"""
Reamostragem de timeframe (fino -> grosso) de OHLCV.

Agrega barras de um timeframe de origem para um alvo mais grosso, gravando
novas partições `timeframe=<alvo>` no MESMO tree de onde leu (raw->data_parquet,
adjusted->data_parquet_adj), consumíveis direto por decomp_pca.load_ohlcv.

Convenções (validadas contra as barras nativas da Alpaca):
  - Alvos intraday (M5,M15,M30,H1,H4): binagem por relógio UTC, left-labeled,
    closed-left. Reproduz exatamente a H1 nativa (OHLC e volume idênticos).
  - Alvos de calendário (D1,W1,MN,Q): agrupados por período em horário de NY
    (DST-aware), rotulados na meia-noite-ET (= 04:00/05:00 UTC), como a nativa.
    W1 ancora na segunda-feira.

Agregação: open=first, high=max, low=min, close=last, volume/tick_volume/
trade_count=soma. vwap = Σ(vwap·volume)/Σvolume — é uma APROXIMAÇÃO do vwap de
trades crus (diferença pequena), por agregar vwaps já agregados.

Extended hours: por padrão agrega tudo (pré/pós-mercado, como a H1 nativa);
--regular-hours restringe a 09:30–16:00 ET (como a D1 nativa).

Uso:
    python resample_tf.py AAPL --from M1 --to M5
    python resample_tf.py AAPL --from M1 --to D1 --regular-hours
    python resample_tf.py --all --from H1 --to W1          # todos os símbolos
    python resample_tf.py NVDA --from M1 --to M15 --raw --dry-run
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import decomp_pca
from csv_to_parquet import _SCHEMA, _parquet_path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ET = "America/New_York"
REG_START_MIN = 9 * 60 + 30      # 09:30 ET
REG_END_MIN   = 16 * 60          # 16:00 ET (inclusivo: barra de fechamento)

# granularidade em minutos — valida origem mais fina que o alvo
_TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240,
           "D1": 1440, "W1": 10080, "MN": 43800, "Q": 131400}

# alvos intraday: regra pandas (relógio UTC)
_INTRADAY_RULE = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h"}
# alvos de calendário: código to_period (agrupado em ET)
_CAL_PERIOD = {"D1": "D", "W1": "W", "MN": "M", "Q": "Q"}

# origem default por alvo (intraday vem do M1; semanal+ vem da D1 nativa/consolidada)
_DEFAULT_SRC = {"M5": "M1", "M15": "M1", "M30": "M1", "H1": "M1", "H4": "M1",
                "D1": "M1", "W1": "D1", "MN": "D1", "Q": "D1"}
# TFs faltantes no alpaca (nativos: M1, H1, D1, W1)
_MISSING_TARGETS = ("M5", "M15", "M30", "H4", "MN", "Q")

_OUT_COLS = ["ts", "open", "high", "low", "close",
             "volume", "tick_volume", "trade_count", "vwap"]


# ── agregação OHLCV de um grouper (resample ou groupby) ───────────────────────

def _aggregate(grouper) -> pd.DataFrame:
    out = pd.DataFrame({
        "open":        grouper["open"].first(),
        "high":        grouper["high"].max(),
        "low":         grouper["low"].min(),
        "close":       grouper["close"].last(),
        "volume":      grouper["volume"].sum(min_count=1),
        "tick_volume": grouper["tick_volume"].sum(min_count=1),
        "trade_count": grouper["trade_count"].sum(min_count=1),
        "_pv":         grouper["_pv"].sum(min_count=1),
    }).dropna(subset=["open"])
    out["vwap"] = out["_pv"] / out["volume"].replace(0, np.nan)
    return out.drop(columns="_pv")


# ── núcleo ────────────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, target_tf: str,
                   regular_hours: bool = False) -> pd.DataFrame:
    """Reamostra um OHLCV para target_tf. df precisa ter colunas do _SCHEMA."""
    d = df.copy()
    d["_pv"] = d["vwap"] * d["volume"]

    if target_tf in _INTRADAY_RULE:
        r = d.set_index("ts").resample(_INTRADAY_RULE[target_tf],
                                       label="left", closed="left")
        out = _aggregate(r)
        out.index.name = "ts"
        out = out.reset_index()
    elif target_tf in _CAL_PERIOD:
        idx = (pd.DatetimeIndex(d["ts"]).tz_localize("UTC").tz_convert(ET))
        d = d.set_index(idx)
        if regular_hours:
            mins = d.index.hour * 60 + d.index.minute
            d = d[(mins >= REG_START_MIN) & (mins <= REG_END_MIN)]
        # período em horário-de-parede ET (tz_localize(None) explicita a intenção
        # e evita o warning de to_period sobre tz-aware index)
        period = d.index.tz_localize(None).to_period(_CAL_PERIOD[target_tf])
        out = _aggregate(d.groupby(period))
        # rótulo = início do período em ET -> UTC naive (04:00/05:00 como a nativa)
        start_et = out.index.to_timestamp(how="start")
        ts = (start_et.tz_localize(ET, nonexistent="shift_forward", ambiguous="NaT")
                      .tz_convert("UTC").tz_localize(None))
        out = out.reset_index(drop=True)
        out.insert(0, "ts", ts)
        out = out.dropna(subset=["ts"]).reset_index(drop=True)
    else:
        raise ValueError(f"target_tf desconhecido: {target_tf!r}")

    return out[_OUT_COLS].sort_values("ts").reset_index(drop=True)


# ── localização da fonte / escrita ────────────────────────────────────────────

def _resolve_source(symbol: str, src_tf: str, adjusted: bool):
    """Espelha a ordem de busca de decomp_pca._load_parquet.
    Retorna (df, root, source) — escreve-se o alvo no MESMO root/source."""
    tf = decomp_pca._TF_TO_PARQUET.get(src_tf, src_tf)
    roots = (decomp_pca.ADJ_ROOT, decomp_pca.DATA_ROOT) if adjusted else (decomp_pca.DATA_ROOT,)
    for root in roots:
        for source in ("alpaca", "metatrader"):
            base = root / f"source={source}" / f"symbol={symbol}" / f"timeframe={tf}"
            parts = sorted(base.glob("year=*/data.parquet")) if base.exists() else []
            if parts:
                df = (pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
                        .sort_values("ts").reset_index(drop=True))
                return df, root, source
    raise FileNotFoundError(f"fonte não encontrada: {symbol} {src_tf} (adjusted={adjusted})")


def _write_partitions(out: pd.DataFrame, root, source: str,
                      symbol: str, target_tf: str, overwrite: bool) -> int:
    n = 0
    out = out.assign(_year=pd.DatetimeIndex(out["ts"]).year)
    for year, grp in out.groupby("_year"):
        dest = _parquet_path(root, source, symbol, target_tf, int(year))
        if dest.exists() and not overwrite:
            continue
        grp = grp.drop(columns="_year").reset_index(drop=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(grp, schema=_SCHEMA, preserve_index=False)
        pq.write_table(table, dest, compression="snappy")
        n += 1
    return n


def process(symbol: str, src_tf: str, target_tf: str, adjusted: bool,
            regular_hours: bool, write: bool, overwrite: bool) -> dict:
    if _TF_MIN[src_tf] >= _TF_MIN[target_tf]:
        raise ValueError(f"origem {src_tf} não é mais fina que o alvo {target_tf}")
    df, root, source = _resolve_source(symbol, src_tf, adjusted)
    out = resample_ohlcv(df, target_tf, regular_hours=regular_hours)
    n_part = _write_partitions(out, root, source, symbol, target_tf, overwrite) if write else 0
    return {"symbol": symbol, "source": source, "src_rows": len(df),
            "out_rows": len(out), "partitions": n_part,
            "root": root.name}


def process_multi(symbol: str, targets: list[str], adjusted: bool,
                  regular_hours: bool, write: bool, overwrite: bool,
                  src_override: str | None = None) -> dict:
    """Gera vários alvos lendo cada origem uma única vez (M1 lido 1×, D1 1×)."""
    by_src: dict[str, list[str]] = {}
    for t in targets:
        s = src_override or _DEFAULT_SRC[t]
        if _TF_MIN[s] >= _TF_MIN[t]:
            raise ValueError(f"origem {s} não é mais fina que {t}")
        by_src.setdefault(s, []).append(t)

    n_part = 0
    done = []
    root_name = source = None
    for s, tgts in by_src.items():
        df, root, source = _resolve_source(symbol, s, adjusted)
        root_name = root.name
        for t in tgts:
            out = resample_ohlcv(df, t, regular_hours=regular_hours)
            n_part += _write_partitions(out, root, source, symbol, t, overwrite) if write else 0
            done.append(t)
    return {"symbol": symbol, "source": source, "root": root_name,
            "targets": done, "partitions": n_part}


# ── main ──────────────────────────────────────────────────────────────────────

def _all_symbols(src_tf: str) -> list[str]:
    tf = decomp_pca._TF_TO_PARQUET.get(src_tf, src_tf)
    base = decomp_pca.DATA_ROOT / "source=alpaca"   # raw tem todos os símbolos
    return sorted(p.name.split("=")[1] for p in base.glob("symbol=*")
                  if (p / f"timeframe={tf}").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="símbolos (ou use --all)")
    ap.add_argument("--all", action="store_true", help="todos os símbolos do tree")
    ap.add_argument("--from", dest="src", default=None,
                    help="timeframe de origem (default: por alvo — M1 p/ intraday, D1 p/ semanal+)")
    ap.add_argument("--to", dest="dst", nargs="+",
                    help="timeframe(s) alvo")
    ap.add_argument("--missing", action="store_true",
                    help=f"gera os TFs faltantes do alpaca: {', '.join(_MISSING_TARGETS)}")
    ap.add_argument("--regular-hours", action="store_true",
                    help="restringe a 09:30–16:00 ET (casa a D1 nativa)")
    ap.add_argument("--raw", action="store_true", help="lê/grava o tree RAW (sem split-adjust)")
    ap.add_argument("--dry-run", action="store_true", help="não grava, só reporta")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    targets = list(_MISSING_TARGETS) if args.missing else [t.upper() for t in (args.dst or [])]
    if not targets:
        ap.error("informe --to <TF...> ou --missing")
    src_override = args.src.upper() if args.src else None
    for tf in targets + ([src_override] if src_override else []):
        if tf not in _TF_MIN:
            ap.error(f"timeframe desconhecido: {tf} (use {list(_TF_MIN)})")
    adjusted = not args.raw

    # com --all e sem origem fixa, usa o conjunto de símbolos que têm M1 (cobre tudo)
    probe_src = src_override or "M1"
    syms = _all_symbols(probe_src) if args.all else args.symbols
    if not syms:
        ap.error("informe símbolos ou --all")

    hours = "regular" if args.regular_hours else "all-hours"
    print(f"alvos {targets} | {len(syms)} símbolos | {hours} | "
          f"{'RAW' if args.raw else 'adjusted'} | {'DRY-RUN' if args.dry_run else 'gravando'}")
    tot = 0
    for k, sym in enumerate(syms, 1):
        try:
            r = process_multi(sym, targets, adjusted, args.regular_hours,
                              write=not args.dry_run, overwrite=args.overwrite,
                              src_override=src_override)
            tot += r["partitions"]
            print(f"[{k}/{len(syms)}] {sym}: {len(r['targets'])} alvos, "
                  f"{r['partitions']} partições ({r['root']}/{r['source']})")
        except (FileNotFoundError, ValueError) as e:
            print(f"[{k}/{len(syms)}] {sym}: SKIP — {e}")
    print(f"\nTotal: {tot} partições escritas.")


if __name__ == "__main__":
    main()
