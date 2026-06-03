"""
Back-adjustment dos parquet pelos splits confirmados em splits.db.

Escreve um tree paralelo data_parquet_adj/ contendo APENAS os símbolos com
split confirmado (mesma estrutura/schema/compressão). Os demais ~450 símbolos
não são copiados — o loader (decomp_pca, adjusted=True) cai no raw para eles.

Regra (back-adjust, convenção total-return):
    cumfac(t) = produto dos factors de todos os splits com ex_date > data(t)
    open/high/low/close/vwap  /=  cumfac
    volume/tick_volume        *=  cumfac
    trade_count               inalterado (split não cria trades)
Bars no/após o ex_date não são tocados (já são pós-split).

Uso:
    python apply_splits.py --symbols AAPL          # prova de 1 símbolo
    python apply_splits.py                          # lote (todos confirmados)
    python apply_splits.py --overwrite              # reescreve partições já feitas
    python apply_splits.py --verify AAPL            # checa resíduo pós-ajuste
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import decomp_pca
import splits_db
from csv_to_parquet import _SCHEMA

RAW_ROOT = decomp_pca.DATA_ROOT
ADJ_ROOT = decomp_pca._HERE.parent / "data_parquet_adj"

PRICE_COLS = ("open", "high", "low", "close", "vwap")
VOL_COLS   = ("volume", "tick_volume")


# ── núcleo do ajuste ──────────────────────────────────────────────────────────

def _cumfac(row_dates: np.ndarray, ex: np.ndarray, fac: np.ndarray) -> np.ndarray:
    """Fator acumulado por linha: produto dos factors com ex_date > data(linha)."""
    suffix = np.concatenate([np.cumprod(fac[::-1])[::-1], [1.0]])  # suffix[j]=prod(fac[j:])
    idx = np.searchsorted(ex, row_dates, side="right")            # ex_date<=data ficam de fora
    return suffix[idx]


def adjust_df(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    ex  = splits["ex_date"].to_numpy()                # ordenado asc (get_splits ordena)
    fac = splits["factor"].to_numpy(float)
    row_dates = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d").to_numpy()
    cf = _cumfac(row_dates, ex, fac)
    out = df.copy()
    for c in PRICE_COLS:
        if c in out:
            out[c] = out[c].to_numpy(float) / cf
    for c in VOL_COLS:
        if c in out:
            out[c] = out[c].to_numpy(float) * cf
    return out


# ── varredura de partições ────────────────────────────────────────────────────

def _raw_partitions(symbol: str, source="alpaca"):
    base = RAW_ROOT / f"source={source}" / f"symbol={symbol}"
    return sorted(base.glob("timeframe=*/year=*/data.parquet"))


def process_symbol(symbol: str, overwrite=False) -> dict:
    splits = splits_db.get_splits(symbol, confirmed_only=True)
    if splits.empty:
        return {"symbol": symbol, "skipped": "sem split confirmado"}
    parts = _raw_partitions(symbol)
    n_written = n_rows = n_adj = 0
    for src in parts:
        rel  = src.relative_to(RAW_ROOT)
        dest = ADJ_ROOT / rel
        if dest.exists() and not overwrite:
            continue
        df  = pd.read_parquet(src)
        adj = adjust_df(df, splits)
        # quantas linhas efetivamente mudaram (auditoria)
        if "close" in df:
            n_adj += int((~np.isclose(df["close"].to_numpy(float),
                                       adj["close"].to_numpy(float))).sum())
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(adj, schema=_SCHEMA, preserve_index=False)
        pq.write_table(table, dest, compression="snappy")
        n_written += 1
        n_rows    += len(adj)
    return {"symbol": symbol, "splits": len(splits), "partitions": n_written,
            "rows": n_rows, "rows_adjusted": n_adj}


# ── verificação ───────────────────────────────────────────────────────────────

def verify(symbol: str, thr=0.18):
    """Compara saltos D1 raw vs ajustado em torno dos ex_dates."""
    splits = splits_db.get_splits(symbol, confirmed_only=True)
    raw = decomp_pca._load_parquet(symbol, "D1", adjusted=False)
    base = ADJ_ROOT / f"source=alpaca" / f"symbol={symbol}" / "timeframe=D1"
    aparts = sorted(base.glob("year=*/data.parquet"))
    adj = pd.concat([pd.read_parquet(p) for p in aparts], ignore_index=True).sort_values("ts")
    def jumps(df):
        c = df["close"].to_numpy(float)
        lr = np.diff(np.log(c))
        d = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d").to_numpy()[1:]
        return d, lr
    dr, lrr = jumps(raw)
    da, lra = jumps(adj)
    print(f"\n=== {symbol}: saltos |logret|>{thr} ===")
    print("  RAW :", [(dr[i], round(float(lrr[i]),3)) for i in np.where(np.abs(lrr)>thr)[0]])
    print("  ADJ :", [(da[i], round(float(lra[i]),3)) for i in np.where(np.abs(lra)>thr)[0]])
    for _, s in splits.iterrows():
        i = np.searchsorted(da, s["ex_date"])
        if 0 < i < len(da):
            print(f"  ex_date {s['ex_date']} (factor {s['factor']:g}): logret ADJ no boundary = {lra[i-1]:+.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", help="subconjunto (default: todos confirmados)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", metavar="SYMBOL")
    args = ap.parse_args()

    if args.verify:
        verify(args.verify)
        return

    syms = args.symbols or splits_db.symbols_with_splits(confirmed_only=True)
    print(f"Ajustando {len(syms)} símbolos -> {ADJ_ROOT}")
    tot_part = tot_rows = 0
    for k, sym in enumerate(syms, 1):
        r = process_symbol(sym, overwrite=args.overwrite)
        if "skipped" in r:
            print(f"[{k}/{len(syms)}] {sym}: {r['skipped']}")
        else:
            tot_part += r["partitions"]; tot_rows += r["rows"]
            print(f"[{k}/{len(syms)}] {sym}: {r['splits']} splits, "
                  f"{r['partitions']} partições, {r['rows_adjusted']} linhas ajustadas")
    print(f"\nTotal: {tot_part} partições escritas, {tot_rows} linhas.")


if __name__ == "__main__":
    main()
