"""
Popula splits.db a partir da Alpaca Corporate Actions API, verificando cada
split contra a nossa própria série D1 (a fonte e os bars são da mesma vendor,
então ex_date casa com o salto).

Fluxo:
  1. API: forward/reverse splits de todos os símbolos do tree (lotes), 2016->hoje.
  2. Verificação: log-ret close-to-close observado no ex_date deve ≈ -ln(factor).
       confirmed=1 se casa (tol), 0 se fora da janela de dados ou sem salto.
  3. Grava via splits_db.upsert_split.
  4. (--scan-orphans) varre os 502 D1 por |log-ret|>thr NÃO explicado por split
     da API — flag de possíveis erros de dado / eventos reais; só reporta.

Uso:
    python fetch_splits.py                 # busca + verifica + grava
    python fetch_splits.py --scan-orphans  # + sweep de saltos órfãos
    python fetch_splits.py --dry-run       # não grava, só relatório
"""
import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.requests import CorporateActionsRequest
from alpaca.data.enums import CorporateActionsType
from alpaca.data.historical.corporate_actions import CorporateActionsClient

import decomp_pca
import splits_db

VERIFY_TOL = 0.12          # |logret_obs - (-ln factor)| aceito p/ confirmar
ORPHAN_THR = 0.18          # |logret| que dispara investigação de órfão
ENV_PATH   = decomp_pca._HERE.parent / "alpaca" / ".env"

# metadados crus da API repassados ao splits_db (chaves em splits_db._META_KEYS)
_META_COLS = ("ca_type", "ca_id", "cusip", "process_date",
              "record_date", "payable_date", "due_bill_redemption_date")


def _d(v):
    """Normaliza data da API (date/datetime/str/None) para ISO 'YYYY-MM-DD'."""
    return None if v is None else str(v)[:10]


# ── símbolos do tree ──────────────────────────────────────────────────────────

def all_symbols(source="alpaca") -> list[str]:
    base = decomp_pca.DATA_ROOT / f"source={source}"
    syms = sorted(p.name.split("=")[1] for p in base.glob("symbol=*"))
    return syms


# ── API ───────────────────────────────────────────────────────────────────────

def _client() -> CorporateActionsClient:
    load_dotenv(ENV_PATH)
    return CorporateActionsClient(os.getenv("ALPACA_API_KEY"),
                                  os.getenv("ALPACA_SECRET_KEY"))


def fetch_api_splits(symbols, start=date(2016, 1, 1), end=None, batch=100):
    """Retorna DataFrame [symbol, ex_date, ratio_num, ratio_den, factor, kind]."""
    end = end or date.today()
    cli = _client()
    rows = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        req = CorporateActionsRequest(
            symbols=chunk,
            types=[CorporateActionsType.FORWARD_SPLIT,
                   CorporateActionsType.REVERSE_SPLIT],
            start=start, end=end,
        )
        res = cli.get_corporate_actions(req).dict()
        for kind in ("forward_splits", "reverse_splits"):
            for ca in res.get(kind) or []:
                num = float(ca["new_rate"])
                den = float(ca["old_rate"])
                rows.append(dict(
                    symbol=ca["symbol"], ex_date=str(ca["ex_date"])[:10],
                    ratio_num=num, ratio_den=den, factor=num / den, kind=kind,
                    # metadados crus da API
                    ca_type=kind,
                    ca_id=None if ca.get("id") is None else str(ca["id"]),
                    cusip=ca.get("cusip"),
                    process_date=_d(ca.get("process_date")),
                    record_date=_d(ca.get("record_date")),
                    payable_date=_d(ca.get("payable_date")),
                    due_bill_redemption_date=_d(ca.get("due_bill_redemption_date")),
                ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["symbol", "ex_date"]).sort_values(["symbol", "ex_date"])
    return df.reset_index(drop=True)


# ── verificação contra a série D1 ─────────────────────────────────────────────

def _d1_close(symbol):
    """(dates_str, close, high, low) do D1 RAW, ou None se ausente.
    Sempre raw (adjusted=False): a verificação precisa ENXERGAR o salto bruto."""
    try:
        df = decomp_pca._load_parquet(symbol, "D1", adjusted=False)
    except FileNotFoundError:
        return None
    d = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d").values
    return d, df["close"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float)


def observed_logret(d1, ex_date):
    """log-ret close-to-close cruzando ex_date (bar do ex_date / bar anterior).
    Retorna (logret, in_range) — in_range=False se ex_date fora da janela."""
    dates, close, *_ = d1
    idx = np.searchsorted(dates, ex_date)
    # idx aponta p/ o 1º bar com data >= ex_date
    if idx == 0 or idx >= len(dates):
        return None, False
    return float(np.log(close[idx]) - np.log(close[idx - 1])), True


def verify_and_store(api_df, dry_run=False):
    report = []
    for sym, grp in api_df.groupby("symbol"):
        d1 = _d1_close(sym)
        for _, r in grp.iterrows():
            factor = r["factor"]
            expected = -np.log(factor)            # logret esperado do salto
            confirmed, note, lr = 0, "", None
            if d1 is None:
                note = "sem D1 no tree"
            else:
                lr, in_range = observed_logret(d1, r["ex_date"])
                if not in_range:
                    note = "ex_date fora da janela de dados"
                elif abs(lr - expected) <= VERIFY_TOL:
                    confirmed = 1
                else:
                    note = f"salto obs {lr:+.3f} != esperado {expected:+.3f}"
            if not dry_run:
                splits_db.upsert_split(
                    sym, r["ex_date"], r["ratio_num"], r["ratio_den"],
                    source="alpaca_ca", detected_logret=lr,
                    confirmed=confirmed, note=note,
                    meta={k: r.get(k) for k in _META_COLS},
                )
            report.append(dict(symbol=sym, ex_date=r["ex_date"],
                               ratio=f"{r['ratio_num']:g}:{r['ratio_den']:g}",
                               factor=round(factor, 4), logret=lr,
                               confirmed=confirmed, note=note))
    return pd.DataFrame(report)


# ── sweep de saltos órfãos (opcional) ─────────────────────────────────────────

def scan_orphans(symbols, api_df, thr=ORPHAN_THR):
    api_dates = set(zip(api_df["symbol"], api_df["ex_date"])) if not api_df.empty else set()
    orphans = []
    for sym in symbols:
        d1 = _d1_close(sym)
        if d1 is None:
            continue
        dates, close, high, low = d1
        lr = np.diff(np.log(close))
        for i in np.where(np.abs(lr) > thr)[0]:
            exd = dates[i + 1]
            if (sym, exd) in api_dates:
                continue
            # range intradiário (Parkinson) pequeno => salto mora no overnight
            park = np.log(high[i + 1] / low[i + 1])
            orphans.append(dict(symbol=sym, date=exd, logret=round(float(lr[i]), 3),
                                intraday_park=round(float(park), 3)))
    return pd.DataFrame(orphans)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scan-orphans", action="store_true")
    ap.add_argument("--symbols", nargs="*", help="subconjunto (default: todos do tree)")
    args = ap.parse_args()

    syms = args.symbols or all_symbols()
    print(f"[1/3] API: buscando splits de {len(syms)} símbolos...")
    api_df = fetch_api_splits(syms)
    print(f"      {len(api_df)} eventos de split na janela.")

    print(f"[2/3] Verificando contra a série D1{' (dry-run)' if args.dry_run else ''}...")
    rep = verify_and_store(api_df, dry_run=args.dry_run)
    if not rep.empty:
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(rep.to_string(index=False))
        n_ok = int(rep["confirmed"].sum())
        print(f"\n      {n_ok}/{len(rep)} confirmados; {len(rep)-n_ok} com ressalva.")
        print(f"      símbolos com >=1 split confirmado: {rep[rep.confirmed==1]['symbol'].nunique()}")
    else:
        print("      nenhum split retornado pela API.")

    if args.scan_orphans:
        print(f"[3/3] Sweep de saltos órfãos (|logret|>{ORPHAN_THR}) nos {len(syms)} D1...")
        orf = scan_orphans(syms, api_df)
        if orf.empty:
            print("      nenhum salto órfão.")
        else:
            print(f"      {len(orf)} saltos NÃO explicados por split da API:")
            with pd.option_context("display.max_rows", None, "display.width", 160):
                print(orf.sort_values("logret").to_string(index=False))


if __name__ == "__main__":
    main()
