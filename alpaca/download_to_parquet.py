"""
Atualiza incrementalmente Parquets da Alpaca em data_parquet/.

Inspeciona data_parquet/source=alpaca/symbol=*/timeframe=*/year=*/ para descobrir
a última barra existente de cada (símbolo, timeframe) e baixa só o que falta.
Reutiliza write_parquet() de src_newest/csv_to_parquet.py (merge+dedup por ano).

Uso:
    python download_to_parquet.py              # baixa todos os SYMBOLS × TIMEFRAMES
    python download_to_parquet.py --dry-run    # mostra janelas sem chamar a API
    python download_to_parquet.py --symbols NVDA AAPL
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# write_parquet vive em src_newest/csv_to_parquet.py
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src_newest"))
from csv_to_parquet import write_parquet, PARQUET_ROOT  # noqa: E402

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────────────

SOURCE = "alpaca"

SYMBOLS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "META", "TSLA",
    "WMT", "BRK-B", "LLY", "JPM", "MU", "AMD", "XOM", "V", "ORCL", "INTC",
    "JNJ", "COST", "MA", "CAT", "BAC", "NFLX", "LRCX", "ABBV", "CSCO", "CVX",
    "PG", "KO", "AMAT", "UNH", "PLTR", "GE", "HD", "MS", "GEV", "MRK", "GS",
    "PM", "TXN", "WFC", "RTX", "KLAC", "LIN", "AXP", "C", "IBM", "PEP",
    "TMUS", "SNDK", "MCD", "ADI", "QCOM", "NEE", "VZ", "DIS", "ANET", "BA",
    "AMGN", "T", "TMO", "STX", "TJX", "APH", "GILD", "BLK", "WDC", "ETN",
    "UBER", "GLW", "ISRG", "SCHW", "DE", "UNP", "APP", "PANW", "BX", "DELL",
    "WELL", "PFE", "CRM", "ABT", "COP", "HON", "VRT", "PLD", "LOW", "BKNG",
    "NEM", "SPGI", "CB", "DHR", "COF", "CRWD", "SBUX", "LMT", "CEG", "PWR",
    "MO", "BMY", "PGR", "PH", "HWM", "SYK", "CVS", "INTU", "TT", "VRTX",
    "EQIX", "ACN", "SO", "CME", "ADBE", "MDT", "CMI", "CDNS", "DUK", "SNPS",
    "HCA", "MAR", "NOW", "CMCSA", "GD", "MCK", "BK", "FDX", "PNC", "KKR",
    "FCX", "WMB", "WM", "JCI", "USB", "ICE", "UPS", "CSX", "AMT", "ABNB",
    "BSX", "EMR", "ADP", "SLB", "SHW", "CIEN", "MPWR", "ELV", "DASH", "NOC",
    "MDLZ", "ORLY", "MRSH", "MCO", "CRH", "RCL", "MMM", "MNST", "FTNT",
    "NXPI", "ITW", "REGN", "ROST", "ECL", "CI", "APO", "HLT", "MSI", "AEP",
    "LITE", "FIX", "GM", "MPC", "HOOD", "CL", "EOG", "DLR", "KMI", "VLO",
    "TDG", "NSC", "CTAS", "PSX", "WBD", "DDOG", "APD", "SPG", "AON", "BKR",
    "NKE", "TRV", "TEL", "TFC", "KEYS", "RSG", "COHR", "SRE", "URI", "PCAR",
    "GWW", "O", "TER", "TGT", "AZO", "AFL", "CARR", "LHX", "VST", "CVNA",
    "AME", "MCHP", "ALL", "CTVA", "PSA", "D", "FANG", "OXY", "OKE", "NUE",
    "TRGP", "ADSK", "AJG", "COIN", "MET", "ETR", "FAST", "ROK", "NDAQ",
    "XEL", "EA", "COR", "F", "DAL", "EBAY", "EW", "GRMN", "EXC", "WAB",
    "FITB", "IDXX", "CAH", "YUM", "AMP", "XYZ", "DHI", "MSCI", "CBRE",
    "ODFL", "BDX", "EME", "VTR", "CMG", "STT", "TTWO", "ON", "AIG", "ZTS",
    "PYPL", "KR", "HPE", "ED", "PEG", "CCI", "JBL", "LYV", "IRM", "KDP",
    "VMC", "IBKR", "CCL", "HSY", "ADM", "WEC", "MLM", "SATS", "HIG", "CBOE",
    "ROP", "PCG", "EQT", "LVS", "STLD", "SYY", "PRU", "KVUE", "WAT", "HBAN",
    "HAL", "A", "ACGL", "KMB", "UAL", "PAYX", "NRG", "CPRT", "MTB", "AXON",
    "CASY", "WDAY", "Q", "EL", "ATO", "RJF", "IR", "VICI", "DOV", "RMD",
    "EXR", "AEE", "DTE", "TDY", "FISV", "TPR", "NTRS", "EXPE", "OTIS",
    "HUM", "IQV", "TPL", "XYL", "DVN", "BIIB", "GEHC", "ARES", "PPL", "CNP",
    "CFG", "VEEV", "CNC", "KHC", "DOW", "MTD", "HUBB", "EIX", "ROL", "STZ",
    "FE", "AVB", "DG", "ES", "SYF", "CINF", "PPG", "FICO", "EQR", "AWK",
    "ALB", "WRB", "BG", "VRSN", "CTSH", "RF", "KEY", "WTW", "TSN", "FIS",
    "FSLR", "DXCM", "ULTA", "LYB", "JBHT", "SBAC", "CMS", "EXE", "PHM",
    "NI", "TROW", "RL", "VRSK", "CHD", "LEN", "WSM", "NTAP", "DRI", "WST",
    "SW", "PFG", "OMC", "L", "VLTO", "DGX", "LH", "STE", "MRNA", "IFF",
    "EFX", "LUV", "CPAY", "DD", "SMCI", "PKG", "CHRW", "INCY", "SNA",
    "EXPD", "HPQ", "CHTR", "BRO", "FFIV", "VTRS", "GPN", "LII", "DLTR",
    "EVRG", "GIS", "LNT", "AMCR", "FTV", "CF", "IP", "BR", "PTC", "WY",
    "TSCO", "ESS", "INVH", "AKAM", "LDOS", "NVR", "IEX", "TXT", "BEN",
    "ZBH", "KIM", "NDSN", "BALL", "GNRC", "LULU", "TRMB", "MAA", "HST",
    "J", "DECK", "MAS", "GPC", "REG", "CDW", "TKO", "CSGP", "EG", "HAS",
    "TYL", "DOC", "APA", "MKC", "PNR", "AVY", "ALGN", "SWK", "BF-B",
    "HII", "DVA", "BBY", "GL", "FOX", "APTV", "PSKY", "FOXA", "SOLV",
    "PNW", "IVZ", "UDR", "GEN", "COO", "AIZ", "ALLE", "HRL", "TTD", "GDDY",
    "ERIE", "RVTY", "ZBRA", "WYNN", "CLX", "DPZ", "PODD", "CPT", "SJM",
    "UHS", "IT", "JKHY", "AES", "FRT", "SWKS", "MGM", "NWSA", "BXP",
    "CRL", "BAX", "BLDR", "AOS", "HSIC", "NCLH", "TAP", "ARE", "FDS",
    "MOS", "TECH", "POOL", "CAG", "CPB", "NWS", "EPAM",
]

# Lista de tuplas (timeframe, label) — TimeFrame.Day/Hour são class-properties
# que retornam instância NOVA a cada acesso, então não dá pra usar como chave de
# dict (hash de identidade).
TIMEFRAMES = [
    (TimeFrame.Minute, "M1"),
    (TimeFrame.Hour, "H1"),
    (TimeFrame.Day,  "D1"),
    (TimeFrame.Week,  "W1"),
    (TimeFrame.Month,  "M1"),
]

DEFAULT_START = datetime(2000, 1, 1, tzinfo=timezone.utc)

# ── Alpaca client ────────────────────────────────────────────────────────────

def get_client() -> StockHistoricalDataClient:
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "Defina ALPACA_API_KEY e ALPACA_SECRET_KEY no arquivo .env"
        )
    return StockHistoricalDataClient(api_key, secret_key)


# ── Inventário do Parquet existente ──────────────────────────────────────────

def find_last_ts(root: Path, symbol: str, tf_label: str) -> pd.Timestamp | None:
    """
    Retorna o maior ts (naive UTC) presente nos parquets de (symbol, tf_label).
    Lê só a coluna `ts` do ano mais recente — custo desprezível.
    """
    base = root / f"source={SOURCE}" / f"symbol={symbol}" / f"timeframe={tf_label}"
    if not base.exists():
        return None
    years = sorted(
        base.glob("year=*"),
        key=lambda p: int(p.name.split("=")[1]),
        reverse=True,
    )
    for year_dir in years:
        f = year_dir / "data.parquet"
        if not f.exists():
            continue
        ts = pd.read_parquet(f, columns=["ts"])["ts"]
        if not ts.empty:
            return ts.max()
    return None


# ── Download e normalização ──────────────────────────────────────────────────

def download_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime | None = None,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame Alpaca → schema unificado (ts naive UTC, tick_volume=NaN)."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)

    nan_series = pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "ts":           idx,
        "open":         pd.to_numeric(df["open"],   errors="coerce"),
        "high":         pd.to_numeric(df["high"],   errors="coerce"),
        "low":          pd.to_numeric(df["low"],    errors="coerce"),
        "close":        pd.to_numeric(df["close"],  errors="coerce"),
        "volume":       pd.to_numeric(df["volume"], errors="coerce"),
        "tick_volume":  np.full(len(df), np.nan),
        "trade_count":  pd.to_numeric(df.get("trade_count", nan_series),
                                      errors="coerce"),
        "vwap":         pd.to_numeric(df.get("vwap", nan_series),
                                      errors="coerce"),
    })
    return (out
            .dropna(subset=["ts", "open", "high", "low", "close"])
            .drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True))


# ── Pipeline ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Só mostra janelas calculadas, sem chamar a API")
    ap.add_argument("--root", type=Path, default=PARQUET_ROOT,
                    help=f"Raiz dos parquets (padrão: {PARQUET_ROOT})")
    ap.add_argument("--symbols", nargs="+",
                    help="Sobrescreve SYMBOLS (lista do código)")
    args = ap.parse_args()

    symbols = args.symbols if args.symbols else SYMBOLS
    client  = None if args.dry_run else get_client()
    now     = datetime.now(timezone.utc)

    total   = len(symbols) * len(TIMEFRAMES)
    skipped, updated, failed = [], [], []

    print(f"\n{'Alpaca → Parquet incremental':^60}")
    print(f"{'─'*60}")
    print(f"Root:   {args.root}")
    print(f"Modo:   {'DRY-RUN' if args.dry_run else 'DOWNLOAD'}")
    print(f"Total:  {len(symbols)} símbolos × {len(TIMEFRAMES)} TFs = {total}\n")

    for i, symbol in enumerate(symbols, 1):
        for tf, tf_label in TIMEFRAMES:
            tag      = f"{symbol}|{tf_label}"
            last_ts  = find_last_ts(args.root, symbol, tf_label)

            if last_ts is None:
                start  = DEFAULT_START
                origin = "vazio"
            else:
                # +1s evita re-baixar exatamente a última barra
                start = (pd.Timestamp(last_ts).tz_localize("UTC")
                         + timedelta(seconds=1)).to_pydatetime()
                origin = f"última={last_ts}"

            if start >= now:
                print(f"  [skip] {tag:14s}  {origin}  (atualizado)")
                skipped.append(tag)
                continue

            window = f"{start.strftime('%Y-%m-%d')} → now"

            if args.dry_run:
                print(f"  [DRY ] {tag:14s}  {origin:32s}  baixaria {window}")
                continue

            try:
                df = download_bars(client, symbol, tf, start, None)
                if df.empty:
                    print(f"  [.   ] {tag:14s}  sem barras novas em {window}")
                    skipped.append(tag)
                    continue
                norm  = normalize(df)
                stats = write_parquet(norm, SOURCE, symbol, tf_label,
                                      root=args.root)
                new = sum(v["new"] for v in stats.values())
                yrs = sorted(stats)
                yr  = f"{yrs[0]}–{yrs[-1]}" if len(yrs) > 1 else str(yrs[0])
                print(f"  [ok  ] {tag:14s}  +{new:>7} barras   "
                      f"anos={yr:9s}  ({window})")
                updated.append((tag, new))
            except Exception as e:
                print(f"  [ERRO] {tag:14s}  {e}")
                failed.append((tag, str(e)))

    print(f"\n{'─'*60}")
    print(f"Resumo: atualizados={len(updated)}  skipped={len(skipped)}  "
          f"falhas={len(failed)}  (total={total})")
    if failed:
        print("\nFalhas:")
        for tag, err in failed[:20]:
            print(f"  • {tag}: {err}")
        if len(failed) > 20:
            print(f"  ... e mais {len(failed) - 20}")


if __name__ == "__main__":
    main()
