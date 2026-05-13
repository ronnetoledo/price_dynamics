"""
Baixa dados históricos de múltiplos ativos via Alpaca Markets API.

Uso:
    python download_assets.py

Configure as listas SYMBOLS e TIMEFRAMES abaixo.
Cada combinação (ativo × timeframe) é salva em data/{SYMBOL}_{timeframe}.csv.
"""

import io
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()


#def get_sp500_symbols() -> list[str]:
#    """Busca a lista atual de componentes do S&P 500 via Wikipedia."""
#    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)"}
#    html = requests.get(url, headers=headers, timeout=15).text
#    symbols = pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()
#    # Alpaca usa '-' em vez de '.' (ex: BRK.B → BRK-B)
#    return [s.replace(".", "-") for s in symbols]


# ── Configurações ────────────────────────────────────────────────────────────
#SYMBOLS = get_sp500_symbols()   # todos os componentes do S&P 500 (~503 ativos)

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


TIMEFRAMES = [
    TimeFrame.Day,
    TimeFrame.Hour,
    #TimeFrame.Minute,
]

START  = datetime(2000, 1, 1, tzinfo=timezone.utc)
END    = None           # None = agora
OUTPUT = Path("data")   # pasta onde os CSVs serão salvos
# ────────────────────────────────────────────────────────────────────────────


def get_client() -> StockHistoricalDataClient:
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "Defina ALPACA_API_KEY e ALPACA_SECRET_KEY no arquivo .env"
        )

    return StockHistoricalDataClient(api_key, secret_key)


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

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df


def save(df: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    tf_name  = str(timeframe).replace(" ", "_").lower()
    filename = OUTPUT / f"{symbol}_{tf_name}.csv"
    df.to_csv(filename)
    return filename


def main() -> None:
    print("Conectando à Alpaca...")
    client = get_client()

    total   = len(SYMBOLS) * len(TIMEFRAMES)
    success = []
    failed  = []

    for i, symbol in enumerate(SYMBOLS, 1):
        for timeframe in TIMEFRAMES:
            tag = f"{symbol} | {timeframe}"
            print(f"[{len(success) + len(failed) + 1}/{total}] Baixando {tag} ...")
            try:
                df   = download_bars(client, symbol, timeframe, START, END)
                path = save(df, symbol, timeframe)
                print(f"  ✓ {df.shape[0]} barras → {path}")
                success.append(tag)
            except Exception as e:
                print(f"  ✗ Erro: {e}")
                failed.append((tag, str(e)))

    print(f"\n{'─' * 50}")
    print(f"Concluído: {len(success)}/{total} downloads com sucesso.")
    if failed:
        print("Falhas:")
        for tag, err in failed:
            print(f"  • {tag}: {err}")


if __name__ == "__main__":
    main()
