"""
Baixa dados históricos do S&P 500 via Alpaca Markets API.

Uso:
    python download_sp500.py

Opções configuráveis nas variáveis abaixo:
    SYMBOL   - SPY (ETF do S&P 500) ou qualquer ticker
    TIMEFRAME - 1 minuto, 1 hora, 1 dia, etc.
    START    - data de início
    END      - data de fim (None = hoje)
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────────────
SYMBOL    = "SPY"           # SPY = ETF que replica o S&P 500
TIMEFRAME = TimeFrame.Minute   # Day | Hour | Minute | Week | Month
START     = datetime(2000, 1, 1, tzinfo=timezone.utc)
END       = None            # None = agora
OUTPUT    = Path("data")    # pasta onde os CSVs serão salvos
# ────────────────────────────────────────────────────────────────────────────


def get_client() -> StockHistoricalDataClient:
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "Defina ALPACA_API_KEY e ALPACA_SECRET_KEY no arquivo .env"
        )

    # StockHistoricalDataClient não precisa de URL base (usa endpoint de dados)
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
    df = bars.df  # MultiIndex: (symbol, timestamp)

    # Se veio MultiIndex, remove o nível do símbolo
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df


def save(df: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    tf_name = str(timeframe).replace(" ", "_").lower()
    filename = OUTPUT / f"{symbol}_{tf_name}.csv"
    df.to_csv(filename)
    return filename


def main() -> None:
    print(f"Conectando à Alpaca...")
    client = get_client()

    print(f"Baixando {SYMBOL} | timeframe={TIMEFRAME} | início={START.date()}")
    df = download_bars(client, SYMBOL, TIMEFRAME, START, END)

    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nÚltimas linhas:")
    print(df.tail())
    print(f"\nShape: {df.shape}  |  Colunas: {list(df.columns)}")

    path = save(df, SYMBOL, TIMEFRAME)
    print(f"\nDados salvos em: {path}")


if __name__ == "__main__":
    main()
