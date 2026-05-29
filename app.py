import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Configuration
DATA_ROOT = Path(r"E:\PROJETOS\DINAMICA DE PREÇOS\data_parquet")

st.set_page_config(layout="wide", page_title="Finance Data Viewer")

st.title("📈 Financial Parquet Data Viewer")

# 1. Data Navigation Logic
def get_symbols():
    # Structure: data_parquet/source=alpaca/symbol=XXX
    alpaca_path = DATA_ROOT / "source=alpaca"
    if not alpaca_path.exists():
        return []

    symbols = []
    for folder in alpaca_path.iterdir():
        if folder.is_dir() and folder.name.startswith("symbol="):
            symbols.append(folder.name.replace("symbol=", ""))
    return sorted(symbols)

def get_timeframes(symbol):
    # Structure: data_parquet/source=alpaca/symbol=XXX/timeframe=YYY
    symbol_path = DATA_ROOT / "source=alpaca" / f"symbol={symbol}"
    if not symbol_path.exists():
        return []

    timeframes = []
    for folder in symbol_path.iterdir():
        if folder.is_dir() and folder.name.startswith("timeframe="):
            timeframes.append(folder.name.replace("timeframe=", ""))
    return sorted(timeframes)

@st.cache_data
def load_data(symbol, timeframe):
    # Structure: data_parquet/source=alpaca/symbol=XXX/timeframe=YYY/year=ZZZZ/data.parquet
    tf_path = DATA_ROOT / "source=alpaca" / f"symbol={symbol}" / f"timeframe={timeframe}"

    all_files = []
    for year_folder in tf_path.iterdir():
        if year_folder.is_dir() and year_folder.name.startswith("year="):
            parquet_file = year_folder / "data.parquet"
            if parquet_file.exists():
                all_files.append(parquet_file)

    if not all_files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in all_files]
    df = pd.concat(dfs).sort_values('ts').reset_index(drop=True)
    return df

# --- SIDEBAR ---
st.sidebar.header("Filters")

symbols = get_symbols()
if not symbols:
    st.error(f"No data found in {DATA_ROOT}")
    st.stop()

selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

timeframes = get_timeframes(selected_symbol)
if not timeframes:
    st.error(f"No timeframes found for {selected_symbol}")
    st.stop()

selected_tf = st.sidebar.selectbox("Select Timeframe", timeframes)

# Load Data with spinner
with st.spinner('Loading data...'):
    df_full = load_data(selected_symbol, selected_tf)

if df_full.empty:
    st.error("No data available for the selected options.")
    st.stop()

# Convert timestamp
df_full['ts'] = pd.to_datetime(df_full['ts'])

# Date Filtering
min_date = df_full['ts'].min().date()
max_date = df_full['ts'].max().date()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Filter DataFrame by date
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df_full['ts'].dt.date >= start_date) & (df_full['ts'].dt.date <= end_date)
    df_full_filtered = df_full.loc[mask]
else:
    df_full_filtered = df_full

# Downsample for plotting if necessary
MAX_PLOT_POINTS = 5000
if len(df_full_filtered) > MAX_PLOT_POINTS:
    step = len(df_full_filtered) // MAX_PLOT_POINTS + 1
    df_plot = df_full_filtered.iloc[::step].reset_index(drop=True)
    st.info(f"Showing every {step}-th point for performance. Full data has {len(df_full_filtered)} points.")
else:
    df_plot = df_full_filtered

# --- MAIN PANEL ---
col1, col2 = st.columns([3, 1])

with col1:
    # Plotting Candlestick + Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_width=[0.2, 0.8])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot['ts'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name="Price"
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=df_plot['ts'],
        y=df_plot['volume'],
        name="Volume",
        marker_color='rgba(100, 149, 237, 0.5)'
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark",
        title=f"{selected_symbol} - {selected_tf}",
        yaxis1_title="Price",
        yaxis2_title="Volume"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Quick Stats")
    st.write(f"**Total Rows:** {len(df_plot)}")
    st.write(f"**Start Date:** {df_plot['ts'].min()}")
    st.write(f"**End Date:** {df_plot['ts'].max()}")
    st.write(f"**Max Price:** {df_plot['high'].max():.2f}")
    st.write(f"**Min Price:** {df_plot['low'].min():.2f}")

    if st.checkbox("Show Raw Data"):
        st.dataframe(df_plot)

# Footer
st.caption(f"Data source: {DATA_ROOT}")
