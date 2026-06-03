"""
Visualizador de candles com a lib nativa do TradingView (lightweight-charts).

Simples e flexível: seleção de símbolo/timeframe, período, zoom/pan/crosshair
nativos, split-adjust. Inclui a análise espectral de Hankel (mesma do projeto)
selecionável entre PREÇO (Δln close) e VOLUME (4 transformações), com o espectro
de uma janela e o expoente de difusão β do processo espectral deslizante.

Rodar:
    streamlit run src_newest/candle_viewer.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import decomp_pca
import hankel_core as hk

_TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN", "Q"]
_TF_LABEL = {"M1": "1 min", "M5": "5 min", "M15": "15 min", "M30": "30 min",
             "H1": "1 hora", "H4": "4 horas", "D1": "Diário", "W1": "Semanal",
             "MN": "Mensal", "Q": "Trimestral"}
_INTRADAY = {"M1", "M5", "M15", "M30", "H1", "H4"}
_UP, _DOWN = "#26a69a", "#ef5350"
_UP_V, _DOWN_V = "rgba(38,166,154,0.45)", "rgba(239,83,80,0.45)"
MAX_CANDLES = 50_000


# ── descoberta de dados ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_symbols() -> list[str]:
    base = decomp_pca.DATA_ROOT / "source=alpaca"
    return sorted(p.name.split("=")[1] for p in base.glob("symbol=*"))


@st.cache_data(show_spinner=False)
def list_timeframes(symbol: str) -> list[str]:
    tfs = set()
    for root in (decomp_pca.DATA_ROOT, decomp_pca.ADJ_ROOT):
        base = root / "source=alpaca" / f"symbol={symbol}"
        if base.exists():
            tfs |= {p.name.split("=")[1] for p in base.glob("timeframe=*")}
    return [tf for tf in _TF_ORDER if tf in tfs]


@st.cache_data(show_spinner=True)
def load(symbol: str, tf: str, adjusted: bool) -> pd.DataFrame:
    df = decomp_pca.load_ohlcv(symbol, tf, adjusted=adjusted)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)


# ── conversão df -> séries do lightweight-charts ──────────────────────────────

def _times(df: pd.DataFrame, tf: str):
    # intraday: UNIX seconds (UTC); diário+: 'YYYY-MM-DD'. cast via datetime64[s]
    # funciona p/ qualquer unidade (o parquet usa us; //1e9 daria ~1000x menor).
    if tf in _INTRADAY:
        return df["ts"].values.astype("datetime64[s]").astype("int64").tolist()
    return df["ts"].dt.strftime("%Y-%m-%d").tolist()


def to_series(df: pd.DataFrame, tf: str):
    t = _times(df, tf)
    o, h, l, c, v = (df[k].tolist() for k in ("open", "high", "low", "close", "volume"))
    candles = [{"time": ti, "open": oo, "high": hh, "low": ll, "close": cc}
               for ti, oo, hh, ll, cc in zip(t, o, h, l, c)]
    volume = [{"time": ti, "value": vv, "color": _UP_V if cc >= oo else _DOWN_V}
              for ti, oo, cc, vv in zip(t, o, c, v)]
    return candles, volume


def chart_config(candles, volume, symbol, tf, show_volume, log_y, markers=None):
    chart = {
        "height": 620,
        "layout": {"background": {"type": "solid", "color": "#131722"},
                   "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "rgba(42,46,57,0.6)"},
                 "horzLines": {"color": "rgba(42,46,57,0.6)"}},
        "crosshair": {"mode": 1},                 # magnet, como o TradingView
        "rightPriceScale": {"borderColor": "rgba(197,203,206,0.3)",
                            "mode": 1 if log_y else 0},
        "timeScale": {"borderColor": "rgba(197,203,206,0.3)",
                      "timeVisible": tf in _INTRADAY, "secondsVisible": False,
                      "rightOffset": 6},
        "watermark": {"visible": True, "text": f"{symbol} · {_TF_LABEL.get(tf, tf)}",
                      "fontSize": 38, "color": "rgba(160,166,180,0.12)",
                      "horzAlign": "center", "vertAlign": "center"},
    }
    series = [{
        "type": "Candlestick", "data": candles,
        "options": {"upColor": _UP, "downColor": _DOWN, "borderVisible": False,
                    "wickUpColor": _UP, "wickDownColor": _DOWN},
        "markers": markers or [],
    }]
    if show_volume:
        series.append({
            "type": "Histogram", "data": volume,
            "options": {"priceFormat": {"type": "volume"}, "priceScaleId": "vol"},
            "priceScale": {"scaleMargins": {"top": 0.8, "bottom": 0.0}},
        })
    return [{"chart": chart, "series": series}]


# ── app ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(layout="wide", page_title="Candle Viewer", page_icon="📈")
    st.markdown("<style>.block-container{padding-top:2.2rem;padding-bottom:0}</style>",
                unsafe_allow_html=True)

    sb = st.sidebar
    sb.header("⚙️ Controles")
    symbols = list_symbols()
    if not symbols:
        st.error("Nenhum símbolo encontrado no tree."); st.stop()

    symbol = sb.selectbox("Símbolo", symbols,
                          index=symbols.index("AAPL") if "AAPL" in symbols else 0)
    tfs = list_timeframes(symbol)
    if not tfs:
        st.error(f"Sem timeframes para {symbol}."); st.stop()
    tf = sb.selectbox("Timeframe", tfs,
                      index=tfs.index("D1") if "D1" in tfs else len(tfs) - 1,
                      format_func=lambda t: f"{t} · {_TF_LABEL.get(t, t)}")
    adjusted = sb.toggle("Split-adjusted", value=True,
                         help="Preços ajustados por split (data_parquet_adj). Desligue p/ o bruto.")

    df = load(symbol, tf, adjusted)
    if df.empty:
        st.error("Sem dados."); st.stop()

    sb.subheader("Período")
    period = sb.select_slider("Janela", options=["1M", "3M", "6M", "1A", "2A", "5A", "Tudo"],
                              value="1A")
    days = {"1M": 30, "3M": 91, "6M": 182, "1A": 365, "2A": 730, "5A": 1825}.get(period)
    view = df if days is None else df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=days)]
    if sb.checkbox("Intervalo de datas customizado"):
        lo, hi = df["ts"].min().date(), df["ts"].max().date()
        dr = sb.date_input("De / até", min_value=lo, max_value=hi,
                           value=(max(lo, (df["ts"].max() - pd.Timedelta(days=365)).date()), hi))
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            view = df[(df["ts"].dt.date >= dr[0]) & (df["ts"].dt.date <= dr[1])]

    sb.subheader("Exibição")
    show_volume = sb.toggle("Volume", value=True)
    log_y       = sb.toggle("Escala log", value=False)

    n = len(view)
    if n > MAX_CANDLES:
        view = view.tail(MAX_CANDLES)
        nxt = _TF_ORDER[min(_TF_ORDER.index(tf) + 1, len(_TF_ORDER) - 1)]
        st.info(f"{n:,} candles no período — mostrando os {MAX_CANDLES:,} mais recentes. "
                f"Para a janela inteira, use um timeframe maior (ex.: {nxt}).")
    if view.empty:
        st.warning("Nenhum candle no período selecionado."); st.stop()
    view = view.reset_index(drop=True)

    # ── análise de Hankel: série (preço/volume), embedding, janela ──────────
    sb.subheader("Espectro (Hankel)")
    series_kind = sb.selectbox(
        "Série", list(hk.SERIES.keys()),
        index=list(hk.SERIES).index("p_dlog"),
        format_func=lambda k: f"{'Preço' if k[0] == 'p' else 'Volume'} — {hk.SERIES[k]}")
    slabel = hk.SERIES[series_kind]
    embed = int(sb.number_input("Embedding (d)", 2, 200, 20, 1,
                                help="Dimensão do embedding de Hankel."))
    mult = int(sb.number_input("Multiplicador da janela", 2, 50, 5, 1,
                               help="window = mult × embed pontos da série. n_samp = window − embed."))
    win = mult * embed                  # pontos da série na janela
    win_bars = win + 1                  # candles (cobre transformações com diff)
    nv = len(view)

    if win_bars > nv:
        st.warning(f"Janela de {win_bars} candles (mult×embed+1) maior que o período ({nv}). "
                   f"Aumente o período ou reduza embed/multiplicador.")
        markers, spec, window, seg = [], None, view.iloc[0:0], None
    else:
        pe = sb.slider("Posição (fim da janela)", win_bars - 1, nv - 1, nv - 1,
                       help="Desliza a janela de tamanho fixo pelo gráfico.")
        p0 = pe - win_bars + 1
        window = view.iloc[p0:pe + 1]
        seg = hk.make_series(window, series_kind)[0][-win:]      # últimos `win` pontos
        spec = hk.hankel_spectrum(seg, embed)
        times = _times(view, tf)
        markers = [
            {"time": times[p0], "position": "belowBar", "color": _UP,
             "shape": "arrowUp", "text": f"janela {win}"},
            {"time": times[pe], "position": "aboveBar", "color": "#f0a020",
             "shape": "arrowDown", "text": "fim"},
        ]

    # ── difusão β (MSD da sequência espectral deslizante) ───────────────────
    sb.subheader("Difusão β")
    do_beta = sb.toggle("Calcular β (MSD deslizante)", value=False,
                        help="Roda a decomposição deslizante sobre o período. Mais pesado.")
    beta_step = int(sb.number_input("Step (stride das janelas)", 1, 200, 1, 1,
                                    help="Aumente para reduzir o viés de sobreposição em β.")) if do_beta else 1

    # ── gráfico de candles ──────────────────────────────────────────────────
    candles, volume = to_series(view, tf)
    cfg = chart_config(candles, volume, symbol, tf, show_volume, log_y, markers=markers)
    renderLightweightCharts(cfg, key=f"{symbol}-{tf}-{adjusted}-{period}-{len(view)}-{embed}-{mult}-{len(window)}")

    c, first = view.iloc[-1], view.iloc[0]
    chg = (c["close"] / first["open"] - 1) * 100
    cols = st.columns(6)
    cols[0].metric("Candles", f"{len(view):,}")
    cols[1].metric("Último close", f"{c['close']:.2f}", f"{chg:+.2f}% no período")
    cols[2].metric("Máx", f"{view['high'].max():.2f}")
    cols[3].metric("Mín", f"{view['low'].min():.2f}")
    cols[4].metric("De", f"{view['ts'].min():%Y-%m-%d}")
    cols[5].metric("Até", f"{view['ts'].max():%Y-%m-%d}")

    # ── figura de Hankel da janela (padrão hankel_demo.png) ─────────────────
    st.divider()
    if spec is None and not window.empty:
        st.warning(f"Janela curta demais para d={embed}.")
    elif spec is not None:
        wdate = f"{window['ts'].iloc[0]:%Y-%m-%d} → {window['ts'].iloc[-1]:%Y-%m-%d}"
        fig = hk.hankel_figure(seg, embed, spec, symbol, _TF_LABEL.get(tf, tf), slabel, wdate)
        st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── figura de β (opt-in) ────────────────────────────────────────────────
    if do_beta:
        st.divider()
        st.markdown(f"#### Expoente de difusão β — {slabel} (período inteiro)")
        S = hk.make_series(view, series_kind)[0]
        try:
            with st.spinner("Decompondo a janela deslizante e ajustando o MSD..."):
                betas = hk.compute_betas(S, embed, mult, step=beta_step)
            bcols = st.columns(4)
            for col, name in zip(bcols, ("total", "struct", "bulk")):
                r = betas[name]
                col.metric(f"β {name}", f"{r['beta']:.3f}", f"± {r['beta_err']:.3f}")
            bcols[3].metric("⟨m⟩ / n_centros", f"{betas['meta']['m_mean']:.1f} / {betas['meta']['n_centers']}")
            figb = hk.beta_figure(betas, symbol, _TF_LABEL.get(tf, tf), slabel)
            st.pyplot(figb, width="stretch"); plt.close(figb)
        except ValueError as e:
            st.warning(f"β indisponível: {e} (amplie o período ou reduza embed/mult/step).")


if __name__ == "__main__":
    main()
