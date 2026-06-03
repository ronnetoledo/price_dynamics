"""
Visualizador de candles com a lib nativa do TradingView (lightweight-charts).

Simples e flexível: seleção de símbolo/timeframe, período de amostragem, zoom/pan
e crosshair nativos. Lê via o loader do projeto (decomp_pca.load_ohlcv) — herda os
preços split-adjusted e todos os TFs gerados por resample_tf.py.

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

_TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN", "Q"]
_TF_LABEL = {"M1": "1 min", "M5": "5 min", "M15": "15 min", "M30": "30 min",
             "H1": "1 hora", "H4": "4 horas", "D1": "Diário", "W1": "Semanal",
             "MN": "Mensal", "Q": "Trimestral"}
_INTRADAY = {"M1", "M5", "M15", "M30", "H1", "H4"}
_UP, _DOWN = "#26a69a", "#ef5350"
_UP_V, _DOWN_V = "rgba(38,166,154,0.45)", "rgba(239,83,80,0.45)"
MAX_CANDLES = 50_000          # lightweight-charts aguenta muito mais que Plotly


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
    # intraday: UNIX seconds (UTC); diário+: 'YYYY-MM-DD' (eixo de datas limpo)
    # cast via datetime64[s] funciona p/ qualquer unidade de origem (us/ns) —
    # o parquet usa timestamp("us"), então dividir int64 por 1e9 daria ~1000x menor.
    if tf in _INTRADAY:
        return df["ts"].values.astype("datetime64[s]").astype("int64").tolist()
    return df["ts"].dt.strftime("%Y-%m-%d").tolist()


def to_series(df: pd.DataFrame, tf: str):
    t = _times(df, tf)
    o, h, l, c, v = (df[k].tolist() for k in ("open", "high", "low", "close", "volume"))
    candles = [{"time": ti, "open": oo, "high": hh, "low": ll, "close": cc}
               for ti, oo, hh, ll, cc in zip(t, o, h, l, c)]
    volume = [{"time": ti, "value": vv,
               "color": _UP_V if cc >= oo else _DOWN_V}
              for ti, oo, cc, vv in zip(t, o, c, v)]
    return candles, volume


# ── espectro de Hankel (mesma convenção de decomp_pca) ────────────────────────

def hankel_spectrum(ret: np.ndarray, embed: int) -> dict | None:
    """Embedding de Hankel dos log-retornos -> covariância d×d -> autovalores.

    Trajetória X[j,k] = ret[j+k] (n_samp, d); centra por coluna sobre as amostras;
    C = Xcᵀ·Xc/(n_samp-1); autovalores decrescentes. Limiar Marchenko-Pastur
    lam = var̄·(1+√q)², q = d/n_samp; m = nº de modos acima do limiar.
    """
    d = int(embed)
    ret = np.ascontiguousarray(ret, dtype=np.float64)
    n_samp = len(ret) - d + 1
    if d < 2 or n_samp < 2:
        return None
    s = ret.strides[0]
    X = np.lib.stride_tricks.as_strided(ret, shape=(n_samp, d), strides=(s, s)).copy()
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc / (n_samp - 1)
    evals = np.linalg.eigvalsh(C)[::-1]                     # decrescente
    q = d / n_samp
    mp = float(Xc.var(axis=0, ddof=1).mean() * (1.0 + np.sqrt(q)) ** 2)
    vp = np.maximum(evals, 1e-15)
    p = vp / vp.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(d))
    return {"evals": evals, "mp": mp, "m": int((evals > mp).sum()),
            "entropy": entropy, "n_samp": n_samp, "q": q, "C": C, "X": X}


def _sym_limit(a, pct=99.0):
    v = float(np.nanpercentile(np.abs(a), pct))
    return v if v > 0 else 1e-12


def hankel_figure(ret: np.ndarray, embed: int, spec: dict,
                  symbol: str, tf_label: str, wdate: str) -> plt.Figure:
    """Figura de 4 painéis no padrão de hankel_demo.png:
    (a) segmento de log-retornos, (b) matriz de Hankel, (c) covariância C,
    (d) espectro de autovalores com modos estruturais/bulk e limiar MP."""
    X, C, ev, mp, m = spec["X"], spec["C"], spec["evals"], spec["mp"], spec["m"]
    n_samp, q, d = spec["n_samp"], spec["q"], int(embed)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))

    # (a) segmento de log-retornos
    ax[0, 0].plot(ret, lw=0.6, color="#1f77b4")
    ax[0, 0].set_title(f"(a) Segmento de log-retornos ({len(ret)} barras)", fontsize=10)
    ax[0, 0].set_xlabel("barra dentro da janela"); ax[0, 0].set_ylabel("retorno")
    ax[0, 0].grid(alpha=0.25)

    # (b) matriz de Hankel K (n_samp snapshots × d atrasos)
    vb = _sym_limit(X)
    imb = ax[0, 1].imshow(X, aspect="auto", cmap="RdBu_r", vmin=-vb, vmax=vb)
    ax[0, 1].set_title(f"(b) Hankel K = ({n_samp} snapshots × {d} atrasos)", fontsize=10)
    ax[0, 1].set_xlabel("atraso (delay)"); ax[0, 1].set_ylabel("snapshot (tempo)")
    fig.colorbar(imb, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # (c) covariância C = XᵀX/(n_samp-1)
    vc = _sym_limit(C, 100)
    imc = ax[1, 0].imshow(C, cmap="RdBu_r", vmin=-vc, vmax=vc)
    ax[1, 0].set_title(f"(c) Covariância C = XᵀX/(n_samp-1)  ({d}×{d})", fontsize=10)
    ax[1, 0].set_xlabel("atraso"); ax[1, 0].set_ylabel("atraso")
    fig.colorbar(imc, ax=ax[1, 0], fraction=0.046, pad=0.04)

    # (d) espectro de C — autovalores ordenados
    idx = np.arange(1, len(ev) + 1)
    struct = ev > mp
    axd = ax[1, 1]
    axd.fill_between(idx, ev, ev.min(), color="#2ca02c", alpha=0.08)
    axd.semilogy(idx, ev, "-", color="#999", lw=1, zorder=1)
    axd.scatter(idx[~struct], ev[~struct], s=14, color="#2ca02c", label="bulk", zorder=2)
    if struct.any():
        axd.scatter(idx[struct], ev[struct], s=26, color="#d62728",
                    label=f"estruturais (λ>λ₊) · m={m}", zorder=3)
    axd.axhline(mp, ls="--", lw=1, color="k", label="limiar MP λ₊")
    axd.set_title("(d) Espectro de C — autovalores ordenados", fontsize=10)
    axd.set_xlabel("índice de modo"); axd.set_ylabel("autovalor (log)")
    axd.legend(fontsize=7, loc="upper right"); axd.grid(alpha=0.25, which="both")

    fig.suptitle(f"{symbol} {tf_label} | janela @ {wdate} | embed={d} window={len(ret)} "
                 f"n_samp={n_samp} q={q:.2f} → {m} modos estruturais", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


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
    if days is not None:
        view = df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=days)]
    else:
        view = df
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

    # ── janela de análise espectral (Hankel) ────────────────────────────────
    sb.subheader("Espectro (Hankel)")
    embed = int(sb.number_input("Embedding (d)", min_value=2, max_value=200,
                                value=20, step=1, help="Dimensão do embedding de Hankel."))
    mult = int(sb.number_input("Multiplicador da janela", min_value=2, max_value=50,
                               value=5, step=1,
                               help="window = mult × embed retornos. n_samp = window − embed."))
    win_ret = mult * embed                       # nº de retornos na janela
    win_bars = win_ret + 1                        # candles (retornos = diff dos closes)
    nv = len(view)

    if win_bars > nv:
        st.warning(f"Janela de {win_bars} candles (mult×embed+1) maior que o período "
                   f"({nv}). Aumente o período ou reduza embed/multiplicador.")
        markers, spec, window = [], None, view.iloc[0:0]
    else:
        pe = sb.slider("Posição (fim da janela)", win_bars - 1, nv - 1, nv - 1,
                       help="Desliza a janela de tamanho fixo pelo gráfico.")
        p0 = pe - win_bars + 1
        window = view.iloc[p0:pe + 1]
        ret = np.diff(np.log(window["close"].to_numpy(float)))
        spec = hankel_spectrum(ret, embed)
        times = _times(view, tf)
        markers = [
            {"time": times[p0], "position": "belowBar", "color": _UP,
             "shape": "arrowUp", "text": f"janela {win_ret}"},
            {"time": times[pe], "position": "aboveBar", "color": "#f0a020",
             "shape": "arrowDown", "text": "fim"},
        ]

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

    # ── figura de Hankel (padrão hankel_demo.png) ───────────────────────────
    st.divider()
    if spec is None and not window.empty:
        st.warning(f"Janela curta demais para d={embed}.")
    elif spec is not None:
        wdate = f"{window['ts'].iloc[0]:%Y-%m-%d %H:%M}"
        fig = hankel_figure(np.diff(np.log(window["close"].to_numpy(float))),
                            embed, spec, symbol, _TF_LABEL.get(tf, tf), wdate)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


if __name__ == "__main__":
    main()
