"""
Núcleo compartilhado da análise espectral de Hankel (preço e volume).

Usado por candle_viewer.py (interativo) e hankel_volume/hankel_volume.py (CLI).
Mesma convenção de decomp_pca: embedding de Hankel -> covariância d×d ->
autovalores -> limiar Marchenko-Pastur -> modos estruturais m e entropia; o
expoente β vem do MSD da sequência espectral deslizante (X_t = cumsum(C_t-⟨C⟩)).

Backend do matplotlib: definido pelo chamador (candle_viewer/hankel_volume usam Agg).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from msd_beta_sliding_window_2 import calculate_X, compute_msd, fit_beta_tail

# séries de entrada: id = "<campo>_<transformação>", campo p=preço(close) v=volume.
# transformações: raw (nível), log (ln), dlog (Δln, incrementos), zlog (z-score de ln).
SERIES = {
    "p_raw":  "close (P)",      "p_log":  "ln(P)",
    "p_dlog": "Δln(P) — retornos", "p_zlog": "z[ln(P)]",
    "v_raw":  "volume (V)",     "v_log":  "ln(V)",
    "v_dlog": "Δln(V)",         "v_zlog": "z[ln(V)]",
}


def make_series(df: pd.DataFrame, kind: str):
    """Retorna (S, ts) para a série escolhida, com timestamps alinhados.

    Preço usa ln (P>0); volume usa log1p (protege V=0). dlog perde a 1ª amostra.
    Aplicar Hankel a níveis (raw/log) de preço capta a TENDÊNCIA como modo
    dominante (séries não-estacionárias); dlog (retornos) é o caso estacionário.
    """
    ts = pd.to_datetime(df["ts"]).to_numpy()
    field, tr = kind.split("_")
    x = (df["close"] if field == "p" else df["volume"]).to_numpy(float)
    logf = np.log if field == "p" else np.log1p
    if tr == "raw":
        return x, ts
    if tr == "log":
        return logf(x), ts
    if tr == "zlog":
        L = logf(x); sd = L.std()
        return (L - L.mean()) / (sd if sd > 0 else 1.0), ts
    if tr == "dlog":
        return np.diff(logf(x)), ts[1:]
    raise ValueError(f"série desconhecida: {kind!r}")


# ── espectro de UMA janela ────────────────────────────────────────────────────

def hankel_spectrum(series: np.ndarray, embed: int) -> dict | None:
    d = int(embed)
    s = np.ascontiguousarray(series, dtype=np.float64)
    n_samp = len(s) - d + 1
    if d < 2 or n_samp < 2:
        return None
    st = s.strides[0]
    X = np.lib.stride_tricks.as_strided(s, shape=(n_samp, d), strides=(st, st)).copy()
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc / (n_samp - 1)
    evals = np.linalg.eigvalsh(C)[::-1]
    q = d / n_samp
    mp = float(Xc.var(axis=0, ddof=1).mean() * (1.0 + np.sqrt(q)) ** 2)
    vp = np.maximum(evals, 1e-15); p = vp / vp.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(d))
    return {"evals": evals, "mp": mp, "m": int((evals > mp).sum()),
            "entropy": entropy, "n_samp": n_samp, "q": q, "C": C, "X": X}


def _sym_limit(a, pct=99.0):
    v = float(np.nanpercentile(np.abs(a), pct))
    return v if v > 0 else 1e-12


def hankel_figure(seg, embed, spec, symbol, tf_label, series_label, wdate):
    """Figura de 4 painéis no padrão de hankel_demo.png."""
    X, C, ev, mp, m = spec["X"], spec["C"], spec["evals"], spec["mp"], spec["m"]
    n_samp, q, d = spec["n_samp"], spec["q"], int(embed)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))

    ax[0, 0].plot(seg, lw=0.6, color="#1f77b4")
    ax[0, 0].set_title(f"(a) Segmento de {series_label} ({len(seg)} pontos)", fontsize=10)
    ax[0, 0].set_xlabel("ponto dentro da janela"); ax[0, 0].set_ylabel(series_label)
    ax[0, 0].grid(alpha=0.25)

    vb = _sym_limit(X)
    imb = ax[0, 1].imshow(X, aspect="auto", cmap="RdBu_r", vmin=-vb, vmax=vb)
    ax[0, 1].set_title(f"(b) Hankel K = ({n_samp} snapshots × {d} atrasos)", fontsize=10)
    ax[0, 1].set_xlabel("atraso (delay)"); ax[0, 1].set_ylabel("snapshot (tempo)")
    fig.colorbar(imb, ax=ax[0, 1], fraction=0.046, pad=0.04)

    vc = _sym_limit(C, 100)
    imc = ax[1, 0].imshow(C, cmap="RdBu_r", vmin=-vc, vmax=vc)
    ax[1, 0].set_title(f"(c) Covariância C = XᵀX/(n_samp-1)  ({d}×{d})", fontsize=10)
    ax[1, 0].set_xlabel("atraso"); ax[1, 0].set_ylabel("atraso")
    fig.colorbar(imc, ax=ax[1, 0], fraction=0.046, pad=0.04)

    idx = np.arange(1, len(ev) + 1); struct = ev > mp
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

    fig.suptitle(f"{symbol} {tf_label} | {series_label} | janela @ {wdate} | "
                 f"embed={d} window={len(seg)} n_samp={n_samp} q={q:.2f} "
                 f"→ {m} modos estruturais", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ── β: MSD da sequência espectral deslizante ──────────────────────────────────

def sliding_covariances(S: np.ndarray, embed: int, window: int, step: int):
    """Para cada centro de janela (espaçados por step), a covariância d×d do
    embedding de Hankel. Retorna (C[T,d,d], n_samp)."""
    d, n_samp = int(embed), window - int(embed)
    s = np.ascontiguousarray(S, dtype=np.float64)
    n_centers = (len(s) - window) // step + 1
    if n_centers < 10:
        raise ValueError(f"poucos centros de janela ({n_centers})")
    st = s.strides[0]
    view = np.lib.stride_tricks.as_strided(
        s, shape=(n_centers, n_samp, d), strides=(step * st, st, st))
    C = np.empty((n_centers, d, d), dtype=np.float64)
    for i in range(0, n_centers, 2000):                # chunk: limita o pico de RAM
        Xc = view[i:i + 2000].copy()
        Xc -= Xc.mean(axis=1, keepdims=True)
        C[i:i + 2000] = Xc.transpose(0, 2, 1) @ Xc / (n_samp - 1)
    return C, n_samp


def split_total_struct_bulk(C: np.ndarray, n_samp: int):
    """Séries total/struct/bulk via projeção espectral (struct = modos > limiar MP)."""
    T, d, _ = C.shape
    w, V = np.linalg.eigh(C)
    w = w[:, ::-1]; V = V[:, :, ::-1]
    q = d / n_samp
    lam = (np.trace(C, axis1=1, axis2=2) / d) * (1.0 + np.sqrt(q)) ** 2
    struct = w > lam[:, None]
    Vt = V.transpose(0, 2, 1)
    C_struct = (V * np.where(struct, w, 0.0)[:, None, :]) @ Vt
    C_bulk = (V * np.where(struct, 0.0, w)[:, None, :]) @ Vt
    return {"total": C, "struct": C_struct, "bulk": C_bulk,
            "m_mean": float(struct.sum(axis=1).mean())}


def beta_of(C_series: np.ndarray, max_lag: int, tail_frac: float) -> dict:
    """C_series (T,d,d) → X=cumsum(C-⟨C⟩) → MSD(s) → β via fit log-log na cauda."""
    X = calculate_X(C_series.astype(np.float64))
    steps = np.arange(1, min(max_lag, len(X) - 1))
    if len(steps) < 5:
        return {"beta": np.nan, "beta_err": np.nan, "msd": np.array([]), "steps": steps}
    msd = np.array([compute_msd(X, s) for s in steps])[:, 0]
    beta, beta_err, s_tail, msd_tail, intercept = fit_beta_tail(steps, msd, tail_frac)
    return {"beta": beta, "beta_err": beta_err, "msd": msd, "steps": steps,
            "s_tail": s_tail, "intercept": intercept}


def compute_betas(S: np.ndarray, embed: int, mult: int, step: int = 1,
                  max_lag: int = 200, tail_frac: float = 0.7) -> dict:
    """Pipeline β completo a partir de uma série S: total/struct/bulk."""
    window = mult * embed
    C, n_samp = sliding_covariances(S, embed, window, step)
    series = split_total_struct_bulk(C, n_samp)
    out = {k: beta_of(series[k], max_lag, tail_frac) for k in ("total", "struct", "bulk")}
    out["meta"] = {"n_centers": len(C), "n_samp": n_samp, "q": embed / n_samp,
                   "m_mean": series["m_mean"], "window": window, "step": step,
                   "embed": embed}
    return out


def beta_figure(betas, symbol, tf_label, series_label):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for axi, name in zip(ax, ("total", "struct", "bulk")):
        r = betas[name]
        if not np.isfinite(r["beta"]):
            axi.set_title(f"({name}) sem fit"); continue
        st, msd = r["steps"], r["msd"]
        good = np.isfinite(msd) & (msd > 0)
        axi.loglog(st[good], msd[good], "o-", ms=3, alpha=0.5, color="#1f77b4")
        s_ref = np.array([r["s_tail"][0], r["s_tail"][-1]], float)
        axi.loglog(s_ref, np.exp(r["beta"] * np.log(s_ref) + r["intercept"]), "r--", lw=1.5)
        axi.set_title(f"({name})  β = {r['beta']:.3f} ± {r['beta_err']:.3f}", fontsize=10)
        axi.set_xlabel("s (lag)"); axi.set_ylabel("MSD"); axi.grid(alpha=0.3, which="both")
    mt = betas["meta"]
    fig.suptitle(f"{symbol} {tf_label} | {series_label} | MSD espectral deslizante | "
                 f"embed={mt['embed']} window={mt['window']} step={mt['step']} "
                 f"n_centers={mt['n_centers']} q={mt['q']:.2f} ⟨m⟩={mt['m_mean']:.1f}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
