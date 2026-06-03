"""
Figura: os quatro MSDs sobrepostos (preco, struct, total, bulk) no mesmo eixo
log-log, evidenciando a clivagem 1o/2o momento (beta_preco < 1 vs beta_espectral > 1).

Eixo x em tempo fisico (minutos): lag espectral de s janelas subamostradas
= s * stride * step barras; para M1, 1 barra = 1 min.

Uso:
    python make_msd_figure.py MSFT M1 --step 20 --embed 70
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # núcleo na raiz de src_newest
import decomp_io
import decomp_pca
from decomp_analysis import msd_beta
from price_msd import variance_time, fit_slope

BARS_PER_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 390}


def main(symbol, tf, step, embed, target_frames, max_lag, tail_frac):
    bpm = BARS_PER_MIN.get(tf, 1)

    # ── espectral ────────────────────────────────────────────────────────────
    arrays = decomp_io.load_as_arrays(symbol, tf, step, embed)
    T = len(arrays["ts"])
    stride = max(1, T // target_frames)
    for k in ("eigenvalues", "eigenvectors", "m", "lam_plus", "entropy", "ts"):
        arrays[k] = arrays[k][::stride]
    lag_min = stride * step * bpm          # minutos por unidade de lag espectral
    print(f"{symbol} {tf}: T={T} janelas -> subamostra {len(arrays['ts'])} "
          f"(stride={stride}); 1 lag = {lag_min} min")

    spec = {}
    for sub in ("struct", "total", "bulk"):
        C = decomp_io.reconstruct_C_series(arrays, subspace=sub)
        r = msd_beta(C, max_lag=max_lag, tail_frac=tail_frac)
        spec[sub] = r
        print(f"  beta_{sub:6s} = {r['beta']:.3f} +/- {r['beta_err']:.3f}")
        del C

    # ── preco ────────────────────────────────────────────────────────────────
    df = decomp_pca.load_ohlcv(symbol, tf).reset_index(drop=True)
    c = df["close"].to_numpy(float)
    r = np.log(c[1:] / c[:-1])
    if np.abs(r).max() > 0.5:                # neutraliza split, se houver
        r = r.copy(); r[np.abs(r) > 0.5] = np.median(r)
    logP = np.concatenate([[0.0], np.cumsum(r)])
    lo, hi = (10, 120) if tf == "M1" else (5, 80)
    taus = np.arange(1, 250)
    Vp = variance_time(logP, taus)
    bp, bpe, _ = fit_slope(taus, Vp, lo, hi)
    print(f"  beta_preco  = {bp:.3f} +/- {bpe:.3f}")

    # ── figura ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    series = [
        ("preço",  taus * bpm,                    Vp,                bp,  bpe, "#000000", "o"),
        ("struct", spec["struct"]["steps"] * lag_min, spec["struct"]["msd"],
                                  spec["struct"]["beta"], spec["struct"]["beta_err"], "#1f77b4", "s"),
        ("total",  spec["total"]["steps"] * lag_min,  spec["total"]["msd"],
                                  spec["total"]["beta"],  spec["total"]["beta_err"],  "#2ca02c", "^"),
        ("bulk",   spec["bulk"]["steps"] * lag_min,   spec["bulk"]["msd"],
                                  spec["bulk"]["beta"],   spec["bulk"]["beta_err"],   "#d62728", "v"),
    ]
    for label, x, y, b, be, col, mk in series:
        good = np.isfinite(y) & (y > 0) & (x > 0)
        x, y = x[good], y[good]
        y = y / y[0]                          # normaliza no primeiro ponto
        ax.loglog(x, y, mk, ms=3, alpha=0.45, color=col)
        ax.loglog(x, y, "-", lw=0.8, alpha=0.6, color=col,
                  label=f"{label}: β = {b:.2f} ± {be:.2f}")

    # guia de difusao normal (slope 1)
    xg = np.array([series[0][1][1], series[3][1][-1]])
    ax.loglog(xg, (xg / xg[0]), "k--", lw=1, alpha=0.5, label="β = 1 (browniano)")

    ax.set_xlabel("lag temporal (min)")
    ax.set_ylabel("MSD normalizado")
    ax.set_title(f"{symbol} {tf} (step={step}, embed={embed})\n"
                 f"preço (1º momento, sub) vs. espectral (2º momento, super)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = Path(__file__).parent / f"msd_4curvas_{symbol}_{tf}_s{step}_e{embed}.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigura: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol"); ap.add_argument("timeframe")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--embed", type=int, default=70)
    ap.add_argument("--frames", type=int, default=12000)
    ap.add_argument("--max-lag", type=int, default=200)
    ap.add_argument("--tail-frac", type=float, default=0.7)
    a = ap.parse_args()
    main(a.symbol, a.timeframe, a.step, a.embed, a.frames, a.max_lag, a.tail_frac)
