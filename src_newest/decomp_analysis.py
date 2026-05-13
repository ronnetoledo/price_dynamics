"""
Análise estatística sobre decomposições espectrais salvas em Parquet.

Consome a saída de decomp_pca.run() via decomp_io.load_as_arrays().
NÃO re-executa PCA — parte diretamente dos eigenvalores/eigenvetores armazenados.

Quantidades calculadas por (symbol, timeframe, step, embed):
    β_T, β_S, β_B   — expoentes de difusão MSD (total, struct, bulk)
    R_FDT ± σ        — razão Flutuação-Dissipação
    α ± err          — expoente da cauda dos tempos de espera entre mudanças de m(τ)
    S_norm           — entropia espectral normalizada (média e série temporal)
    MP_L2 (%)        — erro L2 relativo do bulk vs. lei de Marchenko-Pastur

Uso como script:
    python decomp_analysis.py NVDA 1day
    python decomp_analysis.py NVDA 1day --step 20 --embed 70 --output results/
    python decomp_analysis.py NVDA 1day --no-plots
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import decomp_io
from msd_beta_sliding_window_2 import calculate_X, compute_msd, fit_beta_tail


# ─────────────────────────────────────────────────────────────────────────────
# 1. MSD e expoente β
# ─────────────────────────────────────────────────────────────────────────────

def msd_beta(C_series: np.ndarray,
             max_lag: int = 200,
             tail_frac: float = 0.7) -> dict:
    """
    Calcula MSD(s) e ajusta o expoente de difusão β para uma série (T, d, d).

    Pipeline: C_series → X_t = cumsum(C_t − ⟨C⟩) → MSD(s) → fit log-log

    Retorna
    -------
    beta, beta_err : float   — expoente e incerteza do fit
    msd            : (L,)    — MSD médio para cada lag
    steps          : (L,)    — lags usados
    """
    T = len(C_series)
    _max_lag = min(max_lag, T - 1)

    if _max_lag < 5:
        return dict(beta=np.nan, beta_err=np.nan,
                    msd=np.array([]), steps=np.array([]))

    X_arr = calculate_X(C_series.astype(np.float64))
    steps = np.arange(1, _max_lag)
    msd   = np.array([compute_msd(X_arr, int(s))[0] for s in steps])

    try:
        beta, beta_err, *_ = fit_beta_tail(steps, msd, tail_frac)
    except ValueError:
        beta, beta_err = np.nan, np.nan

    return dict(beta=beta, beta_err=beta_err, msd=msd, steps=steps)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Balanço Flutuação-Dissipação (FDT)
# ─────────────────────────────────────────────────────────────────────────────

def fdt_ratio(C_series: np.ndarray) -> dict:
    """
    Razão Flutuação-Dissipação R = -D/I para a série de matrizes.

    Definições:
        I_t = ||ΔK_t||²_F           (injeção — flutuações novas)
        D_t = 2 ⟨K_t, ΔK_t⟩_F     (dissipação — perda de energia)
        R   = -⟨D⟩/⟨I⟩             (razão média)
        R ≈ 1 → equilíbrio  |  R < 1 → injeção domina

    Incerteza via método delta (propagação de erros covariante).
    """
    K  = C_series.astype(np.float64)
    dK = K[1:] - K[:-1]

    injection   = np.einsum('tij,tij->t', dK,    dK)      # ||ΔK||²_F
    dissipation = 2.0 * np.einsum('tij,tij->t', K[:-1], dK)  # 2⟨K, ΔK⟩

    I_mean = float(np.mean(injection))
    D_mean = float(np.mean(dissipation))

    if I_mean == 0:
        return dict(R_FDT=np.nan, R_FDT_err=np.nan,
                    D_mean=np.nan, I_mean=np.nan,
                    injection=injection, dissipation=dissipation)

    R = -D_mean / I_mean

    n     = len(injection)
    I_std = float(np.std(injection, ddof=1)) / np.sqrt(n)
    D_std = float(np.std(dissipation, ddof=1)) / np.sqrt(n)
    cov_DI = float(np.cov(dissipation, injection, ddof=1)[0, 1]) / n

    sigma_R = np.sqrt(
        D_std**2 / I_mean**2
        + D_mean**2 * I_std**2 / I_mean**4
        - 2 * D_mean * cov_DI / I_mean**3
    )

    return dict(R_FDT=R, R_FDT_err=sigma_R,
                D_mean=-D_mean, I_mean=I_mean,
                injection=injection, dissipation=dissipation)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tempos de espera entre mudanças de regime em m(τ)
# ─────────────────────────────────────────────────────────────────────────────

def _hill_estimator(data: np.ndarray,
                    min_plateau_len: int = 5,
                    stability_threshold: float = 0.05) -> dict:
    """
    Estimativa do expoente de Pareto α pelo método de Hill com detecção de platô.

    Vetorizado com cumsum → O(n). Retorna dict com alpha, alpha_err, ks, alphas.
    """
    tau = np.sort(data)[::-1]
    n   = len(tau)
    if n < 4:
        return dict(alpha=np.nan, alpha_err=np.nan, ks=[], alphas=np.array([]))

    log_tau = np.log(np.maximum(tau, 1e-300))
    cumlog  = np.cumsum(log_tau)

    ks    = np.arange(2, n - 1, dtype=int)
    x_min = tau[ks]
    valid = x_min > 0

    log_xmin = np.where(valid, np.log(np.maximum(x_min, 1e-300)), 0.0)
    sum_log  = cumlog[ks - 1] - ks * log_xmin
    alphas   = np.where(valid & (sum_log > 0), 1.0 + ks / sum_log, np.nan)

    # detecção de platô estável
    dalpha = np.abs(np.diff(alphas))
    best_start, best_len = None, 0
    cur_start, cur_len   = 0, 1
    for i in range(1, len(dalpha)):
        if dalpha[i] < stability_threshold and not np.isnan(alphas[i]):
            cur_len += 1
            if cur_len > best_len:
                best_len  = cur_len
                best_start = cur_start
        else:
            cur_start, cur_len = i, 1

    if best_start is not None and best_len >= min_plateau_len:
        plateau = alphas[best_start: best_start + best_len]
        alpha_hat = float(np.nanmean(plateau))
        alpha_err = float(np.nanstd(plateau))
    else:
        xmin = np.percentile(data, 75)
        tail = data[data >= xmin]
        alpha_hat = float(1 + len(tail) / np.sum(np.log(tail / xmin))) if len(tail) > 0 else np.nan
        alpha_err = np.nan

    return dict(alpha=alpha_hat, alpha_err=alpha_err, ks=ks.tolist(), alphas=alphas)


def regime_stats(m_series: np.ndarray) -> dict:
    """
    Estatísticas dos tempos de espera entre mudanças de m(τ).

    Detecta transições (qualquer Δm ≠ 0) e aplica o estimador de Hill
    à distribuição dos intervalos de espera.

    Retorna
    -------
    n_changes, n_intervals, mean_wt, median_wt,
    alpha, alpha_err, waiting_times,
    hill_ks, hill_alphas,
    indices_up, indices_down   — índices das transições m↑ e m↓
    """
    diff_m = np.diff(m_series.astype(int))

    indices_change = np.where(diff_m != 0)[0] + 1
    indices_up     = np.where(diff_m  > 0)[0] + 1
    indices_down   = np.where(diff_m  < 0)[0] + 1

    n_changes = len(indices_change)

    if n_changes < 2:
        return dict(
            n_changes=n_changes, n_intervals=0,
            mean_wt=np.nan, median_wt=np.nan,
            alpha=np.nan, alpha_err=np.nan,
            waiting_times=np.array([]),
            hill_ks=[], hill_alphas=np.array([]),
            indices_up=indices_up, indices_down=indices_down,
        )

    waiting_times = np.diff(indices_change).astype(float)

    hill = _hill_estimator(waiting_times)

    return dict(
        n_changes=n_changes,
        n_intervals=len(waiting_times),
        mean_wt=float(np.mean(waiting_times)),
        median_wt=float(np.median(waiting_times)),
        alpha=hill["alpha"],
        alpha_err=hill["alpha_err"],
        waiting_times=waiting_times,
        hill_ks=hill["ks"],
        hill_alphas=hill["alphas"],
        indices_up=indices_up,
        indices_down=indices_down,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ajuste Marchenko-Pastur no bulk
# ─────────────────────────────────────────────────────────────────────────────

def _mp_pdf(lam: np.ndarray, var: float, q: float) -> np.ndarray:
    lam_minus = var * (1 - np.sqrt(q)) ** 2
    lam_plus  = var * (1 + np.sqrt(q)) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        density = np.where(
            (lam >= lam_minus) & (lam <= lam_plus),
            np.sqrt(np.maximum((lam_plus - lam) * (lam - lam_minus), 0))
            / (2 * np.pi * q * var * np.maximum(lam, 1e-300)),
            0.0,
        )
    return density


def mp_fit(eigenvalues: np.ndarray, m_series: np.ndarray,
           lam_plus: np.ndarray, embed_dim: int,
           window: int, n_bins: int = 80) -> dict:
    """
    Compara a distribuição empírica dos autovalores do bulk com a lei de MP.

    Os autovalores do bulk (modos acima de m) são rescalonados por σ² antes
    da comparação, tornando a curva MP universal (var=1).

    Parâmetros
    ----------
    eigenvalues : (T, d)  — autovalores em ordem decrescente por janela
    m_series    : (T,)    — número de modos estruturais por janela
    lam_plus    : (T,)    — limiar MP por janela (= σ² (1+√q)²)
    embed_dim   : int     — dimensão d do embedding
    window      : int     — tamanho da janela (para calcular q = d/(window-d))

    Retorna
    -------
    L2_relative_pct, L1_relative, lam_plus_emp, lam_plus_theory,
    bulk_rescaled   — array de todos os autovalores de bulk rescalonados
    """
    q  = embed_dim / (window - embed_dim)
    # σ² estimado por janela a partir do λ+: σ² = λ+ / (1+√q)²
    sigma2 = lam_plus / (1 + np.sqrt(q)) ** 2

    bulk_rescaled = []
    for t in range(len(eigenvalues)):
        m = int(m_series[t])
        if m >= embed_dim:
            continue
        bulk_t = eigenvalues[t, m:]
        s2     = float(sigma2[t])
        if s2 > 0:
            bulk_rescaled.extend((bulk_t / s2).tolist())

    if len(bulk_rescaled) < 20:
        return dict(L2_relative_pct=np.nan, L1_relative=np.nan,
                    lam_plus_emp=np.nan, lam_plus_theory=np.nan,
                    bulk_rescaled=np.array([]))

    bulk = np.array(bulk_rescaled)

    lam_plus_theory  = (1 + np.sqrt(q)) ** 2
    lam_minus_theory = (1 - np.sqrt(q)) ** 2
    lam_plus_emp     = float(np.max(bulk))
    lam_minus_emp    = float(np.min(bulk))

    xs     = np.linspace(lam_minus_emp, lam_plus_emp, 400)
    p_mp   = _mp_pdf(xs, var=1.0, q=q)

    counts, edges = np.histogram(bulk, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p_emp   = np.interp(xs, centers, counts, left=0.0, right=0.0)

    dx           = xs[1] - xs[0]
    L2           = float(np.sqrt(np.sum((p_emp - p_mp) ** 2) * dx))
    L2_mp_norm   = float(np.sqrt(np.sum(p_mp ** 2) * dx))
    L1           = float(np.sum(np.abs(p_emp - p_mp)) * dx)
    L2_relative  = L2 / L2_mp_norm if L2_mp_norm > 0 else np.nan

    return dict(
        L2_relative_pct=float(100 * L2_relative),
        L1_relative=L1,
        lam_plus_emp=lam_plus_emp,
        lam_plus_theory=lam_plus_theory,
        lam_minus_emp=lam_minus_emp,
        lam_minus_theory=lam_minus_theory,
        bulk_rescaled=bulk,
        xs=xs, p_emp=p_emp, p_mp=p_mp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Orquestrador principal
# ─────────────────────────────────────────────────────────────────────────────

def analyze(symbol: str, timeframe: str,
            step: int = 20, embed_dim: int = 70,
            window: int = None,
            max_lag: int = 200, tail_frac: float = 0.7,
            decomp_root: Path = None) -> dict:
    """
    Análise estatística completa para um (symbol, timeframe, step, embed).

    Carrega a decomposição do Parquet e computa:
        β_T, β_S, β_B  (MSD total, estrutural, bulk)
        R_FDT ± σ       (Flutuação-Dissipação)
        α ± err         (cauda dos tempos de espera entre mudanças de m)
        S_norm          (entropia espectral)
        MP_L2 (%)       (ajuste Marchenko-Pastur no bulk)

    Retorna um dict com todos os escalares e séries temporais.
    """
    root       = decomp_root or decomp_io.DECOMP_ROOT
    tf_parquet = timeframe  # já deve estar no formato Parquet (D1, H1, ...)

    if not decomp_io.exists(symbol, tf_parquet, step, embed_dim, root):
        raise FileNotFoundError(
            f"Decomposição não encontrada: {symbol}/{tf_parquet}/step={step}/embed={embed_dim}\n"
            f"  Execute decomp_pca.run('{symbol}', ...) primeiro."
        )

    t0 = time.time()
    print(f"\n[análise] {symbol} | {tf_parquet} | step={step} | embed={embed_dim}")

    # ── carrega arrays ────────────────────────────────────────────────────────
    arrays = decomp_io.load_as_arrays(symbol, tf_parquet, step, embed_dim, root)
    T      = len(arrays["ts"])
    _win   = window if window is not None else 5 * embed_dim

    print(f"  {T} janelas carregadas em {time.time()-t0:.2f}s")

    # ── reconstrói séries de matrizes ────────────────────────────────────────
    print("  reconstruindo C_struct / C_bulk / C_total ...")
    t1 = time.time()
    C_struct = decomp_io.reconstruct_C_series(arrays, subspace="struct")
    C_bulk   = decomp_io.reconstruct_C_series(arrays, subspace="bulk")
    C_total  = decomp_io.reconstruct_C_series(arrays, subspace="total")
    print(f"  reconstrução: {time.time()-t1:.2f}s")

    # ── MSD / β ──────────────────────────────────────────────────────────────
    print("  MSD + beta ...")
    res_beta_T = msd_beta(C_total,  max_lag=max_lag, tail_frac=tail_frac)
    res_beta_S = msd_beta(C_struct, max_lag=max_lag, tail_frac=tail_frac)
    res_beta_B = msd_beta(C_bulk,   max_lag=max_lag, tail_frac=tail_frac)

    def _fmt(b, e):
        return f"{b:.4f} +/- {e:.4f}" if np.isfinite(b) else "N/A"
    print(f"    beta_total  = {_fmt(res_beta_T['beta'], res_beta_T['beta_err'])}")
    print(f"    beta_struct = {_fmt(res_beta_S['beta'], res_beta_S['beta_err'])}")
    print(f"    beta_bulk   = {_fmt(res_beta_B['beta'], res_beta_B['beta_err'])}")

    # ── FDT ──────────────────────────────────────────────────────────────────
    print("  FDT ...")
    res_fdt = fdt_ratio(C_total)
    print(f"    R_FDT = {res_fdt['R_FDT']:.4f} +/- {res_fdt['R_FDT_err']:.4f}")

    # ── Regime / tempos de espera ─────────────────────────────────────────────
    print("  tempos de espera ...")
    res_regime = regime_stats(arrays["m"])
    if np.isfinite(res_regime["alpha"]):
        print(f"    mudancas de m: {res_regime['n_changes']} | "
              f"alpha = {res_regime['alpha']:.4f} +/- {res_regime['alpha_err']:.4f}")
    else:
        print(f"    mudancas de m: {res_regime['n_changes']} | alpha = N/A")

    # ── Entropia ──────────────────────────────────────────────────────────────
    entropy_mean = float(np.mean(arrays["entropy"]))
    print(f"  entropia media: {entropy_mean:.4f}")

    # ── Ajuste MP ────────────────────────────────────────────────────────────
    print("  ajuste Marchenko-Pastur ...")
    res_mp = mp_fit(arrays["eigenvalues"], arrays["m"],
                    arrays["lam_plus"], embed_dim, _win)
    print(f"    L2 relativo: {res_mp['L2_relative_pct']:.2f}%")

    print(f"  total: {time.time()-t0:.1f}s")

    return dict(
        # identificação
        symbol=symbol, timeframe=tf_parquet, step=step, embed_dim=embed_dim,
        n_windows=T,
        # MSD / β
        beta_total=res_beta_T["beta"],       beta_total_err=res_beta_T["beta_err"],
        beta_struct=res_beta_S["beta"],      beta_struct_err=res_beta_S["beta_err"],
        beta_bulk=res_beta_B["beta"],        beta_bulk_err=res_beta_B["beta_err"],
        msd_total=res_beta_T["msd"],         msd_struct=res_beta_S["msd"],
        msd_bulk=res_beta_B["msd"],          msd_steps=res_beta_T["steps"],
        # FDT
        R_FDT=res_fdt["R_FDT"],             R_FDT_err=res_fdt["R_FDT_err"],
        D_mean=res_fdt["D_mean"],            I_mean=res_fdt["I_mean"],
        injection=res_fdt["injection"],       dissipation=res_fdt["dissipation"],
        # Regime
        n_regime_changes=res_regime["n_changes"],
        n_waiting_intervals=res_regime["n_intervals"],
        mean_waiting_time=res_regime["mean_wt"],
        median_waiting_time=res_regime["median_wt"],
        alpha=res_regime["alpha"],           alpha_err=res_regime["alpha_err"],
        waiting_times=res_regime["waiting_times"],
        hill_ks=res_regime["hill_ks"],       hill_alphas=res_regime["hill_alphas"],
        indices_up=res_regime["indices_up"], indices_down=res_regime["indices_down"],
        # Entropia
        entropy_mean=entropy_mean,
        entropy_series=arrays["entropy"],
        # MP
        MP_L2_relative_pct=res_mp["L2_relative_pct"],
        MP_L1_relative=res_mp["L1_relative"],
        MP_lam_plus_emp=res_mp["lam_plus_emp"],
        MP_lam_plus_theory=res_mp["lam_plus_theory"],
        bulk_rescaled=res_mp["bulk_rescaled"],
        mp_xs=res_mp.get("xs"), mp_p_emp=res_mp.get("p_emp"), mp_p_mp=res_mp.get("p_mp"),
        # Séries temporais
        ts=arrays["ts"],
        m_series=arrays["m"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(res: dict, pdf: PdfPages) -> None:
    """Gera as figuras principais e salva no PDF aberto."""
    symbol    = res["symbol"]
    timeframe = res["timeframe"]
    title_base = f"{symbol} | {timeframe} | step={res['step']} embed={res['embed_dim']}"

    # ── Fig 1: MSD log-log ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, label, msd, beta, beta_err in [
        (axes[0], "Total",    res["msd_total"],  res["beta_total"],  res["beta_total_err"]),
        (axes[1], "Struct",   res["msd_struct"], res["beta_struct"], res["beta_struct_err"]),
        (axes[2], "Bulk",     res["msd_bulk"],   res["beta_bulk"],   res["beta_bulk_err"]),
    ]:
        steps = res["msd_steps"]
        if len(steps) == 0 or not np.any(np.isfinite(msd)):
            ax.set_title(f"MSD {label} — sem dados")
            continue
        valid = np.isfinite(msd) & (msd > 0)
        ax.loglog(steps[valid], msd[valid], "o-", ms=3, alpha=0.7, label="MSD")
        if np.isfinite(beta):
            mid   = len(steps[valid]) // 2
            s_ref = np.array([steps[valid][0], steps[valid][-1]], dtype=float)
            y_ref = msd[valid][mid] * (s_ref / steps[valid][mid]) ** beta
            ax.loglog(s_ref, y_ref, "r--", label=f"β={beta:.3f}±{beta_err:.3f}")
        ax.set_xlabel("lag s")
        ax.set_ylabel("MSD")
        ax.set_title(f"MSD {label}")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", ls=":")
    fig.suptitle(title_base)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Fig 2: FDT — injeção e dissipação ────────────────────────────────────
    if len(res["injection"]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        t_idx = np.arange(len(res["injection"]))
        axes[0].plot(t_idx, res["dissipation"], label="Dissipação D", alpha=0.7)
        axes[0].plot(t_idx, res["injection"],   label="Injeção I",    alpha=0.7)
        axes[0].set_title(f"FDT — R={res['R_FDT']:.3f} ± {res['R_FDT_err']:.3f}")
        axes[0].set_xlabel("janela τ")
        axes[0].legend()
        axes[0].grid(ls=":")

        # scatter I vs D
        axes[1].scatter(res["injection"], -res["dissipation"], s=6, alpha=0.5)
        lim = max(res["injection"].max(), (-res["dissipation"]).max())
        axes[1].plot([0, lim], [0, lim], "r--", label="I = D (equilíbrio)")
        axes[1].set_xlabel("Injeção I")
        axes[1].set_ylabel("Dissipação −D")
        axes[1].set_title("Scatter I × D")
        axes[1].legend(fontsize=8)
        axes[1].grid(ls=":")
        fig.suptitle(title_base)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Fig 3: m(τ) e entropia ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(res["m_series"], lw=1)
    axes[0].set_ylabel("m(τ) — modos estruturais")
    axes[0].set_title(f"Dinâmica espectral — {title_base}")
    axes[0].grid(ls=":")

    axes[1].plot(res["entropy_series"], lw=1, color="C1")
    axes[1].set_ylabel("Entropia S norm.")
    axes[1].set_xlabel("janela τ")
    axes[1].axhline(res["entropy_mean"], ls="--", color="C1",
                    label=f"média={res['entropy_mean']:.3f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(ls=":")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Fig 4: tempos de espera + Hill plot ───────────────────────────────────
    wt = res["waiting_times"]
    if len(wt) > 5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(wt, bins="auto", density=True, alpha=0.7, label="tempos de espera")
        axes[0].set_xlabel("Δτ (janelas)")
        axes[0].set_ylabel("densidade")
        alpha_val = res["alpha"]
        axes[0].set_title(
            f"Tempos de espera | α={alpha_val:.3f}" if np.isfinite(alpha_val) else
            "Tempos de espera"
        )
        axes[0].grid(ls=":")

        ks     = np.array(res["hill_ks"])
        alphas = np.array(res["hill_alphas"])
        if len(ks) > 0:
            axes[1].plot(ks, alphas, lw=1.5)
            axes[1].axhline(2.0, ls="--", color="orange", label="α=2")
            if np.isfinite(alpha_val):
                axes[1].axhline(alpha_val, ls=":", color="red",
                                label=f"α estimado={alpha_val:.3f}")
            axes[1].set_xlabel("k (pontos da cauda)")
            axes[1].set_ylabel("α̂(k)")
            axes[1].set_ylim(0, 6)
            axes[1].set_title("Hill plot")
            axes[1].legend(fontsize=8)
            axes[1].grid(ls=":")
        fig.suptitle(title_base)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Fig 5: ajuste Marchenko-Pastur ────────────────────────────────────────
    bulk = res["bulk_rescaled"]
    if len(bulk) > 20 and res["mp_xs"] is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(bulk, bins=80, density=True, alpha=0.6, label="bulk empírico (rescalonado)")
        ax.plot(res["mp_xs"], res["mp_p_mp"],  "r-",  lw=2, label="MP teórico (σ²=1)")
        ax.plot(res["mp_xs"], res["mp_p_emp"], "k--", lw=1.5, label="empírico suavizado")
        ax.axvline(res["MP_lam_plus_emp"],    ls=":", color="blue",  label=f"λ+ emp={res['MP_lam_plus_emp']:.3f}")
        ax.axvline(res["MP_lam_plus_theory"], ls=":", color="green", label=f"λ+ MP={res['MP_lam_plus_theory']:.3f}")
        ax.set_xlabel("λ / σ²")
        ax.set_ylabel("densidade")
        ax.set_title(f"Bulk vs. Marchenko-Pastur — L2={res['MP_L2_relative_pct']:.1f}%\n{title_base}")
        ax.legend(fontsize=7)
        ax.grid(ls=":")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Saída de resultados
# ─────────────────────────────────────────────────────────────────────────────

_SCALAR_KEYS = (
    "symbol", "timeframe", "step", "embed_dim", "n_windows",
    "beta_total", "beta_total_err", "beta_struct", "beta_struct_err",
    "beta_bulk", "beta_bulk_err",
    "R_FDT", "R_FDT_err", "D_mean", "I_mean",
    "n_regime_changes", "n_waiting_intervals",
    "mean_waiting_time", "median_waiting_time",
    "alpha", "alpha_err",
    "entropy_mean",
    "MP_L2_relative_pct", "MP_L1_relative",
    "MP_lam_plus_emp", "MP_lam_plus_theory",
)


def to_scalar_row(res: dict) -> dict:
    """Extrai apenas os campos escalares do resultado (para CSV/tabela)."""
    return {k: res[k] for k in _SCALAR_KEYS if k in res}


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Análise estatística sobre decomposição espectral em Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("symbol",    help="Ticker (ex: NVDA, PETR4)")
    p.add_argument("timeframe", help="Timeframe no formato Parquet (ex: D1, H1)")
    p.add_argument("--step",       type=int,   default=20)
    p.add_argument("--embed",      type=int,   default=70)
    p.add_argument("--window",     type=int,   default=None,
                   help="Tamanho da janela (padrão: 5 × embed)")
    p.add_argument("--max-lag",    type=int,   default=200)
    p.add_argument("--tail-frac",  type=float, default=0.7)
    p.add_argument("--output",     type=Path,  default=Path("."),
                   help="Diretório de saída para CSV e PDF")
    p.add_argument("--no-plots",   action="store_true",
                   help="Pula a geração de figuras")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    res = analyze(
        symbol    = args.symbol,
        timeframe = args.timeframe,
        step      = args.step,
        embed_dim = args.embed,
        window    = args.window,
        max_lag   = args.max_lag,
        tail_frac = args.tail_frac,
    )

    # CSV com escalares
    tag      = f"{args.symbol}_{args.timeframe}_s{args.step}_e{args.embed}"
    csv_path = args.output / f"{tag}_analysis.csv"
    pd.DataFrame([to_scalar_row(res)]).to_csv(csv_path, index=False)
    print(f"\n[salvo] {csv_path}")

    # PDF com figuras
    if not args.no_plots:
        pdf_path = args.output / f"{tag}_analysis.pdf"
        with PdfPages(pdf_path) as pdf:
            plot_results(res, pdf)
        print(f"[salvo] {pdf_path}")
