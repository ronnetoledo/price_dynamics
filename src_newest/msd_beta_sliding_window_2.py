"""
msd_beta_sliding_window.py
==========================
Estimativa do expoente de difusão β a partir do MSD de uma série de matrizes.

Pipeline:
    C_arr  →  calculate_X  →  X_arr
    X_arr  →  compute_msd  →  msd[s]   (para vários lags s)
    msd    →  fit_beta_tail →  β ± σ_β
"""
import gc
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Integração
# ─────────────────────────────────────────────────────────────────────────────

def calculate_X(K_arr):
    """Centraliza e integra a série de matrizes K_t → processo X_t.

    X_t = cumsum(K_t - <K>)

    Usa cumsum in-place sobre K_centered para manter o pico de memória em
    ~1× o tamanho de K_arr (em vez de 2× com a versão original).
    """
    K_centered = K_arr - np.mean(K_arr, axis=0, keepdims=True)
    np.cumsum(K_centered, axis=0, out=K_centered)   # in-place: evita cópia extra
    return K_centered


# ─────────────────────────────────────────────────────────────────────────────
# 2. MSD por lag
# ─────────────────────────────────────────────────────────────────────────────

def compute_msd(X_store, step):
    """MSD de X para um dado passo de subamostragem.

    Usa einsum para calcular ||ΔX||²_F sem alocar array temporário intermediário.

    Retorna
    -------
    mean, std_error, n_samples   (nan, nan, n se n ≤ 3)
    """
    X_sub = X_store[::step]
    dX = X_sub[1:] - X_sub[:-1]
    # einsum evita a alocação do array dX*dX (economia ~50% de memória neste passo)
    sq = np.einsum('tij,tij->t', dX, dX)
    n = len(sq)
    if n > 3:
        return np.mean(sq), np.std(sq) / np.sqrt(n), n
    return np.nan, np.nan, n


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ajuste log-log na cauda
# ─────────────────────────────────────────────────────────────────────────────

def fit_beta_tail(steps, msd, tail_frac=0.3):
    """Ajusta β via regressão log-log nos últimos `tail_frac` dos pontos válidos.

    Retorna
    -------
    beta, beta_err, s_tail, msd_tail, intercept
        intercept : coeficiente linear do fit (log-espaço), reutilizado no plot
                    para evitar um segundo polyfit.
    """
    valid = np.isfinite(msd) & (msd > 0) & (steps > 0)
    s_valid = steps[valid]
    msd_valid = msd[valid]

    n = len(s_valid)
    if n < 5:
        raise ValueError("Pontos válidos insuficientes para o fit.")

    n_tail = max(int(tail_frac * n), 5)
    s_tail = s_valid[-n_tail:]
    msd_tail = msd_valid[-n_tail:]

    coef, cov = np.polyfit(np.log(s_tail), np.log(msd_tail), 1, cov=True)
    beta = coef[0]
    beta_err = np.sqrt(cov[0, 0])

    return beta, beta_err, s_tail, msd_tail, coef[1]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline completo
# ─────────────────────────────────────────────────────────────────────────────

def estimate_beta(pdf, C_arr, build_plots, max_lag=200, tail_frac=0.7, max_frames=10_000):
    """Pipeline: C_arr → X_arr → MSD(s) → β.

    Parâmetros
    ----------
    C_arr      : array (T, m, m)  — série de matrizes de correlação
    build_plots: bool             — gera figura MSD se True
    max_lag    : int              — lag máximo (limitado por T-1)
    tail_frac  : float            — fração da cauda usada no fit
    max_frames : int              — nº máximo de matrizes mantidas em memória;
                                    se T > max_frames, subsamplea uniformemente
                                    para evitar OOM em datasets grandes

    Retorna
    -------
    beta, beta_err, msd, steps
    """
    T_orig = len(C_arr)

    # Retry progressivo: reduz o subsample pela metade a cada falha de memória.
    # Limites testados: max_frames → /2 → /4 → /8 → 500 (mínimo viável).
    _limits = [max_frames, max_frames // 2, max_frames // 4, max_frames // 8, 500]
    _limits = sorted(set(max(lim, 500) for lim in _limits), reverse=True)

    X_arr = None
    for _max_f in _limits:
        if T_orig > _max_f:
            _step = max(1, T_orig // _max_f)
            _C    = C_arr[::_step]
            print(f"    [MSD] subsample {T_orig}→{len(_C)} matrizes (passo={_step}, limite={_max_f})")
        else:
            _C = C_arr

        try:
            gc.collect()
            X_arr = calculate_X(_C)
            break
        except MemoryError:
            print(f"    [MSD] OOM com {len(_C)} matrizes — reduzindo subsample...")
            gc.collect()

    if X_arr is None:
        raise ValueError(
            f"Memória insuficiente para estimate_beta mesmo com subsample mínimo "
            f"({_limits[-1]} matrizes). Considere aumentar STEP ou reduzir EMBED_DIM."
        )
    T = len(X_arr)
    max_lag = min(max_lag, T - 1)
    steps = np.arange(1, max_lag)

    if len(steps) == 0:
        return float('nan'), float('nan'), np.array([]), np.array([])

    # array (max_lag-1, 3): cada linha é [mean, std_err, n]
    msd_results = np.array([compute_msd(X_arr, s) for s in steps])
    if msd_results.ndim < 2:
        return float('nan'), float('nan'), np.array([]), np.array([])
    msd = msd_results[:, 0]

    beta, beta_err, s_tail, msd_tail, intercept = fit_beta_tail(steps, msd, tail_frac)

    b_interp = (
        "difusão normal (Browniano)"   if 0.85 <= beta <= 1.15 else
        "superdifusão — dinâmica persistente" if beta > 1.15 else
        "subdifusão — dinâmica anti-persistente / mean-reverting"
    )
    print(f"    β = {beta:.4f} ± {beta_err:.4f}  ({b_interp})")
    print(f"    Intervalo de s : [{s_tail[0]:.1f}, {s_tail[-1]:.1f}]  — lags usados no fit log-log")

    if build_plots:
        # reutiliza intercept já calculado — sem segundo polyfit
        s_ref = np.array([s_tail[0], s_tail[-1]], dtype=float)
        msd_ref = np.exp(beta * np.log(s_ref) + intercept)

        plt.figure(figsize=(6, 4))
        plt.loglog(steps, msd, 'o-', alpha=0.5, label='MSD completo')
        plt.loglog(s_tail, msd_tail, 'ro', label='região do fit')
        plt.loglog(s_ref, msd_ref, 'r--', label=f'fit: β={beta:.3f}±{beta_err:.3f}')
        plt.xlabel('s')
        plt.ylabel('MSD')
        plt.title('MSD com fit log-log (cauda)')
        plt.legend()
        plt.grid(True, which='both', ls=':')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    return beta, beta_err, msd, steps