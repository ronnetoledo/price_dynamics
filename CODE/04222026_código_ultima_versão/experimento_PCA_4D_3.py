import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import eigh
from scipy.linalg import subspace_angles
from scipy.fft import rfft, rfftfreq
from scipy import stats
from scipy.stats import binomtest
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.stattools import adfuller
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages

class PdfPages(_PdfPages):
    def __exit__(self, *args):
        try:
            super().__exit__(*args)
        except AttributeError:
            pass  # já foi fechado manualmente, ignora

import msd_beta_sliding_window_2 as mbsw
import Extrator_pdf as epdf

# =========================
# FUNÇÕES AUXILIARES
# =========================

def hankel_embed(x, m):
    """Embedding de Hankel via stride_tricks (view zero-copy).

    result[i, j] = x[i + j]  para  i = 0…N-1,  j = 0…m-1
    A matriz é construída sem copiar dados; np.cov fará a cópia internamente.
    """
    x = np.ascontiguousarray(x, dtype=float)
    N = len(x) - m
    shape = (N, m)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def pca_cov(X):
    """PCA via matriz de covariância. Retorna (autovalores desc., autovetores, cov)."""
    C = np.cov(X, rowvar=False)
    vals, vecs = eigh(C)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx], C


def mp_lambda_plus(var, q):
    """Limite superior da distribuição de Marchenko–Pastur."""
    return var * (1 + np.sqrt(q)) ** 2


def estimate_frequency_fourier(v):
    """Frequência média ponderada pela potência espectral (centro de massa)."""
    n_fft = 128
    v_detrend = v - np.mean(v)
    yf = np.abs(rfft(v_detrend, n=n_fft))
    xf = rfftfreq(n_fft, d=1)
    if np.sum(yf) == 0:
        return np.nan
    return np.sum(xf * yf) / np.sum(yf)


def hankel_embed_ohlc(rO, rH, rL, rC, m):
    """Embedding OHLC vetorizado em array único pré-alocado (evita hstack)."""
    N = len(rO) - m
    out = np.empty((N, 4 * m), dtype=float)
    for i in range(m):
        j = 4 * i
        out[:, j]     = rO[i:N + i]
        out[:, j + 1] = rH[i:N + i]
        out[:, j + 2] = rL[i:N + i]
        out[:, j + 3] = rC[i:N + i]
    return out


def hankel_embed_ohlc_indep(rO, rH, rL, rC, m):
    return np.vstack([
        hankel_embed(rO, m),
        hankel_embed(rH, m),
        hankel_embed(rL, m),
        hankel_embed(rC, m),
    ])


def plot_mode(vec, ax, title):
    if MODE == 'vector':
        ax.plot(vec[0::4], label='O')
        ax.plot(vec[1::4], label='H')
        ax.plot(vec[2::4], label='L')
        ax.plot(vec[3::4], label='C')
        ax.legend()
    else:
        ax.plot(vec, label='modo temporal')
        ax.legend()
    ax.set_title(title)


def plot_eigenvector_candles(vec, ax, title, EMBED_current):
    m = EMBED_current
    width = 0.35
    for k in range(m):
        O = vec[4 * k]
        H = vec[4 * k + 1]
        L = vec[4 * k + 2]
        C = vec[4 * k + 3]
        ax.plot([k, k], [L, H], linewidth=1)
        lower = min(O, C)
        height = abs(C - O)
        rect = plt.Rectangle(
            (k - width / 2, lower), width,
            max(height, 1e-8), fill=False
        )
        ax.add_patch(rect)
    ax.set_xlim(-1, m)
    ax.set_title(title)


def validate_fdt(K_series, A_series, dt, verbose=True):
    """Valida o Teorema de Flutuação-Dissipação para uma série de matrizes K_t.

    Parâmetros
    ----------
    K_series : array (T, m, m)
    A_series : array (T-1, m, m)   drift estimado A(K_t)
    dt       : float               passo temporal

    Retorna dict com F_hat, D_hat, erro_relativo, erro_espectral.
    """
    K_series = np.asarray(K_series)
    A_series = np.asarray(A_series)
    m = K_series.shape[1]

    dK = K_series[1:] - K_series[:-1]
    if len(A_series) != len(dK):
        raise ValueError("A_series deve ter tamanho T-1.")

    # --- Resíduos ---
    eps = dK - A_series * dt

    # --- Termo de Flutuação: F = (1/T·dt) Σ_t eps_t eps_t^T ---
    # Equivalente vetorizado de: for t: F += eps[t] @ eps[t].T
    F_hat = np.einsum('til,tkl->ik', eps, eps) / (len(eps) * dt)

    # --- Termo Dissipativo: D = -(1/T) Σ_t (A_t K_t + K_t A_t^T) ---
    T_A = len(A_series)
    K = K_series[:T_A]
    AK  = np.einsum('til,tlj->tij', A_series, K)    # A @ K  por passo
    KAt = np.einsum('til,tjl->tij', K, A_series)    # K @ A^T por passo
    D_hat = -(AK + KAt).sum(axis=0) / T_A

    diff = F_hat - D_hat
    erro_relativo = np.linalg.norm(diff, 'fro') / np.linalg.norm(D_hat, 'fro')
    erro_espectral = np.max(np.abs(np.linalg.eigvalsh(diff)))
    diff_sym = 0.5 * (diff + diff.T)

    if verbose:
        print("==== Validação FDT ====")
        print("Dimensão matriz:", m)
        print("Erro relativo (Frobenius):", erro_relativo)
        print("Erro espectral:", erro_espectral)
        print("Norma F_hat:", np.linalg.norm(F_hat, 'fro'))
        print("Norma D_hat:", np.linalg.norm(D_hat, 'fro'))

    return {
        "F_hat": F_hat, "D_hat": D_hat,
        "erro_relativo": erro_relativo, "erro_espectral": erro_espectral,
        "diff": diff, "diff_sym": diff_sym,
    }


def compute_conditional_drift(K_series, window=20, delta_t=1.0):
    """Estimativa suavizada do drift médio.

    A soma telescopa: Σ_{s=0}^{W-1}(K[t+s+1]-K[t+s]) = K[t+W] - K[t].
    Implementação vetorizada: O(T·m²) em vez de O(T·W·m²).
    """
    K_series = np.asarray(K_series)
    return (K_series[window:] - K_series[:-window]) / (window * delta_t)


def spectral_entropy(eigenvalues):
    lambdas = np.asarray(eigenvalues)
    lambdas = lambdas[lambdas > 0]
    p = lambdas / np.sum(lambdas)
    S = -np.sum(p * np.log(p))
    return S / np.log(len(p))


def hill_estimator(tau_star, min_plateau_len=5, stability_threshold=0.05):
    """Estima α pelo método de Hill com detecção automática de platô.

    Vetorizado com cumsum: O(n) em vez de O(n²).
    Retorna (ks, alphas, alpha_hat, alpha_err, k_lo, k_hi).
    """
    tau_sorted = np.sort(tau_star)[::-1]
    n = len(tau_sorted)

    # cumsum de log para calcular sum(log(tau[:k])) em O(1) por k
    log_tau = np.log(np.maximum(tau_sorted, 1e-300))
    cumlog = np.cumsum(log_tau)

    ks = list(range(2, n - 1))
    k_arr = np.array(ks)
    x_min_arr = tau_sorted[k_arr]
    valid_mask = x_min_arr > 0

    log_xmin = np.where(valid_mask, np.log(np.maximum(x_min_arr, 1e-300)), 0.0)
    # sum(log(tau[:k] / x_min)) = cumlog[k-1] - k*log(x_min)
    sum_log = cumlog[k_arr - 1] - k_arr * log_xmin
    alphas = np.where(valid_mask & (sum_log > 0), 1.0 + k_arr / sum_log, np.nan)

    # --- detecção de platô (varredura linear, já era O(n)) ---
    dalpha = np.abs(np.diff(alphas))
    best_start, best_len = None, 0
    current_start, current_len = 0, 1

    for i in range(1, len(dalpha)):
        if dalpha[i] < stability_threshold and not np.isnan(alphas[i]):
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_start = i
            current_len = 1

    if best_start is not None and best_len >= min_plateau_len:
        k_lo = ks[best_start]
        k_hi = ks[best_start + best_len - 1]
        plateau_vals = alphas[best_start: best_start + best_len]
        return ks, alphas, np.nanmean(plateau_vals), np.nanstd(plateau_vals), k_lo, k_hi
    else:
        xmin = np.percentile(tau_star, 75)
        tail = tau_star[tau_star >= xmin]
        alpha_fallback = 1 + len(tail) / np.sum(np.log(tail / xmin))
        return ks, alphas, alpha_fallback, np.nan, None, None


def hill_plot(pdf,ks, alphas, k_lo, k_hi, label=''):
    plt.figure()
    plt.plot(list(ks), alphas, label=label, lw=1.5)
    plt.axhline(2.0, color='orange', linestyle='--', lw=1, label=r'$\alpha = 2$')
    plt.xlabel('k (número de pontos da cauda)')
    plt.ylabel(r'$\hat{\alpha}(k)$')
    plt.ylim(0, 6)
    plt.legend()
    if k_lo is not None and k_hi is not None:
        plt.axvspan(k_lo, k_hi, alpha=0.15, color='blue', label='platô estável')
    pdf.savefig()
    plt.close()

def build_causal_regime_matrix(center, EMBED_current, WINDOW_current,
                               ret_O, ret_H, ret_L, ret_C):
    blocks = []
    for k in range(N_WINDOWS):
        end = center - k * WINDOW_current
        start = end - WINDOW_current
        if start < 0:
            continue
        segO = ret_O[start:end] - ret_O[start:end].mean()
        segH = ret_H[start:end] - ret_H[start:end].mean()
        segL = ret_L[start:end] - ret_L[start:end].mean()
        segC = ret_C[start:end] - ret_C[start:end].mean()

        if MODE == 'scalar':
            X = hankel_embed(segC, EMBED_current)
        elif MODE == 'indep':
            X = hankel_embed_ohlc_indep(segO, segH, segL, segC, EMBED_current)
        elif MODE == 'vector':
            X = hankel_embed_ohlc(segO, segH, segL, segC, EMBED_current)
        blocks.append(X)

    return np.vstack(blocks) if blocks else None


# ─────────────────────────────────────────────────────────────────────────────
# Funções extraídas de escopos aninhados (eram redefinidas a cada iteração)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ratio(K):
    """Decompõe a variação de energia em injeção e dissipação."""
    dK = K[1:] - K[:-1]
    injection = np.sum(dK * dK, axis=(1, 2))
    dissip = 2 * np.sum(K[:-1] * dK, axis=(1, 2))
    energy_change = np.sum(K[1:] ** 2, axis=(1, 2)) - np.sum(K[:-1] ** 2, axis=(1, 2))
    return dK, injection, dissip, energy_change


def mp_pdf(lam, var, q):
    """Densidade de Marchenko–Pastur."""
    lam_minus = var * (1 - np.sqrt(q)) ** 2
    lam_plus  = var * (1 + np.sqrt(q)) ** 2
    return np.where(
        (lam >= lam_minus) & (lam <= lam_plus),
        np.sqrt((lam_plus - lam) * (lam - lam_minus)) / (2 * np.pi * q * var * lam),
        0,
    )


def calc_FDT(pdf,C_store, W, build_plots):
    """Calcula e plota equilíbrio entre injeção e dissipação."""
    _, injection, dissip, energy_change = compute_ratio(C_store)
    kernel = np.ones(W) / W
    inj_local    = np.convolve(injection,     kernel, mode='valid')
    dis_local    = np.convolve(dissip,        kernel, mode='valid')
    energy_local = np.convolve(energy_change, kernel, mode='valid')
    ratio_local  = dis_local + inj_local - energy_local
    if build_plots:
        plt.figure()
        plt.plot(dis_local, label="Dissipação")
        plt.plot(inj_local, label="Injeção")
        plt.legend()
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(ratio_local)
        plt.ylabel("Ratio")
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(energy_local)
        plt.ylabel("Energy variation")
        pdf.savefig()
        plt.close()
    mean_D = np.mean(dis_local)
    mean_I = np.mean(inj_local)
    std_D = np.std(dis_local) / np.sqrt(len(dis_local))
    std_I = np.std(inj_local) / np.sqrt(len(inj_local))
    cov_DI = np.cov(dis_local, inj_local)[0,1] / len(dis_local)
    ratio = -mean_D / mean_I

    sigma_R = np.sqrt(
        (std_D**2) / (mean_I**2) +
        (mean_D**2 * std_I**2) / (mean_I**4) -
        2 * mean_D * cov_DI / (mean_I**3)
    )
    print("D:", -np.mean(dis_local),
          "I:", np.mean(inj_local),
          "R:", -np.mean(ratio_local),
          "Ratio:",ratio,
          "Ratio_err",sigma_R)
    return ratio,sigma_R


def calc_K(vecs_store, evals_store):
    """Reconstrói K_t = V diag(λ) V^T para cada passo temporal."""
    K_ = []
    for vecs_, lambda_ in zip(vecs_store, evals_store):
        if vecs_.size == 0:
            continue
        # broadcasting evita alocar np.diag(lambda_): (m,m) desnecessário
        K_.append((vecs_ * lambda_) @ vecs_.T)
    return np.array(K_)


def plot_beta_lags(ax, steps, values, label):
    """Gráfico log-log do MSD com referência β=1."""
    mid = len(steps) // 2
    x_ref = np.array([steps[0], steps[-1]], dtype=float)
    y_ref = values[mid] * (x_ref / steps[mid]) ** 1.0
    ax.loglog(x_ref, y_ref, 'k--', lw=0.9, label=r'$\beta=1$')
    ax.loglog(steps, values, 'o-', ms=4)
    ax.set_xlabel(r'Lag $s$')
    ax.set_ylabel(label)
    ax.legend(frameon=False, fontsize=7)


def prob_C_given_M(set_m_type, set_c, window):
    """P(compressão | mudança de m) com janela look-back."""
    if not set_m_type:
        return np.nan
    count = sum(
        any((t - k) in set_c for k in range(1, window + 1))
        for t in set_m_type
    )
    return count / len(set_m_type)


def prob_M_given_C(set_m_type, indices_compressao, window):
    """P(mudança de m | compressão) com janela look-forward."""
    if not indices_compressao:
        return np.nan
    count = sum(
        any((t + k) in set_m_type for k in range(1, window + 1))
        for t in indices_compressao
    )
    return count / len(indices_compressao)


def prob_M_given_CV(set_m_type, set_cv, window):
    """P(mudança de m | compressão ∩ variância alta) com janela look-forward."""
    if not set_cv:
        return np.nan
    count = sum(
        any((t + k) in set_m_type for k in range(1, window + 1))
        for t in set_cv
    )
    return count / len(set_cv)


def test_shift_effect(indices_up, indices_down, returns,
                      window_current, step_current,
                      horizon=5, max_shift=10):
    """Testa o retorno futuro em função do shift após transição de m."""
    results = []
    for shift in range(max_shift + 1):
        ret_up, ret_down = [], []
        for t in indices_up:
            start = window_current + t * step_current + shift
            end = start + horizon
            if end < len(returns):
                ret_up.append(np.sum(returns[start:end]))
        for t in indices_down:
            start = window_current + t * step_current + shift
            end = start + horizon
            if end < len(returns):
                ret_down.append(np.sum(returns[start:end]))

        if len(ret_up) > 5 and len(ret_down) > 5:
            t_stat, p_val = stats.ttest_ind(ret_down, ret_up, equal_var=False)
            results.append((shift, np.mean(ret_down), np.mean(ret_up), t_stat, p_val))
        else:
            results.append((shift, np.nan, np.nan, np.nan, np.nan))
    return results


def excursion_test(indices_up, indices_down, returns,
                   window_current, step_current,
                   horizon=10, shift=0):
    """MFE / MAE por grupo de transição."""
    mfe_up, mae_up, mfe_down, mae_down = [], [], [], []

    for t in indices_up:
        start = window_current + t * step_current + shift
        end = start + horizon
        if end < len(returns):
            path = np.cumsum(returns[start:end])
            mfe_up.append(np.max(path))
            mae_up.append(np.min(path))

    for t in indices_down:
        start = window_current + t * step_current + shift
        end = start + horizon
        if end < len(returns):
            path = np.cumsum(returns[start:end])
            mfe_down.append(np.max(path))
            mae_down.append(np.min(path))

    t_mfe, p_mfe = stats.ttest_ind(mfe_down, mfe_up, equal_var=False)
    t_mae, p_mae = stats.ttest_ind(mae_down, mae_up, equal_var=False)

    print("=== MFE (extensão máxima positiva) ===")
    print("m↓ média:", np.mean(mfe_down))
    print("m↑ média:", np.mean(mfe_up))
    print("t:", t_mfe, "p:", p_mfe)

    print("\n=== MAE (extensão máxima negativa) ===")
    print("m↓ média:", np.mean(mae_down))
    print("m↑ média:", np.mean(mae_up))
    print("t:", t_mae, "p:", p_mae)

    return mfe_up, mfe_down, mae_up, mae_down


# =========================
# PIPELINE PRINCIPAL
# =========================

def main_func(pdf,tf_file, n_value=5):

    df = pd.read_csv(tf_file, sep=';', encoding='utf-16')

    times = pd.to_datetime(df['time']).values
    O = df['open'].values
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values

    #embaralhamento
    #np.random.shuffle(C)
    #ruído branco
    #C_wn=np.random.normal(0.0,1.0,size=len(C))
    #AR(1) phi perto de 1 corelacão, phi=2 -> normal
    #phi=1.0
    if USE_AR1:
        C_ar1 = np.zeros(len(C))
        noise = np.random.normal(0.0,1.0,size=len(C))
        for t in range(1,len(C)): C_ar1[t] = PHI_AR1*C_ar1[t-1] + noise[t]
        C=np.exp(C_ar1)

    if TYPE == 'returns':
        ret_O = np.diff(np.log(O))
        ret_H = np.diff(np.log(H))
        ret_L = np.diff(np.log(L))
        ret_C = np.diff(np.log(C))
        times = times[1:]
        O, H, L, C = O[1:], H[1:], L[1:], C[1:]
    else:
        ret_O, ret_H, ret_L, ret_C = O, H, L, C



    alpha_vs_step    = []
    beta_vs_step     = []
    entropy_vs_step  = []
    mpl2error_vs_step = []
    _tau_sorted = _ccdf = _beta_lags = _beta_values = _beta_err = _beta_mask = None

    loop_values = STEP_LIST if MULTI_STEP_ANALYSIS else [STEP]

    for loop_current in loop_values:

        # ─────────────────────────────────
        # Inicialização de buffers
        # ─────────────────────────────────
        evecs_store      = {}
        tracked_modes    = []
        tau_axis         = []
        gap_t            = []
        var_total_t      = []
        var_struct_t     = []
        var_noise_t      = []
        alpha_t          = []
        bulk_evals_store = []
        bulk_vecs_store  = []
        struct_evals_store = []
        struct_vecs_store  = []
        eigenvalues_series = []
        lambda_plus_t    = []
        var_data_store   = []
        theta_modes_all  = []
        theta_modes_bulk = []
        gap_modes_bulk   = []
        theta_struct_t   = []
        m_t              = []
        prev_vecs_all    = None
        prev_vals_all    = None
        prev_vecs        = None
        prev_m           = None
        gap_min_t        = []
        C_store          = []
        C_store_struct   = []
        C_store_bulk     = []
        entropy_series   = []

        STEP_current   = STEP
        EMBED_current  = EMBED_DIM
        WINDOW_current = 5*EMBED_DIM #WINDOW

        if MULTI_STEP_ANALYSIS:
            if MULTI_STEP_TYPE == 'step':
                STEP_current = loop_current
            elif MULTI_STEP_TYPE == 'embed':
                EMBED_current  = loop_current
                WINDOW_current = n_value * EMBED_current

        print("Calculando STEP:", STEP_current)
        print("Calculando EMBED_DIM:", EMBED_current)
        print("Calculando WINDOW:", WINDOW_current)

        table_df ={}
        table_df['STEP'] = f"{STEP_current}"
        table_df['EMBED_DIM'] = f"{EMBED_current}"
        table_df['WINDOW'] = f"{WINDOW_current}"

        # ─────────────────────────────────
        # Loop principal em τ
        # ─────────────────────────────────
        for center in range(WINDOW_current, len(ret_C) - WINDOW_current, STEP_current):

            Xreg = build_causal_regime_matrix(
                center, EMBED_current, WINDOW_current, ret_O, ret_H, ret_L, ret_C)
            if Xreg is None:
                continue

            vals, vecs, Correlation = pca_cov(Xreg)
            C_store.append(Correlation)

            q = Xreg.shape[1] / Xreg.shape[0]
            var = np.mean(np.var(Xreg, axis=0, ddof=1))
            var_data_store.append(var)
            lam_plus = mp_lambda_plus(var, q)
            lambda_plus_t.append(lam_plus)

            structural = vals > lam_plus
            m = int(np.sum(structural))

            vecs_s    = vecs[:, :m]
            vals_s    = vals[:m]
            vecs_all  = vecs
            vals_all  = vals
            vecs_bulk = vecs[:, m:]
            vals_bulk = vals[m:]

            eigenvalues_series.append(vals)

            # reconstrução K = V diag(λ) V^T via broadcasting (sem np.diag)
            _Ks = (vecs_s    * vals_s)    @ vecs_s.T    #if m > 0 \
                  #else np.zeros((EMBED_current, EMBED_current))
            _Kb = (vecs_bulk * vals_bulk) @ vecs_bulk.T
            C_store_struct.append(_Ks)
            C_store_bulk.append(_Kb)

            if not MULTI_STEP_ANALYSIS:
                bulk_evals_store.append(vals_bulk)
                bulk_vecs_store.append(vecs_bulk)
                struct_evals_store.append(vals_s)
                struct_vecs_store.append(vecs_s)

            entropy_series.append(spectral_entropy(vals))
            m_t.append(m)

            # ─────────────────────────────────
            # Rotações (apenas no modo completo)
            # ─────────────────────────────────
            if BUILD_DEVEL and not MULTI_STEP_ANALYSIS:

                if prev_vecs_all is not None:
                    try:
                        n_modes = min(prev_vecs_all.shape[1], vecs_all.shape[1])

                        # 1) Theta modo-a-modo (vetorizado: sem loop sobre k)
                        dots = np.abs(np.einsum(
                            'ij,ij->j',
                            prev_vecs_all[:, :n_modes],
                            vecs_all[:, :n_modes]
                        ))
                        theta_k = np.arccos(np.clip(dots, 0.0, 1.0))
                        theta_modes_all.append(theta_k)

                        # 2) Theta estrutural (subespaço)
                        if prev_m is not None and m > 0 and prev_m > 0:
                            m_star = min(prev_m, m)
                            ang_struct = subspace_angles(
                                prev_vecs[:, :m_star], vecs_s[:, :m_star])
                            theta_struct_t.append(np.sqrt(np.sum(ang_struct ** 2)))
                        else:
                            theta_struct_t.append(np.nan)

                        # 3) Bulk modo-a-modo + gaps locais
                        if m < n_modes - 2:
                            theta_bulk = theta_k[m + 1:-1]
                            gap_bulk = [
                                min(abs(vals_all[k] - vals_all[k - 1]),
                                    abs(vals_all[k] - vals_all[k + 1]))
                                for k in range(m + 1, n_modes - 1)
                            ]
                            theta_modes_bulk.append(np.array(theta_bulk))
                            gap_modes_bulk.append(np.array(gap_bulk))
                        else:
                            theta_modes_bulk.append(np.array([]))
                            gap_modes_bulk.append(np.array([]))

                    except Exception:
                        theta_struct_t.append(np.nan)
                        theta_modes_all.append(np.array([]))
                        theta_modes_bulk.append(np.array([]))
                        gap_modes_bulk.append(np.array([]))
                else:
                    theta_struct_t.append(np.nan)
                    theta_modes_all.append(np.nan)
                    theta_modes_bulk.append(np.nan)
                    gap_modes_bulk.append(np.nan)

                # Projeção e relação omega–lambda
                omega_lambda_relation = []
                if m > 0:
                    for i in range(min(m, 5)):
                        phi_close = vecs_s[:, i][3::4]
                        mode_ts = Xreg[:, 3::4] @ phi_close
                        w = estimate_frequency_fourier(mode_ts)
                        omega_lambda_relation.append((w, vals_s[i]))

                if len(omega_lambda_relation) > 1:
                    ws = np.array([x[0] for x in omega_lambda_relation])
                    ls = np.array([x[1] for x in omega_lambda_relation])
                    ws = ws[ws > 0]
                    ls = ls[ls > 0]
                    if len(ws) > 1:
                        try:
                            coeffs = np.polyfit(np.log(ws), np.log(ls), 1)
                            alpha_t.append(coeffs[0])
                        except Exception:
                            alpha_t.append(np.nan)
                    else:
                        alpha_t.append(np.nan)
                else:
                    alpha_t.append(np.nan)

                # Tracking de autovalores (max_tracks colunas)
                row = [np.nan] * max_tracks
                for i in range(min(len(vals_s), max_tracks)):
                    row[i] = vals_s[i]
                tracked_modes.append(row)

                if m > 0 and len(evecs_store) < 5:
                    evecs_store[times[center]] = vecs_s.copy()

                # Gap espectral (vetorizado: sem loop sobre k)
                if m > 0:
                    # (m, n_total) — distância de cada modo estrutural a TODOS os outros
                    diff_matrix = np.abs(vals[:m, np.newaxis] - vals[np.newaxis, :])
                    diff_matrix[np.arange(m), np.arange(m)] = np.inf  # exclui auto-comparação
                    deltas = diff_matrix.min(axis=1)
                    gap_min_t.append(np.min(deltas))
                    G_tau = np.mean(1.0 / (deltas ** 2))
                    gap_t.append(G_tau)
                else:
                    gap_min_t.append(np.nan)
                    gap_t.append(np.nan)

                # Energia por componente
                var_total = np.sum(vals)
                var_struct = np.sum(vals[:m]) if m > 0 else 0.0
                var_noise  = np.sum(vals[m:])
                var_total_t.append(var_total)
                var_struct_t.append(var_struct / var_total)
                var_noise_t.append(var_noise / var_total)

                prev_vecs_all = vecs_all.copy()
                prev_vals_all = vals_all.copy()
                prev_vecs = vecs_s.copy()
                prev_m = m

                tau_axis.append(times[center])

        # ─────────────────────────────────────────────────
        # Pós-loop: reescala bulk para MP agregado
        # ─────────────────────────────────────────────────
        tracked_modes = np.array(tracked_modes, dtype=float)
        all_bulk_rescaled = []
        all_vals_rescaled = []
        all_q = []

        for t in range(len(eigenvalues_series)):
            vals = eigenvalues_series[t]
            m    = m_t[t]
            if m >= len(vals):
                continue
            vals_bulk_t = vals[m:]
            if len(vals_bulk_t) < 3:
                continue
            q_emp    = EMBED_current / (WINDOW_current - EMBED_current)
            var_data = var_data_store[t]
            vals_rescaled = vals_bulk_t / var_data
            all_bulk_rescaled.extend(vals_rescaled)
            all_vals_rescaled.append(vals / var_data)
            all_q.append(q_emp)

        all_bulk_rescaled = np.array(all_bulk_rescaled)

        # ─────────────────────────────────────────────────
        # Testes 1 e 2 (apenas modo completo)
        # ─────────────────────────────────────────────────
        if BUILD_DEVEL and not MULTI_STEP_ANALYSIS:
            gap_arr   = np.array(gap_t)
            angle_arr = np.array(theta_struct_t)

            # Teste 1 — theta_bulk,k ~ 1/gap_k²
            theta_list, gap_list = [], []
            for th, gp in zip(theta_modes_bulk, gap_modes_bulk):
                if isinstance(th, np.ndarray) and isinstance(gp, np.ndarray):
                    n = min(len(th), len(gp))
                    theta_list.extend(th[:n])
                    gap_list.extend(gp[:n])

            theta_arr = np.array(theta_list)
            gap_arr_bulk = np.array(gap_list)
            mask = (~np.isnan(theta_arr)) & (~np.isnan(gap_arr_bulk)) \
                   & (gap_arr_bulk > 0) & (theta_arr > 0)
            if np.sum(mask) > 10:
                coeffs = np.polyfit(
                    np.log(1 / gap_arr_bulk[mask] ** 2),
                    np.log(theta_arr[mask]), 1)
                print("Expoente log-log (theta_bulk vs 1/gap²):", coeffs[0])

            # Teste 2 — rotação bulk vs variância bulk
            theta_bulk_rms = np.array([
                np.sqrt(np.mean(th ** 2)) if isinstance(th, np.ndarray) and len(th) > 0
                else np.nan
                for th in theta_modes_bulk
            ])
            var_noise_arr = np.array(var_noise_t)
            mask2 = (~np.isnan(theta_bulk_rms)) & (~np.isnan(var_noise_arr))
            if np.sum(mask2) > 20:
                corr = np.corrcoef(theta_bulk_rms[mask2], var_noise_arr[mask2])[0, 1]
                print("Correlação theta_bulk vs variância do bulk:", corr)

        # ─────────────────────────────────────────────────
        # Índices de mudança de regime
        # ─────────────────────────────────────────────────
        if MULTI_STEP_ANALYSIS:
            indices_m_change = [i for i in range(1, len(m_t)) if m_t[i] != m_t[i - 1]]
        elif not BUILD_DEVEL:
            indices_m_change = [i for i in range(1, len(m_t)) if m_t[i] != m_t[i - 1]]
        elif BUILD_DEVEL:
            # Teste 3 — saltos espectrais
            dtheta = np.diff(angle_arr)
            threshold_theta = np.percentile(np.abs(dtheta), 95)

            jump_modes = jump_compression = jump_theta = jump_spec = 0
            indices_m_change  = []
            indices_compressao = []
            indices_rotacao   = []
            indices_dist      = []
            V_t               = []
            gaps_mean         = []
            gaps_min          = []

            for j in range(len(dtheta)):
                i = j + 1
                vals_now  = all_vals_rescaled[i]
                vals_prev = all_vals_rescaled[i - 1]
                m_prev = m_t[i - 1]
                m_now  = m_t[i]

                # 1) Crossing estrutural
                if m_prev != m_now:
                    jump_modes += 1
                    indices_m_change.append(i)

                # 2) Compressão estrutural
                struct = vals_now[:m_now]
                if len(struct) > 1:
                    gaps = np.abs(struct[:-1] - struct[1:]) / struct[:-1]
                    gap_min = np.min(np.abs(gaps))
                    gaps_mean.append(np.mean(np.abs(gaps)))
                    gaps_min.append(gap_min)
                    if gap_min < 0.05:
                        jump_compression += 1
                        indices_compressao.append(i)
                    mean_val = np.mean(struct)
                    var_val  = np.var(struct)
                    V = var_val / (mean_val ** 2)
                    V_t.append(V)
                else:
                    gaps_mean.append(np.nan)
                    gaps_min.append(np.nan)
                    V_t.append(np.nan)

                # 3) Rotação extrema
                if abs(dtheta[j]) > threshold_theta:
                    jump_theta += 1
                    indices_rotacao.append(i)

                # 4) Distância espectral estrutural
                if m_prev > 0 and m_now > 0:
                    n_common = min(m_prev, m_now)
                    dist = np.linalg.norm(vals_now[:n_common] - vals_prev[:n_common])
                    norm = np.linalg.norm(vals_prev[:n_common])
                    if dist / norm > 0.2:
                        jump_spec += 1
                        indices_dist.append(i)

            print("Candidatos com mudança estrutural:",
                  jump_modes, jump_compression, jump_theta, jump_spec)

            set_m    = set(indices_m_change)
            set_comp = set(indices_compressao)
            set_rot  = set(indices_rotacao)
            print("m ∩ compressão:", len(set_m & set_comp))
            print("m ∩ rotação   :", len(set_m & set_rot))
            print("comp ∩ rotação:", len(set_comp & set_rot))

            window = 5
            antecede   = 0
            gap_before = []
            V_before   = []

            for t in indices_m_change:
                for k in range(1, window + 1):
                    if (t - k) in set_comp:
                        antecede += 1
                        gap_before.append(gaps_min[t - k])
                        V_before.append(V_t[t - k])
                        break

            print("Crossings precedidos por compressão:", antecede)
            print("Gap médio antes de crossing:", np.nanmean(gap_before))
            print("Gap médio:", np.nanmean(gaps_mean))
            print("Média V antes de crossing:", np.nanmean(V_before))
            print("Média V global:", np.nanmean(V_t))

            p_global = len(indices_compressao) / len(gaps_min)
            p_cond   = antecede / len(indices_m_change)

            print("P(compressão):", p_global)
            print("P(compressão | crossing):", p_cond)

            # P(M | C)
            antec_cross = sum(
                any((t + k) in set_m for k in range(1, window + 1))
                for t in indices_compressao
            )
            p_M_given_C = antec_cross / len(indices_compressao)
            print("P(M | C):", p_M_given_C)
            print("P(M):", len(indices_m_change) / len(gaps_min))

            p_C        = p_global
            p_C_given_M = p_cond
            p_M        = len(indices_m_change) / len(gaps_min)
            RR_C_given_M = p_C_given_M / p_C
            RR_M_given_C = p_M_given_C / p_M
            print("Risk ratio C|M:", RR_C_given_M)
            print("Risk ratio M|C:", RR_M_given_C)

            n  = len(indices_m_change)
            k  = antecede
            p0 = p_C
            test = binomtest(k, n, p0, alternative='greater')
            print("p-value binomial:", test.pvalue)
            expected = n * p0
            std      = np.sqrt(n * p0 * (1 - p0))
            z        = (k - expected) / std
            print("z-score:", z)

            # Teste C ∩ V_alto
            N = len(gaps_min)
            set_M = set(indices_m_change)
            V_global = np.nanmean(V_t)
            indices_CV = [
                t for t in indices_compressao
                if not np.isnan(V_t[t - 1]) and V_t[t - 1] > V_global
            ]
            n_CV = len(indices_CV)

            count_M_given_CV = sum(
                any((t + k) in set_M for k in range(1, window + 1))
                for t in indices_CV
            )
            P_M  = len(indices_m_change) / N
            P_CV = n_CV / N
            P_M_given_CV = count_M_given_CV / n_CV if n_CV > 0 else np.nan

            print("P(M):", P_M)
            print("P(C ∩ V_alto):", P_CV)
            print("P(M | C ∩ V_alto):", P_M_given_CV)
            if not np.isnan(P_M_given_CV):
                print("Risk Ratio (M | C ∩ V):", P_M_given_CV / P_M)

            # Separação m↑ e m↓
            indices_M_up   = [j + 1 for j in range(len(m_t) - 1) if m_t[j + 1] > m_t[j]]
            indices_M_down = [j + 1 for j in range(len(m_t) - 1) if m_t[j + 1] < m_t[j]]
            print("Total M_up:", len(indices_M_up))
            print("Total M_down:", len(indices_M_down))

            set_C      = set(indices_compressao)
            set_M_up   = set(indices_M_up)
            set_M_down = set(indices_M_down)
            set_CV_set = set(indices_CV)

            # Usando funções modulares (não mais aninhadas)
            P_C_given_M_up   = prob_C_given_M(set_M_up,   set_C, window)
            P_C_given_M_down = prob_C_given_M(set_M_down, set_C, window)
            print("P(C | M_up):", P_C_given_M_up)
            print("P(C | M_down):", P_C_given_M_down)

            P_M_up_given_C   = prob_M_given_C(set_M_up,   indices_compressao, window)
            P_M_down_given_C = prob_M_given_C(set_M_down, indices_compressao, window)
            print("P(M_up | C):", P_M_up_given_C)
            print("P(M_down | C):", P_M_down_given_C)

            P_M_up_given_CV   = prob_M_given_CV(set_M_up,   set_CV_set, window)
            P_M_down_given_CV = prob_M_given_CV(set_M_down, set_CV_set, window)
            print("P(M_up | C ∩ V):", P_M_up_given_CV)
            print("P(M_down | C ∩ V):", P_M_down_given_CV)

            # Tendência de λ_m antes de m↓ e m↑
            lambda_diffs_down, lambda_diffs_up = [], []
            for t in indices_M_down:
                m_prev_t = m_t[t - 1]
                if m_prev_t > 0:
                    for kk in range(1, window + 1):
                        idx1, idx0 = t - kk, t - kk - 1
                        if idx0 >= 0:
                            lambda_diffs_down.append(
                                all_vals_rescaled[idx1][m_prev_t - 1] -
                                all_vals_rescaled[idx0][m_prev_t - 1])

            for t in indices_M_up:
                m_prev_t = m_t[t - 1]
                if m_prev_t > 0:
                    for kk in range(1, window + 1):
                        idx1, idx0 = t - kk, t - kk - 1
                        if idx0 >= 0:
                            lambda_diffs_up.append(
                                all_vals_rescaled[idx1][m_prev_t - 1] -
                                all_vals_rescaled[idx0][m_prev_t - 1])

            mean_down = np.mean(lambda_diffs_down)
            mean_up   = np.mean(lambda_diffs_up)
            print("Média Δλ antes de m ↓:", mean_down)
            print("Média Δλ antes de m ↑:", mean_up)

            t_stat_down, p_value_down = stats.ttest_1samp(lambda_diffs_down, 0)
            p_value_down_one_sided = p_value_down / 2 if mean_down < 0 else 1 - p_value_down / 2
            print("t-stat m ↓:", t_stat_down)
            print("p-value unilateral m ↓:", p_value_down_one_sided)

            t_stat_up, p_value_up = stats.ttest_1samp(lambda_diffs_up, 0)
            p_value_up_one_sided = p_value_up / 2 if mean_up < 0 else 1 - p_value_up / 2
            print("t-stat m ↑:", t_stat_up)
            print("p-value unilateral m ↑:", p_value_up_one_sided)

            # Retorno após transição
            horizon_ret = 5
            shift_ret   = 0
            returns_down = []
            returns_up   = []
            for t in indices_M_down:
                index = WINDOW_current + t * STEP_current
                if (index + horizon_ret + shift_ret) < len(ret_C):
                    returns_down.append(
                        np.sum(ret_C[index + shift_ret: index + shift_ret + horizon_ret]))
            for t in indices_M_up:
                index = WINDOW_current + t * STEP_current
                if (index + horizon_ret + shift_ret) < len(ret_C):
                    returns_up.append(
                        np.sum(ret_C[index + shift_ret: index + shift_ret + horizon_ret]))

            returns_down = np.array(returns_down)
            returns_up   = np.array(returns_up)
            print("Média retorno após m ↓:", np.mean(returns_down))
            print("Média retorno após m ↑:", np.mean(returns_up))
            t_stat, p_value = stats.ttest_ind(returns_down, returns_up, equal_var=False)
            print("t-stat diferença:", t_stat)
            print("p-value:", p_value)

            results_shift = test_shift_effect(
                indices_M_up, indices_M_down, ret_C,
                WINDOW_current, STEP_current,
                horizon=5, max_shift=12)
            for r in results_shift:
                print(f"Shift={r[0]} | mean↓={r[1]:.6e} | mean↑={r[2]:.6e} "
                      f"| t={r[3]:.3f} | p={r[4]:.4f}")

            mfe_up, mfe_down, mae_up, mae_down = excursion_test(
                indices_M_up, indices_M_down, ret_C,
                WINDOW_current, STEP_current,
                horizon=10, shift=0)

        if BUILD_PLOTS and BUILD_DEVEL:
            plt.figure()
            plt.plot(V_t, label="Variância normalizada dos autovalores")
            plt.legend()

            plt.figure()
            plt.hist(gap_before, bins=30, alpha=0.5, label="Antes crossing")
            plt.legend()

        # ─────────────────────────────────────────────────
        # Tempos de espera entre regimes
        # ─────────────────────────────────────────────────
        indices    = np.array(sorted(indices_m_change))
        waiting_times = np.diff(indices)
        #corre = np.corrcoef(waiting_times[:-1], waiting_times[1:])
        print("Número de intervalos:", len(waiting_times))
        print("Tempo médio:", np.mean(waiting_times))
        print("Mediana:", np.median(waiting_times))
        #print("Correlação:", corre)

        table_df[r'$\bar{\tau^*}$'] = f"{np.mean(waiting_times):.3f}"

        tau = np.array(waiting_times)
        tau_sorted = np.sort(tau)
        ccdf = 1 - np.arange(1, len(tau_sorted) + 1) / len(tau_sorted)

        if BUILD_PLOTS and BUILD_DEVEL:
            plt.figure()
            plt.loglog(tau_sorted, ccdf, marker='.', linestyle='none')
            plt.xlabel("tau")
            plt.ylabel("P(T > tau)")

        ks, alphas, alpha_hat, alpha_err, k_lo, k_hi = hill_estimator(tau)
        print(f"alpha estimado: {alpha_hat:.3f} ± {alpha_err:.3f}  platô: k={k_lo}–{k_hi}")
        if BUILD_PLOTS:
            hill_plot(pdf,ks, alphas, k_lo, k_hi)

        table_df[r'$\hat{\alpha}$'] = f"{alpha_hat:.3f} ± {alpha_err:.3f}"

        if MULTI_STEP_TYPE == 'step':
            alpha_vs_step.append((STEP_current, alpha_hat, alpha_err))
        elif MULTI_STEP_TYPE == 'embed':
            alpha_vs_step.append((EMBED_current, alpha_hat, alpha_err))

        if BUILD_PLOTS:
            plt.figure()
            plt.hist(waiting_times, bins=30, density=True)
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Tempo entre mudanças")
            plt.ylabel("Densidade (log log)")
            pdf.savefig()
            plt.close()
        # ─────────────────────────────────────────────────
        # FDT e balanço de energia (apenas modo completo)
        # ─────────────────────────────────────────────────
        C_store = np.array(C_store)
        W = 200
        #if not MULTI_STEP_ANALYSIS:
        ratio_fdt,ratio_ftd_err=calc_FDT(pdf,C_store, W, BUILD_PLOTS)
        table_df[r"$R_{FDT}$"]=f"{ratio_fdt:.3f} ± {ratio_ftd_err:.3f}"

        # ─────────────────────────────────────────────────
        # Estimativa de beta por subespaço
        # ─────────────────────────────────────────────────
        _C_struct_arr = np.array(C_store_struct)
        _C_bulk_arr   = np.array(C_store_bulk)
        _m_arr        = np.array(m_t)

        beta_total,  beta_err_total,  msd_total,  lags_total  = mbsw.estimate_beta(pdf,C_store,       BUILD_PLOTS)
        beta_struct, beta_err_struct, msd_struct, lags_struct  = mbsw.estimate_beta(pdf,_C_struct_arr, BUILD_PLOTS)
        beta_bulk,   beta_err_bulk,   msd_bulk,   lags_bulk    = mbsw.estimate_beta(pdf,_C_bulk_arr,   BUILD_PLOTS)

        table_df[r'$\hat{\beta_T}$'] = f"{beta_total:.3f} ± {beta_err_total:.3f}"
        table_df[r'$\hat{\beta_S}$'] = f"{beta_struct:.3f} ± {beta_err_struct:.3f}"
        table_df[r'$\hat{\beta_B}$'] = f"{beta_bulk:.3f} ± {beta_err_bulk:.3f}"

        if BUILD_PLOTS and BUILD_DEVEL:
            fig = plt.figure(figsize=(18, 6))
            gs  = gridspec.GridSpec(1, 3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            plot_beta_lags(ax1, lags_total,  msd_total,
                           r'$\mathbb{E}[\|\Delta\mathbf{X}\|_F^2]$')
            plot_beta_lags(ax2, lags_struct, msd_bulk,
                           r'$\mathbb{E}[\|\Delta\mathbf{X_{bulk}}\|_F^2]$')
            plot_beta_lags(ax3, lags_struct, msd_struct,
                           r'$\mathbb{E}[\|\Delta\mathbf{X_{struct}}\|_F^2]$')

        if STEP_current == 20 and MULTI_STEP_ANALYSIS:
            _tau_sorted  = tau_sorted
            _ccdf        = ccdf
            _beta_lags   = lags_struct
            _beta_values = msd_struct
            _beta_err    = beta_err_struct
            _beta_mask   = None

        if MULTI_STEP_TYPE == 'step':
            beta_vs_step.append((STEP_current, beta_struct, beta_err_struct))
        elif MULTI_STEP_TYPE == 'embed':
            beta_vs_step.append((EMBED_current, beta_struct, beta_err_struct))

        if BUILD_DEVEL and not MULTI_STEP_ANALYSIS:
            K_bulk   = calc_K(bulk_vecs_store,   bulk_evals_store)
            K_struct = calc_K(struct_vecs_store, struct_evals_store)

            for label, K in [("BULK", K_bulk), ("STRUCT", K_struct)]:
                _, injection, dissip, energy_change = compute_ratio(K)
                inj_local    = np.convolve(injection,     np.ones(W) / W, mode='valid')
                dis_local    = np.convolve(dissip,        np.ones(W) / W, mode='valid')
                energy_local = np.convolve(energy_change, np.ones(W) / W, mode='valid')
                ratio_local  = dis_local + inj_local - energy_local
                print(f"{label}: D:", -np.mean(dis_local),
                      "I:", np.mean(inj_local),
                      "R:", -np.mean(ratio_local),
                      "Ratio:", -np.mean(dis_local) / np.mean(inj_local))

            gap_min_t = np.array(gap_min_t)[:-1]
            m_mid     = np.array(m_t[:-1])
            mask      = ~np.isnan(dtheta)
            mask2     = ~np.isnan(dtheta) & ~np.isnan(gap_min_t)
            corr      = np.corrcoef(m_mid[mask], dtheta[mask])[0, 1]
            dm        = np.diff(m_t)
            corr2     = np.corrcoef(dm[mask], dtheta[mask])[0, 1]
            corr_gap  = np.corrcoef(gap_min_t[mask2], np.abs(dtheta[mask2]))[0, 1]
            print("Correlações <m,dtheta> <dm,dtheta> <gap_min,|dtheta|>:",
                  corr, corr2, corr_gap)

            # Teste 4 — normalidade do bulk
            C_bulk_sum = None
            count = 0
            for vecs_bulk_t, lambda_bulk_t in zip(bulk_vecs_store, bulk_evals_store):
                if vecs_bulk_t.size == 0:
                    continue
                C_bulk = (vecs_bulk_t * lambda_bulk_t) @ vecs_bulk_t.T
                if C_bulk_sum is None:
                    C_bulk_sum = np.zeros_like(C_bulk)
                C_bulk_sum += C_bulk
                count += 1

            C_bulk_mean = C_bulk_sum / count
            off_diag    = C_bulk_mean - np.diag(np.diag(C_bulk_mean))
            E_off_mean  = np.sum(off_diag ** 2) / np.sum(C_bulk_mean ** 2)
            diag_energy = np.sum(np.diag(C_bulk_mean) ** 2)
            off_energy  = np.sum(off_diag ** 2)
            R_mean      = np.sqrt(off_energy) / np.sqrt(diag_energy)
            print("E_off (médio):", E_off_mean)
            print("R (médio):", R_mean)

            # Teste 5 — variância do ângulo
            max_lag_msd = 20
            msd_angle = [
                np.nanmean((angle_arr[lag:] - angle_arr[:-lag]) ** 2)
                for lag in range(1, max_lag_msd)
            ]
            if BUILD_PLOTS and BUILD_DEVEL:
                plt.figure()
                plt.plot(range(1, max_lag_msd), msd_angle)
                plt.title("MSD da rotação espectral")

        # ─────────────────────────────────────────────────
        # Teste 6 — MP global reescalado
        # ─────────────────────────────────────────────────
        q_mean = np.mean(all_q)
        xs     = np.linspace(np.min(all_bulk_rescaled), np.max(all_bulk_rescaled), 400)

        lambda_minus_theory = (1 - np.sqrt(q_mean)) ** 2
        lambda_plus_theory  = (1 + np.sqrt(q_mean)) ** 2
        lambda_minus_emp    = np.min(all_bulk_rescaled)
        lambda_plus_emp     = np.max(all_bulk_rescaled)

        print(lambda_minus_emp, lambda_minus_theory)
        print(lambda_plus_emp,  lambda_plus_theory)
        print("Diferença percentual:",
              100 * np.abs(lambda_plus_emp - lambda_plus_theory) / lambda_plus_theory, "%")

        hist_vals, bin_edges = np.histogram(all_bulk_rescaled, bins=80, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mp_vals     = mp_pdf(bin_centers, 1.0, q_mean)
        dx          = bin_centers[1] - bin_centers[0]

        L2_error   = np.sum((hist_vals - mp_vals) ** 2) * dx
        L1_error   = np.sum(np.abs(hist_vals - mp_vals)) * dx
        norm_factor = np.sum(mp_vals ** 2) * dx
        L2_relative = L2_error / norm_factor

        print("Erro L2 =", L2_error)
        print("Erro L1 =", L1_error)
        print("Erro L2 relativo =", 100 * L2_relative, "%")

        table_df[r'$L^2$'] = f"{100 * L2_relative:.3f} %"

        table_df = pd.DataFrame([table_df])
        table_df = table_df.T
        table_df.columns = ['Valor']
        add_table(pdf,table_df,'Resultados')

        if MULTI_STEP_TYPE == 'step':
            mpl2error_vs_step.append((STEP_current, 100 * L2_relative))
        elif MULTI_STEP_TYPE == 'embed':
            mpl2error_vs_step.append((EMBED_current, 100 * L2_relative))

        if BUILD_PLOTS:
            fig1_prl, ax_mp = plt.subplots(figsize=(3.375, 2.6))
            lm_thy = (1 - np.sqrt(q_mean)) ** 2
            lp_thy = (1 + np.sqrt(q_mean)) ** 2

            ax_mp.bar(bin_centers, hist_vals, width=dx,
                      color='#4878CF', alpha=0.65, label='Bulk data')
            ax_mp.plot(xs, mp_pdf(xs, 1.0, q_mean),
                       color='#E84646', lw=1.6,
                       label=r'MP law ($q={:.2f}$)'.format(q_mean))
            ax_mp.axvline(lp_thy, color='#888888', lw=0.9, ls='--')
            ax_mp.axvline(lm_thy, color='#888888', lw=0.9, ls='--')
            ax_mp.text(lp_thy + 0.02, ax_mp.get_ylim()[1] * 0.55,
                       r'$\lambda_+$', fontsize=7, color='#555555')
            ax_mp.text(lm_thy - 0.01, ax_mp.get_ylim()[1] * 0.55,
                       r'$\lambda_-$', fontsize=7, color='#555555', ha='right')
            ax_mp.set_xlabel(r'$\lambda\,/\,\hat\sigma^2$')
            ax_mp.set_ylabel('Density')
            ax_mp.legend(frameon=False, fontsize=8)
            for sp in ['top', 'right']:
                ax_mp.spines[sp].set_visible(False)

            ax_mp.text(
                0.93, 0.75,
                f'Rel. $L^2$ error: {100 * L2_relative:.1f}%',#\n'
                #f'$\\lambda_+^{{\\rm emp}}={lambda_plus_emp:.3f}$\n'
                #f'$\\lambda_+^{{\\rm thy}}={lambda_plus_theory:.3f}$',
                transform=ax_mp.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.85))

            fig1_prl.tight_layout()
            if not MULTI_STEP_ANALYSIS:
                out_mp = os.path.splitext(tf_file)[0] + '_fig1_mp.pdf'
                fig1_prl.savefig(out_mp, bbox_inches='tight')
                fig1_prl.savefig(out_mp.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
                print(f"Figura 1 salva: {out_mp}")
            pdf.savefig()
            plt.close()

        if BUILD_DEVEL and not MULTI_STEP_ANALYSIS:
            # Teste 7 — estacionariedade
            theta_clean = angle_arr[~np.isnan(angle_arr)]
            if len(theta_clean) > 20:
                adf_result = adfuller(theta_clean)
                print("ADF stat:", adf_result[0])
                print("p-value:", adf_result[1])
            else:
                print("Série muito curta para ADF")

            # Teste 8 — scaling do MSD
            dtheta_clean = np.diff(theta_clean)
            print("Var(Δtheta):", np.var(dtheta_clean))
            theta_mid = theta_clean[:-1]
            corr = np.corrcoef(theta_mid, dtheta_clean)[0, 1]
            print("Corr(theta, Δtheta):", corr)

            # Teste 9 — drift F(K)
            bins      = np.linspace(min(theta_mid), max(theta_mid), 20)
            digitized = np.digitize(theta_mid, bins)
            drift, centers_drift = [], []
            for i in range(1, len(bins)):
                mask_bin = digitized == i
                if np.sum(mask_bin) > 10:
                    drift.append(np.mean(dtheta_clean[mask_bin]))
                    centers_drift.append(np.mean(theta_mid[mask_bin]))

            if BUILD_PLOTS and BUILD_DEVEL:
                plt.figure()
                plt.plot(centers_drift, drift)
                plt.title("Estimativa de F(theta)")

            # Teste 10 — normalidade dos incrementos
            jb = jarque_bera(dtheta_clean)
            print("JB stat:", jb.statistic)
            print("p-value:", jb.pvalue)

        # Teste 11 — entropia
        S_norm_med = np.mean(entropy_series)
        print("Entropia média:", S_norm_med)
        if MULTI_STEP_TYPE == 'step':
            entropy_vs_step.append((STEP_current, S_norm_med))
        elif MULTI_STEP_TYPE == 'embed':
            entropy_vs_step.append((EMBED_current, S_norm_med))

        if BUILD_PLOTS and BUILD_DEVEL:
            plt.figure()
            plt.plot(entropy_series)
            plt.title("Entropia Espectral Normalizada")

        # ─────────────────────────────────────────────────
        # Gráficos de desenvolvimento
        # ─────────────────────────────────────────────────
        if BUILD_PLOTS and BUILD_DEVEL:
            idx_tau  = np.nanargmax(m_t)
            tau_sel  = tau_axis[idx_tau]
            taus_validos = np.array(sorted(evecs_store.keys()))
            idx      = np.argmin(np.abs(taus_validos - tau_sel))
            vecs_sel = evecs_store[taus_validos[idx]]
            print("Gerando gráficos")

            fig = plt.figure(figsize=(18, 18))
            gs  = gridspec.GridSpec(4, 2)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(theta_struct_t, -np.log(gap_t))
            ax1.set_title("Log do Gap espectral mínimo -ln(Δ(τ))")

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(tau_axis, var_noise_t)
            ax2.set_title("Variância relativa do resíduo σ²η(τ)")

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(tau_axis, theta_struct_t)
            ax3.set_title("Rotação média dos autovetores θ(τ)")

            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(tau_axis, m_t)
            ax4.set_title("Número de modos estruturais m(τ)")

            ax5 = fig.add_subplot(gs[2, 0])
            for i in range(min(max_tracks, tracked_modes.shape[1])):
                ax5.plot(tau_axis, tracked_modes[:, i], label=f"λ{i + 1}")
            ax5.set_title("Autovalores estruturais")
            ax5.legend()

            ax6 = fig.add_subplot(gs[2, 1])
            for i in range(min(6, vecs_sel.shape[1])):
                plot_mode(vecs_sel[:, i], ax6, f"Modo {i + 1}")
            ax6.legend()

            plt.tight_layout()
            out_name = os.path.splitext(tf_file)[0] + "_4D_" + MODE + "_" + TYPE + ".png"
            plt.savefig(out_name, dpi=300)

        if BUILD_PLOTS:
            plt.show()

    if MULTI_STEP_ANALYSIS:
        return {
            'alpha_vs_step':     alpha_vs_step,
            'beta_vs_step':      beta_vs_step,
            'entropy_vs_step':   entropy_vs_step,
            'mpl2error_vs_step': mpl2error_vs_step,
            'tau_sorted':        _tau_sorted,
            'ccdf':              _ccdf,
            'beta_lags':         _beta_lags,
            'beta_values':       _beta_values,
            'beta_err':          _beta_err,
            'beta_mask':         _beta_mask,
        }

#if MULTI_STEP_TYPE == 'step':
def plot_fig_step(step,muti_data,tf_file):
    # ── desempacota multi_data em dicionários por tf_label ───────────
    keys = ('alpha_vs_step', 'beta_vs_step', 'entropy_vs_step',
            'mpl2error_vs_step', 'tau_sorted', 'ccdf',
            'beta_lags', 'beta_values', 'beta_err', 'beta_mask')
    (alpha_vs_step, beta_vs_step, entropy_vs_step, mpl2error_vs_step,
     _tau_sorted, _ccdf, _beta_lags, _beta_values, _beta_err, _beta_mask) = (
        {tf: multi_data[tf][k] for tf in TIMEFRAME_LABELS} for k in keys
    )

    steps_list = {}
    betas_list = {}
    alphas_list = {}
    beta_errs_list  = {}
    alpha_errs_list = {}
    for tf_label in TIMEFRAME_LABELS:
        steps_list[tf_label], betas_list[tf_label], beta_errs_list[tf_label] = \
            zip(*beta_vs_step[tf_label])
        steps_list[tf_label], alphas_list[tf_label], alpha_errs_list[tf_label] = \
            zip(*alpha_vs_step[tf_label])

    fig2_prl = plt.figure(figsize=(3.375, 3.8))
    gs2 = gridspec.GridSpec(2, 2, figure=fig2_prl,
                            left=0.03, right=0.97,
                            bottom=0.03, top=0.97,
                            wspace=0.48, hspace=0.46)

    # painel (a): E[||ΔK_struct||²] vs lag
    ax_a = fig2_prl.add_subplot(gs2[0])
    try:
        for tf_label in TIMEFRAME_LABELS:
            mid   = len(_beta_lags[tf_label]) // 2
            x_ref = np.array([_beta_lags[tf_label][0], _beta_lags[tf_label][-1]], dtype=float)
            y_ref = _beta_values[tf_label][mid] * (x_ref / _beta_lags[tf_label][mid]) ** 1.0
            lw_ref = 0.9
            if tf_label == 'M5':
                ax_a.loglog(x_ref, y_ref, 'k--', lw=lw_ref)
            else:
                ax_a.loglog(x_ref, y_ref, 'k--', lw=lw_ref, label=r'$\beta=1$')
            ax_a.loglog(_beta_lags[tf_label], _beta_values[tf_label], 'o-',
                        color=tf_colors[tf_label], ms=4, label=tf_label)
        ax_a.set_xlabel(r'Lag $s$')
        ax_a.set_ylabel(r'$\mathbb{E}[\|\Delta\mathbf{X}_{struct}\|_F^2]$')
        ax_a.legend(frameon=False, fontsize=7)
    except NameError:
        ax_a.text(0.5, 0.5, 'Rode primeiro\ncom MULTI_STEP_ANALYSIS=False',
                  ha='center', va='center', transform=ax_a.transAxes, fontsize=7)
    ax_a.set_title('(a)', loc='left', fontweight='bold')
    for sp in ['top', 'right']:
        ax_a.spines[sp].set_visible(False)

    # painel (b): CCDF τ*
    ax_b = fig2_prl.add_subplot(gs2[1])
    try:
        ts0 = _tau_sorted[TIMEFRAME_LABELS[0]]
        bins_log = np.logspace(np.log10(max(ts0.min(), 1)), np.log10(ts0.max()), 25)
        ax_b.hist(ts0, bins=bins_log, density=True,
                  color=tf_colors[TIMEFRAME_LABELS[0]], alpha=0.4,
                  label=TIMEFRAME_LABELS[0])
        counts_tmp, edges_tmp = np.histogram(ts0, bins=bins_log, density=True)
        centers_tmp = np.sqrt(edges_tmp[:-1] * edges_tmp[1:])
        i_peak  = np.argmax(counts_tmp)
        A_fit   = counts_tmp[i_peak] * centers_tmp[i_peak] ** 2
        t_fit   = centers_tmp[counts_tmp > 0]
        ax_b.loglog(t_fit, A_fit / t_fit ** 2, 'k--', lw=0.9, label=r'$\alpha=2$')
        ax_b.set_xlabel(r'$\tau^*$')
        ax_b.set_ylabel(r'$p(\tau^*)$')
        ax_b.legend(frameon=False, fontsize=7)
    except NameError:
        ax_b.text(0.5, 0.5, 'Rode primeiro\ncom MULTI_STEP_ANALYSIS=False',
                  ha='center', va='center', transform=ax_b.transAxes, fontsize=7)
    ax_b.set_title('(b)', loc='left', fontweight='bold')
    for sp in ['top', 'right']:
        ax_b.spines[sp].set_visible(False)

    # painel (c): β vs STEP
    ax_c = fig2_prl.add_subplot(gs2[2])
    for tf_label in TIMEFRAME_LABELS:
        ax_c.errorbar(steps_list[tf_label], betas_list[tf_label],
                      yerr=beta_errs_list[tf_label],
                      fmt='o-', color=tf_colors[tf_label],
                      ms=4, lw=1.0, elinewidth=0.8, capsize=2, label=tf_label)
    ax_c.axhline(1.0, color='k', lw=0.9, ls='--', label=r'$\beta=1$')
    ax_c.axhline(0.0, color='#AAAAAA', lw=0.6, ls=':')
    ax_c.set_xlabel(r'$\Delta\tau_0$')
    ax_c.set_ylabel(r'$\hat\beta_{struct}$')
    ax_c.legend(frameon=False, fontsize=7, loc='lower left')
    ax_c.set_title('(c)', loc='left', fontweight='bold')
    ax_c.axvspan(step-5,step+5, alpha=0.2, color='#4DAF4A', label=r'$\Delta\tau_{\rm opt}$')
    for sp in ['top', 'right']:
        ax_c.spines[sp].set_visible(False)

    # painel (d): α vs STEP
    ax_d = fig2_prl.add_subplot(gs2[3])
    for tf_label in TIMEFRAME_LABELS:
        ax_d.errorbar(steps_list[tf_label], alphas_list[tf_label],
                      yerr=alpha_errs_list[tf_label],
                      fmt='o-', color=tf_colors[tf_label],
                      ms=4, lw=1.0, elinewidth=0.8, capsize=2, label=tf_label)
    ax_d.axhline(2.0, color='k', lw=0.9, ls='--', label=r'$\alpha=2$')
    ax_d.axhline(0.0, color='#AAAAAA', lw=0.6, ls=':')
    ax_d.set_xlabel(r'$\Delta\tau_0$')
    ax_d.set_ylabel(r'$\hat\alpha$')
    ax_d.legend(frameon=False, fontsize=7)
    ax_d.set_title('(d)', loc='left', fontweight='bold')
    for sp in ['top', 'right']:
        ax_d.spines[sp].set_visible(False)

    fig2_prl.savefig(tf_file.split('_')[0] + '_fig2_exponents.pdf', bbox_inches='tight')
    fig2_prl.savefig(tf_file.split('_')[0] + '_fig2_exponents.png', dpi=300,
                     bbox_inches='tight')
    print("Figura 2 salva.")
    #pdf.savefig()
    plt.close()

def plot_fig_embed(embed,multi_data,tf_file):
# ── FIGURA 3 PRL: β vs EMBED_DIM ──────────────────────────────────────────
#if MULTI_STEP_TYPE == 'embed':
    steps_list = {}
    betas_list = {}
    beta_errs_list = {}
    for tf_label in TIMEFRAME_LABELS:
        for n_value in N_STEPS:
            index = tf_label + str(n_value)
            steps_list[index], betas_list[index], beta_errs_list[index] = \
                zip(*multi_data[index]['beta_vs_step'])

    fig3_prl, ax3 = plt.subplots(figsize=(3.375, 2.6))
    for tf_label in TIMEFRAME_LABELS:
        for n_value in N_STEPS:
            index = tf_label + str(n_value)
            ax3.errorbar(list(steps_list[index]), list(betas_list[index]),
                         yerr=list(beta_errs_list[index]),
                         fmt='o-', ms=5, lw=1.4, elinewidth=0.8, capsize=2,
                         label=tf_label + " " + str(n_value) + "x")

    ax3.axvspan(embed-10, embed+10, alpha=0.18, color='#F0A500', label=r'$L_{\rm opt}$')
    ax3.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax3.set_xlabel(r'Embedding dimension $L$')
    ax3.set_ylabel(r'$\hat\beta_{struct}$')
    ax3.legend(frameon=False, fontsize=7, loc='lower right')
    for sp in ['top', 'right']:
        ax3.spines[sp].set_visible(False)

    fig3_prl.tight_layout()
    fig3_prl.savefig(tf_file.split('_')[0] + '_fig3_Lopt.pdf', bbox_inches='tight')
    fig3_prl.savefig(tf_file.split('_')[0] + '_fig3_Lopt.png', dpi=300,
                     bbox_inches='tight')
    print("Figura 3 salva.")
    #pdf.savefig()
    plt.close()

def add_table(pdf, df,title):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=14, pad=20)
    pdf.savefig(fig)
    plt.close(fig)

def wait_file_ready(path, timeout=30, interval=0.5):
    """Aguarda até o arquivo existir e parar de crescer."""
    deadline = time.time() + timeout
    prev_size = -1
    while time.time() < deadline:
        if os.path.exists(path):
            curr_size = os.path.getsize(path)
            if curr_size > 0 and curr_size == prev_size:
                return True  # arquivo estável
            prev_size = curr_size
        time.sleep(interval)
    raise TimeoutError(f"Arquivo {path} não ficou pronto em {timeout}s")

# #feitos: T10$D,BGI$D,ITUB4,BIT$D,AAPL34,SEQR11,ERAT5,ERAP5
# #coletados: WSP$D,ICF$D,CCM$D,VALE3,WIN$D,BABA34,MELI34,NVDC34,TSLA34,ABEV3,WDO$D,BITH11,BBOI11,CORN11,GOLD11,NASD11,,MXRF11
# =========================
# CONFIGURAÇÕES
# =========================
ASSET_LIST = ["PETR4"]
#ASSET_LIST = ["BITH11","BBOI11","CORN11","GOLD11","NASD11","MXRF11"]
#["BGI$D","ICF$D","CCM$D","BIT$D","AAPL34","SEQR11","WSP$D","BABA34","MELI34","NVDC34","TSLA34"]
#["T10$D"]
#["BBAS3","BBDC4","B3SA3","WIN$N","DI1$N"]
#["PETR4","ITUB4","VALE3","WIN$D","ABEV3","WDO$N"] - completo
#                "BIT$D","AAPL34","SEQR11","WSP$D",\
#                "BABA34","MELI34","NVDC34","TSLA34",\
#                "BITH11","BBOI11","CORN11","GOLD11","NASD11","MXRF11" ]  # <-- lista de ativos

#ASSET_LIST = ["PETR4","T10$D","BGI$D","ITUB4","BIT$D","AAPL34","SEQR11","WSP$D",\
#                "ICF$D","CCM$D","VALE3","WIN$D","BABA34","MELI34","NVDC34","TSLA34",\
#                "ABEV3","WDO$N","BITH11","BBOI11","CORN11","GOLD11","NASD11","MXRF11" ]  # <-- lista de ativos
BEST_STEP={'PETR4': 20}#,'ITUB4':20,'VALE3':40,'WIN$D':45,'ABEV3':40,'WDO$N':45}
SKIP_STEP=True #para pular a etapa de step e usar os valores otimizados

BEST_EMBED={'PETR4': 80}#,'ITUB4':100,'VALE3':120,'WIN$D':70,'ABEV3':60,'WDO$N':80}
SKIP_EMBED=True #para pular a etapa de embed e usar os valores otimizados

SKIP_SINGLE=False # para pular a etapa single
USE_AR1 = False
PHI_AR1 = 1.0 #0 ruido branco, 1 GBM

WINDOW = 400
N_WINDOWS = 1
STEP = 20
EMBED_DIM = 70
PROJECTION_HORIZON = 50
TYPE = 'returns'
MODE = 'scalar'
max_tracks = 8
BUILD_PLOTS = True
BUILD_DEVEL = False

# ================================
# ANALISE MULTI-STEP
# ================================
MULTI_STEP_ANALYSIS = True
MULTI_STEP_TYPE = 'step'        # <-- valor padrão, garante que sempre existe
STEP_LIST_STEP  = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
#STEP_LIST_STEP  = [80,90,100]
#STEP_LIST_EMBED = [20,30,40]
STEP_LIST_EMBED = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]

# ================================================================
# TIMEFRAMES (usados apenas com MULTI_STEP_ANALYSIS)
# ================================================================
TIMEFRAME_SUFFIX = ['_M5.csv', '_H1.csv']
TIMEFRAME_LABELS = ['M5', 'H1']
tf_colors = {'M5': '#2166AC', 'H1': '#D6604D'}
N_STEPS = [4, 5, 6]
resultados = []
# ================================================================
# LOOP PRINCIPAL SOBRE ATIVOS E MODOS
# ================================================================
for ASSET in ASSET_LIST:
    print("------------------ "+ASSET+" ------------------")
    TIMEFRAME_FILES = [ASSET + sfx for sfx in TIMEFRAME_SUFFIX]

    for MULTI_STEP_ANALYSIS in (True,False):
        if MULTI_STEP_ANALYSIS:
            for MULTI_STEP_TYPE in ('step', 'embed'):

                STEP_LIST = STEP_LIST_STEP if MULTI_STEP_TYPE == 'step' else STEP_LIST_EMBED
                pdf_name  = f"{ASSET}_{MULTI_STEP_TYPE}.pdf"
                multi_data = {}


                if MULTI_STEP_TYPE == 'step':
                    if SKIP_STEP:
                        STEP=BEST_STEP[ASSET]
                    else:
                        EMBED_DIM = 70 #reseta o valor inicial de EMBED_DIM
                        with PdfPages(pdf_name) as pdf:
                            for tf_file, tf_label in zip(TIMEFRAME_FILES, TIMEFRAME_LABELS):
                                multi_data[tf_label] = main_func(pdf, tf_file)
                            #for tf_label in TIMEFRAME_LABELS:
                            #    print(tf_label, multi_data[tf_label].get('beta_lags'))
                            pdf.close()
                        print("calculando STEP ótimo")
                        #time.sleep(10)
                        pdf_path = ASSET+"_step.pdf"
                        output_csv = ASSET+"_step.csv"
                        wait_file_ready(pdf_path)
                        #epdf.pdf_to_csv(pdf_path, output_csv)
                        for _ in range(20):
                            epdf.pdf_to_csv(pdf_path, output_csv)
                            df_check = pd.read_csv(output_csv)
                            if len(df_check) > 0:
                                break
                            print("CSV ainda vazio, aguardando...")
                            time.sleep(1)
                        else:
                            raise RuntimeError("PDF não pôde ser lido após 20 tentativas")
                        merged_step, best_step = epdf.compute_instability_min(output_csv,'step')
                        STEP=int(best_step['STEP'])
                        plot_fig_step(STEP, multi_data,tf_file)
                    print("Step finalizado ", STEP)


                else:  # 'embed'
                    if SKIP_EMBED:
                        EMBED_DIM=BEST_EMBED[ASSET]
                    else:
                        with PdfPages(pdf_name) as pdf:
                            print("começando embed")
                            for tf_file, tf_label in zip(TIMEFRAME_FILES, TIMEFRAME_LABELS):
                                for n_value in N_STEPS:
                                    index = tf_label + str(n_value)
                                    multi_data[index] = main_func(pdf, tf_file, n_value)
                            pdf.close()
                        print("calculando EMBED_DIM ótimo")
                        #time.sleep(10)
                        pdf_path = ASSET+"_embed.pdf"
                        output_csv = ASSET+"_embed.csv"
                        wait_file_ready(pdf_path)

                        for _ in range(20):
                            epdf.pdf_to_csv(pdf_path, output_csv)
                            df_check = pd.read_csv(output_csv)
                            if len(df_check) > 0:
                                break
                            print("CSV ainda vazio, aguardando...")
                            time.sleep(1)
                        else:
                            raise RuntimeError("PDF não pôde ser lido após 20 tentativas")
                        #epdf.pdf_to_csv(pdf_path, output_csv)

                        merged_embed, best_embed = epdf.compute_instability_min(output_csv,'embed')
                        EMBED_DIM=int(best_embed['EMBED_DIM'])
                        plot_fig_embed(EMBED_DIM, multi_data,tf_file)
                    print("Embed finalizado ", EMBED_DIM)


                print(f"[{ASSET}] modo '{MULTI_STEP_TYPE}' concluído → {pdf_name}")

        else:
            if USE_AR1:
                with PdfPages(f"AR1_{PHI_AR1}.pdf") as pdf:
                    main_func(pdf, TIMEFRAME_FILES[0])
            else:
                if not SKIP_SINGLE:
                    for tf_file, tf_label in zip(TIMEFRAME_FILES, TIMEFRAME_LABELS):
                        with PdfPages(os.path.splitext(tf_file)[0] + ".pdf") as pdf:
                            main_func(pdf, tf_file)

            if USE_AR1:
                pdf_path = f"AR1_{PHI_AR1}.pdf"
                output_csv = f"AR1_{PHI_AR1}.csv"
                epdf.pdf_to_csv(pdf_path,output_csv )
                alpha,alpha_std,ratio_fdt,ratio_fdt_std,beta_total,beta_total_std,beta_struct,beta_struct_std,L2 = epdf.collect_data(output_csv)

                resultados.append({
                        'ATIVO': "AR(1)",
                        'PHI': PHI_AR1,
                        'ALPHA': f"{alpha:.3f} ± {alpha_std:.3f}",
                        'RATIO': f"{ratio_fdt:.3f} ± {ratio_fdt_std:.3f}",
                        'BETA_T': f"{beta_total:.3f} ± {beta_total_std:.3f}",
                        'BETA_S': f"{beta_struct:.3f} ± {beta_struct_std:.3f}",
                        'L2': f"{L2:.3f}",
                    })
            else:
                alpha = {}
                alpha_std = {}
                ratio_fdt = {}
                ratio_fdt_std = {}
                beta_total = {}
                beta_total_std = {}
                beta_struct = {}
                beta_struct_std = {}
                L2 = {}
                for tf_file, tf_label in zip(TIMEFRAME_FILES, TIMEFRAME_LABELS):
                    pdf_path = os.path.splitext(tf_file)[0] + ".pdf"
                    output_csv = os.path.splitext(tf_file)[0]+"_results_" + ".csv"
                    epdf.pdf_to_csv(pdf_path,output_csv )
                    alpha[tf_label],alpha_std[tf_label],ratio_fdt[tf_label],ratio_fdt_std[tf_label],beta_total[tf_label],beta_total_std[tf_label],beta_struct[tf_label],beta_struct_std[tf_label],L2[tf_label] = epdf.collect_data(output_csv)

                resultados.append({
                        'ATIVO': ASSET,
                        'BEST_STEP': STEP,
                        'BEST_EMBED_DIM': EMBED_DIM,
                        'ALPHA_H1': f"{alpha['H1']:.3f} ± {alpha_std['H1']:.3f}",
                        'RATIO_H1': f"{ratio_fdt['H1']:.3f} ± {ratio_fdt_std['H1']:.3f}",
                        'BETA_T_H1': f"{beta_total['H1']:.3f} ± {beta_total_std['H1']:.3f}",
                        'BETA_S_H1': f"{beta_struct['H1']:.3f} ± {beta_struct_std['H1']:.3f}",
                        'L2_H1': f"{L2['H1']:.3f}",
                        'ALPHA_M5': f"{alpha['M5']:.3f} ± {alpha_std['M5']:.3f}",
                        'RATIO_M5': f"{ratio_fdt['M5']:.3f} ± {ratio_fdt_std['M5']:.3f}",
                        'BETA_T_M5': f"{beta_total['M5']:.3f} ± {beta_total_std['M5']:.3f}",
                        'BETA_S_M5': f"{beta_struct['M5']:.3f} ± {beta_struct_std['M5']:.3f}",
                        'L2_M5': f"{L2['M5']:.3f}"
                    })

            print(f"[{ASSET}] modo single concluído.")

plt.show()
print("Todos os ativos concluídos.")
df_resultados = pd.DataFrame(resultados)
if USE_AR1:
    df_resultados.to_csv(f"final_results_AR1_{PHI_AR1}.csv", index=False)
else:
    df_resultados.to_csv("final_results.csv", index=False)
print("\nTabela final:")
print(df_resultados)
plt.show()


# # =========================
# # CONFIGURAÇÕES
# # =========================
# ASSET = "PETR4"
# CSV_FILE = ASSET+"_H1.csv"
#
# WINDOW = 400           # tamanho da janela individual
# N_WINDOWS = 1         # quantas janelas vizinhas compõem o mesmo regime
# STEP = 20             # passo do τ
# EMBED_DIM = 70        # embedding temporal
# PROJECTION_HORIZON = 50  # horizonte da projeção
# TYPE = 'returns'      # 'returns' ou 'prices'
# MODE = 'scalar'       # 'scalar', 'indep' ou 'vector'
# max_tracks = 8
# BUILD_PLOTS = True
# BUILD_DEVEL = False   # habilita cálculos e gráficos em desenvolvimento
#
# # ================================
# # ANALISE MULTI-STEP
# # ================================
# MULTI_STEP_ANALYSIS = False
# MULTI_STEP_TYPE = 'step'  # 'step' ou 'embed'
# if MULTI_STEP_TYPE == 'step':
#     STEP_LIST = [5, 10, 20, 30, 50, 100]
# elif MULTI_STEP_TYPE == 'embed':
#     STEP_LIST = [20, 40, 60, 80, 100, 120, 140, 160]
#     # ================================================================
#     # FIGURAS DE PUBLICAÇÃO PRL — sequência de execução:
#     #
#     #  Passo 1 — Figura 1 + salvar dados intermediários:
#     #    MULTI_STEP_ANALYSIS = False  |  BUILD_PLOTS = True
#     #
#     #  Passo 2 — Figura 2 (painéis a, b, c):
#     #    MULTI_STEP_ANALYSIS = True   |  MULTI_STEP_TYPE = 'step'
#     #
#     #  Passo 3 — Figura 3 (β vs L):
#     #    MULTI_STEP_ANALYSIS = True   |  MULTI_STEP_TYPE = 'embed'
#     # ================================================================
# # ================================================================
# # TIMEFRAMES (usados apenas com MULTI_STEP_ANALYSIS)
# # ================================================================
# TIMEFRAME_FILES  = [ASSET+'_M5.csv', ASSET+'_H1.csv']
# TIMEFRAME_LABELS = ['M5', 'H1']
# tf_colors = {'M5': '#2166AC', 'H1': '#D6604D'}
# N_STEPS   = [4, 5, 6] #multiplicadores para o moode embed
#
# multi_data = {}
#
# if MULTI_STEP_ANALYSIS and MULTI_STEP_TYPE == 'step':
#     with PdfPages(CSV_FILE.split('_')[0] + "_" + MULTI_STEP_TYPE + ".pdf") as pdf:
#         for tf_file, tf_label in zip(TIMEFRAME_FILES, TIMEFRAME_LABELS):
#             multi_data[tf_label] = main_func(pdf,tf_file)
#         plot_fig_step(pdf,multi_data)
#
# elif MULTI_STEP_ANALYSIS and MULTI_STEP_TYPE == 'embed':
#     with PdfPages(CSV_FILE.split('_')[0] + "_" + MULTI_STEP_TYPE + ".pdf") as pdf:
#         for tf_file, tf_label in zip(TIMEFRAME_FILES, TIMEFRAME_LABELS):
#             for n_value in N_STEPS:
#                 index = tf_label + str(n_value)
#                 multi_data[index] = main_func(pdf,tf_file, n_value)
#         plot_fig_embed(pdf,multi_data)
#
# else:
#     with PdfPages(os.path.splitext(CSV_FILE)[0] + ".pdf") as pdf:
#         main_func(pdf,CSV_FILE)
#
# # ================================================================
# # FIGURAS DE PUBLICAÇÃO (apenas MULTI_STEP_ANALYSIS)
# # ================================================================
# #if MULTI_STEP_ANALYSIS:
#
# plt.show()
#
# print("Concluído.")