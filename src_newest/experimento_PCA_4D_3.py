import os
import sys
import gc
import time
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS',   str(os.cpu_count()))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(os.cpu_count()))
os.environ.setdefault('MKL_NUM_THREADS',   str(os.cpu_count()))
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
import result_cache

FILE_ENCODING  = 'utf-8'    # encoding para logs e arquivos de texto gerados pelo script
DATA_ENCODING  = 'utf-8'   # encoding dos arquivos CSV de dados de mercado
LOG_INSUF_DATA = "insufficient_data_log.txt"

def _log_insuf_data(asset, tf_file, subspace, error):
    with open(LOG_INSUF_DATA, 'a', encoding=FILE_ENCODING) as f:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{ts} | asset={asset} | tf={tf_file} | subespaço={subspace} | erro={error}\n")


class RunLogger:
    """Espelha sys.stdout no terminal e num arquivo de log simultaneamente.
    Suporta troca de arquivo via switch_file() sem perder o espelho no terminal.
    """
    def __init__(self, filepath):
        self._terminal = sys.stdout
        self._log = open(filepath, 'a', encoding=FILE_ENCODING, buffering=1)
        self.current_path = filepath

    def switch_file(self, filepath):
        """Fecha o log atual e abre um novo, mantendo o espelho no terminal."""
        self._log.flush()
        self._log.close()
        self._log = open(filepath, 'a', encoding=FILE_ENCODING, buffering=1)
        self.current_path = filepath

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()


def _log_header(title, level=1):
    """Imprime cabeçalho estruturado para organizar o log por asset/etapa."""
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    if level == 1:
        sep = '=' * 80
        print(f"\n{sep}\n  ASSET: {title}  |  {ts}\n{sep}")
    elif level == 2:
        sep = '-' * 72
        print(f"\n  {sep}\n  {title}  |  {ts}\n  {sep}")
    elif level == 3:
        print(f"\n    >> {title}  |  {ts}")
    elif level == 4:
        print(f"\n      [{title}]")
    else:
        print(f"        -- {title} --")


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

    if n < 4:
        return [], np.array([]), np.nan, np.nan, None, None
    ks = list(range(2, n - 1))
    k_arr = np.array(ks, dtype=int)
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


def shimazaki_shinomoto_bins(data, bin_range=None):
    """Seleciona o número ótimo de bins via critério de Shimazaki-Shinomoto (2007).

    Minimiza a função custo C(k) = (2k̄ - v) / Δ²  onde:
      Δ  = largura do bin
      k̄  = contagem média por bin
      v  = variância das contagens entre bins

    Parâmetros
    ----------
    data      : array 1-D com os dados a histogramar
    bin_range : sequência de inteiros (nº de bins a testar).
                Padrão: range(10, min(300, len(data)//10))

    Retorna
    -------
    optimal_bins : int   — número ótimo de bins
    bin_range    : array — valores testados
    costs        : array — valor da função custo para cada nº de bins
    """
    data = np.asarray(data, dtype=float)
    if bin_range is None:
        max_bins = min(300, max(10, len(data) // 10))
        bin_range = np.arange(10, max_bins + 1)
    else:
        bin_range = np.asarray(bin_range, dtype=int)

    x_min, x_max = data.min(), data.max()
    costs = np.empty(len(bin_range))

    for i, nbins in enumerate(bin_range):
        delta  = (x_max - x_min) / nbins
        edges  = np.linspace(x_min, x_max, nbins + 1)
        counts, _ = np.histogram(data, bins=edges)
        k_bar  = counts.mean()
        v      = np.sum((counts - k_bar) ** 2) / nbins   # variância enviesada
        costs[i] = (2 * k_bar - v) / (delta ** 2)

    optimal_bins = int(bin_range[np.argmin(costs)])
    return optimal_bins, bin_range, costs


def calc_FDT(pdf,C_store, W, build_plots):
    """Calcula e plota equilíbrio entre injeção e dissipação."""
    if C_store.ndim < 3 or len(C_store) < 2:
        return float('nan'), float('nan')
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
    print(
        "  Legenda FDT:\n"
        "    D (Dissipação) : taxa média com que o estado atual perde energia  = -<2<K_t, ΔK_t>>\n"
        "    I (Injeção)    : taxa média de flutuações novas adicionadas        = <||ΔK_t||²_F>\n"
        "    R (Resíduo)    : balanço energético D+I-ΔE — idealmente ≈ 0\n"
        "    R_FDT (Ratio)  : razão Flutuação-Dissipação = -D/I\n"
        "                     R_FDT ≈ 1 → equilíbrio  |  R_FDT < 1 → injeção domina (não-equilíbrio)\n"
        "    Ratio_err      : incerteza estatística de R_FDT (método delta)"
    )
    equil_flag = (
        "≈ equilíbrio" if 0.8 <= ratio <= 1.2
        else f"injeção {'domina' if ratio < 1 else 'abaixo'} — sistema {'longe' if ratio < 0.5 or ratio > 2 else 'moderadamente afastado'} do equilíbrio"
    )
    print(f"  D  (dissipação média) : {-mean_D:.6e}  — energia dissipada por janela")
    print(f"  I  (injeção média)    : {mean_I:.6e}  — flutuações injetadas por janela")
    print(f"  R  (resíduo balanço)  : {-np.mean(ratio_local):.6e}  — deve ser ≈ 0")
    print(f"  R_FDT                 : {ratio:.6f} ± {sigma_R:.2e}  → {equil_flag}")
    return ratio, sigma_R


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

def main_func(pdf, tf_file, tf_label='', n_value=5):

    df = pd.read_csv(tf_file, sep=',', encoding=DATA_ENCODING)
    print(f"\n  [DADOS] {tf_file} — {len(df)} registros | período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    times = pd.to_datetime(df['timestamp']).values
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

    _data_hash  = result_cache.file_hash(tf_file)
    _cache_mode = MULTI_STEP_TYPE if MULTI_STEP_ANALYSIS else 'single'

    loop_values = STEP_LIST if MULTI_STEP_ANALYSIS else [STEP]
    _n_loop     = len(loop_values)

    for _i_loop, loop_current in enumerate(loop_values):

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

        _log_header(f"[{_i_loop+1}/{_n_loop}] STEP={STEP_current} | EMBED_DIM={EMBED_current} | WINDOW={WINDOW_current}", level=4)

        # ── verificação de cache ──────────────────────────────────────────────
        _cached = result_cache.lookup(
            ASSET, tf_file, _data_hash, _cache_mode,
            STEP_current, EMBED_current, WINDOW_current,
        ) if USE_CACHE else None
        if _cached is not None:
            _sc, _ar = _cached
            print(f"  [cache hit] {ASSET} | {tf_label} | {_cache_mode} "
                  f"| step={STEP_current} embed={EMBED_current} window={WINDOW_current}")
            _all_rows.append({
                'asset': ASSET, 'tf_file': tf_file, 'mode': _cache_mode,
                'step': STEP_current, 'embed_dim': EMBED_current, 'window': WINDOW_current,
                'n_regime_changes':    _sc['n_regime_changes'],
                'n_waiting_intervals': _sc['n_waiting_intervals'],
                'mean_waiting_time':   _sc['mean_waiting_time'],
                'median_waiting_time': _sc['median_waiting_time'],
                'alpha':               _sc['alpha'],
                'alpha_err':           _sc['alpha_err'],
                'R_FDT':               _sc['R_FDT'],
                'R_FDT_err':           _sc['R_FDT_err'],
                'beta_total':          _sc['beta_total'],
                'beta_total_err':      _sc['beta_total_err'],
                'beta_struct':         _sc['beta_struct'],
                'beta_struct_err':     _sc['beta_struct_err'],
                'beta_bulk':           _sc['beta_bulk'],
                'beta_bulk_err':       _sc['beta_bulk_err'],
                'MP_L2_relative_pct':  _sc['MP_L2_relative_pct'],
                'entropy_mean':        _sc['entropy_mean'],
            })
            _lv = STEP_current if MULTI_STEP_TYPE == 'step' else EMBED_current
            if MULTI_STEP_TYPE == 'step':
                alpha_vs_step.append(    (_lv, _sc['alpha'],        _sc['alpha_err']))
                beta_vs_step.append(     (_lv, _sc['beta_struct'],   _sc['beta_struct_err']))
                entropy_vs_step.append(  (_lv, _sc['entropy_mean']))
                mpl2error_vs_step.append((_lv, _sc['MP_L2_relative_pct']))
            elif MULTI_STEP_TYPE == 'embed':
                alpha_vs_step.append(    (_lv, _sc['alpha'],        _sc['alpha_err']))
                beta_vs_step.append(     (_lv, _sc['beta_struct'],   _sc['beta_struct_err']))
                entropy_vs_step.append(  (_lv, _sc['entropy_mean']))
                mpl2error_vs_step.append((_lv, _sc['MP_L2_relative_pct']))
            if STEP_current == 20 and MULTI_STEP_ANALYSIS:
                _tau_sorted  = _ar.get('tau_sorted')
                _ccdf        = _ar.get('ccdf')
                _beta_lags   = _ar.get('beta_lags')
                _beta_values = _ar.get('beta_values')
                _beta_err    = _ar.get('beta_err')
                _beta_mask   = None
            os.makedirs(os.path.dirname(_csv_filename), exist_ok=True)
            _df_snap = pd.DataFrame(_all_rows)
            for _att in range(5):
                try:
                    _df_snap.to_csv(_csv_filename, index=False)
                    break
                except OSError:
                    if _att == 4:
                        raise
                    time.sleep(0.5)
            continue
        # ─────────────────────────────────────────────────────────────────────

        table_df ={}
        table_df['ASSET']     = ASSET
        table_df['TIMEFRAME'] = tf_label
        table_df['STEP']      = f"{STEP_current}"
        table_df['EMBED_DIM'] = f"{EMBED_current}"
        table_df['WINDOW']    = f"{WINDOW_current}"

        # ─────────────────────────────────
        # Loop principal em τ
        # ─────────────────────────────────
        _log_header('LOOP PCA — análise por janela deslizante em τ (PCA + limiar Marchenko-Pastur)', level=5)

        # =======================================================================
        # IMPLEMENTAÇÃO VETORIZADA COM CHUNKS
        # =======================================================================
        # MOTIVAÇÃO:
        #   O loop original calculava PCA de uma janela por vez em Python puro.
        #   Isso mantinha o CPU em ~6% porque as matrizes 70×70 são pequenas
        #   demais para o BLAS paralelizar individualmente.
        #
        # SOLUÇÃO — duas fases:
        #
        #   FASE 1 (pré-computação vetorizada em chunks):
        #     - Constrói a janela Xreg uma por vez e calcula sua covariância
        #       imediatamente, descartando Xreg logo em seguida.
        #       → pico de memória = 1 janela (≈153 KB) + buffer de covariâncias
        #     - A cada _CHUNK_SIZE janelas acumuladas, chama np.linalg.eigh() em
        #       batch sobre o chunk (ex: 1000×70×70). O BLAS processa o lote
        #       inteiro usando todos os cores → CPU a 80-100%.
        #     - Nota: _all_C (matriz de covariância) NÃO é armazenada após o
        #       chunk — seria inviável para datasets grandes (ex: 352K janelas
        #       × 70×70 × 8 bytes = 13 GB). Em vez disso, C é reconstruída na
        #       Fase 2 a partir dos autovetores/autovalores: C = V diag(λ) Vᵀ.
        #
        #   FASE 2 (acumulação sequencial):
        #     - Percorre os resultados pré-computados (vals, vecs, var, m, etc.)
        #       e alimenta os stores e a lógica de tracking, que têm dependências
        #       sequenciais (estado anterior: prev_vecs, prev_m) e não podem ser
        #       vetorizadas.
        # =======================================================================

        _pca_range   = range(WINDOW_current, len(ret_C) - WINDOW_current, STEP_current)
        _n_pca       = len(_pca_range)
        _step_report = max(1, _n_pca // 10)
        _t0_pca      = time.time()

        # -- FASE 1: PRE-COMPUTACAO VETORIZADA (chunked) -------------------------
        _CHUNK_SIZE = 1000  # janelas por chunk de eigendecomposição
        # _store_Xreg: só guarda as janelas brutas quando BUILD_DEVEL=True
        # (necessário para a análise omega-lambda, que projeta modos sobre Xreg)
        _store_Xreg    = BUILD_DEVEL and not MULTI_STEP_ANALYSIS
        _n_emb         = EMBED_current
        _n_samp        = None  # preenchido na primeira janela válida

        _C_buf       = []   # covariâncias do chunk em formação
        _var_buf     = []   # variâncias empíricas (usadas no limiar MP)
        _ctr_buf     = []   # centros τ correspondentes
        _Xreg_buf    = [] if _store_Xreg else None  # janelas brutas (só BUILD_DEVEL)

        _vals_parts  = []   # autovalores de todos os chunks já processados
        _vecs_parts  = []   # autovetores
        _var_parts   = []   # variâncias
        _lam_parts   = []   # limiar MP (λ+) por janela
        _m_parts     = []   # número de modos estruturais m(τ)
        _entr_parts  = []   # entropia espectral
        _valid_centers = []
        _Xreg_parts  = [] if _store_Xreg else None

        def _flush_chunk():
            """Processa o chunk atual: eigendecomposição em batch e acumula resultados.

            Chamada automaticamente a cada _CHUNK_SIZE janelas acumuladas
            e uma última vez ao final do loop para o chunk parcial.
            """
            if not _C_buf or _n_samp is None:
                return
            _q   = _n_emb / _n_samp   # razão aspecto para o limiar MP
            _Ca  = np.array(_C_buf,   dtype=np.float64)  # (chunk, embed, embed)
            _va  = np.array(_var_buf, dtype=np.float64)  # (chunk,)

            # eigendecomposição em batch — ponto onde o BLAS usa múltiplos cores
            _av, _ve = np.linalg.eigh(_Ca)
            _vals = _av[:, ::-1]; _vecs = _ve[:, :, ::-1]  # ordem decrescente
            del _av, _ve, _Ca   # libera imediatamente — não armazenamos _all_C

            # limiar de Marchenko-Pastur e número de modos estruturais
            _lam  = _va * (1 + np.sqrt(_q)) ** 2
            _m    = np.sum(_vals > _lam[:, np.newaxis], axis=1).astype(int)

            # entropia espectral normalizada S ∈ [0,1]
            _vp   = np.maximum(_vals, 1e-15)
            _p    = _vp / _vp.sum(axis=1, keepdims=True)
            _entr = -np.sum(_p * np.log(_p), axis=1) / np.log(_n_emb)
            del _vp, _p

            # acumula resultados do chunk nas listas de partes
            _vals_parts.append(_vals); _vecs_parts.append(_vecs)
            _var_parts.append(_va);   _lam_parts.append(_lam)
            _m_parts.append(_m);      _entr_parts.append(_entr)
            _valid_centers.extend(_ctr_buf)
            if _Xreg_parts is not None:
                _Xreg_parts.extend(_Xreg_buf)

            # limpa buffers do chunk para o próximo
            _C_buf.clear(); _var_buf.clear(); _ctr_buf.clear()
            if _Xreg_buf is not None:
                _Xreg_buf.clear()

        _n_chunks_done = 0
        print(f"        {_n_pca} janelas | chunks de {_CHUNK_SIZE} | "
              f"WINDOW={WINDOW_current} STEP={STEP_current} serie={len(ret_C)} pts")

        # loop de coleta: constrói covariância e descarta Xreg imediatamente
        for center in _pca_range:
            _Xr = build_causal_regime_matrix(center, EMBED_current, WINDOW_current,
                                             ret_O, ret_H, ret_L, ret_C)
            if _Xr is None:
                continue
            if _n_samp is None:
                _n_samp = _Xr.shape[0]  # n_samp é igual para todas as janelas

            # covariância amostral: C = (X - mean)^T (X - mean) / (n-1)
            _Xr_c = _Xr - _Xr.mean(axis=0, keepdims=True)
            _C_buf.append(_Xr_c.T @ _Xr_c / (_n_samp - 1))
            _var_buf.append(float(np.mean(np.var(_Xr, axis=0, ddof=1))))
            _ctr_buf.append(center)
            if _Xreg_buf is not None:
                _Xreg_buf.append(_Xr.copy())
            del _Xr, _Xr_c  # descarta janela — não armazena todas em memória

            if len(_C_buf) >= _CHUNK_SIZE:
                _flush_chunk()
                _n_chunks_done += 1
                _done = _n_chunks_done * _CHUNK_SIZE
                _pct  = min(100, 100 * _done / _n_pca)
                _bar  = chr(9608) * int(_pct / 5) + chr(9617) * (20 - int(_pct / 5))
                print(f"        [{_bar}] {_pct:5.1f}%  chunk {_n_chunks_done}"
                      f"  {time.time()-_t0_pca:5.1f}s", flush=True)

        _flush_chunk()  # processa o último chunk (pode ter menos que _CHUNK_SIZE janelas)

        _n_valid = len(_valid_centers)
        if _n_valid > 0:
            # concatena resultados de todos os chunks num único array contíguo
            _all_vals = np.concatenate(_vals_parts); del _vals_parts
            _all_vecs = np.concatenate(_vecs_parts); del _vecs_parts
            _all_var  = np.concatenate(_var_parts);  del _var_parts
            _all_lam  = np.concatenate(_lam_parts);  del _lam_parts
            _all_m    = np.concatenate(_m_parts);    del _m_parts
            _all_entr = np.concatenate(_entr_parts); del _entr_parts
            _all_Xreg = np.array(_Xreg_parts, dtype=np.float64) if _store_Xreg else None
            if _Xreg_parts is not None:
                del _Xreg_parts
            print(f"        PCA concluido: {_n_valid} janelas em "
                  f"{_n_chunks_done + 1} chunks  {time.time()-_t0_pca:.1f}s  --  acumulando resultados...")

            # -- FASE 2: ACUMULACAO SEQUENCIAL ------------------------------------
            # Esta fase percorre os resultados pré-computados e alimenta os stores.
            # Deve ser sequencial porque a análise de rotação (BUILD_DEVEL) depende
            # do estado da janela anterior (prev_vecs, prev_m, prev_vecs_all).
            for _ip, center in enumerate(_valid_centers):
                if _ip % _step_report == 0 or _ip == _n_valid - 1:
                    _pct    = 100 * (_ip + 1) / _n_valid
                    _filled = int(_pct / 5)
                    _bar    = chr(9608) * _filled + chr(9617) * (20 - _filled)
                    print(f"        [{_bar}] {_pct:5.1f}%  {_ip+1:4d}/{_n_valid}"
                          f"  {time.time()-_t0_pca:5.1f}s", flush=True)

                vals     = _all_vals[_ip]
                vecs     = _all_vecs[_ip]
                # C_raw reconstruída de V diag(λ) Vᵀ — equivalente a np.cov(Xreg)
                # mas sem precisar armazenar _all_C (que custaria ~13 GB para
                # datasets com 352K janelas)
                C_raw    = (vecs * vals) @ vecs.T
                var      = float(_all_var[_ip])
                lam_plus = float(_all_lam[_ip])
                m        = int(_all_m[_ip])
                entropy  = float(_all_entr[_ip])

                C_store.append(C_raw)
                var_data_store.append(var)
                lambda_plus_t.append(lam_plus)

                vecs_s    = vecs[:, :m]
                vals_s    = vals[:m]
                vecs_all  = vecs
                vals_all  = vals
                vecs_bulk = vecs[:, m:]
                vals_bulk = vals[m:]

                eigenvalues_series.append(vals)

                _Ks = (vecs_s * vals_s) @ vecs_s.T if m > 0 else np.zeros((_n_emb, _n_emb))
                _Kb = (vecs_bulk * vals_bulk) @ vecs_bulk.T
                C_store_struct.append(_Ks)
                C_store_bulk.append(_Kb)

                if not MULTI_STEP_ANALYSIS:
                    bulk_evals_store.append(vals_bulk)
                    bulk_vecs_store.append(vecs_bulk)
                    struct_evals_store.append(vals_s)
                    struct_vecs_store.append(vecs_s)

                entropy_series.append(entropy)
                m_t.append(m)

                # -- Rotacoes (apenas modo completo) ------------------------------
                if BUILD_DEVEL and not MULTI_STEP_ANALYSIS:

                    if prev_vecs_all is not None:
                        try:
                            n_modes = min(prev_vecs_all.shape[1], vecs_all.shape[1])
                            dots    = np.abs(np.einsum('ij,ij->j',
                                        prev_vecs_all[:, :n_modes], vecs_all[:, :n_modes]))
                            theta_k = np.arccos(np.clip(dots, 0.0, 1.0))
                            theta_modes_all.append(theta_k)
                            if prev_m is not None and m > 0 and prev_m > 0:
                                m_star     = min(prev_m, m)
                                ang_struct = subspace_angles(prev_vecs[:, :m_star], vecs_s[:, :m_star])
                                theta_struct_t.append(np.sqrt(np.sum(ang_struct ** 2)))
                            else:
                                theta_struct_t.append(np.nan)
                            if m < n_modes - 2:
                                theta_bulk = theta_k[m + 1:-1]
                                gap_bulk   = [min(abs(vals_all[k] - vals_all[k-1]),
                                                  abs(vals_all[k] - vals_all[k+1]))
                                              for k in range(m + 1, n_modes - 1)]
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

                    omega_lambda_relation = []
                    if m > 0:
                        _Xreg_i = _all_Xreg[_ip]
                        for i in range(min(m, 5)):
                            phi_close = vecs_s[:, i][3::4]
                            mode_ts   = _Xreg_i[:, 3::4] @ phi_close
                            w         = estimate_frequency_fourier(mode_ts)
                            omega_lambda_relation.append((w, vals_s[i]))
                    if len(omega_lambda_relation) > 1:
                        ws = np.array([x[0] for x in omega_lambda_relation])
                        ls = np.array([x[1] for x in omega_lambda_relation])
                        ws = ws[ws > 0]; ls = ls[ls > 0]
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

                    row = [np.nan] * max_tracks
                    for i in range(min(len(vals_s), max_tracks)):
                        row[i] = vals_s[i]
                    tracked_modes.append(row)

                    if m > 0 and len(evecs_store) < 5:
                        evecs_store[times[center]] = vecs_s.copy()

                    if m > 0:
                        diff_matrix = np.abs(vals[:m, np.newaxis] - vals[np.newaxis, :])
                        diff_matrix[np.arange(m), np.arange(m)] = np.inf
                        deltas = diff_matrix.min(axis=1)
                        gap_min_t.append(np.min(deltas))
                        G_tau = np.mean(1.0 / (deltas ** 2))
                        gap_t.append(G_tau)
                    else:
                        gap_min_t.append(np.nan)
                        gap_t.append(np.nan)

                    var_total  = np.sum(vals)
                    var_struct = np.sum(vals[:m]) if m > 0 else 0.0
                    var_noise  = np.sum(vals[m:])
                    var_total_t.append(var_total)
                    var_struct_t.append(var_struct / var_total)
                    var_noise_t.append(var_noise / var_total)

                    prev_vecs_all = vecs_all.copy()
                    prev_vals_all = vals_all.copy()
                    prev_vecs     = vecs_s.copy()
                    prev_m        = m

                    tau_axis.append(times[center])

            del _all_vals, _all_vecs, _all_var, _all_lam, _all_m, _all_entr
            if _all_Xreg is not None:
                del _all_Xreg
            gc.collect()

        # ─────────────────────────────────────────────────
        # Pós-loop: reescala bulk para MP agregado
        # ─────────────────────────────────────────────────
        _log_header('PÓS-LOOP — reescala do bulk para distribuição Marchenko-Pastur agregada', level=5)
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
        _log_header('TESTES 1-2 — relação θ_bulk × gap espectral e variância do bulk', level=5)
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
        _log_header('ÍNDICES DE REGIME — detecção de crossings estruturais (mudanças em m(τ))', level=5)
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
        _log_header('TEMPOS DE ESPERA — distribuição entre transições | estimador de Hill para expoente α', level=5)
        print(
            "  Legenda:\n"
            "    Número de intervalos : quantidade de gaps temporais entre transições consecutivas de regime\n"
            "                           (mudanças em m(τ), o número de modos estruturais acima do limiar MP)\n"
            "    Tempo médio          : duração média de um regime em número de janelas temporais\n"
            "    Mediana              : valor central da distribuição — se muito menor que a média,\n"
            "                           confirma cauda pesada à direita (alguns regimes duram muito mais)\n"
            "    α (alpha) estimado   : expoente da cauda da lei de potência P(T > t) ~ t^(-α)\n"
            "                           α < 2 → variância infinita (dinâmica superdifusiva / memória longa)\n"
            "                           α > 2 → variância finita (comportamento próximo ao difusivo)\n"
            "    Platô k=a–b          : intervalo de k (nº de estatísticas de ordem superiores) onde\n"
            "                           o estimador de Hill estabiliza → região confiável para α"
        )
        indices    = np.array(sorted(indices_m_change))
        waiting_times = np.diff(indices)
        n_intervals = len(waiting_times)
        mean_wt     = np.mean(waiting_times)
        median_wt   = np.median(waiting_times)
        skew_flag   = "(assimetria pronunciada — cauda pesada)" if mean_wt > 2 * median_wt else "(distribuição aproximadamente simétrica)"
        print(f"  Número de intervalos : {n_intervals}  → {n_intervals} gaps entre {n_intervals + 1} transições detectadas")
        print(f"  Tempo médio          : {mean_wt:.4f}  janelas por regime")
        print(f"  Mediana              : {median_wt:.1f}  janelas  {skew_flag}")

        table_df[r'$\bar{\tau^*}$'] = f"{mean_wt:.3f}"

        tau = np.array(waiting_times)
        tau_sorted = np.sort(tau)
        ccdf = 1 - np.arange(1, len(tau_sorted) + 1) / len(tau_sorted)

        if BUILD_PLOTS and BUILD_DEVEL:
            plt.figure()
            plt.loglog(tau_sorted, ccdf, marker='.', linestyle='none')
            plt.xlabel("tau")
            plt.ylabel("P(T > tau)")

        ks, alphas, alpha_hat, alpha_err, k_lo, k_hi = hill_estimator(tau)
        alpha_interp = (
            "variância infinita — dinâmica superdifusiva / memória longa" if alpha_hat < 2
            else "variância finita — comportamento próximo ao difusivo"
        )
        print(f"  α estimado           : {alpha_hat:.3f} ± {alpha_err:.3f}  ({alpha_interp})")
        print(f"  Platô do estimador   : k={k_lo}–{k_hi}  → região estável onde α é confiável")
        if BUILD_PLOTS:
            hill_plot(pdf,ks, alphas, k_lo, k_hi)

        table_df[r'$\hat{\alpha}$'] = f"{alpha_hat:.3f} ± {alpha_err:.3f}"

        if MULTI_STEP_TYPE == 'step':
            alpha_vs_step.append((STEP_current, alpha_hat, alpha_err))
        elif MULTI_STEP_TYPE == 'embed':
            alpha_vs_step.append((EMBED_current, alpha_hat, alpha_err))

        if BUILD_PLOTS:
            plt.figure()
            wt_pos = waiting_times[waiting_times > 0]
            if len(wt_pos) > 1:
                plt.hist(wt_pos, bins=30, density=True)
                plt.yscale("log")
                plt.xscale("log")
            else:
                plt.text(0.5, 0.5, 'Dados insuficientes\npara escala log',
                         ha='center', va='center', transform=plt.gca().transAxes)
            plt.xlabel("Tempo entre mudanças")
            plt.ylabel("Densidade (log log)")
            pdf.savefig()
            plt.close()
        # ─────────────────────────────────────────────────
        # FDT e balanço de energia (apenas modo completo)
        # ─────────────────────────────────────────────────
        _log_header('FDT — razão Flutuação-Dissipação (R_FDT) via resposta linear', level=5)

        # ── subsample consistente dos três stores ────────────────────────────────
        # Calcula o número máximo de frames ANTES da alocação, com base no tamanho
        # real da matriz e em um orçamento de memória fixo por store (STORE_MEM_BUDGET_MB).
        # Isso evita tentar alocar e falhar — não depende de capturar exceções numpy.
        _n_store = len(C_store)
        if _n_store > 0:
            _emb_sz   = C_store[0].shape[0]                           # lado da matriz (ex: 70, 100)
            _bytes_pm = _emb_sz * _emb_sz * 8                         # bytes por matriz (float64)
            _budget   = STORE_MEM_BUDGET_MB * 1024 * 1024             # orçamento em bytes
            _max_safe = max(200, _budget // _bytes_pm)                 # frames que cabem no orçamento
            _max_f    = min(MAX_STORE_FRAMES, _max_safe)               # respeita ambos os limites
        else:
            _max_f = MAX_STORE_FRAMES

        _ss = max(1, _n_store // _max_f) if _n_store > _max_f else 1
        if _ss > 1:
            print(f"    [Stores] subsample: {_n_store} → {_n_store // _ss} matrizes "
                  f"(passo={_ss}, embed={_emb_sz}×{_emb_sz}, orçamento={STORE_MEM_BUDGET_MB} MB)")
        gc.collect()
        C_store        = np.array(C_store[::_ss])
        C_store_struct = np.array(C_store_struct[::_ss])
        C_store_bulk   = np.array(C_store_bulk[::_ss])
        gc.collect()

        W = 200
        ratio_fdt,ratio_ftd_err=calc_FDT(pdf,C_store, W, BUILD_PLOTS)
        table_df[r"$R_{FDT}$"]=f"{ratio_fdt:.3f} ± {ratio_ftd_err:.3f}"

        # ─────────────────────────────────────────────────
        # Estimativa de beta por subespaço
        # ─────────────────────────────────────────────────
        _log_header('BETA (MSD) — expoente de difusão β via deslocamento quadrático médio no subespaço', level=5)
        print(
            "  Legenda β:\n"
            "    β (expoente MSD)     : ajuste log-log de MSD(s) ~ s^β sobre os últimos 70% dos lags válidos\n"
            "                           β ≈ 1 → difusão normal (Browniano)  |  β > 1 → superdifusão (persistente)\n"
            "                           β < 1 → subdifusão (anti-persistente / mean-reverting)\n"
            "    Intervalo de s usado : lags [s_min, s_max] utilizados na cauda do fit log-log\n"
            "    Subespaços avaliados :\n"
            "      TOTAL  — matriz de correlação completa C(τ)\n"
            "      STRUCT — subespaço estrutural (autovalores acima do limiar MP, modos informativos)\n"
            "      BULK   — subespaço de ruído (autovalores abaixo do limiar MP, modos aleatórios)"
        )
        _C_struct_arr = C_store_struct   # já convertido e subsampled acima
        _C_bulk_arr   = C_store_bulk     # já convertido e subsampled acima
        _m_arr        = np.array(m_t)

        print("  ► TOTAL (matriz de correlação completa):")
        try:
            beta_total,  beta_err_total,  msd_total,  lags_total  = mbsw.estimate_beta(pdf, C_store,        BUILD_PLOTS)
        except ValueError as e:
            _log_insuf_data(ASSET, tf_file, 'total', e)
            beta_total,  beta_err_total,  msd_total,  lags_total  = float('nan'), float('nan'), np.array([]), np.array([])

        print("  ► STRUCT (subespaço estrutural — modos acima do limiar MP):")
        try:
            beta_struct, beta_err_struct, msd_struct, lags_struct  = mbsw.estimate_beta(pdf, _C_struct_arr, BUILD_PLOTS)
        except ValueError as e:
            _log_insuf_data(ASSET, tf_file, 'struct', e)
            beta_struct, beta_err_struct, msd_struct, lags_struct  = float('nan'), float('nan'), np.array([]), np.array([])

        print("  ► BULK (subespaço de ruído — modos abaixo do limiar MP):")
        try:
            beta_bulk,   beta_err_bulk,   msd_bulk,   lags_bulk    = mbsw.estimate_beta(pdf, _C_bulk_arr,   BUILD_PLOTS)
        except ValueError as e:
            _log_insuf_data(ASSET, tf_file, 'bulk', e)
            beta_bulk,   beta_err_bulk,   msd_bulk,   lags_bulk    = float('nan'), float('nan'), np.array([]), np.array([])

        def _fmt_beta(b, e):
            if np.isnan(b):
                return "N/A"
            return f"{b:.3f} ± {e:.3f}"

        table_df[r'$\hat{\beta_T}$'] = _fmt_beta(beta_total,  beta_err_total)
        table_df[r'$\hat{\beta_S}$'] = _fmt_beta(beta_struct, beta_err_struct)
        table_df[r'$\hat{\beta_B}$'] = _fmt_beta(beta_bulk,   beta_err_bulk)

        if BUILD_PLOTS and BUILD_DEVEL and len(lags_total) > 0 and len(lags_struct) > 0:
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
        _log_header('TESTE 6 — aderência do bulk à lei de Marchenko-Pastur (MP)', level=5)
        L2_relative = float('nan')
        if len(all_bulk_rescaled) == 0 or len(all_q) == 0:
            print("  [aviso] Bulk reescalado vazio — teste MP ignorado (dados insuficientes).")
        else:
            print(
                "  Legenda MP:\n"
                "    λ- / λ+             : limites inferior e superior da distribuição MP para ruído puro\n"
                "                          empírico = extremos observados nos autovalores do bulk reescalado\n"
                "                          teórico  = (1 ± √q)²  onde  q = embed_dim / (window - embed_dim)\n"
                "                          Se empírico ≈ teórico → bulk se comporta como ruído aleatório ✓\n"
                "    Erro L2 / L1        : distância entre distribuição empírica e lei de MP teórica\n"
                "                          L2 = ∫(p_emp - p_MP)² ds  |  L1 = ∫|p_emp - p_MP| ds\n"
                "    Erro L2 relativo(%) : L2 normalizado — quanto % da forma da curva MP é mal ajustada\n"
                "                          < 5% → excelente  |  5–15% → aceitável  |  > 15% → desvio relevante\n"
                "    RuntimeWarning sqrt : ocorre em pontos marginalmente fora do suporte MP (artefato numérico\n"
                "                          tratado pelo np.where — não afeta o resultado)"
            )
            q_mean = np.mean(all_q)
            xs     = np.linspace(np.min(all_bulk_rescaled), np.max(all_bulk_rescaled), 400)

            lambda_minus_theory = (1 - np.sqrt(q_mean)) ** 2
            lambda_plus_theory  = (1 + np.sqrt(q_mean)) ** 2
            lambda_minus_emp    = np.min(all_bulk_rescaled)
            lambda_plus_emp     = np.max(all_bulk_rescaled)

            print(f"λ-  empírico={lambda_minus_emp:.6f}  teórico(MP)={lambda_minus_theory:.6f}")
            print(f"λ+  empírico={lambda_plus_emp:.6f}  teórico(MP)={lambda_plus_theory:.6f}")
            print(f"Diferença percentual λ+: {100 * np.abs(lambda_plus_emp - lambda_plus_theory) / lambda_plus_theory:.2f} %")

            # ── Otimização do nº de bins (Shimazaki-Shinomoto) ─────────────────────
            if BINS_OPTIMIZATION:
                print("        Otimizando número de bins (Shimazaki-Shinomoto)...")
                _opt_bins, _bin_range, _costs = shimazaki_shinomoto_bins(all_bulk_rescaled)
                n_hist_bins = _opt_bins
                print(f"        Bins ótimo: {_opt_bins}  (testados {len(_bin_range[_bin_range >= _bin_range[0]])}: {_bin_range[0]}–{_bin_range[-1]})")

                # CSV e PDF únicos por (asset, tf, step, embed)
                _bins_tag = f"{ASSET}_{tf_label}_s{STEP_current}_e{EMBED_current}"
                _bins_csv = os.path.join(_bin_opt_dir, f"{_bins_tag}_bins_opt.csv")
                pd.DataFrame({'n_bins': _bin_range, 'custo': _costs}).to_csv(_bins_csv, index=False)

                # PDF dedicado à otimização
                _bins_pdf_path = os.path.join(_bin_opt_dir, f"{_bins_tag}_bins_opt.pdf")
                with _PdfPages(_bins_pdf_path) as _bpdf:

                    # Pág 1: função custo vs n_bins
                    _fig_c, _ax_c = plt.subplots(figsize=(7, 4))
                    _ax_c.plot(_bin_range, _costs, lw=2, label='Função custo')
                    _ax_c.axvline(_opt_bins, color='red', ls='--',
                                  label=f'Ótimo = {_opt_bins} bins')
                    _ax_c.set_xlabel('Número de bins')
                    _ax_c.set_ylabel('C(k)  (Shimazaki-Shinomoto)')
                    _ax_c.set_title(f'{ASSET} {tf_label} — Função custo × número de bins\n'
                                    f'Ótimo = {_opt_bins} bins  |  C_min = {_costs.min():.4e}')
                    _ax_c.legend()
                    _ax_c.grid(True, ls=':')
                    _fig_c.tight_layout()
                    _bpdf.savefig(_fig_c)
                    plt.close(_fig_c)

                    # Pág 2: grade de histogramas para bins representativos
                    _sample_bins = np.unique(np.round(
                        np.linspace(_bin_range[0], _bin_range[-1], 9)).astype(int))
                    _sample_bins = np.append(_sample_bins, _opt_bins)
                    _sample_bins = np.unique(_sample_bins)
                    _ncols = 3
                    _nrows = int(np.ceil(len(_sample_bins) / _ncols))
                    _fig_h, _axes_h = plt.subplots(_nrows, _ncols,
                                                   figsize=(4 * _ncols, 3 * _nrows))
                    _axes_h = np.array(_axes_h).flatten()
                    for _ai, _nb in enumerate(_sample_bins):
                        _ax = _axes_h[_ai]
                        _hv, _be = np.histogram(all_bulk_rescaled, bins=int(_nb), density=True)
                        _bc = 0.5 * (_be[:-1] + _be[1:])
                        _dxh = _bc[1] - _bc[0]
                        _ax.bar(_bc, _hv, width=_dxh, color='#4878CF', alpha=0.65)
                        _xs_h = np.linspace(all_bulk_rescaled.min(),
                                            all_bulk_rescaled.max(), 300)
                        _ax.plot(_xs_h, mp_pdf(_xs_h, 1.0, q_mean),
                                 color='#E84646', lw=1.4, label='MP')

                        # erro L2 relativo para este numero de bins
                        _mp_bc  = mp_pdf(_bc, 1.0, q_mean)
                        _l2_h   = np.sum((_hv - _mp_bc) ** 2) * _dxh
                        _nf_h   = np.sum(_mp_bc ** 2) * _dxh
                        _l2r_h  = 100 * _l2_h / _nf_h if _nf_h > 0 else float('nan')

                        _is_opt = (_nb == _opt_bins)
                        _ax.set_title(f'k={_nb}' + (' <- otimo' if _is_opt else ''),
                                      fontsize=9, fontweight='bold' if _is_opt else 'normal',
                                      color='red' if _is_opt else 'black')
                        _ax.set_xlabel(r'$lambda/\hat\sigma^2$', fontsize=7)
                        _ax.tick_params(labelsize=7)
                        _ax.text(0.97, 0.95, f'L2rel={_l2r_h:.1f}%',
                                 transform=_ax.transAxes, fontsize=7,
                                 ha='right', va='top',
                                 color='red' if _is_opt else '#444444',
                                 bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='white', alpha=0.75, edgecolor='none'))
                    for _ai in range(len(_sample_bins), len(_axes_h)):
                        _axes_h[_ai].axis('off')
                    _fig_h.suptitle(f'{ASSET} {tf_label} — Histogramas do bulk vs lei MP',
                                    fontsize=11)
                    _fig_h.tight_layout()
                    _bpdf.savefig(_fig_h)
                    plt.close(_fig_h)

                print(f"        PDF de otimização: {_bins_pdf_path}")
                print(f"        CSV de otimização: {_bins_csv}")
            else:
                n_hist_bins = 80

            hist_vals, bin_edges = np.histogram(all_bulk_rescaled, bins=n_hist_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            mp_vals     = mp_pdf(bin_centers, 1.0, q_mean)
            dx          = bin_centers[1] - bin_centers[0]

            L2_error    = np.sum((hist_vals - mp_vals) ** 2) * dx
            L1_error    = np.sum(np.abs(hist_vals - mp_vals)) * dx
            norm_factor = np.sum(mp_vals ** 2) * dx
            L2_relative = L2_error / norm_factor

            l2_qual = "excelente" if 100 * L2_relative < 5 else ("aceitável" if 100 * L2_relative < 15 else "desvio relevante — bulk não é ruído puro")
            print(f"  Bins utilizados  : {n_hist_bins}{'  (Shimazaki-Shinomoto ótimo)' if BINS_OPTIMIZATION else '  (fixo)'}")
            print(f"  Erro L2          : {L2_error:.6f}  — distância quadrática entre empírico e MP")
            print(f"  Erro L1          : {L1_error:.6f}  — distância absoluta entre empírico e MP")
            print(f"  Erro L2 relativo : {100 * L2_relative:.2f} %  → {l2_qual}")

            table_df[r'$L^2$'] = f"{100 * L2_relative:.3f} %"

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
                    f'Rel. $L^2$ error: {100 * L2_relative:.1f}%',
                    transform=ax_mp.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#CCCCCC', alpha=0.85))

                fig1_prl.tight_layout()
                if not MULTI_STEP_ANALYSIS:
                    out_mp = os.path.join(_fig_dir, os.path.basename(os.path.splitext(tf_file)[0]) + '_fig1_mp.pdf')
                    fig1_prl.savefig(out_mp, bbox_inches='tight')
                    fig1_prl.savefig(out_mp.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
                    print(f"Figura 1 salva: {out_mp}")
                pdf.savefig()
                plt.close()

        # tabela de resultados sempre adicionada ao PDF (necessário para epdf.pdf_to_csv)
        _tdf = pd.DataFrame([table_df]).T
        _tdf.columns = ['Valor']
        add_table(pdf, _tdf, 'Resultados')

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
        _log_header('TESTE 11 — entropia espectral normalizada', level=5)
        S_norm_med = np.mean(entropy_series)
        s_interp = (
            "distribuição de autovalores quase uniforme — matriz dominada por ruído"   if S_norm_med > 0.90 else
            "estrutura moderada — alguns modos dominantes sobre fundo de ruído"         if S_norm_med > 0.70 else
            "estrutura pronunciada — poucos modos concentram a maioria da variância"
        )
        print(
            "  Legenda entropia:\n"
            "    Entropia espectral normalizada S ∈ [0, 1]\n"
            "      S → 1 : autovalores uniformemente distribuídos (máximo desalinhamento — ruído puro)\n"
            "      S → 0 : um único autovalor domina (máxima concentração — estrutura pura)\n"
            "    Valor médio ao longo de todas as janelas temporais τ"
        )
        print(f"  Entropia média : {S_norm_med:.6f}  → {s_interp}")
        if MULTI_STEP_TYPE == 'step':
            entropy_vs_step.append((STEP_current, S_norm_med))
        elif MULTI_STEP_TYPE == 'embed':
            entropy_vs_step.append((EMBED_current, S_norm_med))

        # ── coleta linha para a tabela CSV ────────────────────────────
        _all_rows.append({
            'asset'              : ASSET,
            'tf_file'            : tf_file,
            'mode'               : MULTI_STEP_TYPE if MULTI_STEP_ANALYSIS else 'single',
            'step'               : STEP_current,
            'embed_dim'          : EMBED_current,
            'window'             : WINDOW_current,
            'n_regime_changes'   : len(indices_m_change),
            'n_waiting_intervals': n_intervals,
            'mean_waiting_time'  : round(mean_wt,   4),
            'median_waiting_time': round(median_wt, 4),
            'alpha'              : round(alpha_hat,    4),
            'alpha_err'          : round(alpha_err,    4),
            'R_FDT'              : round(ratio_fdt,    6),
            'R_FDT_err'          : round(ratio_ftd_err, 6),
            'beta_total'         : round(beta_total,   4) if not np.isnan(beta_total)   else np.nan,
            'beta_total_err'     : round(beta_err_total,  4) if not np.isnan(beta_err_total)  else np.nan,
            'beta_struct'        : round(beta_struct,  4) if not np.isnan(beta_struct)  else np.nan,
            'beta_struct_err'    : round(beta_err_struct, 4) if not np.isnan(beta_err_struct) else np.nan,
            'beta_bulk'          : round(beta_bulk,    4) if not np.isnan(beta_bulk)    else np.nan,
            'beta_bulk_err'      : round(beta_err_bulk,   4) if not np.isnan(beta_err_bulk)   else np.nan,
            'MP_L2_relative_pct' : round(100 * L2_relative, 4),
            'entropy_mean'       : round(S_norm_med, 6),
        })
        os.makedirs(os.path.dirname(_csv_filename), exist_ok=True)
        _df_snapshot = pd.DataFrame(_all_rows)
        for _attempt in range(5):
            try:
                _df_snapshot.to_csv(_csv_filename, index=False)
                break
            except OSError:
                if _attempt == 4:
                    raise
                time.sleep(0.5)

        result_cache.save(
            ASSET, tf_file, _data_hash, _cache_mode,
            STEP_current, EMBED_current, WINDOW_current,
            _all_rows[-1],
            {
                'tau_sorted':  tau_sorted  if len(tau_sorted)  > 0 else None,
                'ccdf':        ccdf        if len(ccdf)        > 0 else None,
                'beta_lags':   lags_struct if isinstance(lags_struct, np.ndarray) and len(lags_struct) > 0 else None,
                'beta_values': msd_struct  if isinstance(msd_struct,  np.ndarray) and len(msd_struct)  > 0 else None,
                'beta_err':    np.atleast_1d(beta_err_struct) if not (isinstance(beta_err_struct, float) and np.isnan(beta_err_struct)) else None,
            },
        )

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
            out_name = os.path.join(_fig_dir, os.path.basename(os.path.splitext(tf_file)[0]) + "_4D_" + MODE + "_" + TYPE + ".png")
            plt.savefig(out_name, dpi=300)

        if BUILD_PLOTS:
            plt.show()

        # libera arrays grandes ao fim de cada iteração do loop de steps/embeds
        del C_store, C_store_struct, C_store_bulk
        del eigenvalues_series, bulk_evals_store, bulk_vecs_store
        del struct_evals_store, struct_vecs_store, var_data_store
        del tracked_modes, all_bulk_rescaled, all_vals_rescaled
        gc.collect()

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
def plot_fig_step(step, muti_data, tf_file, tf_labels=None):
    tf_labels = tf_labels if tf_labels is not None else TIMEFRAME_LABELS
    # ── desempacota multi_data em dicionários por tf_label ───────────
    keys = ('alpha_vs_step', 'beta_vs_step', 'entropy_vs_step',
            'mpl2error_vs_step', 'tau_sorted', 'ccdf',
            'beta_lags', 'beta_values', 'beta_err', 'beta_mask')
    (alpha_vs_step, beta_vs_step, entropy_vs_step, mpl2error_vs_step,
     _tau_sorted, _ccdf, _beta_lags, _beta_values, _beta_err, _beta_mask) = (
        {tf: multi_data[tf][k] for tf in tf_labels} for k in keys
    )

    steps_list = {}
    betas_list = {}
    alphas_list = {}
    beta_errs_list  = {}
    alpha_errs_list = {}
    for tf_label in tf_labels:
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
        for tf_label in tf_labels:
            if _beta_lags[tf_label] is None or _beta_values[tf_label] is None:
                raise TypeError("beta_lags/beta_values não disponíveis")
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
    except (NameError, TypeError):
        ax_a.text(0.5, 0.5, 'Dados não disponíveis\n(cache sem arrays)',
                  ha='center', va='center', transform=ax_a.transAxes, fontsize=7)
    ax_a.set_title('(a)', loc='left', fontweight='bold')
    for sp in ['top', 'right']:
        ax_a.spines[sp].set_visible(False)

    # painel (b): CCDF τ*
    ax_b = fig2_prl.add_subplot(gs2[1])
    try:
        ts0 = _tau_sorted[tf_labels[0]]
        if ts0 is None or len(ts0) == 0:
            raise TypeError("tau_sorted não disponível")
        bins_log = np.logspace(np.log10(max(ts0.min(), 1)), np.log10(ts0.max()), 25)
        ax_b.hist(ts0, bins=bins_log, density=True,
                  color=tf_colors[tf_labels[0]], alpha=0.4,
                  label=tf_labels[0])
        counts_tmp, edges_tmp = np.histogram(ts0, bins=bins_log, density=True)
        centers_tmp = np.sqrt(edges_tmp[:-1] * edges_tmp[1:])
        i_peak  = np.argmax(counts_tmp)
        A_fit   = counts_tmp[i_peak] * centers_tmp[i_peak] ** 2
        t_fit   = centers_tmp[counts_tmp > 0]
        ax_b.loglog(t_fit, A_fit / t_fit ** 2, 'k--', lw=0.9, label=r'$\alpha=2$')
        ax_b.set_xlabel(r'$\tau^*$')
        ax_b.set_ylabel(r'$p(\tau^*)$')
        ax_b.legend(frameon=False, fontsize=7)
    except (NameError, TypeError):
        ax_b.text(0.5, 0.5, 'Dados não disponíveis\n(cache sem arrays)',
                  ha='center', va='center', transform=ax_b.transAxes, fontsize=7)
    ax_b.set_title('(b)', loc='left', fontweight='bold')
    for sp in ['top', 'right']:
        ax_b.spines[sp].set_visible(False)

    # painel (c): β vs STEP
    ax_c = fig2_prl.add_subplot(gs2[2])
    for tf_label in tf_labels:
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
    for tf_label in tf_labels:
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

    _f2_base = os.path.join(_fig_dir, tf_file.split('_')[0] + '_fig2_exponents')
    fig2_prl.savefig(_f2_base + '.pdf', bbox_inches='tight')
    fig2_prl.savefig(_f2_base + '.png', dpi=300, bbox_inches='tight')
    print("Figura 2 salva.")
    #pdf.savefig()
    plt.close()

def plot_fig_embed(embed, multi_data, tf_file, tf_labels=None):
    tf_labels = tf_labels if tf_labels is not None else TIMEFRAME_LABELS
# ── FIGURA 3 PRL: β vs EMBED_DIM ──────────────────────────────────────────
#if MULTI_STEP_TYPE == 'embed':
    steps_list = {}
    betas_list = {}
    beta_errs_list = {}
    for tf_label in tf_labels:
        for n_value in N_STEPS:
            index = tf_label + str(n_value)
            steps_list[index], betas_list[index], beta_errs_list[index] = \
                zip(*multi_data[index]['beta_vs_step'])

    fig3_prl, ax3 = plt.subplots(figsize=(3.375, 2.6))
    for tf_label in tf_labels:
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
    _f3_base = os.path.join(_fig_dir, tf_file.split('_')[0] + '_fig3_Lopt')
    fig3_prl.savefig(_f3_base + '.pdf', bbox_inches='tight')
    fig3_prl.savefig(_f3_base + '.png', dpi=300, bbox_inches='tight')
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
#ASSET_LIST = ["PETR4","BOVA11", "VALE3"]
ASSET_LIST = [
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

#BEST_STEP={'PETR4': 20}#,'ITUB4':20,'VALE3':40,'WIN$D':45,'ABEV3':40,'WDO$N':45}
BEST_STEP={asset: 20 for asset in ASSET_LIST}

#SKIP_STEP=True #para pular a etapa de step e usar os valores otimizados
SKIP_STEP=False #para pular a etapa de step e usar os valores otimizados


#BEST_EMBED={'PETR4': 80}#,'ITUB4':100,'VALE3':120,'WIN$D':70,'ABEV3':60,'WDO$N':80}
BEST_EMBED={asset: 80 for asset in ASSET_LIST}

#SKIP_EMBED=True #para pular a etapa de embed e usar os valores otimizados
SKIP_EMBED=False #para pular a etapa de embed e usar os valores otimizados

#SKIP_SINGLE=False # para pular a etapa single
SKIP_SINGLE=True # para pular a etapa single

USE_CACHE = True  # pula o cálculo quando a combinação (asset × tf × mode × step × embed × window) já existe no banco

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
BINS_OPTIMIZATION = True   # otimiza nº de bins do histograma via Shimazaki-Shinomoto
MAX_STORE_FRAMES     = 10_000 # limite absoluto de frames nos stores (independente do tamanho)
STORE_MEM_BUDGET_MB  = 256    # orçamento de RAM por store em MB — determina frames seguros
                              # para embed 70×70: ~8K frames | embed 100×100: ~3.2K frames

# ================================
# ANALISE MULTI-STEP
# ================================
MULTI_STEP_ANALYSIS = True
MULTI_STEP_TYPE = 'step'        # <-- valor padrão, garante que sempre existe

STEP_LIST_STEP  = [20,30,40,50]
STEP_LIST_EMBED = [70,80,90,100]

#STEP_LIST_STEP  = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
#STEP_LIST_EMBED = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]

# ================================================================
# PATHS
# ================================================================
DATA_PATH = r'E:\ronne\Documentos\PROJETOS\DINAMICA DE PREÇOS\alpaca\SPY500_DATA'

# ================================================================
# TIMEFRAMES (usados apenas com MULTI_STEP_ANALYSIS)
# ================================================================
#TIMEFRAME_LABELS = ['D1', 'H1', 'H4', 'M1', 'M5', 'M15', 'M30']  # rótulos — edite aqui para mudar os timeframes
#TIMEFRAME_LABELS = ['1min', '1hour', '1day']  # rótulos — edite aqui para mudar os timeframes
TIMEFRAME_LABELS = ['1day']  # rótulos — edite aqui para mudar os timeframes

TIMEFRAME_SUFFIX = [f'_{lbl}.csv' for lbl in TIMEFRAME_LABELS]   # gerado automaticamente

_COLOR_PALETTE = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3',
                  '#FF7F00', '#A65628', '#F781BF', '#999999']
tf_colors = {lbl: _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
             for i, lbl in enumerate(TIMEFRAME_LABELS)}
N_STEPS = [4, 5, 6]
resultados = []

# ================================================================
# DIRETÓRIO DA RUN E SUBPASTAS
# ================================================================
_run_ts  = time.strftime('%Y%m%d_%H%M%S')
_run_dir = f"run_{_run_ts}"                          # pasta raiz da execução
_log_dir = os.path.join(_run_dir, "logs")            # logs por ativo/timeframe
_fig_dir     = os.path.join(_run_dir, "figures")     # PDFs e PNGs
_bin_opt_dir = os.path.join(_run_dir, "bin_opt")     # CSVs e PDFs de otimização de bins
os.makedirs(_log_dir,     exist_ok=True)
os.makedirs(_fig_dir,     exist_ok=True)
os.makedirs(_bin_opt_dir, exist_ok=True)
result_cache.init()

# redireciona o caminho do log de dados insuficientes para dentro da run
LOG_INSUF_DATA = os.path.join(_run_dir, "insufficient_data_log.txt")

_csv_filename = os.path.join(_run_dir, f"results_{_run_ts}.csv")

# log geral para headers globais; arquivos por timeframe são abertos via switch_file
_general_log = os.path.join(_log_dir, "_run_geral.txt")
_run_logger  = RunLogger(_general_log)
_all_rows    = []   # acumula uma linha por (asset, timeframe, step/embed)
sys.stdout   = _run_logger
print(f"Início da execução  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Estrutura de saída:")
print(f"  {_run_dir}/")
print(f"  ├── results_{_run_ts}.csv")
print(f"  ├── final_results.csv" if not SKIP_SINGLE else f"  (final_results.csv omitido — SKIP_SINGLE=True)")
print(f"  ├── insufficient_data_log.txt")
print(f"  ├── logs/   ← {{ASSET}}_{{TF}}_{{modo}}.txt por ativo × timeframe × modo")
print(f"  └── figures/ ← PDFs e PNGs de cada ativo")
print(f"Ativos na lista : {len(ASSET_LIST)}")

# ================================================================
# LOOP PRINCIPAL SOBRE ATIVOS E MODOS
# ================================================================
for ASSET in ASSET_LIST:
    _log_header(ASSET, level=1)

    # Filtra apenas os timeframes cujo arquivo existe em DATA_PATH
    _all_tf_files = [os.path.join(DATA_PATH, ASSET + sfx) for sfx in TIMEFRAME_SUFFIX]
    _valid_pairs  = [(f, lbl) for f, lbl in zip(_all_tf_files, TIMEFRAME_LABELS)
                     if os.path.exists(f)]
    _missing      = [(f, lbl) for f, lbl in zip(_all_tf_files, TIMEFRAME_LABELS)
                     if not os.path.exists(f)]
    for _mf, _ml in _missing:
        print(f"  [ARQUIVO AUSENTE] {_mf} — timeframe {_ml} ignorado")
        with open(LOG_INSUF_DATA, 'a', encoding=FILE_ENCODING) as _lf:
            _lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | asset={ASSET}"
                      f" | tf={_ml} | arquivo ausente: {_mf}\n")

    if not _valid_pairs:
        print(f"  [AVISO] Nenhum arquivo de dados encontrado para {ASSET} — ativo ignorado.")
        continue

    TIMEFRAME_FILES   = [f   for f, _ in _valid_pairs]   # rebindado por ativo — OK
    _asset_tf_labels  = [lbl for _, lbl in _valid_pairs]  # local: não altera o global TIMEFRAME_LABELS

    for MULTI_STEP_ANALYSIS in (True,False):
        if MULTI_STEP_ANALYSIS:
            for MULTI_STEP_TYPE in ('step', 'embed'):
                _log_header(f"MODO MULTI-STEP: {MULTI_STEP_TYPE.upper()}", level=2)

                STEP_LIST = STEP_LIST_STEP if MULTI_STEP_TYPE == 'step' else STEP_LIST_EMBED
                pdf_name  = os.path.join(_fig_dir, f"{ASSET}_{MULTI_STEP_TYPE}.pdf")
                multi_data = {}


                if MULTI_STEP_TYPE == 'step':
                    if SKIP_STEP:
                        STEP=BEST_STEP[ASSET]
                    else:
                        EMBED_DIM = 70 #reseta o valor inicial de EMBED_DIM
                        with PdfPages(pdf_name) as pdf:
                            for tf_file, tf_label in zip(TIMEFRAME_FILES, _asset_tf_labels):
                                _run_logger.switch_file(os.path.join(_log_dir, f"{ASSET}_{tf_label}_step.txt"))
                                _log_header(ASSET, level=1)
                                _log_header("MODO MULTI-STEP: STEP", level=2)
                                _log_header(f"Timeframe: {tf_label}  ({tf_file})", level=3)
                                multi_data[tf_label] = main_func(pdf, tf_file, tf_label)
                            #for tf_label in _asset_tf_labels:
                            #    print(tf_label, multi_data[tf_label].get('beta_lags'))
                            pdf.close()
                        print("calculando STEP ótimo")
                        pdf_path   = os.path.join(_fig_dir, ASSET + "_step.pdf")
                        output_csv = os.path.join(_run_dir, ASSET + "_step.csv")
                        _pdf_has_pages = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 500
                        if _pdf_has_pages:
                            wait_file_ready(pdf_path)
                            for _ in range(20):
                                epdf.pdf_to_csv(pdf_path, output_csv)
                                try:
                                    df_check = pd.read_csv(output_csv)
                                except pd.errors.EmptyDataError:
                                    df_check = pd.DataFrame()
                                if len(df_check) > 0:
                                    break
                                print("CSV ainda vazio, aguardando...")
                                time.sleep(1)
                            else:
                                raise RuntimeError("PDF não pôde ser lido após 20 tentativas")
                            merged_step, best_step = epdf.compute_instability_min(output_csv,'step')
                            STEP=int(best_step['STEP'])
                            result_cache.save_best_param(ASSET, 'step', STEP)
                        else:
                            _cached_step = result_cache.load_best_param(ASSET, 'step')
                            if _cached_step is not None:
                                STEP = _cached_step
                                print(f"  [cache] STEP ótimo carregado do cache: {STEP}")
                            else:
                                print(f"  [aviso] PDF vazio e sem STEP no cache — mantendo STEP={STEP}")
                        plot_fig_step(STEP, multi_data, tf_file, _asset_tf_labels)
                    print("Step finalizado ", STEP)


                else:  # 'embed'
                    if SKIP_EMBED:
                        EMBED_DIM=BEST_EMBED[ASSET]
                    else:
                        with PdfPages(pdf_name) as pdf:
                            for tf_file, tf_label in zip(TIMEFRAME_FILES, _asset_tf_labels):
                                _run_logger.switch_file(os.path.join(_log_dir, f"{ASSET}_{tf_label}_embed.txt"))
                                _log_header(ASSET, level=1)
                                _log_header("MODO MULTI-STEP: EMBED", level=2)
                                _log_header(f"Timeframe: {tf_label}  ({tf_file})", level=3)
                                for n_value in N_STEPS:
                                    index = tf_label + str(n_value)
                                    multi_data[index] = main_func(pdf, tf_file, tf_label, n_value)
                            pdf.close()
                        print("calculando EMBED_DIM ótimo")
                        pdf_path   = os.path.join(_fig_dir, ASSET + "_embed.pdf")
                        output_csv = os.path.join(_run_dir, ASSET + "_embed.csv")
                        _pdf_has_pages = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 500
                        if _pdf_has_pages:
                            wait_file_ready(pdf_path)
                            for _ in range(20):
                                epdf.pdf_to_csv(pdf_path, output_csv)
                                try:
                                    df_check = pd.read_csv(output_csv)
                                except pd.errors.EmptyDataError:
                                    df_check = pd.DataFrame()
                                if len(df_check) > 0:
                                    break
                                print("CSV ainda vazio, aguardando...")
                                time.sleep(1)
                            else:
                                raise RuntimeError("PDF não pôde ser lido após 20 tentativas")
                            merged_embed, best_embed = epdf.compute_instability_min(output_csv,'embed')
                            EMBED_DIM=int(best_embed['EMBED_DIM'])
                            result_cache.save_best_param(ASSET, 'embed_dim', EMBED_DIM)
                        else:
                            _cached_embed = result_cache.load_best_param(ASSET, 'embed_dim')
                            if _cached_embed is not None:
                                EMBED_DIM = _cached_embed
                                print(f"  [cache] EMBED_DIM ótimo carregado do cache: {EMBED_DIM}")
                            else:
                                print(f"  [aviso] PDF vazio e sem EMBED_DIM no cache — mantendo EMBED_DIM={EMBED_DIM}")
                        plot_fig_embed(EMBED_DIM, multi_data, tf_file, _asset_tf_labels)
                    print("Embed finalizado ", EMBED_DIM)


                print(f"[{ASSET}] modo '{MULTI_STEP_TYPE}' concluído → {pdf_name}")

        else:
            _log_header("MODO SINGLE", level=2)
            if USE_AR1:
                _log_header(f"AR1  phi={PHI_AR1}  ({TIMEFRAME_FILES[0]})", level=3)
                with PdfPages(os.path.join(_fig_dir, f"AR1_{PHI_AR1}.pdf")) as pdf:
                    main_func(pdf, TIMEFRAME_FILES[0], _asset_tf_labels[0])
            else:
                if not SKIP_SINGLE:
                    for tf_file, tf_label in zip(TIMEFRAME_FILES, _asset_tf_labels):
                        _run_logger.switch_file(os.path.join(_log_dir, f"{ASSET}_{tf_label}_single.txt"))
                        _log_header(ASSET, level=1)
                        _log_header("MODO SINGLE", level=2)
                        _log_header(f"Timeframe: {tf_label}  ({tf_file})", level=3)
                        with PdfPages(os.path.join(_fig_dir, os.path.basename(os.path.splitext(tf_file)[0]) + ".pdf")) as pdf:
                            main_func(pdf, tf_file, tf_label)

            if USE_AR1:
                pdf_path   = os.path.join(_fig_dir, f"AR1_{PHI_AR1}.pdf")
                output_csv = os.path.join(_run_dir,  f"AR1_{PHI_AR1}.csv")
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
            elif not SKIP_SINGLE:
                alpha = {}
                alpha_std = {}
                for tf_file, tf_label in zip(TIMEFRAME_FILES, _asset_tf_labels):
                    _tf_base   = os.path.basename(os.path.splitext(tf_file)[0])
                    pdf_path   = os.path.join(_fig_dir, _tf_base + ".pdf")
                    output_csv = os.path.join(_run_dir,  _tf_base + "_results.csv")
                    epdf.pdf_to_csv(pdf_path, output_csv)
                    _a, _as, _r, _rs, _bt, _bts, _bs, _bss, _l2 = epdf.collect_data(output_csv)
                    resultados.append({
                        'ATIVO'         : ASSET,
                        'TIMEFRAME'     : tf_label,
                        'BEST_STEP'     : STEP,
                        'BEST_EMBED_DIM': EMBED_DIM,
                        'ALPHA'         : f"{_a:.3f} ± {_as:.3f}",
                        'RATIO_FDT'     : f"{_r:.3f} ± {_rs:.3f}",
                        'BETA_T'        : f"{_bt:.3f} ± {_bts:.3f}",
                        'BETA_S'        : f"{_bs:.3f} ± {_bss:.3f}",
                        'L2'            : f"{_l2:.3f}",
                    })

            print(f"[{ASSET}] modo single concluído.")

    gc.collect()   # libera memória entre ativos

plt.show()
sep = '=' * 80
print(f"\n{sep}\n  EXECUÇÃO CONCLUÍDA  |  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{sep}")
print("Todos os ativos concluídos.")

# ── tabela resumo por ativo (final_results) ───────────────────────────────────
if not SKIP_SINGLE:
    df_resultados = pd.DataFrame(resultados)
    if USE_AR1:
        df_resultados.to_csv(os.path.join(_run_dir, f"final_results_AR1_{PHI_AR1}.csv"), index=False)
    else:
        df_resultados.to_csv(os.path.join(_run_dir, "final_results.csv"), index=False)
    print("\nTabela resumo por ativo (final_results.csv):")
    print(df_resultados)
else:
    print("\n[INFO] SKIP_SINGLE=True — final_results.csv não gerado. Use results_TIMESTAMP.csv.")

# ── tabela detalhada: uma linha por (asset, timeframe, step/embed) ────────────
df_all = pd.DataFrame(_all_rows)
df_all.to_csv(_csv_filename, index=False)
print(f"\nTabela detalhada salva em: {_csv_filename}")
print(f"  {len(df_all)} linhas  ×  {len(df_all.columns)} colunas")
print(f"  Colunas: {list(df_all.columns)}")

_run_logger.close()
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
# _asset_tf_labels = ['M5', 'H1']
# tf_colors = {'M5': '#2166AC', 'H1': '#D6604D'}
# N_STEPS   = [4, 5, 6] #multiplicadores para o moode embed
#
# multi_data = {}
#
# if MULTI_STEP_ANALYSIS and MULTI_STEP_TYPE == 'step':
#     with PdfPages(CSV_FILE.split('_')[0] + "_" + MULTI_STEP_TYPE + ".pdf") as pdf:
#         for tf_file, tf_label in zip(TIMEFRAME_FILES, _asset_tf_labels):
#             multi_data[tf_label] = main_func(pdf,tf_file)
#         plot_fig_step(pdf,multi_data)
#
# elif MULTI_STEP_ANALYSIS and MULTI_STEP_TYPE == 'embed':
#     with PdfPages(CSV_FILE.split('_')[0] + "_" + MULTI_STEP_TYPE + ".pdf") as pdf:
#         for tf_file, tf_label in zip(TIMEFRAME_FILES, _asset_tf_labels):
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
