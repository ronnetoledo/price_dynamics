import os
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
# =========================
# CONFIGURAÇÕES
# =========================
CSV_FILE = "PETR4_H1.csv"

WINDOW = 400           # tamanho da janela individual
N_WINDOWS = 1        # quantas janelas vizinhas compõem o mesmo regime
STEP = 20             # passo do τ
EMBED_DIM = 80        # embedding temporal
WINDOW_OVERLAP = 400     #passo no empillhamento de janelas é uma estimativa do tempo médio de  variação
PROJECTION_HORIZON = 50 # horizonte da projeção
TYPE = 'returns'#returns, prices
MODE = 'scalar' #scalar,  indep e vector
max_tracks = 8
BUILD_PLOTS = True
# ================================
# ANALISE MULTI-STEP
# ================================
MULTI_STEP_ANALYSIS = False
MULTI_STEP_TYPE = 'embed' #step ou embed
if MULTI_STEP_TYPE == 'step':
    STEP_LIST = [5, 10, 20, 30, 50, 100]
elif MULTI_STEP_TYPE == 'embed':
    STEP_LIST = [20, 40, 60, 80, 100, 120, 140, 160]
    # ================================================================
    # FIGURAS DE PUBLICAÇÃO PRL — sequência de execução:
    #
    #  Passo 1 — Figura 1 + salvar dados intermediários:
    #    MULTI_STEP_ANALYSIS = False
    #    BUILD_PLOTS = True
    #
    #  Passo 2 — Figura 2 (painéis a, b, c):
    #    MULTI_STEP_ANALYSIS = True
    #    MULTI_STEP_TYPE     = 'step'
    #
    #  Passo 3 — Figura 3 (β vs L):
    #    MULTI_STEP_ANALYSIS = True
    #    MULTI_STEP_TYPE     = 'embed'
    # ================================================================

# =========================
# FUNÇÕES
# =========================

def hankel_embed(x, m):
    N = len(x) - m
    return np.column_stack([x[i:N+i] for i in range(m)])

def pca_cov(X):
    C = np.cov(X, rowvar=False)
    vals, vecs = eigh(C)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx], C

def mp_lambda_plus(var, q):
    return var * (1 + np.sqrt(q))**2

def estimate_frequency_fourier(v):
    # Padding para aumentar a resolução espectral
    n_fft = 128
    v_detrend = v - np.mean(v)
    yf = np.abs(rfft(v_detrend, n=n_fft))
    xf = rfftfreq(n_fft, d=1)  # d=1 assume o passo do embedding

    # Centro de massa da frequência (frequência média ponderada pela potência)
    # Isso evita o erro de "pular" entre bins discretos
    if np.sum(yf) == 0: return np.nan
    return np.sum(xf * yf) / np.sum(yf)

def hankel_embed_ohlc(rO, rH, rL, rC, m):
    N = len(rO) - m
    blocks = []

    for i in range(m):
        col = np.column_stack([
            rO[i:N+i],
            rH[i:N+i],
            rL[i:N+i],
            rC[i:N+i],
        ])
        blocks.append(col)

    # concatena tudo lado a lado → dimensão 4*m
    return np.hstack(blocks)

def hankel_embed_ohlc_indep(rO, rH, rL, rC, m):
    XO = hankel_embed(rO, m)
    XH = hankel_embed(rH, m)
    XL = hankel_embed(rL, m)
    XC = hankel_embed(rC, m)
    return np.vstack([XO, XH, XL, XC])

def plot_mode(vec, ax, title):
    if MODE == 'vector':
        O = vec[0::4]
        H = vec[1::4]
        L = vec[2::4]
        C = vec[3::4]

        ax.plot(O, label='O')
        ax.plot(H, label='H')
        ax.plot(L, label='L')
        ax.plot(C, label='C')
        ax.legend()

    else:  # scalar ou indep
        ax.plot(vec, label='modo temporal')
        ax.legend()

    ax.set_title(title)

def plot_eigenvector_candles(vec, ax, title,EMBED_current):
    m = EMBED_current
    width = 0.35

    for k in range(m):
        O = vec[4*k]
        H = vec[4*k + 1]
        L = vec[4*k + 2]
        C = vec[4*k + 3]

        # haste (high-low)
        ax.plot([k, k], [L, H], linewidth=1)

        # corpo (open-close)
        lower = min(O, C)
        height = abs(C - O)

        rect = plt.Rectangle(
            (k - width/2, lower),
            width,
            height if height > 1e-8 else 1e-8,  # evita corpo zero
            fill=False
        )
        ax.add_patch(rect)

    ax.set_xlim(-1, m)
    ax.set_title(title)

def validate_fdt(K_series, A_series, dt, verbose=True):
    """
    Valida o Teorema de Flutuação-Dissipação para uma série de matrizes K_t.

    Parâmetros
    ----------
    K_series : array (T, m, m)
        Série temporal das matrizes K_t.
    A_series : array (T-1, m, m)
        Drift estimado A(K_t) para cada t.
        Deve ter mesmo comprimento que dK.
    dt : float
        Passo temporal.
    verbose : bool
        Se True, imprime métricas.

    Retorna
    -------
    results : dict
        Contém F_hat, D_hat, erro_relativo, erro_espectral.
    """

    K_series = np.asarray(K_series)
    A_series = np.asarray(A_series)

    T = len(K_series)
    m = K_series.shape[1]

    # --- 1) Incrementos ---
    dK = K_series[1:] - K_series[:-1]

    if len(A_series) != len(dK):
        raise ValueError("A_series deve ter tamanho T-1.")

    # --- 2) Resíduos ---
    eps = dK - A_series * dt

    # --- 3) Termo de Flutuação ---
    F_hat = np.zeros((m, m))

    for t in range(len(eps)):
        F_hat += eps[t] @ eps[t].T

    F_hat /= (len(eps) * dt)

    # --- 4) Termo Dissipativo ---
    D_hat = np.zeros((m, m))

    for t in range(len(A_series)):
        Kt = K_series[t]
        At = A_series[t]

        D_hat += -(At @ Kt + Kt @ At.T)

    D_hat /= len(A_series)

    # --- 5) Métricas ---
    diff = F_hat - D_hat

    erro_relativo = np.linalg.norm(diff, 'fro') / np.linalg.norm(D_hat, 'fro')

    # erro espectral (maior autovalor da diferença)
    eigvals = np.linalg.eigvalsh(diff)
    erro_espectral = np.max(np.abs(eigvals))

    # parte simétrica da diferença
    diff_sym = 0.5 * (diff + diff.T)

    if verbose:
        print("==== Validação FDT ====")
        print("Dimensão matriz:", m)
        print("Erro relativo (Frobenius):", erro_relativo)
        print("Erro espectral:", erro_espectral)
        print("Norma F_hat:", np.linalg.norm(F_hat, 'fro'))
        print("Norma D_hat:", np.linalg.norm(D_hat, 'fro'))

    return {
        "F_hat": F_hat,
        "D_hat": D_hat,
        "erro_relativo": erro_relativo,
        "erro_espectral": erro_espectral,
        "diff": diff,
        "diff_sym": diff_sym
    }

def compute_conditional_drift(K_series, window=20, delta_t=1.0):
    """
    Estimativa suavizada do drift médio.
    """
    T = len(K_series) - window

    drift_estimates = []

    for t in range(T):
        dK_mean = 0.0

        for s in range(window):
            dK_mean += (K_series[t + s + 1] - K_series[t + s])

        dK_mean /= window
        drift_estimates.append(dK_mean / delta_t)



    return np.array(drift_estimates)

def spectral_entropy(eigenvalues):
    lambdas = np.array(eigenvalues)
    lambdas = lambdas[lambdas > 0]

    p = lambdas / np.sum(lambdas)
    S = -np.sum(p * np.log(p))
    S_norm = S / np.log(len(p))

    return S_norm

def hill_estimator(tau_star, min_plateau_len=5, stability_threshold=0.05):
    """
    Estima alpha pelo método de Hill com detecção automática de platô.
    Retorna (alpha_hat, k_lo, k_hi) onde [k_lo, k_hi] é a região estável.
    """
    tau_sorted = np.sort(tau_star)[::-1]
    n = len(tau_sorted)
    alphas = []
    ks = list(range(2, n - 1))

    for k in ks:
        tail = tau_sorted[:k]
        x_min = tau_sorted[k]
        if x_min <= 0:
            alphas.append(np.nan)
            continue
        alpha_hat = 1 + k / np.sum(np.log(tail / x_min))
        alphas.append(alpha_hat)

    alphas = np.array(alphas)
    dalpha = np.abs(np.diff(alphas))

    # detecta platô: janela deslizante onde variação máxima < threshold
    best_start = None
    best_len = 0
    current_start = 0
    current_len = 1

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
        alpha_plateau = np.nanmean(alphas[best_start: best_start + best_len])
        return ks,alphas,alpha_plateau, k_lo, k_hi
    else:
        # fallback: percentil 75 como threshold conservador
        xmin = np.percentile(tau_star, 75)
        tail = tau_star[tau_star >= xmin]
        alpha_fallback = 1 + len(tail) / np.sum(np.log(tail / xmin))
        return ks,alphas,alpha_fallback, None, None

def hill_plot(ks, alphas,k_lo,k_hi, label=''):
    """
    Plota o estimador de Hill como função do número de pontos
    da cauda usados na estimativa.
    tau_star: array de tempos entre regimes
    """
    plt.figure()
    plt.plot(list(ks), alphas, label=label, lw=1.5)
    plt.axhline(2.0, color='orange', linestyle='--', lw=1, label=r'$\alpha = 2$')
    plt.xlabel('k (número de pontos da cauda)')
    plt.ylabel(r'$\hat{\alpha}(k)$')
    plt.ylim(0, 6)
    plt.legend()
    # Sombrear a região de platô estável
    # (identificar automaticamente como região onde |dα/dk| < limiar)
    #dalpha = np.abs(np.diff(alphas))
    #stable = np.where(dalpha < 0.05)[0]
    #if len(stable) > 5:
    #    k_lo = list(ks)[stable[0]]
    #    k_hi = list(ks)[stable[-1]]

    if k_lo is not None and k_hi is not None:plt.axvspan(k_lo, k_hi, alpha=0.15, color='blue', label='platô estável')

def build_causal_regime_matrix(center,EMBED_current,WINDOW_current,ret_O,ret_H,ret_L,ret_C):
    blocks = []

    for k in range(N_WINDOWS):
        end = center - k * WINDOW_OVERLAP
        start = end - WINDOW_current
        if start < 0:
            continue

        segO = ret_O[start:end]
        segH = ret_H[start:end]
        segL = ret_L[start:end]
        segC = ret_C[start:end]

        if MODE == 'scalar':
            X = hankel_embed(segC-np.mean(segC), EMBED_current)

        elif MODE == 'indep':
            X = hankel_embed_ohlc_indep(segO-np.mean(segO), segH-np.mean(segH), segL-np.mean(segL), segC-np.mean(segC), EMBED_current)

        elif MODE == 'vector':
            X = hankel_embed_ohlc(segO-np.mean(segO), segH-np.mean(segH), segL-np.mean(segL), segC-np.mean(segC), EMBED_current)

        blocks.append(X)

    if len(blocks) == 0:
        return None

    return np.vstack(blocks)

def main_func(tf_file,n_value=5):

    # recarrega o CSV para este timeframe
    df = pd.read_csv(tf_file, sep=';', encoding='utf-16')

    # =========================
    # LEITURA
    # =========================
    # df = pd.read_csv(CSV_FILE, sep=';', encoding='utf-16')

    times = pd.to_datetime(df['time']).values
    O = df['open'].values
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values

    if TYPE == 'returns':
        # Adicionar filtro de sessão
        times_dt = pd.to_datetime(df['time'])
        ret_O = np.diff(np.log(O))
        ret_H = np.diff(np.log(H))
        ret_L = np.diff(np.log(L))
        ret_C = np.diff(np.log(C))
        times = times[1:]
        O = O[1:]
        H = H[1:]
        L = L[1:]
        C = C[1:]

    else:
        ret_O = O
        ret_H = H
        ret_L = L
        ret_C = C

    prices = C  # continua usando close para reconstrução do preço

    alpha_vs_step = []
    beta_vs_step = []
    entropy_vs_step = []
    mpl2error_vs_step = []
    _tau_sorted = None
    _ccdf = None
    _beta_lags = None
    _beta_values = None
    loop_values = STEP_LIST if MULTI_STEP_ANALYSIS else [STEP]

    for loop_current in loop_values:

        # =========================
        # LOOP PRINCIPAL EM τ
        # =========================
        evecs_store = {}
        tracked_modes = []
        tau_axis = []
        gap_t = []
        var_total_t = []
        var_struct_t = []
        var_noise_t = []
        alpha_t = []
        bulk_evals_store = []
        bulk_vecs_store = []
        struct_evals_store = []
        struct_vecs_store = []
        eigenvalues_series = []
        lambda_plus_t = []
        var_data_store = []
        theta_modes_all = []  # theta_k de todos os modos
        theta_modes_bulk = []  # theta_k apenas do bulk
        gap_modes_bulk = []  # gaps locais do bulk
        theta_struct_t = []
        m_t = []
        prev_vecs_all = None
        prev_vals_all = None
        prev_vecs = None
        prev_m = None
        gap_min_t = []
        C_store = []
        entropy_series = []
        STEP_current = STEP
        EMBED_current = EMBED_DIM
        WINDOW_current = WINDOW
        if MULTI_STEP_ANALYSIS:
            if MULTI_STEP_TYPE == 'step':
                STEP_current = loop_current
            elif MULTI_STEP_TYPE == 'embed':
                EMBED_current = loop_current
                WINDOW_current = n_value * EMBED_current

        print("Calculando STEP: ", STEP_current)
        print("Calculando EMBED_DIM: ", EMBED_current)
        print("Calculando WINDOW: ", WINDOW_current)

        # for center in range(WINDOW, len(ret_C) - WINDOW, STEP):
        for center in range(WINDOW_current, len(ret_C) - WINDOW_current, STEP_current):
            # print("Construindo a matriz do regime")
            Xreg = build_causal_regime_matrix(center, EMBED_current,WINDOW_current,ret_O,ret_H,ret_L,ret_C)
            if Xreg is None:
                continue
            # print("Calculando PCA")
            vals, vecs, Correlation = pca_cov(Xreg)
            C_store.append(Correlation)
            # MP threshold
            # q = EMBED_DIM / Xreg.shape[0]
            q = Xreg.shape[1] / Xreg.shape[0]
            var = np.mean(np.var(Xreg, axis=0, ddof=1))
            var_data_store.append(var)
            # var = np.var(Xreg)
            lam_plus = mp_lambda_plus(var, q)
            lambda_plus_t.append(lam_plus)

            structural = vals > lam_plus
            m = np.sum(structural)

            vals_s = vals[:m]
            vecs_s = vecs[:, :m]

            vecs_all = vecs
            vals_all = vals

            vals_bulk = vals[m:]
            vecs_bulk = vecs[:, m:]

            eigenvalues_series.append(vals)
            if not MULTI_STEP_ANALYSIS:
                bulk_evals_store.append(vals_bulk)
                bulk_vecs_store.append(vecs_bulk)
                struct_evals_store.append(vals_s)
                struct_vecs_store.append(vecs_s)

            S_norm = spectral_entropy(vals)
            entropy_series.append(S_norm)

            m_t.append(m)

            # print("Calculando rotação")
            # =========================
            # Cálculo das rotações
            # =========================
            if not MULTI_STEP_ANALYSIS:

                if prev_vecs_all is not None:

                    try:

                        n_modes = min(prev_vecs_all.shape[1], vecs_all.shape[1])

                        # -------------------------------------------------
                        # 1) Theta modo-a-modo (todos os autovetores)
                        # -------------------------------------------------
                        theta_k = np.zeros(n_modes)

                        for k in range(n_modes):
                            v_prev = prev_vecs_all[:, k]
                            v_curr = vecs_all[:, k]
                            # produto interno com correção de sinal
                            dot = np.abs(np.dot(v_prev, v_curr))
                            dot = np.clip(dot, -1.0, 1.0)
                            theta_k[k] = np.arccos(dot)

                        theta_modes_all.append(theta_k)

                        # -------------------------------------------------
                        # 2) Theta estrutural (subespaço)
                        #    (mantém compatibilidade com seu código atual)
                        # -------------------------------------------------
                        if prev_m is not None and m is not None and m > 0 and prev_m > 0:

                            m_star = min(prev_m, m)

                            A = prev_vecs[:, :m_star]
                            B = vecs_s[:, :m_star]

                            ang_struct = subspace_angles(A, B)

                            theta_struct_t.append(np.sqrt(np.sum(ang_struct ** 2)))

                        else:
                            theta_struct_t.append(np.nan)

                        # -------------------------------------------------
                        # 3) Bulk modo-a-modo + gaps locais
                        # -------------------------------------------------
                        if m < n_modes - 2:

                            theta_bulk = theta_k[m + 1:-1]

                            gap_bulk = []

                            for k in range(m + 1, n_modes - 1):
                                gap_left = abs(vals_all[k] - vals_all[k - 1])
                                gap_right = abs(vals_all[k] - vals_all[k + 1])

                                gap_bulk.append(min(gap_left, gap_right))

                            theta_modes_bulk.append(np.array(theta_bulk))
                            gap_modes_bulk.append(np.array(gap_bulk))

                        else:
                            theta_modes_bulk.append(np.array([]))
                            gap_modes_bulk.append(np.array([]))


                    except:
                        theta_struct_t.append(np.nan)
                        theta_modes_all.append(np.array([]))
                        theta_modes_bulk.append(np.array([]))
                        gap_modes_bulk.append(np.array([]))

                else:
                    theta_struct_t.append(np.nan)
                    theta_modes_all.append(np.nan)
                    theta_modes_bulk.append(np.nan)
                    gap_modes_bulk.append(np.nan)

                # Projeção
                Phi = vecs_s[:, :m]

                price_pred = []
                price_null = []
                price_real = []
                time_series = []

                omega_lambda_relation = []  # Armazena (omega, lambda) para este tau
                if m > 0:
                    for i in range(min(m, 5)):  # Analisamos os primeiros 5 modos
                        phi_close = vecs_s[:, i][3::4]
                        mode_time_series = Xreg[:, 3::4] @ phi_close
                        w = estimate_frequency_fourier(mode_time_series)
                        l = vals_s[i]
                        omega_lambda_relation.append((w, l))

                # Armazenar a média da relação ou o valor do modo principal para o gráfico temporal
                # Vamos guardar a inclinação da relação log-log (se existir) ou apenas o w1
                if len(omega_lambda_relation) > 1:
                    ws = np.array([x[0] for x in omega_lambda_relation])
                    ls = np.array([x[1] for x in omega_lambda_relation])
                    # Evitar log de zero
                    ws = ws[ws > 0]
                    ls = ls[ls > 0]
                    if len(ws) > 1:
                        # Regressão linear no espaço log-log para ver o expoente alpha: lambda ~ omega^alpha
                        try:
                            coeffs = np.polyfit(np.log(ws), np.log(ls), 1)
                            alpha_t.append(coeffs[0])
                        except:
                            alpha_t.append(np.nan)
                    else:
                        alpha_t.append(np.nan)
                else:
                    alpha_t.append(np.nan)

                current = list(vals_s)

                # cria linha vazia fixa
                row = [np.nan] * max_tracks
                for i in range(min(len(current), max_tracks)):
                    row[i] = current[i]
                tracked_modes.append(row)

                # Salva autovetores em alguns τ para inspeção
                if m > 0 and len(evecs_store) < 5:
                    evecs_store[times[center]] = vecs_s.copy()

                # GAP correto: distância de cada estrutural para TODOS os outros modos
                if m > 0:
                    deltas = []
                    for k in range(m):
                        lambda_k = vals[k]  # usa lista completa
                        diffs = np.abs(lambda_k - np.delete(vals, k))
                        delta_k = np.min(diffs)
                        deltas.append(delta_k)

                    deltas = np.array(deltas)
                    gap_min_t.append(np.min(deltas))

                    # quantidade teórica relevante
                    G_tau = np.mean(1.0 / (deltas ** 2))
                    gap_t.append(G_tau)

                else:
                    gap_min_t.append(np.nan)
                    gap_t.append(np.nan)

                # Resíduo
                # Energia total
                var_total = np.sum(vals)
                # Energia estrutural
                var_struct = np.sum(vals[:m]) if m > 0 else 0.0
                # Energia do ruído (bulk)
                var_noise = np.sum(vals[m:])

                var_total_t.append(var_total)
                var_struct_t.append(var_struct / var_total)
                var_noise_t.append(var_noise / var_total)

                prev_vecs_all = vecs_all.copy()
                prev_vals_all = vals_all.copy()
                prev_vecs = vecs_s.copy()
                prev_m = m

                tau_axis.append(times[center])

        tracked_modes = np.array(tracked_modes, dtype=float)
        all_bulk_rescaled = []
        all_vals_rescaled = []
        all_q = []

        for t in range(len(eigenvalues_series)):

            vals = eigenvalues_series[t]
            m = m_t[t]

            if m >= len(vals):
                continue

            vals_bulk = vals[m:]

            if len(vals_bulk) < 3:
                continue

            # Xreg = Xreg_store[t]
            #
            # N_dim = Xreg.shape[1]
            # T_obs = Xreg.shape[0]
            #
            # q_emp = N_dim / T_obs

            # var_data = np.mean(np.var(Xreg, axis=0))
            q_emp    = EMBED_current / (WINDOW_current - EMBED_current)
            var_data = var_data_store[t]
            # reescala os autovalores
            vals_rescaled = vals_bulk / var_data

            all_bulk_rescaled.extend(vals_rescaled)
            all_vals_rescaled.append(vals / var_data)
            all_q.append(q_emp)

        all_bulk_rescaled = np.array(all_bulk_rescaled)
        if not MULTI_STEP_ANALYSIS:
            # =========================
            # TESTE 1 — Amplificação ~ 1/gap²
            # =========================
            gap_arr = np.array(gap_t)
            angle_arr = np.array(theta_struct_t)
            # angle_arr = angle_arr[~np.isnan(angle_arr)]
            # =========================
            # TESTE 1 — Theta_bulk,k ~ 1/gap_k²
            # =========================
            theta_list = []
            gap_list = []

            for th, gp in zip(theta_modes_bulk, gap_modes_bulk):
                if isinstance(th, np.ndarray) and isinstance(gp, np.ndarray):
                    n = min(len(th), len(gp))
                    theta_list.extend(th[:n])
                    gap_list.extend(gp[:n])

            theta_arr = np.array(theta_list)
            gap_arr = np.array(gap_list)

            mask = (~np.isnan(theta_arr)) & (~np.isnan(gap_arr)) & (gap_arr > 0) & (theta_arr > 0)
            if np.sum(mask) > 10:
                x = np.log(1 / (gap_arr[mask] ** 2))
                y = np.log(theta_arr[mask])

                coeffs = np.polyfit(x, y, 1)

                print("Expoente log-log (theta_bulk vs 1/gap²):", coeffs[0])

            # =========================
            # TESTE 2 — Rotação bulk vs variância bulk
            # =========================
            theta_bulk_rms = []
            for th in theta_modes_bulk:
                if isinstance(th, np.ndarray) and len(th) > 0:
                    theta_bulk_rms.append(np.sqrt(np.mean(th ** 2)))
                else:
                    theta_bulk_rms.append(np.nan)
            theta_bulk_rms = np.array(theta_bulk_rms)
            var_noise_arr = np.array(var_noise_t)
            mask = (~np.isnan(theta_bulk_rms)) & (~np.isnan(var_noise_arr))
            if np.sum(mask) > 20:
                corr = np.corrcoef(theta_bulk_rms[mask], var_noise_arr[mask])[0, 1]
                print("Correlação theta_bulk vs variância do bulk:", corr)

        # índices de mudança de regime — versão mínima para MULTI_STEP
        if MULTI_STEP_ANALYSIS:
            indices_m_change = [i for i in range(1, len(m_t))
                                if m_t[i] != m_t[i - 1]]
        else:# TESTE 3 completo (linhas originais 766–963 aqui, sem alteração)
            dtheta = np.diff(angle_arr)
            # ...            # =========================
            # TESTE 3 - saltos espectrais
            # =========================

            dtheta = np.diff(angle_arr)
            mu = np.nanmean(dtheta)
            sigma = np.nanstd(dtheta)
            threshold_theta = np.percentile(np.abs(dtheta), 95)

            # spectral_jump_count = 0
            jump_modes = 0
            jump_compression = 0
            jump_theta = 0
            jump_spec = 0
            indices_m_change = []
            indices_compressao = []
            indices_rotacao = []
            indices_dist = []
            V_t = []
            gaps_mean = []
            gaps_min = []
            for j in range(len(dtheta)):
                i = j + 1

                vals_now = all_vals_rescaled[i]
                vals_prev = all_vals_rescaled[i - 1]

                m_prev = m_t[i - 1]
                m_now = m_t[i]

                jump_flag = False

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
                    var_val = np.var(struct)
                    V = var_val / (mean_val ** 2)
                    Rmed = mean_val / struct[0]
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
                    dist = np.linalg.norm(
                        vals_now[:min(m_prev, m_now)] - vals_prev[:min(m_prev, m_now)]
                    )
                    norm = np.linalg.norm(vals_prev[:min(m_prev, m_now)])
                    if dist / norm > 0.2:
                        jump_spec += 1
                        indices_dist.append(i)

            print("Quantos candidatos tiveram mudança estrutural:", jump_modes, " ", jump_compression, " ", jump_theta,
                  " ", jump_spec)
            set_m = set(indices_m_change)
            set_comp = set(indices_compressao)
            set_rot = set(indices_rotacao)

            print("m ∩ compressão:", len(set_m & set_comp))
            print("m ∩ rotação   :", len(set_m & set_rot))
            print("comp ∩ rotação:", len(set_comp & set_rot))

            window = 5
            antecede = 0
            gap_before = []
            V_before = []

            for t in indices_m_change:
                for k in range(1, window + 1):
                    if (t - k) in set_comp:
                        antecede += 1
                        gap_before.append(gaps_min[t - k])
                        V_before.append(V_t[t - k])
                        break

            print("Crossings precedidos por compressão:", antecede)

            print("Gap médio antes de crossing:", np.nanmean(gap_before))
            print("Gap médio", np.nanmean(gaps_mean))

            print("Média V antes de crossing:", np.nanmean(V_before))
            print("Média V global:", np.nanmean(V_t))

            p_global = len(indices_compressao) / len(gaps_min)
            p_cond = antecede / len(indices_m_change)

            print("P(compressão):", p_global)
            print("P(compressão | crossing):", p_cond)

            # --------------------------
            # P(M | C)
            # --------------------------
            antec_cross = 0
            for t in indices_compressao:
                for k in range(1, window + 1):
                    if (t + k) in set_m:
                        antec_cross += 1
                        break

            p_M_given_C = antec_cross / len(indices_compressao)

            print("P(M | C):", p_M_given_C)
            print("P(M):", len(indices_m_change) / len(gaps_min))

            # --------------------------
            # Razões de risco
            # --------------------------
            p_C = p_global
            p_C_given_M = p_cond
            p_M = len(indices_m_change) / len(gaps_min)

            RR_C_given_M = p_C_given_M / p_C
            RR_M_given_C = p_M_given_C / p_M

            print("Risk ratio C|M:", RR_C_given_M)
            print("Risk ratio M|C:", RR_M_given_C)

            # --------------------------
            # Teste binomial simples
            # H0: compressão independente de crossing
            # --------------------------
            n = len(indices_m_change)
            k = antecede
            p0 = p_C

            test = binomtest(k, n, p0, alternative='greater')
            print("p-value binomial:", test.pvalue)

            # estatística z aproximada
            expected = n * p0
            std = np.sqrt(n * p0 * (1 - p0))
            z = (k - expected) / std
            print("z-score:", z)

            # -------------------------
            # TESTE CONJUNTO: C ∩ V_alto
            # -------------------------

            N = len(gaps_min)
            set_M = set(indices_m_change)

            # Média global de V (ignorando NaN)
            V_global = np.nanmean(V_t)

            indices_CV = []

            # Identificar instantes com compressão + variância alta
            for t in indices_compressao:
                if not np.isnan(V_t[t - 1]) and V_t[t - 1] > V_global:
                    indices_CV.append(t)

            n_CV = len(indices_CV)

            # Contar quantos levam a crossing na janela
            count_M_given_CV = 0

            for t in indices_CV:
                for k in range(1, window + 1):
                    if (t + k) in set_M:
                        count_M_given_CV += 1
                        break

            # Probabilidades
            P_M = len(indices_m_change) / N
            P_CV = n_CV / N

            if n_CV > 0:
                P_M_given_CV = count_M_given_CV / n_CV
            else:
                P_M_given_CV = np.nan

            print("P(M):", P_M)
            print("P(C ∩ V_alto):", P_CV)
            print("P(M | C ∩ V_alto):", P_M_given_CV)

            if not np.isnan(P_M_given_CV):
                print("Risk Ratio (M | C ∩ V):", P_M_given_CV / P_M)

            # ----------------------------------
            # SEPARAÇÃO: m ↑ e m ↓
            # ----------------------------------

            indices_M_up = []
            indices_M_down = []

            for j in range(len(m_t) - 1):
                if m_t[j + 1] > m_t[j]:
                    indices_M_up.append(j + 1)
                elif m_t[j + 1] < m_t[j]:
                    indices_M_down.append(j + 1)

            print("Total M_up:", len(indices_M_up))
            print("Total M_down:", len(indices_M_down))

            # ----------------------------------
            # Probabilidades separadas
            # ----------------------------------

            set_C = set(indices_compressao)
            set_M_up = set(indices_M_up)
            set_M_down = set(indices_M_down)

            # ---- P(C | M_up) e P(C | M_down)

            def prob_C_given_M(set_M_type):
                count = 0
                for t in set_M_type:
                    for k in range(1, window + 1):
                        if (t - k) in set_C:
                            count += 1
                            break
                return count / len(set_M_type) if len(set_M_type) > 0 else np.nan

            P_C_given_M_up = prob_C_given_M(set_M_up)
            P_C_given_M_down = prob_C_given_M(set_M_down)

            print("P(C | M_up):", P_C_given_M_up)
            print("P(C | M_down):", P_C_given_M_down)

            def prob_M_given_C(set_M_type):
                count = 0
                for t in indices_compressao:
                    for k in range(1, window + 1):
                        if (t + k) in set_M_type:
                            count += 1
                            break
                return count / len(indices_compressao)

            P_M_up_given_C = prob_M_given_C(set_M_up)
            P_M_down_given_C = prob_M_given_C(set_M_down)

            print("P(M_up | C):", P_M_up_given_C)
            print("P(M_down | C):", P_M_down_given_C)

            # ----------------------------------
            # Conjunto C ∩ V_alto já calculado como indices_CV
            # ----------------------------------

            set_CV = set(indices_CV)

            def prob_M_given_CV(set_M_type):
                count = 0
                for t in set_CV:
                    for k in range(1, window + 1):
                        if (t + k) in set_M_type:
                            count += 1
                            break
                return count / len(set_CV) if len(set_CV) > 0 else np.nan

            P_M_up_given_CV = prob_M_given_CV(set_M_up)
            P_M_down_given_CV = prob_M_given_CV(set_M_down)

            print("P(M_up | C ∩ V):", P_M_up_given_CV)
            print("P(M_down | C ∩ V):", P_M_down_given_CV)

            # -----------------------------------------
            # TESTE DE TENDÊNCIA DE λ_m ANTES DO m ↓
            # -----------------------------------------

            # window = 5  # use o mesmo que está usando nas outras análises

            lambda_diffs_down = []
            lambda_diffs_up = []

            for t in indices_M_down:

                # último modo estrutural antes do crossing
                m_prev = m_t[t - 1]

                if m_prev > 0:
                    for k in range(1, window + 1):

                        idx1 = t - k
                        idx0 = t - k - 1

                        if idx0 >= 0:
                            lam1 = all_vals_rescaled[idx1][m_prev - 1]
                            lam0 = all_vals_rescaled[idx0][m_prev - 1]

                            lambda_diffs_down.append(lam1 - lam0)

            for t in indices_M_up:

                m_prev = m_t[t - 1]

                if m_prev > 0:
                    for k in range(1, window + 1):

                        idx1 = t - k
                        idx0 = t - k - 1

                        if idx0 >= 0:
                            lam1 = all_vals_rescaled[idx1][m_prev - 1]
                            lam0 = all_vals_rescaled[idx0][m_prev - 1]

                            lambda_diffs_up.append(lam1 - lam0)

            # -----------------------------------------
            # Estatísticas
            # -----------------------------------------

            mean_down = np.mean(lambda_diffs_down)
            mean_up = np.mean(lambda_diffs_up)

            print("Média Δλ antes de m ↓:", mean_down)
            print("Média Δλ antes de m ↑:", mean_up)

            # Teste t unilateral (queremos verificar se média < 0)
            t_stat_down, p_value_down = stats.ttest_1samp(lambda_diffs_down, 0)

            # Ajuste para teste unilateral
            p_value_down_one_sided = p_value_down / 2 if mean_down < 0 else 1 - p_value_down / 2

            print("t-stat m ↓:", t_stat_down)
            print("p-value unilateral m ↓:", p_value_down_one_sided)

            t_stat_up, p_value_up = stats.ttest_1samp(lambda_diffs_up, 0)

            p_value_up_one_sided = p_value_up / 2 if mean_up < 0 else 1 - p_value_up / 2

            print("t-stat m ↑:", t_stat_up)
            print("p-value unilateral m ↑:", p_value_up_one_sided)

            # -----------------------------------------
            # TESTE: RETORNO APÓS TRANSIÇÃO
            # -----------------------------------------

            horizon = 5  # número de períodos à frente
            shift = 0
            returns_down = []
            returns_up = []

            for t in indices_M_down:
                index = WINDOW_current + t * STEP_current
                if (index + horizon + shift) < len(ret_C):
                    r_future = np.sum(ret_C[index + shift:index + shift + horizon])
                    returns_down.append(r_future)

            for t in indices_M_up:
                if (t + horizon + shift) < len(ret_C):
                    index = WINDOW_current + t * STEP_current
                    r_future = np.sum(ret_C[index + shift:index + shift + horizon])
                    returns_up.append(r_future)

            returns_down = np.array(returns_down)
            returns_up = np.array(returns_up)

            print("Média retorno após m ↓:", np.mean(returns_down))
            print("Média retorno após m ↑:", np.mean(returns_up))

            # Teste de diferença de médias
            t_stat, p_value = stats.ttest_ind(returns_down, returns_up, equal_var=False)

            print("t-stat diferença:", t_stat)
            print("p-value:", p_value)

            def test_shift_effect(indices_up, indices_down, returns, horizon=5, max_shift=10):

                results = []

                for shift in range(max_shift + 1):
                    ret_up = []
                    ret_down = []

                    for t in indices_up:
                        index = WINDOW_current + t * STEP_current
                        start = index + shift
                        end = start + horizon
                        if end < len(returns):
                            ret_up.append(np.sum(returns[start:end]))

                    for t in indices_down:
                        index = WINDOW_current + t * STEP_current
                        start = index + shift
                        end = start + horizon
                        if end < len(returns):
                            ret_down.append(np.sum(returns[start:end]))

                    if len(ret_up) > 5 and len(ret_down) > 5:
                        t_stat, p_val = stats.ttest_ind(ret_down, ret_up, equal_var=False)
                        results.append((shift,
                                        np.mean(ret_down),
                                        np.mean(ret_up),
                                        t_stat,
                                        p_val))
                    else:
                        results.append((shift, np.nan, np.nan, np.nan, np.nan))

                return results

            results = test_shift_effect(
                indices_M_up,
                indices_M_down,
                ret_C,
                horizon=5,
                max_shift=12
            )

            for r in results:
                print(f"Shift={r[0]} | mean↓={r[1]:.6e} | mean↑={r[2]:.6e} | t={r[3]:.3f} | p={r[4]:.4f}")

            def excursion_test(indices_up, indices_down, returns,
                               horizon=10, shift=0):

                mfe_up = []
                mae_up = []
                mfe_down = []
                mae_down = []

                for t in indices_up:
                    index = WINDOW_current + t * STEP_current
                    start = index + shift
                    end = start + horizon
                    if end < len(returns):
                        path = np.cumsum(returns[start:end])
                        mfe_up.append(np.max(path))
                        mae_up.append(np.min(path))

                for t in indices_down:
                    index = WINDOW_current + t * STEP_current
                    start = index + shift
                    end = start + horizon
                    if end < len(returns):
                        path = np.cumsum(returns[start:end])
                        mfe_down.append(np.max(path))
                        mae_down.append(np.min(path))

                # Testes estatísticos
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

            mfe_up, mfe_down, mae_up, mae_down = excursion_test(indices_M_up, indices_M_down, ret_C, horizon=10,
                                                                shift=0)

        if BUILD_PLOTS:
            plt.figure()
            plt.plot(V_t, label="Variância normalizada dos autovalores")
            # plt.hist(gap_random, bins=30, alpha=0.5, label="Aleatório")
            plt.legend()

            plt.figure()
            plt.hist(gap_before, bins=30, alpha=0.5, label="Antes crossing")
            # plt.hist(gap_random, bins=30, alpha=0.5, label="Aleatório")
            plt.legend()
        # plt.show()

        indices = np.array(sorted(indices_m_change))

        # tempos entre mudanças
        waiting_times = np.diff(indices)
        corre = np.corrcoef(waiting_times[:-1], waiting_times[1:])
        print("Número de intervalos:", len(waiting_times))
        print("Tempo médio:", np.mean(waiting_times))
        print("Mediana:", np.median(waiting_times))
        print("Correlação: ", corre)

        tau = np.array(waiting_times)

        tau_sorted = np.sort(tau)
        ccdf = 1 - np.arange(1, len(tau_sorted) + 1) / len(tau_sorted)

        if BUILD_PLOTS:
            plt.figure()
            plt.loglog(tau_sorted, ccdf, marker='.', linestyle='none')
            plt.xlabel("tau")
            plt.ylabel("P(T > tau)")

        ks, alphas, alpha_hat, k_lo, k_hi = hill_estimator(tau)
        print(f"alpha estimado: {alpha_hat:.3f}  platô: k={k_lo}–{k_hi}")
        if BUILD_PLOTS: hill_plot(ks, alphas, k_lo, k_hi)
        if MULTI_STEP_TYPE == 'step':
            alpha_vs_step.append((STEP_current, alpha_hat))
        elif MULTI_STEP_TYPE == 'embed':
            alpha_vs_step.append((EMBED_current, alpha_hat))

        if BUILD_PLOTS:
            plt.figure()
            plt.hist(waiting_times, bins=30, density=True)
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Tempo entre mudanças")
            plt.ylabel("Densidade (log log)")
        # plt.show()

        C_store = np.array(C_store)

        def compute_ratio(K):
            dK = K[1:] - K[:-1]

            injection = np.sum(dK * dK, axis=(1, 2))
            dissip = 2 * np.sum(K[:-1] * dK, axis=(1, 2))

            energy_change = np.sum(K[1:] ** 2, axis=(1, 2)) - np.sum(K[:-1] ** 2, axis=(1, 2))
            return dK, injection, dissip, energy_change

        dk, injection, dissip, energy_change = compute_ratio(C_store)
        W = 200

        inj_local = np.convolve(injection, np.ones(W) / W, mode='valid')
        dis_local = np.convolve(dissip, np.ones(W) / W, mode='valid')
        energy_local = np.convolve(energy_change, np.ones(W) / W, mode='valid')
        ratio_local = dis_local + inj_local - energy_local

        print("D: ", -np.mean(dis_local), "I: ", np.mean(inj_local), "R: ", -np.mean(ratio_local), "Ratio: ",
              -np.mean(dis_local) / np.mean(inj_local))

        if BUILD_PLOTS:
            plt.figure()
            plt.plot(dis_local, label="Dissipação")
            plt.plot(inj_local, label="Injeção")
            plt.legend()

            plt.figure()
            plt.plot(ratio_local)
            plt.ylabel("Ratio")

            plt.figure()
            plt.plot(energy_local)
            plt.ylabel("Energy variation")

        def compute_injection(C_store, step):
            C_sub = C_store[::step]
            dK = C_sub[1:] - C_sub[:-1]
            inj = np.sum(dK * dK, axis=(1, 2))
            return np.mean(inj)

        steps = [1, 2, 4, 8, 16]

        for s in steps:
            print(s, compute_injection(C_store, s))
        values = np.array([compute_injection(C_store, s) for s in steps])
        coeff = np.polyfit(np.log(steps), np.log(values), 1)
        print("Expoente beta:", coeff[0])

        # ── armazena valores para a Fig 2 ───────────────────────────
        if STEP_current == 20:
            _tau_sorted = tau_sorted
            _ccdf = ccdf
            _beta_lags = steps  # [1,2,4,8,16]
            _beta_values = values  # E[||ΔK||²] por lag
        _beta_coeff = coeff  # [slope, intercept]

        if MULTI_STEP_TYPE == 'step':
            beta_vs_step.append((STEP_current, coeff[0]))
        elif MULTI_STEP_TYPE == 'embed':
            beta_vs_step.append((EMBED_current, coeff[0]))

        if not MULTI_STEP_ANALYSIS:
            def calc_K(vecs_store, evals_store):
                K_ = []
                for t in range(len(vecs_store)):
                    vecs_ = vecs_store[t]
                    lambda_ = evals_store[t]
                    if vecs_.size == 0:
                        continue
                    K_.append(vecs_ @ np.diag(lambda_) @ vecs_.T)
                K_ = np.array(K_)
                return (K_)

            K_bulk = calc_K(bulk_vecs_store, bulk_evals_store)
            K_struct = calc_K(struct_vecs_store, struct_evals_store)
            dk, injection, dissip, energy_change = compute_ratio(K_bulk)
            inj_local = np.convolve(injection, np.ones(W) / W, mode='valid')
            dis_local = np.convolve(dissip, np.ones(W) / W, mode='valid')
            energy_local = np.convolve(energy_change, np.ones(W) / W, mode='valid')
            ratio_local = dis_local + inj_local - energy_local
            print("BULK: D: ", -np.mean(dis_local), "I: ", np.mean(inj_local), "R: ", -np.mean(ratio_local), "Ratio: ",
                  -np.mean(dis_local) / np.mean(inj_local))
            dk, injection, dissip, energy_change = compute_ratio(K_struct)
            inj_local = np.convolve(injection, np.ones(W) / W, mode='valid')
            dis_local = np.convolve(dissip, np.ones(W) / W, mode='valid')
            energy_local = np.convolve(energy_change, np.ones(W) / W, mode='valid')
            ratio_local = dis_local + inj_local - energy_local
            print("STRUCT: D: ", -np.mean(dis_local), "I: ", np.mean(inj_local), "R: ", -np.mean(ratio_local),
                  "Ratio: ", -np.mean(dis_local) / np.mean(inj_local))

            gap_min_t = np.array(gap_min_t)
            gap_min_t = gap_min_t[:-1]
            m_mid = m_t[:-1]
            m_mid = np.array(m_mid)
            mask = (~np.isnan(dtheta))
            mask2 = (~np.isnan(dtheta)) & (~np.isnan(gap_min_t))
            corr = np.corrcoef(m_mid[mask], dtheta[mask])[0, 1]
            dm = np.diff(m_t)
            corr2 = np.corrcoef(dm[mask], dtheta[mask])[0, 1]
            corr_gap = np.corrcoef(gap_min_t[mask2], np.abs(dtheta[mask2]))[0, 1]
            print("Correlaçoes <m,dtheta> <dm,dtheta> <gap_minimo,|dtheta|>", corr, " ", corr2, " ", corr_gap)

            # teste 4 - normalidade do bulk

            C_bulk_sum = None
            count = 0

            for t in range(len(bulk_vecs_store)):

                vecs_bulk = bulk_vecs_store[t]
                lambda_bulk = bulk_evals_store[t]

                if vecs_bulk.size == 0:
                    continue

                N = vecs_bulk.shape[0]

                C_bulk = vecs_bulk @ np.diag(lambda_bulk) @ vecs_bulk.T

                if C_bulk_sum is None:
                    C_bulk_sum = np.zeros_like(C_bulk)

                C_bulk_sum += C_bulk
                count += 1

            C_bulk_mean = C_bulk_sum / count

            # agora calcula métricas na média
            off_diag = C_bulk_mean - np.diag(np.diag(C_bulk_mean))

            E_off_mean = np.sum(off_diag ** 2) / np.sum(C_bulk_mean ** 2)

            diag_energy = np.sum(np.diag(C_bulk_mean) ** 2)
            off_energy = np.sum(off_diag ** 2)

            R_mean = np.sqrt(off_energy) / np.sqrt(diag_energy)

            print("E_off (médio):", E_off_mean)
            print("R (médio):", R_mean)

            # teste 5 variância do ângulo
            max_lag = 20
            msd = []

            for lag in range(1, max_lag):
                diffs = angle_arr[lag:] - angle_arr[:-lag]
                msd.append(np.nanmean(diffs ** 2))

            if BUILD_PLOTS:
                plt.figure()
                plt.plot(range(1, max_lag), msd)
                plt.title("MSD da rotação espectral")

            # plt.show()

        # =========================
        # TESTE 6 - MP GLOBAL REESCALADO
        # =========================

        def mp_pdf(lam, var, q):
            lam_minus = var * (1 - np.sqrt(q)) ** 2
            lam_plus = var * (1 + np.sqrt(q)) ** 2
            return np.where(
                (lam >= lam_minus) & (lam <= lam_plus),
                np.sqrt((lam_plus - lam) * (lam - lam_minus)) /
                (2 * np.pi * q * var * lam),
                0
            )

        # usar q médio para MP agregado
        q_mean = np.mean(all_q)

        xs = np.linspace(np.min(all_bulk_rescaled),
                         np.max(all_bulk_rescaled), 400)

        lambda_minus_theory = (1 - np.sqrt(q_mean)) ** 2
        lambda_minus_emp = np.min(all_bulk_rescaled)

        print(lambda_minus_emp, lambda_minus_theory)
        lambda_plus_theory = (1 + np.sqrt(q_mean)) ** 2
        lambda_plus_emp = np.max(all_bulk_rescaled)

        print(lambda_plus_emp, lambda_plus_theory)
        print("Diferença percentual: ", 100 * np.abs(lambda_plus_emp - lambda_plus_theory) / lambda_plus_theory,
              "%")

        hist_vals, bin_edges = np.histogram(
            all_bulk_rescaled,
            bins=80,
            density=True
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mp_vals = mp_pdf(bin_centers, 1.0, q_mean)

        dx = bin_centers[1] - bin_centers[0]

        L2_error = np.sum((hist_vals - mp_vals) ** 2) * dx
        L1_error = np.sum(np.abs(hist_vals - mp_vals)) * dx

        print("Erro L2 =", L2_error)
        print("Erro L1 =", L1_error)

        norm_factor = np.sum(mp_vals ** 2) * dx
        L2_relative = L2_error / norm_factor

        print("Erro L2 relativo = ", 100 * L2_relative, "%")
        if MULTI_STEP_TYPE == 'step':
            mpl2error_vs_step.append((STEP_current, 100 * L2_relative))
        elif MULTI_STEP_TYPE == 'embed':
            mpl2error_vs_step.append((EMBED_current, 100 * L2_relative))

        if BUILD_PLOTS:
            # ── FIGURA 1 PRL: MP com inset de erro L2 ──────────────────────────────
            # ── FIGURA 1 PRL ────────────────────────────────────────────────────────
            if not MULTI_STEP_ANALYSIS:
                fig1_prl, ax_mp = plt.subplots(figsize=(3.375, 2.6))

                lm_thy = (1 - np.sqrt(q_mean)) ** 2
                lp_thy = (1 + np.sqrt(q_mean)) ** 2

                ax_mp.bar(bin_centers, hist_vals,
                          width=bin_centers[1] - bin_centers[0],
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

                # inset: barra do erro L2 relativo
                ax_mp.text(0.93, 0.75,
                           f'Rel. $L^2$ error: {100 * L2_relative:.1f}%\n'
                           f'$\\lambda_+^{{\\rm emp}}={lambda_plus_emp:.3f}$\n'
                           f'$\\lambda_+^{{\\rm thy}}={lambda_plus_theory:.3f}$',
                           transform=ax_mp.transAxes,
                           ha='right', va='top',
                           fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', edgecolor='#CCCCCC',
                                     alpha=0.85))

                fig1_prl.tight_layout()
                out_mp = os.path.splitext(tf_file)[0] + '_fig1_mp.pdf'
                fig1_prl.savefig(out_mp, bbox_inches='tight')
                fig1_prl.savefig(out_mp.replace('.pdf', '.eps'), bbox_inches='tight', format='eps')
                fig1_prl.savefig(out_mp.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
                print(f"Figura 1 salva: {out_mp}")
        if not MULTI_STEP_ANALYSIS:
            # ==================================
            # TESTE 7 - ESTACIONARIEDADE
            # ==================================
            theta_clean = angle_arr[~np.isnan(angle_arr)]
            if len(theta_clean) > 20:
                adf_result = adfuller(theta_clean)
                print("ADF stat:", adf_result[0])
                print("p-value:", adf_result[1])
            else:
                print("Série muito curta ara ADF")

            # ==================================
            # TESTE 8 - ESCALING DO MSD
            # ================== ================
            dtheta = np.diff(theta_clean)
            print("Var(Δtheta):", np.var(dtheta))
            theta_mid = theta_clean[:-1]

            corr = np.corrcoef(theta_mid, dtheta)[0, 1]
            print("Corr( theta , Δtheta ):", corr)

            # ==================================
            # TESTE 9 - DRIFT F(K)
            # ==================================

            #dtheta = np.diff(theta_clean)
            #theta_mid = theta_clean[:-1]

            bins = np.linspace(min(theta_mid), max(theta_mid), 20)
            digitized = np.digitize(theta_mid, bins)

            drift = []
            centers = []

            for i in range(1, len(bins)):
                mask = digitized == i
                if np.sum(mask) > 10:
                    drift.append(np.mean(dtheta[mask]))
                    centers.append(np.mean(theta_mid[mask]))

            if BUILD_PLOTS:
                plt.figure()
                plt.plot(centers, drift)
                plt.title("Estimativa de F(theta)")

            # ==================================
            # TESTE 10 - NORMALIDADE DOS INCREMENTOS
            # ==================================

            jb = jarque_bera(dtheta)
            print("JB stat:", jb.statistic)
            print("p-value:", jb.pvalue)

        # ==================================
        # TESTE 11 - Entropia
        # ==================================
        #
        S_norm_med = np.mean(entropy_series)
        print("Entropia média: ", S_norm_med)
        if MULTI_STEP_TYPE == 'step':
            entropy_vs_step.append((STEP_current, S_norm_med))
        elif MULTI_STEP_TYPE == 'embed':
            entropy_vs_step.append((EMBED_current, S_norm_med))

        if BUILD_PLOTS:
            plt.figure()
            plt.plot(entropy_series)
            plt.title("Entropia Espectral Normalizada")
        # =========================
        # GRÁFICOS
        # =========================
        if BUILD_PLOTS:
            # escolhe o τ com maior número de modos
            idx_tau = np.nanargmax(m_t)
            tau_sel = tau_axis[idx_tau]
            # lista ordenada dos taus realmente calculados
            taus_validos = np.array(sorted(evecs_store.keys()))

            # encontra o mais próximo do tau desejado
            idx = np.argmin(np.abs(taus_validos - tau_sel))
            tau_real = taus_validos[idx]

            vecs_sel = evecs_store[tau_real]
            print("Gerando gráficos")

            fig = plt.figure(figsize=(18, 18))
            gs = gridspec.GridSpec(4, 2)  # Aumentado para 4 linhas

            # -----------------------
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(theta_struct_t, -np.log(gap_t))
            ax1.set_title("Log do Gap espectral mínimo -ln(Δ(τ))")

            # -----------------------
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(tau_axis, var_noise_t)
            ax2.set_title("Variância relativa do resíduo σ²η(τ)")

            # -----------------------
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(tau_axis, theta_struct_t)
            ax3.set_title("Rotação média dos autovetores θ(τ)")

            # -----------------------
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(tau_axis, m_t)
            # for idx in sorted(set(indices_m_change) & set(indices_compressao)):
            #    ax4.axvline(tau_axis[idx], linestyle='--', alpha=0.5)
            ax4.set_title("Número de modos estruturais m(τ)")
            # -----------------------
            # ax5: Autovalores
            ax5 = fig.add_subplot(gs[2, 0])
            for i in range(min(max_tracks, tracked_modes.shape[1])):
                ax5.plot(tau_axis, tracked_modes[:, i], label=f"λ{i + 1}")
            # for idx in indices_m_change:
            #    ax5.axvline(tau_axis[idx] , linestyle='--', alpha=0.5)
            ax5.set_title("Autovalores estruturais")
            ax5.legend()

            # -----------------------
            ax6 = fig.add_subplot(gs[2, 1])
            for i in range(min(6, vecs_sel.shape[1])):
                #    plot_eigenvector_candles(vecs_sel[:, i], ax6, f"Modo {i+1}")
                plot_mode(vecs_sel[:, i], ax6, f"Modo {i + 1}")
            ax6.legend()

            plt.tight_layout()

            out_name = os.path.splitext(tf_file)[0] + "_4D_" + MODE + "_ " + TYPE + ".png"
            plt.savefig(out_name, dpi=300)

            plt.show()

    return {
        'alpha_vs_step':  alpha_vs_step,
        'beta_vs_step':   beta_vs_step,
        'entropy_vs_step':  entropy_vs_step,
        'mpl2error_vs_step':   mpl2error_vs_step,
        'tau_sorted':     _tau_sorted,
        'ccdf':           _ccdf,
        'beta_lags':      _beta_lags,
        'beta_values':    _beta_values,
    }


# ================================================================
# TIMEFRAMES PARA FIGURA 2 (usado apenas com MULTI_STEP_ANALYSIS)
# ================================================================
TIMEFRAME_FILES = ['PETR4_M5.csv', 'PETR4_H1.csv']   # ordem importa
TIMEFRAME_LABELS = ['M5', 'H1']
#TIMEFRAME_COLORS = ['#2166AC', '#D6604D']
tf_colors  = {'M5': '#2166AC', 'H1': '#D6604D'}
N_STEPS = [4,5,6]
multi_data={}  # chave = label do timeframe

if MULTI_STEP_ANALYSIS and MULTI_STEP_TYPE == 'step':
    for tf_idx, (tf_file, tf_label) in enumerate(zip(TIMEFRAME_FILES, TIMEFRAME_LABELS)):
        multi_data[tf_label]=main_func(tf_file)
elif MULTI_STEP_ANALYSIS and MULTI_STEP_TYPE == 'embed':
    for tf_idx, (tf_file, tf_label) in enumerate(zip(TIMEFRAME_FILES, TIMEFRAME_LABELS)):
        for n_value in N_STEPS:
                index=tf_label+str(n_value)
                multi_data[index]=main_func(tf_file,n_value)
    # multi_data = main_func(CSV_FILE)
    # alpha_vs_step = multi_data['alpha_vs_step']
    # beta_vs_step = multi_data['beta_vs_step']
    # entropy_vs_step = multi_data['entropy_vs_step']
    # mpl2error_vs_step = multi_data['mpl2error_vs_step']
    # _tau_sorted = multi_data['tau_sorted']
    # _ccdf = multi_data['ccdf']
    # _beta_lags = multi_data['beta_lags']
    # _beta_values = multi_data['beta_values']
elif not MULTI_STEP_ANALYSIS:
    multi_data = main_func(CSV_FILE)
    alpha_vs_step = multi_data['alpha_vs_step']
    beta_vs_step = multi_data['beta_vs_step']
    entropy_vs_step = multi_data['entropy_vs_step']
    mpl2error_vs_step = multi_data['mpl2error_vs_step']
    _tau_sorted = multi_data['tau_sorted']
    _ccdf = multi_data['ccdf']
    _beta_lags = multi_data['beta_lags']
    _beta_values = multi_data['beta_values']

if MULTI_STEP_ANALYSIS:
    if MULTI_STEP_TYPE == 'step':
        alpha_vs_step = {}
        beta_vs_step = {}
        entropy_vs_step = {}
        mpl2error_vs_step = {}
        _tau_sorted = {}
        _ccdf = {}
        _beta_lags = {}
        _beta_values = {}

        for tf_label in TIMEFRAME_LABELS:
            alpha_vs_step[tf_label] = multi_data[tf_label]['alpha_vs_step']
            beta_vs_step[tf_label] = multi_data[tf_label]['beta_vs_step']
            entropy_vs_step[tf_label] = multi_data[tf_label]['entropy_vs_step']
            mpl2error_vs_step[tf_label] = multi_data[tf_label]['mpl2error_vs_step']
            _tau_sorted[tf_label] = multi_data[tf_label]['tau_sorted']
            _ccdf[tf_label] = multi_data[tf_label]['ccdf']
            _beta_lags[tf_label] = multi_data[tf_label]['beta_lags']
            _beta_values[tf_label] = multi_data[tf_label]['beta_values']

        steps_list  = {}
        betas_list  = {}
        alphas_list = {}
        for tf_label in TIMEFRAME_LABELS:
            steps_list[tf_label],  betas_list[tf_label]  = zip(*beta_vs_step[tf_label])
            steps_list[tf_label],  alphas_list[tf_label] = zip(*alpha_vs_step[tf_label])

        fig2_prl = plt.figure(figsize=(3.375, 3.8))
        gs2 = gridspec.GridSpec(2, 2, figure=fig2_prl,
                                left=0.03, right=0.97,
                                bottom=0.03, top=0.97,
                                wspace=0.48,hspace=0.46)

        # ── painel (a): E[||ΔK||²] vs lag ───────────────────────────────────────
        ax_a = fig2_prl.add_subplot(gs2[0])
        try:
            for tf_label in TIMEFRAME_LABELS:
                # linha β=1 passando pelo ponto central
                mid = len(_beta_lags[tf_label])//2
                x_ref = np.array([_beta_lags[tf_label][0], _beta_lags[tf_label][-1]], dtype=float)
                y_ref = _beta_values[tf_label][mid] * (x_ref / _beta_lags[tf_label][mid])**1.0
                if tf_label=='M5': ax_a.loglog(x_ref, y_ref, 'k--', lw=0.9)
                elif tf_label=='H1': ax_a.loglog(x_ref, y_ref, 'k--', lw=0.9, label=r'$\beta=1$')
                ax_a.loglog(_beta_lags[tf_label], _beta_values[tf_label], 'o-',
                            color=tf_colors[tf_label], ms=4, label=tf_label)

            ax_a.set_xlabel(r'Lag $s$')
            ax_a.set_ylabel(r'$\mathbb{E}[\|\Delta\mathbf{K}\|_F^2]$')
            ax_a.legend(frameon=False, fontsize=7)
        except NameError:
            ax_a.text(0.5, 0.5, 'Rode primeiro\ncom MULTI_STEP_ANALYSIS=False',
                      ha='center', va='center', transform=ax_a.transAxes, fontsize=7)
        ax_a.set_title('(a)', loc='left', fontweight='bold')
        for sp in ['top','right']: ax_a.spines[sp].set_visible(False)

        # ── painel (b): CCDF τ* ─────────────────────────────────────────────────
        ax_b = fig2_prl.add_subplot(gs2[1])
        try:
            ts0 = _tau_sorted[TIMEFRAME_LABELS[0]]
            bins_log = np.logspace(np.log10(max(ts0.min(), 1)),
                                   np.log10(ts0.max()), 25)
            ax_b.hist(ts0, bins=bins_log, density=True,
                      color=tf_colors[TIMEFRAME_LABELS[0]], alpha=0.4, label=TIMEFRAME_LABELS[0])
            # reta α=2: ancora no pico do histograma
            counts_tmp, edges_tmp = np.histogram(ts0, bins=bins_log, density=True)
            centers_tmp = np.sqrt(edges_tmp[:-1] * edges_tmp[1:])
            i_peak = np.argmax(counts_tmp)
            A_fit = counts_tmp[i_peak] * centers_tmp[i_peak] ** 2
            t_fit = centers_tmp[counts_tmp > 0]
            ax_b.loglog(t_fit, A_fit / t_fit**2, 'k--', lw=0.9, label=r'$\alpha=2$')

            # ax_b.loglog(_tau_sorted['M5'], _ccdf['M5'], '.', color='#333333', ms=3, alpha=0.7,
            #             label=CSV_FILE.split('_')[0]+" "+'M5')
            # # linha de referência α=2 → CCDF ~ τ^{-1}
            # mask_fit = _tau_sorted['M5'] >= np.median(_tau_sorted['M5'])
            # t_fit = _tau_sorted['M5'][mask_fit]
            # A_fit = _ccdf['M5'][mask_fit][0] * t_fit[0]
            # ax_b.loglog(t_fit, A_fit / t_fit, '--',
            #             color='#E84646', lw=1.2, label=r'$\alpha=2$')
            ax_b.set_xlabel(r'$\tau^*$ ')
            ax_b.set_ylabel(r'$p(\tau^*)$')
            # ax_b.set_ylabel(r'$P(T>\tau^*)$')
            ax_b.legend(frameon=False, fontsize=7)
        except NameError:
            ax_b.text(0.5, 0.5, 'Rode primeiro\ncom MULTI_STEP_ANALYSIS=False',
                      ha='center', va='center', transform=ax_b.transAxes, fontsize=7)
        ax_b.set_title('(b)', loc='left', fontweight='bold')
        for sp in ['top','right']: ax_b.spines[sp].set_visible(False)

        # ── painel (c): β vs STEP ────────────────────────────────────────────────
        ax_c = fig2_prl.add_subplot(gs2[2])
        for tf_label in TIMEFRAME_LABELS:
            ax_c.plot(steps_list[tf_label], betas_list[tf_label], 'o-', color=tf_colors[tf_label], ms=4,
                      label=tf_label)
        ax_c.axhline(1.0, color='k', lw=0.9, ls='--', label=r'$\beta=1$')
        ax_c.axhline(0.0, color='#AAAAAA', lw=0.6, ls=':')
        ax_c.set_xlabel(r'$\Delta\tau_0$ ')
        ax_c.set_ylabel(r'$\hat\beta$')
        ax_c.legend(frameon=False, fontsize=7, loc='lower left')
        ax_c.set_title('(c)', loc='left', fontweight='bold')
        for sp in ['top','right']: ax_c.spines[sp].set_visible(False)
        ax_c.axvspan(15, 25, alpha=0.2, color='#4DAF4A',label=r'$\Delta\tau_{\rm opt}$')

        # ── painel (d): alpha vs STEP ────────────────────────────────────────────────
        ax_d = fig2_prl.add_subplot(gs2[3])
        for tf_label in TIMEFRAME_LABELS:
            ax_d.plot(steps_list[tf_label], alphas_list[tf_label], 'o-', color=tf_colors[tf_label], ms=4,
                      label=tf_label)
        ax_d.axhline(2.0, color='k', lw=0.9, ls='--', label=r'$\alpha=2$')
        ax_d.axhline(0.0, color='#AAAAAA', lw=0.6, ls=':')
        ax_d.set_xlabel(r'$\Delta\tau_0$ ')
        ax_d.set_ylabel(r'$\hat\alpha$')
        ax_d.legend(frameon=False, fontsize=7)
        ax_d.set_title('(d)', loc='left', fontweight='bold')
        for sp in ['top','right']: ax_d.spines[sp].set_visible(False)
        # painel (d)
        #ax_d.axvspan(15, 25, alpha=0.2, color='#4DAF4A')

        fig2_prl.savefig(CSV_FILE.split('_')[0] + '_fig2_exponents.pdf',bbox_inches='tight')
        fig2_prl.savefig(CSV_FILE.split('_')[0] + '_fig2_exponents.png',dpi=300, bbox_inches='tight')
        print("Figura 2 salva.")

    # ── FIGURA 3 PRL: β vs EMBED_DIM com região hachurada ───────────────────
    if MULTI_STEP_TYPE == 'embed':
        alpha_vs_step = {}
        beta_vs_step = {}
        entropy_vs_step = {}
        mpl2error_vs_step = {}
        _tau_sorted = {}
        _ccdf = {}
        _beta_lags = {}
        _beta_values = {}

        for tf_label in TIMEFRAME_LABELS:
            for n_value in N_STEPS:
                index=tf_label+str(n_value)
                alpha_vs_step[index] = multi_data[index]['alpha_vs_step']
                beta_vs_step[index] = multi_data[index]['beta_vs_step']
                entropy_vs_step[index] = multi_data[index]['entropy_vs_step']
                mpl2error_vs_step[index] = multi_data[index]['mpl2error_vs_step']
                _tau_sorted[index] = multi_data[index]['tau_sorted']
                _ccdf[index] = multi_data[index]['ccdf']
                _beta_lags[index] = multi_data[index]['beta_lags']
                _beta_values[index] = multi_data[index]['beta_values']

        steps_list  = {}
        betas_list  = {}
        alphas_list = {}
        for tf_label in TIMEFRAME_LABELS:
            for n_value in N_STEPS:
                index=tf_label+str(n_value)
                steps_list[index],  betas_list[index]  = zip(*beta_vs_step[index])
                steps_list[index],  alphas_list[index] = zip(*alpha_vs_step[index])

        fig3_prl, ax3 = plt.subplots(figsize=(3.375, 2.6))

        for tf_label in TIMEFRAME_LABELS:
            for n_value in N_STEPS:
                index=tf_label+str(n_value)
                ax3.plot(list(steps_list[index]), list(betas_list[index]), 'o-', ms=5, lw=1.4,
                         label=tf_label+" "+ str(n_value)+ "x")

        # região hachurada L_opt = [75, 85]
        ax3.axvspan(75, 85, alpha=0.18, color='#F0A500',
                    label=r'$L_{\rm opt}$')
        ax3.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)

        # anotação da seta no máximo
        #betas_arr = np.array(betas_list)
        #idx_max   = int(np.argmax(betas_arr))
        #L_max     = steps_list[idx_max]
        #b_max     = betas_arr[idx_max]
        #ax3.annotate(rf'$L_{{\rm opt}}={L_max}$',
        #             xy=(L_max, b_max),
        #             xytext=(L_max + 15, b_max - 0.07),
        #             fontsize=7,
       #              arrowprops=dict(arrowstyle='->', lw=0.7, color='#555555'))

        ax3.set_xlabel(r'Embedding dimension $L$')
        ax3.set_ylabel(r'$\hat\beta$')
        ax3.legend(frameon=False, fontsize=7, loc='lower right')
        for sp in ['top','right']: ax3.spines[sp].set_visible(False)

        fig3_prl.tight_layout()
        fig3_prl.savefig(CSV_FILE.split('_')[0] + '_fig3_Lopt.pdf',bbox_inches='tight')
        fig3_prl.savefig(CSV_FILE.split('_')[0] + '_fig3_Lopt.png',dpi=300, bbox_inches='tight')
        print("Figura 3 salva.")

    plt.show()

    # steps, alphas = zip(*alpha_vs_step)
    # _, betas = zip(*beta_vs_step)
    # _, entropy = zip(*entropy_vs_step)
    # _, mpl2error = zip(*mpl2error_vs_step)
    # plt.figure()
    # plt.plot(steps, alphas)
    # plt.xlabel(MULTI_STEP_TYPE)
    # plt.ylabel("Alpha")
    #
    # plt.figure()
    # plt.plot(steps, betas)
    # plt.xlabel(MULTI_STEP_TYPE)
    # plt.ylabel("Beta (dK^2 vs dt)")
    #
    # plt.figure()
    # plt.plot(steps, entropy)
    # plt.xlabel(MULTI_STEP_TYPE)
    # plt.ylabel("Entropy")
    #
    # plt.figure()
    # plt.plot(steps, mpl2error)
    # plt.xlabel(MULTI_STEP_TYPE)
    # plt.ylabel("MP L2 error (%)")

    plt.show()

print("Concluído.")
