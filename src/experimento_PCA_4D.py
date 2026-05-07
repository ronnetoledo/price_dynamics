import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.linalg import subspace_angles
import matplotlib.gridspec as gridspec
import os
from scipy.fft import rfft, rfftfreq

# =========================
# CONFIGURAÇÕES
# =========================
CSV_FILE = "SPY_1hour.csv"

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
MULTI_STEP_TYPE = 'step' #step ou embed
STEP_LIST = [5, 10, 20, 30, 50, 100]
#STEP_LIST = [20, 40, 80, 120, 160]
# =========================
# LEITURA
# =========================
df = pd.read_csv(CSV_FILE, sep=';', encoding='utf-8')

times = pd.to_datetime(df['timestamp']).values
O = df['open'].values
H = df['high'].values
L = df['low'].values
C = df['close'].values

if TYPE=='returns':
    ret_O = np.diff(np.log(O))
    ret_H = np.diff(np.log(H))
    ret_L = np.diff(np.log(L))
    ret_C = np.diff(np.log(C))
    times = times[1:]
    O = O[1:]
    H = H[1:]
    L = L[1:]
    C = C[1:]

else :
    ret_O = O
    ret_H = H
    ret_L = L
    ret_C = C

prices = C  # continua usando close para reconstrução do preço

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


def predict_next_close(a, Phi,EMBED_current):
    m = EMBED_current
    phi_close = Phi[3::4, :]  # parte do close
    last_row = phi_close[-1, :]
    return a @ last_row

def predict_next_ohlc(a, Phi):
    phi_O = Phi[0::4, :]
    phi_H = Phi[1::4, :]
    phi_L = Phi[2::4, :]
    phi_C = Phi[3::4, :]

    rO = a @ phi_O[-1, :]
    rH = a @ phi_H[-1, :]
    rL = a @ phi_L[-1, :]
    rC = a @ phi_C[-1, :]

    return rO, rH, rL, rC

def extract_close_part(vec,EMBED_current):
    m = EMBED_current
    return vec[3::4]


import numpy as np


def compute_tfd(K_series, delta_t=1.0):
    """
    Estima termos do Teorema de Flutuação-Dissipação matricial.

    Parameters:
        K_series : array (T, N, N)
        delta_t  : intervalo temporal entre matrizes

    Returns:
        dict com dissipação, injeção e razão.
    """

    T = len(K_series) - 1

    dissip_sum = 0.0
    inj_sum = 0.0
    ratio=0.0
    for t in range(T):
        K_t = K_series[t]
        K_next = K_series[t + 1]

        dK = K_next - K_t

        Dl=(np.trace(K_next @ dK.T)+np.trace(dK @ K_next.T) )/ delta_t
        Il=np.trace(dK @ dK.T) / (delta_t)
        # Dissipação: -Tr(K dK) / dt
        dissip_sum += Dl

        # Injeção: Tr(dK^2) / (2 dt)
        inj_sum += Il
        ratio+=(-Dl/Il)


    dissip = -dissip_sum / T
    injection = inj_sum / T
    ratio = ratio/T

    #ratio = dissip / injection if injection != 0 else np.nan
    return {
        dissip,
        injection,
        ratio
    }
import numpy as np

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
# =========================
# CONSTRÓI A SUPERMATRIZ DE UM REGIME
# =========================

def build_causal_regime_matrix(center,EMBED_current):
    blocks = []

    for k in range(N_WINDOWS):
        end = center - k * WINDOW_OVERLAP
        start = end - WINDOW
        if start < 0:
            continue

        segO = ret_O[start:end]
        segH = ret_H[start:end]
        segL = ret_L[start:end]
        segC = ret_C[start:end]

        if MODE == 'scalar':
            X = hankel_embed(segC, EMBED_current)

        elif MODE == 'indep':
            X = hankel_embed_ohlc_indep(segO, segH, segL, segC, EMBED_current)

        elif MODE == 'vector':
            X = hankel_embed_ohlc(segO, segH, segL, segC, EMBED_current)

        blocks.append(X)

    if len(blocks) == 0:
        return None

    return np.vstack(blocks)

alpha_vs_step = []
beta_vs_step = []
entropy_vs_step = []
mpl2error_vs_step = []

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
    err_struct_map = {}
    err_null_map = {}
    err_time_map = {}
    range_real = []
    range_pred = []
    body_real = []
    body_pred = []
    dir_hits = []
    bulk_evals_store = []
    bulk_vecs_store = []
    struct_evals_store = []
    struct_vecs_store = []
    eigenvalues_series = []
    lambda_plus_t = []
    Xreg_store = []
    theta_modes_all = []      # theta_k de todos os modos
    theta_modes_bulk = []     # theta_k apenas do bulk
    gap_modes_bulk = []       # gaps locais do bulk
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
    if MULTI_STEP_ANALYSIS:
        if MULTI_STEP_TYPE=='step':
            STEP_current=loop_current
        elif MULTI_STEP_TYPE=='embed':
            EMBED_current=loop_current
            WINDOW = 5* EMBED_current

    print("Calculando STEP: ",STEP_current)
    print("Calculando EMBED_DIM: ",EMBED_current)
    print("Calculando WINDOW: ",WINDOW)

    #for center in range(WINDOW, len(ret_C) - WINDOW, STEP):
    for center in range(WINDOW, len(ret_C) - WINDOW, STEP_current):
        #print("Construindo a matriz do regime")

        Xreg = build_causal_regime_matrix(center,EMBED_current)
        if Xreg is None:
            continue
        Xreg_store.append(Xreg)
        #print("Calculando PCA")
        vals, vecs, Correlation = pca_cov(Xreg)
        C_store.append(Correlation)
        # MP threshold
        #q = EMBED_DIM / Xreg.shape[0]
        q = Xreg.shape[1] / Xreg.shape[0]
        var = np.mean(np.var(Xreg, axis=0))
        #var = np.var(Xreg)
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
        bulk_evals_store.append(vals_bulk)
        bulk_vecs_store.append(vecs_bulk)
        struct_evals_store.append(vals_s)
        struct_vecs_store.append(vecs_s)

        S_norm = spectral_entropy(vals)
        entropy_series.append(S_norm)

        m_t.append(m)

        #print("Calculando rotação")
        # =========================
        # Cálculo das rotações
        # =========================

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

    #Projeção
        Phi = vecs_s[:, :m]

        price_pred = []
        price_null = []
        price_real = []
        time_series = []

        for i in range(0,PROJECTION_HORIZON):  # WINDOW

            xO = ret_O[center + i - EMBED_current: center + i]
            xH = ret_H[center + i - EMBED_current: center + i]
            xL = ret_L[center + i - EMBED_current: center + i]
            xC = ret_C[center + i - EMBED_current: center + i]
            x_t = []
            if MODE in ['scalar', 'indep']:
                for k in range(EMBED_current): x_t.extend([xC[k]])
            else:
                for k in range(EMBED_current):
                    x_t.extend([
                        xO[k],
                        xH[k],
                        xL[k],
                        xC[k]
                    ])

                #x_t = np.hstack([xO, xH, xL, xC])
            x_t = np.array(x_t)

            if m > 0:
                a = x_t @ Phi  # coeficientes modais
                if MODE=='scalar': r_struct = a @ Phi[-1, :]
                elif MODE=='indep': r_struct = a @ Phi[-1, :]
                else:
                    rO,rH,rL,rC = predict_next_ohlc(a, Phi)
            else:
                if MODE=='scalar': r_struct = 0
                elif MODE=='indep': r_struct = 0
                else: rO,rH,rL,rC = 0,0,0,0
            #    r_struct = 0.0
            if MODE == 'vector':
                # preços anteriores reais
                O_prev = O[center + i - 1]
                H_prev = H[center + i - 1]
                L_prev = L[center + i - 1]
                C_prev = C[center + i - 1]
                if TYPE == 'returns':
                    # projeção estrutural
                    O_pred = O_prev * np.exp(rO)
                    H_pred = H_prev * np.exp(rH)
                    L_pred = L_prev * np.exp(rL)
                    C_pred = C_prev * np.exp(rC)
                else:
                    # projeção estrutural
                    O_pred = rO
                    H_pred = rH
                    L_pred = rL
                    C_pred = rC
                # valores reais
                O_real = O[center + i]
                H_real = H[center + i]
                L_real = L[center + i]
                C_real = C[center + i]
                # ===== GEOMETRIA DO CANDLE PREVISTO =====
                range_pred_t = H_pred - L_pred
                body_pred_t = abs(C_pred - O_pred)
                dir_pred = np.sign(C_pred - C_prev)
                range_real_t = H_real - L_real
                body_real_t = abs(C_real - O_real)
                dir_real = np.sign(C_real - C_prev)

                # erro médio da vela inteira (muito mais informativo!)
                e_struct = np.mean([
                #    abs(np.log(O_real / O_pred)),
                #    abs(np.log(H_real / H_pred)),
                #    abs(np.log(L_real / L_pred)),
                    abs(np.log(C_real / C_pred)),
                ])

                e_null = np.mean([
                #    abs(np.log(O_real / O_prev)),
                #    abs(np.log(H_real / H_prev)),
                #    abs(np.log(L_real / L_prev)),
                    abs(np.log(C_real / C_prev)),
                ])
                t_candle = times[center + i]
                if t_candle not in err_struct_map:
                    err_struct_map[t_candle] = e_struct
                    err_null_map[t_candle] = e_null
                    err_time_map[t_candle] = t_candle
                    price_pred.append((O_pred, H_pred, L_pred, C_pred))
                    price_null.append((O_prev, H_prev, L_prev, C_prev))
                    price_real.append((O_real, H_real, L_real, C_real))
                    # erro relativo de RANGE
                    range_real.append(range_real_t)
                    range_pred.append(range_pred_t)

                    # erro relativo de CORPO
                    body_real.append(body_real_t)
                    body_pred.append(body_pred_t)

                    # acerto de DIREÇÃO (1 acerto, 0 erro)
                    dir_hits.append(1 if dir_pred == dir_real else 0)
                time_series.append(t_candle)
            else:
                S_prev = prices[center + i - 1]
                if TYPE=='returns':
                    S_pred = S_prev * np.exp(r_struct)
                else:
                    S_pred = r_struct
                S_real = prices[center + i]
                e_struct = np.abs(np.log(S_real / S_pred))
                e_null = np.abs(np.log(S_real / S_prev))
                t_candle = times[center + i]
                if t_candle not in err_struct_map:
                    err_struct_map[t_candle] = e_struct
                    err_null_map[t_candle] = e_null
                    err_time_map[t_candle] = t_candle
                    price_pred.append(S_pred)
                    price_null.append(S_prev)
                    price_real.append(S_real)
                time_series.append(t_candle)


        # guarde para plotar depois
        if 'window_recons' not in globals():
            window_recons = []

        window_recons.append((times[center], price_real, price_pred,price_null))

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

        # GAP espectral
        #if m > 1:
        #    gaps = [abs(vals_s[i] - vals_s[j])
        #            for i in range(m) for j in range(i+1, m)]
        #    gap_t.append(min(gaps))
        #else:
        #    gap_t.append(np.nan)
        # GAP correto: distância de cada estrutural para TODOS os outros modos
        if m > 0:
            deltas = []
            for k in range(m):
                lambda_k = vals[k]  # usa lista completa
                diffs = np.abs(lambda_k - np.delete(vals, k))
                delta_k = np.min(diffs)
                deltas.append(delta_k)

            deltas = np.array(deltas)
            gap_min_t.append (np.min(deltas))

            # quantidade teórica relevante
            G_tau = np.mean(1.0 / (deltas ** 2))
            gap_t.append(G_tau)

        else:
            gap_min_t.append (np.nan)
            gap_t.append(np.nan)

        # Resíduo
        # Energia total
        var_total = np.sum(vals)

        # Energia estrutural
        var_struct = np.sum(vals[:m]) if m > 0 else 0.0

        # Energia do ruído (bulk)
        var_noise = np.sum(vals[m:])

        var_total_t.append(var_total)
        var_struct_t.append(var_struct/var_total)
        var_noise_t.append(var_noise/var_total)

        prev_vecs_all = vecs_all.copy()
        prev_vals_all = vals_all.copy()
        prev_vecs = vecs_s.copy()
        prev_m = m

        tau_axis.append(times[center])

    tracked_modes = np.array(tracked_modes, dtype=float)
    times_sorted = sorted(err_struct_map.keys())
    err_struct_series = np.array([err_struct_map[t] for t in times_sorted])
    err_null_series   = np.array([err_null_map[t] for t in times_sorted])
    err_time_series   = np.array([err_time_map[t] for t in times_sorted])
    if MODE == 'vector':
        range_real = np.array(range_real)
        range_pred = np.array(range_pred)
        body_real = np.array(body_real)
        body_pred = np.array(body_pred)
        dir_hits = np.array(dir_hits)

        print("\n===== MÉTRICAS GEOMÉTRICAS DO CANDLE =====")
        print(f"Erro médio relativo do RANGE: {100*(np.mean(np.abs(range_real-range_pred)))/np.mean(range_real):.2f}%")
        print(f"Erro médio relativo do CORPO: {100*(np.mean(np.abs(body_real-body_pred)))/np.mean(body_real):.2f}%")
        range_null_err = np.mean(np.abs(range_real - np.mean(range_real)))
        print(100 * range_null_err / np.mean(range_real))
        print(f"Taxa de acerto da DIREÇÃO: {100*np.mean(dir_hits):.2f}%")

    # =========================
    # TESTE 1 — Amplificação ~ 1/gap²
    # =========================
    gap_arr = np.array(gap_t)
    angle_arr = np.array(theta_struct_t)
    #angle_arr = angle_arr[~np.isnan(angle_arr)]
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
    gap_arr   = np.array(gap_list)

    mask = (~np.isnan(theta_arr)) & (~np.isnan(gap_arr)) & (gap_arr > 0) & (theta_arr > 0)

    if np.sum(mask) > 10:

        x = np.log(1 / (gap_arr[mask]**2))
        y = np.log(theta_arr[mask])

        coeffs = np.polyfit(x, y, 1)

        print("Expoente log-log (theta_bulk vs 1/gap²):", coeffs[0])


    # =========================
    # TESTE 2 — Rotação bulk vs variância bulk
    # =========================

    theta_bulk_rms = []

    for th in theta_modes_bulk:
        if isinstance(th, np.ndarray) and len(th) > 0:
            theta_bulk_rms.append(np.sqrt(np.mean(th**2)))
        else:
            theta_bulk_rms.append(np.nan)

    theta_bulk_rms = np.array(theta_bulk_rms)
    var_noise_arr  = np.array(var_noise_t)

    mask = (~np.isnan(theta_bulk_rms)) & (~np.isnan(var_noise_arr))

    if np.sum(mask) > 20:
        corr = np.corrcoef(theta_bulk_rms[mask], var_noise_arr[mask])[0,1]
        print("Correlação theta_bulk vs variância do bulk:", corr)

    #teste 3 - saltos espectrais
    # A principal pergunta é o que são mudanças de regime
    # podemos pensar em dois tipos de mudanças : mudanças nos autovalores, e mudança global - autovetores e autovalores.
    # no caso de termos uma mudança nos autovalores, mas uma mudança pequena nos autovetores, ainda estamos no mesmo regime?
    # a mudança global certamente é uma mudança de regime. Agora, e se os autovetores girarem lentamente, isso também nãoo seria uma mudança
    # lenta de regime?
    # podemos calcular a correlação entre m (número de autovvalores fora do bulk) e dtheta -> rotações estão conectadas com a mudança estrutural
    #temos 3 posibilidades de mudanças estruturais (regime)
    #1) os autovalores se aproximam, a rotação aumenta e isso provoca a mudança de regime, pois os autovalores se tornam degenerados (mudança de base)
    #2) o dW pode gerar um ruído que faz com que um autovalor entre (saia) do bulk, alterando o regime
    #3) o Poisson pode gerar um ruído grande o suficiente para provocar uma reorganização das bases.
    dtheta = np.diff(angle_arr)
    mu = np.nanmean(dtheta)
    sigma = np.nanstd(dtheta)
    threshold_theta = np.percentile(np.abs(dtheta), 95)

    #spectral_jump_count = 0
    jump_modes = 0
    jump_compression = 0
    jump_theta = 0
    jump_spec = 0
    indices_m_change = []
    indices_compressao = []
    indices_rotacao = []
    indices_dist = []
    V_t = []

    for j in range(len(dtheta)):
        i = j + 1

        vals_prev = eigenvalues_series[i-1]
        vals_now  = eigenvalues_series[i]

        m_prev = m_t[i-1]
        m_now  = m_t[i]

        jump_flag = False

        # 1) Crossing estrutural
        if m_prev != m_now:
            jump_modes += 1
            indices_m_change.append(i)
        # 2) Compressão estrutural
        struct_prev = vals_prev[:m_prev]
        if len(struct_prev) > 1:
            gaps = np.diff(np.sort(struct_prev)[::-1])
            if np.min(np.abs(gaps)) < np.mean(struct_prev)*0.05:
                jump_compression += 1
                indices_compressao.append(i)
            mean_val = np.mean(struct_prev)
            var_val = np.var(struct_prev)
            V = var_val / (mean_val ** 2)
            Rmed = mean_val/struct_prev[0]
            V_t.append(V)
        else:
            V_t.append(np.nan)

        # 3) Rotação extrema
        #if dtheta[j] is not np.nan:
        if abs(dtheta[j]) > threshold_theta:
            jump_theta += 1
            indices_rotacao.append(i)

        # 4) Distância espectral estrutural
        if m_prev > 0 and m_now > 0:
            dist = np.linalg.norm(
                vals_now[:min(m_prev,m_now)] - vals_prev[:min(m_prev,m_now)]
            )
            norm = np.linalg.norm(vals_prev[:min(m_prev,m_now)])
            if dist/norm > 0.2:
                jump_spec += 1
                indices_dist.append(i)

        #if jump_flag:
        #    spectral_jump_count += 1

    print("Quantos candidatos tiveram mudança estrutural:", jump_modes," ",jump_compression," ",jump_theta," ",jump_spec)
    set_m   = set(indices_m_change)
    set_comp = set(indices_compressao)
    set_rot  = set(indices_rotacao)

    print("m ∩ compressão:", len(set_m & set_comp))
    print("m ∩ rotação   :", len(set_m & set_rot))
    print("comp ∩ rotação:", len(set_comp & set_rot))

    window = 3
    antecede = 0

    for t in indices_m_change:
        for k in range(1, window+1):
            if (t-k) in set_comp:
                antecede += 1
                break

    print("Crossings precedidos por compressão:", antecede)

    gap_before = []
    gap_random = []
    gap_arr = np.array(gap_min_t)

    # limpar NaN e inf
    valid_mask = np.isfinite(gap_arr)
    for t in indices_m_change:
        if t >= 5:
            window_vals = gap_arr[t-5:t]
            window_vals = window_vals[np.isfinite(window_vals)]
            if len(window_vals) > 0:
                gap_before.append(np.mean(window_vals))

    # amostras aleatórias válidas
    #valid_indices = np.where(valid_mask)[0]
    #random_idx = np.random.choice(valid_indices[5:], size=len(gap_before), replace=False)

    #for t in random_idx:
        #window_vals = gap_arr[t-5:t]
        #window_vals = window_vals[np.isfinite(window_vals)]
        #if len(window_vals) > 0:
            #gap_random.append(np.mean(window_vals))

    print("Gap médio antes de crossing:", np.mean(gap_before))
    #print("Gap médio aleatório:", np.mean(gap_random))
    #from scipy.stats import ttest_ind
    #t_stat, p_val = ttest_ind(gap_before, gap_random, equal_var=False)
    #print("p-value:", p_val)
    V_before = []
    V_global = []
    for t in indices_m_change:
        if t > 0:
            V_before.append(V_t[t - 1])

    print("Média V(t-1):", np.nanmean(V_before))
    print("Média global:", np.nanmean(V_t))

    p_global = len(indices_compressao) / len(gap_min_t)
    p_cond = antecede / len(indices_m_change)

    print("P(compressão):", p_global)
    print("P(compressão | crossing):", p_cond)

    if BUILD_PLOTS:
        plt.figure()
        plt.plot(V_t, label="Variância normalizada dos autovalores")
        #plt.hist(gap_random, bins=30, alpha=0.5, label="Aleatório")
        plt.legend()

        plt.figure()
        plt.hist(gap_before, bins=30, alpha=0.5, label="Antes crossing")
        #plt.hist(gap_random, bins=30, alpha=0.5, label="Aleatório")
        plt.legend()
    #plt.show()

    indices = np.array(sorted(indices_m_change))

    # tempos entre mudanças
    waiting_times = np.diff(indices)
    corre=np.corrcoef(waiting_times[:-1], waiting_times[1:])
    print("Número de intervalos:", len(waiting_times))
    print("Tempo médio:", np.mean(waiting_times))
    print("Mediana:", np.median(waiting_times))
    print("Correlação: ",corre)

    tau = np.array(waiting_times)

    tau_sorted = np.sort(tau)
    ccdf = 1 - np.arange(1, len(tau_sorted)+1)/len(tau_sorted)

    if BUILD_PLOTS:
        plt.figure()
        plt.loglog(tau_sorted, ccdf, marker='.', linestyle='none')
        plt.xlabel("tau")
        plt.ylabel("P(T > tau)")
    #plt.show()

    xmin = np.percentile(tau, 50)  # por exemplo, metade superior
    tail = tau[tau >= xmin]

    alpha = 1 + len(tail) / np.sum(np.log(tail/xmin))

    print("alpha estimado:", alpha)

    if MULTI_STEP_TYPE == 'step':
        alpha_vs_step.append((STEP_current, alpha))
    elif MULTI_STEP_TYPE == 'embed':
        alpha_vs_step.append((EMBED_current, alpha))

    #plt.hist(waiting_times, bins=30)
    #plt.xlabel("Tempo entre mudanças")
    #plt.ylabel("Frequência")
    #plt.show()

    #plt.hist(waiting_times, bins=30, density=True)
    #plt.yscale("log")
    #plt.xlabel("Tempo entre mudanças")
    #plt.ylabel("Densidade (log)")
    #plt.show()

    if BUILD_PLOTS:
        plt.figure()
        plt.hist(waiting_times, bins=30, density=True)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Tempo entre mudanças")
        plt.ylabel("Densidade (log log)")
    #plt.show()

    C_store =  np.array(C_store)
    #dissip,injection,ratio = compute_tfd(C_store)
    #drift =  compute_conditional_drift(C_store)
    #E_series = [0.5 * np.trace(K @ K) for K in C_store]
    #print(validate_fdt(C_store,drift,1,True))
    #print("Dissipação: ",dissip,"Injeção: ",injection,"razão D/I: ",ratio)
    #print(np.mean([np.trace(K @ K) for K in C_store]))
    #injection =[np.trace((C_store[t+1]-C_store[t]) @
    #                        (C_store[t+1]-C_store[t]).T)
    #               for t in range(len(C_store)-1)]
    #dissip =[np.trace(C_store[t] @(C_store[t+1]-C_store[t]).T+(C_store[t+1]-C_store[t])@C_store[t].T)
    #               for t in range(len(C_store)-1)]
    #ratio=[(dissip[t]/injection[t]) for t in range(len(dissip))]
    #print("D: ",-np.mean(dissip),"I: ",np.mean(injection),"R: ",-np.mean(dissip)/np.mean(injection))
    #print(np.mean([np.trace(C_store[t] @
    #                        C_store[t])
    #               for t in range(len(C_store)-1)]))
    def compute_ratio(K):
        dK = K[1:] - K[:-1]

        injection = np.sum(dK*dK, axis=(1,2))
        dissip = 2*np.sum(K[:-1]*dK, axis=(1,2))

        energy_change = np.sum(K[1:]**2, axis=(1,2)) - np.sum(K[:-1]**2, axis=(1,2))
        return dK,injection,dissip,energy_change

    dk,injection,dissip,energy_change=compute_ratio(C_store)
    W = 200

    inj_local = np.convolve(injection, np.ones(W)/W, mode='valid')
    dis_local = np.convolve(dissip, np.ones(W)/W, mode='valid')
    energy_local = np.convolve(energy_change, np.ones(W)/W, mode='valid')
    ratio_local = dis_local + inj_local - energy_local

    print("D: ",-np.mean(dis_local),"I: ",np.mean(inj_local),"R: ",-np.mean(ratio_local),"Ratio: ",-np.mean(dis_local)/np.mean(inj_local))

    if BUILD_PLOTS:
        plt.figure()
        plt.plot(dis_local,label="Dissipação")
        plt.plot(inj_local,label="Injeção")
        plt.legend()


        plt.figure()
        plt.plot(ratio_local)
        plt.ylabel("Ratio")

        plt.figure()
        plt.plot(energy_local)
        plt.ylabel("Energy variation")
    #plt.figure()
    #plt.plot(drift)
    #plt.ylabel("Drift")
    def compute_injection(C_store, step):
        C_sub = C_store[::step]
        dK = C_sub[1:] - C_sub[:-1]
        inj = np.sum(dK*dK, axis=(1,2))
        return np.mean(inj)

    steps = [1,2,4,8,16]

    for s in steps:
        print(s, compute_injection(C_store, s))
    values = np.array([compute_injection(C_store, s) for s in steps])
    coeff = np.polyfit(np.log(steps), np.log(values), 1)
    print("Expoente:", coeff[0])

    if MULTI_STEP_TYPE == 'step':
        beta_vs_step.append((STEP_current, coeff[0]))
    elif MULTI_STEP_TYPE == 'embed':
        beta_vs_step.append((EMBED_current, coeff[0]))

    def calc_K(vecs_store,evals_store):
        K_=[]
        for t in range(len(vecs_store)):
            vecs_ = vecs_store[t]
            lambda_ = evals_store[t]
            if vecs_.size == 0:
                continue
            K_.append(vecs_ @ np.diag(lambda_) @ vecs_.T)
        K_=np.array(K_)
        return(K_)
    K_bulk=calc_K(bulk_vecs_store,bulk_evals_store)
    K_struct=calc_K(struct_vecs_store,struct_evals_store)

    dk,injection,dissip,energy_change=compute_ratio(K_bulk)
    inj_local = np.convolve(injection, np.ones(W)/W, mode='valid')
    dis_local = np.convolve(dissip, np.ones(W)/W, mode='valid')
    energy_local = np.convolve(energy_change, np.ones(W)/W, mode='valid')
    ratio_local = dis_local + inj_local - energy_local
    print("BULK: D: ",-np.mean(dis_local),"I: ",np.mean(inj_local),"R: ",-np.mean(ratio_local),"Ratio: ",-np.mean(dis_local)/np.mean(inj_local))

    dk,injection,dissip,energy_change=compute_ratio(K_struct)
    inj_local = np.convolve(injection, np.ones(W)/W, mode='valid')
    dis_local = np.convolve(dissip, np.ones(W)/W, mode='valid')
    energy_local = np.convolve(energy_change, np.ones(W)/W, mode='valid')
    ratio_local = dis_local + inj_local - energy_local
    print("STRUCT: D: ",-np.mean(dis_local),"I: ",np.mean(inj_local),"R: ",-np.mean(ratio_local),"Ratio: ",-np.mean(dis_local)/np.mean(inj_local))

    gap_min_t = np.array(gap_min_t)
    gap_min_t = gap_min_t[:-1]
    #gap_min_t = np.array(gap_min_t)
    #gap_min_t = gap_min_t[~np.isnan(gap_min_t)]
    m_mid = m_t[:-1]
    m_mid = np.array(m_mid)
    #m_mid = m_mid[~np.isnan(m_mid)]
    mask = (~np.isnan(dtheta))
    mask2 = (~np.isnan(dtheta)) & (~np.isnan(gap_min_t))
    corr = np.corrcoef(m_mid[mask], dtheta[mask])[0,1]
    dm = np.diff(m_t)
    corr2 = np.corrcoef(dm[mask], dtheta[mask])[0,1]
    corr_gap = np.corrcoef(gap_min_t[mask2], np.abs(dtheta[mask2]))[0,1]
    print("Correlaçoes <m,dtheta> <dm,dtheta> <gap_minimo,|dtheta|>",corr," ",corr2," ",corr_gap)

    # jump_idx = np.where(np.abs(dtheta) > mu + 2*sigma)[0]
    #
    # print("Número de candidatos a salto espectral:", len(jump_idx))
    #
    # spectral_jump_count = 0
    #
    # for j in jump_idx: #range(1, len(eigenvalues_series)):
    #     i=j+1
    #     vals_prev = eigenvalues_series[i-1]
    #     vals_now  = eigenvalues_series[i]
    #
    #     m_prev = m_t[i-1]
    #     m_now  = m_t[i]
    #
    #     lambda_plus_prev = lambda_plus_t[i-1]
    #     lambda_plus_now  = lambda_plus_t[i]
    #
    #     jump_flag = False
    #
    #     # 1) Mudança no número estrutural
    #     if m_prev != m_now:
    #         jump_flag = True
    #
    #     # 2) Crossing espectral
    #     # isso aqui verifica se os autovalores se cruzaram?
    #     rank_prev = np.argsort(vals_prev)[::-1]
    #     rank_now  = np.argsort(vals_now)[::-1]
    #     if not np.array_equal(rank_prev, rank_now):
    #         jump_flag = True
    #
    #     # 3) Mudança relevante no limite do bulk - acho que este teste aqui não procede. lambda_plus
    #     # vai depender da volatilidade e de q, q é fixo, então isso aqui está medindo se a volatilidade total mudou
    #     if abs(lambda_plus_now - lambda_plus_prev) > 0.1 * lambda_plus_prev:
    #         jump_flag = True
    #
    #     # 4) Distância espectral global
    #     # isso aqui está medindo a distancia espectral. Não entendi o or que do   0.5*np.mean
    #     # o correto seria medir se essas distâncias se alteraram muito
    #     spec_dist = np.linalg.norm(vals_now - vals_prev)
    #     if spec_dist > np.mean(vals_prev) * 0.5:
    #         jump_flag = True
    #
    #     if jump_flag:
    #         spectral_jump_count += 1
    # print("Quantos candidatos tiveram mudança estrutural:", spectral_jump_count)

    #teste 4 - normalidade do bulk
    from scipy.stats import jarque_bera, skew, kurtosis

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

    E_off_mean = np.sum(off_diag**2) / np.sum(C_bulk_mean**2)

    diag_energy = np.sum(np.diag(C_bulk_mean)**2)
    off_energy = np.sum(off_diag**2)

    R_mean = np.sqrt(off_energy) / np.sqrt(diag_energy)

    print("E_off (médio):", E_off_mean)
    print("R (médio):", R_mean)

    #teste 5 variância do ângulo
    max_lag = 20
    msd = []

    for lag in range(1, max_lag):
        diffs = angle_arr[lag:] - angle_arr[:-lag]
        msd.append(np.nanmean(diffs**2))

    if BUILD_PLOTS:
        plt.figure()
        plt.plot(range(1, max_lag), msd)
        plt.title("MSD da rotação espectral")
    #plt.show()

    # =========================
    # TESTE 6 - MP GLOBAL REESCALADO
    # =========================

    def mp_pdf(lam, var, q):
        lam_minus = var * (1 - np.sqrt(q))**2
        lam_plus  = var * (1 + np.sqrt(q))**2
        return np.where(
            (lam >= lam_minus) & (lam <= lam_plus),
            np.sqrt((lam_plus - lam) * (lam - lam_minus)) /
            (2 * np.pi * q * var * lam),
            0
        )

    all_bulk_rescaled = []
    all_q = []

    for t in range(len(eigenvalues_series)):

        vals = eigenvalues_series[t]
        m = m_t[t]

        if m >= len(vals):
            continue

        vals_bulk = vals[m:]

        if len(vals_bulk) < 3:
            continue

        Xreg = Xreg_store[t]

        N_dim = Xreg.shape[1]
        T_obs = Xreg.shape[0]

        q_emp = N_dim / T_obs

        var_data = np.mean(np.var(Xreg, axis=0))

        # reescala os autovalores
        vals_rescaled = vals_bulk / var_data

        all_bulk_rescaled.extend(vals_rescaled)
        all_q.append(q_emp)

    all_bulk_rescaled = np.array(all_bulk_rescaled)

    # usar q médio para MP agregado
    q_mean = np.mean(all_q)

    xs = np.linspace(np.min(all_bulk_rescaled),
                     np.max(all_bulk_rescaled), 400)
    if BUILD_PLOTS:
        plt.figure()
        plt.hist(all_bulk_rescaled, bins=50,
                 density=True, alpha=0.5)

    lambda_minus_theory = (1 - np.sqrt(q_mean))**2
    lambda_minus_emp = np.min(all_bulk_rescaled)

    print(lambda_minus_emp, lambda_minus_theory)
    lambda_plus_theory = (1 + np.sqrt(q_mean))**2
    lambda_plus_emp = np.max(all_bulk_rescaled)

    print(lambda_plus_emp, lambda_plus_theory)
    print("Diferença percentual: ",100*np.abs(lambda_plus_emp- lambda_plus_theory)/lambda_plus_theory,"%")

    if BUILD_PLOTS:
        plt.plot(xs, mp_pdf(xs, 1.0, q_mean))
        plt.title("MP Global - Bulk Reescalado")

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

    print("Erro L2 relativo = ", 100*L2_relative,"%")
    if MULTI_STEP_TYPE == 'step':
        mpl2error_vs_step.append((STEP_current, 100*L2_relative))
    elif MULTI_STEP_TYPE == 'embed':
        mpl2error_vs_step.append((EMBED_current, 100*L2_relative))
    # ==================================
    # TESTE 7 - ESTACIONARIEDADE
    # ==================================

    from statsmodels.tsa.stattools import adfuller

    theta_clean = angle_arr[~np.isnan(angle_arr)]
    if len(theta_clean)>20:
        adf_result = adfuller(theta_clean)
        print("ADF stat:", adf_result[0])
        print("p-value:", adf_result[1])
    else: print("Série muito curta ara ADF")

    # ==================================
    # TESTE 8 - ESCALING DO MSD
    # ================== ================
    dK=np.diff(theta_clean)
    print("Var(Δtheta):", np.var(dK))
    theta_mid = theta_clean[:-1]

    corr = np.corrcoef(theta_mid, dK)[0,1]
    print("Corr( theta , Δtheta ):", corr)

    # ==================================
    # TESTE 9 - DRIFT F(K)
    # ==================================

    dtheta = np.diff(theta_clean)
    theta_mid = theta_clean[:-1]

    bins = np.linspace(min(theta_mid),max(theta_mid),20)
    digitized = np.digitize(theta_mid,bins)

    drift = []
    centers = []

    for i in range(1,len(bins)):
        mask = digitized==i
        if np.sum(mask)>10:
            drift.append(np.mean(dtheta[mask]))
            centers.append(np.mean(theta_mid[mask]))

    if BUILD_PLOTS:
        plt.figure()
        plt.plot(centers,drift)
        plt.title("Estimativa de F(theta)")

    # ==================================
    # TESTE 10 - NORMALIDADE DOS INCREMENTOS
    # ==================================

    from scipy.stats import jarque_bera

    jb = jarque_bera(dtheta)
    print("JB stat:",jb.statistic)
    print("p-value:",jb.pvalue)

    # ==================================
    # TESTE 11 - Entropia
    # ==================================
    #
    S_norm_med = np.mean(entropy_series)
    print("Entropia média: ",S_norm_med)
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
        gs = gridspec.GridSpec(4, 2) # Aumentado para 4 linhas

        # -----------------------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(theta_struct_t,-np.log(gap_t))
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
        for idx in sorted(set(indices_m_change) & set(indices_compressao)):
            ax4.axvline(tau_axis[idx], linestyle='--', alpha=0.5)
        ax4.set_title("Número de modos estruturais m(τ)")
        # -----------------------
        # ax5: Autovalores
        ax5 = fig.add_subplot(gs[2, 0])
        for i in range(min(max_tracks, tracked_modes.shape[1])):
            ax5.plot(tau_axis, tracked_modes[:, i], label=f"λ{i+1}")
        for idx in indices_m_change:
            ax5.axvline(tau_axis[idx] , linestyle='--', alpha=0.5)
        ax5.set_title("Autovalores estruturais")
        ax5.legend()

        # -----------------------
        ax6 = fig.add_subplot(gs[2, 1])
        for i in range(min(6, vecs_sel.shape[1])):
        #    plot_eigenvector_candles(vecs_sel[:, i], ax6, f"Modo {i+1}")
            plot_mode(vecs_sel[:, i], ax6, f"Modo {i+1}")
        ax6.legend()

        window_err = 2000
        rel_err = (err_null_series - err_struct_series)

        #rolling = pd.Series(rel_err).rolling(window_err).mean()
        rolling = (pd.Series(err_null_series).rolling(window_err).mean()-pd.Series(err_struct_series).rolling(window_err).mean())/pd.Series(err_null_series).rolling(window_err).mean()
        cumsum = (np.cumsum(err_null_series)-np.cumsum(err_struct_series))/np.cumsum(err_null_series)
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(err_time_series,rolling,label=f"{100*cumsum[len(cumsum)-1]}%")
        #ax7.plot(err_time_series,100*cumsum, label=f"{100*cumsum[len(cumsum)-1]}%")
        ax7.set_title('Diferença percentual dos erros absolutos acumulados entre o modelo nulo e a projeção')
        ax7.legend()

        ax8 = fig.add_subplot(gs[3, 1])
        for i in range(1):  # plota 5 janelas exemplo
            t0, p_real, p_rec, p_mean = window_recons[i]
            # if MODE=='vector':
            #     def plot_candles(seq, ax, label, alpha=1.0):
            #         width = 0.3
            #         for k, (O_, H_, L_, C_) in enumerate(seq):
            #             ax.plot([k, k], [L_, H_], alpha=alpha)
            #             lower = min(O_, C_)
            #             height = abs(C_ - O_) if abs(C_ - O_) > 1e-8 else 1e-8
            #             rect = plt.Rectangle((k - width / 2, lower), width, height,
            #                                  fill=False, alpha=alpha)
            #             ax.add_patch(rect)
            #
            #
            #     plot_candles(p_real, ax8, "Real", alpha=0.4)
            #     plot_candles(p_rec, ax8, "Reconstruído", alpha=1.0)
            # else:
            p_real = np.array(p_real)
            p_rec = np.array(p_rec)
            p_mean = np.array(p_mean)
            if MODE=='vector':
                ax8.plot(p_real[:,3], alpha=0.4,label="Real")
                ax8.plot(p_rec[:,3], linewidth=1,label="Reconstruído")
                ax8.plot(p_mean[:,3], linewidth=1, label="Modelo nulo")
            else:
                ax8.plot(p_real, alpha=0.4, label="Real")
                ax8.plot(p_rec, linewidth=1, label="Reconstruído")
                ax8.plot(p_mean, linewidth=1, label="Modelo nulo")
            ax8.legend()
            ax8.set_title("Reconstrução estrutural do preço dentro das janelas")


        plt.tight_layout()

        out_name = os.path.splitext(CSV_FILE)[0] +"_4D_"+MODE+"_ "+TYPE+".png"
        plt.savefig(out_name, dpi=300)

        plt.show()

if MULTI_STEP_ANALYSIS:
    steps, alphas = zip(*alpha_vs_step)
    _, betas = zip(*beta_vs_step)
    _, entropy = zip(*entropy_vs_step)
    _, mpl2error = zip(*mpl2error_vs_step)
    plt.figure()
    plt.plot(steps, alphas)
    plt.xlabel(MULTI_STEP_TYPE)
    plt.ylabel("Alpha")

    plt.figure()
    plt.plot(steps, betas)
    plt.xlabel(MULTI_STEP_TYPE)
    plt.ylabel("Beta (dK^2 vs dt)")

    plt.figure()
    plt.plot(steps, entropy)
    plt.xlabel(MULTI_STEP_TYPE)
    plt.ylabel("Entropy")

    plt.figure()
    plt.plot(steps, mpl2error)
    plt.xlabel(MULTI_STEP_TYPE)
    plt.ylabel("MP L2 error (%)")

    plt.show()

print("Concluído.")
