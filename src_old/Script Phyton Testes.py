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
CSV_FILE = "PETR4_D1.csv"

WINDOW = 200           # tamanho da janela individual
N_WINDOWS = 10        # quantas janelas vizinhas compõem o mesmo regime
STEP = 20             # passo do τ
EMBED_DIM = 20        # embedding temporal
#N_MODES = 6

gap_t = []
resid_t = []
angle_t = []
m_t = []
evecs_store = {}
tracked_modes = []
prev_modes = None
max_tracks = 8


# =========================
# LEITURA
# =========================
df = pd.read_csv(CSV_FILE, sep=';', encoding='utf-16')
prices = df['close'].values
times = pd.to_datetime(df['time']).values

returns = np.diff(np.log(prices))
times = times[1:]

# =========================
# FUNÇÕES
# =========================

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


def estimate_frequency(v):
    """
    Estima a frequência do autovetor contando cruzamentos por zero.
    Como são senoides de Sturm-Liouville, o número de nós define a frequência.
    """
    # Conta quantas vezes o sinal muda de sinal
    zero_crossings = np.where(np.diff(np.sign(v)))[0]
    num_nodes = len(zero_crossings)
    # Frequência normalizada (número de ciclos aproximado)
    return num_nodes / 2.0

def hankel_embed(x, m):
    N = len(x) - m
    return np.column_stack([x[i:N+i] for i in range(m)])

def pca_cov(X):
    C = np.cov(X, rowvar=False)
    vals, vecs = eigh(C)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]

def mp_lambda_plus(var, q):
    return var * (1 + np.sqrt(q))**2

def hankel_diagonal_averaging(X):
    n, m = X.shape
    y = np.zeros(n + m - 1)
    counts = np.zeros(n + m - 1)

    for i in range(n):
        for j in range(m):
            y[i + j] += X[i, j]
            counts[i + j] += 1

    return y / counts

def diagonal_averaging(H):
    L, K = H.shape
    N = L + K - 1
    y = np.zeros(N)
    counts = np.zeros(N)

    for i in range(L):
        for j in range(K):
            y[i+j] += H[i, j]
            counts[i+j] += 1

    return y / counts

# =========================
# CONSTRÓI A SUPERMATRIZ DE UM REGIME
# =========================
def build_regime_matrix_PCA(center):
    blocks = []
    half = N_WINDOWS // 2

    for k in range(-half, half):
        start = center + k * (WINDOW // 4)
        end = start + WINDOW

        if start < 0 or end >= len(returns):
            continue

        seg = returns[start:end]
        X = hankel_embed(seg, EMBED_DIM)
        blocks.append(X)

    if len(blocks) < 3:
        return None

    return np.vstack(blocks)

def build_regime_matrix_SSA(center):
    start = center - WINDOW//2
    end = start + WINDOW

    if start < 0 or end >= len(returns):
        return None

    seg = returns[start:end]
    X = hankel_embed(seg, EMBED_DIM)
    return X


# =========================
# LOOP PRINCIPAL EM τ
# =========================
tau_axis = []
gap_t = []
angle_t = []
var_total_t = []
var_struct_t = []
var_noise_t = []
prev_vecs = None
alpha_t = []
recon_log_ret_t = []

print("Calculando")
for center in range(WINDOW, len(returns) - WINDOW, STEP):
    #print("Construindo a matriz do regime")

    Xreg = build_regime_matrix_PCA(center)
    if Xreg is None:
        continue

    #print("Calculando PCA")
    vals, vecs = pca_cov(Xreg)

    # MP threshold
    q = EMBED_DIM / Xreg.shape[0]
    var = np.var(Xreg)
    lam_plus = mp_lambda_plus(var, q)

    structural = vals > lam_plus
#    m = min(np.sum(structural), N_MODES)
    m = np.sum(structural)

    vals_s = vals[:m]
    vecs_s = vecs[:, :m]

    m_t.append(m)
    # ... (dentro do loop principal)
    # RECONSTRUÇÃO ESTRUTURAL
    # Xreg tem a forma (N_samples, EMBED_DIM)
    # vecs_s tem a forma (EMBED_DIM, m)
    #if m > 0:
        # Projeta no subespaço estrutural e reconstrói (Filtragem PCA)
        #X_proj = (Xreg @ vecs_s) @ vecs_s.T

        # Para voltar ao escalar, pegamos a média das diagonais de Hankel
        # ou simplificamos pegando a última coluna (valor mais recente)
#        recon_returns = X_proj[:, -1]

        # Pegamos o último valor reconstruído para compor a série temporal
        # e integramos para voltar ao preço
#        last_recon_log_ret = recon_returns[-1]
        #recon_series = hankel_diagonal_averaging(X_proj)
        #last_recon_log_ret = recon_series[-1]
        #H_struct = eigvecs_struct @ np.diag(eigvals_struct) @ eigvecs_struct.T
        #recon_series = diagonal_averaging(H_struct)
        #last_recon_log_ret = recon_series[-1]

    #else:
        #last_recon_log_ret = 0.0  # Sem modos, retorno estrutural é nulo

    #recon_log_ret_t.append(last_recon_log_ret)
    # janela central REAL
    #seg_central = returns[center:center + WINDOW]
    #Xc = hankel_embed(seg_central, EMBED_DIM)

    # projeção estrutural da janela central
    #if m > 0:
    #    Xc_proj = (Xc @ vecs_s) @ vecs_s.T
    #    recon_log_ret = Xc_proj[-1, -1]
    #else:
    #    recon_log_ret = 0.0

    #recon_log_ret_t.append(EMBED_DIM*STEP*recon_log_ret)
#    if m > 0:
#        Xc = hankel_embed(returns[center - WINDOW:center], EMBED_DIM)

#        phi = vecs_s[:, :m]  # (EMBED_DIM, m)

        # pega a última linha da Hankel (estado atual do embedding)
#        x_now = Xc[-1, :]  # (EMBED_DIM,)

        # coeficientes temporais a_k(tau)
#        a = x_now @ phi  # (m,)

        # reconstrução do retorno estrutural
#        recon_log_ret = a @ phi[-1, :]  # escalar
#    else:
#        recon_log_ret = 0.0

#    recon_log_ret_t.append(recon_log_ret)
    # ================= RECONSTRUÇÃO DENTRO DA JANELA =================

    #print("Reconstruindo os preços")
    # seg = returns[center - WINDOW:center]
    # price_seg = prices[center - WINDOW:center + 1]
    #
    # Xc = hankel_embed(seg, EMBED_DIM)
    #
    # if m > 0:
    #     Phi = vecs_s[:, :m]
    #     Xc_proj = (Xc @ Phi) @ Phi.T
    # else:
    #     Xc_proj = np.zeros_like(Xc)
    #
    #
    # # ---- diagonal averaging (Hankel -> série temporal) ----
    # def diagonal_averaging(H):
    #     L, K = H.shape
    #     N = L + K - 1
    #     y = np.zeros(N)
    #     counts = np.zeros(N)
    #     for i in range(L):
    #         for j in range(K):
    #             y[i + j] += H[i, j]
    #             counts[i + j] += 1
    #     return y / counts
    #
    #
    # recon_returns = diagonal_averaging(Xc_proj)
    #
    # # ---- integra preço dentro da janela ----
    # log_start = np.log(price_seg[0])
    # #price_recon = np.exp(log_start + np.cumsum(recon_returns))
    # #mu = np.mean(seg)  # drift médio da janela
    # def moving_mean(x, w):
    #     return np.convolve(x, np.ones(w) / w, mode='same')
    #
    #
    # mu_t = moving_mean(seg, w=EMBED_DIM)  # ou EMBED_DIM//2
    # recon_returns_with_drift = recon_returns + mu_t[:len(recon_returns)]
    #
    # #recon_returns_with_drift = recon_returns + mu
    # price_recon = np.exp(log_start + np.cumsum(recon_returns_with_drift))
    seg = returns[center - WINDOW:center]
    price_seg = prices[center - WINDOW:center + 1]

    Phi = vecs_s[:, :m]

    r_pred = []
    mu_series = []

    for i in range(EMBED_DIM, len(seg)):
        # embedding causal (só passado)
        x_t = seg[i - EMBED_DIM:i]

        if m > 0:
            a = x_t @ Phi  # coeficientes modais
            r_struct = a @ Phi[-1, :]  # projeção futura
            #x_rec = Phi @ a
            #r_struct = x_rec[1]
        else:
            r_struct = 0.0

        # média móvel causal
        # def moving_mean(x, w):
        #     return np.convolve(x, np.ones(w) / w, mode='same')
        #
        #
        # mu_t = moving_mean(seg, w=EMBED_DIM)  # ou EMBED_DIM//2
        #mu_t = np.mean(seg[i - EMBED_DIM:i])

        #r_hat = mu_t + r_struct
        #r_hat = seg[i - 1:i] + r_struct
        r_hat = r_struct

        r_pred.append(price_seg[i - 1:i]*np.exp(r_hat))
        #mu_series.append(mu_t)
        mu_series.append(price_seg[i - 1:i])

    log_start = np.log(price_seg[EMBED_DIM])
#    price_recon = np.exp(log_start + np.cumsum(r_pred))
#    price_mu = np.exp(log_start + np.cumsum(mu_series))
#    price_real = price_seg[EMBED_DIM:EMBED_DIM + len(price_recon)]
    price_real = price_seg[EMBED_DIM:EMBED_DIM + len(r_pred)]

#    price_recon = []
#    prev_price = price_seg[EMBED_DIM - 1]  # preço real anterior

#    for r in r_pred:
#        next_price = prev_price * np.exp(r)
#        price_recon.append(next_price)
#        prev_price = next_price

#    price_recon = np.array(price_recon)
    price_mu = mu_series#np.exp(log_start + np.cumsum(mu_series))  # só para não quebrar o gráfico

    # guarde para plotar depois
    if 'window_recons' not in globals():
        window_recons = []

    window_recons.append((times[center], price_real, r_pred,price_mu))

    # ... (dentro do loop, após m = np.sum(structural))
    #print("Calculando expoente")

    omega_lambda_relation = []  # Armazena (omega, lambda) para este tau
    if m > 0:
        for i in range(min(m, 5)):  # Analisamos os primeiros 5 modos
#            w = estimate_frequency_fourier(vecs_s[:, i])
            mode_time_series = Xreg @ vecs_s[:, i]
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

#    if len(current) == 0:
#        tracked_modes.append(row)
#        prev_modes = None
#    else:
#        if prev_modes is None:
#            for i in range(min(len(current), max_tracks)):
#                row[i] = current[i]
#            prev_modes = current.copy()
#        else:
#            used = set()
#            for i, pv in enumerate(prev_modes):
#                if i >= max_tracks:
#                    break
#                dists = [abs(pv - cv) for cv in current]
#                j = int(np.argmin(dists))
#                row[i] = current[j]
#                used.add(j)

            # modos novos
#            k = 0
#            for j, cv in enumerate(current):
#                if j not in used and k < max_tracks:
#                    row[k] = cv
#                    k += 1

#            prev_modes = [x for x in row if not np.isnan(x)]

#        tracked_modes.append(row)

    # Salva autovetores em alguns τ para inspeção
    if m > 0 and len(evecs_store) < 5:
        evecs_store[times[center]] = vecs_s.copy()

    # GAP espectral
    if m > 1:
        gaps = [abs(vals_s[i] - vals_s[j])
                for i in range(m) for j in range(i+1, m)]
        gap_t.append(min(gaps))
    else:
        gap_t.append(np.nan)

    # Resíduo
#    resid_t.append(np.sum(vals[m:]))
    # Energia total
    var_total = np.sum(vals)

    # Energia estrutural
    var_struct = np.sum(vals[:m]) if m > 0 else 0.0

    # Energia do ruído (bulk)
    var_noise = np.sum(vals[m:])

    var_total_t.append(var_total)
    var_struct_t.append(var_struct/var_total)
    var_noise_t.append(var_noise/var_total)

    #print("Calculando rotação")
    # Rotação correta do subespaço estrutural comum
    if prev_vecs is not None and prev_m is not None and m > 1 and prev_m > 1:
        try:
            m_star = min(prev_m, m)

            A = prev_vecs[:, :m_star]
            B = vecs_s[:, :m_star]

            ang = subspace_angles(A, B)
            angle_t.append(np.mean(ang))
        except:
            angle_t.append(np.nan)
    else:
        angle_t.append(np.nan)

    prev_vecs = vecs_s
    prev_m = m

    tau_axis.append(times[center])

tracked_modes = np.array(tracked_modes, dtype=float)
# Integração dos retornos estruturais para obter o preço
# S_recon = S_0 * exp(cumsum(log_ret_estrutural))
#log_price_start = np.log(prices[WINDOW])
#s_recon = np.exp(log_price_start + np.cumsum(recon_log_ret_t))
#plt.figure(figsize=(12,8))


#plt.show()

# =========================
# GRÁFICOS
# =========================

# plt.figure(figsize=(14,10))
#
# plt.subplot(3,1,1)
# plt.plot(tau_axis, gap_t)
# plt.title("Gap espectral mínimo Δ(τ)")
#
# plt.subplot(3,1,2)
# plt.plot(tau_axis, resid_t)
# plt.title("Variância do resíduo σ²η(τ)")
#
# plt.subplot(3,1,3)
# plt.plot(tau_axis, angle_t)
# plt.title("Rotação média dos autovetores θ(τ)")
#
# plt.figure(figsize=(12,4))
# plt.plot(tau_axis, m_t)
# plt.title("Número de modos estruturais m(τ)")
# plt.ylabel("m")
# #plt.show()
#
# # =========================
# # PRIMITIVAS DO KERNEL
# # =========================
# for t_key, vecs in evecs_store.items():
#     plt.figure(figsize=(12,6))
#     n_plot = min(6, vecs.shape[1])
#     for i in range(n_plot):
#         plt.plot(vecs[:, i], label=f"Modo {i+1}")
#     plt.title(f"Primitivas do kernel em τ = {t_key}")
#     plt.legend()
# #    plt.show()
#
# plt.figure(figsize=(14,6))
# for i in range(max_tracks):
#     plt.plot(tau_axis, tracked_modes[:, i], label=f"λ{i+1}")
#
# plt.title("Evolução temporal dos autovalores estruturais (rastreados)")
# plt.xlabel("Tempo")
# plt.ylabel("λ estrutural")
# plt.legend()
# #plt.show()
#
#
# plt.tight_layout()
# plt.show()

# escolhe o τ com maior número de modos
idx_tau = np.nanargmax(m_t)
tau_sel = tau_axis[idx_tau]
# lista ordenada dos taus realmente calculados
taus_validos = np.array(sorted(evecs_store.keys()))

# encontra o mais próximo do tau desejado
idx = np.argmin(np.abs(taus_validos - tau_sel))
tau_real = taus_validos[idx]

vecs_sel = evecs_store[tau_real]
#print("Usando tau_real =", tau_real)
print("Gerando gráficos")

#fig = plt.figure(figsize=(18, 14))
#gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(18, 18))
gs = gridspec.GridSpec(4, 2) # Aumentado para 4 linhas

# -----------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tau_axis, gap_t)
ax1.set_title("Gap espectral mínimo Δ(τ)")

# -----------------------
ax2 = fig.add_subplot(gs[0, 1])
#ax2.plot(tau_axis, resid_t)
#ax2.set_title("Variância do resíduo σ²η(τ)")

#ax2.plot(tau_axis, var_total_t, label='Variância total')
#ax2.plot(tau_axis, var_struct_t, label='Variância estrutural')
#ax2.plot(tau_axis, var_noise_t, label='Variância do ruído (bulk)')
ax2.plot(tau_axis, var_noise_t)
#ax2.legend()
ax2.set_title("Variância relativa do resíduo σ²η(τ)")
#ax2.set_title('Decomposição espectral da variância')

# -----------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(tau_axis, angle_t)
ax3.set_title("Rotação média dos autovetores θ(τ)")

# -----------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(tau_axis, m_t)
ax4.set_title("Número de modos estruturais m(τ)")

# -----------------------
#ax5 = fig.add_subplot(gs[2, 0])
#for i in range(max_tracks):
#    ax5.plot(tau_axis, tracked_modes[:, i], label=f"λ{i+1}")
#ax5.set_title("Autovalores estruturais rastreados")
#ax5.legend()

# ax5: Autovalores
ax5 = fig.add_subplot(gs[2, 0])
for i in range(min(max_tracks, tracked_modes.shape[1])):
    ax5.plot(tau_axis, tracked_modes[:, i], label=f"λ{i+1}")
ax5.set_title("Autovalores estruturais")
ax5.legend()

# -----------------------
ax6 = fig.add_subplot(gs[2, 1])
n_plot = min(6, vecs_sel.shape[1])
for i in range(n_plot):
    ax6.plot(vecs_sel[:, i], label=f"Modo {i+1}")
ax6.set_title(f"Primitivas do kernel em τ = {tau_sel}")
ax6.legend()

# ax7: Relação de Helmholtz (Alpha)
ax7 = fig.add_subplot(gs[3, 0])
ax7.plot(tau_axis, alpha_t, color='brown')
ax7.axhline(y=-2, color='r', linestyle='--')
ax7.set_title(r"Expoente de Dispersão $\alpha$ ($\lambda \propto \omega^\alpha$)")

# ax8: Preço Reconstruído vs Original
#ax8 = fig.add_subplot(gs[3, 1]) # Ocupa a linha inteira
# Ajustar o eixo temporal para o preço original condizente com tau_axis
#prices_segment = prices[WINDOW:WINDOW + len(s_recon)*STEP:STEP]
# Nota: verifique se o slice acima bate com o tamanho de s_recon

#ax8.plot(tau_axis, prices_segment[:len(s_recon)], label='Original', alpha=0.5)
#ax8.plot(tau_axis, s_recon, label='Estrutural', linewidth=2)
#ax8.set_title("Reconstrução do Preço via Modos Estruturais")
#ax8.legend()

ax8 = fig.add_subplot(gs[3, 1])
for i in range(1):  # plota 5 janelas exemplo
    t0, p_real, p_rec, p_mean = window_recons[i]
    ax8.plot(p_real, alpha=0.4,label="Real")
    ax8.plot(p_rec, linewidth=1,label="Reconstruído")
    ax8.plot(p_mean, linewidth=1, label="Modelo nulo")
    ax8.legend()
    ax8.set_title("Reconstrução estrutural do preço dentro das janelas")


plt.tight_layout()

out_name = os.path.splitext(CSV_FILE)[0] + ".png"
plt.savefig(out_name, dpi=300)
plt.show()

print("Concluído.")