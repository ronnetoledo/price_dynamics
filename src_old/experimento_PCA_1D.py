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
CSV_FILE = "PETR4_M5.csv"

WINDOW = 400           # tamanho da janela individual
N_WINDOWS =40        # quantas janelas vizinhas compõem o mesmo regime
STEP = 40             # passo do τ
EMBED_DIM = 10        # embedding temporal
WINDOW_OVERLAP = 10     #passo no empillhamento de janelas é uma eseetimativa do tempo médio de  variação
PROJECTION_HORIZON = 200 # horizonte da rojeção

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

times = pd.to_datetime(df['time']).values
times = times[1:]

mode ='close'
prices = df[mode].values
#Teste com o preço médio
#mode ='avg'
#prices = df['real_volume'].values/df['tick_volume'].values
returns = np.diff(np.log(prices))

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
    return vals[idx], vecs[:, idx]

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

# =========================
# CONSTRÓI A SUPERMATRIZ DE UM REGIME
# =========================

def build_causal_regime_matrix(center):
    blocks = []

    for k in range(N_WINDOWS):
        end = center - k * WINDOW_OVERLAP
        start = end - WINDOW

        if start < 0:
            continue

        seg = returns[start:end]
        X = hankel_embed(seg, EMBED_DIM)

        blocks.append(X)

    return np.vstack(blocks)


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
err_struct_map = {}
err_null_map = {}
err_time_map = {}
print("Calculando")
for center in range(WINDOW, len(returns) - WINDOW, STEP):
    #print("Construindo a matriz do regime")

    Xreg = build_causal_regime_matrix(center)
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


    Phi = vecs_s[:, :m]

    price_pred = []
    price_null = []
    price_real = []
    time_series = []

    for i in range(0,PROJECTION_HORIZON):  # WINDOW

        x_t = returns[center + i - EMBED_DIM: center + i]

        if m > 0:
            a = x_t @ Phi  # coeficientes modais
            r_struct = a @ Phi[-1, :]  # projeção futura
        else:
            r_struct = 0.0

        S_prev = prices[center + i - 1]
        S_pred = S_prev * np.exp(r_struct)
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
        time_series.append(t_candle)
        price_real.append(S_real)

    # guarde para plotar depois
    if 'window_recons' not in globals():
        window_recons = []

    window_recons.append((times[center], price_real, price_pred,price_null))

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
    # Energia total
    var_total = np.sum(vals)

    # Energia estrutural
    var_struct = np.sum(vals[:m]) if m > 0 else 0.0

    # Energia do ruído (bulk)
    var_noise = np.sum(vals[m:])

    var_total_t.append(var_total)
    var_struct_t.append(var_struct/var_total)
    var_noise_t.append(var_noise/var_total)


    prev_vecs = vecs_s
    prev_m = m

    tau_axis.append(times[center])

tracked_modes = np.array(tracked_modes, dtype=float)
times_sorted = sorted(err_struct_map.keys())
err_struct_series = np.array([err_struct_map[t] for t in times_sorted])
err_null_series   = np.array([err_null_map[t] for t in times_sorted])
err_time_series   = np.array([err_time_map[t] for t in times_sorted])

# =========================
# GRÁFICOS
# =========================

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
ax1.plot(tau_axis, gap_t)
ax1.set_title("Gap espectral mínimo Δ(τ)")

# -----------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(tau_axis, var_noise_t)
ax2.set_title("Variância relativa do resíduo σ²η(τ)")

# -----------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(tau_axis, angle_t)
ax3.set_title("Rotação média dos autovetores θ(τ)")

# -----------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(tau_axis, m_t)
ax4.set_title("Número de modos estruturais m(τ)")

# -----------------------
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

ax7 = fig.add_subplot(gs[3, 0])
ax7.plot(err_time_series,100*(np.cumsum(err_null_series)-np.cumsum(err_struct_series))/np.cumsum(err_null_series), alpha=0.7)
ax7.set_title('Diferença percentual dos erros absolutos acumulados entre o modelo nulo e a projeção')

ax8 = fig.add_subplot(gs[3, 1])
for i in range(1):  # plota 5 janelas exemplo
    t0, p_real, p_rec, p_mean = window_recons[i]
    ax8.plot(p_real, alpha=0.4,label="Real")
    ax8.plot(p_rec, linewidth=1,label="Reconstruído")
    ax8.plot(p_mean, linewidth=1, label="Modelo nulo")
    ax8.legend()
    ax8.set_title("Reconstrução estrutural do preço dentro das janelas")


plt.tight_layout()

out_name = os.path.splitext(CSV_FILE)[0] +"_"+mode+ ".png"
plt.savefig(out_name, dpi=300)

plt.show()

print("Concluído.")
