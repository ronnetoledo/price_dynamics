import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Geração dos dados: ruído branco gaussiano
# ============================================================

np.random.seed(0)

N_samples = 100_000
x = np.random.randn(N_samples)

# ============================================================
# 2. Função custo de Shimazaki–Shinomoto
# ============================================================

def shimazaki_shinomoto_cost(data, nbins):
    """
    Calcula a função custo de Shimazaki–Shinomoto
    para um dado número de bins.
    """
    x_min, x_max = data.min(), data.max()
    D = (x_max - x_min) / nbins  # largura do bin

    edges = np.linspace(x_min, x_max, nbins + 1)
    counts, _ = np.histogram(data, bins=edges)

    k_bar = np.mean(counts)
    v = np.sum((counts - k_bar)**2) / nbins  # variância enviesada

    cost = (2 * k_bar - v) / (D**2)
    return cost


# ============================================================
# 3. Varredura no número de bins + busca do ótimo
# ============================================================

bin_range = np.arange(10, 300)          #aqui definimos o numero desde um numero de bins pequenos (10) ate um numero consideravel (300), o que determina tamanhos de box grandes ate bem pequenos.

costs = np.array([
    shimazaki_shinomoto_cost(x, nbins)
    for nbins in bin_range
])

idx_opt = np.argmin(costs)
optimal_bins = bin_range[idx_opt]
optimal_cost = costs[idx_opt]

# largura ótima associada
x_min, x_max = x.min(), x.max()
optimal_width = (x_max - x_min) / optimal_bins


# ============================================================
# 4. Impressão dos resultados
# ============================================================

print("===== RESULTADO DO ESTUDO GAUSSIANO =====")
print(f"Número de amostras          : {N_samples}")
print(f"Número de bins ótimo        : {optimal_bins}")
print(f"Largura ótima do bin (Δ)    : {optimal_width:.5f}")
print(f"Valor mínimo da função custo: {optimal_cost:.5e}")


# ============================================================
# 5. Gráfico da função custo vs número de bins
# ============================================================

plt.figure(figsize=(7, 4))
plt.plot(bin_range, costs, lw=2, label="Função custo")
plt.axvline(optimal_bins, color="red", ls="--", label="Ótimo")
plt.xlabel("Número de bins")
plt.ylabel("Função custo")
plt.title("Função custo × número de bins (ruído gaussiano)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('funcao_custo.png')
plt.show()


# ============================================================
# 6. Boxplot + histograma com bins ótimos
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Boxplot
axes[0].boxplot(x, vert=True)
axes[0].set_title("Boxplot dos dados")
axes[0].set_ylabel("Amplitude")

# Histograma com bins ótimos
axes[1].hist(x, bins=optimal_bins, density=True)
axes[1].set_title(f"Histograma (bins ótimos = {optimal_bins})")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Densidade")
plt.tight_layout()
plt.savefig('box_plot_e_bin_otimizado.png')
plt.show()
