
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import glob
import random
import matplotlib.pyplot as plt

# -----------------------------
# Utilidades
# -----------------------------

def sliding_windows_batch(series, window, batch_size=50000):
    n = len(series) - window
    for start in range(0, n, batch_size):
        X = []
        end = min(start + batch_size, n)
        for i in range(start, end):
            w = series[i:i+window]
            std = np.std(w)
            if std > 0:
                w = (w - np.mean(w)) / std
                X.append(w)
        if X:
            yield np.array(X)


def discrete_laplacian(phi):
    d2 = np.zeros_like(phi)
    d2[1:-1] = phi[2:] - 2*phi[1:-1] + phi[:-2]
    return d2


def marchenko_pastur_bounds(q, sigma2=1.0):
    lambda_min = sigma2 * (1 - np.sqrt(1/q))**2
    lambda_max = sigma2 * (1 + np.sqrt(1/q))**2
    return lambda_min, lambda_max


# -----------------------------
# Classe principal
# -----------------------------

class FullExperiment:

    def _init_(self, window_size=60, max_components=20):
        self.window_size = window_size
        self.max_components = max_components

    def load_returns(self, file_path, shuffle=False):
        df = pd.read_csv(file_path, sep=';')
        r = np.log(df['fechamento'] / df['fechamento'].shift(1)).dropna().values
        if shuffle:
            np.random.shuffle(r)
        return r

    # -----------------------------
    # Treino incremental PCA
    # -----------------------------
    def fit_pca(self, files):
        self.ipca = IncrementalPCA(n_components=self.max_components)
        total_samples = 0

        for f in files:
            r = self.load_returns(f)
            for batch in sliding_windows_batch(r, self.window_size):
                self.ipca.partial_fit(batch)
                total_samples += len(batch)

        self.total_samples = total_samples

    # -----------------------------
    # RMT filtro
    # -----------------------------
    def apply_rmt(self):
        lambdas = self.ipca.explained_variance_
        q = self.total_samples / self.window_size
        _, lmax = marchenko_pastur_bounds(q)
        self.significant = np.where(lambdas > lmax)[0]

    # -----------------------------
    # Erro reconstrução
    # -----------------------------
    def reconstruction_error(self, files, shuffle=False):
        err = []
        for f in files:
            r = self.load_returns(f, shuffle)
            for batch in sliding_windows_batch(r, self.window_size):
                proj = self.ipca.transform(batch)
                recon = self.ipca.inverse_transform(proj)
                err.append(np.mean((batch - recon) ** 2))
        return np.mean(err)

    # -----------------------------
    # Potencial
    # -----------------------------
    def compute_potentials(self):
        potentials = []
        for idx in self.significant:
            phi = self.ipca.components_[idx]
            d2_phi = discrete_laplacian(phi)
            energy = self.ipca.explained_variance_[idx]
            V = d2_phi / (phi + 1e-8) + energy
            potentials.append(np.convolve(V, np.ones(3)/3, mode='same'))
        return potentials

    # -----------------------------
    # Dinâmica K²(t)
    # -----------------------------
    def analyze_dynamics(self, files):
        modes = []
        for f in files:
            r = self.load_returns(f)
            for batch in sliding_windows_batch(r, self.window_size):
                proj = self.ipca.transform(batch)
                dom = np.argmax(np.abs(proj[:, self.significant]), axis=1)
                modes.extend(dom)

        taus = []
        current = modes[0]
        count = 1
        for m in modes[1:]:
            if m == current:
                count += 1
            else:
                taus.append(count)
                current = m
                count = 1
        taus.append(count)
        return taus

    # -----------------------------
    # Validação cruzada Monte Carlo
    # -----------------------------
    def cross_validation(self, files, train_size=80, iterations=3):
        real_errors = []
        null_errors = []

        for i in range(iterations):
            random.shuffle(files)
            train = files[:train_size]
            test = files[train_size:train_size*2]

            print(f"\nIteração {i+1}")

            self.fit_pca(train)
            self.apply_rmt()

            e_real = self.reconstruction_error(test, shuffle=False)
            e_null = self.reconstruction_error(test, shuffle=True)

            print(f"Erro real: {e_real:.6f}")
            print(f"Erro nulo: {e_null:.6f}")

            real_errors.append(e_real)
            null_errors.append(e_null)

        return np.mean(real_errors), np.mean(null_errors)


# -----------------------------
# Execução
# -----------------------------

arquivos = glob.glob("dados/*.csv")

exp = FullExperiment(window_size=60, max_components=20)

real_err, null_err = exp.cross_validation(arquivos)

print("\n==============================")
print(f"Erro médio REAL: {real_err:.6f}")
print(f"Erro médio NULO: {null_err:.6f}")
print("==============================")

# Análise final com todos dados
exp.fit_pca(arquivos)
exp.apply_rmt()

potentials = exp.compute_potentials()
taus = exp.analyze_dynamics(arquivos)

print(f"\nComponentes significativas: {exp.significant}")
print(f"Média τ: {np.mean(taus):.2f}")

plt.plot(potentials[0])
plt.title("Potencial V(s)")
plt.show()

plt.hist(taus, bins=50)
plt.title("Distribuição τ")
plt.show()