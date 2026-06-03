"""
MSD/variancia-tempo do LOG-PRECO -> expoente beta_preco.

V(tau) = Var( ln P_{t+tau} - ln P_t )  ~  tau^beta
  beta = 1  <=> variancia aditiva (Browniano) <=> regra sqrt(time) valida
            <=> constante 1/(4 ln2) de Parkinson bem calibrada.

A variancia (e nao o 2o momento bruto) ja remove o termo de drift (mu*tau)^2,
isolando a difusao. Diferencas sobrepostas em t (mais amostras); o slope e
ajustado numa faixa de tau acima da microestrutura.

Uso:
    python price_msd.py MSFT D1
    python price_msd.py AAPL D1 --winsor 0.5     # capa o retorno do split
"""
import argparse
import numpy as np
import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # núcleo na raiz de src_newest
import decomp_pca


def variance_time(logP, taus):
    """V(tau) = Var das diferencas sobrepostas de logP no lag tau."""
    V = np.empty(len(taus))
    for i, t in enumerate(taus):
        d = logP[t:] - logP[:-t]
        V[i] = d.var(ddof=1)
    return V


def fit_slope(taus, V, lo, hi):
    """slope log-log de V(tau) na faixa [lo, hi] de tau."""
    m = (taus >= lo) & (taus <= hi) & np.isfinite(V) & (V > 0)
    coef, cov = np.polyfit(np.log(taus[m]), np.log(V[m]), 1, cov=True)
    return coef[0], np.sqrt(cov[0, 0]), taus[m]


def main(symbol, tf, winsor, lo, hi, max_tau):
    df = decomp_pca.load_ohlcv(symbol, tf).reset_index(drop=True)
    c = df["close"].to_numpy(float)
    r = np.log(c[1:] / c[:-1])

    n_capped = 0
    if winsor is not None:
        big = np.abs(r) > winsor
        n_capped = int(big.sum())
        r = r.copy()
        r[big] = np.median(r)                 # neutraliza split/erro pontual
    logP = np.concatenate([[0.0], np.cumsum(r)])   # log-preco reconstruido

    taus = np.arange(1, min(max_tau, len(logP) // 4))
    V = variance_time(logP, taus)
    beta, err, used = fit_slope(taus, V, lo, hi)

    interp = ("difusao normal (beta~1) -> sqrt(time) OK" if 0.9 <= beta <= 1.1 else
              "SUPERdifusao (beta>1) -> sqrt(time) SUBESTIMA horizonte longo" if beta > 1.1 else
              "SUBdifusao (beta<1) -> sqrt(time) SUPERESTIMA / mean-revert")
    print(f"\n{symbol} {tf} | n={len(c)} | winsor={winsor} (capou {n_capped}) | "
          f"fit tau in [{used[0]},{used[-1]}]")
    print(f"  beta_preco = {beta:.4f} +/- {err:.4f}   {interp}")
    # eficiencia da regra sqrt(time): razao V(tau)/(V(1)*tau) a tau alto
    j = np.searchsorted(taus, hi)
    ratio = V[j] / (V[0] * taus[j])
    print(f"  V({taus[j]})/(V(1)*{taus[j]}) = {ratio:.3f}  "
          f"(>1 long-memory, <1 anti-persistente; =1 aditivo)")
    return beta, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol"); ap.add_argument("timeframe")
    ap.add_argument("--winsor", type=float, default=None)
    ap.add_argument("--lo", type=int, default=5)
    ap.add_argument("--hi", type=int, default=120)
    ap.add_argument("--max-tau", type=int, default=250)
    a = ap.parse_args()
    main(a.symbol, a.timeframe, a.winsor, a.lo, a.hi, a.max_tau)
