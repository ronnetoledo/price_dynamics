"""
Distribuição preditiva do caminho intradiário do próximo dia, escalada pela vol
prevista (HAR-RV). Testa se a FORMA empírica (perfil-U + caudas) calibra melhor que
GBM — não para bater o ponto de range (morto, ver docs/ohlc_forecast.md §4), mas para
probabilidades de patamares / caudas / range bem-calibradas.

Modelos aninhados, todos com RV total = σ̂²_{t+1} (HAR-RV OOS):
  M0 (GBM)    : perfil plano + z ~ N(0,1) iid
  M1 (+perfil): perfil-U empírico + z ~ N(0,1)
  M2 (+caudas): perfil-U + z ~ bootstrap dos retornos intradiários padronizados

Tudo relativo à abertura do dia (caminho intradiário; gap overnight fora). Forma
estimada no treino; σ̂ expansível; calibração medida no teste (PIT, cobertura, CRPS).

Uso:
    python ohlc_dist.py AAPL M1
    python ohlc_dist.py MSFT H1 --nboot 4000
"""
import argparse
import numpy as np
import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # núcleo na raiz de src_newest
import decomp_pca


# ── agregação diária tf-agnóstica (não depende da decomposição) ───────────────
def build_days(symbol, tf, winsor=0.5):
    df = decomp_pca.load_ohlcv(symbol, tf).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    logc = np.log(df["close"].to_numpy(float))
    r = np.concatenate([[np.nan], np.diff(logc)])
    r[np.abs(r) > winsor] = np.nan
    df["r"] = r

    rets, RV, O, H, L, C, dates = [], [], [], [], [], [], []
    for d, x in df.groupby("date"):
        ri = x["r"].to_numpy()[1:]                 # dropa gap overnight
        if not np.all(np.isfinite(ri)) or len(ri) < 5:
            continue
        rets.append(ri)
        RV.append(float(np.sum(ri ** 2)))
        O.append(float(x["open"].iloc[0])); C.append(float(x["close"].iloc[-1]))
        H.append(float(x["high"].max()));   L.append(float(x["low"].min()))
        dates.append(d)
    W = int(np.median([len(r) for r in rets]))     # comprimento típico de sessão
    # matriz alinhada por barra-do-dia (primeiras W barras); descarta dias curtos
    keep = [i for i, r in enumerate(rets) if len(r) >= W]
    R = np.array([rets[i][:W] for i in keep])       # (ndays, W)
    idx = np.array(keep)
    return dict(R=R, RV=np.array(RV)[idx], W=W,
                O=np.array(O)[idx], H=np.array(H)[idx],
                L=np.array(L)[idx], C=np.array(C)[idx],
                dates=np.array(dates)[idx])


# ── HAR-RV expansível (log RV_{t+1}) ──────────────────────────────────────────
def har_rv_oos(RV, burn=300, refit=5):
    lrv = np.log(RV)
    s = pd.Series(lrv)
    X = np.column_stack([lrv, s.rolling(5).mean(), s.rolling(22).mean()])
    n = len(RV)
    sig2 = np.full(n, np.nan); beta = None
    fin = np.all(np.isfinite(X), 1)
    for t in range(burn, n - 1):
        if not fin[t]:
            continue
        v = fin[:t] & np.isfinite(lrv[1:t + 1])     # alvo lrv[s+1] p/ s<t
        idx = np.arange(t)[v[:t]]
        if len(idx) < 100:
            continue
        if (t - burn) % refit == 0 or beta is None:
            A = np.column_stack([np.ones(len(idx)), X[idx]])
            beta = np.linalg.lstsq(A, lrv[idx + 1], rcond=None)[0]
        sig2[t + 1] = np.exp(beta[0] + beta[1:] @ X[t])   # previsão p/ dia t+1
    return sig2                                      # variância (RV) prevista


# ── bancos de caminhos de RV unitária (forma do treino) ───────────────────────
def make_bank(profile, zpool, model, nboot, W, rng):
    """Amostra nboot caminhos de RV total = 1; devolve (close, high, low) em log."""
    if model == "M0":
        p = np.full(W, 1.0 / W); Z = rng.standard_normal((nboot, W))
    elif model == "M1":
        p = profile; Z = rng.standard_normal((nboot, W))
    else:  # M2
        p = profile; Z = rng.choice(zpool, size=(nboot, W), replace=True)
    steps = np.sqrt(p)[None, :] * Z                  # retornos por barra
    P = np.cumsum(steps, axis=1)                      # caminho (open=0)
    return P[:, -1], P.max(1), P.min(1)               # close, high, low (unit)


# ── CRPS via amostras ─────────────────────────────────────────────────────────
def crps(samples, y):
    a = float(np.mean(np.abs(samples - y)))
    s = np.sort(samples); n = len(s)
    # E|X-X'| = (2/n²) Σ_i (2i-n+1) s_i   (i de 1..n)
    i = np.arange(1, n + 1)
    b = (2.0 / n ** 2) * np.sum((2 * i - n - 1) * s)
    return a - 0.5 * b


def evaluate(symbol, tf, nboot=3000, seed=0):
    rng = np.random.default_rng(seed)
    D = build_days(symbol, tf)
    R, RV, W = D["R"], D["RV"], D["W"]
    n = len(RV)
    if n < 400:
        print(f"{symbol} {tf}: poucos dias ({n})"); return
    sig2 = har_rv_oos(RV)                             # σ̂² OOS

    # alvos relativos à abertura (caminho intradiário)
    y_close = np.log(D["C"] / D["O"])
    y_high  = np.log(D["H"] / D["O"])
    y_low   = np.log(D["L"] / D["O"])
    y_rng   = np.log(D["H"] / D["L"])

    split = n // 2
    tr = slice(0, split)
    # forma estimada SÓ no treino: perfil-U = variância média por barra-do-dia
    profile = (R[tr] ** 2).mean(0)
    profile = profile / profile.sum()
    # pool padronizado (remove nível do dia e perfil) → captura caudas
    z = R[tr] / np.sqrt(RV[tr][:, None] * profile[None, :])
    zpool = z[np.isfinite(z)]
    zpool = zpool / zpool.std()                       # var unitária

    banks = {m: make_bank(profile, zpool, m, nboot, W, rng)
             for m in ("M0", "M1", "M2")}

    test = [t for t in range(split, n) if np.isfinite(sig2[t]) and sig2[t] > 0]
    print(f"{symbol} {tf} | {n} dias (treino {split}, teste {len(test)}) | W={W}")
    print(f"{'modelo':>6} | {'PIT-KS':>7} {'cov95_C':>8} {'cov95_H':>8} {'cov95_L':>8} "
          f"| {'CRPS_C':>8} {'CRPS_H':>8} {'CRPS_L':>8} {'CRPS_rng':>8}")
    for m, (cu, hu, lu) in banks.items():
        pit = []; covC = covH = covL = 0
        cC = cH = cL = cR = 0.0
        for t in test:
            sig = np.sqrt(sig2[t])
            sc, sh, sl = sig * cu, sig * hu, sig * lu
            srng = sh - sl
            pit.append(np.mean(sc <= y_close[t]))
            qC = np.quantile(sc, [0.025, 0.975]); covC += (qC[0] <= y_close[t] <= qC[1])
            qH = np.quantile(sh, [0.025, 0.975]); covH += (qH[0] <= y_high[t] <= qH[1])
            qL = np.quantile(sl, [0.025, 0.975]); covL += (qL[0] <= y_low[t] <= qL[1])
            cC += crps(sc, y_close[t]); cH += crps(sh, y_high[t])
            cL += crps(sl, y_low[t]);   cR += crps(srng, y_rng[t])
        nt = len(test)
        pit = np.array(pit)
        # KS vs uniforme
        ks = np.max(np.abs(np.sort(pit) - (np.arange(1, nt + 1) / nt)))
        print(f"{m:>6} | {ks:>7.3f} {covC/nt:>8.2%} {covH/nt:>8.2%} {covL/nt:>8.2%} "
              f"| {1e3*cC/nt:>8.3f} {1e3*cH/nt:>8.3f} {1e3*cL/nt:>8.3f} {1e3*cR/nt:>8.3f}")
    print("  (CRPS ×1000, menor=melhor; cov95 alvo=95%; PIT-KS menor=mais calibrado)")


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol"); ap.add_argument("timeframe")
    ap.add_argument("--nboot", type=int, default=3000)
    args = ap.parse_args()
    evaluate(args.symbol, args.timeframe, nboot=args.nboot)


if __name__ == "__main__":
    _main()
