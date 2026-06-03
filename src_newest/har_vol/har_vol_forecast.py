"""
Teste preditivo: o sigma^2_bulk da decomposição agrega valor sobre RV crua?

Alvo  : variância realizada diária futura RV_{t+h} (model-free, dos retornos M1).
Feats : agregados diários de sigma^2_bulk, trace/d, sigma^2_struct (da decomposição).
Modelos: RW, EWMA(0.94), GARCH(1,1), HAR-RV, HAR-trace, HAR-bulk, HAR-RV+bulk.
OOS   : janela expansível; métricas R²_OOS (log-RV) e QLIKE (nível).

Uso:
    python har_vol_forecast.py MSFT M1 --step 20 --embed 70
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # núcleo na raiz de src_newest
import decomp_io
import decomp_pca


# ─────────────────────────────────────────────────────────────────────────────
# Dados diários: RV alvo + features espectrais + retorno diário
# ─────────────────────────────────────────────────────────────────────────────

def build_daily(symbol, tf, step, embed, winsor=0.5):
    df = decomp_pca.load_ohlcv(symbol, tf).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    logc = np.log(df["close"].to_numpy(float))
    r = np.concatenate([[np.nan], np.diff(logc)])
    r[np.abs(r) > winsor] = np.nan          # neutraliza split (gap gigante)
    df["r"] = r

    g = df.groupby("date")
    # RV intradiária: soma dos r² dentro do dia (descarta o 1º r = gap overnight)
    daily = pd.DataFrame({
        "RV":     g.apply(lambda x: np.nansum(x["r"].to_numpy()[1:] ** 2), include_groups=False),
        "close":  g["close"].last(),
        "open":   g["open"].first(),
        "high":   g["high"].max(),
        "low":    g["low"].min(),
    })
    r_d = np.concatenate([[np.nan], np.diff(np.log(daily["close"].to_numpy()))])
    r_d[np.abs(r_d) > winsor] = np.nan
    daily["r_d"] = r_d

    # ── features da decomposição ──────────────────────────────────────────────
    ev = decomp_io.read_eigenvalues(symbol, tf, step, embed)
    evals = np.stack(ev["eigenvalues"].to_list()).astype(float)   # (T,d) desc
    m = ev["m"].to_numpy(int)
    d = evals.shape[1]
    trace = evals.sum(1)
    sig2_bulk = np.array([evals[i, mi:].mean() if mi < d else np.nan
                          for i, mi in enumerate(m)])
    f_struct = np.array([evals[i, :mi].sum() / trace[i] if mi > 0 else 0.0
                         for i, mi in enumerate(m)])
    feat = pd.DataFrame({
        "date":     pd.to_datetime(ev["ts"]).dt.normalize(),
        "bulk":     sig2_bulk,
        "trace":    trace / d,
        "m":        m.astype(float),
        "entropy":  ev["entropy"].to_numpy(float),
        "f_struct": f_struct,
    }).groupby("date").mean()

    out = daily.join(feat, how="inner").dropna(subset=["RV", "bulk", "trace"])
    out = out[out["RV"] > 0]
    return out, d


# ─────────────────────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────────────────────

def har_design(x):
    """Features HAR: [x_t, média 5d, média 22d] em log."""
    lx = np.log(x)
    s = pd.Series(lx)
    return np.column_stack([
        lx,
        s.rolling(5).mean().to_numpy(),
        s.rolling(22).mean().to_numpy(),
    ])


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def ols_pred(beta, X):
    A = np.column_stack([np.ones(len(X)), X])
    return A @ beta


def ewma_var(r, lam=0.94):
    """Variância EWMA (RiskMetrics). Retorna h_t = previsão de var p/ t (1-step)."""
    h = np.empty(len(r)); h[0] = np.nanvar(r[:22])
    for t in range(1, len(r)):
        h[t] = lam * h[t - 1] + (1 - lam) * r[t - 1] ** 2
    return h


def garch11_fit(r):
    r = r[np.isfinite(r)]; r = r - r.mean()
    v0 = r.var()
    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = v0
        ll = 0.0
        for x in r:
            h = w + a * x * x + b * h
            ll += np.log(h) + x * x / h
        return 0.5 * ll
    res = minimize(nll, [v0 * 0.05, 0.08, 0.90], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-8})
    return res.x


def garch11_filter(r, p):
    w, a, b = p
    r = np.where(np.isfinite(r), r, 0.0)
    h = np.empty(len(r)); h[0] = r[:22].var() if len(r) > 22 else r.var()
    for t in range(1, len(r)):
        h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
    return h, (w, a, b)


def garch_forecast_h(h_t, p, horizon):
    """Previsão de variância MÉDIA sobre os próximos `horizon` dias (forma fechada)."""
    w, a, b = p
    persist = a + b
    uncond = w / (1 - persist) if persist < 1 else h_t
    fs = [uncond + persist ** k * (h_t - uncond) for k in range(horizon)]
    return np.mean(fs)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest OOS
# ─────────────────────────────────────────────────────────────────────────────

# specs: nome -> lista de blocos de features. Blocos disponíveis:
#   RV, bulk  = HAR-log (x_t, média5d, média22d em log)
#   reg       = m, entropia, f_struct contemporâneos (padronizados)
#   regHAR    = HAR (x, média5d, média22d) de cada feature de regime (padronizado)
SPECS = [
    ("HAR-RV",        ["RV"]),
    ("HAR-bulk",      ["bulk"]),
    ("HAR-regime",    ["reg"]),
    ("HAR-regimeHAR", ["regHAR"]),
    ("HAR-RV+bulk",   ["RV", "bulk"]),
    ("HAR-RV+reg",    ["RV", "reg"]),
    ("HAR-full",      ["RV", "bulk", "reg"]),
    ("HAR-full-HAR",  ["RV", "bulk", "regHAR"]),
]
REGIME_COLS = ["m", "entropy", "f_struct"]


def oos_eval(daily, d, horizon, burn=300, refit_every=5, verbose=True):
    RV = daily["RV"].to_numpy()
    r_d = daily["r_d"].to_numpy()
    n = len(RV)

    y = np.array([RV[t + 1:t + 1 + horizon].mean() if t + horizon < n else np.nan
                  for t in range(n)])
    ly = np.log(y)

    def har_raw(x):
        s = pd.Series(x)
        return np.column_stack([x, s.rolling(5).mean().to_numpy(),
                                s.rolling(22).mean().to_numpy()])

    def zscore(M):
        return (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-12)

    regHAR = np.column_stack([har_raw(daily[c].to_numpy()) for c in REGIME_COLS])
    blocks = {
        "RV":     har_design(daily["RV"].to_numpy()),
        "bulk":   har_design(daily["bulk"].to_numpy()),
        "reg":    zscore(daily[REGIME_COLS].to_numpy()),
        "regHAR": zscore(regHAR),
    }
    Xs = {name: np.column_stack([blocks[b] for b in bl]) for name, bl in SPECS}
    finite_all = np.all(np.isfinite(np.column_stack(list(blocks.values()))), 1)

    h_ewma = ewma_var(r_d, 0.94)
    sc_ewma = np.nanmean(RV[:burn]) / np.nanmean(h_ewma[:burn])

    names = ["RW", "EWMA"] + [s[0] for s in SPECS]
    preds = {m: [] for m in names}
    betas = {s[0]: None for s in SPECS}
    truth = []

    for t in range(burn, n):
        if not np.isfinite(ly[t]) or not finite_all[t]:
            continue
        valid = np.isfinite(ly[:t]) & finite_all[:t]
        idx = np.arange(t)[valid]
        if len(idx) < 100:
            continue
        ytr = ly[idx]

        preds["RW"].append(np.log(RV[max(0, t - horizon + 1):t + 1].mean()))
        preds["EWMA"].append(np.log(max(h_ewma[t] * sc_ewma, 1e-12)))

        refit = (t - burn) % refit_every == 0 or betas[SPECS[0][0]] is None
        for name, _ in SPECS:
            X = Xs[name]
            if refit:
                betas[name] = ols_fit(X[idx], ytr)
            preds[name].append(ols_pred(betas[name], X[t:t + 1])[0])
        truth.append(ly[t])

    truth = np.array(truth); ybar = truth.mean()
    rows = {}
    for m in names:
        p = np.array(preds[m])
        if len(p) != len(truth):
            continue
        r2 = 1 - np.sum((truth - p) ** 2) / np.sum((truth - ybar) ** 2)
        rv_t = np.exp(truth); rv_p = np.exp(p)
        q = float(np.mean(rv_t / rv_p - np.log(rv_t / rv_p) - 1))
        rows[m] = (r2, q)
    if verbose:
        print(f"\n  horizonte h={horizon}d | {len(truth)} previsões OOS")
        print(f"  {'modelo':12s} {'R2_OOS(logRV)':>14s} {'QLIKE':>10s}")
        for m in names:
            if m in rows:
                print(f"  {m:12s} {rows[m][0]:14.4f} {rows[m][1]:10.4f}")
    return rows


def main(symbol, tf, step, embed):
    daily, d = build_daily(symbol, tf, step, embed)
    print(f"{symbol} {tf} | step={step} embed={embed} (d={d}) | "
          f"{len(daily)} dias [{daily.index.min().date()}..{daily.index.max().date()}]")
    print("\n  corr(feature, RV) contemporânea (log):")
    lrv = pd.Series(np.log(daily["RV"].to_numpy()))
    for c in ["trace", "bulk", "m", "entropy", "f_struct"]:
        x = daily[c].to_numpy()
        rho = pd.Series(np.log(np.where(x > 0, x, np.nan)) if c in ("trace", "bulk")
                        else x).corr(lrv)
        print(f"    {c:9s}: {rho:.3f}")
    for h in (1, 5, 22):
        oos_eval(daily, d, h)


def multi(assets, tf, step, embed, horizons=(1, 5)):
    """Roda o backtest em vários ativos e resume R²_OOS por modelo."""
    names = ["RW", "EWMA"] + [s[0] for s in SPECS]
    agg = {h: {m: [] for m in names} for h in horizons}
    ok = []
    for sym in assets:
        try:
            daily, d = build_daily(sym, tf, step, embed)
            if len(daily) < 700:
                print(f"  [skip] {sym}: poucos dias ({len(daily)})"); continue
            for h in horizons:
                rows = oos_eval(daily, d, h, verbose=False)
                for m in names:
                    if m in rows and np.isfinite(rows[m][0]):
                        agg[h][m].append(rows[m][0])
            ok.append(sym)
            print(f"  ok {sym} ({len(daily)} dias)")
        except Exception as e:
            print(f"  [erro] {sym}: {e}")

    print(f"\n=== RESUMO multi-ativo ({len(ok)} ativos) — R²_OOS(logRV) médio ===")
    for h in horizons:
        print(f"\n  h={h}d   {'modelo':12s} {'média':>8s} {'mediana':>8s} "
              f"{'>HAR-RV':>8s}")
        base = np.array(agg[h]["HAR-RV"])
        for m in names:
            v = np.array(agg[h][m])
            if len(v) == 0:
                continue
            win = np.mean(np.array(agg[h][m]) > base) if len(v) == len(base) else np.nan
            print(f"        {m:12s} {v.mean():8.3f} {np.median(v):8.3f} {win:8.2f}")
    return agg, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", help="ticker, ou 'MULTI' para o universo")
    ap.add_argument("timeframe")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--embed", type=int, default=70)
    ap.add_argument("--assets", default="", help="lista separada por vírgula p/ MULTI")
    a = ap.parse_args()
    if a.symbol.upper() == "MULTI":
        import glob, os
        if a.assets:
            uni = a.assets.split(",")
        else:
            uni = sorted(p.split("symbol=")[1].split(os.sep)[0] for p in glob.glob(
                f"../decomp_parquet/symbol=*/timeframe={a.timeframe}/step={a.step}/embed={a.embed}/eigenvalues.parquet"))
        print(f"universo: {len(uni)} ativos")
        multi(uni, a.timeframe, a.step, a.embed)
    else:
        main(a.symbol, a.timeframe, a.step, a.embed)
