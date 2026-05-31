"""
Previsão de High/Low futuros a partir da vol prevista pelo HAR-full.

Decomposição: H,L = centro (≈ close de hoje, martingale) ± semi-amplitude (∝ σ̂).
A largura (range) é previsível pela vol; o centro não (1º momento, cego).

  σ̂_{t+1} = sqrt( HAR-full → RV̂_{t+1} )
  û = ln(H/C) e d̂ = ln(C/L) calibrados OOS sobre σ̂   (absorve discretização e c(H))
  Ĥ = C_t·exp(û),  L̂ = C_t·exp(−d̂)

Métricas: (a) R²_OOS do range previsto; (b) cobertura da banda H/L;
(c) vs benchmarks (RW, MA22). Gera gráfico close + barras H/L previstas.

Uso:
    python har_hl_forecast.py MSFT M1 --step 20 --embed 70 --plot-days 120
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import har_vol_forecast as HV


def har_full_design(daily, use_reg=True):
    parts = [HV.har_design(daily["RV"].to_numpy())]
    if use_reg:
        parts.append(HV.har_design(daily["bulk"].to_numpy()))
        reg = daily[HV.REGIME_COLS].to_numpy()
        reg = (reg - np.nanmean(reg, 0)) / (np.nanstd(reg, 0) + 1e-12)
        parts.append(reg)
    return np.column_stack(parts)


def oos_sigma(daily, use_reg=True, burn=300, refit=5):
    """Previsão OOS de σ̂_{t+1} (raiz da RV prevista). sig[t] = forecast p/ dia t+1."""
    RV = daily["RV"].to_numpy(); n = len(RV)
    ly = np.full(n, np.nan); ly[:-1] = np.log(RV[1:])     # alvo: log RV_{t+1}
    X = har_full_design(daily, use_reg)
    finite = np.all(np.isfinite(X), 1)
    sig = np.full(n, np.nan); beta = None
    for t in range(burn, n):
        if not finite[t]:
            continue
        v = np.isfinite(ly[:t]) & finite[:t]
        idx = np.arange(t)[v]
        if len(idx) < 100:
            continue
        if (t - burn) % refit == 0 or beta is None:
            beta = HV.ols_fit(X[idx], ly[idx])
        sig[t] = np.sqrt(np.exp(HV.ols_pred(beta, X[t:t + 1])[0]))
    return sig


def calibrate_oos(x, y, burn=350, refit=5):
    """ŷ[t] = a+b·x[t], OLS expansível só com passado (s<t)."""
    n = len(x); yh = np.full(n, np.nan); beta = None
    for t in range(burn, n):
        if not (np.isfinite(x[t])):
            continue
        v = np.isfinite(x[:t]) & np.isfinite(y[:t])
        idx = np.arange(t)[v]
        if len(idx) < 100:
            continue
        if (t - burn) % refit == 0 or beta is None:
            A = np.column_stack([np.ones(len(idx)), x[idx]])
            beta = np.linalg.lstsq(A, y[idx], rcond=None)[0]
        yh[t] = beta[0] + beta[1] * x[t]
    return yh


def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def main(symbol, tf, step, embed, plot_days):
    daily, d = HV.build_daily(symbol, tf, step, embed)
    n = len(daily)
    C = daily["close"].to_numpy()
    Hi = daily["high"].to_numpy(); Lo = daily["low"].to_numpy()
    dates = daily.index.to_numpy()
    print(f"{symbol} {tf} | {n} dias")

    # σ̂_{t+1} pelos dois modelos
    sig_full = oos_sigma(daily, use_reg=True)
    sig_rv   = oos_sigma(daily, use_reg=False)

    # alvos do dia t+1 (medidos relativamente ao close de t)
    u_tgt   = np.full(n, np.nan); d_tgt = np.full(n, np.nan); rng_tgt = np.full(n, np.nan)
    u_tgt[:-1]   = np.log(Hi[1:] / C[:-1])      # ln(H_{t+1}/C_t)  ≥ 0
    d_tgt[:-1]   = np.log(C[:-1] / Lo[1:])      # ln(C_t/L_{t+1})  ≥ 0
    rng_tgt[:-1] = np.log(Hi[1:] / Lo[1:])      # ln(H_{t+1}/L_{t+1})
    RNG = np.log(Hi / Lo)                        # range realizado do dia (p/ benchmarks)

    # ── (a) R² do RANGE previsto: HAR-full vs HAR-RV vs benchmarks ────────────
    rng_full = calibrate_oos(sig_full, rng_tgt)
    rng_rv   = calibrate_oos(sig_rv,   rng_tgt)
    rng_rw   = np.concatenate([[np.nan], RNG[:-1]])          # ŷ_{t+1}=RNG_t (persistência)
    rng_ma   = pd.Series(RNG).rolling(22).mean().shift(1).to_numpy()
    print("\n  (a) R²_OOS do log-range previsto:")
    for nm, p in [("RW (range de ontem)", rng_rw), ("MA22", rng_ma),
                  ("σ̂ HAR-RV",  rng_rv), ("σ̂ HAR-full", rng_full)]:
        print(f"      {nm:22s}: {r2(rng_tgt, p):.4f}")

    # ── (b) níveis H/L e cobertura ────────────────────────────────────────────
    uh = calibrate_oos(sig_full, u_tgt)
    dh = calibrate_oos(sig_full, d_tgt)
    Hhat = C * np.exp(uh)        # previsão de H_{t+1} (em t)
    Lhat = C * np.exp(-dh)
    # alinha ao dia previsto t+1
    Hhat_n = np.concatenate([[np.nan], Hhat[:-1]])
    Lhat_n = np.concatenate([[np.nan], Lhat[:-1]])
    m = np.isfinite(Hhat_n) & np.isfinite(Lhat_n)
    print("\n  (b) níveis H/L (R²_OOS) e cobertura:")
    print(f"      R²(H previsto)  : {r2(Hi[m], Hhat_n[m]):.4f}")
    print(f"      R²(L previsto)  : {r2(Lo[m], Lhat_n[m]):.4f}")
    cov_pt = np.mean((Hi[m] <= Hhat_n[m]) & (Lo[m] >= Lhat_n[m]))
    print(f"      cobertura ponto [L̂,Ĥ] contém [L,H]: {cov_pt:.2%}")
    # banda calibrada: escala as semi-amplitudes por c p/ atingir 80% no treino
    idx = np.where(m)[0]; half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    cs = np.linspace(1.0, 3.0, 41)
    def joint_cov(sub, c):
        Hb = C[sub] * np.exp(c * uh[sub]); Lb = C[sub] * np.exp(-c * dh[sub])
        # cuidado: aqui sub indexa o dia da PREVISÃO (t); compara com t+1
        return np.mean((Hi[sub + 1] <= Hb) & (Lo[sub + 1] >= Lb))
    # usa índices t (=dia da previsão) válidos
    valid_t = np.where(np.isfinite(uh) & np.isfinite(dh))[0]
    valid_t = valid_t[valid_t < n - 1]
    tr_t = valid_t[:len(valid_t) // 2]; te_t = valid_t[len(valid_t) // 2:]
    covs = [joint_cov(tr_t, c) for c in cs]
    c_star = cs[int(np.argmin(np.abs(np.array(covs) - 0.80)))]
    cov_te = joint_cov(te_t, c_star)
    print(f"      banda calibrada c={c_star:.2f}: cobertura OOS (2ª metade) = {cov_te:.2%}")

    # ── (c) gráfico ───────────────────────────────────────────────────────────
    sl = slice(max(0, n - 1 - plot_days), n - 1)           # dias da previsão t
    tt = np.arange(sl.start, sl.stop)
    dd = dates[tt + 1]                                      # dia previsto t+1
    fig, ax = plt.subplots(figsize=(12, 5))
    # barras H/L previstas (banda calibrada)
    Hb = C[tt] * np.exp(c_star * uh[tt]); Lb = C[tt] * np.exp(-c_star * dh[tt])
    ax.vlines(dd, Lb, Hb, color="#1f77b4", alpha=0.45, lw=3,
              label=f"banda H/L prevista (c={c_star:.2f}, {cov_te:.0%} cob.)")
    ax.plot(dd, Hi[tt + 1], "_", color="#d62728", ms=7, label="H realizado")
    ax.plot(dd, Lo[tt + 1], "_", color="#2ca02c", ms=7, label="L realizado")
    ax.plot(dd, C[tt + 1], "-", color="black", lw=1.2, label="close")
    ax.set_title(f"{symbol} {tf} — close e banda H/L prevista (σ̂ HAR-full, OOS)")
    ax.set_ylabel("preço"); ax.legend(fontsize=9, loc="best")
    ax.grid(True, ls=":", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = f"hl_forecast_{symbol}_{tf}.png"
    fig.savefig(out, dpi=140)
    print(f"\n  gráfico: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol"); ap.add_argument("timeframe")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--embed", type=int, default=70)
    ap.add_argument("--plot-days", type=int, default=120)
    a = ap.parse_args()
    main(a.symbol, a.timeframe, a.step, a.embed, a.plot_days)
