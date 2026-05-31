"""
Estimadores clássicos de volatilidade (range-based) vs. a volatilidade espectral
sqrt(trace/d) da decomposição PCA.

Confirma numericamente a identidade  vol_histórica² ≈ trace/d  e compara a
eficiência/concordância de Parkinson, Garman-Klass, Rogers-Satchell e Yang-Zhang
contra o close-to-close, janela a janela alinhada à decomposição.

Uso:
    python vol_estimators.py AAPL D1 --step 20 --embed 70
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import decomp_io
import decomp_pca

LN2 = np.log(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Estimadores per-bar (variância por barra, σ²)
# ─────────────────────────────────────────────────────────────────────────────

def per_bar_estimators(o, h, l, c, c_prev):
    """Retorna dict de arrays σ² por barra. c_prev = close da barra anterior."""
    u  = np.log(h / o)          # high  vs open
    d_ = np.log(l / o)          # low   vs open
    co = np.log(c / o)          # close vs open  (intradiária)
    hl = np.log(h / l)          # range high-low
    cc = np.log(c / c_prev)     # close-to-close (overnight + intradiária)
    on = np.log(o / c_prev)     # overnight gap (close anterior -> open)

    park = hl**2 / (4.0 * LN2)
    gk   = 0.5 * hl**2 - (2.0 * LN2 - 1.0) * co**2
    rs   = u * (u - co) + d_ * (d_ - co)   # = ln(H/C)ln(H/O)+ln(L/C)ln(L/O)
    return {
        "CC":  cc**2,           # close-to-close (1 g.l., ineficiente)
        "Park": park,
        "GK":  gk,
        "RS":  rs,
        "_co": co, "_on": on,   # auxiliares para Yang-Zhang
    }


def yang_zhang(co, on, rs):
    """YZ sobre uma janela de N barras: σ²_on + k σ²_open + (1-k) <RS>."""
    n = len(co)
    if n < 2:
        return np.nan
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_on  = on.var(ddof=1)        # variância dos gaps overnight
    var_co  = co.var(ddof=1)        # variância open->close
    return var_on + k * var_co + (1.0 - k) * rs.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de comparação
# ─────────────────────────────────────────────────────────────────────────────

def main(symbol, tf_label, step, embed):
    tf_parquet = decomp_pca._TF_TO_PARQUET.get(tf_label, tf_label)
    window = 5 * embed                       # default do decompose()

    # ── dados OHLC ───────────────────────────────────────────────────────────
    df = decomp_pca.load_ohlcv(symbol, tf_label).reset_index(drop=True)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    # ret_C[k] = ln(C[k+1]/C[k]); a barra que gera esse retorno é a linha k+1
    est = per_bar_estimators(o[1:], h[1:], l[1:], c[1:], c[:-1])
    ret_C = np.log(c[1:] / c[:-1])
    ts_ret = pd.to_datetime(df["ts"].to_numpy())[1:]     # alinhado a ret_C

    # ── autovalores da decomposição ──────────────────────────────────────────
    ev = decomp_io.read_eigenvalues(symbol, tf_parquet, step, embed)
    ev = ev.sort_values("ts").reset_index(drop=True)
    evals = ev["eigenvalues"].to_list()
    d = len(evals[0])
    trace = np.array([np.sum(e) for e in evals])
    sig2_spec = trace / d                      # variância por período (espectral)

    # mapeia cada ts de janela -> posição em ret_C (borda direita da janela)
    pos = pd.Series(np.arange(len(ts_ret)), index=ts_ret)
    win_pos = pos.reindex(pd.to_datetime(ev["ts"].to_numpy())).to_numpy()

    # ── por janela: média dos estimadores sobre os `window` retornos da janela ─
    rows = []
    ann = {"D1": 252, "H1": 252 * 6.5, "M1": 252 * 390}.get(tf_parquet, 252)
    for i, p in enumerate(win_pos):
        if np.isnan(p):
            continue
        p = int(p)
        sl = slice(max(0, p - window), p)      # janela termina em p (exclusive)
        if sl.stop - sl.start < window // 2:
            continue
        co_w = est["_co"][sl]; on_w = est["_on"][sl]; rs_w = est["RS"][sl]
        rows.append({
            "ts": ev["ts"].iloc[i],
            "spec":  sig2_spec[i],
            "varR":  ret_C[sl].var(ddof=1),    # variância amostral dos retornos
            "CC":    est["CC"][sl].mean(),
            "Park":  est["Park"][sl].mean(),
            "GK":    est["GK"][sl].mean(),
            "RS":    rs_w.mean(),
            "YZ":    yang_zhang(co_w, on_w, rs_w),
        })
    R = pd.DataFrame(rows).dropna()

    # ── relatório ──────────────────────────────────────────────────────────────
    def annvol(x):  # σ² médio -> vol anualizada (%)
        return 100.0 * np.sqrt(np.mean(x) * ann)

    print(f"\n{symbol} {tf_parquet} | step={step} embed={embed} (d={d}, "
          f"window={window}) | {len(R)} janelas | ann={ann:.0f}\n")

    print("  vol anualizada média (%):")
    print(f"    spec=sqrt(trace/d) : {annvol(R['spec']):6.2f}")
    print(f"    var(retornos)      : {annvol(R['varR']):6.2f}   <- alvo direto de trace/d")
    print(f"    Close-to-Close     : {annvol(R['CC']):6.2f}")
    print(f"    Parkinson          : {annvol(R['Park']):6.2f}")
    print(f"    Garman-Klass       : {annvol(R['GK']):6.2f}")
    print(f"    Rogers-Satchell    : {annvol(R['RS']):6.2f}")
    print(f"    Yang-Zhang         : {annvol(R['YZ']):6.2f}")

    print("\n  identidade trace/d vs var(retornos):")
    ratio = R["spec"] / R["varR"]
    print(f"    razao spec/varR    : media={ratio.mean():.4f}  std={ratio.std():.4f}")
    print(f"    corr(spec, varR)   : {R['spec'].corr(R['varR']):.4f}")

    print("\n  correlacao (Spearman) com a vol espectral, janela a janela:")
    for col in ["CC", "Park", "GK", "RS", "YZ"]:
        rho = R["spec"].corr(R[col], method="spearman")
        print(f"    spec ~ {col:14s}: {rho:.4f}")

    # eficiência relativa: var do estimador / var do CC (sobre as janelas)
    print("\n  variancia do estimador entre janelas (proxy de ruido; CC=1.0):")
    base = R["CC"].var()
    for col in ["CC", "Park", "GK", "RS", "YZ"]:
        print(f"    {col:14s}: {R[col].var()/base:6.3f}")

    out = Path(__file__).parent / f"vol_cmp_{symbol}_{tf_parquet}_s{step}_e{embed}.csv"
    R.to_csv(out, index=False)
    print(f"\n  salvo: {out}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("timeframe")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--embed", type=int, default=70)
    args = ap.parse_args()
    main(args.symbol, args.timeframe, args.step, args.embed)
