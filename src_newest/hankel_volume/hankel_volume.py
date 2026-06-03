"""
Análise espectral de Hankel aplicada ao VOLUME (análoga à dos retornos).

CLI fino sobre hankel_core: mesmo algoritmo do Hankel dos log-retornos (embedding
-> covariância -> autovalores -> limiar MP -> m/entropia; β do MSD da sequência
espectral deslizante), trocando a série de entrada de Δln(preço) por uma
transformação do volume. window = mult × embed; figuras no padrão hankel_demo.png.

Transformações (--transform): dlogv (Δln V, default), logv (ln V), raw (V),
zlogv (z-score de ln V — estruturalmente ≡ logv: a covariância centra por coluna,
o z-score só reescala os autovalores → mesmos m, entropia e β).

Expoente β (--beta): roda a decomposição DESLIZANTE sobre toda a série de volume
(centros espaçados por --step), forma as séries total/struct/bulk e ajusta
MSD(s) ~ s^β na cauda. Janelas sobrepostas (step pequeno) enviesam β para cima;
aumente --step para reduzir a sobreposição.

Uso:
    python hankel_volume.py NVDA --tf D1 --embed 70 --mult 5
    python hankel_volume.py MSFT --tf D1 --transform all          # 4 figuras
    python hankel_volume.py NVDA --tf D1 --end 2019-06-20         # janela na data
    python hankel_volume.py NVDA --tf D1 --beta --transform all   # β total/struct/bulk
    python hankel_volume.py NVDA --tf D1 --beta --step 5          # menos sobreposição
"""
from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # núcleo na raiz de src_newest

import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import decomp_pca
import hankel_core as hk

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TF_LABEL = {"M1": "1 min", "M5": "5 min", "M15": "15 min", "M30": "30 min",
             "H1": "1 hora", "H4": "4 horas", "D1": "Diário", "W1": "Semanal",
             "MN": "Mensal", "Q": "Trimestral"}
_TRANSFORMS = ("dlogv", "logv", "raw", "zlogv")     # flags de volume
_VOL_MAP = {"dlogv": "v_dlog", "logv": "v_log", "raw": "v_raw", "zlogv": "v_zlog"}


def analyze(symbol, tf, embed, mult, transform, adjusted=True, end=None):
    df = decomp_pca.load_ohlcv(symbol, tf, adjusted=adjusted).sort_values("ts").reset_index(drop=True)
    S, ts = hk.make_series(df, transform)
    win = mult * embed
    if win > len(S):
        raise ValueError(f"janela {win} (mult×embed) > série disponível ({len(S)})")
    if end is not None:
        pos_end = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(end)), side="right")) - 1
        if pos_end < win - 1:
            raise ValueError(f"dados insuficientes antes de {end} para janela {win}")
    else:
        pos_end = len(S) - 1
    pos0 = pos_end - win + 1
    seg = S[pos0:pos_end + 1]
    spec = hk.hankel_spectrum(seg, embed)
    if spec is None:
        raise ValueError(f"janela curta demais para embed={embed}")
    wdate = f"{pd.Timestamp(ts[pos0]):%Y-%m-%d} → {pd.Timestamp(ts[pos_end]):%Y-%m-%d}"
    return seg, spec, wdate


def betas(symbol, tf, embed, mult, transform, adjusted=True, step=1, max_lag=200, tail_frac=0.7):
    df = decomp_pca.load_ohlcv(symbol, tf, adjusted=adjusted).sort_values("ts").reset_index(drop=True)
    S, _ = hk.make_series(df, transform)
    return hk.compute_betas(S, embed, mult, step, max_lag, tail_frac)


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--tf", default="D1")
    ap.add_argument("--embed", type=int, default=20)
    ap.add_argument("--mult", type=int, default=5, help="window = mult × embed")
    ap.add_argument("--transform", choices=(*_TRANSFORMS, "all"), default="dlogv")
    ap.add_argument("--end", default=None, help="data final da janela (YYYY-MM-DD)")
    ap.add_argument("--raw", action="store_true", help="usa volume bruto (sem split-adjust)")
    ap.add_argument("--out", default=None, help="diretório de saída (default: junto do módulo)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--beta", action="store_true", help="calcula β (total/struct/bulk) via MSD deslizante")
    ap.add_argument("--step", type=int, default=1, help="stride entre centros de janela (--beta)")
    ap.add_argument("--max-lag", type=int, default=200)
    ap.add_argument("--tail-frac", type=float, default=0.7)
    args = ap.parse_args()

    tfl = _TF_LABEL.get(args.tf, args.tf)
    outdir = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)
    transforms = list(_TRANSFORMS) if args.transform == "all" else [args.transform]

    for tr in transforms:
        kind = _VOL_MAP[tr]
        label = hk.SERIES[kind]
        try:
            if args.beta:
                b = betas(args.symbol, args.tf, args.embed, args.mult, kind,
                          adjusted=not args.raw, step=args.step,
                          max_lag=args.max_lag, tail_frac=args.tail_frac)
                fig = hk.beta_figure(b, args.symbol, tfl, label)
                fname = outdir / f"beta_vol_{args.symbol}_{args.tf}_{tr}_d{args.embed}m{args.mult}s{args.step}.png"
                fig.savefig(fname, dpi=args.dpi); plt.close(fig)
                bb = {k: b[k]["beta"] for k in ("total", "struct", "bulk")}
                print(f"[{tr}] β_total={bb['total']:.3f} β_struct={bb['struct']:.3f} "
                      f"β_bulk={bb['bulk']:.3f} | ⟨m⟩={b['meta']['m_mean']:.1f} "
                      f"n_centers={b['meta']['n_centers']} -> {fname.name}")
            else:
                seg, spec, wdate = analyze(args.symbol, args.tf, args.embed, args.mult,
                                           kind, adjusted=not args.raw, end=args.end)
                fig = hk.hankel_figure(seg, args.embed, spec, args.symbol, tfl, label, wdate)
                fname = outdir / f"hankel_vol_{args.symbol}_{args.tf}_{tr}_d{args.embed}m{args.mult}.png"
                fig.savefig(fname, dpi=args.dpi); plt.close(fig)
                print(f"[{tr}] n_samp={spec['n_samp']} q={spec['q']:.3f} "
                      f"m={spec['m']} entropia={spec['entropy']:.3f} -> {fname.name}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[{tr}] SKIP — {e}")


if __name__ == "__main__":
    main()
