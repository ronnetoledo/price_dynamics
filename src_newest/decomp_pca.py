"""
Core da decomposição espectral PCA com janela deslizante.

Lê dados OHLCV do Parquet (data_parquet/) com fallback para CSV.
Roda PCA por janela e persiste eigenvalores + eigenvetores via decomp_io.

Uso como script:
    python decomp_pca.py NVDA 1day
    python decomp_pca.py PETR4 D1 --step 20 --embed 70
    python decomp_pca.py AAPL 1day --step 20 --embed 70 --store-mode structural_only
    python decomp_pca.py NVDA 1day --overwrite
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import eigh

import decomp_io

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────────────

_HERE     = Path(__file__).parent
DATA_ROOT = _HERE.parent / "data_parquet"

_CSV_DIRS = {
    "alpaca":      _HERE.parent / "alpaca" / "SPY500_DATA",
    "metatrader":  _HERE / "B3_DATA",
}

# Normalização de rótulos de timeframe → nome na partição Parquet
_TF_TO_PARQUET = {
    "1day":  "D1",  "1d":  "D1",  "D1":  "D1",
    "1hour": "H1",  "1h":  "H1",  "H1":  "H1",
    "4hour": "H4",  "4h":  "H4",  "H4":  "H4",
    "1min":  "M1",  "M1":  "M1",
    "5min":  "M5",  "M5":  "M5",
    "15min": "M15", "M15": "M15",
    "30min": "M30", "M30": "M30",
}


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(symbol: str, tf_parquet: str) -> pd.DataFrame:
    """Lê e concatena todos os anos disponíveis na partição Parquet."""
    for source in ("alpaca", "metatrader"):
        base = DATA_ROOT / f"source={source}" / f"symbol={symbol}" / f"timeframe={tf_parquet}"
        if not base.exists():
            continue
        parts = sorted(base.glob("year=*/data.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        return df.sort_values("ts").reset_index(drop=True)
    raise FileNotFoundError(
        f"Parquet não encontrado: symbol={symbol} timeframe={tf_parquet} em {DATA_ROOT}"
    )


def _load_csv(symbol: str, tf_label: str) -> pd.DataFrame:
    """Fallback: lê CSV bruto da Alpaca ou MetaTrader."""
    for src, folder in _CSV_DIRS.items():
        path = folder / f"{symbol}_{tf_label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        ts_col = next((c for c in df.columns if c in ("timestamp", "time", "date")), None)
        if ts_col:
            df["ts"] = pd.to_datetime(df[ts_col])
            if df["ts"].dt.tz is not None:
                df["ts"] = df["ts"].dt.tz_localize(None)
        return df.sort_values("ts").reset_index(drop=True)
    raise FileNotFoundError(
        f"CSV não encontrado: {symbol}_{tf_label}.csv nos diretórios {list(_CSV_DIRS.values())}"
    )


def load_ohlcv(symbol: str, tf_label: str) -> pd.DataFrame:
    """Carrega OHLCV priorizando Parquet; fallback para CSV."""
    tf_parquet = _TF_TO_PARQUET.get(tf_label, tf_label)
    try:
        return _load_parquet(symbol, tf_parquet)
    except FileNotFoundError:
        return _load_csv(symbol, tf_label)


# ─────────────────────────────────────────────────────────────────────────────
# Pré-processamento
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(df: pd.DataFrame):
    """
    Calcula log-retornos de OHLC.

    Retorna
    -------
    ts   : (N-1,) datetime64  — timestamps alinhados com os retornos
    ret_C: (N-1,) float64     — log-retornos do close (usado no modo scalar)
    """
    C  = df["close"].values.astype(np.float64)
    ts = pd.to_datetime(df["ts"].values)
    return ts[1:], np.diff(np.log(C))


# ─────────────────────────────────────────────────────────────────────────────
# Funções matemáticas do núcleo
# ─────────────────────────────────────────────────────────────────────────────

def hankel_embed(x: np.ndarray, m: int) -> np.ndarray:
    """Embedding de Hankel via stride_tricks — zero-copy, sem alocação extra."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    N = len(x) - m
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=(N, m), strides=strides)


def pca_cov(X: np.ndarray):
    """
    PCA via matriz de covariância amostral.

    Retorna
    -------
    vals : (d,)   float64 — autovalores em ordem decrescente
    vecs : (d, d) float64 — autovetores correspondentes (colunas)
    var  : float          — variância média empírica (usada no limiar MP)
    """
    C    = np.cov(X, rowvar=False)
    var  = float(np.mean(np.var(X, axis=0, ddof=1)))
    vals, vecs = eigh(C)
    idx  = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx], var


def mp_threshold(var: float, q: float) -> float:
    """Limiar superior de Marchenko-Pastur: λ+ = σ²(1 + √q)²."""
    return var * (1.0 + np.sqrt(q)) ** 2


def spectral_entropy(vals: np.ndarray, d: int) -> float:
    """Entropia de Shannon normalizada pela dimensão: S ∈ [0, 1]."""
    lams = np.maximum(vals, 1e-15)
    p    = lams / lams.sum()
    return float(-np.sum(p * np.log(p)) / np.log(d))


def build_window(ret_C: np.ndarray, center: int,
                 window: int, embed: int) -> np.ndarray | None:
    """
    Extrai a janela causal centrada em `center` e constrói o embedding de Hankel.

    Usa `window` pontos à esquerda de `center`, centraliza a série e aplica
    o embedding. Retorna None se a janela sair dos limites.

    Retorna X de shape (window - embed, embed).
    """
    start = center - window
    if start < 0:
        return None
    seg = ret_C[start:center]
    seg = seg - seg.mean()
    return hankel_embed(seg, embed)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal de decomposição
# ─────────────────────────────────────────────────────────────────────────────

def decompose(symbol: str, tf_label: str,
              step: int, embed_dim: int,
              window: int = None,
              store_mode: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decomposição espectral por janela deslizante.

    Para cada janela centrada em τ:
      1. Extrai `window` pontos de log-retorno do close
      2. Constrói embedding de Hankel X ∈ R^{(window-embed) × embed}
      3. Calcula covariância amostral C = X^T X / (n-1)
      4. Eigendecomposição: C = V diag(λ) V^T  (eigh, autovalores reais)
      5. Aplica limiar Marchenko-Pastur para separar modos estruturais (m)
      6. Calcula entropia espectral normalizada

    Parâmetros
    ----------
    symbol     : ticker (ex: 'NVDA', 'PETR4')
    tf_label   : rótulo do timeframe (ex: '1day', 'D1', 'H1')
    step       : passo da janela deslizante
    embed_dim  : dimensão d do embedding de Hankel
    window     : tamanho da janela (padrão: 5 × embed_dim)
    store_mode : 'all'             — armazena todos os d modos por janela
                 'structural_only' — armazena apenas os m modos estruturais

    Retorna
    -------
    (evals_df, evecs_df) prontos para decomp_io.write()
    """
    if window is None:
        window = 5 * embed_dim

    df        = load_ohlcv(symbol, tf_label)
    ts, ret_C = compute_log_returns(df)

    centers = list(range(window, len(ret_C) - window, step))
    n       = len(centers)
    if n == 0:
        raise ValueError(
            f"Série muito curta para window={window}, step={step}: "
            f"len(ret_C)={len(ret_C)}"
        )

    print(f"  {symbol} | {tf_label} | step={step} embed={embed_dim} "
          f"window={window} | {n} janelas")

    # ── pré-alocação ─────────────────────────────────────────────────────────
    ts_valid  = []
    m_arr     = np.empty(n, dtype=np.int16)
    lam_arr   = np.empty(n, dtype=np.float32)
    entr_arr  = np.empty(n, dtype=np.float32)
    evals_mat = np.empty((n, embed_dim), dtype=np.float32)   # (T, d)
    evecs_mat = np.empty((n, embed_dim, embed_dim), dtype=np.float32)  # (T, d, d)

    t0    = time.time()
    count = 0

    # ── loop de decomposição ──────────────────────────────────────────────────
    for center in centers:
        X = build_window(ret_C, center, window, embed_dim)
        if X is None:
            continue

        vals, vecs, var = pca_cov(X)

        q        = embed_dim / X.shape[0]
        lam_plus = mp_threshold(var, q)
        m        = int(np.sum(vals > lam_plus))
        entropy  = spectral_entropy(vals, embed_dim)

        ts_valid.append(ts[center])
        m_arr   [count] = np.int16(m)
        lam_arr [count] = np.float32(lam_plus)
        entr_arr[count] = np.float32(entropy)
        evals_mat[count] = vals.astype(np.float32)
        evecs_mat[count] = vecs.astype(np.float32)  # (d, d): colunas = autovetores
        count += 1

    elapsed = time.time() - t0
    print(f"  concluído: {count}/{n} janelas válidas em {elapsed:.1f}s "
          f"({elapsed/count*1000:.1f}ms/janela)")

    # ── recorta arrays ao tamanho real ────────────────────────────────────────
    ts_valid  = np.array(ts_valid, dtype="datetime64[us]")
    m_arr     = m_arr[:count]
    lam_arr   = lam_arr[:count]
    entr_arr  = entr_arr[:count]
    evals_mat = evals_mat[:count]
    evecs_mat = evecs_mat[:count]

    # ── monta eigenvalues DataFrame (formato largo) ───────────────────────────
    evals_df = pd.DataFrame({
        "ts":          ts_valid,
        "m":           m_arr,
        "lam_plus":    lam_arr,
        "entropy":     entr_arr,
        "eigenvalues": [evals_mat[i].tolist() for i in range(count)],
    })

    # ── monta eigenvectors DataFrame (formato longo) ──────────────────────────
    # Uma linha por (janela, modo). Comprimido eficientemente pelo Parquet
    # porque os campos ts e is_structural têm baixa cardinalidade.
    evec_ts       = []
    evec_midx     = []
    evec_isstruct = []
    evec_vecs     = []

    for i in range(count):
        m_i     = int(m_arr[i])
        n_modes = embed_dim if store_mode == "all" else m_i
        for k in range(n_modes):
            evec_ts.append(ts_valid[i])
            evec_midx.append(np.int16(k))
            evec_isstruct.append(bool(k < m_i))
            evec_vecs.append(evecs_mat[i, :, k].tolist())  # k-ésimo autovetor

    evecs_df = pd.DataFrame({
        "ts":            evec_ts,
        "mode_idx":      np.array(evec_midx, dtype=np.int16),
        "is_structural": evec_isstruct,
        "eigenvector":   evec_vecs,
    })

    return evals_df, evecs_df


# ─────────────────────────────────────────────────────────────────────────────
# Orquestrador
# ─────────────────────────────────────────────────────────────────────────────

def run(symbol: str, tf_label: str,
        step: int = 20, embed_dim: int = 70,
        window: int = None,
        store_mode: str = "all",
        overwrite: bool = False,
        decomp_root: Path = None) -> None:
    """
    Orquestra decomposição e persistência. Pula se já existir e overwrite=False.
    """
    root       = decomp_root or decomp_io.DECOMP_ROOT
    tf_parquet = _TF_TO_PARQUET.get(tf_label, tf_label)

    if not overwrite and decomp_io.exists(symbol, tf_parquet, step, embed_dim, root):
        print(f"  [skip] {symbol} {tf_label} step={step} embed={embed_dim} — já existe")
        return

    print(f"\n[decomp] {symbol} | {tf_label} | step={step} | embed={embed_dim}")
    evals_df, evecs_df = decompose(symbol, tf_label, step, embed_dim,
                                   window=window, store_mode=store_mode)

    decomp_io.write(evals_df, evecs_df, symbol, tf_parquet, step, embed_dim, root)

    n_win = len(evals_df)
    n_evec = len(evecs_df)
    print(f"  [salvo] {symbol}/{tf_parquet}/step={step}/embed={embed_dim} "
          f"— {n_win} janelas, {n_evec} linhas de eigenvetores")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Decomposição espectral PCA de séries financeiras → Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("symbol",    help="Ticker (ex: NVDA, PETR4)")
    p.add_argument("timeframe", help="Timeframe (ex: 1day, D1, H1)")
    p.add_argument("--step",       type=int, default=20,  help="Passo da janela deslizante")
    p.add_argument("--embed",      type=int, default=70,  help="Dimensão do embedding de Hankel")
    p.add_argument("--window",     type=int, default=None,
                   help="Tamanho da janela (padrão: 5 × embed)")
    p.add_argument("--store-mode", choices=["all", "structural_only"], default="all",
                   help="Modos a armazenar: 'all' ou apenas estruturais acima do limiar MP")
    p.add_argument("--overwrite",  action="store_true",
                   help="Refaz a decomposição mesmo se já existir no Parquet")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        symbol     = args.symbol,
        tf_label   = args.timeframe,
        step       = args.step,
        embed_dim  = args.embed,
        window     = args.window,
        store_mode = args.store_mode,
        overwrite  = args.overwrite,
    )
