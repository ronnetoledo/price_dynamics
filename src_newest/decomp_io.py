"""
I/O de decomposição espectral em formato Parquet.

Estrutura de saída:
    <DECOMP_ROOT>/
    └── symbol=<ATIVO>/
        └── timeframe=<TF>/
            └── step=<S>/
                └── embed=<E>/
                    ├── eigenvalues.parquet
                    └── eigenvectors.parquet

Schema — eigenvalues.parquet (uma linha por janela temporal):
    ts          : timestamp[us]  — instante central da janela
    m           : int16          — nº de modos estruturais (acima do limiar MP)
    lam_plus    : float32        — limiar Marchenko-Pastur λ+ = σ²(1+√q)²
    entropy     : float32        — entropia espectral normalizada ∈ [0,1]
    eigenvalues : list<float32>  — autovalores em ordem decrescente (embed_dim valores)

Schema — eigenvectors.parquet (uma linha por (janela, modo)):
    ts            : timestamp[us] — instante central da janela
    mode_idx      : int16         — índice do modo (0 = maior autovalor)
    is_structural : bool          — True se mode_idx < m
    eigenvector   : list<float32> — componentes do autovetor (embed_dim valores)

Por que list<float32> e não colunas separadas?
    Parquet é colunar: filtrar por mode_idx ou ts é O(1) independente de embed_dim.
    Colunas v_0..v_69 criam 70 colunas desnecessárias — as listas são mais compactas
    e permitem embed_dim variável sem alterar o schema.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DECOMP_ROOT = Path(__file__).parent.parent / "decomp_parquet"

_SCHEMA_EVALS = pa.schema([
    ("ts",          pa.timestamp("us")),
    ("m",           pa.int16()),
    ("lam_plus",    pa.float32()),
    ("entropy",     pa.float32()),
    ("eigenvalues", pa.list_(pa.float32())),
])

_SCHEMA_EVECS = pa.schema([
    ("ts",            pa.timestamp("us")),
    ("mode_idx",      pa.int16()),
    ("is_structural", pa.bool_()),
    ("eigenvector",   pa.list_(pa.float32())),
])


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def _partition_path(root: Path, symbol: str, timeframe: str,
                    step: int, embed: int) -> Path:
    return (root
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"step={step}"
            / f"embed={embed}")


# ─────────────────────────────────────────────────────────────────────────────
# Write / Read / Exists
# ─────────────────────────────────────────────────────────────────────────────

def exists(symbol: str, timeframe: str, step: int, embed: int,
           root: Path = DECOMP_ROOT) -> bool:
    p = _partition_path(root, symbol, timeframe, step, embed)
    return (p / "eigenvalues.parquet").exists()


def write(evals_df: pd.DataFrame, evecs_df: pd.DataFrame,
          symbol: str, timeframe: str, step: int, embed: int,
          root: Path = DECOMP_ROOT) -> None:
    """Persiste eigenvalores e eigenvetores em Parquet com compressão Snappy."""
    dest = _partition_path(root, symbol, timeframe, step, embed)
    dest.mkdir(parents=True, exist_ok=True)

    evals_tbl = pa.Table.from_pandas(evals_df, schema=_SCHEMA_EVALS,
                                     preserve_index=False)
    pq.write_table(evals_tbl, dest / "eigenvalues.parquet", compression="snappy")

    evecs_tbl = pa.Table.from_pandas(evecs_df, schema=_SCHEMA_EVECS,
                                     preserve_index=False)
    pq.write_table(evecs_tbl, dest / "eigenvectors.parquet", compression="snappy")


def read_eigenvalues(symbol: str, timeframe: str, step: int, embed: int,
                     root: Path = DECOMP_ROOT) -> pd.DataFrame:
    path = _partition_path(root, symbol, timeframe, step, embed) / "eigenvalues.parquet"
    return pq.read_table(path).to_pandas()


def read_eigenvectors(symbol: str, timeframe: str, step: int, embed: int,
                      root: Path = DECOMP_ROOT) -> pd.DataFrame:
    path = _partition_path(root, symbol, timeframe, step, embed) / "eigenvectors.parquet"
    return pq.read_table(path).to_pandas()


# ─────────────────────────────────────────────────────────────────────────────
# Conversão para arrays numpy (caminho eficiente para análise downstream)
# ─────────────────────────────────────────────────────────────────────────────

def load_as_arrays(symbol: str, timeframe: str, step: int, embed: int,
                   root: Path = DECOMP_ROOT) -> dict:
    """
    Carrega a decomposição e retorna arrays numpy prontos para análise.

    Retorna dict com chaves:
        ts          : (T,)       datetime64[us]
        m           : (T,)       int16   — nº de modos estruturais por janela
        lam_plus    : (T,)       float32 — limiar MP por janela
        entropy     : (T,)       float32 — entropia espectral por janela
        eigenvalues : (T, d)     float32 — autovalores em ordem decrescente
        eigenvectors: (T, d, d)  float32 — eigenvectors[:, :, k] = k-ésimo autovetor

    Nota: eigenvectors só está disponível quando store_mode='all' foi usado
    na decomposição. Se os eigenvetores foram salvos apenas parcialmente
    (structural_only), o campo 'eigenvectors' será None.
    """
    evals_df = read_eigenvalues(symbol, timeframe, step, embed, root)
    evecs_df = read_eigenvectors(symbol, timeframe, step, embed, root)

    T = len(evals_df)
    d = len(evals_df["eigenvalues"].iloc[0])

    ts         = evals_df["ts"].values.astype("datetime64[us]")
    m          = evals_df["m"].values.astype(np.int16)
    lam_plus   = evals_df["lam_plus"].values.astype(np.float32)
    entropy    = evals_df["entropy"].values.astype(np.float32)
    eigenvalues = np.stack(evals_df["eigenvalues"].tolist()).astype(np.float32)

    # Detecta se todos os modos foram armazenados
    n_rows_expected = T * d
    eigenvectors = None
    if len(evecs_df) == n_rows_expected:
        evecs_sorted = evecs_df.sort_values(["ts", "mode_idx"])
        all_vecs = np.stack(evecs_sorted["eigenvector"].tolist()).astype(np.float32)
        # shape: (T*d, d) → (T, d, d); eixo 2 = índice do modo
        eigenvectors = all_vecs.reshape(T, d, d).transpose(0, 2, 1)

    return {
        "ts":           ts,
        "m":            m,
        "lam_plus":     lam_plus,
        "entropy":      entropy,
        "eigenvalues":  eigenvalues,
        "eigenvectors": eigenvectors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reconstrução de matrizes de covariância
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_C_series(arrays: dict,
                         subspace: str = "struct") -> np.ndarray:
    """
    Reconstrói a série de matrizes C para todos os instantes.

    Parâmetros
    ----------
    arrays   : saída de load_as_arrays()
    subspace : 'struct' | 'bulk' | 'total'

    Retorna
    -------
    C_series : (T, d, d) float64
        C_struct[t] = V_s diag(λ_s) V_s^T  (modos estruturais)
        C_bulk[t]   = V_b diag(λ_b) V_b^T  (modos de ruído)
        C_total[t]  = V   diag(λ)   V^T     (espectro completo)
    """
    if arrays["eigenvectors"] is None:
        raise ValueError(
            "eigenvectors=None — decomposição salva com store_mode='structural_only'. "
            "Use subspace='struct' com reconstruct_C_struct_series() neste caso."
        )

    evals = arrays["eigenvalues"].astype(np.float64)   # (T, d)
    evecs = arrays["eigenvectors"].astype(np.float64)  # (T, d, d): evecs[t, :, k] = k-ésimo vetor
    m_arr = arrays["m"]                                 # (T,)
    T, d  = evals.shape

    C_series = np.zeros((T, d, d), dtype=np.float64)

    for t in range(T):
        V  = evecs[t]           # (d, d)
        lm = evals[t]           # (d,)
        m  = int(m_arr[t])

        if subspace == "total":
            C_series[t] = (V * lm) @ V.T
        elif subspace == "struct":
            if m > 0:
                Vs = V[:, :m]; ls = lm[:m]
                C_series[t] = (Vs * ls) @ Vs.T
        elif subspace == "bulk":
            if m < d:
                Vb = V[:, m:]; lb = lm[m:]
                C_series[t] = (Vb * lb) @ Vb.T
        else:
            raise ValueError(f"subspace deve ser 'struct', 'bulk' ou 'total', não '{subspace}'")

    return C_series
