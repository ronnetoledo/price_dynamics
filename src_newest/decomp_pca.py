"""
Core da decomposição espectral PCA com janela deslizante — versão otimizada.

Estratégia de cálculo (duplo batch + streaming):

  FASE 0 — double stride trick (zero-copy)
    Todos os embeddings de Hankel são construídos como uma única view de ret_C
    sem nenhuma cópia de dados:
        all_X_view[i, j, k] = ret_C[i*step + j + k]
    Forma: (n_centers, n_samp, embed_dim) — zero bytes extras além de ret_C.

  FASE 1 — batch de covariâncias (COV_BATCH janelas por vez)
    Copia COV_BATCH janelas → centraliza vetorizadamente → matmul em batch:
        C_batch = Xc.T @ Xc / (n_samp-1)     shape: (COV_BATCH, d, d)
    É uma chamada única de BLAS dgemm em batch → satura todos os cores.
    Peak de memória: COV_BATCH × n_samp × d × 8 bytes.

  FASE 2 — batch eigendecomposição (EIGH_CHUNK covariâncias por vez)
    np.linalg.eigh em batch sobre (EIGH_CHUNK, d, d) → BLAS usa todos os cores.
    Limiar MP, m(τ) e entropia calculados vetorizadamente sobre o batch.

  FASE 3 — streaming Parquet
    Após cada flush do eigh, escreve diretamente no Parquet (PyArrow writer).
    Memória constante: 2 × EIGH_CHUNK × d² × 8 bytes, independente de n_centers.
    Resolve o problema de acumulação (ex: M1 com 88k janelas → ~34 GB sem streaming).

Uso como script:
    python decomp_pca.py NVDA 1day
    python decomp_pca.py NVDA M1 --step 20 --embed 70 --mem-budget 512
    python decomp_pca.py PETR4 D1 --store-mode structural_only --overwrite
"""

# BLAS thread saturation — deve estar antes do import do numpy
import os
import sys
_N_CPU = os.cpu_count() or 1
os.environ.setdefault('OMP_NUM_THREADS',      str(_N_CPU))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_N_CPU))
os.environ.setdefault('MKL_NUM_THREADS',      str(_N_CPU))

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import decomp_io

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────────────

_HERE     = Path(__file__).parent
DATA_ROOT = _HERE.parent / "data_parquet"

_CSV_DIRS = {
    "alpaca":     _HERE.parent / "alpaca" / "SPY500_DATA",
    "metatrader": _HERE / "B3_DATA",
}

_TF_TO_PARQUET = {
    '1day': 'D1', '1d': 'D1', 'D1': 'D1',
    '1hour': 'H1', '1h': 'H1', 'H1': 'H1',
    '4hour': 'H4', '4h': 'H4', 'H4': 'H4',
    '1min':  'M1', 'M1': 'M1',
    '5min':  'M5', 'M5': 'M5',
    '15min': 'M15','M15': 'M15',
    '30min': 'M30','M30': 'M30',
}

# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(symbol: str, tf_parquet: str) -> pd.DataFrame:
    for source in ("alpaca", "metatrader"):
        base  = DATA_ROOT / f"source={source}" / f"symbol={symbol}" / f"timeframe={tf_parquet}"
        parts = sorted(base.glob("year=*/data.parquet")) if base.exists() else []
        if parts:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            return df.sort_values("ts").reset_index(drop=True)
    raise FileNotFoundError(
        f"Parquet nao encontrado: symbol={symbol} timeframe={tf_parquet} em {DATA_ROOT}"
    )


def _load_csv(symbol: str, tf_label: str) -> pd.DataFrame:
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
        f"CSV nao encontrado: {symbol}_{tf_label}.csv em {list(_CSV_DIRS.values())}"
    )


def load_ohlcv(symbol: str, tf_label: str) -> pd.DataFrame:
    """Carrega OHLCV do Parquet (prioritario) com fallback para CSV."""
    tf_parquet = _TF_TO_PARQUET.get(tf_label, tf_label)
    try:
        return _load_parquet(symbol, tf_parquet)
    except FileNotFoundError:
        return _load_csv(symbol, tf_label)


# ─────────────────────────────────────────────────────────────────────────────
# Pré-processamento
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(df: pd.DataFrame):
    """Retorna (ts[1:], ret_C) com log-retornos do close."""
    C  = df["close"].values.astype(np.float64)
    ts = pd.to_datetime(df["ts"].values)
    return ts[1:], np.diff(np.log(C))


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de batch sizes a partir do budget de memória
# ─────────────────────────────────────────────────────────────────────────────

def _batch_sizes(embed_dim: int, n_samp: int, mem_budget_mb: int):
    """
    Retorna (COV_BATCH, EIGH_CHUNK) respeitando o budget de RAM.

    Budget dividido:
        40% → Xc batch  : COV_BATCH × n_samp × d × 8 bytes
        50% → eigh buf  : EIGH_CHUNK × d² × 8 × 2 bytes (C + V float64)
        10% → folga para DataFrames e conversão float32
    """
    budget = mem_budget_mb * 1024 * 1024
    cov_bytes_per_win  = n_samp * embed_dim * 8
    eigh_bytes_per_win = embed_dim * embed_dim * 8 * 2   # C + V (float64)

    cov_batch  = max(50,  int(0.40 * budget / cov_bytes_per_win))
    eigh_chunk = max(100, int(0.50 * budget / eigh_bytes_per_win))

    return min(cov_batch, 5_000), min(eigh_chunk, 10_000)


# ─────────────────────────────────────────────────────────────────────────────
# Escrita Parquet de um chunk (helper interno)
# ─────────────────────────────────────────────────────────────────────────────

def _list_array(mat: np.ndarray) -> pa.ListArray:
    """
    Constrói um list<float32> a partir de (n, width) contíguo, sem objetos Python.

    Cada linha vira uma lista de `width` valores. Os valores compartilham o buffer
    numpy (zero-copy nos floats); só os offsets são alocados. Substitui o padrão
    `[mat[i].tolist() for i in range(n)]`, que materializava n×width floats Python
    (~24 bytes cada) e rodava single-thread — origem do pico de RAM na escrita.
    """
    n, width = mat.shape
    flat    = np.ascontiguousarray(mat, dtype=np.float32).reshape(-1)
    values  = pa.array(flat, type=pa.float32())
    offsets = pa.array(np.arange(0, (n + 1) * width, width, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, values)


def _write_eigh_chunk(writer_evals: pq.ParquetWriter,
                      writer_evecs: pq.ParquetWriter,
                      ts_chunk: np.ndarray,
                      vals: np.ndarray,
                      vecs: np.ndarray,
                      m_arr: np.ndarray,
                      lam_arr: np.ndarray,
                      entr_arr: np.ndarray,
                      embed_dim: int,
                      store_mode: str) -> None:
    """Monta as Tables PyArrow do chunk a partir de buffers numpy e escreve no Parquet."""
    n = len(ts_chunk)

    # ── eigenvalues ──────────────────────────────────────────────────────────
    writer_evals.write_table(pa.Table.from_arrays(
        [
            pa.array(ts_chunk, type=pa.timestamp("us")),
            pa.array(m_arr,    type=pa.int16()),
            pa.array(lam_arr,  type=pa.float32()),
            pa.array(entr_arr, type=pa.float32()),
            _list_array(vals),
        ],
        schema=decomp_io._SCHEMA_EVALS,
    ))

    # ── eigenvectors ─────────────────────────────────────────────────────────
    # vecs: (n, d, d) com vecs[i, :, k] = k-ésimo autovetor → transpor para
    # (n, d_modo, d) de modo que a linha [i, k] seja o autovetor do modo k.
    vt = vecs.transpose(0, 2, 1)                          # (n, d, d_modo→linha)
    mode_grid = np.broadcast_to(np.arange(embed_dim, dtype=np.int16), (n, embed_dim))

    if store_mode == "all":
        ev_vecs_mat = vt.reshape(n * embed_dim, embed_dim)
        ev_ts       = np.repeat(ts_chunk, embed_dim)
        ev_midx     = mode_grid.reshape(-1)
        ev_isstruct = ev_midx < np.repeat(m_arr, embed_dim)
    else:  # structural_only — só os m primeiros modos de cada janela (vetorizado)
        mask        = mode_grid < m_arr[:, np.newaxis]    # (n, d) bool
        flat_mask   = mask.reshape(-1)
        ev_vecs_mat = vt.reshape(n * embed_dim, embed_dim)[flat_mask]
        ev_ts       = np.repeat(ts_chunk, m_arr.astype(np.int64))
        ev_midx     = mode_grid[mask]
        ev_isstruct = np.ones(len(ev_midx), dtype=bool)

    writer_evecs.write_table(pa.Table.from_arrays(
        [
            pa.array(ev_ts,      type=pa.timestamp("us")),
            pa.array(ev_midx,    type=pa.int16()),
            pa.array(ev_isstruct, type=pa.bool_()),
            _list_array(ev_vecs_mat),
        ],
        schema=decomp_io._SCHEMA_EVECS,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal de decomposição — streaming + double batch
# ─────────────────────────────────────────────────────────────────────────────

def decompose(symbol: str, tf_label: str,
              step: int, embed_dim: int,
              dest: Path,
              window: int = None,
              store_mode: str = "all",
              mem_budget_mb: int = 256) -> dict:
    """
    Decomposição espectral com streaming Parquet e double batch (cov + eigh).

    Escreve diretamente em `dest/eigenvalues.parquet` e `dest/eigenvectors.parquet`.
    Retorna dict com estatísticas da execução.

    Parâmetros
    ----------
    dest        : diretório de destino dos Parquet (criado por run())
    store_mode  : 'all' | 'structural_only'
    mem_budget_mb : orçamento de RAM em MB (controla COV_BATCH e EIGH_CHUNK)
    """
    if window is None:
        window = 5 * embed_dim

    df        = load_ohlcv(symbol, tf_label)
    ts, ret_C = compute_log_returns(df)

    n_samp    = window - embed_dim          # linhas em cada X (constante)
    centers   = np.arange(window, len(ret_C) - window, step, dtype=np.int64)
    n_centers = len(centers)

    if n_centers == 0:
        raise ValueError(
            f"Serie muito curta para window={window}, step={step}: len={len(ret_C)}"
        )

    COV_BATCH, EIGH_CHUNK = _batch_sizes(embed_dim, n_samp, mem_budget_mb)
    q = embed_dim / n_samp   # razão aspecto MP — constante

    print(f"  {symbol} | {tf_label} | step={step} embed={embed_dim} "
          f"window={window} | {n_centers} janelas")
    print(f"  COV_BATCH={COV_BATCH}  EIGH_CHUNK={EIGH_CHUNK}  "
          f"budget={mem_budget_mb}MB  store={store_mode}")

    # ── FASE 0: double stride trick — view zero-copy de todos os embeddings ──
    # all_X_view[i, j, k] = ret_C[i*step + j + k]
    # Shape: (n_centers, n_samp, embed_dim)  ← zero bytes extras além de ret_C
    s = ret_C.strides[0]
    all_X_view = np.lib.stride_tricks.as_strided(
        ret_C,
        shape=(n_centers, n_samp, embed_dim),
        strides=(step * s, s, s),
    )

    # Timestamps por janela
    ts_np = ts.values.astype("datetime64[us]")
    ts_centers = ts_np[centers]   # (n_centers,)

    # ── PyArrow writers para streaming ───────────────────────────────────────
    dest.mkdir(parents=True, exist_ok=True)
    writer_evals = pq.ParquetWriter(dest / "eigenvalues.parquet",
                                    decomp_io._SCHEMA_EVALS, compression="snappy")
    writer_evecs = pq.ParquetWriter(dest / "eigenvectors.parquet",
                                    decomp_io._SCHEMA_EVECS, compression="snappy")

    # ── Buffer do eigh — pré-alocado, reusado entre flushes ─────────────────
    C_buf  = np.empty((EIGH_CHUNK, embed_dim, embed_dim), dtype=np.float64)
    v_buf  = np.empty(EIGH_CHUNK, dtype=np.float64)
    ts_buf = np.empty(EIGH_CHUNK, dtype="datetime64[us]")
    buf_n  = 0       # janelas acumuladas no buffer atual
    n_written = 0    # total de janelas escritas no Parquet

    def _flush_eigh(n: int) -> None:
        """Processa n janelas do buffer e escreve no Parquet."""
        nonlocal n_written

        Ca = C_buf[:n]
        va = v_buf[:n]
        ts_chunk = ts_buf[:n]

        # eigendecomposição em batch — BLAS usa todos os cores aqui
        av, ve = np.linalg.eigh(Ca)
        vals = av[:, ::-1].astype(np.float32)   # descrescente, float32
        vecs = ve[:, :, ::-1].astype(np.float32)
        del av, ve

        # limiar MP e m(τ) — vetorizados
        lam  = (va * (1.0 + np.sqrt(q)) ** 2).astype(np.float32)
        m    = np.sum(vals > lam[:, np.newaxis], axis=1).astype(np.int16)

        # entropia espectral — vetorizada
        vp   = np.maximum(vals, 1e-15)
        p    = vp / vp.sum(axis=1, keepdims=True)
        entr = (-np.sum(p * np.log(p), axis=1) / np.log(embed_dim)).astype(np.float32)
        del vp, p

        _write_eigh_chunk(writer_evals, writer_evecs,
                          ts_chunk, vals, vecs, m, lam, entr,
                          embed_dim, store_mode)
        del vals, vecs, lam, m, entr
        n_written += n

    # ── loop principal: COV batches → eigh buffer → flush ────────────────────
    t0            = time.time()
    n_cov_batches = 0
    n_eigh_flushes = 0

    for cov_start in range(0, n_centers, COV_BATCH):
        cov_end = min(cov_start + COV_BATCH, n_centers)
        batch   = cov_end - cov_start

        # FASE 1: batch de covariâncias
        # Xc: cópia necessária (view stride_tricks é read-only por convenção)
        Xc = all_X_view[cov_start:cov_end].copy()        # (batch, n_samp, d)
        Xc -= Xc.mean(axis=1, keepdims=True)             # centra in-place, vetorizado

        # Batched matmul: (batch, d, n_samp) × (batch, n_samp, d) → (batch, d, d)
        # Esta é a chamada que satura o BLAS com múltiplos cores
        C_batch   = Xc.transpose(0, 2, 1) @ Xc / (n_samp - 1)

        # Variância empírica por janela (para limiar MP)
        var_batch = Xc.var(axis=1, ddof=1).mean(axis=-1)   # (batch,)
        ts_batch  = ts_centers[cov_start:cov_end]           # (batch,)
        del Xc

        # FASE 2: alimenta o buffer do eigh, flushing quando cheio
        written = 0
        while written < batch:
            space = EIGH_CHUNK - buf_n
            take  = min(space, batch - written)
            C_buf [buf_n:buf_n+take] = C_batch[written:written+take]
            v_buf [buf_n:buf_n+take] = var_batch[written:written+take]
            ts_buf[buf_n:buf_n+take] = ts_batch[written:written+take]
            buf_n   += take
            written += take

            if buf_n >= EIGH_CHUNK:
                _flush_eigh(EIGH_CHUNK)
                n_eigh_flushes += 1
                buf_n = 0

        del C_batch, var_batch
        n_cov_batches += 1

        # progresso
        elapsed = time.time() - t0
        speed   = n_written / elapsed if elapsed > 0 else 0
        pct     = 100.0 * (cov_end) / n_centers
        eta     = (n_centers - cov_end) / speed if speed > 0 else float("inf")
        filled  = int(pct / 5)
        bar     = '#' * filled + '.' * (20 - filled)
        print(f"    [{bar}] {pct:5.1f}%  "
              f"cov={n_cov_batches}  eigh={n_eigh_flushes}  "
              f"{elapsed:.1f}s  ETA {eta:.0f}s  {speed:.0f} jan/s",
              flush=True)

    # flush do buffer parcial final
    if buf_n > 0:
        _flush_eigh(buf_n)

    writer_evals.close()
    writer_evecs.close()
    gc.collect()

    elapsed = time.time() - t0
    print(f"  concluido: {n_written} janelas | {elapsed:.1f}s "
          f"({1000*elapsed/n_written:.1f} ms/jan)  "
          f"cov_batches={n_cov_batches}  eigh_flushes={n_eigh_flushes+1}")

    return {"n_windows": n_written, "elapsed_s": elapsed,
            "cov_batches": n_cov_batches, "eigh_flushes": n_eigh_flushes + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Orquestrador
# ─────────────────────────────────────────────────────────────────────────────

def run(symbol: str, tf_label: str,
        step: int = 20, embed_dim: int = 70,
        window: int = None,
        store_mode: str = "all",
        mem_budget_mb: int = 256,
        overwrite: bool = False,
        decomp_root: Path = None) -> None:
    """Orquestra decomposição e persistência. Pula se já existir e overwrite=False."""
    root       = decomp_root or decomp_io.DECOMP_ROOT
    tf_parquet = _TF_TO_PARQUET.get(tf_label, tf_label)

    if not overwrite and decomp_io.exists(symbol, tf_parquet, step, embed_dim, root):
        print(f"  [skip] {symbol} {tf_label} step={step} embed={embed_dim} "
              f"-- parquet ja existe")
        return

    dest = decomp_io._partition_path(root, symbol, tf_parquet, step, embed_dim)

    decompose(symbol, tf_label, step, embed_dim,
              dest=dest, window=window,
              store_mode=store_mode, mem_budget_mb=mem_budget_mb)

    print(f"  [salvo] {symbol}/{tf_parquet}/step={step}/embed={embed_dim}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Decomposicao espectral PCA de series financeiras -> Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("symbol",    help="Ticker (ex: NVDA, PETR4)")
    p.add_argument("timeframe", help="Timeframe (ex: 1day, D1, H1, M1)")
    p.add_argument("--step",        type=int, default=20)
    p.add_argument("--embed",       type=int, default=70)
    p.add_argument("--window",      type=int, default=None)
    p.add_argument("--store-mode",  choices=["all", "structural_only"], default="all")
    p.add_argument("--mem-budget",  type=int, default=256,
                   help="Orcamento de RAM para buffers de chunk (MB)")
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        symbol        = args.symbol,
        tf_label      = args.timeframe,
        step          = args.step,
        embed_dim     = args.embed,
        window        = args.window,
        store_mode    = args.store_mode,
        mem_budget_mb = args.mem_budget,
        overwrite     = args.overwrite,
    )
