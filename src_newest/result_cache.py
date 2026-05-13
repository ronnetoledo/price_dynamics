"""
Cache local de resultados: SQLite (escalares) + HDF5 (arrays numpy).

Chave de cache: (asset, tf_file, data_hash, mode, step, embed_dim, window)
  data_hash — MD5 do tamanho+mtime do CSV de entrada; invalida cache se o dado mudar.
  mode      — 'step' | 'embed' | 'single'
"""

import os
import sqlite3
import hashlib
import time
import numpy as np

try:
    import h5py
    _HDF5 = True
except ImportError:
    _HDF5 = False
    print("[result_cache] AVISO: h5py não encontrado — arrays não serão persistidos no cache.")

_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_DB_PATH = os.path.join(_DIR, "cache.db")
_H5_PATH = os.path.join(_DIR, "cache.h5")

# Colunas escalares (sem as colunas da chave primária)
_SCALAR_COLS = (
    "n_regime_changes",
    "n_waiting_intervals",
    "mean_waiting_time",
    "median_waiting_time",
    "alpha",
    "alpha_err",
    "R_FDT",
    "R_FDT_err",
    "beta_total",
    "beta_total_err",
    "beta_struct",
    "beta_struct_err",
    "beta_bulk",
    "beta_bulk_err",
    "MP_L2_relative_pct",
    "entropy_mean",
)


def init():
    """Cria diretório e tabelas SQLite se não existirem."""
    os.makedirs(_DIR, exist_ok=True)
    with _connect() as c:
        cols_ddl = ",\n                ".join(f"{col} REAL" for col in _SCALAR_COLS)
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS results (
                asset        TEXT    NOT NULL,
                tf_file      TEXT    NOT NULL,
                data_hash    TEXT    NOT NULL,
                mode         TEXT    NOT NULL,
                step         INTEGER NOT NULL,
                embed_dim    INTEGER NOT NULL,
                window       INTEGER NOT NULL,
                {cols_ddl},
                computed_at  TEXT    NOT NULL,
                PRIMARY KEY (asset, tf_file, data_hash, mode, step, embed_dim, window)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS best_params (
                asset        TEXT    NOT NULL,
                param_type   TEXT    NOT NULL,
                value        INTEGER NOT NULL,
                computed_at  TEXT    NOT NULL,
                PRIMARY KEY (asset, param_type)
            )
        """)


def _connect():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def file_hash(path):
    """Hash rápido baseado em caminho absoluto + tamanho + mtime do arquivo."""
    stat = os.stat(path)
    tag  = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(tag.encode()).hexdigest()


def _h5key(asset, tf_file, mode, step, embed_dim, window):
    tf_base = os.path.splitext(os.path.basename(tf_file))[0]
    return f"/{asset}/{tf_base}/{mode}/s{step}_e{embed_dim}_w{window}"


def _coerce(v):
    """Converte None (NULL do SQLite) para nan e preserva floats."""
    return float("nan") if v is None else float(v)


def lookup(asset, tf_file, data_hash, mode, step, embed_dim, window):
    """
    Retorna (scalars_dict, arrays_dict) se houver cache hit, ou None.

    scalars_dict — todos os campos da tabela results como floats (None→nan).
    arrays_dict  — {'tau_sorted', 'ccdf', 'beta_lags', 'beta_values', 'beta_err'}
                   como np.ndarray; ausentes ficam como None.
    """
    if not os.path.exists(_DB_PATH):
        return None

    with _connect() as c:
        row = c.execute("""
            SELECT * FROM results
            WHERE asset=? AND tf_file=? AND data_hash=?
              AND mode=? AND step=? AND embed_dim=? AND window=?
        """, (asset, tf_file, data_hash, mode,
              int(step), int(embed_dim), int(window))).fetchone()

    if row is None:
        return None

    scalars = {k: _coerce(row[k]) if k in _SCALAR_COLS else row[k]
               for k in row.keys()}

    arrays = {}
    if _HDF5 and os.path.exists(_H5_PATH):
        key = _h5key(asset, tf_file, mode, step, embed_dim, window)
        try:
            with h5py.File(_H5_PATH, "r") as hf:
                if key in hf:
                    grp = hf[key]
                    for name in grp:
                        arrays[name] = grp[name][:]
        except Exception:
            pass

    return scalars, arrays


def save(asset, tf_file, data_hash, mode, step, embed_dim, window, scalars, arrays):
    """
    Persiste escalares no SQLite e arrays no HDF5.

    scalars — dict com ao menos as chaves de _SCALAR_COLS (valores extras são ignorados).
    arrays  — dict {nome: np.ndarray | None}; None e arrays vazios são pulados.
    """
    os.makedirs(_DIR, exist_ok=True)

    values = (
        asset, tf_file, data_hash, mode, int(step), int(embed_dim), int(window),
        *[scalars.get(col) for col in _SCALAR_COLS],
        time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    placeholders = ",".join(["?"] * len(values))
    col_names    = ("asset,tf_file,data_hash,mode,step,embed_dim,window,"
                    + ",".join(_SCALAR_COLS)
                    + ",computed_at")

    with _connect() as c:
        c.execute(
            f"INSERT OR REPLACE INTO results ({col_names}) VALUES ({placeholders})",
            values,
        )

    if not _HDF5 or not arrays:
        return

    key = _h5key(asset, tf_file, mode, step, embed_dim, window)
    try:
        with h5py.File(_H5_PATH, "a") as hf:
            if key in hf:
                del hf[key]
            grp = hf.require_group(key)
            for name, val in arrays.items():
                if val is None:
                    continue
                try:
                    arr = np.asarray(val, dtype=float)
                    if arr.size > 0:
                        grp.create_dataset(
                            name, data=arr,
                            compression="gzip", compression_opts=4,
                        )
                except Exception:
                    pass
    except Exception:
        pass


def save_best_param(asset, param_type, value):
    """Persiste um parâmetro ótimo (ex: 'step', 'embed_dim') para um ativo."""
    os.makedirs(_DIR, exist_ok=True)
    with _connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO best_params (asset, param_type, value, computed_at)"
            " VALUES (?,?,?,?)",
            (asset, param_type, int(value), time.strftime("%Y-%m-%dT%H:%M:%S")),
        )


def load_best_param(asset, param_type):
    """
    Retorna o valor ótimo cacheado ou None se não houver entrada.

    Fallback automático: quando best_params está vazio (primeira execução com
    código novo), deriva o valor a partir das entradas já existentes em results:
      'step'      → step mais frequente nas entradas mode='embed' deste ativo
      'embed_dim' → embed_dim mais frequente nas entradas mode='single' deste ativo
    """
    if not os.path.exists(_DB_PATH):
        return None
    with _connect() as c:
        row = c.execute(
            "SELECT value FROM best_params WHERE asset=? AND param_type=?",
            (asset, param_type),
        ).fetchone()
    if row:
        return int(row["value"])

    # Fallback: deriva de entradas existentes na tabela results
    with _connect() as c:
        if param_type == "step":
            row = c.execute(
                """SELECT step, COUNT(*) AS cnt FROM results
                   WHERE asset=? AND mode='embed'
                   GROUP BY step ORDER BY cnt DESC LIMIT 1""",
                (asset,),
            ).fetchone()
            col = "step"
        elif param_type == "embed_dim":
            row = c.execute(
                """SELECT embed_dim, COUNT(*) AS cnt FROM results
                   WHERE asset=? AND mode='single'
                   GROUP BY embed_dim ORDER BY cnt DESC LIMIT 1""",
                (asset,),
            ).fetchone()
            col = "embed_dim"
        else:
            return None

    if row:
        value = int(row[col])
        # persiste para evitar re-derivação nas próximas execuções
        save_best_param(asset, param_type, value)
        return value
    return None
