"""
Banco de dados SQLite para resultados da análise espectral.

Chave primária: (symbol, timeframe, step, embed_dim)
Espelha a estrutura de result_cache.py, mas para os resultados de decomp_analysis.
"""

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

_HERE   = Path(__file__).parent
DB_PATH = _HERE.parent / "analysis_results.db"

_KEY_COLS = ("symbol", "timeframe", "step", "embed_dim")

_SCALAR_COLS = (
    "n_windows",
    "beta_total",       "beta_total_err",
    "beta_struct",      "beta_struct_err",
    "beta_bulk",        "beta_bulk_err",
    "R_FDT",            "R_FDT_err",
    "D_mean",           "I_mean",
    "n_regime_changes", "n_waiting_intervals",
    "mean_waiting_time","median_waiting_time",
    "alpha",            "alpha_err",
    "entropy_mean",
    "MP_L2_relative_pct", "MP_L1_relative",
    "MP_lam_plus_emp",    "MP_lam_plus_theory",
)


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    return c


def init(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols_ddl = ",\n        ".join(f"{col} REAL" for col in _SCALAR_COLS)
    with _connect(db_path) as c:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS analysis (
                symbol      TEXT    NOT NULL,
                timeframe   TEXT    NOT NULL,
                step        INTEGER NOT NULL,
                embed_dim   INTEGER NOT NULL,
                {cols_ddl},
                computed_at TEXT    NOT NULL,
                PRIMARY KEY (symbol, timeframe, step, embed_dim)
            )
        """)


def exists(symbol: str, timeframe: str, step: int, embed_dim: int,
           db_path: Path = DB_PATH) -> bool:
    if not db_path.exists():
        return False
    with _connect(db_path) as c:
        row = c.execute(
            "SELECT 1 FROM analysis WHERE symbol=? AND timeframe=? AND step=? AND embed_dim=?",
            (symbol, timeframe, int(step), int(embed_dim)),
        ).fetchone()
    return row is not None


def save(res: dict, db_path: Path = DB_PATH) -> None:
    """Persiste os escalares de um resultado de decomp_analysis.analyze()."""
    init(db_path)
    values = (
        res["symbol"],
        res["timeframe"],
        int(res["step"]),
        int(res["embed_dim"]),
        *[res.get(col) for col in _SCALAR_COLS],
        time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    col_names    = ",".join(_KEY_COLS) + "," + ",".join(_SCALAR_COLS) + ",computed_at"
    placeholders = ",".join(["?"] * len(values))
    with _connect(db_path) as c:
        c.execute(
            f"INSERT OR REPLACE INTO analysis ({col_names}) VALUES ({placeholders})",
            values,
        )


def load(symbol: str = None, timeframe: str = None,
         db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Carrega resultados do banco como DataFrame.
    Filtra por symbol e/ou timeframe se fornecidos.
    """
    if not db_path.exists():
        return pd.DataFrame()

    query  = "SELECT * FROM analysis"
    params = []
    wheres = []
    if symbol:
        wheres.append("symbol=?");    params.append(symbol)
    if timeframe:
        wheres.append("timeframe=?"); params.append(timeframe)
    if wheres:
        query += " WHERE " + " AND ".join(wheres)
    query += " ORDER BY symbol, timeframe, step, embed_dim"

    with _connect(db_path) as c:
        rows = c.execute(query, params).fetchall()

    return pd.DataFrame([dict(r) for r in rows])


def best_step(symbol: str, timeframe: str,
              criterion: str = "mp_l2",
              db_path: Path = DB_PATH) -> int | None:
    """
    Retorna o step ótimo para um (symbol, timeframe) com base nos resultados do DB.

    criterion:
        'mp_l2'     — step com menor erro L2 no ajuste Marchenko-Pastur (bulk mais aleatório)
        'beta_snr'  — step com maior razão beta_struct / beta_struct_err (fit mais estável)
    """
    df = load(symbol, timeframe, db_path)
    if df.empty:
        return None

    if criterion == "mp_l2":
        idx = df["MP_L2_relative_pct"].idxmin()
    elif criterion == "beta_snr":
        snr = df["beta_struct"].abs() / df["beta_struct_err"].replace(0, float("nan"))
        idx = snr.idxmax()
    else:
        raise ValueError(f"criterion desconhecido: {criterion!r}")

    return int(df.loc[idx, "step"])


def best_embed(symbol: str, timeframe: str,
               criterion: str = "mp_l2",
               db_path: Path = DB_PATH) -> int | None:
    """Retorna o embed_dim ótimo. Mesmos critérios de best_step."""
    df = load(symbol, timeframe, db_path)
    if df.empty:
        return None

    if criterion == "mp_l2":
        idx = df["MP_L2_relative_pct"].idxmin()
    elif criterion == "beta_snr":
        snr = df["beta_struct"].abs() / df["beta_struct_err"].replace(0, float("nan"))
        idx = snr.idxmax()
    else:
        raise ValueError(f"criterion desconhecido: {criterion!r}")

    return int(df.loc[idx, "embed_dim"])
