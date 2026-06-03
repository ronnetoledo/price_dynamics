"""
Banco de dados SQLite de splits (desdobramentos/grupamentos) de ações.

Chave primária: (symbol, ex_date)
Espelha o estilo de analysis_db.py. Alimentado por fetch_splits.py (Alpaca
Corporate Actions API + detecção cruzada na própria série) e consumido por
apply_splits.py para o back-adjustment dos parquet.

Convenção de razão e fator:
    factor = ratio_num / ratio_den = preço_pós / preço_pré
    forward split 4:1  -> num=4, den=1 -> factor=4.0   (preço cai ~4x no ex_date)
    reverse  split 1:10-> num=1, den=10-> factor=0.1   (preço sobe ~10x)

Back-adjustment: um bar em data d é dividido pelo PRODUTO dos factors de todos
os splits com ex_date > d (os que ainda estão no "futuro" daquele bar), e seu
volume é multiplicado pelo mesmo produto.
"""

import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_HERE   = Path(__file__).parent
DB_PATH = _HERE.parent / "splits.db"

_KEY_COLS = ("symbol", "ex_date")
_COLS = (
    "ratio_num",        # numerador da razão
    "ratio_den",        # denominador da razão
    "factor",           # ratio_num/ratio_den (preço pós/pré)
    # ── metadados crus da Alpaca Corporate Actions API ──
    "ca_type",          # corporate_action_type: forward_splits | reverse_splits
    "ca_id",            # UUID do evento na Alpaca (chave canônica)
    "cusip",            # CUSIP do papel
    "process_date",     # datas do ciclo do evento (ISO), como vêm da API
    "record_date",
    "payable_date",
    "due_bill_redemption_date",
    # ── campos nossos (verificação) ──
    "source",           # 'alpaca_ca' | 'detected' | 'manual'
    "detected_logret",  # log-ret close-to-close observado no ex_date (auditoria)
    "confirmed",        # 1 se API e detecção concordam
    "note",             # discrepâncias / observações
)


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    return c


def init(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS splits (
                symbol          TEXT    NOT NULL,
                ex_date         TEXT    NOT NULL,   -- ISO 'YYYY-MM-DD'
                ratio_num       REAL    NOT NULL,
                ratio_den       REAL    NOT NULL,
                factor          REAL    NOT NULL,
                ca_type         TEXT,               -- metadados crus da Alpaca CA API
                ca_id           TEXT,
                cusip           TEXT,
                process_date    TEXT,
                record_date     TEXT,
                payable_date    TEXT,
                due_bill_redemption_date TEXT,
                source          TEXT    NOT NULL,
                detected_logret REAL,
                confirmed       INTEGER NOT NULL DEFAULT 0,
                note            TEXT,
                updated_at      TEXT    NOT NULL,
                PRIMARY KEY (symbol, ex_date)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS ix_splits_symbol ON splits(symbol)")


def _norm_date(d) -> str:
    """Normaliza date/datetime/str para 'YYYY-MM-DD'."""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    return pd.Timestamp(d).strftime("%Y-%m-%d")


_META_KEYS = ("ca_type", "ca_id", "cusip", "process_date",
              "record_date", "payable_date", "due_bill_redemption_date")


def upsert_split(symbol: str, ex_date, ratio_num: float, ratio_den: float,
                 source: str, detected_logret: float = None,
                 confirmed: int = 0, note: str = "", meta: dict = None,
                 db_path: Path = DB_PATH) -> None:
    """Insere/atualiza um split. factor é derivado de num/den.

    meta: dict opcional com os metadados crus da API (chaves em _META_KEYS).
    """
    init(db_path)
    ex = _norm_date(ex_date)
    factor = float(ratio_num) / float(ratio_den)
    meta = meta or {}
    values = (
        symbol, ex,
        float(ratio_num), float(ratio_den), factor,
        *[meta.get(k) for k in _META_KEYS],
        source,
        None if detected_logret is None else float(detected_logret),
        int(confirmed), note,
        time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    col_names    = ",".join(_KEY_COLS) + "," + ",".join(_COLS) + ",updated_at"
    placeholders = ",".join(["?"] * len(values))
    with _connect(db_path) as c:
        c.execute(
            f"INSERT OR REPLACE INTO splits ({col_names}) VALUES ({placeholders})",
            values,
        )


def get_splits(symbol: str, db_path: Path = DB_PATH,
               confirmed_only: bool = False) -> pd.DataFrame:
    """Splits de um símbolo, ordenados por ex_date crescente."""
    if not db_path.exists():
        return pd.DataFrame()
    query  = "SELECT * FROM splits WHERE symbol=?"
    params = [symbol]
    if confirmed_only:
        query += " AND confirmed=1"
    query += " ORDER BY ex_date"
    with _connect(db_path) as c:
        rows = c.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def list_all(db_path: Path = DB_PATH, confirmed_only: bool = False) -> pd.DataFrame:
    """Todos os splits da base."""
    if not db_path.exists():
        return pd.DataFrame()
    query = "SELECT * FROM splits"
    if confirmed_only:
        query += " WHERE confirmed=1"
    query += " ORDER BY symbol, ex_date"
    with _connect(db_path) as c:
        rows = c.execute(query).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def symbols_with_splits(db_path: Path = DB_PATH,
                        confirmed_only: bool = True) -> list[str]:
    """Lista de símbolos que têm ao menos um split (confirmado, por padrão)."""
    if not db_path.exists():
        return []
    query = "SELECT DISTINCT symbol FROM splits"
    if confirmed_only:
        query += " WHERE confirmed=1"
    query += " ORDER BY symbol"
    with _connect(db_path) as c:
        return [r[0] for r in c.execute(query).fetchall()]


def cumulative_factor(symbol: str, ts, db_path: Path = DB_PATH,
                      confirmed_only: bool = True) -> float:
    """
    Fator de back-adjustment para um bar em `ts`: produto dos factors de todos
    os splits com ex_date ESTRITAMENTE > data(ts). Bars no/depois do ex_date
    não são ajustados (já estão a preço pós-split).

        preço_ajustado  = preço_bruto  / cumulative_factor
        volume_ajustado = volume_bruto * cumulative_factor
    """
    df = get_splits(symbol, db_path, confirmed_only=confirmed_only)
    if df.empty:
        return 1.0
    bar = _norm_date(ts)
    fut = df[df["ex_date"] > bar]
    if fut.empty:
        return 1.0
    return float(fut["factor"].prod())
