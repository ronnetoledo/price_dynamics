"""
Converte arquivos CSV de dados OHLCV para formato Parquet particionado.

Suporta duas fontes:
  - MetaTrader : separador ';', UTF-16 BOM, ts='YYYY.MM.DD HH:MM:SS'
  - Alpaca     : separador ',', UTF-8,      ts='YYYY-MM-DD HH:MM:SS+00:00'

Estrutura de saída:
  <PARQUET_ROOT>/
  └── source=<fonte>/
      └── symbol=<ATIVO>/
          └── timeframe=<TF>/
              └── year=<ANO>/
                  └── data.parquet

Uso:
  python csv_to_parquet.py                      # processa DATA_DIRS padrão
  python csv_to_parquet.py --dirs path1 path2   # diretórios customizados
  python csv_to_parquet.py --file path/arq.csv  # arquivo único
"""

import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── configuração ──────────────────────────────────────────────────────────────

PARQUET_ROOT = Path(__file__).parent.parent / "data_parquet"

# Diretórios varridos por padrão
DATA_DIRS = [
    #Path(__file__).parent.parent / "src_old"  / "data",
    Path(__file__).parent.parent / "alpaca"   / "SPY500_DATA",
    #Path(__file__).parent        / "data",
]

# Normaliza nomes de timeframe para um padrão único
TF_ALIASES = {
    # Alpaca
    "1min":  "M1",  "1minute": "M1",
    "5min":  "M5",  "5minute": "M5",
    "15min": "M15", "15minute": "M15",
    "30min": "M30", "30minute": "M30",
    "1hour": "H1",  "1h": "H1",
    "4hour": "H4",  "4h": "H4",
    "1day":  "D1",  "1d": "D1",
    # MetaTrader já usa M1/H1/D1 — passam direto
}

# ── detecção de fonte ─────────────────────────────────────────────────────────

def _sniff_encoding(path: Path) -> str:
    """Detecta UTF-16 BOM; caso contrário assume UTF-8."""
    with open(path, "rb") as f:
        bom = f.read(2)
    return "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8"


def _sniff_source(path: Path) -> str:
    """Detecta 'alpaca' ou 'metatrader' pela primeira linha do arquivo."""
    enc = _sniff_encoding(path)
    with open(path, encoding=enc, errors="replace") as f:
        header = f.readline().strip()
    # remove BOM residual se houver
    header = header.lstrip("﻿").strip()
    cols = {c.strip().lower() for c in re.split(r"[;,]", header)}
    if "trade_count" in cols or "vwap" in cols:
        return "alpaca"
    if "tick_volume" in cols or "real_volume" in cols:
        return "metatrader"
    # fallback: tenta pelo separador
    return "alpaca" if "," in header else "metatrader"


def _parse_filename(path: Path) -> tuple[str, str]:
    """
    Extrai (symbol, timeframe) do nome do arquivo.
    Exemplos reconhecidos:
      PETR4_H1.csv      → ('PETR4', 'H1')
      SPY_1hour.csv     → ('SPY', 'H1')
      NVDA_1min.csv     → ('NVDA', 'M1')
      BRK-B_1day.csv    → ('BRK-B', 'D1')
    """
    stem = path.stem  # nome sem extensão
    m = re.match(r'^([A-Z0-9$.\-]+?)_([A-Za-z0-9]+)$', stem, re.IGNORECASE)
    if not m:
        return None, None
    symbol    = m.group(1).upper()
    tf_raw    = m.group(2).lower()
    timeframe = TF_ALIASES.get(tf_raw, tf_raw.upper())
    return symbol, timeframe


# ── leitura e normalização ────────────────────────────────────────────────────

_MT_TS_FMT  = "%Y.%m.%d %H:%M:%S"   # MetaTrader: 2016.02.29 10:00:00
_ALP_TS_FMT = None                   # Alpaca: ISO 8601 — deixa pandas inferir

def _load_metatrader(path: Path) -> pd.DataFrame:
    enc = _sniff_encoding(path)
    df = pd.read_csv(path, sep=";", encoding=enc,
                     skipinitialspace=True, dtype=str)
    df.columns = [c.strip().lstrip("﻿").lower() for c in df.columns]

    # coluna de tempo pode ser 'time' ou 'timestamp'
    ts_col = next((c for c in df.columns if c in ("time", "timestamp", "date")), None)
    if ts_col is None:
        raise ValueError(f"Coluna de timestamp não encontrada em {path.name}: {df.columns.tolist()}")

    df["ts"] = pd.to_datetime(df[ts_col].str.strip(), format=_MT_TS_FMT)
    # MetaTrader não tem timezone — assume horário local do broker (sem tz)

    vol_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
    out = pd.DataFrame({
        "ts":           df["ts"],
        "open":         pd.to_numeric(df["open"],  errors="coerce"),
        "high":         pd.to_numeric(df["high"],  errors="coerce"),
        "low":          pd.to_numeric(df["low"],   errors="coerce"),
        "close":        pd.to_numeric(df["close"], errors="coerce"),
        "volume":       pd.to_numeric(df[vol_col], errors="coerce"),
        "tick_volume":  pd.to_numeric(df.get("tick_volume", pd.Series(dtype=float)),
                                      errors="coerce"),
        "trade_count":  np.full(len(df), np.nan),
        "vwap":         np.full(len(df), np.nan),
    })
    return out


def _load_alpaca(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", encoding="utf-8", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts"] = df["ts"].dt.tz_localize(None)   # remove tz → naive UTC

    out = pd.DataFrame({
        "ts":           df["ts"],
        "open":         pd.to_numeric(df["open"],   errors="coerce"),
        "high":         pd.to_numeric(df["high"],   errors="coerce"),
        "low":          pd.to_numeric(df["low"],    errors="coerce"),
        "close":        pd.to_numeric(df["close"],  errors="coerce"),
        "volume":       pd.to_numeric(df["volume"], errors="coerce"),
        "tick_volume":  np.full(len(df), np.nan),
        "trade_count":  pd.to_numeric(df.get("trade_count",
                                              pd.Series(dtype=str)),
                                      errors="coerce"),
        "vwap":         pd.to_numeric(df.get("vwap",
                                             pd.Series(dtype=str)),
                                      errors="coerce"),
    })
    return out


def load_csv(path: Path, source: str) -> pd.DataFrame:
    """Carrega e normaliza um CSV para o schema unificado."""
    df = _load_metatrader(path) if source == "metatrader" else _load_alpaca(path)
    df = (df
          .dropna(subset=["ts", "open", "high", "low", "close"])
          .drop_duplicates("ts")
          .sort_values("ts")
          .reset_index(drop=True))
    return df


# ── escrita particionada ──────────────────────────────────────────────────────

_SCHEMA = pa.schema([
    ("ts",          pa.timestamp("us")),
    ("open",        pa.float64()),
    ("high",        pa.float64()),
    ("low",         pa.float64()),
    ("close",       pa.float64()),
    ("volume",      pa.float64()),
    ("tick_volume", pa.float64()),
    ("trade_count", pa.float64()),
    ("vwap",        pa.float64()),
])


def _parquet_path(root: Path, source: str,
                  symbol: str, timeframe: str, year: int) -> Path:
    return (root
            / f"source={source}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"year={year}"
            / "data.parquet")


def write_parquet(df: pd.DataFrame, source: str,
                  symbol: str, timeframe: str,
                  root: Path = PARQUET_ROOT) -> dict:
    """
    Grava o DataFrame particionado por ano.
    Faz merge com arquivo existente (deduplicação automática).
    Retorna estatísticas: {year: {'new': N, 'existing': M}}.
    """
    stats = {}
    df["_year"] = df["ts"].dt.year

    for year, group in df.groupby("_year"):
        group = group.drop(columns=["_year"]).reset_index(drop=True)
        dest  = _parquet_path(root, source, symbol, timeframe, year)
        dest.parent.mkdir(parents=True, exist_ok=True)

        existing_rows = 0
        if dest.exists():
            existing = pd.read_parquet(dest)
            existing_rows = len(existing)
            group = (pd.concat([existing, group], ignore_index=True)
                       .drop_duplicates("ts")
                       .sort_values("ts")
                       .reset_index(drop=True))

        table = pa.Table.from_pandas(group, schema=_SCHEMA, preserve_index=False)
        pq.write_table(table, dest, compression="snappy")
        stats[year] = {"written": len(group), "existing": existing_rows,
                       "new": len(group) - existing_rows}
    return stats


# ── entry point ───────────────────────────────────────────────────────────────

def process_file(path: Path, root: Path = PARQUET_ROOT) -> bool:
    """Processa um único arquivo CSV. Retorna True em sucesso."""
    symbol, timeframe = _parse_filename(path)
    if not symbol:
        print(f"  [skip]  {path.name:40s}  nome não reconhecido")
        return False

    source = _sniff_source(path)
    try:
        df    = load_csv(path, source)
        stats = write_parquet(df, source, symbol, timeframe, root)
    except Exception as e:
        print(f"  [ERRO]  {path.name:40s}  {e}")
        return False

    total_new = sum(v["new"] for v in stats.values())
    total_wri = sum(v["written"] for v in stats.values())
    years     = sorted(stats)
    yr_str    = f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])
    print(f"  [ok]    {path.name:40s}  "
          f"source={source:11s} tf={timeframe:4s}  "
          f"{yr_str}  +{total_new:>7} novas  total={total_wri:>8}")
    return True


def process_dirs(dirs: list[Path], root: Path = PARQUET_ROOT):
    """Varre diretórios e converte todos os CSVs encontrados."""
    files = []
    for d in dirs:
        if not d.exists():
            print(f"  [aviso] diretório não encontrado: {d}")
            continue
        files.extend(sorted(d.glob("*.csv")))

    if not files:
        print("Nenhum arquivo CSV encontrado.")
        return

    print(f"\nConvertendo {len(files)} arquivo(s) → {root}\n")
    ok  = sum(process_file(f, root) for f in files)
    print(f"\nConcluído: {ok}/{len(files)} arquivos convertidos.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Converte CSVs OHLCV para Parquet")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--dirs", nargs="+", type=Path,
                     help="Diretórios com CSVs (padrão: DATA_DIRS)")
    grp.add_argument("--file", type=Path,
                     help="Arquivo CSV único")
    p.add_argument("--out", type=Path, default=PARQUET_ROOT,
                   help=f"Raiz de saída Parquet (padrão: {PARQUET_ROOT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.file:
        process_file(args.file, args.out)
    else:
        dirs = [Path(d) for d in args.dirs] if args.dirs else DATA_DIRS
        process_dirs(dirs, args.out)
