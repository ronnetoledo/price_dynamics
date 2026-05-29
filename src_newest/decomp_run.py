"""
Pipeline completo: decomposição PCA + análise estatística para múltiplos ativos.

Espelha a estrutura e as variáveis globais de experimento_PCA_4D_3.py.
Para cada (ativo × timeframe × step × embed):
  1. decomp_pca.run()          → garante parquet da decomposição (skip se já existir)
  2. decomp_analysis.analyze() → calcula beta, FDT, alpha, MP, entropia
  3. analysis_db.save()        → persiste escalares no SQLite
  4. CSV acumulativo da run    → snapshot por execução

Estratégia de CPU:
  N_WORKERS = 1  (padrão): BLAS usa todos os cores internamente via chunked eigh.
                            Ideal para TFs com muitas janelas (H1, M5...).
  N_WORKERS > 1:           ProcessPoolExecutor paraleliza assets. Cada worker
                            recebe cpu_count//N_WORKERS threads BLAS.
                            Ideal para D1 (~100 janelas/asset, chunk pequeno).

Uso:
    python decomp_run.py                          # configurações padrão
    python decomp_run.py --no-plots               # pula geração de PDFs
    python decomp_run.py --workers 4              # 4 assets em paralelo
    python decomp_run.py --overwrite-analysis     # refaz análise mesmo se no DB
    python decomp_run.py --overwrite-decomp       # refaz decomposição mesmo se parquet existir
"""

# BLAS thread saturation — deve estar antes do import do numpy
import os
import sys
_N_CPU = os.cpu_count() or 1
os.environ.setdefault('OMP_NUM_THREADS',      str(_N_CPU))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_N_CPU))
os.environ.setdefault('MKL_NUM_THREADS',      str(_N_CPU))

import time
import argparse
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages

import decomp_pca
import decomp_analysis
import analysis_db

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES — espelha experimento_PCA_4D_3.py
# ─────────────────────────────────────────────────────────────────────────────

ASSET_LIST = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "META", "TSLA",
    "WMT", "BRK-B", "LLY", "JPM", "MU", "AMD", "XOM", "V", "ORCL", "INTC",
    "JNJ", "COST", "MA", "CAT", "BAC", "NFLX", "LRCX", "ABBV", "CSCO", "CVX",
    "PG", "KO", "AMAT", "UNH", "PLTR", "GE", "HD", "MS", "GEV", "MRK", "GS",
    "PM", "TXN", "WFC", "RTX", "KLAC", "LIN", "AXP", "C", "IBM", "PEP",
    "TMUS", "SNDK", "MCD", "ADI", "QCOM", "NEE", "VZ", "DIS", "ANET", "BA",
    "AMGN", "T", "TMO", "STX", "TJX", "APH", "GILD", "BLK", "WDC", "ETN",
    "UBER", "GLW", "ISRG", "SCHW", "DE", "UNP", "APP", "PANW", "BX", "DELL",
    "WELL", "PFE", "CRM", "ABT", "COP", "HON", "VRT", "PLD", "LOW", "BKNG",
    "NEM", "SPGI", "CB", "DHR", "COF", "CRWD", "SBUX", "LMT", "CEG", "PWR",
    "MO", "BMY", "PGR", "PH", "HWM", "SYK", "CVS", "INTU", "TT", "VRTX",
    "EQIX", "ACN", "SO", "CME", "ADBE", "MDT", "CMI", "CDNS", "DUK", "SNPS",
    "HCA", "MAR", "NOW", "CMCSA", "GD", "MCK", "BK", "FDX", "PNC", "KKR",
    "FCX", "WMB", "WM", "JCI", "USB", "ICE", "UPS", "CSX", "AMT", "ABNB",
    "BSX", "EMR", "ADP", "SLB", "SHW", "CIEN", "MPWR", "ELV", "DASH", "NOC",
    "MDLZ", "ORLY", "MRSH", "MCO", "CRH", "RCL", "MMM", "MNST", "FTNT",
    "NXPI", "ITW", "REGN", "ROST", "ECL", "CI", "APO", "HLT", "MSI", "AEP",
    "LITE", "FIX", "GM", "MPC", "HOOD", "CL", "EOG", "DLR", "KMI", "VLO",
    "TDG", "NSC", "CTAS", "PSX", "WBD", "DDOG", "APD", "SPG", "AON", "BKR",
    "NKE", "TRV", "TEL", "TFC", "KEYS", "RSG", "COHR", "SRE", "URI", "PCAR",
    "GWW", "O", "TER", "TGT", "AZO", "AFL", "CARR", "LHX", "VST", "CVNA",
    "AME", "MCHP", "ALL", "CTVA", "PSA", "D", "FANG", "OXY", "OKE", "NUE",
    "TRGP", "ADSK", "AJG", "COIN", "MET", "ETR", "FAST", "ROK", "NDAQ",
    "XEL", "EA", "COR", "F", "DAL", "EBAY", "EW", "GRMN", "EXC", "WAB",
    "FITB", "IDXX", "CAH", "YUM", "AMP", "XYZ", "DHI", "MSCI", "CBRE",
    "ODFL", "BDX", "EME", "VTR", "CMG", "STT", "TTWO", "ON", "AIG", "ZTS",
    "PYPL", "KR", "HPE", "ED", "PEG", "CCI", "JBL", "LYV", "IRM", "KDP",
    "VMC", "IBKR", "CCL", "HSY", "ADM", "WEC", "MLM", "SATS", "HIG", "CBOE",
    "ROP", "PCG", "EQT", "LVS", "STLD", "SYY", "PRU", "KVUE", "WAT", "HBAN",
    "HAL", "A", "ACGL", "KMB", "UAL", "PAYX", "NRG", "CPRT", "MTB", "AXON",
    "CASY", "WDAY", "Q", "EL", "ATO", "RJF", "IR", "VICI", "DOV", "RMD",
    "EXR", "AEE", "DTE", "TDY", "FISV", "TPR", "NTRS", "EXPE", "OTIS",
    "HUM", "IQV", "TPL", "XYL", "DVN", "BIIB", "GEHC", "ARES", "PPL", "CNP",
    "CFG", "VEEV", "CNC", "KHC", "DOW", "MTD", "HUBB", "EIX", "ROL", "STZ",
    "FE", "AVB", "DG", "ES", "SYF", "CINF", "PPG", "FICO", "EQR", "AWK",
    "ALB", "WRB", "BG", "VRSN", "CTSH", "RF", "KEY", "WTW", "TSN", "FIS",
    "FSLR", "DXCM", "ULTA", "LYB", "JBHT", "SBAC", "CMS", "EXE", "PHM",
    "NI", "TROW", "RL", "VRSK", "CHD", "LEN", "WSM", "NTAP", "DRI", "WST",
    "SW", "PFG", "OMC", "L", "VLTO", "DGX", "LH", "STE", "MRNA", "IFF",
    "EFX", "LUV", "CPAY", "DD", "SMCI", "PKG", "CHRW", "INCY", "SNA",
    "EXPD", "HPQ", "CHTR", "BRO", "FFIV", "VTRS", "GPN", "LII", "DLTR",
    "EVRG", "GIS", "LNT", "AMCR", "FTV", "CF", "IP", "BR", "PTC", "WY",
    "TSCO", "ESS", "INVH", "AKAM", "LDOS", "NVR", "IEX", "TXT", "BEN",
    "ZBH", "KIM", "NDSN", "BALL", "GNRC", "LULU", "TRMB", "MAA", "HST",
    "J", "DECK", "MAS", "GPC", "REG", "CDW", "TKO", "CSGP", "EG", "HAS",
    "TYL", "DOC", "APA", "MKC", "PNR", "AVY", "ALGN", "SWK", "BF-B",
    "HII", "DVA", "BBY", "GL", "FOX", "APTV", "PSKY", "FOXA", "SOLV",
    "PNW", "IVZ", "UDR", "GEN", "COO", "AIZ", "ALLE", "HRL", "TTD", "GDDY",
    "ERIE", "RVTY", "ZBRA", "WYNN", "CLX", "DPZ", "PODD", "CPT", "SJM",
    "UHS", "IT", "JKHY", "AES", "FRT", "SWKS", "MGM", "NWSA", "BXP",
    "CRL", "BAX", "BLDR", "AOS", "HSIC", "NCLH", "TAP", "ARE", "FDS",
    "MOS", "TECH", "POOL", "CAG", "CPB", "NWS", "EPAM",
]

# Parâmetros ótimos por ativo — atualizados automaticamente após cada varredura
BEST_STEP  = {asset: 20 for asset in ASSET_LIST}
BEST_EMBED = {asset: 70 for asset in ASSET_LIST}

# Controle de fases
SKIP_STEP   = False   # True → pula varredura de step, usa BEST_STEP
SKIP_EMBED  = False   # True → pula varredura de embed, usa BEST_EMBED
SKIP_SINGLE = True    # True → pula análise com params ótimos (geralmente redundante)

# Parâmetros de análise
STEP        = 20
EMBED_DIM   = 70
# WINDOW = 5 * EMBED_DIM (padrão em decomp_pca — não precisa ser definido aqui)

STEP_LIST_STEP  = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
STEP_LIST_EMBED = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]

# Timeframes — mesmo formato que experimento_PCA_4D_3.py
TIMEFRAME_LABELS = ['M1']

# Mapeamento para nome na partição Parquet
_TF_TO_PARQUET = {
    '1day': 'D1', '1d': 'D1', 'D1': 'D1',
    '1hour': 'H1', '1h': 'H1', 'H1': 'H1',
    '1week': 'W1', '1w': 'W1', 'W1': 'W1',
    '4hour': 'H4', '4h': 'H4', 'H4': 'H4',
    '1min': 'M1',  'M1': 'M1',
    '5min': 'M5',  'M5': 'M5',
    '15min': 'M15','M15': 'M15',
    '30min': 'M30','M30': 'M30',
}

# Parâmetros de análise estatística
MAX_LAG   = 200
TAIL_FRAC = 0.7

# ── Paralelismo e memória ─────────────────────────────────────────────────────
# N_WORKERS = 1  → sequencial; BLAS usa todos os cores via chunked eigh (padrão)
# N_WORKERS > 1  → assets em paralelo; BLAS threads = cpu_count // N_WORKERS
#   Use > 1 para D1 (poucas janelas/asset); mantenha 1 para H1/M5 (muitas janelas)
N_WORKERS     = 1
MEM_BUDGET_MB = 256   # orçamento de RAM por worker para o buffer de chunks

FILE_ENCODING = 'utf-8'


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de log (análogos ao experimento)
# ─────────────────────────────────────────────────────────────────────────────

class RunLogger:
    """Espelha sys.stdout no terminal e num arquivo de log simultaneamente."""
    def __init__(self, filepath: str):
        self._terminal = sys.stdout
        self._log      = open(filepath, 'a', encoding=FILE_ENCODING, buffering=1)
        self.current_path = filepath

    def switch_file(self, filepath: str):
        self._log.flush(); self._log.close()
        self._log = open(filepath, 'a', encoding=FILE_ENCODING, buffering=1)
        self.current_path = filepath

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()


def _log_header(title: str, level: int = 1):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    if level == 1:
        sep = '=' * 80
        print(f"\n{sep}\n  ASSET: {title}  |  {ts}\n{sep}")
    elif level == 2:
        sep = '-' * 72
        print(f"\n  {sep}\n  {title}  |  {ts}\n  {sep}")
    elif level == 3:
        print(f"\n    >> {title}  |  {ts}")
    else:
        print(f"\n      [{title}]")


# ─────────────────────────────────────────────────────────────────────────────
# Funções de nível de módulo para ProcessPoolExecutor (precisam ser picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init(n_blas: int) -> None:
    """Configura BLAS threads no processo filho antes de qualquer cálculo."""
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(n_blas)


def _asset_task(asset: str, valid_tfs: list,
                embed_fixo: int,
                step_list_step: list, step_list_embed: list,
                skip_step: bool, skip_embed: bool, skip_single: bool,
                best_step_init: int, best_embed_init: int,
                build_plots: bool, fig_dir: str, log_dir: str,
                overwrite_decomp: bool, overwrite_analysis: bool,
                max_lag: int, tail_frac: float,
                mem_budget_mb: int) -> list[dict]:
    """
    Processa um asset completo (todas as fases e TFs) num processo filho.

    Retorna lista de dicts escalares para o processo principal agregar no CSV/DB.
    Não usa RunLogger (cada worker escreve no próprio arquivo de log).
    """
    # redireciona stdout do worker para arquivo de log por asset
    _log_path = os.path.join(log_dir, f"{asset}_worker.txt")
    _orig_stdout = sys.stdout
    try:
        sys.stdout = open(_log_path, 'a', encoding='utf-8', buffering=1)
    except OSError:
        sys.stdout = _orig_stdout

    rows = []
    best_step  = best_step_init
    best_embed = best_embed_init

    try:
        for tf_label, tf_parquet in valid_tfs:
            print(f"\n[{asset}] {tf_label} -> {tf_parquet}  |  {time.strftime('%H:%M:%S')}")

            # FASE 1: varredura de STEP
            if not skip_step:
                for step in step_list_step:
                    row = _run_one_return(
                        asset, tf_label, tf_parquet, step, embed_fixo,
                        build_plots, fig_dir, overwrite_decomp, overwrite_analysis,
                        max_lag, tail_frac, mem_budget_mb,
                    )
                    if row:
                        rows.append(row)
                # melhor step pelo MP_L2 dos resultados já computados
                if rows:
                    df_step = pd.DataFrame(
                        [r for r in rows if r.get('timeframe') == tf_parquet
                         and r.get('embed_dim') == embed_fixo]
                    )
                    if not df_step.empty and 'MP_L2_relative_pct' in df_step:
                        best_step = int(df_step.loc[
                            df_step['MP_L2_relative_pct'].idxmin(), 'step'
                        ])
                print(f"  BEST_STEP[{asset}] = {best_step}")

            # FASE 2: varredura de EMBED
            if not skip_embed:
                for embed in step_list_embed:
                    row = _run_one_return(
                        asset, tf_label, tf_parquet, best_step, embed,
                        build_plots, fig_dir, overwrite_decomp, overwrite_analysis,
                        max_lag, tail_frac, mem_budget_mb,
                    )
                    if row:
                        rows.append(row)
                if rows:
                    df_embed = pd.DataFrame(
                        [r for r in rows if r.get('timeframe') == tf_parquet
                         and r.get('step') == best_step]
                    )
                    if not df_embed.empty and 'MP_L2_relative_pct' in df_embed:
                        best_embed = int(df_embed.loc[
                            df_embed['MP_L2_relative_pct'].idxmin(), 'embed_dim'
                        ])
                print(f"  BEST_EMBED[{asset}] = {best_embed}")

            # FASE 3: single params ótimos
            if not skip_single:
                row = _run_one_return(
                    asset, tf_label, tf_parquet, best_step, best_embed,
                    build_plots, fig_dir, overwrite_decomp, overwrite_analysis,
                    max_lag, tail_frac, mem_budget_mb,
                )
                if row:
                    rows.append(row)

    finally:
        if sys.stdout is not _orig_stdout:
            sys.stdout.close()
            sys.stdout = _orig_stdout

    return rows


def _run_one_return(symbol: str, tf_label: str, tf_parquet: str,
                    step: int, embed: int,
                    build_plots: bool, fig_dir: str,
                    overwrite_decomp: bool, overwrite_analysis: bool,
                    max_lag: int, tail_frac: float,
                    mem_budget_mb: int) -> dict | None:
    """
    Versão de _run_one() que retorna o dict escalar em vez de mutacionr all_rows.
    Usada tanto no path sequencial quanto no path paralelo.
    """
    tag = f"{symbol}/{tf_parquet}/step={step}/embed={embed}"

    # 1. Decomposição
    try:
        decomp_pca.run(
            symbol=symbol, tf_label=tf_label,
            step=step, embed_dim=embed,
            mem_budget_mb=mem_budget_mb,
            overwrite=overwrite_decomp,
        )
    except Exception as exc:
        print(f"  [ERRO decomp] {tag}: {exc}")
        return None

    # 2. Análise — skip se já no DB
    if not overwrite_analysis and analysis_db.exists(symbol, tf_parquet, step, embed):
        print(f"  [skip analise] {tag} — ja existe no DB")
        df_row = analysis_db.load(symbol, tf_parquet)
        match  = df_row[(df_row.step == step) & (df_row.embed_dim == embed)]
        return match.iloc[0].to_dict() if not match.empty else None

    try:
        res = decomp_analysis.analyze(
            symbol=symbol, timeframe=tf_parquet,
            step=step, embed_dim=embed,
            max_lag=max_lag, tail_frac=tail_frac,
        )
    except Exception as exc:
        print(f"  [ERRO analise] {tag}: {exc}")
        return None

    # 3. Figuras
    if build_plots:
        pdf_path = os.path.join(fig_dir, f"{symbol}_{tf_parquet}_s{step}_e{embed}.pdf")
        try:
            with _PdfPages(pdf_path) as pdf:
                decomp_analysis.plot_results(res, pdf)
        except Exception as exc:
            print(f"  [AVISO plots] {tag}: {exc}")

    # salva no DB (idempotente — INSERT OR REPLACE)
    analysis_db.save(res)

    print(f"  [ok] {tag} | beta_struct={res['beta_struct']:.3f} "
          f"R_FDT={res['R_FDT']:.3f} alpha={res['alpha']:.3f}")
    return decomp_analysis.to_scalar_row(res)


# ─────────────────────────────────────────────────────────────────────────────
# Verificação de dados disponíveis
# ─────────────────────────────────────────────────────────────────────────────

def _data_available(symbol: str, tf_parquet: str) -> bool:
    """Verifica se existe pelo menos um arquivo Parquet de dados para (symbol, tf)."""
    base = decomp_pca.DATA_ROOT
    for source in ("alpaca", "metatrader"):
        p = base / f"source={source}" / f"symbol={symbol}" / f"timeframe={tf_parquet}"
        if p.exists() and any(p.rglob("*.parquet")):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Gravação segura do CSV acumulativo (retry em caso de bloqueio de arquivo)
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(rows: list, path: str, retries: int = 5):
    df = pd.DataFrame(rows)
    for attempt in range(retries):
        try:
            df.to_csv(path, index=False)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Núcleo: decomposição + análise para uma combinação (symbol, tf, step, embed)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(symbol: str, tf_label: str, tf_parquet: str,
             step: int, embed: int,
             build_plots: bool, fig_dir: str,
             overwrite_decomp: bool, overwrite_analysis: bool,
             all_rows: list, csv_path: str) -> None:
    """Wrapper sequencial: chama _run_one_return e agrega resultado no CSV."""
    row = _run_one_return(
        symbol, tf_label, tf_parquet, step, embed,
        build_plots, fig_dir, overwrite_decomp, overwrite_analysis,
        MAX_LAG, TAIL_FRAC, MEM_BUDGET_MB,
    )
    if row is not None:
        all_rows.append(row)
        _save_csv(all_rows, csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(build_plots: bool = True,
                 overwrite_decomp: bool = False,
                 overwrite_analysis: bool = False,
                 n_workers: int = None) -> None:

    n_workers = n_workers if n_workers is not None else N_WORKERS
    n_blas    = max(1, _N_CPU // n_workers)

    # ── estrutura de diretórios da run ────────────────────────────────────
    _run_ts    = time.strftime('%Y%m%d_%H%M%S')
    _src_dir   = Path(__file__).parent
    _run_dir   = _src_dir / f"run_{_run_ts}"
    _log_dir   = _run_dir / "logs"
    _fig_dir   = _run_dir / "figures"
    _log_insuf = _run_dir / "insufficient_data_log.txt"
    _csv_path  = str(_run_dir / f"results_{_run_ts}.csv")

    _run_dir.mkdir(parents=True, exist_ok=True)
    _log_dir.mkdir(exist_ok=True)
    _fig_dir.mkdir(exist_ok=True)

    analysis_db.init()

    _all_rows: list[dict] = []

    # banner inicial (sempre no terminal original, antes de redirecionar stdout)
    _banner = (
        f"\nInicio  |  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Ativos  : {len(ASSET_LIST)}\n"
        f"TFs     : {TIMEFRAME_LABELS}\n"
        f"Steps   : {STEP_LIST_STEP}  (SKIP={SKIP_STEP})\n"
        f"Embeds  : {STEP_LIST_EMBED}  (SKIP={SKIP_EMBED})\n"
        f"Workers : {n_workers}  |  BLAS threads/worker: {n_blas}  |  CPUs: {_N_CPU}\n"
        f"Memoria : {MEM_BUDGET_MB} MB/worker para buffer de chunks\n"
        f"DB      : {analysis_db.DB_PATH}\n"
        f"Saida   : {_run_dir}/\n"
    )
    print(_banner)

    # ── pré-filtragem: quais assets têm dados disponíveis ─────────────────
    asset_tfs: list[tuple[str, list]] = []   # [(asset, [(tf_label, tf_parquet), ...]), ...]
    with open(_log_insuf, 'a', encoding=FILE_ENCODING) as _lf:
        for asset in ASSET_LIST:
            valid_tfs = []
            for tf_label in TIMEFRAME_LABELS:
                tf_parquet = _TF_TO_PARQUET.get(tf_label, tf_label)
                if _data_available(asset, tf_parquet):
                    valid_tfs.append((tf_label, tf_parquet))
                else:
                    _lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                              f"asset={asset} | tf={tf_label} | dados nao encontrados\n")
            if valid_tfs:
                asset_tfs.append((asset, valid_tfs))
            else:
                print(f"  [AUSENTE] {asset} — nenhum dado disponivel, ignorado")

    print(f"{len(asset_tfs)}/{len(ASSET_LIST)} ativos com dados disponiveis\n")

    # ── PATH SEQUENCIAL (N_WORKERS == 1) ─────────────────────────────────
    if n_workers == 1:
        _run_logger = RunLogger(str(_log_dir / "_run_geral.txt"))
        sys.stdout  = _run_logger
        print(_banner)

        for asset, valid_tfs in asset_tfs:
            _log_header(asset, level=1)
            _run_logger.switch_file(str(_log_dir / f"{asset}.txt"))
            _log_header(asset, level=1)

            for tf_label, tf_parquet in valid_tfs:
                _log_header(f"Timeframe: {tf_label} -> {tf_parquet}", level=3)

                if not SKIP_STEP:
                    _log_header("FASE 1: varredura STEP", level=2)
                    for step in STEP_LIST_STEP:
                        _log_header(f"step={step} embed={EMBED_DIM}", level=4)
                        _run_one(asset, tf_label, tf_parquet, step, EMBED_DIM,
                                 build_plots, str(_fig_dir),
                                 overwrite_decomp, overwrite_analysis,
                                 _all_rows, _csv_path)
                    best = analysis_db.best_step(asset, tf_parquet)
                    if best:
                        BEST_STEP[asset] = best
                        print(f"  BEST_STEP[{asset}] = {best}")

                if not SKIP_EMBED:
                    _log_header("FASE 2: varredura EMBED", level=2)
                    for embed in STEP_LIST_EMBED:
                        _log_header(f"step={BEST_STEP[asset]} embed={embed}", level=4)
                        _run_one(asset, tf_label, tf_parquet, BEST_STEP[asset], embed,
                                 build_plots, str(_fig_dir),
                                 overwrite_decomp, overwrite_analysis,
                                 _all_rows, _csv_path)
                    best = analysis_db.best_embed(asset, tf_parquet)
                    if best:
                        BEST_EMBED[asset] = best
                        print(f"  BEST_EMBED[{asset}] = {best}")

                if not SKIP_SINGLE:
                    _log_header("FASE 3: single (params otimos)", level=2)
                    _run_one(asset, tf_label, tf_parquet,
                             BEST_STEP[asset], BEST_EMBED[asset],
                             build_plots, str(_fig_dir),
                             overwrite_decomp, overwrite_analysis,
                             _all_rows, _csv_path)

        _run_logger.close()

    # ── PATH PARALELO (N_WORKERS > 1) ────────────────────────────────────
    else:
        # Cada worker: processa um asset completo (todas TFs + todas fases)
        # BLAS threads reduzidas para não sobre-subscrever os cores
        n_done = 0
        n_total_assets = len(asset_tfs)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(n_blas,),
        ) as pool:
            futures = {
                pool.submit(
                    _asset_task,
                    asset, valid_tfs,
                    EMBED_DIM,
                    STEP_LIST_STEP, STEP_LIST_EMBED,
                    SKIP_STEP, SKIP_EMBED, SKIP_SINGLE,
                    BEST_STEP[asset], BEST_EMBED[asset],
                    build_plots, str(_fig_dir), str(_log_dir),
                    overwrite_decomp, overwrite_analysis,
                    MAX_LAG, TAIL_FRAC, MEM_BUDGET_MB,
                ): asset
                for asset, valid_tfs in asset_tfs
            }

            for future in as_completed(futures):
                asset = futures[future]
                n_done += 1
                try:
                    rows = future.result()
                    _all_rows.extend(rows)
                    if _all_rows:
                        _save_csv(_all_rows, _csv_path)
                    pct = 100 * n_done / n_total_assets
                    bar = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
                    print(f"[{bar}] {pct:5.1f}%  {asset} concluido "
                          f"({n_done}/{n_total_assets})  +{len(rows)} resultados",
                          flush=True)
                except Exception as exc:
                    print(f"  [ERRO worker] {asset}: {exc}", flush=True)

    print(f"\nConcluido | {len(_all_rows)} combinacoes | "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DB   : {analysis_db.DB_PATH}")
    print(f"CSV  : {_csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline decomposicao + analise para multiplos ativos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-plots",           action="store_true",
                   help="Pula geracao de PDFs de figuras")
    p.add_argument("--overwrite-decomp",   action="store_true",
                   help="Refaz decomposicao mesmo se parquet existir")
    p.add_argument("--overwrite-analysis", action="store_true",
                   help="Refaz analise mesmo se resultado estiver no DB")
    p.add_argument("--workers",            type=int, default=N_WORKERS,
                   help="Numero de assets processados em paralelo "
                        "(1=sequencial, >1=ProcessPoolExecutor com BLAS reduzido)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        build_plots        = not args.no_plots,
        overwrite_decomp   = args.overwrite_decomp,
        overwrite_analysis = args.overwrite_analysis,
        n_workers          = args.workers,
    )
