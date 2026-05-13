"""
Pipeline completo: decomposição PCA + análise estatística para múltiplos ativos.

Espelha a estrutura e as variáveis globais de experimento_PCA_4D_3.py.
Para cada (ativo × timeframe × step × embed):
  1. decomp_pca.run()          → garante parquet da decomposição (skip se já existir)
  2. decomp_analysis.analyze() → calcula beta, FDT, alpha, MP, entropia
  3. analysis_db.save()        → persiste escalares no SQLite
  4. CSV acumulativo da run    → snapshot por execução

Uso:
    python decomp_run.py                       # configurações padrão
    python decomp_run.py --no-plots            # pula geração de PDFs de figuras
    python decomp_run.py --overwrite-analysis  # refaz análise mesmo se já estiver no DB
    python decomp_run.py --overwrite-decomp    # refaz decomposição mesmo se parquet existir
"""

import os
import sys
import time
import argparse
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

STEP_LIST_STEP  = [20, 30, 40, 50]
STEP_LIST_EMBED = [70, 80, 90, 100]

# Timeframes — mesmo formato que experimento_PCA_4D_3.py
TIMEFRAME_LABELS = ['1day']

# Mapeamento para nome na partição Parquet
_TF_TO_PARQUET = {
    '1day': 'D1', '1d': 'D1', 'D1': 'D1',
    '1hour': 'H1', '1h': 'H1', 'H1': 'H1',
    '4hour': 'H4', '4h': 'H4', 'H4': 'H4',
    '1min': 'M1',  'M1': 'M1',
    '5min': 'M5',  'M5': 'M5',
    '15min': 'M15','M15': 'M15',
    '30min': 'M30','M30': 'M30',
}

# Parâmetros de análise estatística
MAX_LAG   = 200
TAIL_FRAC = 0.7

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
             build_plots: bool,
             fig_dir: str,
             overwrite_decomp: bool,
             overwrite_analysis: bool,
             all_rows: list,
             csv_path: str) -> dict | None:
    """
    Executa decomposição + análise para uma combinação e salva no DB.
    Retorna o dict de resultados escalares, ou None em caso de erro.
    """
    tag = f"{symbol}/{tf_parquet}/step={step}/embed={embed}"

    # ── 1. Decomposição (skip se parquet já existir) ───────────────────────
    try:
        decomp_pca.run(
            symbol     = symbol,
            tf_label   = tf_label,
            step       = step,
            embed_dim  = embed,
            overwrite  = overwrite_decomp,
        )
    except Exception as exc:
        print(f"  [ERRO decomp] {tag}: {exc}")
        return None

    # ── 2. Análise (skip se já no DB e não forçar reescrita) ──────────────
    if not overwrite_analysis and analysis_db.exists(symbol, tf_parquet, step, embed):
        print(f"  [skip análise] {tag} — já existe no DB")
        # recupera do DB para incluir no CSV da run
        df_row = analysis_db.load(symbol, tf_parquet)
        match  = df_row[(df_row.step == step) & (df_row.embed_dim == embed)]
        if not match.empty:
            row = match.iloc[0].to_dict()
            all_rows.append(row)
            _save_csv(all_rows, csv_path)
        return None

    try:
        res = decomp_analysis.analyze(
            symbol    = symbol,
            timeframe = tf_parquet,
            step      = step,
            embed_dim = embed,
            max_lag   = MAX_LAG,
            tail_frac = TAIL_FRAC,
        )
    except Exception as exc:
        print(f"  [ERRO análise] {tag}: {exc}")
        return None

    # ── 3. Figuras (opcional) ─────────────────────────────────────────────
    if build_plots:
        pdf_path = os.path.join(fig_dir, f"{symbol}_{tf_parquet}_s{step}_e{embed}.pdf")
        try:
            with _PdfPages(pdf_path) as pdf:
                decomp_analysis.plot_results(res, pdf)
        except Exception as exc:
            print(f"  [AVISO plots] {tag}: {exc}")

    # ── 4. Persistência ───────────────────────────────────────────────────
    analysis_db.save(res)

    row = decomp_analysis.to_scalar_row(res)
    all_rows.append(row)
    _save_csv(all_rows, csv_path)

    print(f"  [ok] {tag} | beta_struct={res['beta_struct']:.3f} "
          f"R_FDT={res['R_FDT']:.3f} alpha={res['alpha']:.3f}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(build_plots: bool = True,
                 overwrite_decomp: bool = False,
                 overwrite_analysis: bool = False) -> None:

    # ── estrutura de diretórios da run ────────────────────────────────────
    _run_ts      = time.strftime('%Y%m%d_%H%M%S')
    _src_dir     = Path(__file__).parent
    _run_dir     = _src_dir / f"run_{_run_ts}"
    _log_dir     = _run_dir / "logs"
    _fig_dir     = _run_dir / "figures"
    _log_insuf   = _run_dir / "insufficient_data_log.txt"
    _csv_path    = str(_run_dir / f"results_{_run_ts}.csv")

    _run_dir.mkdir(parents=True, exist_ok=True)
    _log_dir.mkdir(exist_ok=True)
    _fig_dir.mkdir(exist_ok=True)

    analysis_db.init()

    _run_logger = RunLogger(str(_log_dir / "_run_geral.txt"))
    sys.stdout  = _run_logger

    _all_rows: list[dict] = []

    print(f"Inicio da execucao  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ativos : {len(ASSET_LIST)}")
    print(f"TFs    : {TIMEFRAME_LABELS}")
    print(f"Steps  : {STEP_LIST_STEP}  (SKIP={SKIP_STEP})")
    print(f"Embeds : {STEP_LIST_EMBED}  (SKIP={SKIP_EMBED})")
    print(f"DB     : {analysis_db.DB_PATH}")
    print(f"Saida  : {_run_dir}/")

    # ── loop principal ────────────────────────────────────────────────────
    for ASSET in ASSET_LIST:
        _log_header(ASSET, level=1)
        _run_logger.switch_file(str(_log_dir / f"{ASSET}_geral.txt"))
        _log_header(ASSET, level=1)

        # verifica disponibilidade de dados para cada timeframe
        valid_tfs: list[tuple[str, str]] = []
        for tf_label in TIMEFRAME_LABELS:
            tf_parquet = _TF_TO_PARQUET.get(tf_label, tf_label)
            if _data_available(ASSET, tf_parquet):
                valid_tfs.append((tf_label, tf_parquet))
            else:
                msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | asset={ASSET} | tf={tf_label} | dados nao encontrados\n"
                print(f"  [AUSENTE] {ASSET} {tf_label} — ignorado")
                with open(_log_insuf, 'a', encoding=FILE_ENCODING) as f:
                    f.write(msg)

        if not valid_tfs:
            print(f"  [AVISO] Nenhum dado disponivel para {ASSET} — ativo ignorado")
            continue

        for tf_label, tf_parquet in valid_tfs:
            _run_logger.switch_file(str(_log_dir / f"{ASSET}_{tf_parquet}.txt"))
            _log_header(ASSET, level=1)
            _log_header(f"Timeframe: {tf_label} -> {tf_parquet}", level=3)

            # ── FASE 1: varredura de STEP (embed fixo) ────────────────────
            if not SKIP_STEP:
                _log_header("FASE 1: varredura STEP", level=2)
                embed_fixo = EMBED_DIM
                for step in STEP_LIST_STEP:
                    _log_header(f"step={step} embed={embed_fixo}", level=4)
                    _run_one(
                        symbol=ASSET, tf_label=tf_label, tf_parquet=tf_parquet,
                        step=step, embed=embed_fixo,
                        build_plots=build_plots, fig_dir=str(_fig_dir),
                        overwrite_decomp=overwrite_decomp,
                        overwrite_analysis=overwrite_analysis,
                        all_rows=_all_rows, csv_path=_csv_path,
                    )

                # determina BEST_STEP a partir do DB
                best = analysis_db.best_step(ASSET, tf_parquet, criterion='mp_l2')
                if best is not None:
                    BEST_STEP[ASSET] = best
                    print(f"  BEST_STEP[{ASSET}] = {best} (criterio: MP_L2 minimo)")

            # ── FASE 2: varredura de EMBED (step ótimo fixo) ──────────────
            if not SKIP_EMBED:
                _log_header("FASE 2: varredura EMBED", level=2)
                step_fixo = BEST_STEP[ASSET]
                for embed in STEP_LIST_EMBED:
                    _log_header(f"step={step_fixo} embed={embed}", level=4)
                    _run_one(
                        symbol=ASSET, tf_label=tf_label, tf_parquet=tf_parquet,
                        step=step_fixo, embed=embed,
                        build_plots=build_plots, fig_dir=str(_fig_dir),
                        overwrite_decomp=overwrite_decomp,
                        overwrite_analysis=overwrite_analysis,
                        all_rows=_all_rows, csv_path=_csv_path,
                    )

                # determina BEST_EMBED a partir do DB
                best = analysis_db.best_embed(ASSET, tf_parquet, criterion='mp_l2')
                if best is not None:
                    BEST_EMBED[ASSET] = best
                    print(f"  BEST_EMBED[{ASSET}] = {best} (criterio: MP_L2 minimo)")

            # ── FASE 3: análise com parâmetros ótimos (single) ────────────
            if not SKIP_SINGLE:
                _log_header("FASE 3: single (params otimos)", level=2)
                step_opt  = BEST_STEP[ASSET]
                embed_opt = BEST_EMBED[ASSET]
                _log_header(f"step={step_opt} embed={embed_opt}", level=4)
                _run_one(
                    symbol=ASSET, tf_label=tf_label, tf_parquet=tf_parquet,
                    step=step_opt, embed=embed_opt,
                    build_plots=build_plots, fig_dir=str(_fig_dir),
                    overwrite_decomp=overwrite_decomp,
                    overwrite_analysis=overwrite_analysis,
                    all_rows=_all_rows, csv_path=_csv_path,
                )

    _run_logger.close()
    print(f"\nConcluido | {len(_all_rows)} combinacoes processadas")
    print(f"DB     : {analysis_db.DB_PATH}")
    print(f"CSV    : {_csv_path}")


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
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        build_plots        = not args.no_plots,
        overwrite_decomp   = args.overwrite_decomp,
        overwrite_analysis = args.overwrite_analysis,
    )
