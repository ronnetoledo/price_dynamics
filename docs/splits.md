# Splits — detecção, base de dados e ajuste dos parquet

**Data:** 2026-06-02 · **Escopo:** fonte `alpaca` (502 símbolos S&P 500), timeframes `D1/H1/M1/W1`.

## 1. Problema

Os preços da fonte `alpaca` no `data_parquet/` vinham com **splits NÃO ajustados** — o
preço cai/sobe de um dia para o outro pelo fator do desdobramento, gerando um log-retorno
artificial gigante (|ret| ≫ 0.5). Ao deslizar pela matriz de Hankel, um único salto desses
aparece em ~`embed` snapshots consecutivos e injeta autovalores espúrios acima do limiar
Marchenko-Pastur, inflando o número de modos estruturais `m` e contaminando β/MSD e a
entropia — exatamente as quantidades do artigo. Diagnóstico original e mecanismo: ver
[volatilidade.md](volatilidade.md) (o gap entre vol close-based e vol de range puro flagra
o salto, pois Parkinson/GK/RS nunca cruzam a fronteira close→open).

## 2. Arquitetura da solução

Quatro peças, todas em [src_newest/](../src_newest/):

| Peça | Arquivo | Função |
|---|---|---|
| Base de dados | [splits_db.py](../src_newest/splits_db.py) + `splits.db` (SQLite, raiz) | armazena os splits, calcula o fator acumulado |
| Coleta + verificação | [fetch_splits.py](../src_newest/fetch_splits.py) | Alpaca Corporate Actions API, cruzada com a série D1 |
| Ajuste dos dados | [apply_splits.py](../src_newest/apply_splits.py) | back-adjust → tree paralelo `data_parquet_adj/` |
| Carregador | [decomp_pca.py](../src_newest/decomp_pca.py) `load_ohlcv(adjusted=True)` | usa o ajustado, cai no raw p/ não-splitados |

### 2.1 Base de dados (`splits.db`)

Tabela `splits`, PK `(symbol, ex_date)`:

```
symbol, ex_date, ratio_num, ratio_den, factor,           -- razão/fator
ca_type, ca_id, cusip,                                    -- metadados crus da API
process_date, record_date, payable_date, due_bill_redemption_date,
source, detected_logret, confirmed, note, updated_at      -- verificação (nosso)
```

Convenção de fator: `factor = ratio_num / ratio_den = preço_pós / preço_pré`.
- forward split 4:1 → `num=4, den=1, factor=4.0` (preço cai ~4×)
- reverse split 1:8 → `num=1, den=8, factor=0.125` (preço sobe ~8×)

**Metadados crus da Alpaca CA API** (preservados para rastreabilidade/auditoria do PRL):
- `ca_type` — classificação da própria Alpaca (`forward_splits` / `reverse_splits`);
  91 forward / 7 reverse. Mais autoritativo que inferir pelo sinal do factor.
- `ca_id` — UUID canônico do evento na Alpaca · `cusip` — CUSIP do papel (7 nulos).
- `process_date`, `record_date`, `payable_date`, `due_bill_redemption_date` — datas do
  ciclo do evento, como vêm da API. `record/payable` nulas em 42 eventos e `due_bill` em
  46 (eventos antigos / spin-offs sem ciclo de liquidação completo).
- `detected_logret` e `note` são **nossos** (verificação), não da API: `note` é vazio nos
  confirmados e traz a mensagem de discrepância nos 3 ressalvados.

### 2.2 Coleta e verificação

Fonte: **Alpaca Corporate Actions API** (`CorporateActionsClient`, mesma vendor dos bars,
credenciais em `alpaca/.env`). Como API e bars vêm da mesma fonte, o `ex_date` casa
exatamente com o salto na série.

Cada evento da API é **verificado contra a nossa série D1**: o log-retorno close-to-close
observado no `ex_date` deve bater com o esperado `−ln(factor)` (tolerância 0.12, que
absorve o retorno real do dia). `confirmed=1` se casa; `0` se o `ex_date` está fora da
janela de dados ou se o salto não bate (reestruturação disfarçada).

### 2.3 Ajuste (back-adjustment, convenção total-return)

Para um bar em data `t`:

```
cumfac(t) = ∏ factor de todos os splits com ex_date > data(t)

open, high, low, close, vwap   ÷=  cumfac(t)
volume, tick_volume            ×=  cumfac(t)
trade_count                    inalterado   (split não cria trades)
```

Bars **no** ex_date ou depois não são tocados (já estão a preço pós-split). O mesmo
`ex_date` é aplicado a todos os timeframes por comparação de data.

### 2.4 Escrita e carregamento

Os dados ajustados vão para um **tree paralelo** `data_parquet_adj/` (mesma estrutura
`source=/symbol=/timeframe=/year=`, schema e compressão snappy idênticos), contendo
**apenas os 76 símbolos com split confirmado**. Os ~426 símbolos sem split não são
copiados.

`load_ohlcv(symbol, tf, adjusted=True)` é o **default** (`decomp_pca.USE_ADJUSTED = True`):
tenta `data_parquet_adj/` e cai automaticamente no `data_parquet/` raw para os símbolos
sem partição ajustada. Para recuperar o RAW (ex.: o próprio detector de splits do
[vol_estimators.py](../src_newest/vol_estimators/vol_estimators.py)), passar `adjusted=False`.

## 3. Resultados

- **98 eventos** retornados pela API (forward + reverse), janela 2016→2026.
- **95 confirmados** em **76 símbolos** → aplicados.
- **3 ressalvas** (`confirmed=0`) — reestruturações, **não** aplicadas.
- `data_parquet_adj/`: **3.274 partições, 76,7M linhas**.
- Verificação pós-ajuste: **zero** salto residual `|logret| > 0.18` em todos os boundaries.

Decisão do usuário: **ajustar todos os 95** (splits limpos + spin-offs), porque todo gap
overnight contamina o espectro da mesma forma; o back-adjustment é a convenção total-return.

## 4. Splits aplicados

### 4.1 Forward splits — 77 eventos, 62 símbolos

Desdobramentos "limpos" (razão simples). Multi-split marcados.

| Símbolo | ex_date | Razão | factor | Símbolo | ex_date | Razão | factor |
|---|---|---|---|---|---|---|---|
| AAPL | 2020-08-31 | 4:1 | 4 | MNST ² | 2016-11-10 | 3:1 | 3 |
| ACGL | 2018-06-21 | 3:1 | 3 | MNST ² | 2023-03-28 | 2:1 | 2 |
| AFL | 2018-03-19 | 2:1 | 2 | NDAQ | 2022-08-29 | 3:1 | 3 |
| AMZN | 2022-06-06 | 20:1 | 20 | NEE | 2020-10-27 | 4:1 | 4 |
| ANET ² | 2021-11-18 | 4:1 | 4 | NFLX | 2025-11-17 | 10:1 | 10 |
| ANET ² | 2024-12-04 | 4:1 | 4 | NOW | 2025-12-18 | 5:1 | 5 |
| AOS | 2016-10-06 | 2:1 | 2 | NVDA ² | 2021-07-20 | 4:1 | 4 |
| APH ² | 2021-03-05 | 2:1 | 2 | NVDA ² | 2024-06-10 | 10:1 | 10 |
| APH ² | 2024-06-12 | 2:1 | 2 | ODFL ² | 2020-03-25 | 3:2 | 1.5 |
| AVGO | 2024-07-15 | 10:1 | 10 | ODFL ² | 2024-03-28 | 2:1 | 2 |
| BKNG | 2026-04-06 | 25:1 | 25 | ORLY | 2025-06-10 | 15:1 | 15 |
| BRO | 2018-03-29 | 2:1 | 2 | PANW ² | 2022-09-14 | 3:1 | 3 |
| CHD | 2016-09-02 | 2:1 | 2 | PANW ² | 2024-12-16 | 2:1 | 2 |
| CMCSA | 2017-02-21 | 2:1 | 2 | PCAR | 2023-02-08 | 3:2 | 1.5 |
| CMG | 2024-06-26 | 50:1 | 50 | RJF | 2021-09-22 | 3:2 | 1.5 |
| CNC | 2019-02-07 | 2:1 | 2 | ROL ² | 2018-12-11 | 3:2 | 1.5 |
| COO | 2024-02-20 | 4:1 | 4 | ROL ² | 2020-12-11 | 3:2 | 1.5 |
| CPRT ³ | 2017-04-11 | 2:1 | 2 | SHW | 2021-04-01 | 3:1 | 3 |
| CPRT ³ | 2022-11-04 | 2:1 | 2 | SMCI | 2024-10-01 | 10:1 | 10 |
| CPRT ³ | 2023-08-22 | 2:1 | 2 | SRE | 2023-08-22 | 2:1 | 2 |
| CSGP | 2021-06-28 | 10:1 | 10 | TECH | 2022-11-30 | 4:1 | 4 |
| CSX | 2021-06-29 | 3:1 | 3 | TJX | 2018-11-07 | 2:1 | 2 |
| CTAS | 2024-09-12 | 4:1 | 4 | TPL ² | 2024-03-27 | 3:1 | 3 |
| CVNA | 2026-05-08 | 5:1 | 5 | TPL ² | 2025-12-23 | 3:1 | 3 |
| DECK | 2024-09-17 | 6:1 | 6 | TSCO | 2024-12-20 | 5:1 | 5 |
| DXCM | 2022-06-13 | 4:1 | 4 | TSLA ² | 2020-08-31 | 5:1 | 5 |
| ETR | 2024-12-13 | 2:1 | 2 | TSLA ² | 2022-08-25 | 3:1 | 3 |
| EW | 2020-06-01 | 3:1 | 3 | TTD | 2021-06-17 | 10:1 | 10 |
| FAST ² | 2019-05-23 | 2:1 | 2 | WMT | 2024-02-26 | 3:1 | 3 |
| FAST ² | 2025-05-22 | 2:1 | 2 | WRB ³ | 2019-04-03 | 3:2 | 1.5 |
| FISV | 2018-03-20 | 2:1 | 2 | WRB ³ | 2022-03-24 | 3:2 | 1.5 |
| FTNT | 2022-06-23 | 5:1 | 5 | WRB ³ | 2024-07-11 | 3:2 | 1.5 |
| GOOG | 2022-07-18 | 20:1 | 20 | WSM | 2024-07-09 | 2:1 | 2 |
| GOOGL | 2022-07-18 | 20:1 | 20 | HRL | 2016-02-10 | 2:1 | 2 |
| HSIC ² | 2017-09-15 | 2:1 | 2 | IBKR | 2025-06-18 | 4:1 | 4 |
| ICE | 2016-11-04 | 5:1 | 5 | ISRG ² | 2017-10-06 | 3:1 | 3 |
| ISRG ² | 2021-10-05 | 3:1 | 3 | LNT | 2016-05-20 | 2:1 | 2 |
| LRCX | 2024-10-03 | 10:1 | 10 | MCHP | 2021-10-13 | 2:1 | 2 |
| MKC | 2020-12-01 | 2:1 | 2 | | | | |

² 2 splits no período · ³ 3 splits no período.

### 4.2 Reverse splits — 2 eventos

Grupamentos (preço sobe). Note `HSIC` aparece também em 4.1 (forward 2017) e 4.3 (spin-off
2019); cada evento é tratado independentemente.

| Símbolo | ex_date | Razão | factor | Evento |
|---|---|---|---|---|
| GE | 2021-08-02 | 1:8 | 0.125 | reverse split 1-for-8 |
| AMCR | 2026-01-15 | 1:5 | 0.2 | reverse split 1-for-5 |

### 4.3 Spin-offs e distribuições — 16 eventos, 13 símbolos

**Caso especial.** O feed `forward_splits`/`reverse_splits` da Alpaca **mistura spin-offs e
distribuições** com os splits limpos: o "fator de split" é, na verdade, o ajuste de preço
que o evento provoca (valor saindo da empresa). A matemática do back-adjustment é idêntica,
e como ambos criam um gap overnight, todos foram aplicados (decisão do usuário). Os fatores
"tortos" (≠ razão simples) denunciam esses casos.

| Símbolo | ex_date | factor | Evento (corporativo) |
|---|---|---|---|
| DHR | 2016-07-05 | 1.319 | spin-off da **Fortive** (FTV) |
| JCI | 2016-09-06 | 0.955 | fusão Johnson Controls + Tyco / spin **Adient** |
| HON | 2016-10-03 | 1.005 | distribuição |
| APD | 2016-10-03 | 1.081 | distribuição (Versum Materials, ~2016) |
| CAG | 2016-11-10 | 1.285 | spin-off da **Lamb Weston** (LW) |
| HLT | 2017-01-04 | 0.487 | spin-offs **Park Hotels** + **Hilton Grand Vacations** |
| HPE | 2017-04-03 | 1.335 | spin-merge **CSC** (→ DXC) |
| MET | 2017-08-07 | 1.122 | spin-off da **Brighthouse Financial** |
| HPE | 2017-09-01 | 1.289 | spin-merge **Micro Focus** (Software) |
| DOV | 2018-05-09 | 1.238 | spin-off da **Apergy** (depois ChampionX) |
| HON | 2018-10-01 | 1.011 | spin-off da **Garrett Motion** |
| HON | 2018-10-29 | 1.032 | spin-off da **Resideo** |
| HSIC | 2019-02-08 | 1.275 | spin-off / fusão **Covetrus** (animal health) |
| TT | 2020-03-02 | 1.289 | spin/fusão Ingersoll-Rand → **Trane Technologies** |
| FTV | 2020-10-09 | 1.195 | spin-off da **Vontier** |
| DELL | 2021-11-02 | 1.973 | spin-off da **VMware** |

> A maioria desses fatores é próxima de 1 (gap pequeno); só DELL (1.97), HLT (0.49), HPE,
> CAG e TT produzem gaps grandes o suficiente para contaminar o espectro de forma relevante.

## 5. Casos não aplicados / a vigiar

### 5.1 Ressalvas — `confirmed=0` (3 eventos, na base mas NÃO aplicados)

Reestruturações em que o salto observado na série não bate com o fator da API — sinal
e/ou magnitude divergem. Ficam registrados para auditoria, mas fora do ajuste automático.

| Símbolo | ex_date | factor | Observado | Esperado | Evento |
|---|---|---|---|---|---|
| EQT | 2018-11-13 | 0.800 | −0.624 | +0.223 | spin-off do midstream (Equitrans/ETRN) — sinal oposto |
| DD | 2019-06-03 | 0.4725 | +0.914 | +0.750 | separação DowDuPont → Dow / DuPont / Corteva |
| IR | 2020-03-02 | 0.8824 | +0.000 | +0.125 | reverse-merger Gardner Denver → Ingersoll Rand (sem gap na série) |

Se algum desses ativos for usado, tratar o evento manualmente (ou via winsorização do
retorno na data).

### 5.2 Saltos órfãos (sweep `--scan-orphans`)

A varredura de `|logret| > 0.18` não explicados por split é dominada por **moves reais**
(crash COVID mar/2020, CVNA na quase-falência de 2022 e recuperação, earnings). Não são
splits e não foram tocados. Um caso de **dado problemático**:

- **EXE** (Expand Energy, ex-**Chesapeake Energy / CHK**): passou por Chapter 11 em 2020-21
  e reuso de ticker. Apresenta log-retornos absurdos (>1, até +4.8) que **não** são splits.
  Os dados pré-2021 de EXE não são confiáveis — tratar à parte antes de usar.

## 6. Reproduzir

```bash
cd src_newest

# (1) popular splits.db a partir da Alpaca CA API + verificação D1
python fetch_splits.py                 # grava; --dry-run só relata; --scan-orphans p/ o sweep

# (2) aplicar o back-adjust -> data_parquet_adj/
python apply_splits.py --symbols AAPL  # prova de 1 símbolo
python apply_splits.py                 # lote completo (todos confirmados); --overwrite p/ refazer
python apply_splits.py --verify NVDA   # checa resíduo pós-ajuste

# (3) consumir no pipeline (default já é adjusted=True)
python -c "import decomp_pca; df=decomp_pca.load_ohlcv('NVDA','D1'); print(df.close.iloc[0])"
```

## 7. Limitações

- **Apenas `source=alpaca`.** B3/metatrader (PETR4, VALE3, BOVA11) não estão no tree atual;
  desdobramentos da B3 precisariam de outra fonte de corporate actions.
- **Dividendos em dinheiro não são ajustados** — só splits/spin-offs. Para retorno total
  com reinvestimento de dividendos seria preciso incluir `cash_dividend` da mesma API.
- O ajuste é **back-adjustment** (preços recentes = reais; histórico reescalado). Os preços
  pré-split no tree ajustado **não** são os preços nominais que se negociaram à época.
