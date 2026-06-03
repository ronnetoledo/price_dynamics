# Previsão de OHLC do próximo dia via vol prevista + forma espectral intradiária

Estratégia (continuação do [har_hl_forecast.py](../src_newest/har_hl_forecast.py)) para
prever o **OHLC do próximo dia** — foco no **range (High/Low)** — combinando a vol
prevista (HAR diário) com a *forma* da distribuição intradiária (decomposição
espectral). É um "martingale de distribuição": a vol é prevista, a forma é congelada.

Relacionado a [direcao_e_espectro.md](direcao_e_espectro.md) (direção = martingale,
range = previsível) e [step_embed_window.md](step_embed_window.md).

---

## 0. A ideia (enquadramento conceitual)

**Núcleo:** dada uma **vol projetada** e uma **forma de distribuição**, tem-se a
*distribuição preditiva inteira* do caminho do próximo dia → probabilidades de cada
patamar de preço, quantis de range, P(tocar barreira). A questão é *qual forma usar*.

**GBM e β.** "β=1" significa **difusão normal** (MSD ~ t). GBM é um caso específico
disso: **incrementos iid gaussianos**. Um processo pode ter β=1 e não ser GBM (ex.:
iid de cauda gorda). Os preços reais desviam de GBM por **três** vias independentes:
1. **difusão anômala** (β≠1) — incrementos autocorrelacionados;
2. **não-gaussianidade** — caudas gordas (mesmo com β=1);
3. **heterocedasticidade** — perfil-U intradiário, clustering de vol.

**Qual β?** O **β>1 medido no projeto é do processo ESPECTRAL** `X_t=cumsum(Kₜ−⟨K⟩)`,
não do preço (ver vol-historica-e-trace). O β do **caminho de preço** é ≈1 (diário,
martingale) ou **<1** (subdifusivo no intradiário, por mean-reversion de
microestrutura). Logo **o preço está perto de GBM na *difusão*** — desvia pela
não-gaussianidade (2) e pelo perfil-U (3), não por β anômalo.

**O que o HAR faz (e não faz).** O HAR-RV(-full) prevê o **nível de vol** (RV do
próximo dia) explorando o **clustering/memória longa da volatilidade** (lags 1/5/22d).
Ele **não** modela β nem a forma da distribuição, e **não** sabe do β>1 espectral —
são objetos diferentes. Arquitetura natural, **ortogonal**:
**σ̂ (HAR) cuida do *nível* × a forma cuida do *formato* da distribuição.** GBM usa
forma iid-gaussiana; a proposta deste doc usa forma **empírica** (das janelas
anteriores ou de uma média delas), escalada para σ̂.

**Proposta = martingale de distribuição.** A vol é prevista (parte previsível); a
*forma* é congelada na da(s) janela(s) anterior(es) (martingale). Duas leituras do
objetivo, com destinos diferentes (ver §4–5):
- **bater o *ponto* de range** (range/σ melhor que ∝σ̂) — exige que a forma *varie e
  persista* dia-a-dia → testado, **falha** (§4): a forma não persiste;
- **distribuição preditiva calibrada** (probabilidades de patamares/caudas/barreiras)
  — não precisa prever a variação da forma, só ter a forma média certa → **funciona**
  (§5), embora GBM+HAR já seja difícil de bater (CLT).

---

## 1. Por que o intradiário é o objeto certo para H/L

O High/Low de um dia é, por definição, o **máx/mín do caminho intradiário** que o
preço percorre — uma estatística de *caminho*, e o caminho é intradiário. A
decomposição **diária** descreve a dinâmica dia-a-dia e não gera a trajetória dentro
do dia; a **intradiária** carrega a forma desse caminho.

**O insight central (a melhoria sobre o estado atual):** o `har_hl_forecast` calibra
`û,d̂ ∝ σ̂` por OLS — assume **range ≈ constante × σ̂**, o modelo iid/Browniano
(`E[range] = √(8/π)·σ ≈ 1,596·σ`, relação de Parkinson). Mas o **ratio range/σ não é
constante**: intradiário **persistente** (tendência no dia) → caminho vai mais longe
→ range/σ alto; **mean-reverting** → caminho confinado → range/σ baixo. Dois dias com
a mesma vol diária e autocorrelações intradiárias diferentes têm ranges esperados
diferentes. A decomposição intradiária **prevê esse ratio** — é a hipótese testável.

---

## 2. Método (decisões travadas com o usuário)

- **Frequência da previsão:** diária. A única peça preditiva (vol) é diária — o
  HAR-full prevê `RV_{t+1}` do próximo dia (medida de retornos intradiários); ver §3.
- **(1) Nível:** `σ̂²_{t+1}` = vol prevista do próximo dia (HAR; full onde há features
  espectrais, RV caso contrário).
- **(2) Forma (martingale):** autocorrelação intradiária `ρ̂(τ)` recente persiste —
  `ρ̂_{t+1} = ρ̂_t`. Equivale a "congelar os autovetores" da covariância intradiária.
- **(3) Escala:** covariância dos retornos intradiários do próximo dia
  `Σ_r = (σ̂²_{t+1}/W)·Toeplitz(ρ̂)` (forma congelada, nível escalado à vol prevista
  ⇔ "autovalores corrigidos proporcionalmente p/ trace = σ̂²").
- **(4) Banda distribucional (opção B):** Monte Carlo — amostra caminhos intradiários
  `r ~ N(0, Σ_r)`, **centrados** (sem drift), `cumsum` ancorado na abertura → registra
  máx/mín/fech de cada caminho.
- **(5) Saída:** `Ĥ = abertura·exp(E[máx])`, `L̂ = abertura·exp(E[mín])`, banda por
  quantis; **fechamento ≈ abertura** (centrado/martingale — direção é imprevisível).
  Prevê-se **só o dia novo** (informação até o fim da janela atual).
- **Âncora (abertura):** janela desliza 1 dia no diário; a abertura da janela futura
  é o fechamento do 2º termo da janela atual.

**Teste decisivo:** R²_OOS do range e cobertura da banda H/L **vs. o baseline ∝σ̂**
(ratio fixo) do `har_hl_forecast`. Se o ratio variável (da forma espectral) vencer, a
decomposição intradiária agregou.

> **Fator confundidor a controlar:** o **perfil de vol intradiário** (U-shape:
> abertura/fechamento agitados) também infla o range e pode se misturar aos modos
> líderes. Sequência sugerida: primeiro a versão de autocorrelação **de-sazonalizada**
> (forma pura), depois testar se reintroduzir o perfil U melhora.

---

## 3. Inventário de dados (verificado em 2026-05-31)

A `forma` (autocorrelação) é computável **direto dos retornos brutos** — a estratégia
**não depende** do DB de decomposição, ficando timeframe-agnóstica. O DB de decomposição
só seria necessário para os autovetores "literais" e para as features do HAR-full.

| timeframe | bruto (`data_parquet`) | decomposição (`decomp_parquet`) | testável |
|---|---|---|---|
| **M1** | ✓ alpaca (~502 ativos) | ✓ **embed=70** (window=5×70=350) | já |
| **H1** | ✓ alpaca (~502 ativos) | ✗ ausente | já (forma crua + HAR-RV) |
| **M5** | ✗ (só CSVs B3 não-convertidos) | ✗ ausente | após converter 3 ativos B3 |
| D1 | ✓ | ✓ embed=70 | é o alvo, não intraday |

- **Padrão de decomposição = embed=70** (não 20). `embed=20` só num sweep especial.
  Autovetores dim 70 → `ρ(τ)` até lag 70 (suficiente; estende-se `ρ≈0` além, pois a
  autocorrelação intradiária morre antes disso). A sessão tem ~390 barras M1.
- **`data_parquet` só tem alpaca** {M1, H1, D1, W1}. Sem M5; B3/metatrader não
  convertido (CSVs M5 existem em `src_newest/B3_DATA/` para PETR4/VALE3/BOVA11).

### Resolução do H/L (viés do teste por timeframe)
O H/L diário realizado é o extremo tick-a-tick. Um caminho em barras `Δ` tem
`~sessão/Δ` pontos: **M1 ~390** (resolve bem), **M5 ~78**, **H1 ~7** (subestima o range
por construção). O teste "qual tf é melhor" favorece o fino para H/L — M1 deve vencer;
o interessante é quanto H1/M5 perdem e se a forma espectral compensa.

---

## 4. Teste de viabilidade (feito antes de implementar) — NEGATIVO

Antes de construir o Monte Carlo, mediu-se se o ratio `k = ln(H/L)/√RV` (a quantidade
que a forma deveria prever além de `∝σ̂`) **varia e é previsível**. 5 ativos alpaca,
M1 e H1, ~2600 dias cada. Forma = autocorr lag-1 intradiária `ρ1`.

| (M1) | cv(k) | corr(k,ρ1) contemp. | persist. ρ1 (acf1) | **corr(k_{t+1}, ρ1_t)** | corr(k_{t+1}, ⟨ρ1⟩₂₂) |
|---|---|---|---|---|---|
| AAPL | 0.34 | 0.31 | 0.14 | **0.04** | 0.09 |
| MSFT | 0.31 | 0.30 | 0.13 | **0.07** | 0.04 |
| NVDA | 0.31 | 0.29 | 0.11 | **0.07** | 0.12 |
| JPM | 0.32 | 0.28 | 0.05 | **0.02** | 0.04 |
| KO | 0.32 | 0.31 | 0.01 | **0.02** | 0.01 |

H1: todas as correlações defasadas ≈ 0.

**Resultado:** `k` varia (cv≈0.32) e é **contemporaneamente** dirigido pela forma
intradiária (~0.30 corr ⇒ ~9% da variância) — o mecanismo é real. **Mas não é
previsível:** a forma de ontem/suavizada carrega quase nada sobre o `k` de amanhã
(corr defasado 0.02–0.12, R²<1.5%) e o próprio `k` não persiste (acf≈0). A forma que
importa é a **do próprio dia**, desconhecida na previsão.

**Conclusão:** o "martingale da forma" falha — a forma não persiste dia-a-dia.
Out-of-sample a estratégia colapsa para `range ∝ σ̂` (ratio ≈ constante por ativo),
que é **o que o `har_hl_forecast` já faz**. Construir o Monte Carlo daria ~1% de R²
sobre o baseline, inconsistente entre ativos → **não implementar**. Mesma lição do
detector de regime: a estrutura espectral é informativa *contemporaneamente*, não
*preditivamente*; o único grau de liberdade previsível é a **vol** (σ̂).

> Fio solto (não fecha a porta): `ρ1` é descritor cru. Variance-ratio/Hurst
> intradiário ou o perfil-U poderiam ser mais persistentes — mas o perfil-U é
> ~constante por ativo (prevê a *média* de `k`, que o baseline já calibra), então a
> expectativa é que não mude o veredito.

---

## 5. Pivô: distribuição preditiva do caminho (não o ponto de range) — FUNCIONA

Implementado em [ohlc_dist.py](../src_newest/ohlc_dist.py). Em vez de bater o *ponto*
de range (morto, §4), constrói a **distribuição preditiva** do caminho intradiário do
próximo dia (probabilidades de patamares, caudas, range, P(barreira)), escalada pela
σ̂ do HAR-RV. Modelos aninhados, todos com RV total = σ̂²_{t+1}:
`M0 (GBM)` plano+gaussiano · `M1 (+perfil-U)` · `M2 (+caudas)` bootstrap dos retornos
intradiários padronizados. Truque scale-free: banco de caminhos de RV unitária gerado
1× (forma do treino), escalado por σ̂ por dia. OOS: forma no treino, σ̂ expansível,
calibração medida no teste (PIT-KS, cobertura 95%, CRPS).

### Resultado — POSITIVO (o objeto funciona)
Diário (base M1, W~400–700), 4 ativos, ~670 dias de teste cada:

| | cov95 (C) | PIT-KS M0 (GBM) | PIT-KS M2 (forma) | CRPS_range M0→M2 |
|---|---|---|---|---|
| AAPL | 95% | 0.078 | 0.066 | 7.28 → 7.16 |
| MSFT | 96% | 0.076 | 0.056 | 6.86 → 6.71 |
| NVDA | 93% | 0.031 | 0.017 | 9.81 → 9.79 |
| KO | 98% | 0.077 | 0.061 | 4.19 → 4.04 |

- **GBM + HAR já é bem calibrado** (cobertura ~95%, PIT-KS ~0.03–0.08). A ideia "dada
  σ̂, probabilidades de cada patamar" **funciona** — é um entregável útil (risco,
  barreiras), validado por calibração.
- **A forma empírica (M2) melhora pouco mas consistente** o PIT-KS (calibração
  central) e o CRPS de range (~1–3%). Refinamento menor, não divisor de águas.
- **Por quê GBM basta no diário: CLT.** O close = soma de ~500–700 retornos M1 →
  quase gaussiano; as caudas individuais do M1 se lavam na agregação.
- **H1 (W~15):** cobertura cai p/ ~90% (bandas apertadas) — mais **ruído de medição da
  vol** (RV de 15 barras) que problema de forma; ganhos do M2 inconsistentes
  (ajuda high, piora low/range). **Diário (base M1) é o ponto ótimo.**

### Conclusão
Diferente dos becos anteriores, **este objeto funciona**: distribuição preditiva
calibrada do caminho diário. Mas "bater o GBM" é difícil — GBM+HAR já é bom (CLT), e a
forma empírica adiciona margem pequena. **O ganho real é a vol (σ̂); a forma é 2ª
ordem** — coerente com o tema do projeto: a vol é o grau de liberdade previsível.

> Refinamento possível (não perseguido): o bootstrap iid destrói a autocorrelação
> intradiária; um **block bootstrap** (preserva a forma do caminho) poderia melhorar o
> range/low do M2 — mas, dada a margem pequena, baixa prioridade.
