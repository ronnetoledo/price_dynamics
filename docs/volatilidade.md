# Volatilidade: a histórica é o traço espectral, e o que os estimadores de range revelam

Documento de referência sobre **o que a volatilidade clássica mede**, sua
identidade exata com a decomposição espectral PCA, e o que a comparação com os
estimadores de range (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
revela sobre os dados. Relacionado a [direcao_e_espectro.md](direcao_e_espectro.md)
e [step_embed_window.md](step_embed_window.md). Scripts:
[src_newest/vol_estimators.py](../src_newest/vol_estimators.py),
[src_newest/price_msd.py](../src_newest/price_msd.py) e
[src_newest/har_vol_forecast.py](../src_newest/har_vol_forecast.py).

---

## 1. Volatilidade histórica = desvio-padrão dos log-retornos, sob GBM

A vol histórica canônica é o desvio-padrão amostral dos log-retornos `rₜ = ln(Pₜ/Pₜ₋₁)`,
anualizado:

```
σ̂ = sqrt( (1/(n−1)) Σ (rₜ − r̄)² ) · sqrt(A)        (A = períodos/ano)
```

A construção pressupõe o **movimento browniano geométrico** (GBM): `dP = μP dt + σP dW`,
i.e. log-retornos i.i.d. gaussianos com variância ∝ Δt. Por Itô,
`d(ln P) = (μ − σ²/2) dt + σ dW` — log-preço é browniano aritmético. Duas
consequências fundam toda a teoria clássica:

1. **Variância aditiva no tempo** → desvio ∝ sqrt(Δt) (a "regra da raiz do tempo",
   que justifica o `·sqrt(A)`). É exatamente **difusão normal, β = 1** no MSD do
   log-preço. *O GBM assume β = 1 por construção.*
2. **σ constante** — um único escalar caracteriza o ativo na janela.

Os dois pressupostos são onde o conceito quebra (heterocedasticidade, caudas
pesadas, memória, saltos) — e onde a decomposição espectral entra.

---

## 2. Identidade exata: σ²_histórica = trace/d

Cada janela embeda os retornos numa matriz de Hankel de dimensão `d = embed`; a
covariância é `C = Xcᵀ·Xc/(n_samp−1)`. As d colunas são cópias atrasadas da mesma
série, então cada entrada diagonal de C é a variância marginal ≈ σ². Logo:

```
trace = Σλₖ = Σ_{j=1}^{d} Var(coluna j) ≈ d · σ²
     ⇒   σ²_histórica = trace / d              (variância por período)
     ⇒   vol = sqrt(trace/d) · sqrt(A)
```

É o mesmo `σ² = trace/d` usado como piso de ruído no limiar MP (ver
[direcao_e_espectro.md](direcao_e_espectro.md) §4d). **Não é analogia — é igualdade.**

**Confirmação empírica** (D1, step=20, embed=70, 96 janelas; vol anualizada %):

| | MSFT (sem split) | AAPL (com split 2020) |
|---|---|---|
| `sqrt(trace/d)` (espectral) | **27.81** | 57.26 |
| `var(retornos)` (alvo) | 27.80 | 56.66 |
| Close-to-Close | 27.82 | 56.61 |

ratio spec/varR ≈ 0.985, Spearman(spec, CC) ≈ 0.97. O **traço da decomposição é a
variância histórica** — a vol clássica é a projeção 0-dimensional do espectro; os
autovalores resolvem essa mesma energia modo a modo.

---

## 3. Estimadores de range: ideia e calibração

O close-to-close usa **um ponto por barra** (o fechamento). O range intradiário
(máxima H, mínima L) tapa a **estatística de valores extremos** do caminho — muito
mais informação sobre σ por barra. Os clássicos (O,H,L,C):

```
Parkinson:        σ²_P  = (ln H/L)² / (4 ln 2)
Garman-Klass:     σ²_GK = 0.5(ln H/L)² − (2 ln2 − 1)(ln C/O)²
Rogers-Satchell:  σ²_RS = ln(H/C)ln(H/O) + ln(L/C)ln(L/O)      (independe de drift)
Yang-Zhang:       σ²_YZ = σ²_overnight + k·σ²_open + (1−k)·σ²_RS  (drift- e gap-indep.)
```

A constante `1/(4 ln 2)` de Parkinson vem do segundo momento do range de um
browniano sem drift: `E[R²] = 4 ln2 · σ²T` (Feller, via reflexões nas duas
barreiras max/min; o `ln 2` é a série alternada Σ(−1)^{k+1}/k que aparece ao
integrar r²·f_R). Não-viesado por construção sob GBM; eficiência ~5× (Parkinson),
~7–8× (GK), ~14× (YZ) vs. close-to-close.

---

## 4. O que a comparação revela (achados)

### 4a. O gap range-vs-close é um detector de splits não ajustados

Park/GK/RS medem **só a variância intradiária** — **nunca cruzam a fronteira
close→open**. Splits não ajustados (ver memória de splits) injetam o salto
*exatamente* nessa fronteira. Resultado:

| estimador | cruza overnight? | MSFT | AAPL |
|---|---|---|---|
| Close-to-Close, `trace/d`, YZ | **sim** | 27.8 | **57** (contaminado) |
| Parkinson / GK / RS | **não** | 21.8 | **22.5** (imune) |
| gap (close − range) | — | ~6 pp | ~35 pp |

O split 4:1 de AAPL (2020-08-31, `r = −1.35`) sozinho infla a vol de todo o
histórico de 28.9% → 51%. Como `trace/d` é close-based, ele **herda** a
contaminação — é o mecanismo concreto pelo qual splits injetam modos espúrios na
PCA. **Rodar Parkinson em paralelo flagra ações corporativas não ajustadas sem
precisar de lista de splits:** gap normal ≈ overnight (~6 pp em D1); gap anômalo
denuncia o salto.

### 4b. Mesmo sem split, range < close: o overnight

No controle limpo (MSFT) Park/GK/RS ficam ~6 pp abaixo do CC. Isso **não é erro** —
é a variância overnight (close→open) que os estimadores intradiários excluem:
`(21.8/27.8)² ≈ 61%` da variância diária é intradiária, ~39% overnight. Só o
**Yang-Zhang** reincorpora esse termo (27.6% ≈ CC). Range estimators são ainda
~4–5× menos ruidosos por janela (eficiência confirmada).

---

## 5. Quando β ≠ 1: viés dos estimadores e da anualização

Toda a calibração (incl. `1/4ln2`) pressupõe browniano (H = ½, β = 2H = 1). Sob
fBm, `E[R_T²] = c(H)·σ²·T^{2H}`, `c(½) = 4ln2`. O viés entra em dois lugares:

- **(a) constante intra-barra:** `H < ½` (anti-persistente, e.g. bid-ask bounce)
  → range grande p/ dado σ → Parkinson **superestima**; `H > ½` (trending) →
  **subestima**. A **discretização** dos ticks (H/L observados < extremos do
  caminho contínuo) puxa para baixo e **cancela parcialmente** o efeito de
  microestrutura — daí Park/GK serem robustos na prática.
- **(b) anualização:** `·sqrt(A)` assume aditividade (β=1). Sob fBm a variância de
  τ-períodos cresce como `τ^{2H}` → o fator correto é `A^H`. `H>½` subestima
  horizonte longo; `H<½` superestima.

**β_preço medido** (variância-tempo `V(τ)=Var(ln P_{t+τ}−ln P_t) ∝ τ^β`,
[price_msd.py](../src_newest/price_msd.py), D1, fit τ∈[5,120]):

| ativo | β_preço | leitura |
|---|---|---|
| MSFT | **0.907 ± 0.003** | subdifusão leve |
| AAPL (split winsorizado) | **0.883 ± 0.009** | subdifusão leve |

`V(120)/(V(1)·120) ≈ 0.55–0.63 < 1` → variância **sub-aditiva**: **sqrt(time)
superestima** a vol de horizonte longo (~+15–34% em 120 dias). Anti-persistência
leve do log-preço diário.

> **Dois β distintos (a ressalva de [direcao_e_espectro.md](direcao_e_espectro.md)):**
> o β_preço (caminho de retornos) é **< 1** (subdifusivo), enquanto o **β
> espectral** do pipeline — do processo `Xₜ = cumsum(Kₜ−⟨K⟩)` — é **> 1**
> (superdifusivo, persistência de *regime*). São objetos diferentes: o preço
> reverte de leve; a dinâmica de regime persiste. Os estimadores de vol e a
> anualização dependem do β_preço, não do β espectral.

---

## 5b. Os quatro β no mesmo ativo (MSFT, sem split)

Confronto direto, mesmo ativo e timeframe, step=20, embed=70. Espectrais do
`analysis_results.db`; β_preço de [price_msd.py](../src_newest/price_msd.py);
figura dos quatro MSDs sobrepostos em
[src_newest/msd_4curvas_MSFT_M1_s20_e70.png](../src_newest/msd_4curvas_MSFT_M1_s20_e70.png)
([make_msd_figure.py](../src_newest/make_msd_figure.py)).

| β | **M1** (75.706 jan.) | **D1** (96 jan.) | objeto |
|---|---|---|---|
| **β_preço** | **0.958 ± 0.000** | 0.89–0.91 ± 0.004 | caminho de preço (1º mom., ímpar) |
| **β_struct** | **1.116 ± 0.020** | 1.103 ± 0.211 | subespaço estrutural (sinal) |
| **β_total** | **1.628 ± 0.007** | 1.170 ± 0.181 | espectro completo |
| **β_bulk** | **1.690 ± 0.006** | 1.280 ± 0.105 | bulk (ruído/energia) |

Ordenação em todo TF: `β_preço < β_struct < β_total ≲ β_bulk`. No D1 os
espectrais têm σ_β ~0.2 (96 janelas só) — o **M1 é a comparação significativa**.

- **Clivagem 1º/2º momento, quantitativa:** preço **subdifusivo** (0.96 < 1) vs.
  estrutura espectral **fortemente superdifusiva** (1.6–1.7). `Δβ(preço,bulk) ≈ 0.7`
  no M1. Direção (ímpar, imprevisível) vs. energia/regime (par, memória longa).
- **β_bulk dirige β_total e ≈ memória longa da volatilidade:** os autovalores de
  bulk são o mar MP ≈ σ², então `C_bulk` rastreia o *nível de vol*; sua dinâmica
  herda o volatility clustering (β_bulk o mais alto). Como o bulk é ~87% da
  energia, `β_total ≈ β_bulk`.
- **β_struct é o mais browniano e estável** (~1.1 em M1 e D1): os modos de sinal
  difundem quase normalmente; a memória longa vive no bulk, não no sinal.
- **Regime tem memória bem mais forte intraday:** β_total cai de 1.63 (M1) → 1.17
  (D1); β_preço fica ~estável (0.89–0.96).

> Nota de reprodução: a figura subamostra para ~12k janelas (stride 6), o que
> reduz levemente os slopes medidos (struct 1.05 / total 1.53 / bulk 1.56) vs. os
> valores full-sample do DB acima; a ordenação e a clivagem se preservam.

### 5c. Dois eixos: "aleatório no espaço" ≠ "aleatório no tempo"

A leitura intuitiva (struct = sinal, bulk = ruído) é correta **no instante**, mas
não se transfere para a *dinâmica*. São dois eixos independentes, e eles se
**cruzam invertidos**:

| eixo | struct | bulk |
|---|---|---|
| **instantâneo** (forma do espectro num dado t) | **sinal / estruturado** (modos > λ₊) | **ruído / aleatório** (mar MP ≈ σ²) |
| **temporal** (β, persistência da dinâmica) | β ≈ 1.1 → **quase browniano** | β ≈ 1.7 → **superdifusivo / memória longa** |

- O **bulk é o "ruído" no instante** (autovalores sem estrutura), **mas sua
  evolução é a menos aleatória** (β mais alto). Razão: `Σλ_bulk ≈ energia total ≈
  nível de volatilidade`, e o nível de vol **agrupa no tempo** (volatility
  clustering) → o bulk herda a memória longa.
- O **struct é o "sinal" no instante**, **mas sua dinâmica é a mais parecida com
  passeio aleatório** (β ≈ 1): os modos estruturais aparecem/desaparecem quase
  brownianamente.

> **Cuidado com "difusivo":** difusão *normal* (β = 1) **é** o caso
> browniano/aleatório; superdifusão (β > 1) é o **oposto** de aleatório — é
> persistência/memória. Portanto o bulk não é "a parte aleatória difusiva"; é a
> parte **superdifusiva persistente**.

**Síntese honesta:** a *direção* do preço é aleatória (1º momento, β≈0.96); a
*volatilidade* não é (2º momento, tem memória). Dentro dela, o **nível de vol
(bulk)** carrega a memória longa, enquanto os **modos estruturais (struct)** se
movem quase como passeio aleatório. O que é ruído no espaço é o que tem mais ordem
no tempo — essa troca de eixos é o que distingue a decomposição de medir só o σ
escalar.

---

## 5d. Vale para prever volatilidade? Teste OOS (HAR)

Teste preditivo out-of-sample (janela expansível, alvo = variância realizada
diária futura `RV_{t+h}`, M1 agregado ao dia, retornos winsorizados p/ neutralizar
split). Benchmarks: random-walk, EWMA(0.94) e **HAR-RV** (padrão-ouro de RV).
Features da decomposição: `σ²_bulk`, `trace/d`, e as de **regime** (`m`, entropia,
`f_struct`). MSFT M1, ~2.345 previsões, R²_OOS sobre log-RV:

| modelo | h=1d | h=5d | h=22d |
|---|---|---|---|
| EWMA | 0.04 | 0.11 | −0.11 |
| **HAR-RV** (benchmark) | 0.257 | 0.383 | 0.241 |
| HAR-bulk | 0.285 | 0.395 | 0.233 |
| HAR-RV+bulk | 0.295 | 0.394 | 0.228 |
| HAR-regime (m,S,f contemporâneos) | 0.031 | 0.056 | 0.045 |
| HAR-regimeHAR (m,S,f c/ lags 5d/22d) | 0.126 | 0.248 | 0.231 |
| HAR-full (RV+bulk+regime) | 0.308 | 0.423 | 0.272 |
| **HAR-full-HAR** (RV+bulk+regimeHAR) | **0.310** | **0.440** | **0.324** |

**Três resultados:**

1. **σ²_bulk sobre RV: ganho marginal.** `corr(bulk, RV) = 0.79 ≈ corr(trace, RV)`
   — o bulk *é* quase todo o nível de vol, então prever com bulk ≈ prever com RV.
   O HAR-RV+bulk supera o HAR-RV só por ~+3 pontos no diário. O bulk-nível não é o
   diferencial.
2. **A estrutura é fraca contemporânea, forte com escalas temporais.** As features
   de regime *contemporâneas* preveem mal sozinhas (R² 0.03–0.06). Mas dando a elas
   a **estrutura HAR (médias 5d/22d)**, a estrutura *sozinha* (sem nível!) sobe para
   **0.23 no mensal — quase o HAR-RV (0.241)**. As features de regime são lentas
   (regime muda devagar): suas médias multi-escala capturam a tendência, o valor
   contemporâneo é ruído. `m` e `f_struct` correlacionam **negativamente** com a RV
   (−0.16, −0.14: alta vol = espectro difuso — coerente com §5b) → info **ortogonal
   ao nível**.
3. **O modelo completo (HAR-full-HAR) domina.** Bate o HAR-RV em todo horizonte, e
   o ganho **cresce com o horizonte**: +5 pts (h=1), +6 (h=5), **+8 no mensal**
   (0.324 vs 0.241). QLIKE também é o menor em todos os h. O regime estrutural é
   especialmente valioso em horizonte longo, onde a persistência da RV se esgota.

> **Atenção — MSFT não é representativo.** Os números acima são *um* ativo. O teste
> multi-ativo (§5e) mostra que o sinal **standalone** do regimeHAR (R² 0.23 no
> mensal aqui) **não generaliza**: na média de 46 ativos, regime-sozinho dá R² ≈ 0
> ou negativo. O valor robusto do regime é apenas como **complemento** à RV, e a
> versão com lags (regimeHAR) **não supera** o regime contemporâneo no painel —
> chega a overfittar. A conclusão honesta está em §5e.

---

## 5e. Robustez multi-ativo: o que generaliza (e o que não)

Mesmo teste OOS em **46 ativos S&P500** (M1, s20 e70; AAPL excluído por Parquet
corrompido). R²_OOS(logRV) **médio** e **fração de ativos em que cada modelo bate o
HAR-RV**:

| modelo | h=1d média | >HAR-RV | h=5d média | >HAR-RV |
|---|---|---|---|---|
| **HAR-RV** (benchmark) | 0.348 | — | 0.402 | — |
| HAR-bulk | 0.346 | 0.48 | 0.410 | 0.43 |
| HAR-regime (m,S,f contemp.) | **−0.040** | 0.00 | **−0.066** | 0.00 |
| HAR-regimeHAR (m,S,f c/ lags) | **−0.004** | 0.00 | **−0.013** | 0.00 |
| HAR-RV+bulk | 0.366 | 0.72 | 0.417 | 0.65 |
| HAR-RV+regime | 0.354 | 0.85 | 0.407 | 0.83 |
| **HAR-full** (RV+bulk+regime) | **0.375** | **1.00** | **0.426** | **0.96** |
| HAR-full-HAR (RV+bulk+regimeHAR) | 0.373 | 1.00 | 0.419 | 0.67 |

**Veredito honesto (corrige a impressão do MSFT):**

- **Regime sozinho NÃO prevê vol.** Em 46 ativos, regime-only (contemporâneo *ou*
  com lags HAR) tem R² médio **≈ 0 ou negativo** e **nunca** bate o HAR-RV. O 0.23
  do regimeHAR no MSFT (§5d) era **idiossincrático**, não universal. Coerente com a
  tese: a estrutura é cega ao nível, e o nível é o que prevê vol.
- **bulk sozinho ≈ RV** (bate o HAR-RV em só 43–48% — cara-ou-coroa). O nível
  espectral não agrega sobre a RV.
- **O valor robusto do regime é como COMPLEMENTO.** O **HAR-full** (RV+bulk+regime
  *contemporâneo*) bate o HAR-RV em **100% (h=1) e 96% (h=5)**, ganho médio ~+0.03.
  Esse é o resultado universal.
- **Os lags HAR do regime overfittam no painel.** HAR-full-HAR ≈ HAR-full em h=1
  (ambos 100%), mas **pior e menos robusto em h=5** (0.419, só 67% vs 0.426, 96%) —
  os 9 regressores de regimeHAR adicionam variância OOS em ativos ruidosos. **O
  regime contemporâneo (3 features) é a escolha melhor e mais robusta.**

> **Resultado para o PRL:** a decomposição **não fornece um preditor de vol
> independente** (regime-sozinho ≈ 0; bulk ≈ RV), mas as **3 features de regime
> contemporâneas** (`m`, entropia, `f_struct`) adicionam um ganho **pequeno (~+0.03
> R²) porém universal em sinal** (100% dos ativos em h=1) sobre o padrão-ouro
> HAR-RV. É um complemento ortogonal robusto, não um modelo autônomo — e a versão
> "esperta" com lags não melhora. Afirmação defensável e honesta: *o 2º momento
> resolvido adiciona uma dimensão que a RV não vê, mas só junto com a RV.*

---

## 5f. Da vol prevista para High/Low futuros

Aplicação direta: usar σ̂_{t+1} do HAR-full para prever H e L do dia seguinte.
Script [src_newest/har_hl_forecast.py](../src_newest/har_hl_forecast.py); figura
[hl_forecast_MSFT_M1.png](../src_newest/hl_forecast_MSFT_M1.png).

**Separação largura/centro** (a assimetria 1º/2º momento aplicada a H/L):
```
H, L = centro (≈ C_t, martingale — IMPREVISÍVEL) ± semi-amplitude (∝ σ̂ — PREVISÍVEL)
```
Calibração OOS (absorve discretização e c(H) do β≠1), prevendo o dia t+1 a partir
do close de t:
```
û = ln(H/C),  d̂ = ln(C/L)  regredidos sobre σ̂   →   Ĥ = C_t·e^û,  L̂ = C_t·e^{−d̂}
ln(Ĥ/L̂) = range previsto ≈ 1.6·σ̂  (E[range] do BM = √(8/π)·σ; ver §3 e item c(H))
```

**Resultados OOS (MSFT M1):**

| (a) R²_OOS do log-range | |
|---|---|
| RW (range de ontem) | −0.17 |
| MA22 | 0.147 |
| σ̂ HAR-RV | 0.303 |
| **σ̂ HAR-full** | **0.321** |

- **O range é previsível pela vol** (R²~0.32, muito acima do ingênuo), e o **edge
  do regime propaga** (HAR-full > HAR-RV, +1.7 pts) — a melhoria de vol vira
  melhoria de range.
- **(b) Níveis H/L:** `R²(H)=R²(L)=0.999` é **trivial** (H,L dominados pelo nível
  de preço; `Ĥ≈C_t`). A skill real é o **range**, não o nível. Cobertura pontual
  `[L̂,Ĥ]⊇[L,H]` = 26% (Ĥ,L̂ são *médias*, não limites); **banda calibrada c=2.25
  → cobertura OOS 81.6%** (alvo 80%, a calibração se mantém fora da amostra).
- **(c)** Ordenação: `HAR-full > HAR-RV > MA22 ≫ RW`.

> **Leitura:** prevê-se o **envelope, não o caminho**. A vol (com o edge universal
> do regime) dá a *largura* da banda de amanhã; o *centro* fica preso no close de
> hoje porque a direção é martingale. Na figura, a banda alarga nos sell-offs (alta
> vol prevista) e contém ~82% dos H/L realizados.

---

## 6. TL;DR

- **Vol histórica = sqrt(trace/d)** — igualdade exata (ratio ≈ 0.99, ρ ≈ 0.97).
- O **gap range-vs-close** detecta **splits não ajustados** (imunes ao overnight)
  e mede o peso do **overnight** (~40% da variância diária).
- **Yang-Zhang** é o único range estimator que reincorpora o overnight → casa com
  o close-to-close.
- **β_preço ≈ 0.88–0.96 < 1** (subdifusão leve) → **sqrt(time) superestima** a vol
  de horizonte longo. É o *oposto* do β espectral (> 1) — processos distintos.
- **No mesmo ativo (MSFT M1):** `β_preço 0.96 < β_struct 1.12 < β_total 1.63 ≲
  β_bulk 1.69` — Δβ(preço,bulk) ≈ 0.7. O bulk (≈ nível de vol) carrega a memória
  longa; o sinal estrutural difunde quase normalmente.
- **Dois eixos cruzados:** o bulk é ruído no *instante* mas o mais persistente no
  *tempo*; o struct é sinal no instante mas o mais browniano no tempo. Ruído no
  espaço = ordem no tempo.
- **Previsão de vol (OOS, 46 ativos):** o bulk-nível ≈ RV (não agrega), mas as
  features de **regime** (`m`, entropia, `f_struct`) fazem o HAR-full **bater o
  HAR-RV em 100% dos ativos (h=1)** — ganho pequeno (~+0.03 R²) e universal. O
  valor preditivo está no 2º momento *resolvido*, não no escalar.
- **High/Low futuros:** σ̂ do HAR-full prevê o **range** (R²~0.32, ≫ ingênuo; edge
  do regime propaga), mas só a *largura* — o *centro* é martingale. Banda calibrada
  contém ~82% dos H/L realizados OOS. Prevê-se o envelope, não o caminho.
