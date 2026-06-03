# Detecção de regime em streaming (tick-a-tick) para HFT

Documento de design da versão **online/causal** do detector espectral de regime.
É a contraparte em tempo real do `build_causal_regime_matrix` do pipeline batch
([experimento_PCA_4D_3.py](../src_newest/experimento_PCA_4D_3.py)): mesma física
(Hankel → C → espectro → limiar de Marchenko-Pastur), porém com **estado
incremental** atualizado a cada barra (alvo: 1 barra/segundo) em vez de recortar
janelas. Objetivo: disparar uma mudança de regime **o mais cedo possível**, com
falso-alarme controlado, dentro de um orçamento de latência fixo.

Relacionado a [step_embed_window.md](step_embed_window.md) (de onde vêm `embed`,
`window`, `C`, `λ₊`) e a [direcao_e_espectro.md](direcao_e_espectro.md) (por que o
espectro **não** dá direção — ver §5).

---

## 0. Por que não basta re-rodar o pipeline batch a cada tick

A cada barra nova, a covariância da janela seguinte é a anterior perturbada por um
update **de posto baixo** (ver [step_embed_window.md](step_embed_window.md), seção
sobre sobreposição). Refazer `eigh(C)` do zero a cada segundo é O(m³) jogado fora,
e o sliding window duro tem dois defeitos para uso live:

- o *downdate* do snapshot que sai (`−s₀·s₀ᵀ`) pode quebrar a positividade em ponto
  flutuante e obriga a guardar o buffer inteiro da janela;
- o descarte abrupto de s₀ injeta um degrau espúrio na trajetória espectral.

Ambos se resolvem trocando a janela dura por **esquecimento exponencial** (§2).

---

## 1. Arquitetura: três camadas, três latências

O trade-off é físico e inescapável: **quanto mais cedo se dispara, mais falsos
positivos**. Em vez de um detector único, usa-se uma cascata da camada mais
rápida/ruidosa para a mais lenta/confiável, fundindo as decisões (§7).

```
tick r_t
  │
  ├─ Camada 1 (latência 1 tick):  resíduo de inovação   → estado WATCH
  ├─ Camada 2 (latência ~1/α):    tracker de autopares  → sinais espectrais
  └─ Camada 3 (decisão):          change-point (CUSUM)  → estado CONFIRMED
```

---

## 2. Decisão de design nº 1 — covariância EWMA (posto-1 puro)

Substitui a janela deslizante dura por média móvel exponencial:

```
C_t = (1−α)·C_{t−1} + α·sₙ·sₙᵀ          sₙ = snapshot novo (últimos `embed` retornos)
```

- update de **posto 1 puro** (só soma → sempre PSD); sem downdate instável;
- "janela efetiva" ≈ `1/α` — botão **contínuo** de responsividade;
- esquecimento gradual → sem degrau quando dados velhos saem;
- estado O(m²) fixo, sem ring buffer da janela inteira (só os últimos `embed`
  retornos para montar sₙ).

> **Variante de dois tempos:** rodar α_f (rápido) e α_s (lento) em paralelo e
> monitorar a divergência entre os dois espectros é, por si só, um detector de
> mudança quase de graça (um "MACD espectral").

---

## 3. Camada 1 — resíduo de inovação (latência de 1 tick)

A peça que dá a antecipação. **Antes** de atualizar C, projeta-se o snapshot que
chega no subespaço estrutural *atual* `Q` (top-p autovetores) e mede-se o resíduo:

```
ê_t = ‖ sₙ − Q·Qᵀ·sₙ ‖ / ‖ sₙ ‖        ∈ [0,1]
```

`Q` resume "como o mercado vinha se movimentando". Se a barra nova é bem explicada
por esse subespaço, `ê_t` é pequeno. Quando o regime começa a virar, **a barra
nova é a primeira a ficar inconsistente com o passado** → `ê_t` salta no mesmo
tick, muito antes de o autovalor integrar a mudança (que leva ~`1/α` ticks). É o
limite inferior de latência: 1 tick. O custo é falso-alarme — `ê_t` sozinho é
ruidoso, por isso ele só **arma** (WATCH), não opera.

Equivalente estatístico melhor calibrado (distância de Mahalanobis do snapshot):
`T²_t = sₙᵀ·C⁻¹·sₙ`. Mas inverter C é caro/instável no bulk; na prática `ê_t` no
subespaço de topo + a energia no bulk separadamente dão o mesmo sinal sem inverter
nada.

---

## 4. Camada 2 — tracker de autopares (subspace iteration com warm-start)

Como `C_t ≈ C_{t−1}`, partir dos autovetores anteriores e fazer **1 iteração** já
converge. Rastreia-se só os top-`p` modos (`p ≪ m`: nº esperado de modos
estruturais + folga), o que evita o gargalo O(m³) da reconstrução de autovetores e
a degenerescência do bulk:

```
def step(C, Q, p):
    Z = C @ Q                 # O(m²p)
    Q, _ = qr(Z)              # O(mp²)  reortonormaliza
    H = Q.T @ (C @ Q)         # p×p
    θ, U = eigh(H)            # O(p³)   Rayleigh–Ritz
    return θ[::-1], Q @ U[:, ::-1]
```

Custo **O(m²·p) por tick**, determinístico (sem cauda de latência) — requisito de
HFT. Re-ancorar com um `eigh(C)` completo a cada N ticks (ex.: N=1000) zera a
deriva acumulada; pode rodar fora do caminho crítico (thread separado).

---

## 5. Os sinais espectrais e o que cada um detecta

Da camada 2 saem escalares causais. Cada um capta um *tipo* de transição
(alinhados ao que o pipeline batch já calcula):

| Sinal | Fórmula | Detecta |
|---|---|---|
| Traço | `tr C = Σλ` | regime de **volatilidade** (= vol clássica, vol²=trace/d) |
| λ₁ | maior autovalor | concentração de energia / início de **tendência** |
| Contagem `m` | `#{λ_k > λ₊}` | mudança de **dimensionalidade** estrutural |
| Entropia | `S = −Σ p_k ln p_k`, `p_k = λ_k/Σλ` | **compressão ↔ expansão** (squeeze de vol) |
| Gap | `λ_p − λ₊` e sua velocidade | **modo nascendo** do bulk (antecede `m` subir) |
| Rotação | `θ = subspace_angles(Q_t, Q_{t−τ})` | regime **girando** de direção |

`λ₊ = σ²_bulk·(1+√q)²` (Marchenko-Pastur, ver `mp_lambda_plus`), com `σ²_bulk`
estimado de forma robusta pela mediana dos autovalores **abaixo** de λ₊; atualiza
junto. No segundo-bar o bulk literalmente *é* o ruído de microestrutura (bid-ask
bounce); os modos acima de λ₊ são o sinal real.

> ⚠️ **O espectro é cego à direção** (ver [direcao_e_espectro.md](direcao_e_espectro.md)):
> ele detecta *que* o regime mudou e *quão* concentrado/disperso está, mas **não**
> para que lado. Para a direção, combinar com o drift condicional (sinal da
> projeção dos retornos recentes em q₁, como em `compute_conditional_drift`).
> Regra: detector de regime = "algo grande vem aí"; drift = "é para cima".

---

## 6. Camada 3 — change-point com latência mínima (CUSUM / Page-Hinkley)

Cada sinal `y_t`, padronizado pela média/variância correntes, alimenta um
acumulador CUSUM — detector ótimo de *onset* com falso-alarme controlado:

```
g_t = max(0, g_{t−1} + (y_t − ȳ − ν))      # ν = folga (drift mínimo relevante)
alarme  quando  g_t > h
```

`h` controla diretamente o trade-off latência × falso-alarme (ARL). O CUSUM
dispara em ~O(σ / deslocamento) amostras após a mudança — o ótimo teórico.

---

## 7. Fusão das decisões (máquina de estados)

Dois limiares combinam a antecipação da camada 1 com a confiabilidade da camada 3:

```
NORMAL  ── ê_t alto (camada 1) ───────────────▶ WATCH       (arma, não opera)
WATCH   ── CUSUM espectral dispara (camada 3) ─▶ CONFIRMED   (sinal de trade)
WATCH   ── ê_t volta ao baseline por k ticks ──▶ NORMAL      (era ruído)
```

Operar só em CONFIRMED, mas pré-posicionar/cancelar ordens já em WATCH. É assim
que se espreme latência sem se afogar em ruído.

---

## 8. Estado mantido por ativo

```
buf      : ring buffer dos últimos `embed` retornos   (para montar sₙ)
C        : covariância EWMA  (m×m),  m = embed
Q, Λ     : top-p autopares   (Q: m×p, Λ: p)
σ²_bulk  : nível de ruído (mediana robusta dos λ abaixo de λ₊)
g[·]     : acumuladores CUSUM por sinal monitorado
base[·]  : média/variância correntes de cada sinal (padronização)
estado   : NORMAL | WATCH | CONFIRMED
```

Pipeline por tick: ingerir r_t → montar sₙ → **(1)** resíduo ê_t → atualizar C
(EWMA) → **(2)** subspace iteration → atualizar σ²_bulk e λ₊ → derivar sinais (§5)
→ **(3)** CUSUM → fundir estado (§7) → emitir regime + confiança. Re-ancoragem
periódica fora do caminho crítico.

---

## 9. Caveats honestos para HFT

1. **Limite de informação.** Não há almoço grátis: distinguir regime de ruído
   exige um mínimo de dados novos. O resíduo `ê_t` compra latência *pagando* em
   falso-alarme — escolhe-se um ponto na curva ROC, não se fura o limite de SNR.
2. **Microestrutura no segundo-bar.** Bid-ask bounce, ticks sem trade, horários
   ralos. Validar que os modos acima de λ₊ são sinal e não artefato; usar
   mid-price e tratar gaps. Filtrar `|ret|` extremos (splits/erros de fonte)
   antes da decomposição.
3. **Latência alarme → execução.** Um sinal "200 ms antes do movimento" só vale se
   a stack executa em < 200 ms. O backtest **tem** que modelar essa latência e os
   custos, senão o "early detection" é ilusório.
4. **Causalidade estrita.** Tudo é causal (EWMA, CUSUM, subspace só olham o
   passado); a re-ancoragem com `eigh` deve usar apenas dados até `t`. Sem
   look-ahead.
5. **Determinismo de latência.** O(m²p) é bom por ser *bounded*; evitar qualquer
   passo com nº de iterações variável no caminho crítico (p pequeno e fixo).

---

## 10. Implementação

Implementado em [stream_regime.py](../src_newest/stream_regime.py): classe
`StreamRegimeDetector` (EWMA + subspace tracker warm-started + Page-Hinkley),
reusando `mp_lambda_plus`; harness `replay()`/CLI sobre o Parquet do projeto;
`vol_breakouts()` e `score()` para calibração; `plot_states()` para inspeção.

```
python stream_regime.py --symbol NVDA --tf M1 --embed 70 --alpha 0.02 \
       --h 15 --nu 1.0 --limit 8000 --plot stream_nvda_m1.png
```

Custo medido: **~450 µs/tick** (embed=70, p=8) — folga de >2000× para barras de
1 s.

---

## 11. Calibração (NVDA M1) — resultados

Ground-truth = **breakouts de volatilidade realizada** (`vol_breakouts`: vol
forward / trailing > 2 em janelas de 60 barras). Métricas: recall, precisão e
**lead** (= t_gt − t_evento; positivo = antecipou).

### Achado 1 — sazonalidade intradiária domina
Em dados intradiários crus, "regime" ≈ **hora-do-dia**: o traço (vol²·d) faz um
dente-de-serra diário (pico na abertura, decaimento) e os eventos agrupam na
abertura. Tanto o detector quanto qualquer alvo baseado em vol medem, em primeira
ordem, o perfil determinístico. **De-sazonalizar os retornos** (dividir por um
perfil de vol por minuto-do-dia) antes de detectar é passo recomendado quando o
alvo é regime *estocástico*, não o ciclo intradiário.

### Achado 2 — a estrutura (`m`) ANTECIPA a vol; a vol (trace) é atrasada
Decompondo o detector por sinal contra breakouts de vol crus:

| sinal | recall | precisão | lead |
|---|---|---|---|
| trace (vol) | 0.07 | 0.04 | **−11** (atrasado) |
| vol-EWMA puro (sem espectro) | 0.07 | 0.12 | −4 (atrasado) |
| **m (contagem de modos)** | **0.31** | **0.20** | **+22 (antecipa)** |
| entropia | 0.17 | 0.19 | +1 |
| novelty (resíduo) | 0.14 | 0.21 | 0 |

A contagem de modos estruturais lidera o breakout em ~22 barras, enquanto a
volatilidade em si é coincidente/atrasada — só reage depois do burst começar.
Confirma, em versão streaming/causal, que features de regime preveem vol (HAR).

### Achado 3 — correção da máquina de estados
O design original assumia o resíduo de inovação como o precursor mais cedo
(camada 1 arma → espectral confirma). Os dados mostram o contrário: **`m` lidera
(+22), a novidade é coincidente (lead 0)**. Gatear `m` atrás do resíduo destruía a
antecipação. Correção: o sinal **estrutural confirma direto** NORMAL→CONFIRMED; o
resíduo vira apenas pré-arme opcional (WATCH). Por isso `confirm_signals`
default = `("m", "entropy")` — trace/λ₁ ficam de fora do gatilho (servem de
contexto de vol e, com o drift, de direção).

### Config recomendada
`alpha=0.02` (n_eff≈99), `h=15`, `nu=1.0`, `confirm_signals=("m","entropy")`:
recall **0.33**, precisão **0.19**, **lead +24 barras**. Subir `h`/`nu` corta
eventos mas zera o recall junto (Pareto desfavorável). Os valores absolutos de
recall/precisão dependem da densidade do ground-truth; o resultado robusto é o
**lead positivo grande do sinal estrutural** vs. o lead negativo da vol.

---

## 12. Direção do trade (drift condicional)

O espectro é **cego à direção** (par sob `r→−r`); ela vem de objetos **ímpares**,
condicionados pelo regime — ver [direcao_e_espectro.md](direcao_e_espectro.md) §4b.
`_direction()` implementa `E[r_{t+h}] ≈ f(regime)·sinal_linear`, **gauge-safe**:

- **sinal linear (ímpar):** `drift` = retorno líquido das últimas `embed` barras
  (momentum bruto) e a inclinação do retorno filtrado no subespaço estrutural via o
  **projetor** `P = VₛVₛᵀ` (invariante ao sinal dos autovetores — nunca usa
  `sign(vₖ)` cru);
- **modo momentum/reversão:** `direction_mode='auto'` usa a autocorr lag-1 (`persist`)
  para decidir; ou força `'momentum'`/`'reversion'`;
- **confiança (peso ∈ [0,1]):** `(1−entropia)` se `m≥1`, senão 0 — entropia baixa +
  modos estruturais ⇒ sinal confiável.

### Validação (NVDA M1, hit-rate vs. retorno forward)

**Incondicional não há edge (~0.49–0.51), como o framework prevê** (1º momento ≈
martingale). Reversão fica levemente acima de 0.5 (mean-reversion de microestrutura
no minuto), momentum abaixo. **O edge é regime-dependente** e some no agregado.

Estratificando os eventos CONFIRMED (modo reversão, h=60) por **confiança**:

| confiança `(1−S, m>0)` | n | hit-rate | ret. assinado médio |
|---|---|---|---|
| baixa | 182 | 0.473 | −8.6e-04 |
| média | 182 | 0.500 | +1.3e-03 |
| **alta** | 187 | **0.567** | **+2.9e-03** |

Padrão monotônico: **com regime espectral bem-definido (entropia baixa, m>0), a
reversão acerta ~57%**, contra ~47% em baixa confiança. Confirma a tese: o espectro
não dá direção, mas **gateia quando o sinal direcional é confiável**. Leverage
(viés de baixa em breakout) e persistência aparecem fracos; **a confiança é o knob
que funciona** → operar só os eventos de alta confiança e dimensionar posição ∝
`|direction|` (que já embute `1/vol` implícito via entropia + escala por trace).

> Marcadores de direção no plot: ▲ verde (long) / ▼ vermelho (short), tamanho ∝
> confiança, no painel de preço.

---

## 13. Backtest (custos + latência) — in-sample vs out-of-sample

`backtest()`: causal, **uma posição por vez** (flat entre trades, sem look-ahead).
Cada evento CONFIRMED com `|direction| ≥ conf_min` entra `latency` barras após o
sinal (atraso alarme→execução), segura `horizon` barras, sai; PnL em log-retorno
menos `cost_bps` round-trip. CLI: `--backtest --conf-min --cost-bps --latency`.

### Achados in-sample (NVDA M1 2016, 100k ticks, modo reversão)
- **filtro de confiança é decisivo:** `conf_min=0` (todo sinal) **perde** (Sharpe<0,
  PF<1 — custo come o ~0.5); `conf_min=0.2` ganha (hit 0.60, PF 1.97, maxDD baixo).
- **robusto a custo:** 0→5 bps quase não degrada (PF 2.00→1.83). O hold de 60 min dá
  ganho bruto ~37 bps ≫ custo → **swing intradiário, não scalping** (custo só morde
  em holds curtos). A latência 1→5 barras degrada mas mantém o edge positivo.
- Sharpe/trade 0.215 ⇒ **Sharpe anualizado ≈ 2.1** (95 trades em ~1 ano).

### ⚠️ Out-of-sample: o edge NÃO se replica
Mesma config (sem re-tunar), `conf_min=0.2`, custo 1 bp, hold 60:

| segmento | trades | hit | Sharpe_ann | PF |
|---|---|---|---|---|
| NVDA 2016 (IS) | 95 | 0.600 | **2.10** | 1.97 |
| NVDA OOS #1 | 26 | 0.423 | **−0.48** | 0.73 |
| NVDA OOS #2 | 11 | 0.545 | 1.28 | 2.81 |
| AAPL OOS | 27 | 0.667 | 0.67 | 1.58 |
| MSFT OOS | 52 | 0.519 | 1.13 | 1.65 |

**Veredito honesto:** o Sharpe ~2.1 in-sample é otimista/overfit. OOS é fraco e
inconsistente — 3 de 4 segmentos positivos (Sh_ann 0.67–1.28), mas um perde, e as
contagens de trades são baixas (11–52) → estatística ruidosa. Há *indício* de edge
condicional ao regime, **muito abaixo** do número in-sample e ainda **não
negociável**. O backtest cumpriu seu papel: evitou acreditar num edge falso.

**Veredito:** o Sharpe ~2.1 in-sample é overfit. OOS em ativo único é fraco/ruidoso
(11–52 trades). Precisa de poder estatístico → §14.

---

## 14. Walk-forward + portfólio (poder estatístico)

`portfolio_backtest()` roda o detector em vários ativos num slice e **agrupa os
trades** (pool) para um t-stat; `walkforward()` re-calibra `conf_min` por fold de
calibração e aplica no fold seguinte (OOS), agrupando os trades OOS. Aceleração:
`track_rotation=False` (o ângulo de subespaço não entra no backtest) → ~2× (282
µs/tick). Guarda de splits via `clip=0.5`.

### Resultados (15 ativos líquidos do S&P500)
**Portfólio** (conf=0.2 fixo, OOS [100k,180k]): pooled **771 trades**, hit 0.532,
Sharpe/trade 0.087, **t-stat 2.42**, PF 1.50. Por ativo muito heterogêneo (META hit
0.76; TSLA/JPM/PG perdem).

**Walk-forward** (3 folds OOS, conf re-calibrado): pooled **2244 trades**, hit
0.528, Sharpe/trade 0.049, **t-stat 2.32**, PF 1.19. Por fold inconsistente (2 de 3
não-significantes; só o fold com conf=0.3 deu t-stat 2.5).

### Veredito
- **Edge detectável:** agrupando milhares de trades, t-stat ~2.3–2.4. A
  diversificação resgata o sinal do cara-ou-coroa do ativo único → coerente com "o
  edge se cancela no agregado a menos que **condicionado**" (confiança +
  diversificação).
- **Fino e (por fold) instável:** Sharpe/trade 0.05–0.09 (o 0.21 in-sample era
  overfit), PF 1.2–1.5, folds inconsistentes.
- **⚠️ `net_total` não é NAV:** soma de log-retornos de trades sobrepostos no tempo
  em ativos diferentes (alavancagem implícita), não curva de capital.

---

## 15. Direção SSA-no-nível + t-stat honesto (block bootstrap)

### A/B drift vs SSA (portfólio OOS [100k,180k], reversão, conf=0.2)
`direction_source='ssa'` = inclinação da reconstrução SSA do nível log-preço
(`_ssa_level`), gauge-free. Mesmos 771 trades (a confiança/filtro são iguais; só o
**sinal** muda):

| fonte | hit | net_tot | Sharpe/tr | t-stat | PF |
|---|---|---|---|---|---|
| drift | 0.532 | 0.796 | 0.087 | 2.42 | 1.50 |
| **ssa** | 0.527 | **1.114** | **0.123** | **3.40** | **1.78** |

**SSA tem hit-rate menor mas Sharpe/PF/t maiores** → não acerta *mais vezes*, acerta
nos *movimentos maiores*. A tendência denoised pega melhor a direção quando o
movimento é grande/persistente — valor de magnitude, não de contagem. (In-sample
deu empate; o ganho aparece no portfólio OOS.)

### Block bootstrap (t honesto vs. correlação cross-asset)
`block_bootstrap()`: cluster bootstrap por tempo — agrupa trades em blocos
(≥ horizonte), reamostra blocos inteiros, mede o SE robusto vs. o ingênuo.

| fonte | bloco | t_naive | t_honesto | se_ratio | p | signif95 |
|---|---|---|---|---|---|---|
| drift | 1d / 3d | 2.42 | 2.51 / 2.38 | 0.97 / 1.02 | 0.004 | SIM |
| ssa | 1d / 3d | 3.40 | 3.53 / 3.47 | 0.96 / 0.98 | <0.001 | SIM |

**`se_ratio ≈ 1` ⇒ a correlação cross-asset NÃO inflou o t** (a preocupação prévia
estava empiricamente errada). Razão: o livro long/short de reversão é
quase **market-neutral** — a cada instante há posições para os dois lados, o beta de
mercado se cancela e o PnL é majoritariamente **idiossincrático**, então os trades
pooled comportam-se quase como independentes. Robusto a bloco 1d vs 3d.

**Conclusão (parcial — ver §16):** com config **fixa**, o edge sobrevive ao
bootstrap (drift p=0.004, ssa p<0.001) e o SSA-no-nível é superior ao drift (t 3.5 vs
2.4). MAS esta config foi escolhida com conhecimento in-sample (viés de seleção). O
§16 refaz tudo com seleção de config **OOS** e o edge **não sobrevive** — o bootstrap
corrige correlação cross-asset, não viés de seleção. Ler §15 e §16 juntos: SSA é um
estimador de direção melhor *condicionalmente*, mas não há edge negociável.

### Gate por trendiness — testado, NÃO ajuda (resultado negativo)
`conf_gate='trend'`: confiança = `(1−S)·trendiness` (trendiness = λ₁/trace do nível
SSA). A/B no portfólio OOS, **a contagem de trades casada** (trend cm=0.1 ≈ entropy
cm=0.2, ambos 771 trades):

| gate | n | hit | Sh/tr | PF | t_honesto |
|---|---|---|---|---|---|
| `entropy` (1−S) | 771 | 0.527 | **0.123** | **1.78** | **3.53** |
| `trend` (1−S)·trend | 771 | 0.512 | 0.093 | 1.54 | 2.59 |

O gate **piora** (t 3.53→2.59). Estratificando os trades por tercil de trendiness, o
hit de reversão é **plano** (0.508 / 0.538 / 0.527) — trendiness **não discrimina**
trades vencedores (a hipótese "reversão prefere choppy" foi refutada; o hit nem cai
nem inverte). Provável causa: `λ₁/trace` do nível reflete a geometria do embedding (a
cumsum é intrinsecamente low-rank) mais que força de tendência real. **Decisão:
manter `conf_gate='entropy'`** — `(1−S)` carrega informação, trendiness não, e
multiplicá-las dilui o bom seletor. (O parâmetro fica disponível, default `entropy`.)

---

## 16. Walk-forward com seleção de config 100% OOS — VEREDITO: edge NÃO sobrevive

O teste mais rigoroso: `walkforward(direction_source='ssa')` escolhe a config
(`direction_mode` ∈ {reversion, momentum} × `conf_min`) que maximiza o t-stat **no
fold de calibração** e aplica no fold de **teste** seguinte — toda decisão de config
é OOS. 12 ativos, 4 folds de 40k ticks.

| fold | modo* | conf* | n | hit | Sh/tr | t-stat |
|---|---|---|---|---|---|---|
| 0 | reversion | 0.3 | 121 | 0.496 | 0.034 | 0.38 |
| 1 | reversion | 0.1 | 852 | 0.522 | −0.018 | −0.54 |
| 2 | reversion | 0.3 | 107 | 0.551 | 0.268 | 2.77 |
| **pooled** | — | — | 1080 | 0.522 | 0.043 | 1.43 |

Block bootstrap no pool OOS: **t_honest 1.33–1.34, IC95 inclui 0, p≈0.08 → NÃO
significante a 95%** (robusto a bloco 1d/3d).

**Veredito final da linha:** com a seleção de config feita 100% OOS, **o edge não
sobrevive**. Os t=3.40 (SSA) e t=2.42 (drift) do §14–15 tinham **viés de seleção
embutido** — "reversão, conf=0.2" foi fixado com conhecimento in-sample. Distinções:
- **o que generaliza:** o walk-forward escolheu **reversão nos 3 folds** sozinho —
  essa escolha não era overfit;
- **o que não transfere:** o `conf_min` ótimo pula (0.3/0.1/0.3) e o OOS é
  inconsistente (0.38 / −0.54 / 2.77); o pooled positivo é carregado por **um único
  fold**. Assinatura de edge que se concentra num período e não replica.
- **ponto metodológico:** o block bootstrap corrige **correlação cross-asset**, não
  **viés de seleção** — eram dois vieses distintos, e o de seleção era o maior. Por
  isso §15 (config fixa) deu p<0.001 e §16 (config OOS) dá p=0.08. Só o walk-forward
  expõe isso.

> **Lição registrada:** não recaçar o t=3.40 — ele era seleção. O valor desta linha
> é a **metodologia** (detector causal + backtest honesto + walk-forward + bootstrap
> que pegaram o overfit), não uma estratégia negociável. O SSA-no-nível continua um
> estimador de direção melhor *condicionalmente* (§15), mas isso não basta para edge.

### Próximos passos (se retomar)
- mais dados / outros regimes de mercado (a amostra é ~1,5 ano de M1; 1 fold
  significante em 3 é fraco) e timeframes (H1/D1) antes de concluir ausência de edge;
- features ortogonais ao preço (fluxo de ordens, microestrutura) — o 1º momento é
  ≈ martingala, o teto direcional do espectro puro é baixo por construção;
- de-sazonalização causal embutida no stream (perfil do dia anterior);
- testar B3 (PETR4) quando houver Parquet M1/M5.
