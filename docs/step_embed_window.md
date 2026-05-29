# Parâmetros da decomposição: `embed`, `window` e `step`

Os três parâmetros que governam a decomposição espectral em [src_newest/decomp_pca.py](../src_newest/decomp_pca.py)
respondem a perguntas diferentes sobre *como olhar* para a série de log-retornos
`ret = [r₀, r₁, r₂, …]`.

O objetivo do pipeline é **acompanhar como a estrutura espectral local dos
retornos evolui no tempo**. Para isso, em cada instante é preciso (a) transformar
a série escalar num objeto multidimensional, (b) estimar uma covariância local,
(c) repetir deslizando no tempo. Cada parâmetro controla uma dessas etapas.

> Resumo de uma linha: **`embed` define o tamanho de cada espectro, `window`
> define quão bem cada espectro é estimado, e `step` define com que frequência
> ele é reestimado ao longo do tempo.**

---

## 1. `embed` — dimensão do mergulho (time-delay embedding)

Uma série escalar de retornos não tem "estrutura espectral" sozinha. Para extrair
modos, ela é **mergulhada** num espaço de dimensão `embed` por atrasos temporais
(delay embedding, à la Takens / SSA). Cada ponto no espaço passa a ser um pedaço
de `embed` retornos consecutivos:

```
snapshot = [rₜ, rₜ₊₁, rₜ₊₂, …, rₜ₊embed₋₁]     ← vetor de dimensão embed
```

`embed` é o **comprimento do padrão** tratado como unidade. Ele define:

- a dimensão do espaço de fase;
- o tamanho da matriz de covariância `C` → `(embed × embed)`;
- quantos modos espectrais podem ser resolvidos (no máximo `embed`).

`embed` grande → captura correlações temporais mais longas, mas `C` fica maior e
mais cara e exige mais amostras para ser bem estimada.

## 2. `window` — trecho local que vira uma covariância

Dentro de um trecho de `window` barras, o snapshot de comprimento `embed` é
deslizado de 1 em 1, gerando vários snapshots **sobrepostos** — as *linhas* da
matriz X:

```
n_samp  = window − embed       ← nº de snapshots (amostras) por covariância
X.shape = (n_samp, embed)
```

X é uma **matriz de Hankel** (cada linha desloca a anterior em 1; valores
constantes ao longo das anti-diagonais). A covariância
`C = Xᵀ·X / (n_samp − 1)` resume a estrutura desses `n_samp` snapshots.

`window` é o **tamanho da amostra estatística local**: quantos snapshots
alimentam *uma* estimativa espectral. Trade-off:

- `window` grande → mais amostras → `C` menos ruidosa, mas mistura mais tempo
  (pior resolução temporal, borra regimes);
- `window` pequeno → mais responsivo no tempo, mas `C` ruidosa.

O padrão `window = 5·embed` fixa a razão de aspecto `q = embed / n_samp = 1/4`
constante, que entra no limiar de Marchenko-Pastur `λ₊ = σ²·(1 + √q)²`.

## 3. `step` — passada entre covariâncias consecutivas

Estimada uma `C` num trecho, o bloco inteiro avança `step` barras e a próxima é
estimada. `step` é a **passada temporal** entre estimativas:

```
centers = window, window + step, window + 2·step, …
```

`step` **não muda o tamanho de nada** — só a densidade no tempo:

- `step` pequeno → janelas muito sobrepostas → evolução densa/suave, porém muito
  mais janelas (custo);
- `step` grande → janelas espaçadas → série espectral mais esparsa, menos
  redundante.

Detalhe importante: `step` é contado **em barras, não em tempo de calendário**.
`step = 20` significa uma janela a cada 20 horas no H1, mas a cada 20 minutos no
M1 — por isso o mesmo `step` deixa ~60× mais janelas no M1 (ver
[a relação entre timeframes](#por-que-m1-tem-muito-mais-janelas-que-h1)).

---

## Exemplo numérico

Com `embed = 3`, `window = 6` (logo `n_samp = 3`), `step = 4`, sobre
`ret = [r₀, r₁, r₂, r₃, r₄, r₅, …]`:

```
Bloco 1 (começa em r₀):          Bloco 2 (começa em r₄ = +step):
  X₁ = [ r₀ r₁ r₂ ]                X₂ = [ r₄ r₅ r₆ ]
       [ r₁ r₂ r₃ ]   (Hankel)         [ r₅ r₆ r₇ ]
       [ r₂ r₃ r₄ ]                    [ r₆ r₇ r₈ ]
  → C₁ = X₁ᵀX₁/2  (3×3)            → C₂ = X₂ᵀX₂/2  (3×3)
  → eigh(C₁) → espectro em t₁      → eigh(C₂) → espectro em t₂
```

- `embed = 3` → 3 colunas, `C` é 3×3, até 3 modos;
- `window = 6` → `n_samp = 3` linhas por bloco (a amostra local);
- `step = 4` → o bloco 2 começa 4 barras adiante.

---

## Tabela-resumo

| Parâmetro | Pergunta que responde | Afeta | Depende de `step`? |
|---|---|---|---|
| `embed` | comprimento de um padrão / dimensão do espaço | tamanho de C, nº de modos, alcance temporal | não |
| `window` | quantos snapshots locais estimam uma covariância | qualidade estatística vs. resolução temporal | não |
| `step` | de quanto em quanto tempo reestimar | densidade da série espectral, nº de janelas | — |

Fórmulas (padrão `window = 5·embed`):

```
n_samp     = window − embed            = 4·embed
X.shape    = (n_samp, embed)           = (4·embed, embed)
C.shape    = (embed, embed)
q          = embed / n_samp            = 0.25
λ₊         = σ²·(1 + √q)²              (limiar Marchenko-Pastur)
n_centers ≈ (len(ret) − 2·window) / step
```

Se `--window` for passado explicitamente, `n_samp = window − embed` e
`q = embed / (window − embed)` mudam; `step` continua sem afetar tamanho algum.

---

## Por que M1 tem muito mais janelas que H1

Para `step` e `embed` fixos, **o custo por janela é idêntico** entre timeframes
(mesma forma de X e de C). O que muda é só a *quantidade* de janelas, que é
proporcional ao número de barras:

```
T ≈ n_centers × custo_por_janela        com  n_centers ∝ nº de barras
```

Exemplo (NVDA, `step=20`, `embed=70`):

| TF | Barras | Janelas |
|---|---|---|
| M1 | 1.769.013 | 88.416 |
| H1 | 39.504 | 1.941 |
| D1 | 2.606 | 96 |

M1 gera ~45× mais janelas que H1 → leva ~45× mais tempo. Para a mesma densidade
temporal de janelas que o H1, o M1 precisaria de `step` ~60× maior.

---

## Visualização

Uma janela real do NVDA, com a Hankel X, a covariância C e o espectro com o
limiar MP separando modos estruturais (acima de `λ₊`) do bulk de ruído, pode ser
gerada para inspeção — ver figura de exemplo em `src_newest/hankel_demo.png`
(artefato, fora do versionamento).

> **Cuidado com splits não ajustados:** retornos extremos (`|ret| > 0.5`,
> tipicamente splits de ações na fonte Alpaca) deslizam pela Hankel e injetam
> modos estruturais espúrios, inflando `m`. Tratar antes da decomposição.
