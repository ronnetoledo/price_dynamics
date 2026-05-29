# Direção do mercado e o espectro: o que dá (e o que não dá) para extrair

Documento de referência sobre **se e como** a decomposição espectral PCA pode
informar a *direção* do movimento de preços. A conclusão central tem uma parte
matemática rigorosa e uma parte empírica. Relacionado a
[step_embed_window.md](step_embed_window.md).

---

## 1. Resultado central: autovalores são cegos à direção (por construção)

A covariância é uma forma **quadrática** nos dados; direção é informação **ímpar**
(de sinal). Considere espelhar todos os log-retornos, `r → −r` (uma alta vira a
queda de mesma forma):

```
X  → −X            (a matriz de Hankel é linear em r)
Xc → −Xc           (a centragem preserva a linearidade)
C  = Xcᵀ·Xc/(n−1)  →  (−Xc)ᵀ(−Xc)/(n−1) = C        ← INVARIANTE
```

Portanto **autovalores E autovetores são idênticos** para uma trajetória e seu
espelho de sinal. Nenhuma função de `{λₖ, vₖ}` distingue alta de queda. A direção
está, literalmente, no **núcleo do mapa** que produz os autovalores.

Além disso, a **média `⟨r⟩` (o drift) é removida** antes de formar C — e o drift é
o sinal direcional mais direto. A direção é descartada duas vezes: ao centrar e ao
elevar ao quadrado.

> **Corolário prático:** não busque direção em `λₖ`, `m`, entropia espectral ou
> em qualquer escalar derivado só dos autovalores. Eles medem *energia/geometria
> de flutuação*, não sinal.

---

## 2. Onde a direção realmente mora

Direção só vem de objetos **ímpares sob `r → −r`** (lineares nos dados):

1. **Drift `⟨r⟩`** da janela — o termo que é subtraído. Sinal direcional bruto.
2. **Projeções / scores `aₖ = Xc·vₖ`** — lineares em Xc, trocam de sinal com os
   retornos. É aqui que o espectro vira direção: os autovetores dão o *eixo*
   (gauge ±), os scores dão a *amplitude com sinal*.
3. **Reconstrução SSA do modo líder, `X̂₁ = a₁·v₁ᵀ`** — o autovetor dominante de
   uma covariância de atrasos costuma capturar a tendência lenta; sua reconstrução
   é uma série com sinal → indica subida/descida do componente de tendência.

Resumo: **os autovetores dão o eixo do movimento; o sinal da projeção sobre esse
eixo é o que aponta a direção.**

---

## 3. Assimetria de previsibilidade (a física do problema)

| Momento | Quantidade | Previsibilidade | O espectro vê? |
|---|---|---|---|
| 1º (sinal do retorno) | direção | ≈ martingale, quase imprevisível | **não** |
| 2º (volatilidade, flutuação) | regime/energia | forte (clustering, memória longa) | **sim** |

O método é poderoso justamente na parte previsível (regime/volatilidade) e cego
justamente na parte difícil (sinal). Isso é a estrutura do problema, não uma
falha do método.

**Único canal real (e fraco) ligando autovalores a direção — *leverage effect*:**
em ações, surtos de volatilidade acompanham **quedas** mais que altas. Um pico nos
autovalores estruturais carrega então um *viés probabilístico* contemporâneo (ou
levemente defasado) para movimento negativo. É "indicação" estatística, não
previsão de sinal.

---

## 4. Como usar no framework: espectro como classificador de regime

O uso defensável é **espectro condiciona um sinal direcional**, não "autovalor →
direção":

1. **Regime via espectro** — `m(t)`, entropia espectral e principalmente o **β do
   MSD** já calculado no pipeline:
   - β > 1 (superdifusão) → regime persistente/tendencioso → *seguir* tendência;
   - β < 1 (subdifusão)   → mean-reverting → *reverter*.
   Isso é informação direcional **condicional** (diz qual estratégia vale, não o
   sinal em si).
2. **Direção via reconstrução** — dentro de regime tendencioso, usar `sign(a₁)` da
   reconstrução do modo líder (ou o `⟨r⟩`) como direção candidata.
3. **Pontos de virada via rotação de subespaço** — `scipy.linalg.subspace_angles`
   entre janelas consecutivas detecta *quando* a estrutura roda; útil para
   antecipar mudança de regime, que costuma coincidir com inflexões.

Esquema de decisão:

```
espectro (m, entropia, β)        →  QUE regime? (tendência vs ruído vs reversão)
ângulos de subespaço             →  QUANDO o regime está mudando?
sign(a₁) / ⟨r⟩  (linear em r)    →  QUAL direção, condicionada ao regime
```

---

## 4b. Os objetos lineares condicionados ao regime

A própria PCA já **fatoriza** a parte direcional e a parte de regime. Cada janela,
após a eigh, escreve-se como

```
Xc = Σₖ aₖ · vₖᵀ          (decomposição espectral / SSA)
```

e os dois fatores têm paridades opostas sob `r → −r`:

| Objeto | O que é | Paridade sob r→−r | Carrega |
|---|---|---|---|
| `{vₖ, λₖ}` | autovetores (formas) + autovalores | **par** (invariante) | regime / geometria |
| `aₖ = Xc·vₖ` | scores (amplitudes projetadas) | **ímpar** (troca de sinal) | direção |

**`vₖ` e `λₖ` dizem qual subespaço e quanta energia (regime); os scores `aₖ`
dizem onde ao longo dele (sinal).** Direção = sinal dos scores, sobre o subespaço
que o regime selecionou.

### Objetos lineares (ímpares = direcionais)

Funcionais lineares dos retornos — trocam de sinal com `r`:

1. **Drift** `μ = ⟨r⟩` — momentum bruto.
2. **Scores** `aₖ = Xc·vₖ` — amplitude do k-ésimo modo temporal.
3. **Projeção no subespaço estrutural** `r̂ = Vₛ·Vₛᵀ·r`, `Vₛ = [v₁…v_m]` — retorno
   filtrado, mantendo só os `m` modos que o limiar MP julgou sinal.
4. **Inclinação da reconstrução** `slope(cumsum(r̂))` — taxa direcional do
   componente estrutural.
5. **Filtro linear genérico** `s = wᵀ·r`, com `w` montado da estrutura espectral.

### Objetos de regime (pares = cegos à direção)

Todos já no Parquet: `m`, entropia espectral, `trace`/`λ₁` (energia ≈ vol),
expoente de persistência (Hurst/β), ângulos de subespaço entre janelas.

### Como o regime condiciona — modelo `E[r_{t+h}] ≈ f(regime_t) · sinal_linear_t`

O regime modula o ganho e o sinal do objeto linear, nunca fornece direção sozinho:

1. **Inverte tendência ↔ reversão (principal):** a persistência decide seguir ou
   inverter o sinal linear —
   `H > 1/2` (persistente) → `+sign(μ)` (momentum);
   `H < 1/2` (anti-persist.) → `−sign(μ)` (reversão).
2. **Confiança (peso):** entropia baixa / `m > 0` → sinal confiável; entropia alta
   / `m = 0` → sem subespaço direcional, não apostar.
3. **Escala de risco:** posição ∝ 1/vol via `trace`/`λ₁` (cego ao sinal, define
   magnitude).
4. **Gate de transição:** ângulo de subespaço grande → estrutura rodando → sinal
   antigo obsoleto, reduzir exposição.

> **Ressalva:** o β do pipeline é calculado sobre o processo espectral
> `Xₜ = cumsum(Kₜ − ⟨K⟩)` — descreve a persistência da *dinâmica de regime*, não
> diretamente do preço. Para condicionar direção de preço, o knob direto é a
> persistência do *caminho de retornos* (Hurst dos retornos), que também pode ser
> regime-dependente. São complementares, não o mesmo objeto.

> **Por que isso importa após o resultado nulo da seção 5b:** o hit-rate de 0.49
> é *incondicional* (média sobre todos os regimes). Se o edge for regime-dependente
> (momentum em `H>½`, reversão em `H<½`), a média **se cancela** e some no agregado.
> Condicionalizar = estratificar o sinal por regime e medir dentro de cada um.

## 4c. Reconstrução de posto baixo (SSA): o que os modos capturam

Reconstruir o sinal com os `K` maiores autovalores = **reconstrução SSA de posto
K**: projeta-se a Hankel centrada no subespaço dos `K` autovetores dominantes e
faz-se a **média diagonal** (hankelização) de volta a uma série 1D:

```
Xc_K = Xc · V_K · V_Kᵀ          (V_K = [v₁…v_K])
sinal_K = hankelize(Xc_K)        (média sobre anti-diagonais)
```

Demonstração em NVDA D1 (janela @ 2019-06-20, embed=70, m=6; figura
`src_newest/recon3.png`), com K=3:

- **Os top-3 capturam só ~22% da variância** (λ/trace = 0.086, 0.085, 0.051). O
  espectro é quase plano → retornos diários são **dominados por ruído**, com baixa
  compressibilidade de posto baixo. Coerente com a cegueira direcional e com o
  caráter de 2º momento do sinal.
- **`λ₁ ≈ λ₂` (par quase degenerado) → modo oscilatório.** É a assinatura SSA
  clássica de um *ciclo*: dois autovalores próximos correspondem a um par
  seno/cosseno. Visível nos componentes elementares (modos 1 e 2 em quase
  quadratura) e nos autovetores `v₁,v₂` (EOFs oscilatórios de frequência similar).
- **A direção aparece pela integração.** A decomposição é de **retornos**
  centrados, então o sinal reconstruído é um sinal de *retornos*. A
  tendência/direção é propriedade do **caminho integrado** (log-preço = cumsum dos
  retornos): para vê-la, **integra-se o sinal reconstruído**. Como a centragem
  remove o drift por janela, a flutuação de posto-K integra-se a **excursões
  cíclicas** (o par degenerado vira uma oscilação no preço), enquanto o **trend
  linear sustentado vem da integração do drift** (a média removida). Integrar é a
  operação linear que leva retorno → preço — mesma fatoração par/ímpar da 4b.

> **Uso prático:** para extrair tendência via posto baixo, reconstrua o sinal de
> retorno (retendo o drift, se quiser o trend sustentado) e **integre**; ou faça
> SSA **não-centrado** direto sobre o log-preço, onde o modo 1 já é o trend. Sobre
> retornos centrados, os modos dão a *forma* dos ciclos; a integração revela como
> eles se acumulam no preço.

## 4d. Energia no subespaço estrutural — quanto é sinal

Os autovalores *são* energias (variâncias); `trace = Σλ` é a energia total da
janela. A **fração de energia estrutural** mede quanto disso vive acima do limiar
MP:

```
f_struct = (Σ_{k=1}^{m} λₖ) / trace                  (fração no subespaço estrutural)
f_excess = f_struct − m/d                            (corrigida pelo ruído: σ²=trace/d)
```

`f_excess` desconta a energia que `m` modos de puro ruído já teriam (`m·σ²`),
isolando o excesso genuíno de sinal. Ambas saem direto do `eigenvalues.parquet`.

Medido em MSFT M1 (75.706 janelas, d=70; figura `src_newest/energy_struct.png`):

| quantidade | valor | leitura |
|---|---|---|
| janelas com m=0 (só ruído) | 14% | parte do tempo não há estrutura |
| `m` | média 3.2, mediana 3, máx 21 | poucos modos estruturais |
| `f_struct` | média **0.13** (mediana 0.11) | ~13% da energia é estrutural |
| `f_excess` | média **0.08** | ~8% é sinal além do ruído |
| `f_top1` | 0.041 | modo líder ≈ 3× o piso de ruído (1/d=0.014) |
| energia por modo estrutural | 0.039 do trace | ≈ 2.8× a energia de um modo de ruído |

**Interpretação:**

- **~87% da energia está no bulk (ruído).** A estrutura é uma casca fina — coerente
  com a cegueira direcional e com os ~22% da reconstrução posto-3.
- **Cada modo estrutural carrega ~3× a energia de um modo de ruído** (`f_struct`
  cresce com `m` bem *acima* do piso `m/d` — o gap é o sinal genuíno). Confirma que
  os modos acima de λ₊ não são acidente de amostragem.
- **O MP é conservador.** No perfil espectral médio, a energia só cai abaixo do
  nível uniforme `1/d` por volta do modo ~28, mas o limiar MP marca em média só ~3
  modos. MP exige `λ > λ₊` (não apenas `λ > σ²`), filtrando agressivamente o ruído.

`f_struct` é, ele próprio, um bom **objeto de regime** (par, cego à direção): serve
de medida de confiança no mecanismo 2 da seção 4b — alto `f_struct` → janela
estruturada, sinal mais confiável; baixo → quase ruído, não apostar.

### Série temporal f_struct(t) e a armadilha da sobreposição

(MSFT M1; figura `src_newest/fstruct_ts.png`)

- `f_struct(t)` oscila em torno da média (0.127) com variação lenta (regimes) e um
  padrão intradiário tipo **dente-de-serra** (a estrutura cresce ao longo da sessão
  e reseta) — sazonalidade intradiária.
- `m(t)` co-move com `f_struct(t)`.
- **Armadilha da sobreposição:** janelas consecutivas compartilham
  `(window−step)/window = 330/350 = 94%` dos dados. O `ACF(1) = 0.80` é quase todo
  **mecânico**, não memória de regime. As janelas só deixam de se sobrepor em
  `lag = window/step ≈ 17`. Além desse lag o ACF cai para ~0.07–0.09 num platô
  lento → a memória de regime **genuína é fraca, porém long-memory** (não decai a
  zero).

> **Relação com o MSD/β (núcleo do artigo):** o mesmo efeito de sobreposição infla
> a correlação de *curto* lag de qualquer estatística de janela deslizante —
> inclusive o processo espectral `Kₜ`. **Porém o expoente β está protegido:** o
> ajuste em `msd_beta_sliding_window_2.py` usa a cauda (`tail_frac=0.7`, lags ~60–199
> janelas), bem acima do limite de sobreposição (~17 = window/step). Nessa região a
> sobreposição apenas renormaliza o *prefator* de difusão, não o *expoente*. O viés
> de curto lag afeta estatísticas como o ACF de `f_struct`, não o β da cauda —
> **desde que** o fit permaneça acima de `window/step` (cuidado com séries curtas,
> onde `max_lag = T−1` pode empurrar o fit para dentro da zona sobreposta).

## 5. Experimento mínimo (com o null correto)

Antes de qualquer backtest, **medir o que é previsível**:

1. `corr(m(t) ou entropia, volatilidade realizada futura)` → esperado **forte**
   (confirma o canal de 2º momento).
2. `corr(m(t) ou entropia, sinal do retorno futuro)` → esperado **≈ 0**
   (confirma a cegueira direcional).
3. *Leverage*: `corr(Δλ₁, retorno)` contemporâneo e defasado → esperado **negativo
   e fraco**.
4. Sinal candidato: `sign(a₁)` da reconstrução do modo líder vs. retorno forward,
   **comparado contra o null de embaralhar os sinais dos retornos** — para não
   confundir drift estrutural (prêmio de risco, viés de alta perene em ações) com
   previsibilidade real.

> **Armadilha do null:** ações têm drift positivo (equity risk premium). Qualquer
> sinal "sempre comprado" parece lucrar. O teste honesto compara contra
> embaralhamento de sinais / contra o buy-and-hold, não contra zero.

---

## 5b. Confirmação empírica (MSFT M1, 75.705 janelas)

O experimento da seção 5 foi rodado sobre MSFT M1 (step=20, embed=70), excluindo
janelas com split. Correlações de Spearman; horizontes em barras (minutos):

**(1) Espectro → volatilidade futura — FORTE, como esperado:**

| feature | h=20 | h=60 | h=240 |
|---|---|---|---|
| trace (energia total) | +0.39 | +0.31 | +0.28 |
| λ₁ (modo líder) | +0.37 | +0.29 | +0.25 |
| entropia | +0.12 | +0.15 | +0.17 |
| m | −0.10 | −0.12 | −0.13 |

A energia espectral total (trace ≈ variância realizada da janela) é o melhor
preditor de vol futura — clustering de volatilidade. Nota: `m` correlaciona
*negativo* e entropia *positivo* → regimes de alta vol futura são mais
*difusos/ruidosos* (espectro espalhado, poucos modos claramente estruturais),
porém de maior energia.

**(2) Espectro → retorno/sinal futuro — NULO, como esperado:**
todas as correlações com retorno futuro e com `sign(retorno)` ficaram em
|r| ≤ 0.01 (ex.: `m~sign` = +0.004). Cegueira direcional confirmada em escala de
75 mil janelas.

**(3) Leverage:** `corr(Δλ₁, retorno) = −0.005` (≈ 0) — o leverage effect
**não aparece em escala intradiária M1** (é fenômeno de baixa frequência/diário).
Sanidade: `corr(Δλ₁, |retorno|) = +0.16` (positivo), confirmando que os
autovalores *rastreiam* a magnitude do movimento, mas são cegos ao sinal.

**(4) Sinal momentum `sign(drift)`:** hit-rate ≈ 0.49, **abaixo** do null de
embaralhamento (0.498) e bem abaixo do buy&hold (0.53), com z = −5.4 em h=240.
Ou seja, o sinal de momentum da janela **não tem edge direcional** (intradiário é,
se algo, levemente anti-persistente). O viés de alta perene do ativo (buy&hold)
supera o sinal — exatamente a armadilha do null descrita acima.

> **Conclusão do experimento:** confirma quantitativamente a tese — o espectro
> prevê o 2º momento (volatilidade/regime), é cego ao 1º (direção). Figura:
> `src_newest/direction_experiment.png`. Caveat: testar o leverage em D1 (onde o
> efeito vive) exigiria mais janelas que o D1 fornece por ativo — agregar vários
> ativos resolveria.

## 6. TL;DR

- Autovalores **não** dão direção — são quadráticos, invariantes a `r → −r`.
- Direção mora em objetos lineares: **drift, projeções `aₖ`, reconstruções**.
- Espectro prevê **volatilidade/regime** (2º momento), não **sinal** (1º momento).
- Uso correto: **espectro classifica o regime → condiciona** um sinal direcional
  vindo da reconstrução/drift; ângulos de subespaço sinalizam viradas.
