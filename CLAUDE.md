# Dinâmica de Preços — Contexto do Projeto

## O que é este projeto
Pesquisa em física computacional aplicada a mercados financeiros. O objetivo é caracterizar a dinâmica espectral de preços de ativos usando métodos de física estatística (PCA, MSD, expoentes de difusão β, espaços de Hilbert). Há um artigo em elaboração para o Physical Review Letters (PRL).

## Metodologia central
- **PCA 4D**: séries temporais de preços são embedidas em subespaços de dimensão 4 via janelas deslizantes. A estrutura espectral dessas matrizes de covariância é analisada ao longo do tempo.
- **MSD (Mean Squared Displacement)**: mede a difusão do processo X_t = cumsum(K_t − ⟨K⟩), onde K_t são as matrizes espectrais. O expoente β caracteriza o regime de difusão (normal, sub, super).
- **Otimização de bins**: método de Shimazaki-Shinomoto para binagem ótima de histogramas.
- **Ângulos de subespaço**: `scipy.linalg.subspace_angles` para medir rotação entre subespaços espectrais consecutivos.
- **Parâmetros `embed`/`window`/`step`**: explicação conceitual (embedding de Hankel, amostra local, passada temporal) em `docs/step_embed_window.md`.
- **Direção vs. espectro**: por que autovalores são cegos à direção (forma quadrática, invariante a r→−r) e como usar o espectro como classificador de regime — `docs/direcao_e_espectro.md`.

## Arquitetura de dados
- **Dados históricos**: arquivos Parquet particionados em `data_parquet/source=<fonte>/symbol=<ATIVO>/timeframe=<TF>/year=<ANO>/data.parquet`
  - Fontes: `metatrader` (B3 — PETR4, VALE3, BOVA11) e `alpaca` (EUA — SPY, NVDA, AAPL, MSFT, AMZN etc.)
  - Conversão feita por `src_newest/csv_to_parquet.py`
- **Dados live (planejado)**: QuestDB para ingestão de stream de preços em tempo real
- **CSVs originais**: `src_newest/B3_DATA/` (MetaTrader) e `alpaca/SPY500_DATA/` (Alpaca)

## Estrutura de código principal (`src_newest/`)
- `experimento_PCA_4D_3.py` — script principal de análise; orquestra o pipeline completo
- `msd_beta_sliding_window_2.py` — cálculo do MSD e ajuste do expoente β
- `csv_to_parquet.py` — conversor CSV → Parquet (já executado)
- `Extrator_pdf.py` — extração de figuras/resultados para PDF
- `result_cache.py` — cache de resultados intermediários

## Estado atual (maio 2026)
- Conversão CSV → Parquet concluída para dados B3 e Alpaca (SPY500)
- Pipeline de análise rodando: últimos resultados em `src_newest/run_20260512_144857/`
  - Ativos analisados: NVDA, AAPL, MSFT, AMZN (dados diários, timeframe D1)
- Fase atual: **exploração dos dados Parquet com o pipeline de cálculo espectral**
- Artigo LaTeX: `CODE/04302026_tex_ultima_versão/spectral_dynamics_NP_2.tex`

## Ativos e timeframes disponíveis
- **B3**: PETR4, VALE3, BOVA11 — M1, M5, M15, M30, H1, H4, D1
- **EUA (Alpaca)**: SPY e ações do S&P500 — H1, D1

## Instruções para o assistente
- Ao aprender algo novo sobre o projeto, decisões arquiteturais ou preferências do usuário, **salvar memória** em `C:\Users\ronne\.claude\projects\e--ronne-Documentos-PROJETOS-DINAMICA-DE-PRE-OS\memory\`
- O usuário tem formação em física e domina os conceitos de álgebra linear, séries temporais e métodos espectrais — não precisa de explicações básicas dessas áreas
- Preferir editar arquivos existentes a criar novos
- Código sem comentários óbvios; comentar apenas invariantes não-óbvias
