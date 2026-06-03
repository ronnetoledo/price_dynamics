# Reamostragem de timeframe (`resample_tf.py`)

Agrega OHLCV de um timeframe fino para um mais grosso e grava novas partições
`timeframe=<alvo>` no mesmo tree de origem, consumíveis direto por
`decomp_pca.load_ohlcv`. Módulo: [src_newest/resample_tf.py](../src_newest/resample_tf.py).

## Agregação

`open=first, high=max, low=min, close=last, volume/tick_volume/trade_count=soma`.
`vwap = Σ(vwap·volume)/Σvolume` — **aproximação** do vwap de trades crus (agrega
vwaps já agregados); a diferença é pequena mas não-nula.

## Convenções de binagem (validadas contra as barras nativas da Alpaca)

- **Alvos intraday** (`M5, M15, M30, H1, H4`): binagem por relógio **UTC**,
  left-labeled, closed-left. **Reproduz a H1 nativa exatamente** (M1→H1 em 2021
  inteiro: OHLC e volume idênticos, dif. máx. 0).
- **Alvos de calendário** (`D1, W1, MN, Q`): agrupados por período em horário de
  **Nova York** (DST-aware), rotulados na **meia-noite-ET** (= 04:00 EDT / 05:00
  EST em UTC), como a nativa. `W1` ancora na **segunda-feira**; `MN`/`Q` no 1º dia
  do mês/trimestre.

## Origem recomendada por alvo

| Alvo | Origem ideal | Casa a nativa? |
|---|---|---|
| M5, M15, M30, H1, H4 | M1 (ou intraday mais fino) | **Sim, exato** |
| W1, MN, Q | **D1** | **Sim, exato** (D1→W1: 261 semanas, dif. 0) |
| D1 (all-hours) | M1 | Não — ver abaixo |

A Alpaca constrói a `W1` nativa reamostrando a própria `D1`; por isso semanal/
mensal/trimestral devem sair da **D1**, não do intraday.

## Por que D1-a-partir-de-intraday NÃO bate com a D1 nativa

A `D1`/`W1` nativas vêm do **feed diário consolidado** (leilões oficiais de
abertura/fechamento e volume diário reportado), que **não** é a soma das barras de
minuto:
- o **close** nativo é o print do leilão das 16:00 ET (≈ abertura da barra de
  minuto 16:00), não o close do último minuto;
- o **open** nativo é o leilão de abertura, ≠ open do 1º minuto;
- o **volume** diário difere alguns % (odd-lots, reporte de leilão).

Logo, reconstruir `D1` do `M1` dá uma diária **all-hours** (que não existe
nativamente — útil!), mas ~0.05–0.7% diferente da `D1` nativa. Para análise que
precise casar a nativa, usar a `D1` do tree e reamostrar dela para cima.

## Extended hours

Por padrão agrega **tudo** (pré/pós-mercado, 08:00–23:59 UTC — como a H1 nativa).
`--regular-hours` restringe a **09:30–16:00 ET** (aproxima a sessão da D1 nativa).

## Split-adjust

Lê via o tree resolvido por `load_ohlcv` (default `adjusted=True`): símbolos com
split saem de `data_parquet_adj/` → o alvo herda o split-adjust automaticamente e
é gravado no mesmo tree ajustado. Símbolos sem split saem/gravam no raw
`data_parquet/`. `--raw` força o tree bruto. Ver [splits.md](splits.md).

## Uso

```bash
cd src_newest

python resample_tf.py AAPL --from M1 --to M5            # intraday, adjusted
python resample_tf.py --all --from D1 --to W1           # semanal de todos (da D1)
python resample_tf.py AAPL --from D1 --to Q             # trimestral
python resample_tf.py NVDA --from M1 --to D1 --regular-hours   # diária só-pregão
python resample_tf.py MSFT --from M1 --to M15 --raw --dry-run  # raw, sem gravar
```

Flags: `--regular-hours`, `--raw`, `--dry-run`, `--overwrite`, `--all`.
A origem precisa ser mais fina que o alvo (validado por `_TF_MIN`).
