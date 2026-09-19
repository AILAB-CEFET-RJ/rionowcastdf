# CorrDiff - Fase 5: Sazonalidade

Esta fase foi desenhada a partir das conclusões das Fases 0-4, em especial da
heterogeneidade da cobertura temporal (69% no dataset reconstruído).

## Pergunta central

A sazonalidade do radar deve ser calculada como frequência **condicionada à
disponibilidade**:

`P(evento | radar disponível, mês/estação)`

e não como contagem bruta de eventos.

## Arquivos

- `scripts/05_compute_seasonality.py`
- `scripts/05_summarize_seasonality.py`
- `notebooks/05_seasonality_analysis.ipynb`

## Execução

```bash
python scripts/05_compute_seasonality.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/05_seasonality \
  --overwrite
```

A etapa ERA5 lê o input completo para calcular uma média espacial por timestamp
dos 12 canais brutos e 8 derivados. Se quiser executar inicialmente apenas
cobertura + radar:

```bash
python scripts/05_compute_seasonality.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/05_seasonality \
  --skip-predictors \
  --overwrite
```

## Resumo textual

```bash
python scripts/05_summarize_seasonality.py \
  --phase5-dir analysis_outputs/05_seasonality \
  --output analysis_outputs/summary_phase5.txt
```

## Métricas principais

Para cada mês e estação:

- `coverage_ratio`
- `pixel_event_rate`
- `timestamp_any_event_rate`
- `patch_any_event_rate`
- `mean_positive_radar`
- `mean_timestamp_max_radar`

Eventos:

- `gt_0`
- `ge_20`
- `ge_30`
- `ge_40`
- `ge_45`

O script também grava `ge_25` e `ge_35`.

## Estações

- DJF = verão
- MAM = outono
- JJA = inverno
- SON = primavera

A climatologia sazonal usa apenas `season_years` completos dentro do período
configurado para não misturar DJF parcial nas bordas de 2011/2024.

## Semântica do radar

Limiares fixos são avaliados no domínio `stored_log1p_float32`, exatamente como
na Fase 4 v3.1. Estatísticas contínuas usam `expm1(target)` e permanecem
descritas como **valor numérico da legenda do radar**, sem assumir unidade física
dBZ.

## Interpretação

- `timestamp_any_event_rate` é a principal métrica temporal para responder se
  um evento estava presente em um timestamp disponível.
- `pixel_event_rate` continua representando a distribuição armazenada em patches
  sobrepostos.
- Não interpretar contagens brutas como climatologia quando a disponibilidade
  varia entre meses/anos.
