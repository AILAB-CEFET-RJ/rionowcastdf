# CorrDiff — Fase 6: Ciclo Diurno

A Fase 6 estuda o ciclo diurno da refletividade radar **em dBZ** e dos preditores ERA5, sempre condicionando as frequências à disponibilidade observacional.

## Métrica principal

`P(evento | radar disponível, hora)`

A principal medida temporal é `timestamp_any_event_rate`.

## UTC e hora civil local

São gerados perfis em UTC e em `America/Sao_Paulo`. A conversão usa regras históricas de timezone, incluindo o antigo horário de verão brasileiro. O denominador esperado também é convertido a partir da grade UTC.

## Controle sazonal

Como a Fase 5 mostrou forte sazonalidade, a Fase 6 produz:

1. perfil diurno bruto;
2. perfil hora × estação;
3. perfil local padronizado por estação, dando peso igual a DJF/MAM/JJA/SON e usando apenas season-years completos.

## Frequência, extensão e intensidade

Para cada limiar são calculados:

- `timestamp_any_event_rate` — frequência temporal;
- `pixel_event_rate` — frequência nos pixels armazenados;
- `patch_any_event_rate`;
- `mean_event_pixel_fraction_given_event_timestamp` — extensão espacial média condicionada à ocorrência;
- `mean_positive_dbz`;
- `mean_timestamp_max_dbz`.

## Radar

O builder usa `target = log1p(clip(dBZ, 0, None))`. Os limiares fixos 20/25/30/35/40/45 dBZ são avaliados em `stored_log1p_float32` para evitar artefatos numéricos em fronteiras.

## Execução

```bash
python scripts/06_compute_diurnal_cycle.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/06_diurnal \
  --local-timezone America/Sao_Paulo \
  --overwrite
```

Para uma execução inicial só com radar/cobertura:

```bash
python scripts/06_compute_diurnal_cycle.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/06_diurnal \
  --local-timezone America/Sao_Paulo \
  --skip-predictors \
  --overwrite
```

## Resumo textual

```bash
python scripts/06_summarize_diurnal_cycle.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --output analysis_outputs/summary_phase6.txt
```

## Saídas principais

- `hourly_coverage_utc.parquet`
- `hourly_coverage_local.parquet`
- `hourly_event_rates_local.parquet`
- `season_hour_event_rates_local.parquet`
- `standardized_hourly_event_rates_local.parquet`
- `hourly_predictor_statistics_local.parquet`
- `analysis_summary.json`

Comece a interpretação pelo perfil local padronizado por estação. Depois compare-o ao perfil bruto para identificar quanto do sinal diurno pode decorrer de composição sazonal desigual.
