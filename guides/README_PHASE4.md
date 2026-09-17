# CorrDiff — Fase 4 v2: extremos com amostragem estratificada

A versão anterior reutilizava os 300 mil pixels da Fase 3. Essa amostra representou razoavelmente a ocorrência de radar, mas sub-representou fortemente os limiares mais intensos.

A v2 corrige isso lendo o Zarr diretamente e mantendo reservoirs independentes para estratos disjuntos de intensidade:

```text
0
(0,20)
[20,25)
[25,30)
[30,35)
[35,40)
[40,45)
[45,50)
>=50
```

Cada observação retida recebe um **peso inverso da fração de amostragem do estrato**. Assim, médias, quantis, SMD e curvas por decil são calculados de forma ponderada, reconstruindo a distribuição dos blocos Zarr amostrados.

Para os limiares fixos, o script também usa `analysis_outputs/00_quality/target_event_rates_global.parquet` como referência exata do dataset completo quando disponível.

## Execução recomendada

```bash
python scripts/04_compute_extreme_statistics.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase0-dir analysis_outputs/00_quality \
  --output-dir analysis_outputs/04_extremes \
  --sample-patches 65536 \
  --stratum-size 20000 \
  --overwrite
```

Se houver memória/tempo disponível, aumente `--sample-patches` para 131072.

## Saídas principais

- `stratum_sampling.parquet` — população observada, amostra e peso por estrato
- `event_prevalence.parquet` — prevalência ponderada e referência exata da Fase 0
- `conditional_predictor_stats.parquet` — estatísticas evento/não-evento ponderadas
- `effect_sizes.parquet` — SMD ponderado
- `event_rate_by_predictor_decile.parquet` — curvas condicionais ponderadas
- `extreme_samples.npz` — amostra estratificada local com pesos
- `analysis_summary.json`

## Verificações importantes

Depois da execução, observe:

- `max_abs_fixed_threshold_rate_diff_vs_phase0`
- `stratum_sampling.parquet`
- `effective_n_event` em `effect_sizes.parquet`
- eventos `>=40` e `>=45`

O `event_rate` dos limiares fixos usa a Fase 0 exata quando essa referência está disponível.
