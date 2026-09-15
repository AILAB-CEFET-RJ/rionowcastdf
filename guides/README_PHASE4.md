# CorrDiff — Fase 4: Eventos intensos e extremos

A Fase 4 procura identificar quais condições ERA5 e diagnósticos derivados diferenciam eventos mais intensos de radar.

## Definições de evento

São construídas três famílias:

- limiares fixos: `>=20`, `>=25`, `>=30`, `>=35`, `>=40`, `>=45`, `>=50`;
- quantis globais: `>P90`, `>P95`, `>P99` considerando todos os pixels válidos;
- quantis positivos: `>P90+`, `>P95+`, `>P99+` calculados somente onde `radar > 0`.

A família positiva é importante porque a distribuição do radar é muito esparsa e os quantis globais P90/P95 podem coincidir com zero.

## Análises

- prevalência por definição de evento;
- estatísticas dos preditores em evento versus não-evento;
- diferença média e mediana;
- diferença média padronizada;
- probabilidade do evento por decil do preditor;
- comparação entre limiares fixos e quantílicos.

## Execução recomendada

A Fase 4 reutiliza `relationship_samples.npz` da Fase 3 quando disponível:

```bash
python scripts/04_compute_extreme_statistics.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase3-dir analysis_outputs/03_joint \
  --output-dir analysis_outputs/04_extremes \
  --deciles 10 \
  --overwrite
```

Caso a Fase 3 não exista, o script faz uma amostragem direta do Zarr.

Depois abra:

```text
notebooks/04_extreme_event_conditions.ipynb
```

## Saídas

```text
analysis_outputs/04_extremes/
├── analysis_summary.json
├── predictor_catalog.parquet
├── threshold_definitions.parquet
├── event_prevalence.parquet
├── conditional_predictor_stats.parquet
├── effect_sizes.parquet
├── event_rate_by_predictor_decile.parquet
├── extreme_samples.npz
└── phase4.log
```

## Observação sobre unidade do radar

Os limiares são denominados **unidades da legenda do radar**. Não use `dBZ` nos gráficos ou texto científico até a unidade física ser confirmada na documentação da fonte do produto.
