# CorrDiff — Fase 3: Relações ERA5 × Radar

A Fase 3 mede associações descritivas entre os 12 canais ERA5, oito diagnósticos derivados e o radar. O radar é reconstruído com `expm1(target)` e tratado como **valor numérico da legenda do radar** até confirmação documental da unidade física.

## Análises

- Pearson.
- Spearman.
- Informação mútua com a intensidade do radar.
- Associação com ocorrência `radar > 0`.
- Relações condicionais por decis do preditor.
- Taxas condicionais `>0`, `>=20`, `>=30` e `>=40`.
- Amostra conjunta para gráficos `hexbin` no notebook.
- Métricas separadas para todos os pixels válidos e para radar positivo.

As relações são descritivas; não representam causalidade. Como os patches se sobrepõem, as estatísticas descrevem a distribuição efetivamente vista pelo treinamento.

## Execução

```bash
python scripts/03_compute_joint_statistics.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/03_joint \
  --sample-patches 16384 \
  --sample-pixels 300000 \
  --mi-sample 60000 \
  --conditional-bins 10 \
  --overwrite
```

Depois abra:

```text
notebooks/03_era5_radar_relationships.ipynb
```

## Saídas

```text
analysis_outputs/03_joint/
├── analysis_summary.json
├── predictor_catalog.parquet
├── sampling_blocks.parquet
├── association_metrics.parquet
├── conditional_bins.parquet
├── relationship_samples.npz
└── phase3.log
```

## Dependências

`scipy`, `scikit-learn`, `pyarrow`, `zarr`, `numpy`, `pandas`, `matplotlib`.
