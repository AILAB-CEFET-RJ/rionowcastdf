# CorrDiff — Fase 2: Variáveis derivadas

A Fase 2 constrói diagnósticos meteorológicos a partir dos 12 canais ERA5 já auditados nas Fases 0 e 1. O dataset original não é alterado.

## Variáveis derivadas

- `wind_speed_10 = sqrt(u10² + v10²)`
- `wind_speed_850 = sqrt(u_850² + v_850²)`
- `wind_speed_500 = sqrt(u_500² + v_500²)`
- `delta_t_500_850 = t_500 - t_850`
- `delta_t_850_surface = t_850 - t2m`
- `delta_r_500_850 = r_500 - r_850`
- `bulk_wind_diff_10_850 = sqrt((u_850-u10)² + (v_850-v10)²)`
- `bulk_wind_diff_850_500 = sqrt((u_500-u_850)² + (v_500-v_850)²)`

As duas diferenças vetoriais de vento são proxies de cisalhamento em m/s. Como não são normalizadas pela distância vertical, não devem ser interpretadas como taxas de cisalhamento em s⁻¹.

## Execução

```bash
python scripts/02_compute_derived_variables.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase1-dir analysis_outputs/01_univariate \
  --output-dir analysis_outputs/02_derived \
  --sample-patches 32768 \
  --reservoir-values 300000 \
  --hist-bins 120 \
  --overwrite
```

Depois abra:

```text
notebooks/02_derived_variables.ipynb
```

## Saídas

```text
analysis_outputs/02_derived/
├── analysis_summary.json
├── formula_catalog.parquet
├── sampling_blocks.parquet
├── derived_summary.parquet
├── derived_quantiles.parquet
├── derived_histograms.parquet
├── derived_correlations.parquet
├── derived_samples.npz
└── phase2.log
```

## Dependências

Além do ambiente das fases anteriores: `scipy`, `pyarrow`, `zarr`, `numpy`, `pandas`, `matplotlib`.
