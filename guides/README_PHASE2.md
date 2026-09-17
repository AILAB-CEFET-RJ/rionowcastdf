# CorrDiff — Fase 2 v2: variáveis derivadas

Esta versão corrige a correlação entre as variáveis derivadas.

## Correção

Na versão anterior, cada variável mantinha um reservoir aleatório independente. A correlação era calculada juntando esses arrays por posição, embora cada posição pudesse representar pixels diferentes.

Na v2 existe um **reservoir multivariado alinhado**: quando um pixel é selecionado, todas as 8 variáveis derivadas daquele mesmo pixel são armazenadas juntas. `derived_correlations.parquet` passa a ser válido.

As estatísticas univariadas da Fase 2 anterior não estavam afetadas por esse problema.

## Execução

```bash
python scripts/02_compute_derived_variables.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase1-dir analysis_outputs/01_univariate \
  --output-dir analysis_outputs/02_derived \
  --sample-patches 32768 \
  --reservoir-values 300000 \
  --overwrite
```

## Principais saídas

- `derived_summary.parquet`
- `derived_quantiles.parquet`
- `derived_histograms.parquet`
- `derived_correlations.parquet` — Pearson + Spearman em pixels alinhados
- `derived_samples.npz` — reservoir alinhado
- `analysis_summary.json`

O `analysis_summary.json` deve mostrar:

```text
correlation_alignment_verified: true
```

e informar `aligned_reservoir_rows`.
