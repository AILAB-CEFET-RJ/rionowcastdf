# CorrDiff - Fase 15: Split Formal de Treino / Validação / Teste / Stress-OOD

Versão: `phase15-formal-splits-v1-purged-temporal-ood2024`

A Fase 15 inaugura o bloco diretamente orientado ao treinamento. Ela congela o protocolo temporal que será usado nas fases seguintes e transforma as conclusões da Fase 14 em uma política operacional de separação de dados.

## Protocolo principal

```text
train            = 2011-2020
validation       = 2021-2022
test_primary     = 2023
test_stress_ood  = 2024
```

O racional é:

- `2011-2020`: período histórico amplo para ajuste;
- `2021-2022`: seleção de arquitetura e hiperparâmetros sem tocar nos testes;
- `2023`: teste cronológico principal;
- `2024`: teste de stress/OOD separado, porque a Fase 14 mostrou forte perda de coverage e covariate shift.

O conjunto de 2024 **não deve ser misturado ao headline metric principal**.

## Purge temporal

Por padrão:

```text
48 h antes da fronteira
+
48 h depois da fronteira
```

são excluídas em cada transição entre splits.

Isso é deliberadamente conservador e reduz o risco de o mesmo sistema meteorológico persistente aparecer dos dois lados da fronteira.

As fronteiras são:

```text
2021-01-01 00 UTC
2023-01-01 00 UTC
2024-01-01 00 UTC
```

O purge é aplicado em nível de timestamp; todos os patches do mesmo timestamp herdam o mesmo status.

## Princípio central

A Fase 15 **não estratifica temporalmente pelo target**.

Não serão movidos timestamps entre anos para equilibrar eventos. Diferenças naturais de prevalência fazem parte do teste de generalização.

Desbalanceamento de extremos deverá ser tratado apenas dentro do treino, por exemplo com:

```text
sampler
class/event weighting
curriculum
loss weighting
oversampling de eventos
```

sem alterar validation/test.

## Inputs

```text
datasets/corrdiff_2011_2024/train.zarr

analysis_outputs/06_diurnal/
  timestamp_event_metrics.parquet
  predictor_timestamp_means.parquet

analysis_outputs/13_regimes/
  regime_assignments.parquet

analysis_outputs/14_distribution_shift/
  coverage_by_year.parquet
  regime_composition_shift.parquet
  predictor_shift_anomaly_metrics.parquet
  event_rate_decomposition.parquet
```

## Outputs principais

### Manifesto temporal

```text
timestamp_split_manifest.parquet
```

Uma linha por timestamp radar disponível, contendo:

```text
timestamp_utc
year_utc
month_local
hour_local
season_code
nominal_split
split
is_purged
purge_boundary
role
locked
model_selection_access
final_refit_eligible_pre_primary_test
```

### Índices de patches

```text
train_patch_indices.npy
validation_patch_indices.npy
test_primary_patch_indices.npy
test_stress_ood_patch_indices.npy
excluded_purge_patch_indices.npy
```

Esses arrays podem ser usados diretamente por um Dataset/Sampler no treinamento.

Também é gerado:

```text
patch_split_codes.npy
```

com um byte por patch:

```text
0 = excluded_purge
1 = train
2 = validation
3 = test_primary
4 = test_stress_ood
```

### Auditorias

```text
split_boundary_audit.parquet
split_coverage.parquet
split_event_balance.parquet
split_event_by_season.parquet
split_regime_balance.parquet
split_regime_shift_vs_train.parquet
split_predictor_shift_vs_train.parquet
split_patch_summary.parquet
```

## Integridade de patches

O dataset atual possui 12 patches por timestamp. Por padrão, a Fase 15 exige:

```text
expected_patches_per_timestamp = 12
```

Se a geometria do builder for alterada intencionalmente, pode-se usar:

```bash
--allow-variable-patches
```

mas isso deve ser tratado como mudança de versão do dataset.

## Uso correto dos conjuntos

### train

Pode ser usado para:

```text
ajuste dos pesos do modelo
normalização
estatísticas de treino
sampler
loss weighting
qualquer transformação data-dependent
```

### validation

Pode ser usado para:

```text
seleção de arquitetura
seleção de hiperparâmetros
early stopping
escolha de checkpoint
```

Não deve ser usado para ajustar os pesos do modelo durante o protocolo de desenvolvimento.

### test_primary

2023 permanece bloqueado para:

```text
avaliação cronológica principal
```

Não deve influenciar escolhas de arquitetura, hiperparâmetros, thresholds ou preprocessing.

### test_stress_ood

2024 é uma avaliação separada de stress/OOD.

A Fase 14 mostrou aproximadamente:

```text
coverage 2023 ~57%
coverage 2024 ~36%
```

além de covariate shift relevante, especialmente em `t_500` e `tcwv`.

Por isso, resultados de 2024 devem sempre ser acompanhados de coverage e contagem de timestamps.

## Protocolo de final refit

Depois que arquitetura, hiperparâmetros, loss, sampler e preprocessing estiverem congelados, é permitido declarar um experimento separado:

```text
final_refit = train + validation = 2011-2022
```

seguido de **uma avaliação** em:

```text
test_primary = 2023
```

O stress test 2024 permanece separado.

Esse protocolo deve ser explicitamente distinguido do desenvolvimento normal para evitar vazamento de informação.

## Execução

```bash
python scripts/15_compute_formal_splits.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase6-dir analysis_outputs/06_diurnal \
  --phase13-dir analysis_outputs/13_regimes \
  --phase14-dir analysis_outputs/14_distribution_shift \
  --output-dir analysis_outputs/15_formal_splits \
  --purge-hours 48 \
  --expected-patches-per-timestamp 12 \
  --local-timezone America/Sao_Paulo \
  --overwrite
```

Depois:

```bash
python scripts/15_summarize_formal_splits.py \
  --phase15-dir analysis_outputs/15_formal_splits \
  --output analysis_outputs/summary_phase15.txt
```

## O que deve ser verificado no resumo

1. quantidade de timestamps e patches por split;
2. coverage de cada período;
3. separação real produzida pelo purge;
4. prevalência de `>=30/40/45 dBZ`;
5. balanço sazonal;
6. prevalência dos regimes;
7. shift de predictors versus train;
8. confirmação de 12 patches por timestamp;
9. ausência de sobreposição entre arrays de índices;
10. 2024 claramente identificado como stress/OOD.

## Relação com as próximas fases

A partir desta fase o protocolo temporal não deve ser alterado em resposta ao desempenho do modelo.

```text
Fase 16 -> baselines formais no split congelado
Fase 17 -> ablações no mesmo protocolo
Fase 18 -> configuração final CorrDiff e matriz experimental
```

Qualquer alteração posterior dos splits deve gerar nova versão explícita da Fase 15.
