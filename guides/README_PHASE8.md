# CorrDiff - Fase 8: Baselines Temporais + Auditoria de Continuidade

Versão: `phase8-persistence-baselines-v2-continuity-audited`

Esta versão revisa a Fase 8 anterior. Ela mantém os baselines temporais e adiciona
uma auditoria direta do `train.zarr` para verificar se a persistência quase
perfeita observada nas frações >=30/40/45 dBZ pode estar associada a targets
exatamente repetidos entre `t` e `t-1h`.

## Objetivos

A Fase 8 agora responde duas perguntas:

1. Quanto skill existe apenas usando persistência temporal?
2. Esse skill extremamente alto pode ser explicado por repetição exata do radar
   armazenado?

## Fonte dos baselines

```text
analysis_outputs/06_diurnal/
  timestamp_event_metrics.parquet
  analysis_summary.json
```

## Fonte da auditoria direta

```text
datasets/corrdiff_2011_2024/train.zarr
```

A auditoria lê apenas:

```text
target
timestamps
```

e calcula, para cada timestamp, um hash `BLAKE2b-128` sobre os bytes `float32`
exatamente armazenados no target, incluindo todos os patches daquele timestamp
na ordem armazenada.

Não é usado `expm1`, arredondamento ou threshold durante o hash.

## O que é comparado em t versus t-1h

Para todos os pares UTC exatamente consecutivos:

- igualdade exata do hash do target;
- igualdade de `max_dbz`;
- igualdade da fração positiva;
- igualdade das frações >=30, >=40 e >=45 dBZ.

Os resultados são separados em:

```text
all
both_dry
any_wet
either_ge_30
either_ge_40
either_ge_45
```

A condição `both_dry` é separada porque targets totalmente secos idênticos são
esperados e não constituem, isoladamente, evidência de duplicação artificial.

## Auditoria por ano

Os mesmos indicadores são calculados separadamente para cada ano, permitindo
verificar se a persistência quase perfeita está concentrada em 2022, 2023 ou
2024, ou se ocorre ao longo de toda a série.

## Runs consecutivos idênticos

Também são identificados runs de targets byte-a-byte idênticos em timestamps
consecutivos de 1 h.

São gravados:

```text
start_timestamp_utc
end_timestamp_utc
length_timestamps
duration_hours
dry_target
start_year_utc
```

A análise mais importante é o tamanho dos runs `wet`, porque runs secos podem
ser naturais.

## Baselines temporais

A parte preditiva permanece igual à Fase 8 anterior.

### Eventos

```text
>0 dBZ
>=20 dBZ
>=30 dBZ
>=40 dBZ
>=45 dBZ
```

### Baselines binários

```text
train_prevalence
climatology_month_hour
persistence_1h
persistence_mean_lags
persistence_exp_decay
logistic_temporal
```

### Baselines contínuos

```text
train_mean
climatology_month_hour
persistence_1h
persistence_mean_lags
persistence_exp_decay
ridge_temporal
```

## Split cronológico provisório

```text
Treino:    até 2019-12-31 23:00 UTC
Validação: 2020-01-01 até 2021-12-31 23:00 UTC
Teste:     2022-01-01 em diante
```

Esse split continua provisório. O protocolo final permanece reservado para a
Fase 15.

## Cohort temporal

Por padrão:

```text
lags = 1,2,3,6 h
```

O timestamp alvo só entra no experimento preditivo se todos os lags existirem
exatamente.

A auditoria de continuidade, por outro lado, usa **todos os pares exatos de 1 h**
existentes no dataset e não fica limitada ao cohort comum dos baselines.

Essa distinção é intencional.

## Execução recomendada

```bash
python scripts/08_compute_persistence_baselines.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/08_persistence_baselines \
  --lags 1,2,3,6 \
  --train-end "2019-12-31T23:00:00Z" \
  --val-end "2021-12-31T23:00:00Z" \
  --overwrite
```

A auditoria de hashes percorre o target e pode demorar mais do que a Fase 8
anterior.

Para depuração, é possível pular a auditoria:

```bash
python scripts/08_compute_persistence_baselines.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --output-dir analysis_outputs/08_persistence_baselines \
  --skip-target-audit \
  --overwrite
```

## Resumo

```bash
python scripts/08_summarize_persistence_baselines.py \
  --phase8-dir analysis_outputs/08_persistence_baselines \
  --output analysis_outputs/summary_phase8.txt
```

## Novos outputs da auditoria

```text
radar_timestamp_hashes.parquet
radar_continuity_pair_audit.parquet
radar_continuity_summary.parquet
radar_continuity_by_year.parquet
radar_identical_target_runs.parquet
```

## Outputs dos baselines

```text
evaluation_cohort_summary.parquet
binary_baseline_metrics.parquet
binary_baseline_metrics_by_season.parquet
binary_calibration.parquet
continuous_baseline_metrics.parquet
continuous_baseline_metrics_by_condition.parquet
model_status.parquet
analysis_summary.json
```

## Interpretação da auditoria

### Caso A — igualdade alta apenas em `both_dry`

Isso é compatível com muitos timestamps sem eco e não explica, por si só, a
persistência dos eventos intensos.

### Caso B — igualdade baixa em `any_wet` e `either_ge_40/45`

A persistência elevada provavelmente representa continuidade temporal real da
estrutura radar agregada, ainda sujeita às limitações do dataset.

### Caso C — igualdade alta também em `any_wet` ou `either_ge_40/45`

Antes de interpretar a persistência fisicamente, deve-se investigar:

- seleção do PNG/cache;
- arredondamento/quantização do radar;
- reutilização de arquivo;
- regra de associação do timestamp;
- duplicação no cache;
- possíveis fallbacks na construção do dataset.

## Observação importante

Igualdade de uma fração espacial não significa igualdade do campo. Por isso esta
versão compara o hash do target completo. O hash é a evidência mais forte de
repetição exata dos arrays armazenados.

## Próximo passo

Somente depois desta auditoria a Fase 8 deve ser congelada. Se os targets wet não
estiverem sendo repetidos artificialmente, a Fase 9 pode iniciar a análise de
estrutura espacial do radar.
