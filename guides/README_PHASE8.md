# CorrDiff - Fase 8: Baselines Temporais de Persistência

A Fase 8 transforma a dependência temporal observada na Fase 7 em baselines
preditivos explícitos.

## Objetivo

Responder:

> Quanto skill é obtido apenas carregando informação temporal recente do radar,
> antes de atribuir qualquer ganho ao CorrDiff?

## Fonte

A fase usa:

```text
analysis_outputs/06_diurnal/timestamp_event_metrics.parquet
```

e opcionalmente:

```text
analysis_outputs/06_diurnal/analysis_summary.json
```

Ela não reabre o Zarr.

## Split cronológico provisório

Padrão:

```text
Treino:    até 2019-12-31 23:00 UTC
Validação: 2020-01-01 até 2021-12-31 23:00 UTC
Teste:     2022-01-01 em diante
```

Este split serve apenas para medir baselines da Fase 8.

A definição formal do split experimental permanece reservada para a **Fase 15**.

## Cohort comum

Por padrão são usados os lags:

```text
1, 2, 3 e 6 horas
```

Um timestamp alvo entra na avaliação somente se todos esses timestamps anteriores
existirem exatamente em UTC.

Isso garante comparação justa entre os baselines temporais.

## Eventos

```text
>0 dBZ
>=20 dBZ
>=30 dBZ
>=40 dBZ
>=45 dBZ
```

## Baselines binários

### train_prevalence

Probabilidade constante igual à prevalência do evento no treino.

### climatology_month_hour

Probabilidade estimada apenas no treino:

```text
P(evento | mês local, hora local)
```

É a referência para o Brier Skill Score.

### persistence_1h

```text
evento(t) ~= evento(t-1h)
```

### persistence_mean_lags

Média dos indicadores nos lags configurados.

### persistence_exp_decay

Média ponderada com:

```text
w(k) ∝ exp(-k / tau)
```

com `tau=3 h` por padrão.

### logistic_temporal

Regressão logística simples usando:

```text
evento(t-1h)
evento(t-2h)
evento(t-3h)
evento(t-6h)
climatologia(mês,hora)
```

A regressão é ajustada somente no treino.

## Baselines contínuos

Para:

```text
max_dbz
positive_pixel_fraction
event_pixel_fraction_ge_30
event_pixel_fraction_ge_40
event_pixel_fraction_ge_45
```

São avaliados:

```text
train_mean
climatology_month_hour
persistence_1h
persistence_mean_lags
persistence_exp_decay
ridge_temporal
```

O Ridge usa os lags da própria métrica mais a climatologia mês × hora.

## Métricas binárias

- Brier score
- Brier Skill Score versus climatologia
- log loss
- ROC AUC
- Average Precision
- Expected Calibration Error com 10 bins

Para eventos raros, dê maior peso interpretativo a:

```text
Brier
Brier Skill Score
Average Precision
calibração
```

## Métricas contínuas

- MAE
- RMSE
- bias
- Pearson
- Spearman
- skill MAE versus climatologia
- skill RMSE versus climatologia

Também são produzidas métricas condicionadas a:

```text
all
gt_0
ge_30
ge_40
ge_45
```

Isso reduz o risco de uma conclusão ser dominada por timestamps secos.

## Execução

```bash
python scripts/08_compute_persistence_baselines.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --output-dir analysis_outputs/08_persistence_baselines \
  --lags 1,2,3,6 \
  --train-end "2019-12-31T23:00:00Z" \
  --val-end "2021-12-31T23:00:00Z" \
  --overwrite
```

## Resumo textual

```bash
python scripts/08_summarize_persistence_baselines.py \
  --phase8-dir analysis_outputs/08_persistence_baselines \
  --output analysis_outputs/summary_phase8.txt
```

## Principais outputs

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

## Como interpretar skill

Para Brier:

```text
BSS = 1 - Brier_model / Brier_climatologia
```

Logo:

```text
BSS > 0  -> melhora sobre climatologia
BSS = 0  -> equivalente à climatologia
BSS < 0  -> pior que climatologia
```

Para MAE/RMSE, a mesma convenção é usada:

```text
skill = 1 - erro_modelo / erro_climatologia
```

## Limitações

1. O cohort comum favorece períodos com sequência temporal completa.
2. A disponibilidade observacional não é necessariamente aleatória.
3. Frações espaciais são calculadas sobre patches sobrepostos.
4. Este experimento mede baselines timestamp-level, não skill espacial
   pixel-a-pixel do CorrDiff.
5. O split é provisório e será revisitado na Fase 15.

## Relação com as próximas fases

A Fase 8 fecha o bloco temporal inicial:

```text
Fase 5 -> sazonalidade
Fase 6 -> ciclo diurno
Fase 7 -> dependência temporal
Fase 8 -> skill de persistência
```

A Fase 9 deve iniciar a análise da estrutura espacial do radar.
