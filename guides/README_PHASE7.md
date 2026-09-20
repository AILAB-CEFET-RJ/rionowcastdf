# CorrDiff - Fase 7: Dependência Temporal e Lags

A Fase 7 investiga quanto do estado do radar em `t` está associado ao radar e ao estado atmosférico em `t-k`.

## Base metodológica

Esta fase usa os outputs agregados da Fase 6 e **não reabre o Zarr**. Isso evita assumir que a posição interna dos patches seja espacialmente alinhada entre timestamps diferentes.

Entradas esperadas:

```text
analysis_outputs/06_diurnal/
  timestamp_event_metrics.parquet
  predictor_timestamp_means.parquet
  analysis_summary.json
```

## Lags padrão

```text
0, 1, 2, 3, 6, 12 e 24 horas
```

Um par só é usado quando existem exatamente `t` e `t-k`. Gaps temporais nunca são tratados como observações contíguas.

## Blocos de análise

### 1. Autocorrelação do radar

Para:

- fração de pixels positivos;
- máximo em dBZ por timestamp;
- frações espaciais >=30, >=40 e >=45 dBZ.

São calculados Pearson e Spearman, tanto brutos quanto ajustados.

### 2. Dependência temporal dos eventos

Eventos:

- `>0 dBZ`
- `>=20 dBZ`
- `>=30 dBZ`
- `>=40 dBZ`
- `>=45 dBZ`

Para cada lag são produzidos:

```text
P(evento_t | evento_t-k)
P(evento_t | não-evento_t-k)
diferença absoluta
risco relativo
coeficiente phi
autocorrelação residual do evento
```

A análise também é repetida por estação.

### 3. Preditor(t-k) -> radar(t)

Os 12 canais ERA5 e 8 variáveis derivadas são comparados com:

- ocorrência dos eventos;
- `max_dbz`;
- fração de pixels positivos;
- frações >=30, >=40 e >=45 dBZ.

## Ajuste sazonal e diurno

Como as Fases 5 e 6 mostraram sazonalidade e ciclo diurno fortes, a Fase 7 repete as associações depois de remover a climatologia média de cada combinação:

```text
mês local x hora local
```

A anomalia é:

```text
valor - média(mês_local, hora_local)
```

Esse ajuste reduz confundimento climatológico, mas continua sendo descritivo e não causal.

## Unidade do radar

A unidade física é **dBZ**. O builder armazena:

```text
log1p(clip(dBZ, min=0))
```

A Fase 7 trabalha apenas com as métricas já reconstruídas pela Fase 6.

## Execução

```bash
python scripts/07_compute_temporal_lags.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --output-dir analysis_outputs/07_lags \
  --lags 0,1,2,3,6,12,24 \
  --overwrite
```

## Resumo textual

```bash
python scripts/07_summarize_temporal_lags.py \
  --phase7-dir analysis_outputs/07_lags \
  --output analysis_outputs/summary_phase7.txt
```

## Principais outputs

```text
lag_pair_coverage.parquet
radar_lag_autocorrelation.parquet
event_lag_dependence.parquet
event_lag_dependence_by_season.parquet
predictor_event_lag_associations.parquet
predictor_radar_lag_associations.parquet
analysis_summary.json
```

## Limite entre Fase 7 e Fase 8

A Fase 7 caracteriza dependência temporal. A Fase 8 deve transformar isso em um experimento preditivo formal de persistência, usando splits temporais sem vazamento e comparando baselines como:

```text
radar(t) ≈ radar(t-1h)
evento(t) ≈ evento(t-1h)
climatologia sazonal/diurna
persistência + climatologia
```
