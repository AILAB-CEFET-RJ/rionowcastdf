# CorrDiff - Fase 14: Distribution Shift e Estabilidade Temporal

Versão: `phase14-distribution-shift-v1-coverage-regime-conditional`

A Fase 14 encerra o bloco de diagnóstico do dataset antes da definição formal
dos splits de treinamento.

O objetivo é separar quatro fontes de mudança temporal que podem afetar o
desempenho do CorrDiff:

```text
1. coverage / disponibilidade do radar
2. shift marginal dos predictors ERA5
3. mudança de composição dos regimes meteorológicos
4. mudança de P(radar | regime)
```

A fase foi desenhada especialmente para investigar a deterioração observada no
fold F5 da Fase 12.

---

## 1. Inputs

A fase reutiliza outputs agregados já existentes.

### Fase 6

```text
analysis_outputs/06_diurnal/
├── predictor_timestamp_means.parquet
└── timestamp_event_metrics.parquet
```

### Fase 13

```text
analysis_outputs/13_regimes/
└── regime_assignments.parquet
```

Nenhuma releitura do Zarr completo é necessária.

---

## 2. Janelas de comparação

Duas famílias de comparação são usadas.

### Comparações anuais contra referência fixa

```text
2011-2020  vs  2021
2011-2020  vs  2022
2011-2020  vs  2023
2011-2020  vs  2024
```

Isso permite comparar os anos recentes contra a mesma referência histórica.

### Comparação orientada ao F5

```text
2011-2022  vs  2023-2024
```

Essa janela reproduz conceitualmente o problema temporal observado no último
fold da Fase 12:

```text
treino histórico
      ↓
avaliação 2023-2024
```

---

## 3. Coverage shift

A fase reconstrói a grade horária esperada entre:

```text
2011-01-01 00 UTC
e
2024-12-31 23 UTC
```

e marca cada timestamp como:

```text
available
missing
```

São gerados:

```text
coverage_by_year.parquet
coverage_by_month.parquet
missing_gap_runs.parquet
```

Isso responde:

- qual o percentual de horas disponíveis por ano;
- quais meses são particularmente incompletos;
- quais são os maiores gaps contínuos;
- se 2024 possui coverage suficientemente diferente para contaminar análises de
  shift do target.

A interpretação física de qualquer shift no radar deve ocorrer somente depois
desse diagnóstico.

---

## 4. Shift marginal dos predictors ERA5

Para cada predictor são comparadas as distribuições de referência e avaliação.

Métricas:

```text
standardized mean difference
std ratio
KS statistic
PSI
p10 shift / sigma_ref
p50 shift / sigma_ref
p90 shift / sigma_ref
```

### Standardized Mean Difference

```text
SMD = (mean_eval - mean_ref) / std_ref
```

Facilita comparar predictors com unidades diferentes.

### KS

Mede a maior distância entre as CDFs empíricas.

É sensível não somente à média, mas à forma geral da distribuição.

### PSI

O Population Stability Index usa bins quantílicos derivados da referência.

Na Fase 14 o PSI é tratado como um indicador complementar.

Não há classificação automática como "baixo/moderado/alto" porque thresholds
de PSI são convenções de domínio e não leis estatísticas universais.

---

## 5. Absolute vs anomaly shift

O shift de predictor é calculado de duas formas.

### `absolute`

Usa diretamente os valores da Fase 6.

Pode refletir:

```text
mudança de estação observada
mudança do ciclo diurno amostrado
mudança meteorológica real
```

### `reference_month_hour_anomaly`

Para cada comparação, uma climatologia:

```text
mês local × hora local
```

é ajustada **somente no período de referência**.

A anomalia é então:

```text
predictor(t) - climatologia_ref(mês, hora)
```

Esse espaço reduz o efeito de mudanças meramente sazonais ou diurnas.

---

## 6. Shift de composição dos regimes

A Fase 14 reutiliza os regimes da Fase 13.

Para cada espaço:

```text
absolute
anomaly_month_hour
```

mede:

```text
Jensen-Shannon divergence
Total Variation distance
máximo delta de prevalência
delta por regime
```

### Total Variation

```text
TV = 1/2 * sum_r |P_eval(r) - P_ref(r)|
```

Pode ser interpretada como o tamanho total da redistribuição entre regimes.

### Jensen-Shannon

É uma medida simétrica de diferença entre distribuições discretas.

Ela não depende da ordenação dos regimes.

---

## 7. Shift marginal do radar

Para os eventos:

```text
gt_0
ge_20
ge_30
ge_40
ge_45
```

são comparados:

```text
event rate reference
event rate evaluation
difference
risk ratio
```

com intervalos de confiança obtidos por block bootstrap.

Também são avaliados:

```text
max_dbz
positive_pixel_fraction
event_pixel_fraction_ge_30
event_pixel_fraction_ge_40
event_pixel_fraction_ge_45
```

quando disponíveis.

---

## 8. Block bootstrap temporal

As observações horárias não são independentes.

Por isso a fase não usa bootstrap por linha.

O bloco padrão é:

```text
UTC day
```

Todos os timestamps do mesmo dia permanecem juntos.

Por padrão:

```text
500 bootstrap replications
```

Isso preserva parte da dependência temporal intra-diária.

Ainda existe dependência entre dias consecutivos pertencentes ao mesmo sistema
meteorológico, portanto os intervalos não devem ser tratados como independência
perfeita.

---

## 9. Conditional shift dentro dos regimes

Essa é uma das partes centrais da Fase 14.

Para cada regime:

```text
P(event | regime, reference)
vs
P(event | regime, evaluation)
```

é calculado com bootstrap diário.

Isso permite separar duas hipóteses.

### Composition shift

```text
P(regime) mudou
```

mas:

```text
P(event | regime)
```

permaneceu aproximadamente estável.

### Conditional / concept-like shift

```text
P(event | regime)
```

também mudou.

Essa segunda situação é particularmente relevante para o treinamento porque
significa que o mesmo tipo de ambiente ERA5 passou a corresponder a uma
distribuição radar diferente.

---

## 10. Decomposição da mudança de taxa do evento

Para cada regime `r`:

```text
p0(r) = prevalência do regime na referência
p1(r) = prevalência do regime na avaliação

q0(r) = P(event | regime, referência)
q1(r) = P(event | regime, avaliação)
```

A mudança total:

```text
Delta = P1(event) - P0(event)
```

é decomposta simetricamente.

### Componente de composição

```text
sum_r (p1 - p0) * (q0 + q1) / 2
```

### Componente within-regime

```text
sum_r (p0 + p1) / 2 * (q1 - q0)
```

Os dois componentes somam exatamente a mudança observada, salvo erro numérico.

Essa decomposição é equivalente a uma alocação Shapley simétrica da interação.

Ela responde:

> quanto da mudança de `>=45 dBZ` ocorreu porque regimes diferentes se tornaram
> mais frequentes?

versus:

> quanto ocorreu porque a taxa de `>=45 dBZ` mudou dentro dos próprios regimes?

Isso é uma decomposição estatística, não causal.

---

## 11. Execução

```bash
python scripts/14_compute_distribution_shift.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --phase13-dir analysis_outputs/13_regimes \
  --output-dir analysis_outputs/14_distribution_shift \
  --dataset-start-utc "2011-01-01 00:00:00+00:00" \
  --dataset-end-utc "2024-12-31 23:00:00+00:00" \
  --local-timezone America/Sao_Paulo \
  --bootstrap-reps 500 \
  --overwrite
```

---

## 12. Resumo textual

```bash
python scripts/14_summarize_distribution_shift.py \
  --phase14-dir analysis_outputs/14_distribution_shift \
  --output analysis_outputs/summary_phase14.txt
```

---

## 13. Outputs

```text
comparison_windows.parquet

coverage_by_year.parquet
coverage_by_month.parquet
missing_gap_runs.parquet

predictor_shift_metrics.parquet
predictor_shift_anomaly_metrics.parquet

regime_composition_shift.parquet
regime_prevalence_delta.parquet

radar_event_shift.parquet
radar_continuous_shift.parquet

conditional_event_shift_by_regime.parquet
event_rate_decomposition.parquet

analysis_summary.json
phase14.log
```

---

## 14. Ordem de interpretação

A ordem recomendada é:

```text
1. coverage por ano
2. coverage por mês e maiores gaps
3. predictor shift absoluto
4. predictor shift em anomalia mês×hora
5. regime composition shift
6. radar marginal shift
7. conditional >=45 dentro dos regimes
8. decomposição composition vs within-regime
```

Essa ordem evita interpretar como mudança física algo que pode ser consequência
de missingness.

---

## 15. Foco especial: 2023-2024

A comparação:

```text
RECENT_2023_2024_vs_pre2023
```

é a principal para conectar a Fase 14 à degradação do F5 da Fase 12.

Devem ser avaliados conjuntamente:

```text
predictor SMD / KS / PSI
regime TV / JS
P(>=45)
P(>=45 | regime)
decomposição composition/within
coverage 2023/2024
```

---

## 16. Interpretação dos possíveis cenários

### Cenário A - composition dominates

```text
P(regime) muda
P(event | regime) relativamente estável
```

Implica que o modelo encontrou uma distribuição de ambientes diferente.

Consequência para Fase 15:

```text
split deve preservar/regulamentar composição de regimes
```

### Cenário B - within-regime dominates

```text
P(regime) pouco muda
P(event | regime) muda
```

Isso sugere mudança mais profunda na relação condicionamento-target.

Consequência:

```text
split temporal deve permanecer estrito
e avaliação de extrapolação ganha importância
```

### Cenário C - coverage dominates

Se 2024 tiver coverage muito diferente:

```text
shift aparente no radar
pode ser seleção temporal
```

Nesse caso a Fase 15 deve controlar explicitamente:

```text
coverage
meses disponíveis
estações
eventos
```

### Cenário D - mixed shift

É provavelmente o caso mais realista:

```text
coverage
+
predictor shift
+
regime shift
+
conditional shift
```

A Fase 14 foi construída justamente para quantificar essa mistura.

---

## 17. Relação com a Fase 15

A Fase 14 encerra o bloco de análise diagnóstica.

A Fase 15 deve usar os resultados daqui para definir formalmente:

```text
train
validation
test
```

controlando:

```text
tempo
regime
estação
extremos
coverage
gaps
possible shift
```

A partir da Fase 15 o trabalho passa a ser diretamente orientado ao protocolo
de treinamento e avaliação do CorrDiff.
