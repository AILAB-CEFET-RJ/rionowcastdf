# CorrDiff - Fase 13: Regimes Meteorológicos

Versão: `phase13-meteorological-regimes-v1-pca-kmeans-dualspace`

A Fase 13 identifica estados atmosféricos recorrentes a partir dos predictors
ERA5 e mede como ocorrência, intensidade e extensão radar variam entre esses
estados.

Ela foi desenhada para responder às questões que surgiram na Fase 12:

- por que a umidade domina eventos moderados;
- por que a estrutura térmica ganha importância em `>=45 dBZ`;
- em quais estados atmosféricos aparecem os maiores enriquecimentos de extremos;
- se a degradação de 2023–2024 pode estar associada a mudança de composição de
  regimes.

A Fase 13 é descritiva. A Fase 14 continua responsável pelos testes formais de
distribution shift.

## 1. Unidade de análise

A unidade passa a ser o **timestamp**, e não o patch.

A fase reutiliza os outputs agregados da Fase 6:

```text
analysis_outputs/06_diurnal/
├── predictor_timestamp_means.parquet
└── timestamp_event_metrics.parquet
```

Isso evita:

- reler o Zarr completo;
- duplicar estruturas devido a patches sobrepostos;
- definir regimes diretamente a partir do radar.

## 2. Regimes definidos somente pelo ERA5

O clustering usa exclusivamente os 12 canais raw:

```text
tcwv
t2m
u10
v10
t_850
r_850
u_850
v_850
t_500
r_500
u_500
v_500
```

As 8 variáveis derivadas são utilizadas depois para caracterização:

```text
wind_speed_10
wind_speed_850
wind_speed_500
delta_t_500_850
delta_t_850_surface
delta_r_500_850
bulk_wind_diff_10_850
bulk_wind_diff_850_500
```

Isso reduz o risco de variáveis derivadas redundantes dominarem a geometria do
clustering.

O radar entra somente após a definição dos clusters.

## 3. Dois espaços de regime

A fase constrói dois sistemas de regimes.

### `absolute`

Usa os predictors absolutos padronizados.

Responde:

> quais estados atmosféricos de larga escala são recorrentes no período
> histórico?

Esse espaço pode naturalmente refletir:

```text
estação
ciclo anual
ciclo diurno
regime sinótico
```

### `anomaly_month_hour`

Antes do PCA, cada predictor tem removida sua média:

```text
mês local × hora local
```

Responde:

> quais estados atmosféricos anômalos aparecem além do background
> sazonal-diurno esperado?

Essa segunda visão é importante porque evita que o clustering seja apenas uma
separação verão/inverno ou dia/noite.

## 4. PCA

Cada espaço é:

1. padronizado;
2. transformado por PCA;
3. reduzido ao número mínimo de componentes necessário para reter, por padrão,
   `90%` da variância.

Isso diminui:

- colinearidade;
- peso duplicado de grupos de variáveis altamente correlacionadas;
- sensibilidade do KMeans à dimensionalidade bruta.

Outputs:

```text
pca_explained_variance.parquet
pca_loadings.parquet
```

## 5. KMeans

Configuração de referência:

```text
k = 5
```

O valor 5 não é tratado como "número verdadeiro" de regimes.

Também são calculados diagnósticos para:

```text
k = 3,4,5,6,7,8
```

com:

```text
inertia
silhouette
Calinski-Harabasz
Davies-Bouldin
```

O objetivo é avaliar sensibilidade ao número de clusters sem automatizar uma
escolha excessivamente forte.

## 6. Estabilidade

Para o `k` de referência são ajustados múltiplos KMeans com seeds diferentes.

A estabilidade é medida com:

```text
Adjusted Rand Index
```

Saídas:

```text
reference_k_stability_ari_mean
reference_k_stability_ari_min
```

Valores altos indicam que a partição é pouco sensível à inicialização.

## 7. Identificadores dos regimes

Depois do clustering, os regimes são renomeados:

```text
R1 ... Rk
```

ordenando primeiro pela média de `tcwv` e depois por `t2m`.

Isso serve somente para tornar a saída reprodutível e legível.

`R1`, `R2`, etc. não significam:

```text
melhor
pior
mais intenso
menos intenso
```

nem possuem interpretação física ordinal automática.

## 8. Perfil meteorológico

Para cada regime são calculados, nos 20 predictors:

```text
média
mediana
desvio-padrão
anomalia da média em unidades de sigma global
```

O campo:

```text
regime_mean_anomaly_sigma
```

é particularmente útil para interpretar a assinatura de cada regime.

Exemplo conceitual:

```text
tcwv       +1.1 sigma
r850       +0.8 sigma
t2m        +0.5 sigma
ws10       -0.4 sigma
```

## 9. Radar por regime

Depois que os regimes já foram definidos sem radar, são calculados:

```text
P(>0 dBZ | regime)
P(>=20 dBZ | regime)
P(>=30 dBZ | regime)
P(>=40 dBZ | regime)
P(>=45 dBZ | regime)
```

além de:

```text
max_dbz médio
fração positiva média
frações >=30/40/45
```

quando as colunas estão disponíveis na Fase 6.

## 10. Enriquecimento

Para cada evento:

```text
risk_ratio_vs_global =
    P(event | regime) / P(event)
```

e:

```text
absolute_rate_difference =
    P(event | regime) - P(event)
```

Para eventos raros como `>=45 dBZ`, olhe os dois.

Um risk ratio elevado pode ocorrer simplesmente porque a taxa global é muito
pequena.

## 11. Sazonalidade

São produzidas duas leituras complementares:

```text
prevalence_regime_within_season
season_fraction_within_regime
```

A primeira pergunta:

> dentro de DJF, MAM, JJA ou SON, qual fração pertence a cada regime?

A segunda:

> dentro de um regime, qual é sua composição sazonal?

Isso ajuda a separar regimes meteorológicos de simples classificações por
estação.

## 12. Prevalência anual

A Fase 13 calcula:

```text
P(regime | ano)
```

e também:

```text
P(>=45 dBZ | regime, ano)
```

Essas tabelas devem ser inspecionadas especialmente em:

```text
2021
2022
2023
2024
```

por causa da deterioração temporal observada no F5 da Fase 12.

Isso é triagem para a Fase 14, não teste formal de shift.

## 13. Persistência dos regimes

A fase usa apenas pares:

```text
t
t + 1h
```

com timestamps UTC exatamente consecutivos.

Gaps não são preenchidos.

É produzida a matriz:

```text
P(regime(t+1h) | regime(t))
```

e a probabilidade de permanência:

```text
P(mesmo regime em t+1h)
```

Isso fornece uma primeira medida da escala temporal dos estados atmosféricos.

## 14. Crosswalk absolute × anomaly

A saída:

```text
regime_crosswalk.parquet
```

mede quanto cada regime absoluto se distribui pelos regimes de anomalia.

Se houver uma correspondência quase 1:1, o espaço anômalo pouco acrescenta.

Se a correspondência for difusa, o espaço anômalo está capturando estrutura
meteorológica distinta do background sazonal.

## 15. Execução

```bash
python scripts/13_compute_meteorological_regimes.py \
  --phase6-dir analysis_outputs/06_diurnal \
  --output-dir analysis_outputs/13_regimes \
  --n-regimes 5 \
  --candidate-k 3,4,5,6,7,8 \
  --pca-variance 0.90 \
  --local-timezone America/Sao_Paulo \
  --overwrite
```

## 16. Resumo textual

```bash
python scripts/13_summarize_meteorological_regimes.py \
  --phase13-dir analysis_outputs/13_regimes \
  --output analysis_outputs/summary_phase13.txt
```

## 17. Outputs

```text
input_schema.parquet

clustering_diagnostics.parquet
pca_explained_variance.parquet
pca_loadings.parquet

regime_assignments.parquet
regime_predictor_profiles.parquet
regime_raw_centroids.parquet

regime_radar_metrics.parquet
regime_event_enrichment.parquet

regime_seasonality.parquet
regime_monthly_prevalence.parquet
regime_hourly_prevalence.parquet
regime_yearly_prevalence.parquet
regime_event_metrics_by_year.parquet

regime_transition_1h.parquet
regime_crosswalk.parquet

analysis_summary.json
phase13.log
```

## 18. Como interpretar

A ordem recomendada é:

```text
1. candidate-k diagnostics
2. estabilidade ARI do k=5
3. PCA
4. perfil meteorológico de cada regime
5. ocorrência radar por regime
6. enriquecimento >=30/40/45
7. composição sazonal
8. absolute × anomaly crosswalk
9. persistência 1h
10. prevalência anual 2021–2024
11. >=45 por regime e ano
```

## 19. O que procurar

A fase deve responder perguntas como:

- existe um regime claramente úmido no qual `>=30 dBZ` é enriquecido?
- existe um regime termicamente distinto associado a `>=45 dBZ`?
- o comportamento de `r500` muda entre regimes?
- `tcwv` continua relevante dentro de estados semelhantes?
- os extremos de JJA pertencem a regimes diferentes dos extremos de DJF?
- a frequência dos regimes muda em 2023–2024?
- o evento `>=45 dBZ` muda dentro do mesmo regime em 2023–2024?
- os regimes são persistentes por algumas horas?

## 20. Limitações

### Radar availability

A análise cobre somente timestamps presentes na Fase 6.

Portanto:

```text
P(regime)
```

é condicionado à disponibilidade radar.

A missingness pode não ser aleatória em relação ao tempo meteorológico.

### KMeans

KMeans pressupõe clusters aproximadamente convexos no espaço PCA.

Estados atmosféricos reais podem:

```text
se sobrepor
formar contínuos
ter geometrias não esféricas
```

Portanto, "regime" aqui é uma representação estatística útil, não uma divisão
ontológica da atmosfera.

### Anomaly climatology

A climatologia mês × hora é calculada usando todo o período porque esta fase é
descritiva.

Isso não deve ser transportado diretamente para um pipeline operacional de
previsão sem refit apenas no treino.

### Causalidade

Nenhuma associação regime-radar estabelece mecanismo causal.

## 21. Relação com a Fase 14

A Fase 13 responde:

> quais estados atmosféricos recorrentes existem e como o radar se comporta
> dentro deles?

A Fase 14 deve responder formalmente:

> a distribuição desses estados, dos predictors ou do target mudou ao longo do
> tempo?

Ela deve investigar especialmente:

```text
2023–2024
mudança de prevalência dos regimes
shift dentro do mesmo regime
shift predictor marginal
shift radar
coverage/missingness
```

## 22. Relação com treinamento CorrDiff

Após a Fase 14, o estudo passa do bloco predominantemente descritivo para o
bloco de preparação/avaliação do treinamento:

```text
Fase 15 -> split formal
Fase 16 -> baselines de treinamento/avaliação
Fase 17 -> ablações de predictors/representações
Fase 18 -> desenho experimental e configuração CorrDiff
```

Ou seja, 15–17 ainda são etapas de engenharia experimental e protocolo de
avaliação; a Fase 18 é a ponte direta para o treinamento CorrDiff.
