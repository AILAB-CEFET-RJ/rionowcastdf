# CorrDiff - Fase 12: Análise Multivariada

Versão: `phase12-multivariate-v1-forward-temporal-weighted`

A Fase 12 inaugura o bloco multivariado do estudo após:

```text
Fase 9  -> estrutura espacial do radar
Fase 10 -> relações espaciais ERA5 × Radar
Fase 11 -> análise formal de escalas espaciais
```

O objetivo agora é medir **quanto sinal conjunto** existe nos 12 canais ERA5 e
nas 8 variáveis derivadas quando observados simultaneamente.

## Princípio metodológico

Esta fase é uma **triagem multivariada temporalmente controlada**.

Ela não substitui:

```text
Fase 15 -> desenho formal de train/validation/test
Fase 16 -> baselines finais
Fase 17 -> ablações formais
```

Por isso:

- não há tuning extensivo de hiperparâmetros;
- são usados modelos simples/regularizados;
- os folds são forward-chaining provisórios;
- a interpretação prioriza estabilidade entre períodos, não apenas a melhor
  métrica média.

---

## 1. Unidade de análise

A unidade é o **patch 32×32**.

Isso foi escolhido porque as Fases 10–11 mostraram que o ERA5 descreve melhor:

```text
ambiente / regime do patch
```

do que:

```text
posição pixel a pixel do núcleo radar
```

Para cada predictor são extraídos:

```text
média espacial do patch
desvio-padrão espacial do patch
```

---

## 2. Representações multivariadas

### `means_only`

20 features:

```text
12 canais raw
+
8 variáveis derivadas
```

usando somente a média de cada predictor no patch.

### `means_plus_std`

40 features:

```text
20 médias
+
20 desvios-padrão dentro do patch
```

A segunda representação testa se a variabilidade espacial interna do ERA5
interpolado acrescenta sinal além do estado médio do ambiente.

Isso **não** é ainda a ablação formal da Fase 17.

---

## 3. Canais raw

Ordem canônica:

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

Recomenda-se informar explicitamente:

```bash
--channel-names tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500
```

---

## 4. Variáveis derivadas

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

Os `bulk_wind_diff` são diferenças vetoriais de vento em m/s.

Eles **não** representam shear normalizado por altura.

---

## 5. Famílias conceituais

Para facilitar a interpretação multivariada, os predictors são classificados em:

### Umidade

```text
tcwv
r_850
r_500
delta_r_500_850
```

### Térmica

```text
t2m
t_850
t_500
delta_t_500_850
delta_t_850_surface
```

### Dinâmica

```text
u/v
wind_speed
bulk_wind_diff
```

Essa classificação serve para interpretação agregada dos coeficientes.

Não implica seleção automática de features.

---

## 6. Targets binários

Em nível de patch:

```text
gt_0
ge_20
ge_30
ge_40
ge_45
```

Um target vale 1 quando existe pelo menos um pixel do patch acima do limiar.

Os thresholds continuam sendo avaliados diretamente no target armazenado:

```text
log1p(clip(dBZ, min=0))
```

evitando o problema de arredondamento identificado na Fase 4 v3.

---

## 7. Targets contínuos

```text
max_target_log1p
positive_pixel_fraction
event_pixel_fraction_ge_30
event_pixel_fraction_ge_40
event_pixel_fraction_ge_45
```

`max_target_log1p` usa diretamente o máximo no domínio armazenado.

As frações continuam descrevendo os patches sobrepostos da distribuição de
treinamento.

Não devem ser chamadas de climatologia de campo radar de-duplicado.

---

## 8. Amostragem

A PASS A percorre todos os patches e obtém os estratos exatos:

```text
dry
gt0_lt20
ge20_lt30
ge30_lt40
ge40_lt45
ge45
```

A PASS B usa, por padrão:

```text
3000 patches por estrato
seed = 42
```

Estratos menores são incluídos integralmente.

Pesos inversos da fração de amostragem são utilizados em:

```text
PCA/correlação
ajuste dos modelos
métricas
climatologia
```

---

## 9. Colinearidade

A fase calcula a matriz de correlação ponderada das **20 médias de predictors**.

Isso é importante porque existem dependências estruturais fortes, por exemplo:

```text
t_500 - t_850
r_500 - r_850
velocidades derivadas de u/v
```

Logo, coeficientes individuais não devem ser interpretados sem considerar
redundância.

---

## 10. PCA

A PCA é feita sobre a matriz de correlação ponderada dos 20 predictors médios.

Outputs:

```text
eigenvalue
explained_variance_ratio
cumulative_explained_variance_ratio
loadings
```

A PCA responde:

> quantas dimensões independentes aproximadamente existem no conjunto de
> predictors?

Ela **não** mede importância supervisionada para o radar.

---

## 11. Folds temporais

A avaliação usa forward-chaining:

```text
F1: treino <= 2014 | avaliação 2015-2016
F2: treino <= 2016 | avaliação 2017-2018
F3: treino <= 2018 | avaliação 2019-2020
F4: treino <= 2020 | avaliação 2021-2022
F5: treino <= 2022 | avaliação 2023-2024
```

Isso evita treinar em dados futuros para avaliar períodos passados.

Todos os patches do mesmo timestamp permanecem naturalmente no mesmo período.

Ainda existe dependência entre timestamps adjacentes do mesmo sistema
meteorológico. A Fase 15 tratará formalmente esse problema.

---

## 12. Referência climatológica

Em cada fold é criada uma climatologia somente a partir do treino:

```text
mês local × hora local
```

Se uma célula não estiver presente no treino, usa-se a média ponderada global
do treino.

Essa climatologia é a referência para:

```text
Brier Skill Score
RMSE skill
MAE skill
```

---

## 13. Modelo linear binário

Para ocorrência:

```text
LogisticRegression L2
C = 1.0
features padronizadas com treino ponderado
```

São avaliadas as duas representações:

```text
means_only
means_plus_std
```

Os coeficientes são armazenados por fold.

Como as features estão padronizadas, a magnitude é comparável dentro de cada
modelo. Porém, devido à colinearidade, predictors correlacionados podem dividir
ou trocar peso entre si.

Por isso a fase também calcula:

```text
média do coeficiente
desvio entre folds
média do |coeficiente|
consistência de sinal
```

---

## 14. Modelo não linear

A fase inclui como triagem:

```text
HistGradientBoostingClassifier
```

somente com:

```text
means_plus_std
```

e hiperparâmetros fixos.

O propósito é responder:

> existe ganho multivariado não linear evidente em relação à logística?

Não há tuning de hiperparâmetros nesta fase.

---

## 15. Modelos contínuos

Para os targets contínuos é utilizado:

```text
Ridge regression
```

com:

```text
alpha = 1.0
```

nas representações:

```text
means_only
means_plus_std
```

A regressão Ridge é usada principalmente como ferramenta de screening
multivariado linear.

---

## 16. Métricas binárias

```text
Brier score
Brier Skill Score vs climatologia
ROC AUC
Average Precision
ECE em 10 bins
```

Para eventos raros como `>=45 dBZ`, priorize:

```text
Brier
Average Precision
calibração
estabilidade entre folds
```

ROC AUC isoladamente pode parecer elevada em problemas muito desbalanceados.

---

## 17. Métricas contínuas

```text
RMSE
MAE
bias
RMSE skill vs climatologia
MAE skill vs climatologia
```

Targets de fração espacial continuam altamente zero-inflated.

Assim, os resultados devem ser lidos junto das análises de ocorrência.

---

## 18. O que procurar nos resultados

A leitura recomendada é:

```text
1. correlações entre predictors
2. PCA e dimensionalidade efetiva
3. logística means_only
4. logística means_plus_std
5. ganho do HistGradientBoosting
6. estabilidade dos coeficientes entre folds
7. famílias moisture / thermal / dynamics
8. Ridge para intensidade/extensão
9. comportamento específico de >=45 dBZ
```

### Evidência multivariada forte

Um predictor é mais interessante quando reúne:

```text
coeficiente não desprezível
+
sinal consistente entre folds
+
coerência com outras fases
```

Um coeficiente grande em apenas um fold não deve ser tratado como robusto.

---

## 19. Execução

```bash
python scripts/12_compute_multivariate_analysis.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/12_multivariate \
  --sample-per-stratum 3000 \
  --channel-names tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500 \
  --overwrite
```

---

## 20. Resumo textual

```bash
python scripts/12_summarize_multivariate_analysis.py \
  --phase12-dir analysis_outputs/12_multivariate \
  --output analysis_outputs/summary_phase12.txt
```

---

## 21. Feature table opcional

Por padrão a fase **não grava** a tabela row-level das features.

Isso reduz:

```text
volume
risco de compartilhamento acidental de dados linha a linha
```

Se necessário localmente:

```bash
--save-feature-table
```

gera:

```text
patch_feature_table.parquet
```

Não é necessária para o resumo textual.

---

## 22. Outputs

```text
patch_strata_counts.parquet
sampled_patch_manifest.parquet
input_channel_metadata.parquet
temporal_fold_definition.parquet

weighted_predictor_correlation.parquet
pca_explained_variance.parquet
pca_loadings.parquet

binary_model_metrics.parquet
continuous_model_metrics.parquet

logistic_coefficients.parquet
ridge_coefficients.parquet
coefficient_stability.parquet

model_status.parquet
analysis_summary.json
phase12.log
```

Opcional:

```text
patch_feature_table.parquet
```

---

## 23. Limitações

- amostra estratificada representa menos de 5% dos patches;
- pesos corrigem a distribuição marginal dos estratos, não tornam observações
  independentes;
- patches se sobrepõem;
- timestamps adjacentes podem pertencer ao mesmo sistema meteorológico;
- folds são provisórios;
- derived variables introduzem colinearidade deliberada;
- coeficientes regularizados não são efeitos causais;
- o HistGradientBoosting é apenas screening;
- esta fase não decide o conjunto final de features.

---

## 24. Relação com a Fase 13

A Fase 12 responde:

> quais relações multivariadas são estáveis no conjunto completo?

A Fase 13 deve responder:

> essas relações mudam entre regimes meteorológicos distintos?

Assim, PCA, estabilidade de coeficientes e comportamento de `>=40/45 dBZ`
servirão como ponte direta para a análise de regimes.
