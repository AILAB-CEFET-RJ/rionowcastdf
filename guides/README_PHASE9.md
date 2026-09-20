# CorrDiff - Fase 9: Estrutura Espacial do Radar

Versão: `phase9-radar-spatial-structure-v1-patch-distribution`

A Fase 9 inicia o bloco espacial da análise do dataset CorrDiff.

## Objetivo

Caracterizar:

- escala espacial de dependência do target radar;
- anisotropia direcional;
- fragmentação e conectividade de áreas de eco;
- comportamento morfológico de eventos fortes;
- impacto das bordas dos patches sobre a interpretação.

A análise é feita **dentro dos patches 32x32** armazenados em `train.zarr`.

Ela não tenta reconstruir o campo completo 70x84.

## Por que não reconstruir o campo completo nesta fase?

Os patches do CorrDiff são sobrepostos. A ordem/posição geométrica exata dos
patches não precisa ser presumida para responder às perguntas locais da Fase 9.

Isso evita introduzir uma hipótese espacial não documentada sobre a montagem do
mosaico.

A Fase 10 poderá usar a informação espacial estabelecida aqui para estudar
relações ERA5 × Radar.

## Target

Unidade física: **dBZ**.

O builder armazena:

```text
target = log1p(clip(dBZ, min=0))
```

Os limiares `20`, `30`, `40` e `45 dBZ` são comparados diretamente no domínio
`log1p float32`, seguindo a correção usada a partir da Fase 4 v3.1.

## PASS A: estratos exatos

Todos os patches são lidos uma vez e classificados pelo máximo do patch:

```text
dry
gt0_lt20
ge20_lt30
ge30_lt40
ge40_lt45
ge45
```

Isso fornece as taxas exatas de patches contendo eventos.

## Amostragem espacial

A parte mais cara da análise usa amostragem estratificada reprodutível.

Padrão:

```text
5000 patches por estrato
seed = 42
```

Estratos com menos de 5000 patches são incluídos integralmente.

Os resumos globais usam peso inverso da fração de amostragem de cada estrato.

## Correlação espacial

São calculados deslocamentos:

```text
1, 2, 4, 8 e 16 pixels
```

Com resolução padrão de `2 km/pixel`, isso corresponde a escalas ortogonais de:

```text
2, 4, 8, 16 e 32 km
```

Direções:

```text
EW
NS
NW_SE
NE_SW
```

Para diagonais, a distância física é multiplicada por `sqrt(2)`.

Para cada deslocamento são calculados:

```text
Pearson
semivariância
mismatch probability para máscaras binárias
```

Campos analisados:

```text
target_log1p
>0 dBZ
>=20 dBZ
>=30 dBZ
>=40 dBZ
>=45 dBZ
```

## Interpretação do semivariograma

Para o target contínuo:

```text
gamma(h) = 0.5 * E[(Z(x) - Z(x+h))^2]
```

O cálculo é realizado no target `log1p`.

Para máscaras binárias, a semivariância corresponde à metade da probabilidade
de os dois pixels terem estados diferentes.

## Morfologia

Para cada limiar, em cada patch amostrado:

```text
event_pixels
event_fraction
component_count
largest_component_pixels
largest_component_fraction_of_event
internal_edge_transition_density
touches_border
largest_component_touches_border
largest_component_bbox_fraction
largest_component_elongation
```

A conectividade utilizada é **4-neighbor**.

## Bordas dos patches

Um componente que toca a borda pode continuar fora do patch.

Por isso:

```text
component_count
largest_component_pixels
elongation
```

devem ser tratados como diagnósticos da distribuição de treinamento, e não como
morfologia completa de uma célula meteorológica.

A própria Fase 9 calcula a taxa de eventos que tocam a borda para quantificar
essa limitação.

## Sazonalidade

A morfologia também é resumida por:

```text
DJF
MAM
JJA
SON
```

A conversão temporal usa `America/Sao_Paulo`, preservando o histórico de horário
de verão.

## Execução

```bash
python scripts/09_compute_spatial_structure.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/09_spatial_structure \
  --sample-per-stratum 5000 \
  --spatial-steps 1,2,4,8,16 \
  --radar-resolution-km 2 \
  --overwrite
```

## Resumo textual

```bash
python scripts/09_summarize_spatial_structure.py \
  --phase9-dir analysis_outputs/09_spatial_structure \
  --output analysis_outputs/summary_phase9.txt
```

## Outputs

```text
patch_strata_counts.parquet
sampled_patch_manifest.parquet
spatial_lag_statistics.parquet
patch_morphology_sample.parquet
morphology_summary.parquet
morphology_by_season.parquet
analysis_summary.json
phase9.log
```

## Perguntas científicas principais

A Fase 9 deve permitir responder:

1. Em que distância a correlação espacial do radar cai substancialmente?
2. Essa escala muda com o limiar de refletividade?
3. Existe anisotropia EW × NS ou diagonal?
4. Eventos fortes são mais compactos ou mais fragmentados?
5. A morfologia muda entre estações?
6. Quanto das estruturas está sendo cortado pelas bordas dos patches?

## Limitação central

Os patches são sobrepostos.

Assim, pares físicos podem ser contados várias vezes em patches diferentes.
Os resultados representam a **distribuição espacial apresentada ao modelo**,
não uma climatologia espacial de campo completo.

Essa distinção deve permanecer explícita no texto da dissertação.

## Próximo passo

Após a Fase 9:

```text
Fase 10 -> estrutura espacial ERA5 × Radar
Fase 11 -> escalas espaciais
```

Os resultados da Fase 9 servirão para escolher escalas e vizinhanças relevantes
para essas etapas.
