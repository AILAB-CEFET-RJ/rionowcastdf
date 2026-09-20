# CorrDiff - Fase 10: Relações Espaciais ERA5 × Radar

Versão: `phase10-spatial-era5-radar-v1-aligned-centered-offset`

A Fase 10 conecta o diagnóstico espacial do radar da Fase 9 aos campos ERA5
que entram no CorrDiff.

## Pergunta central

> Os padrões ERA5 presentes no patch estão espacialmente associados às regiões
> de refletividade do radar, e essa associação persiste quando removemos o
> background médio do próprio patch?

Essa distinção é importante porque uma correlação global pode refletir apenas:

- estação do ano;
- hora do dia;
- regime sinótico;
- diferença média entre patches.

A Fase 10 adiciona uma associação `within_patch_centered`, em que predictor e
radar são centralizados separadamente dentro de cada patch antes da correlação.

## Dados

A fase lê diretamente:

```text
datasets/corrdiff_2011_2024/train.zarr

input       [N, 12, 32, 32]
target      [N, 1, 32, 32]
timestamps  [N]
```

Não é necessário reabrir ERA5 ou os PNGs originais.

## Preditores raw

Ordem canônica esperada:

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

O script tenta primeiro ler nomes em atributos do Zarr. Se não encontrar e o
input possuir 12 canais, usa essa ordem canônica como fallback.

Para eliminar qualquer ambiguidade, é possível executar com:

```bash
--channel-names tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500
```

## Preditores derivados

Os mesmos oito usados desde a Fase 2:

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

Os `bulk_wind_diff` são diferenças vetoriais de vento em m/s. Não são shear
normalizado pela altura.

## Radar

O target permanece:

```text
log1p(clip(dBZ, min=0))
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

Os thresholds são aplicados diretamente no domínio `log1p float32`, como na
Fase 4 v3.1.

## PASS A

Todos os 1.016.208 patches são varridos para classificação exata em:

```text
dry
gt0_lt20
ge20_lt30
ge30_lt40
ge40_lt45
ge45
```

## Amostra da PASS B

Padrão:

```text
3000 patches por estrato
seed = 42
```

Estratos menores são incluídos integralmente.

Pesos inversos da fração de amostragem de cada estrato são usados nos resumos
globais.

## 1. Associação pixel a pixel bruta

Calcula Pearson entre cada predictor e cada campo radar no mesmo pixel da grade
CorrDiff.

Essa associação ainda pode conter diferenças entre regimes e patches.

## 2. Associação centralizada dentro do patch

Para cada patch:

```text
X' = X - mean_patch(X)
Y' = Y - mean_patch(Y)
```

e depois é calculada a correlação global ponderada entre `X'` e `Y'`.

Isso enfatiza co-localização espacial **dentro do mesmo patch** e reduz a
contribuição de diferenças médias entre patches.

Não transforma a relação em causal.

## 3. Contexto local por suavização

Janelas padrão:

```text
1, 3, 5, 9, 17 pixels
```

Com 2 km/pixel, os raios geométricos são aproximadamente:

```text
0, 2, 4, 8, 16 km
```

Somente pixels cuja janela completa cabe dentro do patch são avaliados. Assim,
não há contribuição de padding nas estatísticas finais.

Esta parte é uma triagem de contexto local. A decomposição formal de escalas é
reservada para a **Fase 11**.

## 4. Contraste evento × background no mesmo patch

Para cada predictor e threshold:

```text
delta = mean(X | pixels do evento)
      - mean(X | pixels fora do evento, mesmo patch)
```

Também é calculado:

```text
delta / std_patch(X)
```

Essa métrica é particularmente útil porque compara regiões dentro do mesmo
ambiente de patch.

## 5. Cross-offset

A Fase 9 mostrou anisotropia espacial do radar. Por isso a Fase 10 calcula:

```text
predictor(x) × radar(x + delta)
```

tanto no modo bruto quanto após centralização dentro do patch, nos deslocamentos padrão:

```text
2, 4 e 8 pixels
```

ou:

```text
4, 8 e 16 km
```

nas direções N, S, E e W, além de `CENTER`.

Um máximo fora do centro **não deve ser interpretado automaticamente como
advecção ou deslocamento causal**. Interpolação, geometria dos patches,
organização espacial e circulação podem contribuir.

## 6. Associação em nível de patch

A média espacial de cada predictor é comparada a:

```text
max_dbz
positive_pixel_fraction
event_pixel_fraction_ge_30
event_pixel_fraction_ge_40
event_pixel_fraction_ge_45
```

Essa tabela conecta a análise espacial local aos resultados agregados das Fases
3–8.

## Sazonalidade

A associação pixelwise centralizada e os contrastes evento-background também
são calculados por:

```text
DJF
MAM
JJA
SON
```

## Execução

```bash
python scripts/10_compute_spatial_era5_radar.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/10_spatial_era5_radar \
  --sample-per-stratum 3000 \
  --smoothing-windows 1,3,5,9,17 \
  --offset-steps 2,4,8 \
  --radar-resolution-km 2 \
  --overwrite
```

Para fixar explicitamente a ordem dos canais:

```bash
python scripts/10_compute_spatial_era5_radar.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/10_spatial_era5_radar \
  --channel-names tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500 \
  --overwrite
```

## Resumo textual

```bash
python scripts/10_summarize_spatial_era5_radar.py \
  --phase10-dir analysis_outputs/10_spatial_era5_radar \
  --output analysis_outputs/summary_phase10.txt
```

## Outputs

```text
patch_strata_counts.parquet
sampled_patch_manifest.parquet
input_channel_metadata.parquet
pixel_spatial_associations.parquet
pixel_spatial_associations_by_season.parquet
spatial_cross_offset_associations.parquet
within_patch_event_contrasts.parquet
within_patch_event_contrasts_by_season.parquet
patch_level_associations.parquet
analysis_summary.json
phase10.log
```

## Interpretação recomendada

A ordem de leitura científica deve ser:

```text
1. associação raw
2. associação within_patch_centered
3. contraste evento-background
4. comportamento sazonal
5. cross-offset
6. efeito preliminar das janelas
```

Se uma associação for forte apenas no modo raw, mas desaparecer após
centralização, ela provavelmente representa background/regime mais do que
co-localização espacial local.

Se persistir no modo centralizado e no contraste evento-background, existe
evidência descritiva mais forte de alinhamento espacial entre predictor e radar.

## Limitações

- patches sobrepostos duplicam pixels físicos;
- `weighted_observation_mass` não é tamanho amostral independente;
- ERA5 foi mapeado para a grade CorrDiff;
- valores interpolados em 2 km não criam informação meteorológica independente
  em 2 km;
- nenhum resultado desta fase é causal;
- o target continua com valores negativos de dBZ clipados para zero pelo
  builder.

## Relação com a Fase 11

A Fase 10 responde **onde e em que direção** existem associações espaciais.

A Fase 11 deverá formalizar **em quais escalas espaciais** a informação é
carregada, evitando confundir interpolação com resolução meteorológica efetiva.
