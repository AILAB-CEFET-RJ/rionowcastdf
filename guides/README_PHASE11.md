# CorrDiff - Fase 11: Análise Formal de Escalas Espaciais

Versão: `phase11-spatial-scales-v1-haar-coarsegraining`

A Fase 11 formaliza a análise de escalas espaciais após a Fase 10 mostrar que
os campos ERA5 descrevem bem o ambiente do patch, mas têm pouca correspondência
pixel a pixel com os núcleos de radar.

## Pergunta central

> Em quais escalas nominais da grade CorrDiff está concentrada a variabilidade
> espacial de cada predictor e do radar, e em quais escalas as duas fontes
> apresentam associação espacial?

A fase foi construída para não confundir:

```text
grade CorrDiff de 2 km
```

com:

```text
resolução física/nativa efetiva do ERA5
```

Os valores ERA5 foram mapeados para a grade CorrDiff. Isso produz um campo em
2 km, mas não cria observações meteorológicas independentes a cada 2 km.

## Dados

A fase lê diretamente:

```text
datasets/corrdiff_2011_2024/train.zarr
```

com:

```text
input       [N, 12, 32, 32]
target      [N, 1, 32, 32]
timestamps  [N]
```

O target permanece:

```text
log1p(clip(dBZ, min=0))
```

e os eventos são:

```text
>0
>=20
>=30
>=40
>=45 dBZ
```

Os limiares são aplicados no domínio armazenado `log1p float32`, como nas fases
4 v3.1, 9 e 10.

## Preditores

Raw:

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

Derivados:

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

Os `bulk_wind_diff` são diferenças vetoriais de vento em m/s, e não shear
vertical normalizado pela altura.

---

# 1. Decomposição espacial Haar 2D

A principal ferramenta da Fase 11 é uma decomposição Haar 2D ortonormal.

Como os patches têm `32x32`, existem cinco níveis dyádicos:

```text
nível 1 -> suporte nominal de  2 pixels ->  4 km
nível 2 -> suporte nominal de  4 pixels ->  8 km
nível 3 -> suporte nominal de  8 pixels -> 16 km
nível 4 -> suporte nominal de 16 pixels -> 32 km
nível 5 -> suporte nominal de 32 pixels -> 64 km
```

Esses valores são **suportes nominais na grade CorrDiff**.

Eles não devem ser descritos como resolução física do ERA5.

## Detalhes direcionais

A cada nível são produzidos:

```text
EW_DETAIL
NS_DETAIL
DIAGONAL_DETAIL
```

e também um resumo:

```text
ALL_DETAIL
```

que acumula as três orientações.

A normalização é ortonormal, de modo que a soma dos quadrados dos coeficientes
de detalhe pode ser usada como decomposição da energia espacial do campo.

---

# 2. Energia espacial por escala

Para cada predictor e cada campo radar é calculado:

```text
weighted_energy
rms_coefficient
detail_energy_fraction
```

`detail_energy_fraction` é definida usando somente os níveis de detalhe Haar:

```text
energia do nível / soma da energia de todos os níveis
```

Logo, para cada campo:

```text
soma das frações dos níveis 1..5 = 1
```

desconsiderando erros numéricos.

Isso permite comparar, por exemplo:

```text
fração de energia espacial em <= 8 km
fração em 16 km
fração em >= 32 km
```

## Interpretação importante

Se um predictor ERA5 possuir energia não nula em 4 km, isso não significa que
o ERA5 contenha fenômenos meteorológicos realmente resolvidos em 4 km.

Essa energia pode surgir de:

```text
interpolação
gradientes de larga escala
geometria da projeção
combinação não linear de canais
```

A métrica descreve apenas a estrutura espacial **apresentada ao modelo**.

---

# 3. Associação predictor-radar na mesma escala

Os coeficientes Haar de predictor e radar são comparados no mesmo:

```text
nível
orientação
posição dentro do patch
```

A principal saída é:

```text
pearson_r
```

para:

```text
predictor detail × radar detail
```

Isso é mais rigoroso do que correlacionar diretamente os valores finos da
grade interpolada, pois separa explicitamente as escalas dyádicas.

A análise é feita para:

```text
target_log1p
>0
>=20
>=30
>=40
>=45 dBZ
```

e também por estação:

```text
DJF
MAM
JJA
SON
```

---

# 4. Correlação entre energia de escalas diferentes

Para cada patch é calculado o RMS dos coeficientes de detalhe em cada escala.

Depois é avaliada a relação:

```text
energia predictor na escala A
versus
energia radar na escala B
```

Isso gera uma matriz de escalas:

```text
predictor 4 km  -> radar 4/8/16/32/64 km
predictor 8 km  -> radar 4/8/16/32/64 km
...
```

Essa análise não mede co-localização pixel a pixel.

Ela responde se patches que possuem mais estrutura espacial no predictor em
determinada escala também tendem a possuir mais estrutura radar em outra.

---

# 5. Coarse graining

A segunda abordagem usa média em blocos não sobrepostos.

Padrão:

```text
1 pixel  ->  2 km
2 pixels ->  4 km
4 pixels ->  8 km
8 pixels -> 16 km
16 pixels -> 32 km
```

Para cada escala são calculadas associações em dois modos.

## raw

Correlação entre os valores agregados.

Pode conter:

```text
diferenças de regime
sazonalidade
gradientes de patch
nível médio do ambiente
```

## within_patch_centered

Após agregar:

```text
X' = X - mean_patch(X)
Y' = Y - mean_patch(Y)
```

A correlação passa a enfatizar a covariação espacial interna ao patch.

Esse modo é especialmente importante para comparar com a Fase 10.

---

# 6. Amostragem

A PASS A percorre todos os patches para obter estratos exatos:

```text
dry
gt0_lt20
ge20_lt30
ge30_lt40
ge40_lt45
ge45
```

A PASS B usa amostragem estratificada reprodutível.

Padrão:

```text
3000 patches por estrato
seed = 42
```

Estratos menores entram integralmente.

As estatísticas globais usam pesos inversos da fração de amostragem.

---

# 7. Execução

```bash
python scripts/11_compute_spatial_scales.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --output-dir analysis_outputs/11_spatial_scales \
  --sample-per-stratum 3000 \
  --coarse-blocks 1,2,4,8,16 \
  --radar-resolution-km 2 \
  --channel-names tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500 \
  --overwrite
```

Se os nomes dos canais estiverem corretamente disponíveis nos atributos do
Zarr, `--channel-names` pode ser omitido.

---

# 8. Resumo textual

```bash
python scripts/11_summarize_spatial_scales.py \
  --phase11-dir analysis_outputs/11_spatial_scales \
  --output analysis_outputs/summary_phase11.txt
```

---

# 9. Outputs

```text
patch_strata_counts.parquet
sampled_patch_manifest.parquet
input_channel_metadata.parquet

haar_scale_energy.parquet
haar_scale_energy_by_season.parquet

haar_same_scale_associations.parquet
haar_same_scale_associations_by_season.parquet

haar_patch_energy_cross_scale.parquet

coarse_grain_associations.parquet
coarse_grain_associations_by_season.parquet

analysis_summary.json
phase11.log
```

---

# 10. Ordem recomendada de interpretação

A sequência científica recomendada é:

```text
1. energia espacial do radar por escala
2. energia espacial dos predictors por escala
3. associação Haar predictor-radar na mesma escala
4. orientação EW/NS/diagonal
5. coarse graining raw
6. coarse graining within_patch_centered
7. matriz predictor-scale × radar-scale
8. sazonalidade, principalmente >=40/45 dBZ
```

## Perguntas principais

A fase deve permitir responder:

1. O radar concentra mais variabilidade em 4–8 km ou em escalas maiores?
2. Quais predictors ERA5 são espacialmente mais suaves?
3. Quanto da variabilidade dos predictors aparece em escalas <=8 km somente por
   causa da representação interpolada?
4. A associação ERA5-Radar melhora quando os campos são agregados?
5. Existe uma escala onde a associação local centralizada se torna claramente
   mais forte?
6. A orientação da associação reproduz a anisotropia EW > NS observada no radar?
7. A escala dominante muda em eventos >=40/45 dBZ?
8. A distribuição de escalas muda entre DJF e JJA?

---

# 11. Limitações metodológicas

## Escala Haar não é comprimento de onda de Fourier

O `nominal_support_km` significa a largura do suporte dyádico na grade.

Não deve ser chamado de:

```text
comprimento de onda físico
resolução nativa
escala espectral exata
```

sem qualificação.

## 64 km

O nível Haar de 64 km cobre praticamente o patch completo.

Ele é especialmente sensível a:

```text
posição do patch
borda
gradiente de grande escala
```

e deve ser interpretado como contraste na escala do patch.

## Patches sobrepostos

Os patches CorrDiff se sobrepõem.

Assim:

```text
pixels físicos
estruturas meteorológicas
coeficientes espaciais
```

podem aparecer em múltiplos patches.

O `weighted_observation_mass` não deve ser usado como tamanho amostral
independente.

## ERA5 interpolado

A principal regra de interpretação da Fase 11 é:

> estrutura espacial na grade interpolada não equivale a informação
> meteorológica independente nessa mesma escala.

---

# 12. Relação com as próximas fases

A Fase 11 fecha o bloco inicial de análise espacial:

```text
Fase 9  -> estrutura espacial do radar
Fase 10 -> relação espacial ERA5 × Radar
Fase 11 -> decomposição por escalas
```

A sequência prevista continua com:

```text
Fase 12 -> análise multivariada
Fase 13 -> regimes
Fase 14 -> shift
Fase 15 -> splits formais
```

A Fase 12 deve usar as conclusões das Fases 9–11 para evitar colocar em modelos
multivariados variáveis/representações redundantes sem uma justificativa
espacial clara.
