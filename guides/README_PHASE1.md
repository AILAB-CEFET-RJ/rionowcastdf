# CorrDiff — Fase 1: caracterização estatística univariada

Esta fase caracteriza, de forma detalhada, as distribuições marginais dos **12 canais ERA5** e do **alvo de radar** do dataset CorrDiff 2011–2024.

Ela foi desenhada para aproveitar a Fase 0 já executada e evitar uma nova varredura integral desnecessária de dezenas de bilhões de valores ERA5. O script seleciona blocos Zarr completos e distribuídos ao longo de todo o período, calcula estatísticas detalhadas sobre essa amostra e compara as médias/desvios amostrais com as estatísticas exatas obtidas na Fase 0.

## Objetivos

A Fase 1 responde principalmente:

- como se distribuem `tcwv`, `t2m`, `u10`, `v10`, `t_850`, `r_850`, `u_850`, `v_850`, `t_500`, `r_500`, `u_500` e `v_500`;
- quais variáveis apresentam maior assimetria e caudas pesadas;
- quão representativa é a amostra usada nas análises visuais;
- qual a frequência de valores de umidade relativa abaixo de 0% ou acima de 100%;
- qual a distribuição do `target` no domínio armazenado (`log1p`);
- qual a distribuição do radar no domínio reconstruído por `expm1(target)`;
- qual a massa em zero do radar;
- quão raros são eventos acima dos níveis numéricos 20, 25, 30, 35, 40, 45 e 50 da legenda;
- quanto do desbalanceamento ocorre no nível de pixel e no nível de patch.

A unidade física da legenda do radar permanece explicitamente **não declarada pelo builder**. Até a documentação da fonte confirmar a unidade, os valores não devem ser rotulados automaticamente como dBZ.

## Arquivos

```text
scripts/
└── 01_compute_univariate_stats.py

notebooks/
└── 01_univariate_analysis.ipynb
```

## Dependências

O ambiente de análise utilizado na Fase 0 já deve conter praticamente tudo:

```bash
pip install \
  numpy \
  pandas \
  pyarrow \
  zarr \
  matplotlib \
  jupyterlab \
  ipykernel \
  tqdm
```

## Execução recomendada no CEFET

A partir da raiz do projeto:

```bash
python scripts/01_compute_univariate_stats.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase0-dir analysis_outputs/00_quality \
  --output-dir analysis_outputs/01_univariate \
  --sample-patches 32768 \
  --reservoir-values 300000 \
  --hist-bins 120 \
  --sampling-strategy systematic \
  --overwrite
```

O `block-size` padrão é obtido automaticamente da primeira dimensão do chunk Zarr. No dataset atual, o builder foi configurado com `chunk_size=256`, portanto a amostragem tende a ler blocos alinhados de 256 patches, reduzindo I/O aleatório.

## Amostragem

A estratégia padrão é:

```text
systematic
```

Como o builder gravou as amostras em ordem temporal, a estratégia sistemática seleciona blocos ao longo de todo o dataset, em vez de concentrar a análise em uma pequena janela temporal.

Com:

```text
sample-patches = 32768
block-size ≈ 256
```

serão lidos aproximadamente:

```text
128 blocos
× 256 patches
≈ 32.768 patches
```

Cada patch possui `32×32` pixels e 12 canais, portanto a análise ainda observa dezenas de milhões de valores por variável, sem precisar reler integralmente os 1.132.464 patches.

## Validação da representatividade

Quando existe:

```text
analysis_outputs/00_quality/channel_summary.parquet
```

o script compara:

```text
média Fase 1 amostrada
vs
média Fase 0 exata
```

em unidades do desvio padrão exato, além de comparar os desvios padrão relativos.

Os campos principais são:

```text
sample_mean_diff_in_phase0_std
sample_std_relative_diff_vs_phase0
```

O notebook marca para revisão diferenças maiores que aproximadamente:

```text
0,10 σ na média
10% no desvio padrão
```

Esses são limiares diagnósticos, não testes estatísticos formais.

## Saídas

```text
analysis_outputs/01_univariate/
├── analysis_summary.json
├── sampling_blocks.parquet
├── input_summary.parquet
├── input_quantiles.parquet
├── input_histograms.parquet
├── relative_humidity_quality.parquet
├── target_summary.parquet
├── target_quantiles.parquet
├── target_histograms.parquet
├── target_threshold_rates_sampled.parquet
├── target_threshold_rates_reference.parquet
├── univariate_samples.npz
└── phase1.log
```

### `input_summary.parquet`

Para cada canal ERA5 contém, entre outros:

```text
sampled_mean
sampled_std
sampled_min
sampled_max
sampled_finite_ratio
sampled_skewness_from_reservoir
sampled_excess_kurtosis_from_reservoir
phase0_exact_mean
phase0_exact_std
sample_mean_diff_in_phase0_std
sample_std_relative_diff_vs_phase0
```

### `input_quantiles.parquet`

Quantis aproximados a partir do reservoir:

```text
P0.1
P1
P5
P10
P25
P50
P75
P90
P95
P99
P99.9
```

### `relative_humidity_quality.parquet`

Para `r_850` e `r_500`:

```text
below_0_ratio
above_100_ratio
outside_0_100_ratio
within_0_100_ratio
```

Os dados **não são corrigidos nem recortados**; a fase apenas mede e documenta o fenômeno.

## Domínios do target

A análise mantém três representações separadas:

```text
target_stored_valid
    = valor do Zarr
    = log1p(valor pós-clip)


target_radar_legend_valid
    = expm1(target_stored_valid)
    = valor numérico pós-clip da legenda


target_radar_legend_positive
    = apenas valores reconstruídos > 0
```

Isso é importante porque a distribuição completa é fortemente dominada por zeros. A distribuição positiva precisa ser estudada separadamente para que a estrutura dos eventos não desapareça no histograma global.

## Taxas por limiar

A Fase 1 calcula na amostra:

```text
> 0
>= 20
>= 25
>= 30
>= 35
>= 40
>= 45
>= 50
```

Quando o arquivo abaixo existe:

```text
analysis_outputs/00_quality/target_event_rates_global.parquet
```

as taxas **exatas da varredura integral da Fase 0** são copiadas para:

```text
target_threshold_rates_reference.parquet
```

Esses valores são preferidos pelo notebook para interpretar o desbalanceamento global.

São usadas duas métricas:

- `event_pixel_ratio`: fração dos pixels válidos que atingem o limiar;
- `event_patch_ratio`: fração dos patches que possuem pelo menos um pixel no limiar.

## Atenção ao overlap dos patches

Com:

```text
patch_size = 32
stride = 16
```

os patches se sobrepõem. Portanto, as distribuições desta fase descrevem a **distribuição efetivamente apresentada ao modelo durante o treinamento**.

Elas não devem ser interpretadas automaticamente como climatologia espacial do campo radar completo, pois pixels centrais podem aparecer em múltiplos patches.

A análise espacial/climatológica sem duplicação será tratada em uma fase posterior.

## Notebook

Depois de executar o script, abra:

```text
notebooks/01_univariate_analysis.ipynb
```

O notebook contém:

1. validação da amostra contra a Fase 0;
2. tabela estatística dos 12 canais;
3. quantis;
4. histogramas dos 12 canais;
5. temperaturas também visualizadas em °C;
6. diagnóstico de umidade relativa fora de `[0,100]`;
7. assimetria e curtose;
8. target no domínio `log1p` e no domínio reconstruído;
9. massa em zero;
10. distribuição somente dos eventos positivos;
11. ECDF dos valores positivos;
12. taxas por limiar no nível de pixel e patch;
13. comparação padronizada das formas das distribuições;
14. checklist automático para encerramento da Fase 1.

## Resultado esperado

Ao encerrar a Fase 1 devemos conhecer:

- escalas e dispersões dos 12 canais;
- variáveis fortemente assimétricas ou de cauda pesada;
- possíveis anomalias físicas em `r_850` e `r_500`;
- grau real de esparsidade do radar;
- distribuição condicional do radar quando existe evento;
- frequência de eventos progressivamente intensos;
- se uma amostra relativamente pequena de blocos é suficiente para representar a distribuição completa observada na Fase 0.

A próxima etapa natural é a **Fase 2 — variáveis derivadas fisicamente**, com magnitude do vento, diferenças verticais de temperatura/umidade e cisalhamento entre 10 m, 850 hPa e 500 hPa.
