# CorrDiff - Fase 16: Baselines Formais e Protocolo de Avaliação

Versão: `phase16-baselines-v1.1-zarr-oindex-frozen-split`

## Hotfix v1.1 — compatibilidade com Zarr v2

A versão v1.1 substitui seleções em lote no formato NumPy:

```python
array[idx]
array[idx, 0]
```

por **orthogonal indexing** (`oindex`) através de
`zarr_take_first_axis(...)`.

Isso é necessário em ambientes Zarr v2, onde `__getitem__` com
`numpy.ndarray` é interpretado como basic indexing e gera:

```text
IndexError: unsupported selection item for basic indexing
```

A alteração não muda os índices, splits ou cálculos; apenas a forma de
materializar batches do Zarr.


A Fase 16 é a primeira etapa diretamente orientada a medir o que o CorrDiff
precisa superar no protocolo temporal congelado da Fase 15.

O objetivo não é fazer tuning agressivo. É construir referências com níveis
crescentes de informação e complexidade, congelar métricas e separar claramente:

```text
referências triviais
climatologia
persistência observacional
mapeamento local ERA5→Radar
mapeamento espacial determinístico
baseline probabilístico simples
```

---

## 1. Split utilizado

A Fase 16 herda integralmente a Fase 15:

```text
train            = 2011-2020
validation       = 2021-2022
test_primary     = 2023
test_stress_ood  = 2024
purge            = ±48 h em cada fronteira
```

Durante desenvolvimento:

```text
train + validation
```

são os únicos conjuntos acessíveis para model selection.

Os testes:

```text
2023
2024
```

permanecem bloqueados.

---

## 2. Baselines

### B0 — zero

```text
radar previsto = 0 dBZ
```

É essencial devido à extrema esparsidade pixelwise do radar.

Se um modelo treinado não superar zero em métricas apropriadas, existe um
problema sério de collapse.

---

### B1 — train_global_mean_field

Campo médio do target em `log1p`, calculado somente no treino.

Não usa:

```text
mês
hora
posição/slot
ERA5
```

---

### B2 — train_slot_mean

Cada uma das 12 posições de patch recebe seu campo médio histórico de treino.

Isso controla efeitos geográficos/geométricos associados à posição do patch no
domínio.

---

### B3 — climatology_month_hour_slot

Climatologia:

```text
mês local × hora local × patch slot
```

ajustada somente em `train`.

Fallback:

```text
slot mean
→
global mean field
```

Ela representa uma climatologia espaço-temporal forte, mas sem predictors ERA5
do timestamp.

---

### B4 — persistence_1h_radar_reference

```text
radar(t) ≈ radar(t-1h)
```

apenas quando existe timestamp UTC exato em `t-1h`.

Este baseline é propositalmente chamado de:

```text
radar_reference
```

porque ele usa **radar observado anterior**.

Portanto ele **não é input-equivalente** ao CorrDiff condicionado somente por
ERA5.

Seu papel é responder:

> quanto da dificuldade já poderia ser resolvida se soubéssemos a estrutura
> radar imediatamente anterior?

A comparação deve ocorrer na coorte:

```text
persistence_common
```

e, para comparação rigorosa, os outros baselines também podem ser avaliados
nessa mesma coorte com:

```text
--with-persistence-common
```

---

### B5 — pixel_mlp

Rede composta apenas por convoluções `1×1`.

Ela enxerga os 12 predictors ERA5 do mesmo pixel, mas não possui contexto
espacial.

Esse baseline responde:

> quanto da previsão radar pode ser obtido por um mapeamento local
> ERA5→Radar?

Arquitetura:

```text
12
↓
1x1 Conv
GELU
↓
1x1 Conv
GELU
↓
1x1 Conv
↓
softplus
↓
target log1p
```

Loss:

```text
MSE(log1p target)
```

---

### B6 — unet_deterministic

U-Net pequeno com três níveis de downsampling.

Ele recebe:

```text
[12, 32, 32]
```

e produz:

```text
[1, 32, 32]
```

Loss:

```text
MSE no target armazenado em log1p
```

Seu objetivo é estabelecer um baseline espacial determinístico capaz de usar
contexto dentro do patch.

---

### B7 — unet_gaussian

Mesma família U-Net, mas produz:

```text
mu(x)
log_sigma(x)
```

assumindo:

```text
Y_log1p ~ Normal(mu, sigma)
```

Loss:

```text
heteroscedastic Gaussian NLL
```

É propositalmente simples.

O target real é:

```text
zero-inflated
fortemente não-Gaussiano
multimodal espacialmente
```

Portanto esse modelo não pretende substituir CorrDiff. Ele serve para medir
quanto um baseline probabilístico pixelwise simples consegue fazer.

---

## 3. Normalização

A normalização dos 12 canais é ajustada **somente no treino**:

```text
mean_c
std_c
```

sobre todos os pixels dos patches de treino.

Outputs:

```text
train_input_normalization.npz
```

Essa mesma normalização é aplicada a validation/test.

Não é permitido recalcular estatísticas em 2021-2024 durante o protocolo de
desenvolvimento.

---

## 4. Sampler

Por padrão, os modelos aprendidos usam:

```text
natural training patch distribution
```

sem oversampling por evento.

Isso torna os baselines mais simples e interpretáveis.

A Fase 15 já estabeleceu que qualquer oversampling/class weighting futuro deve
ocorrer somente dentro do treino.

Sampler/loss para eventos raros será uma decisão experimental posterior, não
uma correção do split.

---

## 5. Checkpoint

O checkpoint de cada modelo aprendido é selecionado por:

```text
mínimo validation loss
```

onde:

```text
pixel_mlp          → MSE(log1p)
unet_deterministic → MSE(log1p)
unet_gaussian      → Gaussian NLL(log1p)
```

2023 e 2024 nunca participam dessa seleção.

---

# 6. Métricas congeladas

## 6.1 Métricas pontuais

Nos dois domínios:

```text
stored log1p
reconstructed dBZ = expm1(target)
```

são calculados:

```text
RMSE
MAE
bias
```

O domínio `log1p` é o domínio efetivamente apresentado ao modelo.

O domínio dBZ facilita interpretação física.

---

## 6.2 Métricas categóricas

Thresholds:

```text
20 dBZ
30 dBZ
40 dBZ
45 dBZ
```

Métricas:

```text
Precision
POD / Recall
FAR
CSI
F1
Frequency Bias
ETS
Brier determinístico
```

Essas métricas são fundamentais porque RMSE pixelwise é fortemente influenciado
pela grande massa de zeros.

---

## 6.3 Fractions Skill Score

O FSS é calculado para os mesmos thresholds em suportes nominais:

```text
2 km
4 km
8 km
16 km
```

correspondentes a kernels:

```text
1
2
4
8 pixels
```

na grade CorrDiff de 2 km.

O FSS mede skill espacial tolerante a pequenos deslocamentos e é particularmente
relevante para nowcasting/geração espacial de precipitação.

---

## 6.4 Máximo por patch

Para cada patch:

```text
max target dBZ
max predicted dBZ
```

são comparados.

Métricas:

```text
RMSE
MAE
bias
```

em:

```text
todos os patches
patches target >=30
patches target >=40
patches target >=45
```

Isso reduz o risco de um bom score global esconder falha sistemática nos
extremos.

---

## 6.5 Métricas probabilísticas

Para `unet_gaussian`:

```text
Gaussian NLL no log1p
Gaussian CRPS no log1p
Brier probabilístico >=20/30/40/45
ECE 10 bins
```

Essas métricas estabelecem uma referência probabilística para a futura avaliação
do CorrDiff.

---

## 6.6 Diagnóstico sazonal

O máximo por patch e as taxas de evento são reportados separadamente em:

```text
DJF
MAM
JJA
SON
```

sempre acompanhados do número de patches.

---

# 7. Preparação

Execute primeiro:

```bash
python scripts/16_prepare_baseline_artifacts.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --output-dir analysis_outputs/16_baselines \
  --local-timezone America/Sao_Paulo \
  --expected-patches-per-timestamp 12 \
  --overwrite
```

Outputs principais:

```text
train_input_normalization.npz
train_climatology_month_hour_slot.npz

patch_slot_codes.npy
patch_month_local.npy
patch_hour_local.npy

baseline_preparation.json
metric_protocol.json
phase16_prepare.log
```

---

# 8. Treino dos modelos aprendidos

Configuração inicial recomendada:

```bash
python scripts/16_train_baselines.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase16-dir analysis_outputs/16_baselines \
  --models pixel_mlp,unet_deterministic,unet_gaussian \
  --epochs 8 \
  --batch-size 128 \
  --num-workers 4 \
  --base-channels 32 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --amp \
  --overwrite-checkpoints
```

Se CUDA não estiver disponível:

```text
--device cpu
```

e remova `--amp`.

Outputs:

```text
checkpoints/
  pixel_mlp_best.pt
  unet_deterministic_best.pt
  unet_gaussian_best.pt

training_history.parquet
learned_baseline_training_manifest.json
phase16_train.log
```

---

# 9. Avaliação durante desenvolvimento

Durante a Fase 16, rode primeiro **apenas validation**:

```bash
python scripts/16_evaluate_baselines.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase16-dir analysis_outputs/16_baselines \
  --splits validation \
  --with-persistence-common \
  --amp \
  --overwrite-metrics
```

Isso avalia:

```text
zero
train_global_mean_field
train_slot_mean
climatology_month_hour_slot
persistence_1h_radar_reference
pixel_mlp
unet_deterministic
unet_gaussian
```

---

# 10. Abrindo o teste primário

Somente depois de congelar:

```text
baseline definitions
arquitetura
loss
learning rate
número de épocas
checkpoint rule
metric protocol
```

rode:

```bash
python scripts/16_evaluate_baselines.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase16-dir analysis_outputs/16_baselines \
  --splits test_primary \
  --with-persistence-common \
  --allow-locked-splits
```

---

# 11. Stress/OOD 2024

Depois do primary test:

```bash
python scripts/16_evaluate_baselines.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase16-dir analysis_outputs/16_baselines \
  --splits test_stress_ood \
  --with-persistence-common \
  --allow-locked-splits
```

2024 deve ser reportado separadamente.

Nunca combine:

```text
2023 + 2024
```

como uma única métrica principal.

---

# 12. Resumo

```bash
python scripts/16_summarize_baselines.py \
  --phase16-dir analysis_outputs/16_baselines \
  --output analysis_outputs/summary_phase16.txt
```

---

# 13. Outputs de avaliação

```text
point_metrics.parquet
threshold_metrics.parquet
fss_metrics.parquet
patch_max_metrics.parquet
probabilistic_metrics.parquet
group_patch_metrics.parquet
evaluation_cohorts.parquet
evaluation_manifest.json
```

---

# 14. Como interpretar os baselines

A sequência mede incrementos de informação:

```text
zero
  ↓
mean field
  ↓
patch slot
  ↓
climatologia mês×hora×slot
  ↓
ERA5 local sem contexto espacial
  ↓
ERA5 + contexto espacial determinístico
  ↓
ERA5 + incerteza pixelwise simples
```

A persistência é uma referência lateral:

```text
radar(t-1h)
```

e não deve ser colocada como se utilizasse as mesmas entradas.

---

# 15. O que seria um resultado importante

Alguns cenários serão especialmente informativos.

### U-Net ≈ Pixel MLP

Sugere que o contexto espacial ERA5 interpolado adiciona pouco além do estado
local, coerente com as Fases 10-11.

### U-Net > Pixel MLP em FSS

Sugere que mesmo campos ERA5 suaves contêm gradientes/contexto espacial útil.

### Climatologia próxima do U-Net

Mostra que grande parte do skill provém de sazonalidade/geografia e não dos
predictors instantâneos.

### U-Net melhora RMSE mas falha em CSI/FSS >=40/45

Mostra que um modelo determinístico suaviza extremos, uma justificativa direta
para uma abordagem generativa.

### Gaussian U-Net melhora CRPS/Brier mas continua mal calibrado em >=45

Mostra que incerteza pixelwise Gaussiana é insuficiente para a distribuição
zero-inflated/multimodal do radar.

### Persistência muito superior espacialmente

Reforça a conclusão das Fases 7-8: informação sobre localização da célula
convectiva reside fortemente no estado radar recente, enquanto ERA5 descreve o
ambiente.

---

# 16. Relação com CorrDiff

A Fase 16 congela o piso experimental.

O CorrDiff futuro deverá ser comparado contra esses baselines usando exatamente:

```text
mesmos splits
mesmos patches
mesmos thresholds
mesmos FSS supports
mesma reconstrução dBZ
mesmos grupos sazonais
```

Sem alterar a métrica depois de observar o test.

---

# 17. Próxima fase

A Fase 17 deve ser a **ablação controlada de condicionamento**:

```text
12 raw
raw + derived
moisture only
thermal only
dynamics only
combinações
event-aware sampling/loss se decidido
```

sempre usando apenas train/validation para seleção.

A Fase 18 então fecha a configuração final do CorrDiff.
