# CorrDiff — Fase 18: Integração Nativa NVIDIA + Smoke Tests

Versão: `phase18-nvidia-corrdiff-native-integration-v1-smoke-only`

A Fase 18 conecta formalmente o contrato experimental das Fases 15–17 ao
**checkout real do NVIDIA PhysicsNeMo CorrDiff**, mas ainda **não executa
treinamento completo** e **não abre 2023/2024**.

A estratégia segue a interface oficial atual do CorrDiff para datasets
customizados:

```text
custom dataset class
    ↓
DownscalingDataset
    ↓
dataset.type = /path/file.py::Class
    ↓
train.py + Hydra
    ↓
regression
    ↓
diffusion
```

A integração evita editar arquivos core da NVIDIA. Ela usa:

```text
custom dataset hook oficial
+
novos configs Hydra
```

---

## 1. O que é validado nesta fase

```text
checkout real PhysicsNeMo/CorrDiff
custom Dataset/DataLoader contract
channel selection C00/C07
derived channels
train-only normalization
Hydra config composition
native regression smoke
native diffusion smoke
locked split protection
```

Não é objetivo da Fase 18:

```text
treinar C00..C10 completamente
selecionar melhor conditioning
abrir 2023
abrir 2024
produzir resultados finais
```

---

# 2. Base NVIDIA utilizada

O código oficial atual organiza CorrDiff em:

```text
examples/weather/corrdiff/

train.py
generate.py
score_samples.py

datasets/
  base.py

conf/
  config_training_custom.yaml
  config_generate_custom.yaml

  base/
    model/
      regression.yaml
      diffusion.yaml

    model_size/
      mini.yaml

    training/
      regression.yaml
      diffusion.yaml
```

O `DownscalingDataset` oficial define os métodos principais:

```text
longitude()
latitude()
input_channels()
output_channels()
time()
image_shape()
__len__()
__getitem__()
```

O CorrDiff oficial suporta dataset customizado declarando:

```yaml
dataset:
  type: /path/to/custom_dataset.py::CustomDataset
```

e executa treinamento em duas etapas:

```text
1. regression model
2. diffusion model usando checkpoint da regressão
```

Esta Fase 18 implementa exatamente esse ponto de integração.

---

# 3. Arquivos do pacote

```text
adapter/
  rionowcast_corrdiff_dataset.py

configs/
  phase18_contract.json

scripts/
  18_inspect_nvidia_corrdiff_checkout.py
  18_prepare_conditioning_stats.py
  18_smoke_dataset_adapter.py
  18_install_nvidia_smoke_configs.py
  18_run_native_smoke.py
  18_summarize_integration.py

notebooks/
  18_nvidia_integration_review.ipynb
```

---

# 4. Adapter customizado

Classe:

```text
RioNowcastCorrDiffDataset
```

herda diretamente de:

```text
datasets.base.DownscalingDataset
```

e retorna:

```python
target, conditioning
```

com shapes:

```text
target:
[1, 32, 32]

C00 conditioning:
[12, 32, 32]

C07 conditioning:
[20, 32, 32]
```

O target permanece:

```text
log1p(clipped radar dBZ)
```

na Fase 18.

Nenhuma transformação do target é adicionada nesta fase.

---

# 5. Proteção explícita dos testes

O adapter só aceita por padrão:

```text
train
validation
```

e recusa:

```text
test_primary
test_stress_ood
```

se:

```text
allow_locked_split = false
```

que é o valor obrigatório da Fase 18.

Além disso, o smoke test verifica que os índices train/validation não possuem
interseção com os arrays de índice de 2023/2024.

---

# 6. Derived channels

O adapter implementa dinamicamente os oito derivados congelados na Fase 17:

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

Eles são calculados **a partir dos raw12 em cada amostra**.

Não são armazenados em uma cópia separada do dataset.

---

# 7. Normalização

A Fase 17 determinou que a normalização deve ser ajustada separadamente para
os canais presentes em cada experimento.

Para evitar ler o Zarr 11 vezes, a Fase 18 calcula uma única vez estatísticas
dos:

```text
12 raw
+
8 derived
=
20 canais
```

usando **somente os 763.044 patches de train**.

Depois emite:

```text
conditioning_stats/
  all20_train_stats.npz
  C00.npz
  C01.npz
  ...
  C10.npz
  conditioning_stats_audit.json
```

Execute:

```bash
python scripts/18_prepare_conditioning_stats.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase17-dir analysis_outputs/17_conditioning_ablations \
  --output-dir analysis_outputs/18_nvidia_integration/conditioning_stats \
  --overwrite
```

---

# 8. Localizar o checkout NVIDIA real

A Fase 18 **não assume** onde o PhysicsNeMo está instalado.

Você deve fornecer o diretório:

```text
examples/weather/corrdiff
```

do checkout real.

Exemplo hipotético:

```text
/home/vsantos/repositories/physicsnemo/examples/weather/corrdiff
```

Execute:

```bash
python scripts/18_inspect_nvidia_corrdiff_checkout.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff
```

O script valida:

```text
train.py
generate.py
score_samples.py
datasets/base.py
custom configs
regression config
diffusion config
mini model config
training configs
```

e registra, quando disponível:

```text
git commit
git branch
git status
```

Output:

```text
analysis_outputs/18_nvidia_integration/
  phase18_nvidia_checkout_audit.json
```

---

# 9. Smoke do Dataset antes da GPU

Antes de tocar no treinamento NVIDIA, valide C00 e C07 diretamente contra a
classe `DownscalingDataset` do checkout real:

```bash
python scripts/18_smoke_dataset_adapter.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff \
  --adapter-path adapter/rionowcast_corrdiff_dataset.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase17-dir analysis_outputs/17_conditioning_ablations \
  --stats-dir analysis_outputs/18_nvidia_integration/conditioning_stats
```

Esse teste exige:

```text
C00 train       [12,32,32]
C00 validation  [12,32,32]

C07 train       [20,32,32]
C07 validation  [20,32,32]

target           [1,32,32]
```

e testa explicitamente que:

```text
test_primary -> refused
test_stress_ood -> refused
```

---

# 10. Instalar somente novos configs Hydra

Após o adapter smoke:

```bash
python scripts/18_install_nvidia_smoke_configs.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff \
  --adapter-path adapter/rionowcast_corrdiff_dataset.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase17-dir analysis_outputs/17_conditioning_ablations \
  --stats-dir analysis_outputs/18_nvidia_integration/conditioning_stats \
  --training-duration 64 \
  --batch-size-per-gpu 2 \
  --overwrite
```

Serão criados no `conf/` do checkout NVIDIA:

```text
config_training_rionowcast_C00_regression_smoke.yaml
config_training_rionowcast_C00_diffusion_smoke.yaml

config_training_rionowcast_C07_regression_smoke.yaml
config_training_rionowcast_C07_diffusion_smoke.yaml
```

Nenhum arquivo core NVIDIA é alterado.

---

# 11. Por que C00 e C07 no smoke?

A Fase 18 precisa validar os dois caminhos estruturalmente distintos:

```text
C00
raw12 only

C07
raw12 + derived8
```

Se ambos passam pelo Dataset, Hydra e modelos nativos, então:

```text
channel subset path
derived-channel path
normalization path
```

estão cobertos.

Não é necessário smoke de C01...C10 individualmente nesta fase.

---

# 12. Hydra composition smoke

Primeiro teste sem GPU/treinamento:

```bash
python scripts/18_run_native_smoke.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff \
  --experiment-id C00 \
  --mode compose \
  --output analysis_outputs/18_nvidia_integration/native_smoke_audit_C00_compose.json
```

Repita para C07:

```bash
python scripts/18_run_native_smoke.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff \
  --experiment-id C07 \
  --mode compose \
  --output analysis_outputs/18_nvidia_integration/native_smoke_audit_C07_compose.json
```

Isso chama o `train.py` **real** da NVIDIA com:

```text
--cfg job
```

portanto Hydra compõe o config, mas treinamento não começa.

---

# 13. Native regression smoke

Com GPU e o ambiente oficial CorrDiff ativado:

```bash
python scripts/18_run_native_smoke.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff \
  --experiment-id C00 \
  --mode regression \
  --output analysis_outputs/18_nvidia_integration/native_smoke_audit_C00_regression.json
```

Config congelado para smoke:

```text
model_size = mini
training_duration = 64 samples
batch_size_per_gpu = 2
total_batch_size = 2
wandb = disabled
```

Isso é apenas:

```text
import
dataset construction
DataLoader
model construction
forward
loss
backward
optimizer
validation/checkpoint path
```

em escala mínima.

Repita C07 depois que C00 estiver funcionando.

---

# 14. Regression checkpoint para smoke da diffusion

O pipeline oficial de diffusion exige um checkpoint de regressão.

Depois do regression smoke, localize:

```bash
find analysis_outputs/18_nvidia_integration/native_smoke \
  -type f \( -name '*.mdlus' -o -name '*.pt' -o -name '*.pth' \) \
  -print
```

Se a versão concreta do CorrDiff não salvar checkpoint com apenas 64 amostras,
não aumente arbitrariamente o treino ainda.

Nesse caso, inspecione o config base real:

```text
conf/base/training/regression.yaml
```

e ajuste **somente a frequência de checkpoint do smoke** na fase 18.

---

# 15. Native diffusion smoke

Com um regression checkpoint compatível:

```bash
python scripts/18_run_native_smoke.py \
  --corrdiff-dir /CAMINHO/physicsnemo/examples/weather/corrdiff \
  --experiment-id C00 \
  --mode diffusion \
  --regression-checkpoint /CAMINHO/REGRESSION_CHECKPOINT.mdlus \
  --output analysis_outputs/18_nvidia_integration/native_smoke_audit_C00_diffusion.json
```

Esse teste confirma:

```text
regression checkpoint load
residual/diffusion model construction
conditioning channel count
forward/loss/backward path
```

sem treinamento completo.

---

# 16. Item de desenho ainda aberto: posição absoluta do patch

Este ponto é propositalmente **não escondido** pela Fase 18.

O dataset possui:

```text
12 patches sobrepostos por timestamp
```

e cada slot representa uma região espacial diferente do domínio.

Entretanto, a interface base oficial expõe:

```text
longitude()
latitude()
```

no nível do dataset, e não por sample.

Na integração inicial o adapter retorna:

```text
relative_patch coordinates
0, 2, 4, ... km
```

apenas para satisfazer metadata/smoke.

Assim:

```text
absolute_patch_position_in_conditioning = False
```

na Fase 18.

Antes do treinamento formal, devemos decidir explicitamente entre:

```text
A. nenhuma posição absoluta
B. slot embedding / one-hot
C. coordenadas absolutas como conditioning auxiliar
D. reconstruir/amostrar domínio completo e usar patching nativo CorrDiff
```

A Fase 16 mostrou que `slot mean` adiciona apenas pequeno ganho climatológico,
mas isso **não é suficiente para concluir** que posição absoluta é irrelevante
para um modelo espacial.

Esse ponto deve ser congelado antes da campanha Stage R completa.

---

# 17. Target normalization

Outro ponto que a Fase 18 mantém deliberadamente simples:

```text
target = stored log1p(clipped dBZ)
normalize_output = identity
```

Isso preserva exatamente o target estudado nas fases anteriores.

Antes do treinamento formal podemos testar/justificar se o regression/diffusion
nativo demanda transformação adicional. Qualquer mudança deverá ser:

```text
fit train-only
congelada antes de 2023/2024
aplicada a todas as ablações
```

---

# 18. Critérios de PASS da Fase 18

A fase só fecha quando:

```text
[ ] checkout NVIDIA audit PASS

[ ] conditioning stats C00..C10 gerados train-only

[ ] C00 Dataset smoke PASS
[ ] C07 Dataset smoke PASS

[ ] 2023 explicitamente recusado
[ ] 2024 explicitamente recusado

[ ] C00 regression Hydra composition PASS
[ ] C00 diffusion Hydra composition PASS

[ ] C00 native regression tiny smoke PASS
[ ] C00 native diffusion tiny smoke PASS

[ ] C07 native regression smoke PASS
[ ] C07 native diffusion smoke PASS

[ ] nenhum treinamento completo realizado
[ ] nenhum test split acessado
```

---

# 19. Resumo

```bash
python scripts/18_summarize_integration.py \
  --phase18-dir analysis_outputs/18_nvidia_integration \
  --output analysis_outputs/summary_phase18.txt
```

---

# 20. Próxima fase

Depois do PASS:

```text
Fase 19
```

passa a executar a campanha formal:

```text
Stage R
C00...C10
seeds 42,43,44
regression model nativo NVIDIA
train 2011–2020
validation 2021–2022
```

Ainda mantendo:

```text
2023 LOCKED
2024 LOCKED
```

O CorrDiff completo só entra na campanha confirmatória após a triagem Stage R.
