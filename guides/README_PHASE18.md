# Fase 18 — integração nativa NVIDIA CorrDiff (v2)

Pacote recriado em 17/10/2026 após indisponibilidade do download anterior.
Esta é uma reconstrução das alterações descritas na conversa, novamente testada;
não é uma cópia byte a byte do patch anterior.

Base RioNowcastDF: `cd2f81719df80a08726c2840d14ec7f1d83c3579`.
NVIDIA: `beafeecbd01fcac3c76bc6f3391a6af1872ea6bf`, exemplo
`examples/weather/corrdiff`. O exemplo está marcado deprecated nesse checkout;
mantemos o código solicitado, fixado nesse commit.

## Objetivo e limites

Integrar o dataset ao código real, validar Dataset/DataLoader/Hydra, executar
smokes mínimos de regressão/difusão e gerar um patch de validation após recarga
dos checkpoints. A campanha completa fica para as fases 19/20. 2023/2024 permanecem
bloqueados. Nenhum treino GPU com o Zarr real foi realizado nesta reconstrução.

## O que o pacote implementa

- Adapter `adapter/rionowcast_corrdiff_dataset.py`, ausente na base do projeto.
- Interface real `DownscalingDataset`; retorno `(target, conditioning)` em tensors.
- C00 com 12 canais, C07 com 20; todas as seleções C00..C10 seguem a fase 17.
- Derivados calculados antes da normalização, em unidades físicas.
- Estatísticas exclusivamente train, com acumulação estável e hash do manifesto.
- YAML corrigido: `arquivo.py::Classe` como string completa; campos custom
  obrigatórios preenchidos; `_self_` explícito.
- Composição Hydra de regression e diffusion para C00/C07.
- Saídas isoladas, frequências de checkpoint/validation para treino de 64 amostras.
- Ponte explícita de seeds para inicialização e sampler, sem editar arquivos NVIDIA.
- Verificação de todas as máscaras train/validation e recusa dos testes bloqueados.
- Relatório que exige evidências de todos os estágios antes de emitir PASS.

## Contrato preservado

| Item | Regra |
|---|---|
| Geometria | 32 × 32; sem resize nem novo patching |
| Target | log1p(clipped radar dBZ); normalização de saída identidade |
| Treino | 2011–2020 |
| Validation | 2021–2022 |
| Testes | 2023/2024 recusados inclusive com override |
| Splits | Manifestos fase 15, purge e 12 patches por timestamp |
| Timestamp | Unix seconds, conforme o builder |
| Normalização | Ajustada no train e reutilizada na validation |
| Máscara | Todos os pixels devem ser válidos para usar a loss nativa |

A auditoria pode ler índices/calendário dos testes para confirmar separação;
não lê input/target desses testes. O Zarr deve permanecer imutável após a
validação: o fingerprint cobre código, manifestos e estatísticas, não todos
os chunks do dataset.

**Máscaras:** a loss nativa não recebe uma máscara. Se houver pixel inválido em
train/validation, o smoke para. Não convertemos inválido em ausência de eco,
não descartamos patches e não mudamos a loss silenciosamente. A ocorrência
exige implementar uma loss mascarada antes de concluir a integração.

**Posição:** sem condicionamento geográfico absoluto adicional neste contrato.
Embeddings locais NVIDIA continuam ativos. Como os samples representam 12 slots,
uma única grade de latitude/longitude não é válida para todo o dataset; os métodos
retornam NaN. `sample_metadata()` preserva índice, timestamp, linha e coluna.
O smoke de geração usa índice explícito, evitando a ambiguidade de seleção
somente por timestamp no `generate.py` oficial.

**Seeds:** o treino NVIDIA chama `set_seed(rank + cur_nimg)` e usa sampler seed 0.
O launcher acrescenta a seed experimental ao primeiro ponto e passa a mesma seed
ao sampler, registrando evidência em `*.seeds.json`. O smoke usa uma GPU/processo,
saída nova e seed 42 por padrão; suporta 42/43/44. DDP não é coberto nesta fase.

## Preparação do ambiente

Na raiz do RioNowcastDF, use o ambiente CUDA correspondente ao PhysicsNeMo.
O Zarr e os outputs das fases 15/17 são necessários e não fazem parte deste ZIP.

```bash
# Se não existir um checkout NVIDIA, crie um em pasta separada:
git clone https://github.com/NVIDIA/physicsnemo.git ../physicsnemo
git -C ../physicsnemo checkout beafeecbd01fcac3c76bc6f3391a6af1872ea6bf
export CORRDIFF_DIR="$(realpath ../physicsnemo/examples/weather/corrdiff)"
```

Para checkout já existente, preserve alterações locais; prefira outro checkout
nesse commit. Instale PhysicsNeMo correspondente e as dependências de
`examples/weather/corrdiff/requirements.txt`. Preserve seu PyTorch CUDA.

```bash
python -m pip install -r requirements_phase18.txt
python scripts/18_inspect_nvidia_corrdiff_checkout.py --corrdiff-dir "$CORRDIFF_DIR"
```

O inspector recusa outro commit ou arquivos versionados NVIDIA modificados.

## 1. Normalização train-only

```bash
python scripts/18_prepare_conditioning_stats.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase15-dir analysis_outputs/15_formal_splits \
  --phase17-dir analysis_outputs/17_conditioning_ablations \
  --output-dir analysis_outputs/18_nvidia_integration/conditioning_stats
```

O script lê train uma vez para os 20 canais e emite estatísticas C00..C10.
Default: chunks de 128 patches. Use `--overwrite` para recalcular estatísticas
existentes. Estatísticas v1 devem ser recriadas, pois não têm a proveniência v2.

## 2. Dataset e máscaras

```bash
python scripts/18_smoke_dataset_adapter.py \
  --corrdiff-dir "$CORRDIFF_DIR" \
  --adapter-path adapter/rionowcast_corrdiff_dataset.py
```

Valida C00/C07 em train/validation, batches, finitude, shapes, todas as máscaras
e recusa dos testes. Registra fingerprint de código/manifestos/estatísticas;
alterações exigem nova auditoria antes do treino.

## 3. Instalação e composição Hydra

```bash
python scripts/18_install_nvidia_smoke_configs.py \
  --corrdiff-dir "$CORRDIFF_DIR" \
  --adapter-path adapter/rionowcast_corrdiff_dataset.py \
  --training-duration 64 --batch-size-per-gpu 2

for EXP in C00 C07; do
  python scripts/18_run_native_smoke.py \
    --corrdiff-dir "$CORRDIFF_DIR" --experiment-id "$EXP" --mode compose
done
```

Cada composição verifica os dois estágios com a API Hydra e configs NVIDIA reais.
Não importa os modelos nem valida CUDA/checkpoints; usa placeholder de checkpoint
apenas para compor diffusion. Configs smoke: mini, fp32, 0 workers, batch 2,
64 amostras e uma etapa validation. Limite de orçamento: 256 amostras.
Esses parâmetros não são uma seleção de hiperparâmetros para a campanha formal.

## 4. Regressão, difusão e geração

Comece por C00. Depois repita os comandos com `EXP=C07`.

```bash
EXP=C00
python scripts/18_run_native_smoke.py \
  --corrdiff-dir "$CORRDIFF_DIR" --experiment-id "$EXP" \
  --mode regression --seed 42

REG_CKPT=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS"; assert len(d["checkpoints"])==1; print(d["checkpoints"][0])' \
  "analysis_outputs/18_nvidia_integration/native_smoke_audit_${EXP}_regression.json")

python scripts/18_run_native_smoke.py \
  --corrdiff-dir "$CORRDIFF_DIR" --experiment-id "$EXP" \
  --mode diffusion --seed 42 --regression-checkpoint "$REG_CKPT"

python scripts/18_smoke_generation.py \
  --corrdiff-dir "$CORRDIFF_DIR" --experiment-id "$EXP" \
  --regression-audit "analysis_outputs/18_nvidia_integration/native_smoke_audit_${EXP}_regression.json" \
  --diffusion-audit "analysis_outputs/18_nvidia_integration/native_smoke_audit_${EXP}_diffusion.json"
```

O runner exige saída bem-sucedida, checkpoint, perdas finitas de treino/validation
e seeds verificadas. O gerador usa `regression_step`, `diffusion_step` e
`deterministic_sampler` nativos: um patch, quatro passos Heun, seed latente 42.
Salva regressão, residual, previsão log1p e expm1 sem clipping silencioso, e
reporta a fração negativa. Resultados de smoke não são evidência de skill.

Para repetir treino, use novo `--smoke-output-root` no instalador e `--overwrite`
para atualizar os configs. O runner recusa saída não vazia para evitar retomada.

## 5. Fechamento e testes

```bash
python scripts/18_summarize_integration.py
```

Enquanto faltar evidência, retorna BLOCKED e código 2. Exige checkout,
estatísticas, adapter/máscaras, bloqueio de testes, composição, regressão,
difusão e geração de C00/C07. Não edite os audits para obter PASS.
Após PASS: fase 19 com C00 × seeds 42/43/44, depois 33 runs Stage R;
fase 20 com 15 runs obrigatórios Stage D.

Para reproduzir a validação CPU com dados sintéticos:

```bash
CORRDIFF_DIR="$CORRDIFF_DIR" python -m pytest tests/test_phase18_integration.py -q
```

Use checkout NVIDIA de teste sem os quatro configs RioNowcast já instalados:
os testes criam/removem esses configs temporários. Os 14 testes foram novamente
executados nesta reconstrução. Não substituem execução GPU com dados reais.
