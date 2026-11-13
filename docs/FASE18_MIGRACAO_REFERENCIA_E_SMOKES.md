# Fase 18 — migração segura à referência Zarr e smoke de modelos PhysicsNeMo

**Base de revisão:** `AILAB-CEFET-RJ/rionowcastdf`, branch `chore/reorganizar-estrutura`, commit `e534726b7129feb8e5252b13d4fc53def657ca91` (12/11/2026).  
**Ambiente aprovado:** `~/corrdiff_reference_reproduced_cu129_v4` (bundle `reference_cu129_v6`, Doctor 11/11 PASS).  
**Status desta entrega:** implementação nova para inspeção e testes; os testes unitários CPU foram executados localmente, **não** os smoke CUDA com Zarr real no CEFET.

## Por que a integração antiga não pode ser usada diretamente

O fluxo anterior (`18_prepare_conditioning_stats.py`, `18_smoke_dataset_adapter.py`, `18_run_native_smoke.py`, configs geradas pelo instalador antigo) presume o dataset `corrdiff_2011_2024`, 12 patches/hora, índices de split de uma estrutura Zarr única e alvo `log1p`. A referência auditada é `datasets/corrdiff_reference_v1`, quatro Zarrs por split, 48 patches/hora, alvo **`target_code/50`**, máscara booleana e normalização de ERA5 ajustada exclusivamente em treino. Além disso, o checkout PhysicsNeMo aprovado é `c15bd23eb56216c32bd3e30180593e5685be9bc2`, diferente do commit registrado no contrato legado `configs/phase18_contract.json`.

**Não modificar `configs/phase18_contract.json` isoladamente nem executar a implementação legada com a referência nova.** A entrega introduz `configs/phase18_reference_contract.json` com identidade explícita e uma cadeia de scripts com nomes distintos. Nenhum arquivo legado, dataset congelado, contrato do builder ou checkout PhysicsNeMo é sobrescrito.

O `train.py` original utiliza perdas sem máscara. Portanto, passar `(target, input)` na API nativa antiga com pixels inválidos seria metodologicamente incorreto. A execução nova usa **as classes reais** `CorrDiffRegressionUNet` e `EDMPrecondSuperResolution` do PhysicsNeMo e os módulos `adapter/reference_masked_losses.py`/`adapter/reference_masked_flow.py` já aprovados pelo Doctor. Isso é **smoke de modelos PhysicsNeMo com treinador mascarado do projeto**, não smoke do entrypoint original `train.py`. Para fechar a Fase 18 com exigência de execução do orquestrador nativo, ainda é necessário projetar/validar uma integração compatível com máscara ou justificar cientificamente o treinador próprio.

## Arquivos desta entrega

| Arquivo | Finalidade |
|---|---|
| `configs/phase18_reference_contract.json` | Contrato separado; code/50, 48 patches/h, splits bloqueados, commit PhysicsNeMo |
| `adapter/reference_conditioning.py` | Derivados ERA5 a partir dos campos **físicos**, não dos canais padronizados |
| `adapter/reference_derived_dataset.py` | Dataset C07: 20 canais, raw12 com normalização oficial, 8 derivados com stats train-only, alvo/máscara intactos |
| `scripts/18_fit_reference_derived_stats.py` | Ajuste train-only sobre **pontos espaciais únicos por hora**, sem viés de sobreposição |
| `scripts/18_reference_preflight.py` | Verificação somente de treino/validação e travas de splits de teste |
| `scripts/18_reference_model_smoke.py` | Forward, masked loss, backward, atualização, validação, checkpoint técnico em GPU |
| `tests/test_phase18_reference_migration.py` | Testes unitários CPU para contratos, fórmulas, integridade e travas |

### Política de normalização C07

O **C00** usa `ReferenceCorrDiffDataset`, leitor de 12 canais que utiliza as médias e desvios congelados no `normalization.json` da referência. **C07** usa esse mesmo ajuste dos canais brutos, acrescentando oito derivadas computadas em espaço físico a partir de `raw12` **antes** da padronização. As médias e os desvios das derivadas são estimados apenas nas horas de `train` (cada ponto da grade contado uma vez, sem multiplicidade artificial por patches). O estimador lê `train.zarr` por meio de `ReferenceReader.hour()` e não toca os testes.

A opção padrão `--max-hours 96` produz estatísticas **amostrais exclusivas de smoke**. O NPZ registra `scope=smoke_sampled_train_hours` e **não pode** ser usado em treinamento completo: o leitor recusa essas estatísticas, a menos que `allow_sampled_stats=True` seja habilitado explicitamente no script de smoke. A versão definitiva do C07 exige `--all-train-hours`, que terá custo alto de I/O e deve ser realizada **antes da Fase 19**.

## Preparação no CEFET

1. Transfira o ZIP para a máquina; aplique **somente no checkout correto**:

```bash
cd "$HOME/repositories/rionowcastdf"
git branch --show-current
git rev-parse HEAD
git status --short

unzip -n /CAMINHO/phase18_reference_migration_v1.zip -d .
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
"$PYTHON" -m unittest discover -s tests -p 'test_phase18_reference_migration.py' -v
```

2. Verifique as entradas sem reconstruir o dataset:

```bash
DATA="$PWD/datasets/corrdiff_reference_v1"
ls -lh "$DATA/summary.json" "$DATA/normalization.json" "$DATA/contract.json"
"$PYTHON" -m pip check
```

## Etapa A — ajuste dos canais derivados (somente treino)

```bash
cd "$HOME/repositories/rionowcastdf"
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
OUT="$PWD/analysis_outputs_reference/phase18_smoke_v1"
mkdir -p "$OUT"

nohup "$PYTHON" -u scripts/18_fit_reference_derived_stats.py \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --output "$OUT/derived96_train.npz" \
  --max-hours 96 \
  > "$OUT/derived96_train.log" 2>&1 < /dev/null &
echo "PID: $!"
```

Acompanhar: `tail -f "$OUT/derived96_train.log"`. Resultado: `derived96_train.npz` e `derived96_train.json`. **Não usar no C07 completo.**

## Etapa B — preflight de dados C00/C07

```bash
"$PYTHON" -u scripts/18_reference_preflight.py \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --stats-file "$OUT/derived96_train.npz" \
  --output "$OUT/data_preflight.json"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('analysis_outputs_reference/phase18_smoke_v1/data_preflight.json').read_text())
print('DATA PREFLIGHT:',r['status'],'| locked accessed:',r['locked_splits_accessed'])
for key,value in r['datasets'].items():print(key,value['length'],value['samples_checked'])
PY
```

Parar diante de qualquer falha. **Apenas `train`/`validation` podem ser abertas**.

## Etapa C — smoke de regressão C00

O modelo é de tamanho reduzido (`model_channels=32`, batch 1, FP32), com no máximo oito passos, para respeitar a VRAM de cada RTX 2080 Ti. Usamos uma GPU por execução e **não** iniciamos treinamento científico.

```bash
nohup "$PYTHON" -u scripts/18_reference_model_smoke.py \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --experiment C00 --stage regression --steps 2 --gpu 0 \
  --output-dir "$OUT/C00_regression" \
  > "$OUT/C00_regression.log" 2>&1 < /dev/null &
```

Acompanhar: `tail -f "$OUT/C00_regression.log"` e `nvidia-smi`. Para difusão, **só após `C00_regression/audit.json` retornar PASS**:

```bash
nohup "$PYTHON" -u scripts/18_reference_model_smoke.py \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --experiment C00 --stage diffusion --steps 2 --gpu 0 \
  --regression-checkpoint "$OUT/C00_regression/technical_state_dict.pt" \
  --output-dir "$OUT/C00_diffusion" \
  > "$OUT/C00_diffusion.log" 2>&1 < /dev/null &
```

O checkpoint `technical_state_dict.pt` **não** tem o formato `.mdlus` do PhysicsNeMo. Ele se destina somente a testar a continuidade entre regressão e difusão no treinador mascarado.

## Etapa D — smoke C07 (20 canais)

```bash
nohup "$PYTHON" -u scripts/18_reference_model_smoke.py \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --experiment C07 --stage regression --steps 2 --gpu 0 \
  --stats-file "$OUT/derived96_train.npz" \
  --output-dir "$OUT/C07_regression" \
  > "$OUT/C07_regression.log" 2>&1 < /dev/null &
```

```bash
nohup "$PYTHON" -u scripts/18_reference_model_smoke.py \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --experiment C07 --stage diffusion --steps 2 --gpu 0 \
  --stats-file "$OUT/derived96_train.npz" \
  --regression-checkpoint "$OUT/C07_regression/technical_state_dict.pt" \
  --output-dir "$OUT/C07_diffusion" \
  > "$OUT/C07_diffusion.log" 2>&1 < /dev/null &
```

## Conferência dos quatro resultados

```bash
"$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('analysis_outputs_reference/phase18_smoke_v1')
for stage in ('C00_regression','C00_diffusion','C07_regression','C07_diffusion'):
    f=root/stage/'audit.json'
    if not f.exists():print(stage,'PENDENTE');continue
    d=json.loads(f.read_text())
    print(stage, d['status'], 'losses', [x['loss'] for x in d.get('train_steps',[])],
          'GPU_peak_MB', round(d.get('peak_gpu_memory_bytes',0)/1024**2), 'error',d.get('error'))
PY
```

A execução gera em cada pasta um `audit.json` e, em caso de sucesso, um `technical_state_dict.pt`. Não reutilize caminhos de saída já existentes: isso evita confundir evidências de tentativas diferentes; renomeie o diretório na nova execução em vez de usar `--overwrite`.

## Limitações explícitas e condição de avanço

- Os **11 testes unitários CPU** aprovam lógica local; não comprovam GPU, performance ou compatibilidade final com os Zarrs do CEFET.
- As classes dos modelos são do PhysicsNeMo real, mas este pacote **não chama o `examples/weather/corrdiff/train.py`** e não executa o sampler de geração; sua aprovação é **parcial** para a Fase 18.
- Não usar checkpoints técnicos ou estatísticas C07 amostrais como resultados científicos. Os experimentos C00/C07 **completos** pertencem à Fase 19.
- Antes de concluir a Fase 18, avaliar a integração com um *loop* final mask-aware ou uma implementação segura do orquestrador, incluindo geração/amostragem, checkpoint nativo, métricas e organização de saída; nunca retirar máscaras só para satisfazer a assinatura antiga do dataset.
- O bundle v6 e o venv v4 permanecem congelados. Este pacote altera somente o código do repositório e escreve novos arquivos em diretórios de resultados; não modifica o Zarr ou o checkout da NVIDIA.

## Política Git

Versionar os códigos, o contrato novo, este documento e o `tests/test_phase18_reference_migration.py`. Resultados de smoke (checkpoints/logs) permanecem no armazenamento local. O commit é responsabilidade do mantenedor; a conexão GitHub usada na revisão possui permissão de leitura no repositório.
