# RioNowCastDF — Fase 18: ambiente CorrDiff de referência

**Estado: bundle e instalação aprovados no CEFET em 12/11/2026; smoke nativo C00/C07 pendente.**

Este documento registra a história técnica da reprodução do ambiente PhysicsNeMo e o procedimento para reutilizar os artefatos definitivos. Substitui o roteiro transitório v6.3 que pressupunha o ambiente `corrdiff_reference_reproduced_cu129_v3`, já descartado após a validação da versão v4.

## 1. Resultado alcançado

A instalação foi efetuada a partir de `reference_cu129_v6`, usando `--no-auto-repair` em uma nova virtualenv `corrdiff_reference_reproduced_cu129_v4`. A saída registrada foi:

```text
No broken requirements found.
Environment installed and native integrity PASS: /home/vsantos/corrdiff_reference_reproduced_cu129_v4/bin/python. Run doctor.
```

O Doctor retornou `passed: true`, com **11 verificações aprovadas**:

```text
checkout_clean                   True
dependencies                    True
native_imports                 True
gpu_fp32                       True
runtime                        True
dataset_contract               True
installed_commit_receipt       True
native_wheel_integrity         True
warp_cuda                      True
test_reference_adapter.py      True
test_reference_masked_losses.py True
```

Isso comprova consistência do ambiente Python/CUDA, integridade dos binários, leitura do contrato e testes de GPU, Warp, adaptador e funções de perda. **Não comprova treinamento nativo do CorrDiff, convergência de modelos nem validade física da região de cobertura radar.**

## 2. Configuração de referência

| Recurso | Versão/identidade |
|---|---|
| Bundle | `~/corrdiff_environment_bundles/reference_cu129_v6` |
| Venv aprovado | `~/corrdiff_reference_reproduced_cu129_v4` |
| Python | 3.12 |
| PyTorch | `2.13.0+cu129` |
| PhysicsNeMo | `2.3.0a0`, checkout `c15bd23eb56216c32bd3e30180593e5685be9bc2` |
| cuDNN | `nvidia-cudnn-cu12==9.20.0.48` |
| Warp | `1.19.0+cu12` |
| Zarr (venv treinável) | `3.1.6` |
| TensorDict | `0.15.5` |
| GPUs | 2 × NVIDIA GeForce RTX 2080 Ti, 11 GiB por GPU, SM 7.5 |

Não instalar dependências extras nesse venv congelado. Versões do driver devem ser confirmadas pelos registros do Doctor e pela máquina na ocasião de cada reprodução.

## 3. Por que foi necessário criar o bundle v6?

A configuração de origem gerou colisão nativa: `nvidia-cudnn-cu12==9.20.0.48` e `nvidia-cudnn-cu13==9.24.0.43` instalavam bibliotecas no mesmo caminho `nvidia/cudnn/lib/libcudnn.so.9`. Assim, metadados Python podiam indicar a versão 9.20 enquanto o runtime carregava binários 9.24.

A implementação definitiva de `scripts/reference_environment.py` introduziu:

- autorização explícita de wheel remoto por `--wheel-url`, incluindo verificação da origem e do fragmento SHA-256;
- tratamento de extensões nativas Python válidas (por exemplo `_duckdb.cpython-312-...so`);
- verificação de nomes nativos sobrepostos e integridade dos membros `.so` dos wheels;
- projeção explícita `--cuda-family cu12` mantendo `source-freeze.txt` original e relatório de exclusões;
- preflight offline com resolução e hashes, antes de aprovar o manifesto;
- auditoria da instalação sem correção automática via `--no-auto-repair`;
- Doctor com convolução/backpropagation nas duas GPUs e kernels Warp reais.

O `finalize` da versão v6 reportou:

```text
No broken requirements found.
CUDA target projection: excluded nvidia-cudnn-cu13, nvidia-cusparselt-cu13, nvidia-nccl-cu13, nvidia-nvshmem-cu13
Unlinked 4 CUDA 13 wheel(s) from incomplete bundle only
Bundle created and preflight passed: /home/vsantos/corrdiff_environment_bundles/reference_cu129_v6. Installation and doctor remain required.
```

O pacote v6 é a **projeção aprovada**, não um snapshot literal integral do ambiente de origem misto. Algumas distribuições de CUDA usam nomes sem sufixo `-cu13`; não classificar versões apenas pelo texto do nome. O bundle possui auditorias de colisão e o ambiente passou em execução real de GPU, mas outras ferramentas CUDA devem ser validadas conforme forem utilizadas.

## 4. Evidências que devem ser preservadas

```text
~/corrdiff_environment_bundles/
├── reference_cu129_v6/
│   ├── manifest.json
│   ├── requirements.lock
│   ├── source-freeze.txt
│   ├── preflight.log
│   ├── cuda-target-exclusions.json
│   └── wheels/
├── reference_cu129_v6_finalize_cu12.log
├── install_cu129_v4.log
├── doctor_reproduced_cu129_v4.json
├── doctor_reproduced_cu129_v4.log
├── zarr2_reference_baseline.json
├── zarr3_reference_comparison.json
└── history/

~/corrdiff_reference_reproduced_cu129_v4/
├── bin/python
├── reference_installation.json
└── reference_native_audit.json
```

Não mover a pasta de venv esperando que seus caminhos absolutos continuem válidos. Para reconstrução, criar **novo** ambiente de destino a partir do bundle completo e refazer Doctor. Não excluir o bundle v6 após a limpeza de versões anteriores.

## 5. Verificações de consulta (não destrutivas)

```bash
cd "$HOME/repositories/rionowcastdf"
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
BUNDLES="$HOME/corrdiff_environment_bundles"

"$PYTHON" -m pip check
"$PYTHON" -m unittest discover -s tests -p 'test_reference_environment.py' -v

python - <<'PY'
import json
from pathlib import Path
p = Path.home()/'corrdiff_environment_bundles/doctor_reproduced_cu129_v4.json'
r=json.loads(p.read_text())
print('DOCTOR PASSED:',r.get('passed'))
for name,passed in r.get('checks',{}).items():
    print(f'{name}: {passed}')
PY
```

Para reexecutar o Doctor sem substituir evidência anterior:

```bash
nohup "$PYTHON" -u scripts/reference_environment.py doctor \
  --project-dir "$HOME/repositories/rionowcastdf" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --gpus 0,1 \
  --output "$BUNDLES/doctor_recheck.json" \
  > "$BUNDLES/doctor_recheck.log" 2>&1 < /dev/null &
```

### Reconstrução a partir do bundle preservado

Usar um Python 3.12 compatível (por exemplo, o venv v4 **já aprovado**) como lançador; o destino precisa **não existir**. Não usar `/home/vsantos/corrdiff_reference_reproduced_cu129_v3`, que foi eliminado.

```bash
cd "$HOME/repositories/rionowcastdf"
SOURCE="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
BUNDLES="$HOME/corrdiff_environment_bundles"
DEST="$HOME/corrdiff_reference_reproduced_cu129_recheck"
test ! -e "$DEST" || { echo "Destino já existe: $DEST"; exit 1; }

nohup "$SOURCE" -u scripts/reference_environment.py install \
  --bundle "$BUNDLES/reference_cu129_v6" \
  --venv "$DEST" \
  --no-auto-repair \
  > "$BUNDLES/install_recheck.log" 2>&1 < /dev/null &
```

Só considere sucesso após integridade nativa aprovada e Doctor independente. Pacotes externos, checkout PhysicsNeMo e dados necessários ao leitor precisam permanecer disponíveis e identificados. **Não executar esse comando apenas por rotina; o v4 já está aprovado.**

## 6. Limite atual da Fase 18

O Doctor aprovado não executa `train.py` nativo. Antes dos smoke C00/C07, é obrigatório corrigir a divergência entre o Zarr de referência (`target=code/50`) e o contrato/entrypoints antigos da Fase 18 (`log1p`, dataset `corrdiff_2011_2024` e commit NVIDIA defasado). Essa atualização deve ser feita com testes e sem contaminar `test_primary` / `test_stress_ood`.

Critério futuro de conclusão da Fase 18: evidências reprodutíveis de batches reais, máscaras, forward/backward, perdas finitas, checkpoint e configuração Hydra/seed para os estágios nativos de regressão e difusão. C00/C07 completos pertencem à campanha de treinamento posterior, não aos smoke tests.

Veja também [`README.md`](README.md), [`configs/phase17_experiments.json`](configs/phase17_experiments.json) e [`docs/PLANO_LIMPEZA_REPOSITORIO_2026-11-12.md`](docs/PLANO_LIMPEZA_REPOSITORIO_2026-11-12.md).
