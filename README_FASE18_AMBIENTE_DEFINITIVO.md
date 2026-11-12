# CorrDiff — Fase 18: correção de mistura CUDA 12/13 (v6.3)

## Diagnóstico

O ambiente Python de origem contém dois pacotes que escrevem o mesmo caminho nativo:

- `nvidia-cudnn-cu12==9.20.0.48`
- `nvidia-cudnn-cu13==9.24.0.43`

Essa é a causa dos binários cuDNN 9.24 encontrados na instalação que declarava cuDNN 9.20. **Não desativar a auditoria de sobreposição** nem confiar apenas em `pip check`. A instalação da máquina de origem pode conter uma combinação funcional após reparo manual, mas o snapshot bruto não é reproduzível como instalação limpa.

## Modificação v6.3

A opção explícita `--cuda-family cu12`, em `bundle` e `finalize`, cria uma **projeção auditável CUDA 12** do snapshot de origem. Exclui os pacotes cujos nomes indicam CUDA 13, como `nvidia-cudnn-cu13`, `nvidia-curand-cu13`, `cupy-cuda13x` e `nvidia-dali-cuda130`. O `source-freeze.txt` original permanece inalterado. A lista de exclusões é gravada em `cuda-target-exclusions.json` e no `manifest.json`.

O `finalize` de um bundle incompleto retira **somente do diretório `v6/wheels`** os wheels excluídos. Quando eram hardlinks dos arquivos do v5, o original continua intacto. O método verifica que o snapshot atual é o mesmo que iniciou o bundle, não apaga outros wheels e mantém a validação de colisões de binários nativos.

Crucial: a projeção não substitui o resolvedor. Antes do manifesto final, o comando executa um preflight de instalação `--dry-run --ignore-installed --no-index --require-hashes`, com somente os wheels preservados. Se algum pacote restante exigir CUDA 13, **a conclusão falhará** e será necessário investigar quem o exige; não o force com `--no-deps`.

O script inclui os controles anteriores: URL autorizada e hash do wheel Warp, suporte a extensões nativas na raiz do wheel, hashes de binários nativos pós-instalação e smoke test Warp no Doctor.

## Aplicação na máquina CEFET

Transfira o ZIP e execute, na raiz de `rionowcastdf`:

```bash
cd "$HOME/repositories/rionowcastdf"
cp -p scripts/reference_environment.py scripts/reference_environment.py.backup_v6_2
unzip -o /CAMINHO/phase18_reference_environment_definitivo_v6_3.zip -d .

SOURCE="$HOME/corrdiff_reference_reproduced_cu129_v3/bin/python"
"$SOURCE" -m unittest discover -s tests -p 'test_reference_environment.py' -v
```

Confira que não há outra execução ativa do bundle:

```bash
pgrep -af '[r]eference_environment.py (bundle|finalize|install)'
```

## Recuperar o bundle v6, preservando espaço

**Não execute novamente o `bundle`, não apague a pasta `v6` e não modifique o ambiente v3 que passou no Doctor.** A falha de colisão ocorreu antes da criação do manifesto final.

```bash
cd "$HOME/repositories/rionowcastdf"
SOURCE="$HOME/corrdiff_reference_reproduced_cu129_v3/bin/python"
BUNDLES="$HOME/corrdiff_environment_bundles"

test -f "$BUNDLES/reference_cu129_v6/source-freeze.txt" || { echo 'Snapshot v6 não encontrado'; exit 1; }
test ! -f "$BUNDLES/reference_cu129_v6/manifest.json" || { echo 'v6 já finalizado; não modificar'; exit 1; }

nohup "$SOURCE" -u scripts/reference_environment.py finalize \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --cuda-family cu12 \
  --bundle "$BUNDLES/reference_cu129_v6" \
  > "$BUNDLES/reference_cu129_v6_finalize_cu12.log" 2>&1 \
  < /dev/null &
echo "PID: $!"
```

Acompanhe:

```bash
tail -f "$HOME/corrdiff_environment_bundles/reference_cu129_v6_finalize_cu12.log"
```

**Critério de aprovação:** log termina em `Bundle created and preflight passed` e `v6/manifest.json` existe. É possível inspecionar as exclusões:

```bash
cat "$HOME/corrdiff_environment_bundles/reference_cu129_v6/cuda-target-exclusions.json"
```

Se o preflight reclamar de dependência `cu13` necessária a outro pacote, não instale o v6; investigue a exigência antes de alterar a projeção.

## Se for preciso iniciar outro bundle a partir do zero

Apenas se o `v6` não puder ser recuperado:

```bash
nohup "$SOURCE" -u scripts/reference_environment.py bundle \
  --cuda-family cu12 \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --reuse-bundle "$BUNDLES/reference_cu129_v5" \
  --wheel-url 'warp-lang=https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --output "$BUNDLES/reference_cu129_v7" \
  > "$BUNDLES/reference_cu129_v7.log" 2>&1 < /dev/null &
```

## Instalação limpa após preflight PASS

Uma instalação limpa **sem reparação automática** é a evidência mais forte de que o problema cuDNN foi resolvido na origem:

```bash
cd "$HOME/repositories/rionowcastdf"
SOURCE="$HOME/corrdiff_reference_reproduced_cu129_v3/bin/python"
BUNDLES="$HOME/corrdiff_environment_bundles"

nohup "$SOURCE" -u scripts/reference_environment.py install \
  --bundle "$BUNDLES/reference_cu129_v6" \
  --venv "$HOME/corrdiff_reference_reproduced_cu129_v4" \
  --no-auto-repair \
  > "$BUNDLES/install_cu129_v4.log" 2>&1 < /dev/null &
echo "PID: $!"
```

```bash
tail -f "$HOME/corrdiff_environment_bundles/install_cu129_v4.log"
```

Se a pasta de destino já existir de tentativa anterior, **não reutilize**; escolha outro caminho. Somente após `Environment installed and native integrity PASS` e `reference_installation.json` criado siga para o Doctor.

```bash
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
nohup "$PYTHON" -u scripts/reference_environment.py doctor \
  --project-dir "$HOME/repositories/rionowcastdf" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --gpus 0,1 \
  --output "$BUNDLES/doctor_reproduced_cu129_v4.json" \
  > "$BUNDLES/doctor_reproduced_cu129_v4.log" 2>&1 < /dev/null &
```

O `doctor` espera o dataset no caminho padrão `datasets/corrdiff_reference_v1`. Se estiver em outro local, informar `--dataset-dir /CAMINHO/REAL`.

Critérios: `passed: true`, `native_wheel_integrity: true`, `warp_cuda: true`, e `gpu_fp32: true` no campo `checks`, sem reparação automática no `install`.

## Escopo e limitações

O script e os testes unitários foram executados em ambiente local de desenvolvimento, sem os wheels reais do CEFET e sem suas GPUs. Nenhum teste nativo C00 ou C07 foi executado por esta alteração. A aprovação depende do preflight, da instalação limpa e do Doctor no servidor.
