# Fase 18 — smoke de geração C00/C07 com checkpoints técnicos

**Estado de entrada:** C00/C07 regressão e difusão concluídos com PASS no ambiente `corrdiff_reference_reproduced_cu129_v4`. SHA-256 dos quatro checkpoints, identidade da referência e perdas de validação já verificados no CEFET. Isto **não** é treinamento completo.

## Escopo

O novo `scripts/18_reference_generation_smoke.py` carrega exatamente as classes PhysicsNeMo usadas por `scripts/18_reference_model_smoke.py`, mas reconstrói os modelos a partir dos `technical_state_dict.pt` do projeto, não de `.mdlus`. Reutiliza `build_model()` e `check_contract()` do script anterior para impedir divergência arquitetural silenciosa.

Gera **um patch de validação** (default `sample-index=0`) usando `CorrDiffRegressionUNet` seguido de `deterministic_sampler` com `EDMPrecondSuperResolution` e condicionamento pela média da regressão. A composição final é `regressão + residual`. Todo o processo conserva o alvo **`target_code / 50`**, sem `expm1`, sem ajuste ad hoc das classes discretas e sem usar o alvo como entrada da geração. O alvo é usado apenas para métrica depois da predição.

A implementação executa o **amostrador real do PhysicsNeMo**, mas não chama `examples/weather/corrdiff/generate.py`. A amostragem é curta e limitada (`4` passos, `sigma_min=0.01`, `sigma_max=2.0`) para um teste funcional de checkpoints **não treinados**; não é o protocolo científico EDM definitivo. Não interpretar o RMSE obtido como resultado de previsão.

O script impede leitura de `test_primary`/`test_stress_ood`, verifica o contrato atual, commit PhysicsNeMo, hashes SHA-256, identidade dos dois checkpoints, experimento, seed, número de canais, versão das estatísticas C07 e vínculo da difusão ao checkpoint exato de regressão. Ao gerar novos resultados, não sobrescreve diretórios já existentes.

## Instalação segura

```bash
cd "$HOME/repositories/rionowcastdf"
git status --short
unzip -n /CAMINHO/phase18_generation_addon_v1.zip -d .
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
"$PYTHON" -m unittest discover -s tests -p 'test_phase18_reference_generation.py' -v
```

Os oito testes unitários passaram em ambiente local **CPU**, sem o hardware e o dataset CEFET. O teste de GPU só será considerado PASS após execução real.

## C00 — geração a partir de checkpoints técnicos

```bash
cd "$HOME/repositories/rionowcastdf"
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
OUT="$PWD/analysis_outputs_reference/phase18_smoke_v1"

# Os quatro itens devem existir
for name in C00_regression C00_diffusion; do
  test -f "$OUT/$name/audit.json" || exit 1
  test -f "$OUT/$name/technical_state_dict.pt" || exit 1
done

test ! -e "$OUT/C00_generation" || { echo 'C00_generation já existe'; exit 1; }

nohup "$PYTHON" -u scripts/18_reference_generation_smoke.py \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --experiment C00 \
  --regression-run "$OUT/C00_regression" \
  --diffusion-run "$OUT/C00_diffusion" \
  --sample-index 0 --num-steps 4 --gpu 0 \
  --output-dir "$OUT/C00_generation" \
  > "$OUT/C00_generation.log" 2>&1 < /dev/null &
echo "PID: $!"
```

Acompanhar: `tail -f "$OUT/C00_generation.log"`. Em caso de FAIL, consultar `$OUT/C00_generation/audit.json` e corrigir a causa; nunca trocar checkpoints para forçar PASS.

## C07 — geração com os oito canais derivados

```bash
cd "$HOME/repositories/rionowcastdf"
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
OUT="$PWD/analysis_outputs_reference/phase18_smoke_v1"
STATS="$OUT/derived96_train.npz"
test -f "$STATS" || { echo 'Estatísticas C07 indisponíveis'; exit 1; }
for name in C07_regression C07_diffusion; do
  test -f "$OUT/$name/audit.json" || exit 1
  test -f "$OUT/$name/technical_state_dict.pt" || exit 1
done

test ! -e "$OUT/C07_generation" || { echo 'C07_generation já existe'; exit 1; }

nohup "$PYTHON" -u scripts/18_reference_generation_smoke.py \
  --dataset-dir "$PWD/datasets/corrdiff_reference_v1" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --experiment C07 \
  --stats-file "$STATS" \
  --regression-run "$OUT/C07_regression" \
  --diffusion-run "$OUT/C07_diffusion" \
  --sample-index 0 --num-steps 4 --gpu 0 \
  --output-dir "$OUT/C07_generation" \
  > "$OUT/C07_generation.log" 2>&1 < /dev/null &
echo "PID: $!"
```

**Importante:** `derived96_train.npz` contém estatísticas amostrais **somente para smoke**. Antes do treinamento C07 científico, gerar estatísticas sobre **todos** os horários de treino (células únicas), com uma nova identidade, e reavaliar proveniência.

## O que deve aparecer nos resultados

Para cada experimento, o novo diretório terá `audit.json` e `generated_validation_patch.npz`. O NPZ contém:

- `conditioning`: `[1,12,32,32]` no C00 e `[1,20,32,32]` no C07
- `target_code50` e `mask`: referência e suporte válido, apenas para avaliação
- `regression_code50`: predição determinística
- `residual_code50`: amostra residual de difusão
- `prediction_code50_unclipped`: soma da regressão + residual, **sem clipping** para não esconder saídas fora do domínio

O JSON registra os SHA-256 dos checkpoints e da predição, o número de pixels válidos, métricas **mascaradas** no código normalizado (`MSE`, `RMSE`, `MAE`), fração de predições fora de `[0,1]` e a fração de pixels sem eco. Não interpreta números contínuos intermediários como refletividade medida e não aplica `expm1`.

## Verificação consolidada

```bash
"$PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
base=Path('analysis_outputs_reference/phase18_smoke_v1')
for exp in ('C00','C07'):
    r=json.loads((base/f'{exp}_generation/audit.json').read_text())
    print(exp, r['status'], r.get('metrics'), r.get('error'))
    assert r['status']=='PASS' and r['locked_splits_accessed'] is False
    artifact=Path(r['generated_npz'])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==r['generated_npz_sha256']
print('GERAÇÃO TÉCNICA C00/C07: PASS')
PY
```

## Critério científico para avançar

Se C00 e C07 retornarem PASS, terão sido comprovados os caminhos de inferência e geração (checkpoints técnicos, carregamento, regressão, residual, composição, métricas mascaradas e sampler). Isso **não** comprova convergência, adequação científica de `sigma_max`, estabilidade em dados extensos, custo do treinamento completo, capacidade de retomada de otimizador, formato `.mdlus` ou validade física da cobertura radar.

A consolidação da Fase 18 requer ainda decidir e testar o treinador definitivo com máscara, normalização C07 completa, persistência/reinício de otimizador e validação periódica. A campanha e o piloto de aprendizado pertencem à Fase 19.
