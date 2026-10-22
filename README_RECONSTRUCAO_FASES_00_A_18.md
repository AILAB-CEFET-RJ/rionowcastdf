# RioNowcast: reconstrução do dataset e execução das fases 0–18

Roteiro de recuperação elaborado em 21/10/2026, a partir do código do repositório AILAB-CEFET-RJ/rionowcastdf no commit `0bd42394e461ae233bdd6fcc9076fcea00f8208b`.

Este documento deve ser colocado na raiz do **rionowcastdf**, ao lado de `scripts/`, `src/`, `configs/` e `adapter/`. Todos os comandos são executados nessa raiz, salvo indicação explícita. Nenhum dado remoto foi reconstruído durante a elaboração deste documento: a execução depende dos arquivos e do ambiente da máquina do CEFET.

## 1. Estratégia de recuperação

Não é necessário repetir download de ERA5, conversão dos PNGs ou treinamento para descobrir se o dataset foi recuperado. Primeiro preservamos as referências, reconstruímos a partir do cache existente e executamos as fases 0, 6 e 8. Elas permitem comparar estrutura, cobertura, estatísticas por horário e hashes do target. Só depois refazemos as demais análises e os artefatos de treino. A sequência numérica completa também está disponível abaixo.

1. Preservar resultados antigos, código, configuração, ambiente e fontes sobreviventes.
2. Construir em um diretório novo; reutilizar cache sem completá-lo automaticamente.
3. Auditar integralmente e comparar com as referências antigas.
4. Investigar divergências antes de mudar filtros, datas, canais ou tolerâncias.
5. Regerar fases 1–17 e respectivos manifestos a partir do novo Zarr.
6. Executar a integração nativa NVIDIA da fase 18 em GPU, usando treino e validação.

Não copie índices de patches antigos sem verificar timestamps, posições e ordem. Não use a normalização global do builder para treinar os modelos formais. Não use `--overwrite` como solução para uma execução interrompida: escolha outro diretório de saída e mantenha os logs anteriores.

## 2. Contrato histórico verificável

Fonte: `analysis_outputs/00_quality/dataset_summary.json` do commit acima. São referências de comparação, não resultados já obtidos na reconstrução.

| Item | Referência |
|---|---|
| Período solicitado | 2011-01-01 00:00 a 2024-12-31 23:00, UTC, frequência horária |
| Horários esperados / presentes / ausentes | 122.736 / 84.684 / 38.052 |
| Primeiro horário observado | 2011-01-17 19:00 UTC |
| Patches | 1.016.208; 12 por horário |
| Grade / resolução nominal | 70 × 84; 2 km |
| Patch / stride | 32 × 32 / 16 |
| Input | float32, (1016208, 12, 32, 32) |
| Target e máscara | float32, (1016208, 1, 32, 32) |
| Chunk principal | 256 patches |
| Máscara válida | 100% dos pixels armazenados |
| Contadores | missing_radar=38049; invalid_radar=3; missing_era5=0 |
| Rejeições por patch / shape mismatch | zero na referência |
| Cobertura espacial pelos patches | aproximadamente 87,07% da grade |

Ordem obrigatória dos canais:

```text
tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500
```

Os padrões do builder atual selecionam outras variáveis. Por isso os comandos abaixo explicitam os quatro campos de superfície e os quatro campos em cada um dos níveis 850 e 500 hPa.

O target é `float32(log1p(clip(nan_to_num(cache), min=0)))`; a máscara é calculada antes da substituição dos valores inválidos. `expm1(target)` recupera o valor não negativo do cache, com arredondamento numérico. A fase 0 registra que o builder sozinho **não comprova a unidade física dBZ**, embora os contratos posteriores usem esse nome. Não confundir essa variável com precipitação em mm/h, nem aplicar conversão Z–R sem validação externa.

## 3. Preparar uma sessão recuperável

Recomenda-se uma sessão `tmux` no servidor, se disponível. Execute os blocos no mesmo Bash. Ajuste os caminhos abaixo para as fontes que realmente sobreviveram. O cache usado na auditoria histórica estava em outro caminho; o nome do diretório não comprova que o conteúdo seja o mesmo.

```bash
set -euo pipefail
export REPO="$HOME/repositories/rionowcastdf"
cd "$REPO"
export RUN="$REPO/recovery_runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN/reference" "$RUN/logs" "$RUN/analysis_outputs"
export DATASET="$RUN/dataset"
export A="$RUN/analysis_outputs"
export ERA5_ROOT="/home/vsantos/data/reanalysis/cds/era5"
export RADAR_CACHE="/home/vsantos/data/radar_cache"
export RADAR_SOURCE="/home/vsantos/data/radar_sumare"
test -d "$ERA5_ROOT/single"
test -d "$ERA5_ROOT/pressure"
test -d "$RADAR_CACHE"
git rev-parse HEAD > "$RUN/reference/rionowcast_commit.txt"
git status --short > "$RUN/reference/git_status.txt"
git diff HEAD > "$RUN/reference/local_changes.patch"
python --version > "$RUN/reference/python.txt"
python -m pip freeze > "$RUN/reference/pip_freeze_before.txt"
cp -a configs "$RUN/reference/"
if [ -d analysis_outputs ]; then
  cp -a analysis_outputs "$RUN/reference/analysis_outputs"
fi
for f in corrdiff_builder_2011_2024.log radar_cache_builder.log; do
  if [ -f "$f" ]; then cp -a "$f" "$RUN/reference/"; fi
done
printf 'export REPO=%q
export RUN=%q
export DATASET=%q
export A=%q
export ERA5_ROOT=%q
export RADAR_CACHE=%q
export RADAR_SOURCE=%q
'   "$REPO" "$RUN" "$DATASET" "$A" "$ERA5_ROOT" "$RADAR_CACHE" "$RADAR_SOURCE" > "$RUN/session.env"
df -h "$RUN"
du -sh "$ERA5_ROOT" "$RADAR_CACHE"
```

Anote o caminho de `session.env`. Após reconectar: `source /caminho/da/execucao/session.env`, `cd "$REPO"`, reative o ambiente e execute novamente `set -euo pipefail`. Não execute novamente o bloco que cria RUN para retomar uma execução.

Compare o commit local com o commit documentado. Se diferir ou houver alterações locais, preserve-as e revise os scripts antes de executar. Não faça reset do repositório. Para reproduzir estritamente a versão examinada, é possível criar um worktree separado com `git worktree add --detach <novo-diretorio> 0bd42394e461ae233bdd6fcc9076fcea00f8208b`, ajustando REPO e copiando as referências antigas para RUN.

A cópia de `analysis_outputs` preserva artefatos locais, inclusive fases 15–17 se ainda existirem. No snapshot remoto examinado, encontrei resultados versionados até a fase 14; scripts de uma fase não provam que seus resultados estejam preservados. Faça também uma cópia das fontes/cache em armazenamento independente, se houver espaço. `cp` no mesmo disco não substitui backup externo.

## 4. Ambiente e entradas

Prefira o ambiente que criou o dataset anterior. A auditoria histórica registra Python 3.12.14. As versões antigas de todas as bibliotecas não estão recuperadas aqui; instalar versões novas pode introduzir diferenças numéricas.

Se for necessário um ambiente novo para builder e análises:

```bash
python3 -m venv "$RUN/venv-analysis"
source "$RUN/venv-analysis/bin/activate"
python -m pip install -r requirements_corrdiff_builder.txt
python -m pip install scikit-learn matplotlib seaborn
python -m pip check
python -m pip freeze > "$RUN/reference/pip_freeze_analysis.txt"
python -m src.corrdiff.builder.dataset_builder --help > "$RUN/reference/builder_help.txt"
```

Mantenha Zarr 2 (`zarr<3`) e os limites de numcodecs do requirements. A fase 18 usa outro ambiente, compatível com CUDA/PhysicsNeMo. Não atualize o PyTorch de um ambiente funcional indiscriminadamente.

O builder consome **ERA5 em Parquet**, não GRIB/NetCDF diretamente. Espera `ERA5_ROOT/single` e `ERA5_ROOT/pressure`; aceita partições mensais como `year=2011/month=1`, `2011/01` e nomes `era5_single_2011_01.parquet` / `era5_pressure_2011_01.parquet`. Os níveis ficam em coluna `pressure_level`; as variáveis são colunas. Se só restaram GRIB/NetCDF, é necessário recuperar/reexecutar o ETL original antes deste roteiro; não renomeie arquivos para Parquet.

O cache esperado contém `YYYYMMDD_HH_MM.npy`, em grade 70×84. Os PNGs brutos seguem `YYYY/MM/DD/YYYY_MM_DD_HH_MM.png`. `never` impede geração de cache; `auto` não garante completude, pois basta encontrar cache existente; `always` pode acrescentar horários antes ausentes. Para comparar o dataset anterior, não complete o cache nesta primeira reconstrução.

Os limites abaixo são os padrões geográficos atuais do builder. O resumo antigo confirma o tamanho da grade, mas não registra os limites. Confira metadados/logs antigos e a configuração que gerou o cache: mesmo shape não garante mesma geolocalização. Havendo outra configuração histórica, use-a e registre a diferença.

## 5. Construção piloto e completa

Primeiro teste um intervalo que contém dados históricos. O piloto usa saída própria e não será concatenado ao dataset final.

```bash
BUILD_ARGS=(
  --era5_root "$ERA5_ROOT" --radar_cache "$RADAR_CACHE"
  --radar_source "$RADAR_SOURCE" --radar_cache_policy never
  --surface_variables tcwv,t2m,u10,v10
  --pressure_variables t,r,u,v --pressure_levels 850,500
  --lat_range -23.5 -22.25 --lon_range -44.0 -42.5
  --radar_res 2 --time_frequency 1h --patch_size 32 --stride 16
  --chunk_size 256 --write_batch_size 2048
  --minimum_radar_valid_ratio 0.05 --minimum_input_valid_ratio 1
)
python -m src.corrdiff.builder.dataset_builder   "${BUILD_ARGS[@]}" --begin '2011-01-17 00:00:00' --end '2011-01-18 23:00:00'   --output_dir "$RUN/pilot_dataset" 2>&1 | tee "$RUN/logs/build_pilot.log"
python scripts/00_audit_dataset.py --dataset-dir "$RUN/pilot_dataset"   --radar-cache-dir "$RADAR_CACHE" --output-dir "$RUN/pilot_quality"   2>&1 | tee "$RUN/logs/audit_pilot.log"
```

Confira canais, grade, dados finitos e transformação cache→target. A cobertura curta não deve ser comparada às contagens de 14 anos. Os limiares de validade acima são os defaults atuais; caso um log histórico registre valores diferentes, restaure-os antes do build completo.

```bash
# Reexecute a definição BUILD_ARGS se tiver reconectado a sessão.
test ! -e "$DATASET"
python -m src.corrdiff.builder.dataset_builder   "${BUILD_ARGS[@]}" --begin '2011-01-01 00:00:00' --end '2024-12-31 23:00:00'   --output_dir "$DATASET" 2>&1 | tee "$RUN/logs/build_full.log"
test -f "$DATASET/metadata.json"
test -d "$DATASET/train.zarr"
```

O volume lógico dos tensores float32 é aproximadamente 58 GB decimais, antes da compressão, além das fontes, análises e checkpoints. A compressão real varia; não há estimativa confiável de duração sem medir a máquina. Interrupção pode deixar Zarr parcial: não o promova nem presuma retomada automática. Guarde-o, escolha nova saída e reconstrua.

## 6. Fases 0–15

Os comandos preservam os defaults científicos do commit examinado, incluindo seeds e tamanhos amostrais. Saídas ficam em RUN; nenhuma fase deve consumir acidentalmente `analysis_outputs` antigo. Para recuperação rápida, execute 0 → 6 → 8, faça a comparação da seção 7 e então complete a sequência. Não use `--skip-predictors` na fase 6, pois fases posteriores precisam dos preditores agregados.

### Fase 0 — Integridade e semântica

Varre o Zarr, confere dimensões, finitude, máscara, cobertura temporal/espacial e transformação do cache. PASS estrutural não é equivalência ao dataset antigo.

```bash
python scripts/00_audit_dataset.py \
  --dataset-dir "$DATASET" --radar-cache-dir "$RADAR_CACHE" \
  --output-dir "$A/00_quality" \
  2>&1 | tee "$RUN/logs/phase00.log"
```

### Fase 1 — Distribuições univariadas

Resume cada canal e o target; combina estatísticas exatas da fase 0 com distribuições amostradas. Compare também a configuração amostral.

```bash
python scripts/01_compute_univariate_stats.py \
  --dataset-dir "$DATASET" --phase0-dir "$A/00_quality" \
  --output-dir "$A/01_univariate" \
  2>&1 | tee "$RUN/logs/phase01.log"
```

### Fase 2 — Variáveis derivadas

Analisa velocidades do vento, diferenças térmicas/umidade e diferenças vetoriais. O código atual usa amostra conjunta alinhada para correlações. Não modifica os canais do Zarr.

```bash
python scripts/02_compute_derived_variables.py \
  --dataset-dir "$DATASET" --phase1-dir "$A/01_univariate" \
  --output-dir "$A/02_derived" \
  2>&1 | tee "$RUN/logs/phase02.log"
```

### Fase 3 — Relações conjuntas

Calcula associações, informação mútua e distribuições condicionais ERA5–radar; associação não implica causalidade ou habilidade futura.

```bash
python scripts/03_compute_joint_statistics.py \
  --dataset-dir "$DATASET" \
  --output-dir "$A/03_joint" \
  2>&1 | tee "$RUN/logs/phase03.log"
```

### Fase 4 — Extremos

Varre eventos e constrói amostras estratificadas para diagnóstico dos preditores. Verifique definição de limiar e tratamento float32 antes de comparar taxas de versões antigas.

```bash
python scripts/04_compute_extreme_statistics.py \
  --dataset-dir "$DATASET" --phase0-dir "$A/00_quality" \
  --output-dir "$A/04_extremes" \
  2>&1 | tee "$RUN/logs/phase04.log"
```

### Fase 5 — Sazonalidade

Relaciona disponibilidade, eventos e preditores ao calendário; diferenças de cobertura não devem ser interpretadas automaticamente como clima.

```bash
python scripts/05_compute_seasonality.py \
  --dataset-dir "$DATASET" \
  --output-dir "$A/05_seasonality" \
  2>&1 | tee "$RUN/logs/phase05.log"
```

### Fase 6 — Ciclo diurno

Produz timestamp_event_metrics.parquet e predictor_timestamp_means.parquet, essenciais à comparação e às fases temporais. Usa America/Sao_Paulo, inclusive regras históricas de horário de verão.

```bash
python scripts/06_compute_diurnal_cycle.py \
  --dataset-dir "$DATASET" --local-timezone America/Sao_Paulo \
  --output-dir "$A/06_diurnal" \
  2>&1 | tee "$RUN/logs/phase06.log"
```

### Fase 7 — Defasagens temporais

Examina relações em lags 0,1,2,3,6,12,24 horas a partir da fase 6. Lacunas precisam ser tratadas por tempo real, não por linha adjacente.

```bash
python scripts/07_compute_temporal_lags.py \
  --phase6-dir "$A/06_diurnal" \
  --output-dir "$A/07_lags" \
  2>&1 | tee "$RUN/logs/phase07.log"
```

### Fase 8 — Persistência exploratória e hashes

Compara referências temporais e audita continuidade diretamente no target. Não pule a auditoria: ela gera radar_timestamp_hashes.parquet. O split exploratório padrão termina treino em 2019 e validação em 2021; não substitui o protocolo formal da fase 15.

```bash
python scripts/08_compute_persistence_baselines.py \
  --dataset-dir "$DATASET" --phase6-dir "$A/06_diurnal" \
  --output-dir "$A/08_persistence_baselines" \
  2>&1 | tee "$RUN/logs/phase08.log"
```

### Fase 9 — Estrutura espacial do radar

Analisa organização espacial por estratos de intensidade. A amostragem estratificada não representa diretamente a frequência natural dos eventos.

```bash
python scripts/09_compute_spatial_structure.py \
  --dataset-dir "$DATASET" \
  --output-dir "$A/09_spatial_structure" \
  2>&1 | tee "$RUN/logs/phase09.log"
```

### Fase 10 — Associação espacial ERA5–radar

Examina suavizações e deslocamentos espaciais; mantém a ordem canônica dos 12 canais.

```bash
python scripts/10_compute_spatial_era5_radar.py \
  --dataset-dir "$DATASET" \
  --output-dir "$A/10_spatial_era5_radar" \
  2>&1 | tee "$RUN/logs/phase10.log"
```

### Fase 11 — Escalas espaciais

Compara coarse graining e escalas de associação. Interpolar ERA5 para 2 km não cria informação atmosférica observada nessa resolução.

```bash
python scripts/11_compute_spatial_scales.py \
  --dataset-dir "$DATASET" \
  --output-dir "$A/11_spatial_scales" \
  2>&1 | tee "$RUN/logs/phase11.log"
```

### Fase 12 — Análise multivariada

Compara capacidade explicativa conjunta com modelos exploratórios. Seus resultados não são a avaliação formal do CorrDiff.

```bash
python scripts/12_compute_multivariate_analysis.py \
  --dataset-dir "$DATASET" \
  --output-dir "$A/12_multivariate" \
  2>&1 | tee "$RUN/logs/phase12.log"
```

### Fase 13 — Regimes meteorológicos

Aplica redução dimensional e agrupamento aos preditores por timestamp, incluindo diagnósticos de estabilidade. Rótulos de clusters podem permutar: compare partições e centroides, não apenas números de rótulo.

```bash
python scripts/13_compute_meteorological_regimes.py \
  --phase6-dir "$A/06_diurnal" \
  --output-dir "$A/13_regimes" \
  2>&1 | tee "$RUN/logs/phase13.log"
```

### Fase 14 — Mudança de distribuição

Compara cobertura, eventos, preditores e regimes entre períodos. Trata 2024 como potencial stress/OOD; não ajusta automaticamente o split.

```bash
python scripts/14_compute_distribution_shift.py \
  --phase6-dir "$A/06_diurnal" --phase13-dir "$A/13_regimes" \
  --output-dir "$A/14_distribution_shift" \
  2>&1 | tee "$RUN/logs/phase14.log"
```

### Fase 15 — Partições formais e purga

Gera manifestos temporais e índices de patches: treino 2011–2020, validação 2021–2022, teste primário 2023 e stress/OOD 2024, com purga de 48 h em cada lado das fronteiras. Não separa aleatoriamente patches sobrepostos.

```bash
python scripts/15_compute_formal_splits.py \
  --dataset-dir "$DATASET" --phase6-dir "$A/06_diurnal" --phase13-dir "$A/13_regimes" --phase14-dir "$A/14_distribution_shift" --purge-hours 48 --expected-patches-per-timestamp 12 \
  --output-dir "$A/15_formal_splits" \
  2>&1 | tee "$RUN/logs/phase15.log"
```

## 7. Comparar a reconstrução com o que sobreviveu

Uma contagem igual não comprova valores iguais. Use três níveis:

1. **Estrutural/temporal:** canais e ordem, shapes, tipos, posições, timestamps, máscaras e contadores; compare `temporal_coverage.parquet` e `patch_position_counts.parquet` da fase 0.
2. **Numérico agregado:** `channel_summary.parquet`, eventos por timestamp e médias ERA5 por timestamp. Compare chaves e valores; médias iguais não provam campos iguais.
3. **Target exato:** se sobreviveu `08_persistence_baselines/radar_timestamp_hashes.parquet`, compare `target_hash_blake2b128` alinhado por `timestamp_utc`. O script atual usa bytes float32 dos patches ordenados por horário. Igualdade desses hashes oferece evidência forte de igualdade do target coberto, não do input ou máscara. Confirme que o algoritmo antigo era o mesmo.

O bloco abaixo compara tabelas lógicas, ignorando apenas a ordem das linhas (ordena por chaves). Campos inteiros, strings e datas devem coincidir; floats usam rtol=1e-6 e atol=1e-6 para diagnóstico. Mudança de schema é divergência, não conversão silenciosa. Os hashes são strings e portanto comparados exatamente. Falta de referência é INCONCLUSIVO.

```bash
python - <<'PYCOMPARE'
import os, json
from pathlib import Path
import pandas as pd
old=Path(os.environ['RUN'])/'reference/analysis_outputs'
new=Path(os.environ['A'])
jobs=[
 ('00_quality/temporal_coverage.parquet',['timestamp']),
 ('00_quality/patch_position_counts.parquet',['patch_row','patch_col']),
 ('06_diurnal/timestamp_event_metrics.parquet',['timestamp_utc']),
 ('06_diurnal/predictor_timestamp_means.parquet',['timestamp_utc']),
 ('08_persistence_baselines/radar_timestamp_hashes.parquet',['timestamp_utc']),
]
results=[]
for name, keys in jobs:
    a,b=old/name,new/name
    row={'file':name}
    if not a.exists() or not b.exists():
        row.update(status='INCONCLUSIVO', reason='referencia ou resultado ausente')
    else:
        try:
            x,y=pd.read_parquet(a),pd.read_parquet(b)
            # A fase 0 pode usar nome de coluna diferente: inspecione antes de alterar a chave.
            for frame in (x,y):
                if not set(keys)<=set(frame.columns):
                    raise ValueError(f'Chaves {keys} ausentes; colunas={list(frame.columns)}')
                if frame.duplicated(keys).any():
                    raise ValueError('Chave nao unica; definir chave completa antes de comparar')
            x=x.sort_values(keys).set_index(keys).sort_index(axis=1)
            y=y.sort_values(keys).set_index(keys).sort_index(axis=1)
            pd.testing.assert_index_equal(x.index,y.index)
            pd.testing.assert_index_equal(x.columns,y.columns)
            floats=[c for c in x if pd.api.types.is_float_dtype(x[c])]
            exact=[c for c in x if c not in floats]
            pd.testing.assert_frame_equal(x[exact],y[exact],check_exact=True)
            pd.testing.assert_frame_equal(x[floats],y[floats],check_exact=False,rtol=1e-6,atol=1e-6)
            row.update(status='COMPATIVEL_NESTA_TABELA', rows=len(x))
        except (AssertionError,ValueError,KeyError) as exc:
            row.update(status='DIVERGENTE_OU_SCHEMA_INCOMPATIVEL', reason=str(exc)[:2500])
    results.append(row)
path=Path(os.environ['RUN'])/'comparison_tables.json'
path.write_text(json.dumps(results,indent=2,ensure_ascii=False))
print(path.read_text())
PYCOMPARE
```

Compare também `dataset_summary.json` antigo e novo: `dataset`, `configuration`, `temporal`, `patch_geometry`, `mask`, `target_integrity`, `builder_counters`. Datas de execução, duração, caminhos e hostname naturalmente mudam; não devem ser confundidos com mudanças de dados. As taxas de evento podem mudar se a implementação dos limiares foi corrigida, mesmo com hashes de target iguais.

Se houver divergência: localize primeiro horários adicionados/removidos, depois patches/posições, depois valores. Verifique cache incompleto ou reparado, Parquet alterado, limites geográficos, ordem dos canais, níveis, duplicatas ERA5, interpolação, versões e arredondamento. Não ajuste critérios só para obter as contagens antigas. Registre a causa e, se a mudança for intencional, atribua uma nova versão ao dataset.

Sem Zarr antigo nem hashes completos de input/mask, a equivalência integral desses arrays permanece não demonstrada. Se restarem hashes antigos de target, ainda é possível verificar essa parte. Salve um inventário SHA-256 dos arquivos do novo dataset para futuras verificações; ele identifica a serialização exata, que pode mudar com compressão/chunking mesmo quando os valores não mudam:

```bash
(cd "$DATASET" && find . -type f -print0 | sort -z | xargs -0 sha256sum)   > "$RUN/reference/new_dataset_files.sha256"
# Em outra ocasião, verificar sem alterar o dataset:
# (cd "$DATASET" && sha256sum -c "$RUN/reference/new_dataset_files.sha256")
```

## 8. Fase 16 — baselines clássicos e handoff

O guia antigo `guides/README_PHASE16.md` ainda descreve redes locais. Para a integração atual, use os scripts **classical**, cuja avaliação declara v2.2. Eles estabelecem zero, média global, média por posição, climatologia mês/hora/posição e persistência de 1 hora. A persistência usa radar anterior observado: é uma referência com informação diferente do modelo condicionado somente por ERA5.

A preparação ajusta normalização/climatologia somente em treino; a avaliação usa validação e a coorte comum de persistência. Não habilite testes bloqueados durante recuperação ou seleção.

```bash
python scripts/16_prepare_classical_baselines.py --dataset-dir "$DATASET"   --phase15-dir "$A/15_formal_splits" --output-dir "$A/16_baselines"   2>&1 | tee "$RUN/logs/phase16_prepare.log"
python scripts/16_evaluate_classical_baselines.py --dataset-dir "$DATASET"   --phase15-dir "$A/15_formal_splits" --phase16-dir "$A/16_baselines"   --splits validation --with-persistence-common   2>&1 | tee "$RUN/logs/phase16_evaluate.log"
python scripts/16_build_nvidia_handoff.py --dataset-dir "$DATASET"   --phase15-dir "$A/15_formal_splits" --phase16-dir "$A/16_baselines"   2>&1 | tee "$RUN/logs/phase16_handoff.log"
python scripts/16_summarize_classical_baselines.py --phase16-dir "$A/16_baselines"   --output "$A/summary_phase16.txt"
```

Inspecione preparação, métricas, coortes e `nvidia_handoff/`. RMSE global isolado é insuficiente: analise também eventos raros, CSI/FSS e sazonalidade. Não execute `16_train_baselines.py` como pré-requisito automático desse fluxo clássico.

## 9. Fase 17 — contrato de ablações

Valida catálogo/canais e materializa C00–C10, manifestos e vínculo com o handoff da fase 16. Esta etapa define experimentos; não treina CorrDiff. C00 usa os 12 canais raw; C07 acrescenta as oito derivadas, totalizando 20. Não use a opção de pular validação do handoff para contornar resultados ausentes.

```bash
python scripts/17_validate_ablation_plan.py --config-dir configs   2>&1 | tee "$RUN/logs/phase17_validate.log"
python scripts/17_build_ablation_manifest.py --package-config-dir configs   --phase16-handoff-dir "$A/16_baselines/nvidia_handoff"   --output-dir "$A/17_conditioning_ablations"   2>&1 | tee "$RUN/logs/phase17_build.log"
python scripts/17_summarize_ablation_plan.py --phase17-dir "$A/17_conditioning_ablations"   --output "$A/summary_phase17.txt"
```

## 10. Fase 18 — integração nativa com NVIDIA CorrDiff

**Sim: nesta fase usamos o código da NVIDIA.** O projeto fornece o adapter e contratos; o treinamento mínimo de regressão e difusão é executado pelo CorrDiff nativo. A etapa comprova compatibilidade de dados, normalização, seeds, checkpoints e geração, sem realizar a campanha científica completa.

O contrato fixa PhysicsNeMo em `beafeecbd01fcac3c76bc6f3391a6af1872ea6bf`. O diretório passado em `--corrdiff-dir` é **examples/weather/corrdiff**, não a raiz do repositório. Se o clone se chama `corrdiff`, ajuste NVIDIA_REPO abaixo; se se chama `physicsnemo`, use esse nome. Ambos podem continuar como irmãos de rionowcastdf dentro de repositories.

Para preservar alterações e configurações antigas, crie um worktree NVIDIA exclusivo desta recuperação:

```bash
export NVIDIA_REPO="$HOME/repositories/physicsnemo"  # ou "$HOME/repositories/corrdiff"
export NVIDIA_PIN=beafeecbd01fcac3c76bc6f3391a6af1872ea6bf
git -C "$NVIDIA_REPO" cat-file -e "$NVIDIA_PIN^{commit}"
# Se o commit não existir localmente, obtenha-o do remoto NVIDIA antes de prosseguir.
git -C "$NVIDIA_REPO" worktree add --detach "$RUN/physicsnemo" "$NVIDIA_PIN"
export CORRDIFF_DIR="$RUN/physicsnemo/examples/weather/corrdiff"
export P18="$A/18_nvidia_integration"
mkdir -p "$P18"
test -f "$CORRDIFF_DIR/train.py"
printf 'export CORRDIFF_DIR=%q
export P18=%q
' "$CORRDIFF_DIR" "$P18" >> "$RUN/session.env"
```

Ative o ambiente NVIDIA/CUDA já validado na máquina. Se não existir, siga as instruções de instalação do checkout fixado, compatíveis com o driver/CUDA local; não há um comando universal seguro para escolher o wheel CUDA nesta documentação. Instale as dependências do exemplo e do adapter no mesmo ambiente:

```bash
cd "$REPO"
python -m pip install -r "$CORRDIFF_DIR/requirements.txt"
python -m pip install -r requirements_phase18.txt
python -m pip check
python -c 'import torch, physicsnemo; print(torch.__version__); print("CUDA:",torch.cuda.is_available()); assert torch.cuda.is_available()'
python -m pip freeze > "$RUN/reference/pip_freeze_phase18.txt"
nvidia-smi > "$RUN/reference/nvidia_smi.txt"
```

A importação de physicsnemo deve corresponder ao ambiente compatível com o checkout fixado. O requirements do adapter não instala sozinho todo o stack NVIDIA. Não prossiga se importações/dependências falharem.

### 18.1 Checkout e estatísticas exclusivas de treino

```bash
python scripts/18_inspect_nvidia_corrdiff_checkout.py --corrdiff-dir "$CORRDIFF_DIR"   --output "$P18/phase18_nvidia_checkout_audit.json"
python scripts/18_prepare_conditioning_stats.py --dataset-dir "$DATASET"   --phase15-dir "$A/15_formal_splits" --phase17-dir "$A/17_conditioning_ablations"   --output-dir "$P18/conditioning_stats"   2>&1 | tee "$RUN/logs/phase18_stats.log"
```

As derivadas são calculadas em unidades originais antes da normalização. Estatísticas dos 20 canais são ajustadas no treino e selecionadas por experimento. O target permanece em log1p com normalização identidade no smoke test; não aplicar uma segunda transformação.

### 18.2 Adapter, máscaras e bloqueio dos testes

```bash
python scripts/18_smoke_dataset_adapter.py --corrdiff-dir "$CORRDIFF_DIR"   --adapter-path "$REPO/adapter/rionowcast_corrdiff_dataset.py"   --dataset-dir "$DATASET" --phase15-dir "$A/15_formal_splits"   --phase17-dir "$A/17_conditioning_ablations" --stats-dir "$P18/conditioning_stats"   --output "$P18/dataset_smoke_audit.json"   2>&1 | tee "$RUN/logs/phase18_adapter.log"
```

O adapter retorna `(target, conditioning)` para a interface nativa. A auditoria verifica máscaras de treino/validação; a loss nativa não aplica a máscara. Se houver pixels inválidos, o caminho deve bloquear: será necessário implementar/validar loss mascarada, e não silenciar inválidos. Testes 2023/2024 devem ser recusados. Coordenadas absolutas não são inventadas: o adapter registra metadados de patch e usa NaN para geografia desconhecida.

### 18.3 Configurações e composição Hydra

```bash
python scripts/18_install_nvidia_smoke_configs.py --corrdiff-dir "$CORRDIFF_DIR"   --adapter-path "$REPO/adapter/rionowcast_corrdiff_dataset.py"   --dataset-dir "$DATASET" --phase15-dir "$A/15_formal_splits"   --phase17-dir "$A/17_conditioning_ablations" --stats-dir "$P18/conditioning_stats"   --smoke-output-root "$P18/native_smoke"   --training-duration 64 --batch-size-per-gpu 2
for EXP in C00 C07; do
  python scripts/18_run_native_smoke.py --corrdiff-dir "$CORRDIFF_DIR"     --experiment-id "$EXP" --mode compose     --output "$P18/native_smoke_audit_${EXP}_compose.json"
done
```

São quatro arquivos novos de configuração no worktree NVIDIA. O smoke usa modelo mini, fp32, batch 2, uma GPU e orçamento de 64 amostras processadas, não 64 épocas. Composição bem-sucedida não comprova treino.

### 18.4 Regressão, difusão residual e geração

```bash
for EXP in C00 C07; do
  python scripts/18_run_native_smoke.py --corrdiff-dir "$CORRDIFF_DIR"     --experiment-id "$EXP" --mode regression --seed 42     --output "$P18/native_smoke_audit_${EXP}_regression.json"
  REG_CKPT=$(python - "$P18/native_smoke_audit_${EXP}_regression.json" <<'PYCKPT'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='PASS', x
assert len(x['checkpoints'])==1, x['checkpoints']
print(x['checkpoints'][0])
PYCKPT
)
  python scripts/18_run_native_smoke.py --corrdiff-dir "$CORRDIFF_DIR"     --experiment-id "$EXP" --mode diffusion --seed 42     --regression-checkpoint "$REG_CKPT"     --output "$P18/native_smoke_audit_${EXP}_diffusion.json"
  python scripts/18_smoke_generation.py --corrdiff-dir "$CORRDIFF_DIR"     --experiment-id "$EXP"     --regression-audit "$P18/native_smoke_audit_${EXP}_regression.json"     --diffusion-audit "$P18/native_smoke_audit_${EXP}_diffusion.json"     --output-dir "$P18"
done
python scripts/18_summarize_integration.py --phase18-dir "$P18"   --output "$A/summary_phase18.txt"
```

Mantenha todos os JSONs acima no mesmo P18: o runner procura a auditoria do adapter ao lado do arquivo de auditoria nativa. Saídas/checkpoints de outra execução não devem ser reutilizados. O runner recusa diretório de treino não vazio e checkout NVIDIA divergente/modificado. Se houver falha, preserve a evidência e use nova raiz de smoke/configuração para a repetição.

O fechamento exige PASS de checkout, estatísticas train-only, adapter/máscaras, recusa dos testes, composição/regressão/difusão/geração de C00 e C07, fingerprints e evidência de seeds. Falta de GPU/evidência resulta em BLOCKED, não em sucesso parcial. Seeds 43/44 são suportadas, mas não são necessárias para este smoke inicial; exigem execuções isoladas se forem usadas.

## 11. Resumos das fases e critérios de encerramento

```bash
python scripts/summarize_phase0_4.py --analysis-root "$A" --output "$A/summary_phase0_4.txt"
python scripts/05_summarize_seasonality.py --phase5-dir "$A/05_seasonality" --output "$A/summary_phase5.txt"
python scripts/06_summarize_diurnal_cycle.py --phase6-dir "$A/06_diurnal" --output "$A/summary_phase6.txt"
python scripts/07_summarize_temporal_lags.py --phase7-dir "$A/07_lags" --output "$A/summary_phase7.txt"
python scripts/08_summarize_persistence_baselines.py --phase8-dir "$A/08_persistence_baselines" --output "$A/summary_phase8.txt"
python scripts/09_summarize_spatial_structure.py --phase9-dir "$A/09_spatial_structure" --output "$A/summary_phase9.txt"
python scripts/10_summarize_spatial_era5_radar.py --phase10-dir "$A/10_spatial_era5_radar" --output "$A/summary_phase10.txt"
python scripts/11_summarize_spatial_scales.py --phase11-dir "$A/11_spatial_scales" --output "$A/summary_phase11.txt"
python scripts/12_summarize_multivariate_analysis.py --phase12-dir "$A/12_multivariate" --output "$A/summary_phase12.txt"
python scripts/13_summarize_meteorological_regimes.py --phase13-dir "$A/13_regimes" --output "$A/summary_phase13.txt"
python scripts/14_summarize_distribution_shift.py --phase14-dir "$A/14_distribution_shift" --output "$A/summary_phase14.txt"
python scripts/15_summarize_formal_splits.py --phase15-dir "$A/15_formal_splits" --output "$A/summary_phase15.txt"
```

Antes de aceitar a recuperação, registre: comparação estrutural/temporal, tabelas divergentes/ausentes, igualdade ou não dos hashes de target, limitações para input/mask, parâmetros efetivos, versões, fontes e qualquer diferença deliberada. Se os dados mudaram, não misture métricas antigas e novas sob uma única versão.

As análises exploratórias 0–14 descrevem todo o período e não constituem seleção formal com teste cego. O protocolo 15–18 mantém testes bloqueados para model selection. Normalização de builder é descritiva global; normalização formal deve ser train-only. Regimes exploratórios e features não podem ser transferidos para um modelo supervisionado como se tivessem sido ajustados apenas no treino sem conferir isso.

Limitações científicas persistentes: radar derivado de imagem e legenda, incerteza de unidade/calibração, resolução nominal da grade, interpolação de ERA5, ausência de horários, seleção condicionada à disponibilidade, forte massa de zeros, overlap de patches e perda das bordas. Patches do mesmo timestamp não são observações independentes. O acoplamento ERA5(t)→radar(t) é reconstrução/downscaling contemporâneo; não demonstra previsão operacional com antecedência.

Após fase 18 PASS, a infraestrutura está apta a planejar treino completo de regressão e difusão e a campanha de ablações. PASS do smoke não demonstra qualidade meteorológica nem conclusão dessas campanhas.

## 12. Proveniência e manutenção

Fontes do projeto examinadas:

- [Repositório no commit documentado](https://github.com/AILAB-CEFET-RJ/rionowcastdf/tree/0bd42394e461ae233bdd6fcc9076fcea00f8208b).
- `src/corrdiff/builder/dataset_builder.py`, `src/config/paths.py` e requirements.
- Scripts `00_` a `18_`, incluindo os baselines clássicos e `phase16_common.py`.
- `adapter/conditioning.py`, `adapter/rionowcast_corrdiff_dataset.py` e contratos `configs/phase17_*.json`, `configs/phase18_contract.json`.
- Resumo histórico da fase 0 e resumos versionados das fases seguintes.
- `guides/README_PHASE0.md` a `README_PHASE18.md`; em divergências, este roteiro privilegia os argumentos e comportamento do código examinado.

Os comandos foram conferidos contra os parsers do código; não foram executados com os dados remotos nem com GPU. Mudanças posteriores no repositório exigem revisar este documento. Guarde README, session.env, logs, comparação e hashes junto ao snapshot final e em backup independente.
