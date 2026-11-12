# RioNowCastDF — ERA5, radar Sumaré e CorrDiff

**Pesquisa de mestrado · CEFET-RJ · estado documentado em 12/11/2026**

Repositório científico para construir, auditar, caracterizar e modelar uma referência pareada entre **reanálise ERA5** e **produto gráfico de refletividade do radar Sumaré**. A linha experimental utiliza regressão e modelos de difusão da família **CorrDiff / NVIDIA PhysicsNeMo**.

> **Escopo científico.** O pareamento é contemporâneo: variáveis ERA5 no instante de referência condicionam um campo radar associado a esse instante, sujeito às regras documentadas de pareamento temporal. A tarefa principal é **geração/downscaling espacial condicionado entre modalidades meteorológicas**. Esta implementação não demonstra, por si só, previsão futura (*nowcasting*). Tampouco converte os códigos de refletividade radar diretamente em taxas de precipitação.

> **Estado do projeto.** A referência Zarr foi construída e aprovada em auditoria técnica integral; os diagnósticos de referência 0–17 possuem execuções concluídas nos escopos explicitados nos seus relatórios; o ambiente computacional CUDA 12 foi reconstruído a partir do bundle `reference_cu129_v6`, instalado sem reparação automática e aprovado no Doctor (11/11 verificações). **Os smoke tests nativos C00/C07 e os treinamentos completos ainda não estão concluídos.** `production_ready=false` e `spatial_validity_confirmed=false` continuam sendo limitações explícitas, não campos a editar manualmente.

## Sumário

1. [Situação e evidências](#1-situação-e-evidências)
2. [Questão científica e metodologia](#2-questão-científica-e-metodologia)
3. [Fontes, inventário e contratos](#3-fontes-inventário-e-contratos)
4. [Construção da referência Zarr](#4-construção-da-referência-zarr)
5. [Auditoria integral e diagnósticos 0–17](#5-auditoria-integral-e-diagnósticos-017)
6. [Integração CorrDiff e ambiente reproduzível — Fase 18](#6-integração-corrdiff-e-ambiente-reproduzível--fase-18)
7. [Plano de experimentos C00–C10](#7-plano-de-experimentos-c00c10)
8. [Organização do repositório e política de artefatos](#8-organização-do-repositório-e-política-de-artefatos)
9. [Reprodução e verificações](#9-reprodução-e-verificações)
10. [Limitações, pendências e referências internas](#10-limitações-pendências-e-referências-internas)

## 1. Situação e evidências

**Base consultada:** branch `chore/reorganizar-estrutura`, commit `7029e758e4366c00fcbb7e7f0e1457af50cb581c`. Os resultados computacionais mais recentes da máquina `librae` são registros externos ao Git; o bundle completo e o ambiente virtual **não** devem ser commitados.

| Entrega | Estado verificado | Evidência principal |
|---|---|---|
| Seleção temporal e manifesto congelado (14p) | Concluídos | `analysis_outputs_restored/14p_temporal_contract_v1/` |
| Cache compacto e quatro Zarrs de referência | Construídos | `datasets/corrdiff_reference_v1/summary.json`, `*.complete.json` (armazenamento externo ao Git) |
| Auditoria integral da referência | **PASS** | `analysis_outputs_reference/integrity_full_v1/audit/summary.json` |
| Leitor compartilhado | **PASS** | `analysis_outputs_reference/shared_reader_v2/summary.json` |
| Diagnósticos 0–17 da referência | **PASS por fase, com diferentes escopos** | `analysis_outputs_reference/phases_v1/<fase>/summary.json` |
| Bundle CUDA 12 `reference_cu129_v6` | **Preflight PASS** | `manifest.json`, `requirements.lock`, `preflight.log`, `cuda-target-exclusions.json` |
| Instalação isolada `corrdiff_reference_reproduced_cu129_v4` | **PASS, sem auto-repair** | `reference_installation.json`, `reference_native_audit.json`, `install_cu129_v4.log` |
| Doctor (duas GPUs) | **11/11 PASS** | `corrdiff_environment_bundles/doctor_reproduced_cu129_v4.json` |
| Treino nativo C00/C07 — smoke | **Pendente** | Exigir evidências de forward/backward, perdas, checkpoint e mascaramento |
| Treinos completos / avaliação final | **Não executados** | Fases experimentais posteriores |

A aprovação do Doctor confirma integridade do ambiente, imports, operações em GPU, Warp, contrato do leitor e testes automatizados. **Não comprova qualidade preditiva, convergência de CorrDiff ou execução de `train.py` sobre todos os dados.**

### Volume e partições do dataset de referência

| Partição | Período de referência | Horários | Patches |
|---|---|---:|---:|
| `train` | 2011–2020 | 75.319 | 3.615.312 |
| `validation` | 2021–2022 | 14.425 | 692.400 |
| `test_primary` | 2023 | 4.931 | 236.688 |
| `test_stress_ood` | 2024 | 4.423 | 212.304 |
| **Total** | **2011–2024, horários selecionados** | **99.098** | **4.756.704** |

Há **48 patches por horário**. As janelas temporais possuem **purga de 48 horas de cada lado** das fronteiras definidas pelo contrato. Os testes de 2023/2024 são bloqueados para ajuste e seleção de modelos; auditorias descritivas e de metadados podem ter escopos próprios, sem que isso autorize usar os alvos na otimização. Conferir datas UTC exatas em `analysis_outputs_reference/phases_v1/15_split_audit/summary.json`.

## 2. Questão científica e metodologia

O problema pode ser descrito conceitualmente como estimativa da distribuição condicional

\[
p(R_t\mid E_t),
\]

em que `E_t` é um conjunto de 12 variáveis atmosféricas do ERA5 no instante de referência e `R_t` é o código de refletividade radar pareado. A hipótese experimental é que a combinação regressão + difusão possa representar estrutura espacial e variabilidade condicionada não capturadas por uma predição determinística isolada. **Essa hipótese ainda requer avaliação experimental**.

Cadeia metodológica:

```text
Acervo radar PNG + ERA5 (Parquet)
  -> inventário, verificação de fonte e decisões do produtor (14b–14n)
  -> contrato espacial/numérico e geometria (14o)
  -> pareamento/seleção temporal e partições congeladas (14p)
  -> cache radar por conteúdo + Zarrs por split
  -> auditoria independente integral da referência
  -> diagnósticos estatísticos e baselines (0–17)
  -> ambiente PhysicsNeMo/CUDA reproduzível (Fase 18: infraestrutura PASS)
  -> smoke nativo C00/C07 (pendente)
  -> treinamento experimental + avaliação (pendente)
```

A **unidade documental** é o arquivo PNG; a **unidade temporal** é o horário ERA5–radar pareado; a **amostra computacional** é o patch, identificado pelo horário e sua origem na grade. Patches espacialmente sobrepostos e horários meteorologicamente correlacionados **não constituem observações independentes**. Relatórios estatísticos devem declarar se os denominadores são pixels únicos/hora, patches ou eventos.

### Fases de investigação do acervo e congelamento da referência

- **14b–14c:** revisão de extremos e auditoria da proveniência das imagens.
- **14d–14g:** qualidade, reconstrução e análises de impacto de regras anteriores; **históricas**, não o contrato vigente.
- **14h–14i:** verificações espaciais e de alinhamento entre imagem, grade e horários.
- **14j:** decodificação/cache intermediário e geometria documentada.
- **14k–14l:** inventário ERA5, pareamento e smoke técnico de fontes.
- **14m:** inventário integral dos PNGs e classificação de casos ambíguos.
- **14n:** política de codificação e suporte com evidências do produtor.
- **14o:** contrato espacial, numérico e testes de compatibilidade.
- **14p:** contrato temporal, seleção congelada e separação treino/validação/testes.

Arquivos de `pipelines/dataset_preparation/` e contratos relevantes em `analysis_outputs_restored/` documentam essas decisões. **Não remover ou reescrever indiscriminadamente esses artefatos**, pois alguns são entradas do builder e outros estabelecem rastreabilidade científica.

## 3. Fontes, inventário e contratos

### 3.1 ERA5

As entradas são provenientes da reanálise ERA5, em arquivos Parquet locais, com variáveis de superfície e de níveis de pressão. A grade original possui resolução nominal de 0,25°. A referência usa estes 12 canais, nesta ordem imutável:

| Índice | Canal | Descrição |
|---:|---|---|
| 0 | `tcwv` | Água precipitável total da coluna |
| 1 | `t2m` | Temperatura a 2 m |
| 2 | `u10` | Vento zonal a 10 m |
| 3 | `v10` | Vento meridional a 10 m |
| 4 | `t_850` | Temperatura em 850 hPa |
| 5 | `r_850` | Umidade relativa em 850 hPa |
| 6 | `u_850` | Vento zonal em 850 hPa |
| 7 | `v_850` | Vento meridional em 850 hPa |
| 8 | `t_500` | Temperatura em 500 hPa |
| 9 | `r_500` | Umidade relativa em 500 hPa |
| 10 | `u_500` | Vento zonal em 500 hPa |
| 11 | `v_500` | Vento meridional em 500 hPa |

Os campos são interpolados para a grade de referência por **interpolação bilinear, sem extrapolação**. Não há agregação temporal para transformar entrada horária em previsão futura. As estatísticas de padronização são calculadas **somente em treino**, considerando cada horário e ponto da grade uma vez, antes de repetir os pontos em patches sobrepostos.

### 3.2 Radar Sumaré

A origem é um **produto gráfico PNG** de refletividade, não uma matriz bruta de medições instrumentais. A decodificação respeita uma paleta discreta de sete classes: **20, 25, 30, 35, 40, 45 e 50 dBZ**. O código adicional `0` representa **ausência de eco exibido**, quando válida segundo a política do produtor, e **não** uma observação física de refletividade de 0 dBZ nem necessariamente precipitação nula.

O pixel apresenta três estados lógicos:

1. eco exibido, com código de classe;
2. ausência de eco exibido, com suporte válido;
3. pixel desconhecido ou fora do suporte, que **não participa da perda/métrica**.

A máscara preserva a distinção. Um `target` numericamente igual a zero com máscara falsa é apenas um preenchimento; não pode ser agregado como dado válido. Imagens cartográficas/cores ambíguas não são automaticamente interpretadas como ausência de precipitação. As exceções aprovadas estão no manifesto e na política 14n/14p.

**A representação oficial atual do alvo é `target_code / 50`**, com valores discretos em `[0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`. **Não aplicar `log1p`, `expm1` ou uma segunda divisão por 50** aos tensores da referência. Documentos e scripts da geração anterior que usavam `log1p` não devem ser tratados como compatíveis sem revisão.

### 3.3 Domínio, projeção e patches

- Grade de referência: **70 × 84**, aproximadamente 2 km por célula, entre latitude inicial `-23.5` e longitude inicial `-44.0`, com incremento angular `2/111` grau; limites superiores exclusivos.
- Referência geométrica do produto radar: projeção azimutal equidistante esférica, centro aproximadamente `(-22.955139°, -43.248278°)`, raio de exibição de cerca de **138,9 km**, conforme geometria contratada.
- Patches: **32 × 32**, stride **8**, 6 origens de linha (`0,8,16,24,32,38`) × 8 origens de coluna (`0,8,16,24,32,40,48,52`), total de **48** patches por horário, sem padding.
- A grade pode ser completamente coberta geometricamente pelos patches, mas a **cobertura física/instrumental** continua marcada como `provisional_geometric_region` / `spatial_validity_confirmed=false`.

Essa referência **não** deve ser confundida com o dataset histórico `corrdiff_2011_2024`, que utilizava outra geometria de extração/transformação do alvo.

### 3.4 Pareamento temporal

O manifesto selecionado associa horário de referência ERA5 UTC, horário real da fonte PNG, `offset_seconds`, hash de conteúdo e split. Os nomes históricos dos PNGs usam horário local `America/Sao_Paulo`, incluindo as transições de horário de verão. A seleção não deve ser reexecutada por aproximação ou indexação aleatória: **utilizar o manifesto congelado**. Não preencher com zeros observações sem fonte válida.

## 4. Construção da referência Zarr

### 4.1 Ambientes distintos

Existem **dois ambientes de propósito diferente**:

- **Construção e auditoria Zarr:** `requirements_reference_zarr.txt`, com dependências compatíveis com **Zarr 2.x**. Não misturar com o venv de treinamento.
- **Integração/treinamento PhysicsNeMo:** ambiente reproduzido `corrdiff_reference_reproduced_cu129_v4`, a partir do bundle v6; nesse ambiente, **Zarr 3.1.6** foi validado no leitor de referência. Não executar um `pip install -r requirements_phase18.txt` nesse venv, porque o arquivo ainda contém restrições herdadas de Zarr 2.

Os dados não precisam ser reconstruídos para iniciar o treinamento. Os procedimentos abaixo servem para reprodução/auditoria, não para sobrescrever o produto congelado.

### 4.2 Entradas e arquivos protegidos

Configuração oficial: `configs/corrdiff_reference_zarr.json`. Na máquina `librae`:

```bash
cd "$HOME/repositories/rionowcastdf"
export RADAR_ROOT="$HOME/data/radar_sumare"
export ERA5_ROOT="$HOME/data/reanalysis/cds/era5"
```

Entradas contratadas:

| Configuração | Proveniência |
|---|---|
| `temporal_dir` | `analysis_outputs_restored/14p_temporal_contract_v1` — `selected_manifest.parquet`, contratos e resumos |
| `geometry_file` | `data/radar_intermediate_14j_v1/geometry.npz` |
| `coverage_file` | `analysis_outputs_restored/14n_policy_v1/coverage.npz` |
| `spatial_contract_file` | `analysis_outputs_restored/14o_prepared_s8_v1/contract.json` |
| `pairing_dir` | `analysis_outputs_restored/14k_pairing_v2` |
| `reuse_14j_dir` | Cache histórico 14j para reuso por SHA-256 |
| `radar_cache_dir` | `data/radar_reference_v1/` |
| `output_dir` | `datasets/corrdiff_reference_v1/` |

Os arquivos citados podem estar ignorados no Git por tamanho, mas são obrigatórios para reconstrução. Não interpretar ausência no GitHub como inexistência no armazenamento do CEFET. O `project_root` e os caminhos do JSON controlam sua resolução.

### 4.3 Construir o cache radar

No ambiente do builder:

```bash
python3 -m venv .venv-zarr-reference
source .venv-zarr-reference/bin/activate
python -m pip install -r requirements_reference_zarr.txt
python -m unittest discover -s tests -p 'test_reference_zarr.py' -v

mkdir -p logs
nohup python -u src/corrdiff/builder/radar_cache_builder.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  > logs/reference_radar_cache.log 2>&1 < /dev/null &
```

Acompanhar com `tail -f logs/reference_radar_cache.log`. O cache compacto deduplica arquivos pelo hash de conteúdo, verifica reuso da 14j e preserva as decisões do manifesto. Para uma execução interrompida, `--resume` exige entradas e contrato idênticos.

### 4.4 Construir os quatro splits

```bash
python -u src/corrdiff/builder/dataset_builder.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  --check-only

nohup python -u src/corrdiff/builder/dataset_builder.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  > logs/reference_zarr.log 2>&1 < /dev/null &
```

A configuração real utiliza **patch_stride=8**. A retomada utiliza `--resume`; meses de staging completos podem ser reaproveitados. Não executar dois builders sobre o mesmo `output_dir`. Alteração de canal, contrato, seleção temporal ou fonte exige uma nova identidade/saída; **não sobrescrever a referência auditada**.

### 4.5 Saídas e contrato dos tensores

Em `datasets/corrdiff_reference_v1/`:

```text
datasets/corrdiff_reference_v1/
├── train.zarr/
├── validation.zarr/
├── test_primary.zarr/
├── test_stress_ood.zarr/
├── summary.json
├── contract.json
├── selected_manifest.parquet
├── normalization.json
├── normalization.npz
├── era5_identity.json
├── *.complete.json
└── staging/                 # preservar enquanto necessário à auditoria/retomada
```

| Array | Formato | Interpretação |
|---|---|---|
| `input` | `[N,12,32,32]`, float32 | ERA5 interpolado, em unidades originais no Zarr |
| `target` | `[N,1,32,32]`, float32 | Código radar / 50 |
| `mask` | `[N,1,32,32]` | Suporte válido da observação/alvo |
| `timestamps`, `source_timestamps` | `[N]` | UTC da referência ERA5 e do PNG selecionado |
| `offset_seconds` | `[N]` | Deslocamento entre fontes |
| `patch_row`, `patch_col` | `[N]` | Origem espacial do patch |
| `source_sha256` | `[N]` | Identidade de conteúdo do PNG |
| `grid_lat`, `grid_lon`, `grid_support` | Grade completa | Geometria/suporte contratado |

O **loader** aplica uma vez a padronização ERA5 `(input - mean_train) / std_train`; o alvo já sai na escala numérica contratada. A interpretação precisa usar a máscara. Exemplo no ambiente de leitura compatível:

```python
from src.corrdiff.builder.contract_dataset import ReferenceDataset

ds = ReferenceDataset('datasets/corrdiff_reference_v1', split='train')
sample = ds[0]
print(len(ds), sample['input'].shape, sample['target'].shape, sample['mask'].sum())
```

A referência em Zarr ocupa armazenamento fora do Git. Metadados e hashes pequenos podem ser versionados. Para transportar a referência pronta à máquina de treino, preservam-se splits, manifestos, normalização, contrato e identidades; `staging/` pode ser dispensado **somente em uma cópia exclusiva para inferência/treino** depois de manter a cópia integral exigida pela auditoria.

## 5. Auditoria integral e diagnósticos 0–17

### 5.1 Auditoria integral independente

A auditoria é executada por `audit_reference_zarr.py` e o resultado de referência encontra-se em `analysis_outputs_reference/integrity_full_v1/audit/summary.json`:

- `complete=true` e **`integrity_passed=true`**;
- **99.098** horários de fontes conferidos;
- todos os tensores de patches comparados com `staging` e os alvos/máscaras com o cache;
- `era5_raw_recomputed=true` e `normalization_recomputed=true`;
- caminhos PNG selecionados verificados; **75.682** conteúdos únicos selecionados e decodificados;
- `coverage_state=provisional_geometric_region` e `spatial_validity_confirmed=false`.

A verificação comprova consistência com contratos e decodificador/interpolador adotados; **não é validação independente da cobertura física ou da meteorologia instrumental**.

Para reproduzir, no ambiente do builder e com todas as fontes/caches presentes:

```bash
nohup python -u audit_reference_zarr.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  --output-dir analysis_outputs_restored/reference_integrity_nova_execucao \
  > logs/reference_integrity_nova_execucao.log 2>&1 < /dev/null &
```

Utilizar novo diretório de saída; nunca sobrescrever resultados auditados. O escopo exato da auditoria depende das opções de execução registradas no resumo.

### 5.2 Diagnósticos da referência

Os resumos `analysis_outputs_reference/phases_v1/` reportam `complete=true` e `passed=true` em **todas as fases 0–17**. As bases de cálculo diferem; alguns diagnósticos usam amostras apenas de treino, e a fase 17 **planeja** experimentos sem treiná-los.

| Fase | Finalidade | Escopo declarado na referência |
|---|---|---|
| 0 | Estrutura, qualidade, cobertura e normalização | Varredura de **99.098 horas** e tensores de referência |
| 1 | Estatísticas univariadas | Amostras de treino, pontos/horários únicos |
| 2 | Variáveis derivadas | Amostras de horas/grade de treino |
| 3 | Relações conjuntas | Médias espaciais horárias de treino; análise descritiva |
| 4 | Extremos | Limiares fixos da paleta, denominadores por hora/pixel |
| 5 | Sazonalidade | Horários observados e sua disponibilidade |
| 6 | Ciclo diurno | Horários observados e sua disponibilidade |
| 7 | Defasagens | Junção por UTC exato, sem atravessar lacunas |
| 8 | Persistência | Pares exatos de treino/validação, amostrados e mascarados |
| 9 | Estrutura espacial | Campos amostrados de treino; componentes/variogramas |
| 10 | Relações espaciais ERA5–radar | Correlações amostrais no treino, por ponto da grade |
| 11 | Escalas espaciais | Blocos não sobrepostos, médias ponderadas pela máscara |
| 12 | Análise multivariada | PCA ajustada somente no treino |
| 13 | Regimes meteorológicos | KMeans ajustado em treino e aplicado sem reajuste |
| 14 | Mudanças de distribuição | Diagnóstico descritivo entre partições, sem ajuste no teste |
| 15 | Auditoria formal de partições | Todos os manifestos e metadados dos patches; purga de 48 h |
| 16 | Baselines clássicos | Ajuste amostral em treino; avaliação treino/validação, sem CorrDiff |
| 17 | Matriz de ablação | Planejamento de **66 execuções**; `training_executed=false` |

**Concluir uma fase de diagnóstico não equivale a executar todos os treinamentos por ela planejados.** Resultados do antigo `analysis_outputs/` ou `analysis_outputs_restored/` podem pertencer a outra versão/escala do dataset; verificar identidade e transformação antes de citá-los na dissertação.

### 5.3 Princípios estatísticos

- Estatísticas descritivas não devem ser confundidas com inferência causal.
- Tabelas por patch precisam indicar multiplicidade de cobertura; resultados principais devem privilegiar horários e pixels únicos quando o objetivo exigir independência aproximada.
- Eventos extremos são raros e dependem da paleta e da máscara. Não extrapolar quantitativamente para intensidade contínua ou chuva em mm/h.
- Ajustar normalização, seleção de atributos e modelos somente em treino; validação orienta escolhas; testes são reservados ao relatório final.
- Registrar denominador, unidade, intervalo temporal, tamanho amostral efetivo, versão de contrato e incerteza para interpretações meteorológicas.

## 6. Integração CorrDiff e ambiente reproduzível — Fase 18

### 6.1 Objetivo e situação

A Fase 18 cobre tanto a preparação de software quanto a integração nativa. **A preparação de software está concluída e aprovada; os smoke tests nativos de regressão/difusão ainda estão pendentes**.

A reconstrução do ambiente foi necessária porque o snapshot Python inicial misturava pacotes que gravavam os mesmos caminhos `nvidia/cudnn/lib/*.so` (`cu12` e `cu13`). Houve também correções de seleção de wheel CUDA para Warp, de resolução `tqdm` e `tensordict`/Zarr. O bundle v6 projeta explicitamente a família CUDA 12, registra exclusões, valida hashes e exige `pip --dry-run --no-index --require-hashes`. Uma instalação nova com `--no-auto-repair` passou na auditoria de integridade de binários.

**Referências operacionais no CEFET:**

```text
~/corrdiff_environment_bundles/reference_cu129_v6/
~/corrdiff_environment_bundles/reference_cu129_v6_finalize_cu12.log
~/corrdiff_environment_bundles/install_cu129_v4.log
~/corrdiff_environment_bundles/doctor_reproduced_cu129_v4.json
~/corrdiff_reference_reproduced_cu129_v4/
~/repositories/physicsnemo/
```

| Componente | Identidade validada |
|---|---|
| Sistema-alvo | Linux x86_64, Python 3.12 |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti (SM 7.5, ~11 GiB cada) |
| PyTorch | `2.13.0+cu129` |
| PhysicsNeMo | `2.3.0a0` — checkout esperado `c15bd23eb56216c32bd3e30180593e5685be9bc2` |
| CUDA | Wheels PyTorch CUDA 12.9; driver NVIDIA compatível, ver Doctor |
| cuDNN | `nvidia-cudnn-cu12==9.20.0.48` |
| Warp | `1.19.0+cu12` |
| Zarr (leitura e treinamento) | `3.1.6`, validado contra amostras da referência |
| Tensordict | `0.15.5` |

**Resultado do Doctor no servidor:** `passed=true`, verificações `checkout_clean`, `dependencies`, `native_imports`, `gpu_fp32`, `runtime`, `dataset_contract`, `installed_commit_receipt`, `native_wheel_integrity`, `warp_cuda`, `test_reference_adapter.py` e `test_reference_masked_losses.py` aprovadas (11/11).

### 6.2 Como inspecionar a instalação aprovada

```bash
cd "$HOME/repositories/rionowcastdf"
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
BUNDLE="$HOME/corrdiff_environment_bundles/reference_cu129_v6"

"$PYTHON" -m pip check
"$PYTHON" -m unittest discover -s tests -p 'test_reference_environment.py' -v
ls -lh "$BUNDLE/manifest.json" "$BUNDLE/requirements.lock" "$BUNDLE/preflight.log"
```

Para refazer o Doctor **sem alterar o ambiente**:

```bash
nohup "$PYTHON" -u scripts/reference_environment.py doctor \
  --project-dir "$HOME/repositories/rionowcastdf" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --gpus 0,1 \
  --output "$HOME/corrdiff_environment_bundles/doctor_recheck.json" \
  > "$HOME/corrdiff_environment_bundles/doctor_recheck.log" 2>&1 < /dev/null &
```

Para reconstruir uma **nova** instalação a partir do bundle já pronto, usar `reference_environment.py install` com `--bundle` apontando para v6, `--venv` apontando para um diretório inexistente e `--no-auto-repair`. Nunca instalar por cima do v4 aprovado.

### 6.3 Pendências bloqueantes da integração nativa

**Atenção — a branch contém duas gerações do contrato.** Antes de usar os scripts `18_*` para smoke nativo, reconciliar explicitamente:

1. `configs/phase18_contract.json` ainda informa alvo `radar_reflectivity_log1p` e `stored_space=log1p(clipped radar dBZ)`, além de um `nvidia_commit` antigo (`beafeec...`). **Esses valores não descrevem a referência validada.** O checkout do Doctor usa `c15bd23...` e o alvo é `code/50`.
2. `scripts/18_prepare_conditioning_stats.py` e partes da integração `18_*` ainda referenciam `datasets/corrdiff_2011_2024`, índices anteriores ou `log1p`. É necessário migrar esses acessos para os quatro Zarrs atuais e as estatísticas de treino do contrato vigente, com testes.
3. `adapter/reference_corrdiff_dataset.py` é leitor **mask-aware** para os 12 canais originais, retornando um dicionário, não a tupla esperada pela API legada NVIDIA. Não equivale a integração nativa aprovada. O C07 requer oito canais derivados e normalização exclusiva de treino.
4. A função de perda do treinamento NVIDIA deve respeitar `mask`, e qualquer escolha de filtrar patches deve ter impacto de seleção explicitado. Não usar perda não mascarada em observações com pixels desconhecidos.
5. O modo `compose` de Hydra não prova `train.py`: exigir uma execução real curta por estágio, com logs de perdas finitas, gradientes, checkpoints, seed e registros de GPU.

**Não alterar contratos/artefatos congelados só para fazer um smoke passar.** Migrar adaptadores/entrypoints para o contrato de referência e deixar auditável a diferença entre a geração legada e a vigente.

## 7. Plano de experimentos C00–C10

A matriz de planejamento está em `configs/phase17_experiments.json` e as definições de canais em `configs/phase17_channel_catalog.json`.

- **C00:** 12 variáveis originais; experimento de referência.
- **C01–C03:** suficiência de um grupo de variáveis (umidade, térmicas ou dinâmica).
- **C04–C06:** necessidade de grupos, retirando um deles de cada vez.
- **C07:** 12 originais + **8 derivadas** (20 canais), comparação de representação.
- **C08–C10:** 12 originais + famílias específicas de canais derivados.

Derivadas: velocidades do vento em 10 m/850/500 hPa, diferenças de temperatura em níveis, contraste de umidade relativa e magnitudes de diferenças vetoriais de vento. São **funções determinísticas** das variáveis originais: podem ajudar a representação/aprendizado, mas não acrescentam informação observacional nova. Diferenças vetoriais entre níveis não são necessariamente cisalhamento normalizado por altitude, e diferenças de temperatura não devem ser chamadas automaticamente de instabilidade.

A Fase 17 registra **66 execuções planejadas**, não resultados de treinamento. A sequência recomendada é: regressão C00 → difusão C00 → integração/validação de 20 canais C07 → smoke C07 → campanha completa e baselines, respeitando capacidade das GPUs e orçamento computacional. Nos smoke tests, usar apenas `train` e `validation`; não liberar splits de teste por conveniência. Repetições com sementes controladas não substituem agregação estatística por tempo/evento.

### Avaliação científica futura

Comparar regressão, difusão e baselines nas representações compatíveis; reportar métricas mascaradas, propriedades de extremos e distribuição, estrutura espacial, incertezas, variação por horário/estação/regime e custo computacional. Os resultados de teste devem ser usados apenas após congelar modelo, protocolo e seleção de hiperparâmetros. **Não há ainda alegação de superioridade de CorrDiff.**

## 8. Organização do repositório e política de artefatos

| Diretório | Finalidade | Política |
|---|---|---|
| `src/corrdiff/builder/` | Builder contratado de cache/Zarr | Proteger: código participa de hashes congelados |
| `src/corrdiff/diagnostics/` | Leitor e fases da referência | Ativo |
| `adapter/` | Leitor e funções auxiliares CorrDiff | Ativo; interface nativa ainda requer ajuste |
| `configs/` | Contratos e matriz experimental | Ativo; corrigir divergências da Fase 18 em alteração versionada |
| `scripts/` | Auditorias, diagnósticos e entrypoints | Manter executáveis oficiais; não versionar backups pessoais |
| `tests/` | Testes automatizados | Manter |
| `pipelines/dataset_preparation/` | Proveniência e preparação histórica 14b–14p | Preservar; não consolidar sem rastrear dependências |
| `analysis_outputs_reference/` | Resumos dos diagnósticos da referência vigente | Manter pequenos artefatos verificáveis |
| `analysis_outputs_restored/` | Evidências da reconstrução, **incluindo entradas ativas** | Proteger; revisar por arquivo, nunca excluir em lote |
| `analysis_outputs/` | Resultados históricos | Arquivar ou retirar binários do índice após backup |
| `exports/` | Cópias para dissertação/intercâmbio | Evitar ZIP e extração duplicados no Git após backup |
| `experimental/` | Investigações anteriores (inclui dados brutos) | Separar código de grandes dados; manter cópia externa |
| `recovery_runs/`, `recovery_comparison/` | Recuperação/linhagem | Preservar manifestos, hashes, resumos; retirar logs gigantes após backup |
| `docs/`, `figures/`, `notebooks/` | Documentação, ilustrações, exploração | Manter o que sustenta a dissertação; não confundir figuras legadas com novos resultados |
| `data/`, `datasets/`, `logs/` | Armazenamento operacional | Fora do Git; manter no servidor conforme política de retenção |

**Orientações e levantamento de candidatos à limpeza:** [`docs/PLANO_LIMPEZA_REPOSITORIO_2026-11-12.md`](docs/PLANO_LIMPEZA_REPOSITORIO_2026-11-12.md). A remoção de arquivos já rastreados exige `git rm --cached`; `.gitignore` sozinho não os remove e a operação **não reduz o tamanho do histórico Git anterior**. Antes de retirar dados de pesquisa do índice, verificar backup e possibilidade de reconstrução.

## 9. Reprodução e verificações

Roteiro mínimo para um checkout que **já possui os dados externos necessários**:

```bash
cd "$HOME/repositories/rionowcastdf"
git status --short
git rev-parse HEAD

# Contrato do dataset e artefatos de auditoria
test -f datasets/corrdiff_reference_v1/summary.json
test -f analysis_outputs_reference/integrity_full_v1/audit/summary.json

# Suite do ambiente e smoke do leitor/da perda — não é treino nativo
PYTHON="$HOME/corrdiff_reference_reproduced_cu129_v4/bin/python"
"$PYTHON" -m pip check
"$PYTHON" -m unittest discover -s tests -p 'test_reference_adapter.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_reference_masked_losses.py' -v

# Evidência original do Doctor
python - <<'PY'
import json
from pathlib import Path
p = Path.home() / 'corrdiff_environment_bundles/doctor_reproduced_cu129_v4.json'
r = json.loads(p.read_text())
print('DOCTOR:', r['passed'])
for name, ok in r['checks'].items():
    print(name, ok)
PY
```

O comando acima **consulta** evidência, não executa novamente o Doctor. Os testes completos de Zarr e integridade possuem ambiente/dependências separados. A ordem obrigatória para treinamento é: integridade do contrato → alinhamento dos adaptadores → smoke dataset/máscaras → smoke regressão → smoke difusão → campanha completa.

Os artefatos de treino e seus metadados devem registrar commit, hash do contrato, hashes do manifesto e da normalização, seed, versão de PhysicsNeMo, opções de máscara, partições utilizadas, consumo de memória, tempos, métricas e localização dos checkpoints. Preservar o bundle e o relatório Doctor mesmo após limpeza de instalações temporárias.

## 10. Limitações, pendências e referências internas

1. **Validade física do suporte radar:** a geometria ainda é classificada como provisória; auditoria computacional não substitui confirmação instrumental.
2. **Alvo discretizado e ausência de eco:** não usar métricas meteorológicas que pressupõem chuva contínua ou refletividade física de 0 dBZ.
3. **Amostragem dependente e disponibilidade temporal desigual:** analisar correlações, incerteza, eventos extremos e períodos sem cobertura com denominadores explícitos.
4. **Mudança de contrato entre gerações:** `log1p`/dataset legado e `code/50`/referência atual são incompatíveis sem migração. O `phase18_contract.json` e seus consumidores precisam ser reconciliados antes do smoke nativo.
5. **Evidência computacional versus resultado científico:** Doctor PASS e diagnósticos concluídos não comprovam treinamento ou generalização de CorrDiff.
6. **Limite computacional:** duas RTX 2080 Ti (~11 GiB cada), sem assumir memória combinada ou peer-to-peer disponível; planejar batches, checkpoints e tempo com base em execução real.

Referências no próprio repositório:

- [`docs/REPOSITORIO.md`](docs/REPOSITORIO.md): mapa da reorganização e caminhos protegidos.
- [`docs/DADOS_E_RESULTADOS.md`](docs/DADOS_E_RESULTADOS.md): retenção e publicação de evidências.
- [`docs/MIGRACAO_ESTRUTURA.md`](docs/MIGRACAO_ESTRUTURA.md): origem dos módulos movidos.
- [`docs/history/README_RECONSTRUCAO_FASES_00_A_18.md`](docs/history/README_RECONSTRUCAO_FASES_00_A_18.md): histórico técnico, **não** contrato atual de treinamento.
- [`README_FASE18_AMBIENTE_DEFINITIVO.md`](README_FASE18_AMBIENTE_DEFINITIVO.md): história do ambiente e recuperação do bundle v6.
- [`analysis_outputs_reference/`](analysis_outputs_reference/): auditoria e diagnósticos da referência.

**Critério de avanço:** concluir a migração dos scripts/contratos nativos à referência `code/50`, executar os smoke tests C00 e C07 com evidências reais e só então iniciar o treinamento completo dos experimentos. Essa sequência mantém consistente a implementação, o Capítulo 4 (contrato de dados), o Capítulo 5 (diagnósticos) e o Capítulo 6 (metodologia experimental).
