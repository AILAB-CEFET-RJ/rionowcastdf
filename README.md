# RioNowCastDF — ERA5 e radar do Sumaré

Repositório de construção, auditoria e avaliação de um dataset para downscaling meteorológico com regressão e difusão CorrDiff. O condicionamento usa reanálise ERA5 e o alvo usa o produto gráfico de refletividade do radar do Sumaré. O pareamento atual é contemporâneo; ele não estabelece, por si só, um experimento de previsão futura (nowcasting).

Este README é o guia operacional central. Os READMEs internos explicam interfaces específicas; os documentos históricos preservam a rastreabilidade e não substituem este contrato.

## Conteúdo

- [Estado atual](#estado-atual)
- [Mapa dos módulos](#mapa-dos-módulos)
- [Construção e contrato do dataset](#construção-e-contrato-do-dataset)
- [Validação da integridade](#validação-da-integridade)
- [Diagnósticos e integração nativa](#diagnósticos-e-integração-nativa)
- [Preservação e histórico](#preservação-e-histórico)

## Estado atual

A construção de referência foi concluída, segundo o resumo da execução no CEFET. A organização do código e seus testes sintéticos não constituem aprovação científica do dataset.

| Etapa | Situação |
|---|---|
| Auditoria PNG e decisões do produtor | Incorporadas à política; preservar evidências das etapas 14m–14p |
| Cache compacto e Zarr | Construídos; leitura de retorno realizada pelo builder |
| Auditoria independente integral | Programa disponível; resultado integral ainda precisa ser conferido |
| Diagnósticos 0–18 para a referência | Adaptação e reexecução pendentes |
| CorrDiff nativo | Integração e treinamento ainda não certificados |
| Cobertura física do radar | Suporte geométrico provisório; não inferir validade instrumental de toda a grade |

| Conjunto | Horas | Patches |
|---|---:|---:|
| Treino | 75.319 | 3.615.312 |
| Validação | 14.425 | 692.400 |
| Teste principal | 4.931 | 236.688 |
| Teste de estresse/OOD | 4.423 | 212.304 |
| Total | 99.098 | 4.756.704 |

O resultado de referência informa `production_ready=false`, `corrdiff_native_tested=false` e `diagnostics_0_18_rerun=false`. Esses campos devem refletir verificações efetivamente concluídas, sem alteração manual para contornar pendências.

## Mapa dos módulos

Todos os comandos são executados na raiz do checkout. Na máquina local informada: `/home/sangonvi/Cefet/repositories/rionowcastdf`; no CEFET: `/home/vsantos/repositories/rionowcastdf`. A presença do código na máquina local não implica disponibilidade dos PNGs, Parquets, caches ou Zarrs.

| Caminho | Papel e tratamento |
|---|---|
| src/corrdiff/builder/ | Fluxo oficial de cache/Zarr no modo --contract-config. Preservar caminhos e conteúdo durante a auditoria. |
| configs/ | Configurações; corrdiff_reference_zarr.json é a entrada da referência atual. |
| audit_reference_zarr.py | Auditoria integral de tensores, staging e cache; PNGs por amostragem. |
| scripts/ | Diagnósticos 0–18; requerem adaptação ao alvo code/50, máscara e Zarrs separados. |
| adapter/ | Integração existente; ainda precisa de validação nativa com a referência. |
| tests/ | Testes automatizados do builder e da integração; testes locais não substituem auditoria do acervo. |
| notebooks/, experimental/ | Exploração e experimentos; não são o fluxo oficial. |
| docs/documents/, docs/guides/, figures/ | Documentação e figuras existentes. Não atribuir resultados antigos ao novo dataset. |
| docs/history/README_RECONSTRUCAO_FASES_00_A_18.md | Roteiro histórico preservado; não é uma certificação de compatibilidade com o novo Zarr. |

### Preparação e evidências históricas

| Caminho | Função |
|---|---|
| pipelines/dataset_preparation/corrdiff_extreme_audit/, pipelines/dataset_preparation/corrdiff_source_audit/ | Investigação de extremos e rastreabilidade das fontes. |
| pipelines/dataset_preparation/corrdiff_radar_qc/, pipelines/dataset_preparation/14g_cruzar_radar.py | Políticas e cruzamentos anteriores à confirmação do produtor. |
| pipelines/dataset_preparation/verificacao_espacial_14h/, pipelines/dataset_preparation/corrdiff_alignment_14i/ | Inspeções espaciais e temporais; evidências da correção. |
| pipelines/dataset_preparation/corrdiff_cache_14j/ | Cache intermediário decodificado e geometria de referência. |
| pipelines/dataset_preparation/corrdiff_pairing_14k/ | Inventário ERA5 e pareamento. |
| pipelines/dataset_preparation/corrdiff_smoke_14l/ | Teste técnico anterior à política final do produtor. |
| pipelines/dataset_preparation/corrdiff_png_audit_14m/ | Auditoria completa do acervo PNG. |
| pipelines/dataset_preparation/corrdiff_policy_14n/ | Política baseada nas respostas do produtor. |
| pipelines/dataset_preparation/corrdiff_contract_14o/ | Preparação do contrato espacial e numérico. |
| pipelines/dataset_preparation/corrdiff_temporal_14p/ | Seleção temporal e congelamento do manifesto. |
| examples/radar_sumare/amostras_orientador_sumare/ | Exemplos de fonte, incluindo arquivos deliberadamente inválidos para demonstração. |
| recovery_runs/, recovery_comparison/ | Recuperação de versões anteriores, fora do fluxo oficial. |

As etapas 14j–14p não são lixo: algumas produzem entradas ainda usadas e outras documentam decisões científicas. Os módulos foram movidos em conjunto, preservando o conteúdo Python e os imports locais. Consulte `docs/MIGRACAO_ESTRUTURA.md` para os caminhos anteriores.


| Outros caminhos | Responsabilidade |
|---|---|
| `analysis_outputs/` | Resultados anteriores; não representam automaticamente o Zarr atual |
| `analysis_outputs_restored/` | Auditorias da reconstrução e artefatos contratados; vários são entradas ativas |
| `analysis_outputs_reference/` | Área dedicada aos diagnósticos da nova referência, identificando execução e contrato |
| `data/`, `datasets/` | Caches e datasets locais, mantidos fora do Git |
| `logs/` e logs soltos | Registros de execução; não substituem os resumos e manifests |
| `archive/` | Backups e materiais históricos preservados |
| `requirements_reference_zarr.txt` | Ambiente reproduzível de construção/auditoria da referência |
| `requirements_corrdiff_builder.txt`, `requirements_phase18.txt` | Requisitos dos fluxos correspondentes; não misturar ambientes sem verificar versões |
| `reference_builder_existing.patch`, `reference_package_sha256.json` | Materiais da entrega anterior; não reaplicar automaticamente sobre o código atual |
| `Untitled.ipynb`, `download_pressure.txt` | Materiais auxiliares; fora da entrada operacional oficial |


## Construção e contrato do dataset

O modo oficial exige `--contract-config` nos builders existentes; o modo legado não implementa o contrato atual.

### 1. Instalação

Use um checkout atualizado. Dados e manifests volumosos precisam ser disponibilizados separadamente. A referência já construída não precisa ser reconstruída para iniciar sua auditoria.

```bash
cd /home/vsantos/repositories/rionowcastdf
python3 -m venv .venv-zarr-reference
source .venv-zarr-reference/bin/activate
python -m pip install -r requirements_reference_zarr.txt
python -m unittest discover -s tests -p test_reference_zarr.py -v
mkdir -p logs
export RADAR_ROOT=/home/vsantos/data/radar_sumare
export ERA5_ROOT=/home/vsantos/data/reanalysis/cds/era5
```

Python 3.11+ em Linux/WSL; testes executados com Python 3.12. O ambiente de construção é separado do ambiente de treinamento. Não requer GPU, PyTorch ou CUDA para construir o Zarr. Zarr fica na série 2.x; não instalar Zarr 3 neste ambiente. A versão de xarray foi fixada para compatibilidade com esse formato. Congele o ambiente realmente utilizado com `python -m pip freeze > logs/reference_environment.txt` para reproduzir a execução em outras máquinas.

### 2. Conferir as entradas

Edite `configs/corrdiff_reference_zarr.json` caso seus caminhos difiram. Caminhos relativos são resolvidos a partir de `project_root`, relativo à pasta do JSON; variáveis de ambiente são expandidas. `RADAR_ROOT` aponta para a pasta que contém os anos; `ERA5_ROOT` contém `single/` e `pressure/`.

Entradas obrigatórias:

| Entrada | Conteúdo necessário |
|---|---|
| `temporal_dir` | `summary.json`, `temporal_contract.json`, **`selected_manifest.parquet` integral**, da 14p congelada |
| `geometry_file` | `geometry.npz` da 14j |
| `coverage_file` | `coverage.npz` da 14n |
| `spatial_contract_file` | `contract.json` **produzido** em `14o_prepared_s8_v1`, com hashes de `policy_artifacts`; não o modelo genérico do código |
| `pairing_dir` | `run_config.json` e `era5_inventory_snapshot.sqlite` da 14k |
| `radar_dir` | PNGs necessários para hashes que não estão no cache 14j |
| `reuse_14j_dir` | opcional: `summary.json`, `geometry.npz`, `manifest.sqlite` e árvore `decoded/` da 14j |
| `era5_root` | Parquets das duas famílias do ERA5 |

O manifesto congelado contém 99.098 horas: treino 75.319, validação 14.425, teste principal 4.931 e teste OOD 4.423. Estes números são conferidos contra o manifesto utilizado; o builder não seleciona novamente o vizinho temporal. Arquivos excluídos ou ausentes não são preenchidos com zero.

O período inicial já vem limitado a `2011-01-17T02:00:00Z`. O nome do PNG permanece em horário histórico de `America/Sao_Paulo`, incluindo horário de verão. Cada par guarda o horário de referência UTC, o horário real do radar e o deslocamento em segundos.

### 3. Criar o cache compacto

```bash
nohup python -u src/corrdiff/builder/radar_cache_builder.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  > logs/reference_radar_cache.log 2>&1 &
echo "PID: $!"
```

```bash
tail -f logs/reference_radar_cache.log
```

Saída padrão: `data/radar_reference_v1/`. Reutiliza objetos decodificados da 14j por SHA-256, verifica seu checksum, e lê PNGs apenas para os demais hashes. Várias horas com o mesmo conteúdo usam um único objeto. Não altera a 14j, a 14n ou a 14p. Não percorre todo o acervo: utiliza apenas os hashes selecionados.

Para construir sem a 14j, configure `reuse_14j_dir` como `null`; ainda são necessários os pequenos arquivos de geometria e cobertura. Todos os PNGs selecionados deverão estar disponíveis.

Para retomar uma execução interrompida, repita o comando acrescentando `--resume`. Sem essa opção, um diretório de saída existente é rejeitado. A retomada exige contrato, código e entradas idênticos. Mudanças de contrato devem produzir uma nova saída.

### 4. Conferir e construir os Zarrs

Depois de `complete: true` no resumo do cache:

```bash
python -u src/corrdiff/builder/dataset_builder.py \
  --contract-config configs/corrdiff_reference_zarr.json --check-only
```

A pré-verificação confere contratos, manifesto e índice do cache e apresenta volumes. A validação dos valores ERA5 e dos objetos individuais ocorre durante a construção.

```bash
nohup python -u src/corrdiff/builder/dataset_builder.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  > logs/reference_zarr.log 2>&1 &
echo "PID: $!"
```

```bash
tail -f logs/reference_zarr.log
```

Para retomar, acrescente `--resume` ao mesmo comando. O builder verifica e reutiliza meses completos de campos interpolados. Um split Zarr ainda incompleto é regravado a partir desse estágio local; os splits concluídos são verificados e preservados. Não são necessárias novas leituras de valores ERA5 para meses já preparados.

Não execute duas construções sobre a mesma saída: há bloqueio exclusivo. Para outras configurações, copie o JSON e troque `output_dir`. Uma seleção diferente de canais pode usar o mesmo cache compacto se manifesto, período e demais contratos forem iguais. Alteração do período requer uma saída de cache correspondente. Subconjuntos precisam conter horas de treino para estimar a normalização.

### 5. Conteúdo das saídas

Em `datasets/corrdiff_reference_v1/`:

- `train.zarr`, `validation.zarr`, `test_primary.zarr`, `test_stress_ood.zarr`.
- `normalization.json` e `normalization.npz`: estatísticas de entrada estimadas somente no treino selecionado pela 14p.
- `contract.json`: identidade do contrato, código e entradas; `selected_manifest.parquet`: pares usados.
- `era5_identity.json`: checksums de conteúdo dos Parquets lidos e metadados disponíveis das fontes.
- `<split>.complete.json`: contagens e checksums dos arrays efetivamente lidos de volta.
- `summary.json`: conclusão, contagens e limites de validação.
- `staging/YYYYMM/`: arrays horários intermediários para retomada. São mantidos; não apague durante a construção. A retomada pressupõe sua presença.

Cada Zarr tem arrays planos:

| Array | Formato / significado |
|---|---|
| `input` | `[N,C,32,32]`, float32, unidades físicas originais, ainda não padronizado |
| `target` | `[N,1,32,32]`, float32, código do alvo dividido por 50 |
| `mask` | `[N,1,32,32]`, booleano, suporte válido para a perda |
| `timestamps` | `[N]`, segundos Unix UTC da referência ERA5 |
| `source_timestamps` | `[N]`, segundos Unix UTC do PNG escolhido |
| `offset_seconds` | diferença entre PNG e referência |
| `patch_row`, `patch_col` | início do patch na grade 70×84 |
| `source_sha256` | hash do PNG original |
| `grid_lat`, `grid_lon`, `grid_support` | grade completa e suporte herdado |

A grade permanece 70×84, passo angular `2/111` grau, projeção de referência e amostragem do pixel nativo mais próximo. ERA5 usa interpolação bilinear, sem extrapolação. Não há média temporal, conversão para chuva, suavização de cores ou escolha pelo máximo.

Patches: 32×32, passo 8, últimos inícios ancorados em linha 38 e coluna 52. São 48 posições por hora, sem padding, cobrindo a grade. Patches sem qualquer suporte são descartados; patches totalmente sem eco, mas com suporte, são mantidos. Sobreposição aumenta as amostras de treino, mas não cria observações independentes.

O alvo tem códigos `[0,20,25,30,35,40,45,50]` antes de dividir por 50. **Zero é ausência de eco exibido**, não refletividade medida de 0 dBZ nem garantia de precipitação nula. Pixels desconhecidos continuam mascarados; o zero numérico nesses lugares é apenas um placeholder. A cobertura herdada ainda informa `provisional_geometric_region`, sem alegar comprovação física de toda a região.

Cartografia pendente permanece excluída. Somente `reviewed_map_no_echo` já autorizado no manifesto congelado pode virar alvo sem eco, e o programa rejeita esse caso se houver pixels da paleta meteorológica. Cores opacas desconhecidas em imagens normais não se tornam automaticamente “sem chuva”.

Normalização: média e desvio populacional por canal, considerando cada hora de treino e cada ponto da grade uma vez, antes dos patches sobrepostos. O loader aplica `(input - mean) / std` uma única vez. O alvo **já está** dividido por 50 no Zarr; não aplicar `log1p`, uma segunda divisão por 50 ou normalização estimada do alvo. O NPZ registra `target_mean=0`, `target_std=1` por esse motivo.

Exemplo de leitura sem Torch:

```python
from src.corrdiff.builder.contract_dataset import ReferenceDataset

ds = ReferenceDataset('datasets/corrdiff_reference_v1', split='train')
batch = ds[0]  # input já padronizado; target já dividido por 50
print(len(ds), batch['input'].shape, batch['target'].shape, batch['mask'].sum())
```

A perda deve utilizar `mask`. Esta classe demonstra o contrato de leitura; não é um adaptador nativo completo do CorrDiff.

### 6. Espaço, tempo e portabilidade

Com 99.098 horas e 12 canais: até 4.756.704 patches, cerca de **240 GiB** de tensores sem compressão (entrada, alvo e máscara), mais metadados. Os campos horários intermediários ocupam aproximadamente **29 GiB**. Compressão Zstd reduz o volume, mas a taxa depende dos dados. Reserve espaço adicional para tabelas temporárias DuckDB e saída em construção. O programa informa a estimativa para a configuração efetiva.

O tempo será dominado por leitura/interpolação ERA5 e escrita/verificação dos patches, não pelo treino. Não há estimativa confiável antes de observar os primeiros meses. `workers` controla decodificação de PNG; `duckdb_threads` e `duckdb_memory` controlam o leitor ERA5, que pode usar disco temporário. Os tensores não são carregados integralmente em RAM: os campos são preparados em arquivos mapeados por mês e os patches gravados por hora. O manifesto e os índices compactos permanecem em memória.

Outra máquina pode receber:

1. **Zarr pronto:** copie toda a pasta de saída (pode omitir `staging/` e `duckdb_tmp/` em uma cópia exclusiva para treino, após preservar a cópia completa necessária à auditoria e à retomada), código do loader e dependências. Nenhum PNG ou Parquet é necessário para ler/treinar.
2. **Reconstrução:** copie contratos e manifesto congelado, cobertura/geometria, inventário, cache compacto ou PNGs e Parquets. Edite caminhos no JSON. Preserve tamanho e mtime dos Parquets usando `rsync -a`; o inventário original registra esses valores. Depois da primeira construção, `era5_identity.json` pode ser fornecido em `era5_identity_file` para validar cópias por SHA-256 mesmo quando mtime mudou. Sem identidade confiável, não se ignora uma divergência de arquivo: recopie preservando metadados ou refaça o inventário/pareamento.

Não colocar caches, staging e Zarrs no Git. Acrescente ao `.gitignore`, se ainda não houver regras equivalentes:

```gitignore
/data/radar_reference*/
/datasets/corrdiff_reference*/
/.venv-zarr-reference/
```

Isso não remove arquivos já rastreados nem apaga histórico de commits grandes. Publique os resumos pequenos e o código; mantenha os dados volumosos no armazenamento de dados.

### 7. O que enviar ao terminar

Envie o `summary.json` do cache e do dataset, `normalization.json`, e os quatro `<split>.complete.json` (ou suas contagens). O status `complete=true` significa construção e leitura de verificação concluídas. `production_ready=false` permanece deliberado até os diagnósticos científicos e a integração nativa.

**As fases 0–18 precisam ser reexecutadas sobre esta referência**, adaptando seus leitores ao alvo `code/50`, aos Zarrs separados e à máscara. Não execute interpretadores antigos que presumem `log1p` sobre o novo alvo. Esta entrega não afirma que as fases foram adaptadas ou executadas; também não define ainda o orquestrador de casos regressão/difusão. O próximo passo é verificar estes resultados e atualizar os diagnósticos com base na referência consolidada.

## Validação da integridade

Execute no ambiente da construção, onde estejam disponíveis as fontes e os artefatos completos.

### Execução no CEFET

```bash
cd /home/vsantos/repositories/rionowcastdf
source .venv-zarr-reference/bin/activate
export RADAR_ROOT=/home/vsantos/data/radar_sumare
export ERA5_ROOT=/home/vsantos/data/reanalysis/cds/era5
mkdir -p logs
nohup python -u audit_reference_zarr.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  --output-dir analysis_outputs_restored/reference_integrity_v1 \
  > logs/reference_integrity_v1.log 2>&1 &
echo "PID: $!"
```

```bash
tail -n 60 -f logs/reference_integrity_v1.log
```

O diretório de saída precisa ser novo. Em nova execução use, por exemplo, `reference_integrity_v2`. Não execute simultaneamente com uma construção ou alteração dos arquivos de entrada. Não remova a pasta `staging` do dataset antes da auditoria.

### Escopo

- Verifica hashes dos artefatos, contrato/configuração/código, manifesto congelado e manifesto do dataset.
- Verifica horários UTC, nomes locais com horário de verão, tolerância temporal, partições e purge conforme contrato do builder. Caminhos de aquisição não podem cruzar partições; imagens transparentes idênticas em datas distintas não são automaticamente tratadas como vazamento.
- Verifica hashes das grades horárias em `staging` e recalcula a normalização com todas as horas de treino, antes dos patches; compara JSON e NPZ.
- Lê todos os patches dos quatro Zarrs: shapes, dtypes, dimensões, ausência de fill-value que mascare zeros válidos, valores finitos, máscara, posições, horário fonte, deslocamento e SHA da imagem.
- Compara cada patch de entrada com a grade ERA5 interpolada preservada em staging; compara cada alvo e máscara com o cache do radar. Não arredonda valores do alvo para aceitar classes próximas.
- Recalcula todos os hashes lógicos dos arrays, contagens de patches, suporte e pixels sem eco.
- Decodifica até duas imagens de conteúdo distinto por combinação split/política, distribuídas temporalmente, e compara com o cache. Lista a amostra em `png_samples.json`. Pode ajustar `--png-samples-per-policy`.

### Resultados

`summary.json` registra o progresso e, no sucesso, `complete: true` e `integrity_passed: true`. Erros encerram com código diferente de zero e ficam registrados no resumo e log. O programa interrompe na primeira inconsistência, sem reparar ou alterar os dados.

A verificação integral lê todos os tensores e pode demorar devido ao volume descomprimido. Progresso a cada 500 horas por split; a checagem inicial de staging/normalização também pode levar algum tempo.

PASS significa consistência técnica com o contrato e os artefatos preservados. Não confirma cobertura física do radar, não executa CorrDiff, não refaz independentemente a interpolação dos Parquets ERA5 e não substitui os diagnósticos científicos. A comparação PNG/cache é amostral e usa o decodificador contratado. A seleção nearest2 é confrontada com o manifesto congelado; não se refaz a busca de candidatos no acervo inteiro.

### Testes locais

O teste usa o fixture do builder em `tests/test_reference_zarr.py`:

```bash
python -m unittest discover -s tests -p test_reference_integrity.py -v
```

Foram verificados um dataset sintético completo, a detecção de alteração numérica em um patch e a recusa de gravar relatórios dentro do dataset.

## Diagnósticos e integração nativa

A sequência de trabalho é: **integridade → adaptação dos diagnósticos → execução conforme dependências → CorrDiff nativo**. A fase 18 já integra esse último passo; não se trata de executar 0–18 e depois repetir uma segunda fase 18.

### Regras comuns aos leitores e métricas

1. Ler os quatro Zarrs e seus manifests sem reconstruir splits a partir de índices antigos. Preservar purge e exclusões do contrato; patches da mesma hora não podem ser redistribuídos aleatoriamente entre conjuntos.
2. Usar a máscara em perdas, estatísticas e métricas. Zero com máscara verdadeira significa sem eco exibido; zero com máscara falsa não é observação.
3. O alvo armazenado é `target_code / 50`. Para métricas na escala codificada, recuperar `50 * target`; não aplicar `expm1` ou `log1p`. Classes exibidas são 20–50 dBZ; o código zero não é uma medida física de 0 dBZ.
4. Distinguir entradas físicas lidas diretamente do Zarr das entradas padronizadas pelo loader. Aplicar normalização de treino somente uma vez.
5. Separar contagens por patch das contagens por hora/pixel único. Patches sobrepostos não são novas observações independentes. Reconstruções espaciais devem tratar a multiplicidade de cobertura e a máscara.
6. Usar horários de referência UTC e timestamps reais da fonte; verificar deslocamentos nearest2 e gaps antes de calcular persistência, defasagens ou sequências.
7. Ajustar transformações, seleção de variáveis, agrupamentos e modelos somente no treino. Validação orienta decisões; testes ficam reservados à avaliação final.
8. Gravar versão do código, contrato, manifesto, split, unidade, transformação, máscara e regra de ponderação em cada resultado. Não sobrescrever evidências antigas.

### Fases e dependências

Os scripts abaixo já existem em `scripts/`. Sua existência não significa compatibilidade certificada com a referência. Os comandos de execução devem ser publicados após adaptar e testar as interfaces.

| Fase | Finalidade | Dependência ou ajuste principal |
|---|---|---|
| 0 | Estrutura e auditoria do dataset | Integridade independente e leitor do novo contrato |
| 1 | Estatísticas univariadas | Fase 0; unidades físicas e ponderação explícita |
| 2 | Variáveis derivadas | Fase 0; ordem de canais, unidades e níveis de pressão |
| 3 | Estatísticas conjuntas | Fases 1–2; máscara e associação espacial |
| 4 | Extremos | Fases 0–1; limiares na escala do produto, sem converter para chuva |
| 5 | Sazonalidade | Cobertura temporal e denominadores por período |
| 6 | Ciclo diurno | UTC/local histórico e disponibilidade por hora |
| 7 | Defasagens temporais | Gaps, horários reais e pares temporalmente válidos |
| 8 | Persistência | Fase 7; horizonte definido e ausência de acesso ao futuro |
| 9 | Estrutura espacial | Geometria, máscara e sobreposição dos patches |
| 10 | Relações espaciais ERA5–radar | Fases 2 e 9; interpolação e escalas distintas |
| 11 | Escalas espaciais | Fase 9; distância física aproximada e amostragem angular |
| 12 | Análise multivariada | Fases 1–3; ajustes somente no treino |
| 13 | Regimes meteorológicos | Fase 12; atribuição dos demais conjuntos sem refit |
| 14 | Mudança de distribuição | Comparação dos splits congelados e cobertura desigual |
| 15 | Splits formais | Validar os splits da referência, sem reparticionar silenciosamente |
| 16 | Baselines e handoff NVIDIA | Fases 0/15 aprovadas, máscara e normalização consistentes |
| 17 | Manifesto de ablações | Fase 16; canais/casos versionados e comparabilidade |
| 18 | Integração CorrDiff | Contrato do loader, ambiente GPU e checkout NVIDIA identificados |

As fases 5–14 podem ser executadas em grupos após suas dependências e o leitor comum estarem validados. A fase 15 deve ser conferida desde o início porque os splits já fazem parte do dataset construído.

### Critérios para o teste nativo CorrDiff

- Registrar commit do checkout NVIDIA/PhysicsNeMo, versões de PyTorch/CUDA, GPU e driver; verificar compatibilidade antes do treinamento.
- Exercitar o adaptador com formas `[B,12,32,32]` e `[B,1,32,32]`, normalização única, máscara, metadados e separação dos conjuntos.
- Verificar onde a máscara entra na perda nativa. Não treinar pixels inválidos como zeros observados.
- Executar primeiro um treino curto de regressão: perdas finitas, gradientes, parâmetros atualizados, checkpoint salvo e retomada.
- Em seguida, testar difusão residual condicionada à regressão, carregamento dos checkpoints e geração de amostras, respeitando o contrato numérico.
- Confirmar desnormalização/interpretação dos resultados e métricas com máscara. Saídas contínuas do modelo não devem ser tratadas automaticamente como classes observadas exatas.
- Usar treino e validação nesses testes. Nenhum teste TinyRegressor anterior substitui essas verificações.

Entregáveis esperados: relatório de integridade, relatórios diagnósticos por fase, manifesto de ablações e relatório de integração nativa com logs, configurações e checkpoints identificados. O planejamento de casos distribuídos por máquinas vem depois dessa referência validada.

## Preservação e histórico

O builder, seus módulos `contract_*.py`, as configurações e os artefatos congelados não são alterados por esta consolidação documental. Seus hashes continuam relevantes. Não apague `staging/`, manifests ou caches necessários à auditoria.

- [Migração de caminhos](docs/MIGRACAO_ESTRUTURA.md).
- [Dados e resultados](docs/DADOS_E_RESULTADOS.md).
- [Validação da reorganização](docs/VALIDACAO_REORGANIZACAO.md).
- [Validação sintética original do builder](docs/history/VALIDACAO_ZARR_REFERENCIA.md).
- [Roteiro histórico de reconstrução](docs/history/README_RECONSTRUCAO_FASES_00_A_18.md): usa outro contrato, incluindo log1p e stride 16; não executar como roteiro do Zarr atual.

Caches, Zarrs, bancos SQLite e grandes manifests não devem ser publicados no Git. Os READMEs internos dos módulos e os guias históricos permanecem disponíveis para consulta; somente este README concentra a orientação operacional da referência atual.

### Leitor compartilhado para diagnósticos

`src/corrdiff/diagnostics/reference_reader.py` fornece leitura em unidades físicas
ou padronizadas, alvo `code/50`, máscaras, metadados e reconstrução por hora sem
contagem duplicada das sobreposições. Consulte
[uso e verificação do leitor](docs/validation/LEITOR_COMPARTILHADO.md).
