# Zarr de referência: contrato 14o + seleção temporal 14p

Esta atualização acrescenta `--contract-config` aos **dois scripts existentes**:

- `src/corrdiff/builder/radar_cache_builder.py`: cache compacto de alvos do radar.
- `src/corrdiff/builder/dataset_builder.py`: ERA5 interpolado, normalização e Zarrs por split.

Os módulos `contract_*.py` implementam o novo modo. Sem `--contract-config`, os scripts mantêm o caminho legado, que não implementa este contrato. Não usar o modo legado para construir esta referência. A importação antiga, inexistente, de `CorrdiffDatasetBuilder` no gerador de cache foi corrigida.

Base consultada: commit `24b93eaf6d2c24369ce85023ae96bb3f8043a5be` de `AILAB-CEFET-RJ/rionowcastdf`. Os testes desta entrega usam arquivos sintéticos; a construção integral deve ser executada no servidor onde estão os dados.

## 1. Instalação

Extraia os arquivos deste pacote na raiz de uma cópia atualizada do repositório. O pacote contém arquivos substitutos e novos módulos, mas não contém os dados nem os manifestos grandes.

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

## 2. Conferir as entradas

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

## 3. Criar o cache compacto

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

## 4. Conferir e construir os Zarrs

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

## 5. Conteúdo das saídas

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

## 6. Espaço, tempo e portabilidade

Com 99.098 horas e 12 canais: até 4.756.704 patches, cerca de **240 GiB** de tensores sem compressão (entrada, alvo e máscara), mais metadados. Os campos horários intermediários ocupam aproximadamente **29 GiB**. Compressão Zstd reduz o volume, mas a taxa depende dos dados. Reserve espaço adicional para tabelas temporárias DuckDB e saída em construção. O programa informa a estimativa para a configuração efetiva.

O tempo será dominado por leitura/interpolação ERA5 e escrita/verificação dos patches, não pelo treino. Não há estimativa confiável antes de observar os primeiros meses. `workers` controla decodificação de PNG; `duckdb_threads` e `duckdb_memory` controlam o leitor ERA5, que pode usar disco temporário. Os tensores não são carregados integralmente em RAM: os campos são preparados em arquivos mapeados por mês e os patches gravados por hora. O manifesto e os índices compactos permanecem em memória.

Outra máquina pode receber:

1. **Zarr pronto:** copie toda a pasta de saída (pode omitir `staging/` e `duckdb_tmp/` se não for retomar a construção), código do loader e dependências. Nenhum PNG ou Parquet é necessário para ler/treinar.
2. **Reconstrução:** copie contratos e manifesto congelado, cobertura/geometria, inventário, cache compacto ou PNGs e Parquets. Edite caminhos no JSON. Preserve tamanho e mtime dos Parquets usando `rsync -a`; o inventário original registra esses valores. Depois da primeira construção, `era5_identity.json` pode ser fornecido em `era5_identity_file` para validar cópias por SHA-256 mesmo quando mtime mudou. Sem identidade confiável, não se ignora uma divergência de arquivo: recopie preservando metadados ou refaça o inventário/pareamento.

Não colocar caches, staging e Zarrs no Git. Acrescente ao `.gitignore`, se ainda não houver regras equivalentes:

```gitignore
/data/radar_reference*/
/datasets/corrdiff_reference*/
/.venv-zarr-reference/
```

Isso não remove arquivos já rastreados nem apaga histórico de commits grandes. Publique os resumos pequenos e o código; mantenha os dados volumosos no armazenamento de dados.

## 7. O que enviar ao terminar

Envie o `summary.json` do cache e do dataset, `normalization.json`, e os quatro `<split>.complete.json` (ou suas contagens). O status `complete=true` significa construção e leitura de verificação concluídas. `production_ready=false` permanece deliberado até os diagnósticos científicos e a integração nativa.

**As fases 0–18 precisam ser reexecutadas sobre esta referência**, adaptando seus leitores ao alvo `code/50`, aos Zarrs separados e à máscara. Não execute interpretadores antigos que presumem `log1p` sobre o novo alvo. Esta entrega não afirma que as fases foram adaptadas ou executadas; também não define ainda o orquestrador de casos regressão/difusão. O próximo passo é verificar estes resultados e atualizar os diagnósticos com base na referência consolidada.
