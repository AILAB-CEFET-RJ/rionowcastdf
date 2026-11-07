# Entrega 2 — inventário ERA5 e pareamento temporal 14k

## Objetivo e limite

Determinar quais horas do novo cache 14j têm os 12 preditores ERA5 disponíveis,
em UTC, e atribuir as divisões cronológicas. Esta entrega não gera patches,
não interpola os campos ERA5, não normaliza valores, não treina modelos e não
resolve a semântica dos pixels transparentes. Ela produz um manifesto rastreável
para essas operações posteriores. `training_ready` permanece falso em todas as linhas.

O inventário pode ser executado enquanto o cache radar é construído. Os dois
processos fazem leitura do disco; isso pode aumentar o tempo de ambos. Se o
servidor ficar sobrecarregado, execute o inventário depois do cache.

## Instalação

Copie o ZIP para a raiz do repositório no CEFET e execute no ambiente dataset:

```bash
cd /home/vsantos/repositories/rionowcastdf
unzip corrdiff_pairing_14k.zip
python -m pip install -r pipelines/dataset_preparation/corrdiff_pairing_14k/requirements.txt
mkdir -p logs
```

Dependências: pandas, numpy, PyArrow e DuckDB. O programa usa SQLite e fcntl da
biblioteca padrão e foi preparado para Linux/WSL. Não modifica o builder de produção.

## 1. Inventariar o ERA5 — pode começar agora

```bash
nohup python -u pipelines/dataset_preparation/corrdiff_pairing_14k/pair_inventory.py inventory \
  --single-dir /home/vsantos/data/reanalysis/cds/era5/single \
  --pressure-dir /home/vsantos/data/reanalysis/cds/era5/pressure \
  --output-dir analysis_outputs_restored/14k_era5_inventory \
  > logs/14k_era5_inventory.log 2>&1 &
```

Os caminhos `single` e `pressure` correspondem ao carregador atual do repositório.
Se a instalação usa caminhos diferentes, informe as raízes reais. A busca é
recursiva por `*.parquet`, inclusive partições `year=.../month=...`.

```bash
tail -n 40 logs/14k_era5_inventory.log
cat analysis_outputs_restored/14k_era5_inventory/inventory_summary.json
```

O intervalo padrão é inclusivo: 2011-01-01T00:00Z a 2024-12-31T23:00Z.
Para um piloto, adicione, por exemplo:
`--start-utc 2012-12-11T00:00:00Z --end-utc 2012-12-11T23:00:00Z`
e use outro diretório de saída. O programa ainda inspeciona os arquivos encontrados,
mas agrega valores apenas no intervalo solicitado.

Repetir o mesmo comando retoma automaticamente o inventário: arquivos sem mudança
em tamanho e mtime_ns e com leitura anterior bem-sucedida são reutilizados. Erros
são tentados novamente. Arquivos alterados são reprocessados, arquivos removidos
são retirados do inventário. A transação de cada arquivo evita duplicar progresso.
`--rescan` força nova leitura de todos os valores. Tamanho/mtime não é hash de
conteúdo: use `--rescan` se os arquivos tiverem sido substituídos preservando ambos.
Contrato temporal, raízes, unidades e código diferentes exigem outro diretório.
Não altere o acervo durante a leitura.

### Verificações realizadas

- Detecta, sem adivinhação, uma coluna temporal entre valid_time/time/datetime/date.
- Rejeita múltiplas colunas temporais candidatas, timestamps nulos/não interpretáveis
  e tempo numérico sem unidade de época especificada.
- Timestamps sem fuso são interpretados como UTC conforme o ETL fornecido; timestamps
  com fuso são convertidos para UTC. Metadata global de timezone contraditório é erro.
  Ausência de metadata não comprova UTC: a origem desta regra é o contrato do ETL.
- Detecta latitude/lat e longitude/lon; converte longitude 0–360 para −180–180.
- Identifica os canais por nomes curtos ou aliases do builder. Mais de um alias
  para uma mesma variável no arquivo é ambiguidade, não escolha silenciosa.
- Níveis padrão em hPa: 850 e 500. Use `--pressure-unit Pa` somente se a coluna
  estiver em pascal. Não converte níveis por heurística de magnitude. Metadata de
  unidade Pa/hPa contraditório é erro. Níveis encontrados constam em era5_files.csv.
- Lê os valores; conta linhas, coordenadas distintas, latitudes/longitudes distintas,
  valores finitos e coordenadas inválidas por arquivo, timestamp e canal.
- As unidades das variáveis são registradas quando disponíveis, mas não se faz
  conversão de temperatura/vento/umidade ou teste de plausibilidade física nesta entrega.

Ordem dos canais: `tcwv, t2m, u10, v10, t_850, r_850, u_850, v_850,
t_500, r_500, u_500, v_500`.

O script ETL single compartilhado não solicita t2m/u10/v10; o inventário mostrará
se estes canais existem nos arquivos efetivos de outra versão. Se faltarem, não
serão fabricados nem substituídos por outra variável. Será necessário obtê-los
antes de montar o condicionamento de 12 canais.

O processamento usa uma tabela temporária por arquivo, dois threads DuckDB e limite
configurado de 1 GB para o mecanismo DuckDB, com spill em disco. Isso não é um teto
para todo o processo Python. O inventário SQLite e as exportações também ocupam espaço.

## 2. Parear depois de o inventário e o cache terminarem

```bash
python -u pipelines/dataset_preparation/corrdiff_pairing_14k/pair_inventory.py pair \
  --radar-cache-dir data/radar_intermediate_14j_v1 \
  --inventory-dir analysis_outputs_restored/14k_era5_inventory \
  --output-dir analysis_outputs_restored/14k_pairing_v1 \
  > logs/14k_pairing.log 2>&1
```

Para acompanhar em outro terminal:

```bash
tail -n 40 logs/14k_pairing.log
cat analysis_outputs_restored/14k_pairing_v1/summary.json
cat analysis_outputs_restored/14k_pairing_v1/split_pairing_coverage.csv
cat analysis_outputs_restored/14k_pairing_v1/transparency_scenarios.csv
cat analysis_outputs_restored/14k_pairing_v1/channel_coverage.csv
```

O pareamento é uma exportação, sem progresso por hora no log. Pode demorar para
carregar/agrupar o inventário e gravar os arquivos. Use `nohup ... &` se precisar
fechar a sessão SSH durante essa etapa.

Para uma prévia durante o cache, adicione `--allow-partial` e use um diretório novo,
por exemplo `analysis_outputs_restored/14k_pairing_preview_01`. O inventário ERA5
precisa estar concluído mesmo na prévia. Horas ainda não visitadas pelo 14j recebem
`not_processed_in_snapshot`; elas não são contabilizadas como PNG ausente. O resultado
é estático, não acompanha o avanço do cache: faça nova exportação ao final.

O SQLite Backup API cria cópias consistentes, inclusive com WAL. Não copie apenas
o arquivo .sqlite de um processo ativo usando cp. A etapa de pareamento bloqueia
leitura simultânea do inventário enquanto ele é atualizado. O cache radar pode
continuar escrevendo; a cópia é uma fotografia transacional de um instante.
O summary do 14j é um indicador conservador: uma fotografia parcial não vira final
só porque todas as linhas já existem durante uma retomada.

Cada saída de pair precisa ser um diretório novo; não sobrescreve exportações.
Se pair for interrompido, execute com outro diretório. Não há retomada dessa etapa.

## 3. Critério de disponibilidade dos preditores

Cada timestamp/canal recebe um dos estados:

| Estado | Significado |
|---|---|
| finite_rectangular_cover | Em um único arquivo: grade retangular completa, pelo menos 2 latitudes e 2 longitudes, sem coordenadas duplicadas/inválidas, todos os valores finitos, limites cobrindo toda a grade radar 14j |
| incomplete_or_invalid_grid | Falta algum desses requisitos; consultar métricas no snapshot SQLite |
| multiple_files_review | Mais de um arquivo fornece o mesmo canal no mesmo timestamp; não escolhemos uma versão nem juntamos mosaicos automaticamente |
| missing_channel_or_time | Nenhum registro desse canal/horário foi obtido; pode ser variável ausente, nível ausente ou horário ausente |

Uma malha regular pode ter resolução diferente da grade do radar e ainda cobrir
seus limites; a interpolação propriamente dita será feita na montagem dos tensores.
A checagem desta entrega usa os limites efetivos de `geometry.npz`, não somente as
bordas nominais da configuração. Não exige espaçamento uniforme do ERA5, mas exige
produto cartesiano completo de coordenadas distintas. Não trata domínios que cruzam
o antimeridiano; o contrato é o domínio regional do Rio.

Arquivos divididos espacialmente, expver/member duplicados ou versões sobrepostas
são sinalizados para revisão. Esta política é conservadora: não significa que os
arquivos sejam inutilizáveis, e sim que não foi aplicada uma regra arbitrária de fusão.

`era5_complete_12` exige os 12 canais com finite_rectangular_cover.
`era5_technical_candidate` exige isso e inventário sem arquivos com erro. Um arquivo
ilegível pode esconder uma versão sobreposta; por isso erros de inventário impedem
certificação técnica global até correção ou delimitação adequada das raízes.

`paired_palette_candidate` exige candidato ERA5, status radar com paleta amostrada
na grade e nome local não ambíguo. Isso NÃO certifica todos os pixels do alvo.
`train_smoke_candidate` acrescenta split=train; é uma seleção técnica para testar
futuramente o loader/loss mascarada, não uma autorização de treino científico.
O programa não abre todos os NPZs nem verifica novamente seus hashes: validade dos
objetos permanece sob responsabilidade da entrega14j; o pareamento conserva caminhos
relativos e hashes registrados para leitura/verificação posterior.

## 4. Horário e divisões

O join é por timestamp UTC exato. Não há aproximação, deslocamento fixo de três horas,
interpolação temporal, média ou escolha do minuto vizinho. O nome local do PNG já
foi resolvido pelo14j usando America/Sao_Paulo, inclusive horário de verão.

| Split | Anos nominais UTC | Papel |
|---|---|---|
| train | 2011–2020 | Ajuste de parâmetros e futura normalização |
| validation | 2021–2022 | Seleção de modelo/hiperparâmetros |
| test_primary | 2023 | Teste bloqueado |
| test_stress_ood | 2024 | Teste stress/OOD bloqueado |
| excluded_purge | Fronteiras | Exclusão temporal |

A purga padrão é [fronteira−48h, fronteira+48h), igual à fase15 consultada.
Há três fronteiras: 2021-01-01, 2023-01-01 e 2024-01-01 UTC. Na linha temporal
completa são 96 horas por fronteira, 288 horas ao todo. O relatório antigo podia
ter menos horas purgadas porque contava apenas timestamps disponíveis no dataset.
Aqui todas as 122.736 horas solicitadas entram no denominador, inclusive ausentes
ou não processadas. Não compare percentuais antigos e novos sem igualar denominadores.

Nenhum índice antigo de patch é reaproveitado. Depois da reconstrução dos patches,
todos os patches do mesmo timestamp herdarão o split do novo manifesto. Não se
assume que haverá 12 patches utilizáveis por hora quando a validade do alvo é parcial.
Os testes são inventariados, mas não usados para normalização, ajustes ou seleção
meteorológica. Não são calculadas distribuições de intensidades de teste.

## 5. Arquivos e resultados esperados

No inventário:

- inventory_config.json: contrato, canais, raízes, UTC, unidade de pressão e hash do código.
- era5_inventory.sqlite: arquivos e métricas por hora/canal, com retomada transacional.
- era5_files.csv: situação, schema, aliases, níveis, metadata e erros por arquivo.
- inventory_summary.json: conclusão da varredura, quantidade de arquivos e erros.

No pareamento:

- paired_manifest.parquet e .csv: uma linha por hora esperada, radar, caminhos/hashes,
  estados de cada canal, candidatos técnicos, split, ambiguidade e flags de bloqueio.
- era5_channel_diagnostics.parquet: estados e contagens dos canais encontrados.
  Canais ausentes aparecem explicitamente no paired_manifest e no channel_coverage.
- channel_coverage.csv: horas por canal/estado, com denominador de todas as horas solicitadas.
- split_pairing_coverage.csv: horas por split, completude ERA5, candidatos e ambiguidades.
- split_radar_status.csv e annual_radar_status.csv: disponibilidade por split/ano.
- transparency_scenarios.csv: transparentes por split, percentuais e quantos têm ERA5
  e horário não ambíguo; cenário hipotético de acrescentá-los aos candidatos com paleta.
  `hypothesis_applied=false`: nenhuma transparência foi transformada em zero. O cenário
  quantifica exposição à decisão pendente, não o número final de exemplos treináveis.
- source_hash_cross_split.csv: PNGs com hash idêntico presentes em mais de um split.
  Repetição de PNG transparente/mapa não prova vazamento meteorológico. Não há remoção automática.
- train_timestamps.parquet, validation_timestamps.parquet, test_primary_timestamps.parquet,
  test_stress_ood_timestamps.parquet, excluded_purge_timestamps.parquet: todos os timestamps
  da divisão, incluindo flags; não são listas exclusivamente de exemplos aprovados.
- radar_manifest_snapshot.sqlite, era5_inventory_snapshot.sqlite: bases utilizadas.
- radar_config_snapshot.json, radar_summary_snapshot.json: situação do cache observado.
- run_config.json: argumentos, instante UTC, hashes do código/config/geometry/snapshots.
- summary.json: totais e limitações. `complete` indica que pair concluiu; confira também
  `radar_snapshot_complete`, `provisional`, `era5_error_files` e production_ready=false.

Snapshots duplicam as bases SQLite por execução, mas não copiam todos os objetos PNG/NPZ.
Guarde o cache14j referenciado. Não é preciso colocar bases/cache grandes no Git.

## 6. Critério de saída e próximos passos

Exit 0: execução finalizada sem erros de arquivos ERA5. Não significa que existem
12 canais, alvo válido ou treino liberado. Exit 2: finalizada com erros de arquivos,
ver era5_files.csv. Exit1: falha de contrato/execução; exit130: interrupção.

Esperamos responder objetivamente: quais canais existem, quais horas são pareáveis,
onde estão perdas por ausência/NaN/duplicação/geometria, quanto cada split mantém,
e quantas horas transparentes mudariam de categoria sob uma regra futura documentada.

A próxima entrega pode exercitar leitura de tensors e máscaras nos candidatos de
TREINO. A validação física da transparência e da cobertura continua pendente do
produtor. Se existir máscara inválida, a loss precisa tratá-la explicitamente; não
preencher NaN com zero para satisfazer a interface nativa.

## Verificação do pacote

```bash
python -m unittest discover -s pipelines/dataset_preparation/corrdiff_pairing_14k/tests -v
```

Veja VALIDACAO.md. Testes locais sintéticos não substituem a execução no acervo CEFET.
