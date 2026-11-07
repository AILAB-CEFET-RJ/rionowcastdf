# Piloto 14i — fuso, geometria e sequências do Sumaré

Implementação isolada e executável para comparar o processamento legado com o contrato espacial/temporal do projeto rionowcast-radar-stations. Não modifica o builder, o cache ou o dataset existentes. Não constitui liberação para treinamento nem reconstrução completa.

## Execução no CEFET

Extraia o ZIP na raiz de /home/vsantos/repositories/rionowcastdf. O ZIP já contém a pasta corrdiff_alignment_14i. Execute na raiz do repositório:

```bash
python -m pip install -r pipelines/dataset_preparation/corrdiff_alignment_14i/requirements.txt
python -u pipelines/dataset_preparation/corrdiff_alignment_14i/14i_alignment_pilot.py \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir analysis_outputs_restored/14i_alignment_pilot_v1
```

O destino deve ser novo. Para repetir, use um novo sufixo, por exemplo v2. Não é necessário reconstruir o cache antigo. A execução padrão tenta apenas 66 arquivos, 11 por caso; não precisa percorrer o acervo inteiro.

## O que foi corrigido neste piloto

1. O nome do PNG representa America/Sao_Paulo. Cada caso local é convertido para UTC com regras históricas de horário de verão. Cada instante UTC é convertido de volta para o nome local correto ao procurar o arquivo. Horários locais inexistentes ou repetidos são detectados por round-trip; casos centrais ambíguos exigem âncora UTC explícita e os arquivos de hora repetida permanecem em revisão.
2. O domínio de saída preserva a grade 70x84, latitude [-23.5,-22.25), longitude [-44,-42.5), passo angular 2/111 grau. Isso mantém a compatibilidade de coordenadas com a grade regional existente; não afirma espaçamento métrico exato de 2 km em ambos os eixos.
3. As coordenadas da grade são projetadas em AEQD esférica centrada no radar (-22.955139,-43.248278), raio terrestre 6371008.8 m, raio de imagem 138.9 km, norte para cima e leste à direita. Imagem esperada: 656 colunas e 654 linhas. Amostragem pelo pixel nativo mais próximo, conforme o método do projeto de referência.
4. O mapeamento legado também é calculado, com a mesma decodificação conservadora, para isolar o efeito da geometria. Não reproduz a antiga interpolação RGB que convertia cartografia em extremos.
5. Transparência e cores desconhecidas permanecem NaN. Somente cores opacas exatas da paleta diagnóstica são decodificadas. O indicador de cor bege é apenas uma triagem reversível de cartografia. Não se preenchem faltantes com zero.

Referência fixada: AILAB-CEFET-RJ/rionowcast-radar-stations, commit 65201b6787bb26997da5bc2145807d1a8fd4be33, configs/radar_sumare_historical_audited_v1.json e src/nowcasting/radar_georeferencing.py. Comparação legada: rionowcastdf commit 554c019356481832a62d7d0d46d3d07df1bc29fe.

## Casos padrão (nomes locais, todos do período de treino)

- 2011_08_27_11_00: sinal esparso da auditoria anterior.
- 2016_10_02_14_00: sinal esparso intermediário.
- 2014_10_02_04_00: maior grupo da seleção esparsa anterior.
- 2012_02_25_03_00: PNG totalmente transparente já inspecionado.
- 2012_02_25_04_00: PNG cartográfico já inspecionado.
- 2012_12_11_17_00: exemplo com cores de eco, para controle.

Para cada caso: -10,-8,-6,-4,-2,0,+2,+4,+6,+8,+10 minutos. As sequências são construídas em UTC, depois convertidas para nomes locais. Não há média, interpolação ou substituição automática de um arquivo ausente.

Pode selecionar outros casos locais:

```bash
python pipelines/dataset_preparation/corrdiff_alignment_14i/14i_alignment_pilot.py \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir analysis_outputs_restored/14i_outros_casos \
  --local-cases 2012_02_25_03_00 2012_12_11_17_00 \
  --radius-minutes 10 --step-minutes 2
```

Ou selecionar instantes ERA5 em UTC explicitamente com --utc-cases 2012-02-25T05:00:00Z. Os valores ERA5 não são lidos neste piloto; a referência UTC demonstra a associação temporal e não comprova disponibilidade dos preditores.

## Saídas

- config.json: contrato completo e hash.
- geometry.npz: latitude/longitude, índices corrigidos e legados, máscara geométrica e ROI no raster original.
- frames.csv: horários UTC/local, nomes, hashes, status, cores no PNG inteiro, no domínio projetado e nas duas grades.
- cases.csv: resumo de cada sequência, disponibilidade e nome que seria buscado incorretamente usando UTC como nome local.
- transitions.csv: estados sucessivos, igualdade de hashes e Jaccard das máscaras de cores no domínio. Se não houver cor em nenhum dos dois frames, Jaccard fica vazio; não é prova de semelhança meteorológica. Intervalos são separados por caso.
- cache_utc/*.npz: campos diagnósticos por timestamp UTC. Formato deliberadamente distinto do cache .npy legado, para não ser consumido como target validado.
- panels/*_sequence.png: sequência do domínio corrigido. Xadrez significa transparência.
- panels/*_comparison.png: imagem central completa, grade legada e grade corrigida. Cores codificadas pela paleta; branco nas grades é desconhecido.
- panels/geometry_comparison.png: pontos de amostragem dos dois métodos no PNG.
- summary.json: totais, erros, hashes e indicadores de conclusão.

Nos NPZs, dbz_palette_corrected_raw é a correspondência de paleta; dbz_review também aplica quarentena de cartografia/horário ambíguo. palette_mask não é máscara de observação válida. geometric_mask só representa limites geométricos do produto: não identifica bloqueios, qualidade, atenuação nem operação do equipamento. sampled_alpha fora da máscara geométrica não deve ser usado.

## Interpretação objetiva

- native_palette_pixels > 0, native_roi_palette_pixels = 0: as cores estão fora do nosso domínio sob a geometria de referência; não é falha de amostragem desse domínio.
- native_roi_palette_pixels > 0, corrected_grid_palette_points = 0: existem cores dentro do domínio entre os pontos amostrados. A existência de cores não prova sua origem meteorológica.
- corrected_grid_palette_points diferente de legacy_grid_palette_points: efeito da escolha de pixels de origem, mantendo a mesma paleta diagnóstica.
- transparent_ambiguous: transparência total preservada como desconhecida.
- map_screen_review: presença de fundo cartográfico acima do critério de triagem; requer revisão, sem converter mapa em chuva.
- missing_png: o arquivo daquele minuto não foi encontrado. Não representa automaticamente ausência de varredura; pode ser ausência no acervo.
- ambiguous_local_filename_review: o nome não distingue as duas ocorrências de uma hora repetida no horário de verão.

complete=true significa que todas as solicitações foram tentadas, mesmo com arquivos ausentes. all_requested_files_present indica se todos foram encontrados. source_read_errors indica falhas de leitura/dimensão. production_ready=false e dry_rule_documented=false são esperados: nenhuma semântica de tempo seco foi inventada.

## Verificações realizadas

Nove testes unitários passam: hora local/UTC, horário de verão, mudança de data, horas inexistentes e repetidas, rejeição de UTC sem fuso, centro/orientação da projeção, domínio/grade, transparência/cartografia, paleta com alfa e dimensões.

Também foi executado o fluxo completo nos três PNGs reais publicados na raiz do repositório. O arquivo de controle examples/radar_sumare/2012_12_11_17_00.png teve 229 pontos da paleta na grade legada e 714 na corrigida. O PNG transparente permaneceu ambíguo; o cartográfico foi sinalizado para revisão. Esses três nomes são UTC-2 nas datas correspondentes. Testou-se também a ausência dos minutos adjacentes na cópia local: foram registrados como missing_png, sem preenchimento ou crash.

Os arquivos reais adjacentes não estavam nesta cópia local: a inspeção das sequências completas depende da execução no CEFET. Os resultados locais estão em verificacao_local/ apenas para rastrear o teste; não representam a disponibilidade do acervo remoto.

```bash
python -m unittest discover -s pipelines/dataset_preparation/corrdiff_alignment_14i/tests -v
```

## Próximo resultado a revisar

Publique cases.csv, frames.csv, transitions.csv, summary.json e a pasta panels/ em analysis_outputs_restored/14i_alignment_pilot_v1. Não é necessário publicar cache_utc/. A análise dessas saídas definirá a política de seleção temporal e o tratamento dos casos ambíguos antes de uma única reconstrução completa. O builder principal e a fase 15 ainda não foram alterados: a incorporação em produção exige manifestos UTC novos, associação ERA5 correta e regra de validade do alvo.
