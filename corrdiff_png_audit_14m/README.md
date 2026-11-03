# 14m — Auditoria integral dos PNGs do Radar Sumaré

## Escopo e limites

Percorre recursivamente TODOS os arquivos com extensão .png (sem distinguir maiúsculas/minúsculas) dentro de --radar-dir. Não seleciona só hora cheia, não limita anos e não consulta cache antigo ou ERA5. Arquivos originais são somente lidos; caches 14j e relatórios 14k/14l não são alterados. Sem acesso à rede e sem GPU.

Não classifica automaticamente transparência como tempo seco. Paleta exata não comprova observação meteorológica válida. A heurística de cartografia é a fração de pixels opacos da cor bege RGB(242,239,233), limiar 1%, usada nas auditorias anteriores; NÃO detecta toda cartografia possível. Falhar nessa heurística NÃO valida uma imagem. Uma inspeção visual de todos os milhões de arquivos não é realizada: todos recebem diagnóstico numérico; amostras limitadas são apresentadas visualmente.

A auditoria mede distribuição dos pixels coloridos (caixa envolvente, centroide, quadrantes), e amostragem pela geometria de referência 14j: AEQD esférica, centro -22.955139/-43.248278, alcance centro-borda 138900 m, bitmap 656 colunas x 654 linhas, grade regional 70x84. Não infere posição real da antena ou valida georreferenciamento com marcos geográficos; não gera máscara de cobertura física. A classificação paleta-no-PNG-sem-amostra não implica que a paleta esteja DENTRO da região de interesse: pixels externos à região também podem gerar essa classificação.

## Instalação no CEFET

Extraia o ZIP na raiz /home/vsantos/repositories/rionowcastdf. O ZIP contém a pasta corrdiff_png_audit_14m.

```bash
cd /home/vsantos/repositories/rionowcastdf
conda activate dataset
unzip corrdiff_png_audit_14m.zip
python -m pip install -r corrdiff_png_audit_14m/requirements.txt
python corrdiff_png_audit_14m/test_audit.py
mkdir -p logs
```

## Execução completa

Use uma pasta de saída NOVA. Não acrescente filtros de ano/minuto: a varredura completa é o padrão.

```bash
nohup python -u corrdiff_png_audit_14m/audit_pngs.py \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir analysis_outputs_restored/14m_full_png_audit_v1 \
  > logs/14m_full_png_audit_v1.log 2>&1 &
echo "PID da auditoria: $!"
```

Acompanhar:

```bash
tail -f logs/14m_full_png_audit_v1.log
cat analysis_outputs_restored/14m_full_png_audit_v1/progress.json
```

O programa registra progresso a cada 1000 arquivos. Não conhece o total antes de terminar a travessia, portanto não imprime porcentagem/ETA fictícia. Após a leitura, gera relatórios, exportação CSV gzip e painéis; summary.json indica stage=reports durante essa etapa. O processamento usa um processo e evita decodificar novamente bytes idênticos. Não foi medido desempenho no acervo completo; a duração depende da quantidade de arquivos distintos e do armazenamento.

## Retomada

Depois de confirmar que a execução anterior encerrou, repita o MESMO comando e acrescente --resume. Não reinstale/atualize dependências entre execuções: configuração, código e versões NumPy/Pillow são conferidos. Não remova o banco ou o arquivo config.json.

```bash
nohup python -u corrdiff_png_audit_14m/audit_pngs.py \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir analysis_outputs_restored/14m_full_png_audit_v1 \
  --resume \
  >> logs/14m_full_png_audit_v1.log 2>&1 &
echo "PID da retomada: $!"
```

A retomada percorre novamente os nomes, reaproveitando os diagnósticos se tamanho, mtime_ns e ctime_ns não mudaram. Para uma nova leitura dos bytes de TODOS os arquivos, acrescente --rehash junto de --resume. Hashes conhecidos reutilizam o diagnóstico decodificado. Erros de leitura transitórios são tentados novamente. Erros de decodificação de bytes idênticos são preservados, pois repetição não os corrige.

Há trava exclusiva por pasta de saída, transações SQLite e checkpoints a cada 1000 arquivos. Uma interrupção pode exigir reler o último lote. Arquivos removidos desde uma passagem anterior são mantidos no banco para rastreabilidade e excluídos dos relatórios da passagem atual. Mudanças de SHA são registradas na tabela changes.

A pasta de entrada deve permanecer estável durante a auditoria. Isso é um censo por travessia, não um snapshot atômico do filesystem. Alterações durante a leitura de um arquivo são sinalizadas; mudanças posteriores não podem ser garantidas sem uma nova leitura.

## Categorias

Ordem de precedência quando há múltiplos sinais:

1. read_error / source_changed_during_read / symlink_file_skipped / oversized_file_review: acesso ou limite de recursos.
2. decode_error: falha de verificação estrutural/CRC ou decodificação Pillow.
3. not_png_format_review: extensão PNG com conteúdo de outro formato.
4. multiframe_review: imagem animada/múltiplos frames; métricas referem-se ao primeiro frame.
5. unexpected_dimensions_review: diferente de 656x654 (largura x altura).
6. map_screen_review: heurística bege >= 1%.
7. transparent_ambiguous: todos os pixels com alpha zero.
8. no_exact_palette_review: nenhum pixel opaco com cor exata das sete classes.
9. palette_in_png_not_sampled_in_corrected_grid: há paleta no PNG, nenhuma na grade regional.
10. palette_sampled_observation_validity_unconfirmed: há suporte de paleta na grade; ainda NÃO é campo meteorológico aprovado.

As métricas preservam sinais simultâneos: fração bege, transparência parcial, pixels opacos fora da paleta, contagem por classe, hashes RGBA/alpha/máscara, paleta esparsa (1–20 pixels, somente diagnóstico). Um frame opaco não é automaticamente cartografia. Valores RGB invisíveis sob alpha=0 não contam como eco.

Paleta RGB -> dBZ: (197,0,197)->50; (227,6,5)->45; (255,112,0)->40; (195,230,0)->35; (4,85,4)->30; (19,122,19)->25; (0,167,12)->20. Só alpha=255 é contado como paleta exata. Não utiliza distância RGB nem inventa classe para outras cores.

## Horários, repetições e lacunas

Nome esperado YYYY_MM_DD_HH_MM.png em horário local America/Sao_Paulo. Datas inválidas, nomes não interpretáveis e horas ambíguas/inexistentes no horário de verão ficam sinalizados. UTC é preenchido só quando há conversão única. Nomes sem timestamp continuam sendo auditados como imagens.

Contagens são por ARQUIVO e não por hora. Duplicatas de timestamp e de conteúdo são relatadas separadamente. Arquivos idênticos são contabilizados tantas vezes quantos caminhos possuírem. Não se conclui falha, replay ou vazamento só pelo hash repetido.

Sequências em temporal_runs.csv agrupam estados consecutivos com diferença exata de dois minutos nos nomes locais. O campo same_file_hash indica se TODOS os arquivos daquela sequência têm o mesmo hash. Horários ambíguos/inexistentes interrompem a sequência; timestamps duplicados são excluídos dessa análise. observed_gaps.csv lista intervalos entre nomes locais, sem inferir falha do radar: horário de verão, acervo parcial ou frequência variável podem explicar lacunas. Não inclui lacunas antes do primeiro ou depois do último arquivo. Não gera matriz completa de transições/Jaccard entre imagens.

## Resultados

- summary.json: contagens finais; complete só é verdadeiro com pelo menos um PNG, travessia finalizada e sem diretórios inacessíveis/symlinks ignorados. Pode haver erros de arquivo registrados mesmo com complete=true. production_ready=false sempre.
- progress.json: checkpoint de leitura; complete=false por definição (consulte summary para finalização).
- config.json: parâmetros, SHA256 do script e versões das dependências.
- audit.sqlite: resultado por caminho, diagnóstico por SHA e histórico de mudanças; manter LOCAL.
- files.csv.gz: exportação integral, com métricas por arquivo em JSON na coluna metrics; pode ser grande.
- counts_by_year_status.csv: contagens por ano do nome e categoria.
- dimensions.csv: dimensões/modos ponderados por número de arquivos.
- timestamp_quality.csv e duplicate_local_timestamps.csv: qualidade dos horários.
- repeated_file_hashes.csv: grupos de bytes repetidos, primeiro/último horário observado.
- temporal_runs.csv e observed_gaps.csv: sequências e lacunas nos nomes locais.
- spatial_palette_by_year.csv: contagens por quadrante e paleta amostrada (não valida cobertura; inclui frames cartográficos para diagnóstico).
- files_absent_on_rescan.csv: arquivos de execuções anteriores não vistos na atual.
- traversal_errors.jsonl: diretórios inacessíveis e links de diretórios não seguidos.
- review/<execução>/samples.csv: seleção e origem de cada exemplo.
- review/<execução>/samples/: cópias binárias de exemplos, verificadas por SHA.
- review/<execução>/panel_*.jpg: prévias sobre fundo quadriculado, sem alterar os originais.
- review_bundle.zip: pacote limitado para compartilhar, com resumos, exemplos e painéis; sem SQLite nem exportações integrais.

Amostras: até dois hashes distintos por ano/categoria, até 160 exemplos no total. Seleção determinística pelo caminho, em rodadas para priorizar diversidade de grupos; não é amostra aleatória nem estimador de prevalência. Cópias de amostras limitadas a 96 MiB somadas e 10 MiB por arquivo. Falhas/cópias recusadas aparecem em samples.csv. O ZIP também inclui painéis e resumos, logo pode exceder 96 MiB em alguns MiB. Arquivo bruto acima de 64 MiB é inventariado como oversized_file_review e não decodificado/hasheado; altere --max-file-mib em uma nova saída se precisar examiná-lo. Proteção Pillow contra imagens gigantes permanece ativa.

Código de saída 0: censo terminou dentro do escopo, inclusive se identificou PNGs ilegíveis. Código 2: argumentos/trava/configuração ou censo incompleto. Código 130: interrupção pelo teclado. Outros erros não tratados retornam código diferente de zero e summary permanece incompleto.

## O que enviar para revisão

Primeiro:

```bash
cat analysis_outputs_restored/14m_full_png_audit_v1/summary.json
cat analysis_outputs_restored/14m_full_png_audit_v1/counts_by_year_status.csv
cat analysis_outputs_restored/14m_full_png_audit_v1/dimensions.csv
```

Depois compartilhar review_bundle.zip via Drive ou GitHub (confira o tamanho antes; para GitHub, prefira os resumos e os painéis separados). Ele contém nomes de arquivos e caminhos relativos; summary/config contêm a raiz local fornecida. Não envie audit.sqlite ou files.csv.gz ao Git.

Sugestão de .gitignore na RAIZ do repositório:

```gitignore
/analysis_outputs_restored/14m_full_png_audit*/audit.sqlite*
/analysis_outputs_restored/14m_full_png_audit*/files.csv.gz
/analysis_outputs_restored/14m_full_png_audit*/repeated_file_hashes.csv
/analysis_outputs_restored/14m_full_png_audit*/temporal_runs.csv
/analysis_outputs_restored/14m_full_png_audit*/observed_gaps.csv
/analysis_outputs_restored/14m_full_png_audit*/duplicate_local_timestamps.csv
/analysis_outputs_restored/14m_full_png_audit*/files_absent_on_rescan.csv
/analysis_outputs_restored/14m_full_png_audit*/review_bundle.zip*
/analysis_outputs_restored/14m_full_png_audit*/.audit.lock
```

Nunca execute git add . para enviar resultados sem conferir a seleção. As regras não retiram arquivos já versionados. Não exclua seus dados para resolver tamanho de push.
