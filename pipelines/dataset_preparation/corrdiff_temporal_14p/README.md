# 14p — política do produtor e decisão temporal

Implementa os cinco encaminhamentos após o censo 14m. Não altera 14j/14k/14m/14n/14o, não cria Zarr nem inicia treinamento. Execute no Linux do servidor CEFET, a partir da raiz do repositório. Python 3.10+.

## 1. Preservação

`analyze` exige auditoria completa e usa SQLite backup para obter snapshot consistente, sob o lock compartilhado da 14m. Copia configuração, resumo e manifesto de pareamento; calcula SHA256 dos artefatos. Rejeita diretório de saída existente e árvores de entrada/saída sobrepostas. O resumo só recebe complete=true após todos os relatórios. Em falha, preserve os logs e escolha outro diretório de saída; não há resume nesta fase.

O snapshot SQLite ocupa aproximadamente o tamanho do banco 14m. A saída ainda inclui um índice SQLite de candidatos, manifesto comprimido dos arquivos e comparação Parquet. Reserve vários GB conforme o banco real. O programa percorre metadados; não repete a decodificação de três milhões de imagens.

## 2. Política

Transparentes: ausência de eco exibido, codificada como zero, apenas no suporte espacial adotado. Cores no PNG não amostradas pela grade: elegibilidade de frame, não prova de erro espacial. Nenhum pixel desconhecido deve ser convertido em zero pelo futuro loader. A política numérica e as máscaras da 14n/14o permanecem necessárias.

Cartografia e ausência de paleta exata ficam excluídas por padrão. `review_template.csv` permite decisões por SHA256: pending, reject, accept_no_echo. Aceitação exige evidence, reviewer, reviewed_at; só aceita cartografia com zero pixels meteorológicos. Arquivo com cores desconhecidas não pode ser liberado como seco automaticamente. Para incorporar revisões, execute analyze em nova saída com `--reviews caminho/revisoes.csv`.

`review_groups.csv` agrupa hash/ano/status com frequência e métricas. `review_samples.csv` seleciona primeiro representantes dos grupos ano/status, depois hashes frequentes, com limite padrão de 40. Não é amostra aleatória. `--radar-dir` copia os PNGs dessas amostras, verificando hash; falhas constam na tabela. Nenhuma imagem é alterada ou automaticamente aprovada. O template contém uma linha por hash; não use diretamente review_groups como decisões, pois o mesmo hash pode aparecer em vários anos.

## 3. Comparação temporal

Compara exact, nearest2 (±120s) e nearest5 (±300s). Cada hora escolhe UM PNG elegível: mínimo deslocamento absoluto, empate pelo instante anterior. Hora exata tem prioridade natural. Sem média, máximo ou interpolação temporal. Arquivo ausente nunca vira zero.

Usa America/Sao_Paulo e regras históricas de horário de verão. Exclui timestamps ambíguos/inexistentes, nomes não interpretáveis e qualquer duplicidade de rótulo local. Limite inicial: 2011-01-17 00h local = 02h UTC. Mantém os splits do manifesto 14k e confere o contrato vigente (treino até 2020, validação 2021–22, teste principal 2023, stress 2024, purga ±48h). Impede seleção de PNG de outro split. Manifestos com outra regra falham explicitamente.

Denominador: horas dentro do acervo presentes no manifesto 14k, incluindo horas sem ERA5 completo. Elegibilidade de pareamento exige era5_technical_candidate=true e split diferente de excluded_purge. Relatórios por ano/split não autorizam escolha por desempenho dos testes: decision_support contém somente ganhos de treino. A comparação mede disponibilidade e deslocamento, não habilidade meteorológica do modelo.

`minute_distribution.csv` mostra TODOS os arquivos por mês local/minuto (inclusive suspeitos); não é frequência de varreduras válidas. `offset_distribution.csv` mostra deslocamentos dos candidatos encontrados.

## Execução

```bash
cd /home/vsantos/repositories/rionowcastdf
python -m pip install -r pipelines/dataset_preparation/corrdiff_temporal_14p/requirements.txt
mkdir -p logs
nohup python -u pipelines/dataset_preparation/corrdiff_temporal_14p/pipeline.py analyze \
  --audit-dir analysis_outputs_restored/14m_full_png_audit_v1 \
  --pairing-dir analysis_outputs_restored/14k_pairing_v2 \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir analysis_outputs_restored/14p_temporal_v1 \
  > logs/14p_temporal_v1.log 2>&1 &
echo "PID: $!"
```

Ajuste somente o diretório do pareamento se a versão completa estiver em outro local. A auditoria deve conter audit.sqlite, summary.json, config.json e .audit.lock; o pareamento precisa de paired_manifest.parquet e summary.json. Não basta ter os CSVs publicados no GitHub.

```bash
tail -n 60 -f logs/14p_temporal_v1.log
```

## 4. Revisão dirigida

Enviar para análise: summary.json, decision_support.json, temporal_coverage.csv, offset_distribution.csv, minute_distribution.csv e review_samples.csv; inspecionar review_pngs com review_groups.csv. As amostras devem ter copy_status=copied antes da inspeção. As decisões humanas são registradas no template com evidência, nome e data. Não extrapolar a resposta sobre um exemplo de cartografia para todos os hashes.

## 5. Fechamento da regra

Após avaliar a comparação no treino, `freeze` registra a escolha explícita. Exemplo SOMENTE caso a decisão seja nearest2:

```bash
python -u pipelines/dataset_preparation/corrdiff_temporal_14p/pipeline.py freeze \
  --analysis-dir analysis_outputs_restored/14p_temporal_v1 \
  --strategy nearest2 \
  --rationale 'PREENCHER com justificativa baseada no relatório de treino' \
  --radar-cache-dir data/radar_intermediate_14j_v1 \
  --output-dir analysis_outputs_restored/14p_temporal_contract_v1
```

Também aceita exact ou nearest5. Não execute o exemplo com justificativa fictícia. A escolha não é automática: aumento de disponibilidade, por si só, não demonstra superioridade meteorológica. Janelas simétricas permitem usar observação posterior à referência; isso deve constar na metodologia de downscaling. Se o experimento exigir causalidade de previsão, o contrato precisará de janela somente passada antes de produzir o dataset.

Entregas: temporal_contract.json, selected_manifest.parquet, selected_counts.csv, required_sources.csv e, quando informado o cache 14j, additional_sources_to_decode.csv. Reuso do cache é identificado pelo hash do PNG, independentemente do nome temporal; o loader deve validar os bytes dos objetos decodificados. O comando não altera nem expande o cache. Esses manifestos preparam a futura materialização Zarr; não substituem o dataset da 14o automaticamente.

## Organização do repositório

Manter código, documentação, configurações, resumos, tabelas agregadas e decisões de revisão. Manter audit.sqlite original, snapshots e PNGs brutos no servidor com backup. Não apagar auditorias anteriores: documentam a evolução metodológica. Não reexecutar fases 0–14 sobre o dataset antigo para avaliar esta versão.

Adicionar ao .gitignore da raiz antes de git add:

```gitignore
/analysis_outputs_restored/14p_*/audit_snapshot.sqlite*
/analysis_outputs_restored/14p_*/temporal_index.sqlite*
/analysis_outputs_restored/14p_*/policy_files.csv.gz
/analysis_outputs_restored/14p_*/*.parquet
/analysis_outputs_restored/14p_*/review_pngs/
/analysis_outputs_restored/14p_*/review_groups.csv
/analysis_outputs_restored/14p_*/required_sources.csv
/analysis_outputs_restored/14p_*/additional_sources_to_decode.csv
```

Ignore não remove arquivos já rastreados. Não usar git add . sem conferir git status e tamanhos. Nenhuma limpeza destrutiva é executada por este pacote.

## Testes

```bash
python -m unittest discover -s pipelines/dataset_preparation/corrdiff_temporal_14p -p 'test_*.py' -v
```

Os testes usam banco sintético no formato 14m, pareamento Parquet e fluxos analyze/freeze. O acervo completo só pode ser executado no servidor onde audit.sqlite está disponível. production_ready permanece false: decisão temporal não certifica treinamento, validação meteorológica ou integração Zarr.
