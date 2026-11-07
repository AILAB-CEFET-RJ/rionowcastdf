# Correção e auditoria radar — v2

Pacote para executar na raiz de AILAB-CEFET-RJ/rionowcastdf. Não altera o builder original nem os dados antigos. Não executa treino neural.

## O que foi implementado

- Conversão conservadora: reconhece somente as sete cores opacas da paleta de eco; não interpola cores arbitrárias para dBZ.
- Cores desconhecidas permanecem NaN e, no dataset, são representadas pela máscara de validade. Não são observações de tempo seco.
- Triagem de fundo cartográfico com quarentena reversível, revisão ligada ao SHA256 da imagem e justificativa.
- Varredura de todas as horas de 2011–2024, incluindo fontes ausentes; cache novo, hashes, cobertura anual, fila de revisão e painéis.
- Reconstrução usando configuração ERA5/geometria do relatório original e exclusivamente o cache QC. A conversão RGB antiga não é acionada.
- Auditoria antes/depois, denominadores válidos, separação temporal original e purge de 48 h de cada lado.
- Baselines ajustados com máscara somente em treino, avaliação somente em validação, persistência na coorte comum e normalização de inputs train-only.
- Adaptador para fases 13–15: evento é determinado pela contagem de pixels da fase 6, evitando erros de comparação após expm1 em float32.

## Instalação e primeira execução

Extraia o ZIP na raiz do repositório, mantendo a pasta corrdiff_radar_qc.
Use o ambiente Python do projeto: o arquivo de requisitos adiciona dependências deste pacote, não substitui as dependências do builder/ERA5.

```bash
python -m pip install -r pipelines/dataset_preparation/corrdiff_radar_qc/requirements.txt
bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh test
mkdir -p logs
nohup bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh scan > logs/14d_qc_v2.log 2>&1 &
```

O scan lê PNGs de /home/vsantos/data/radar_sumare e cria data/radar_cache_qc_v2.
Uma execução completa pode demorar: percorre todas as horas, não apenas os 224 frames anteriores.
Inspecione qc_summary.json, coverage_by_year.csv, source_review_queue.csv e os painéis.
Erros de leitura/conversão encerram com código não zero. Quarentena e fontes ausentes são contabilizadas separadamente.

## Revisão da fonte: pendência científica

Os painéis auditados mostraram fundo cartográfico convertido para extremos. Igualdade PNG→cache→target demonstra reprodução do conversor, não validade física.
O limiar de 1% de bege na grade é uma regra de triagem, não prova automática de artefato.
Não interprete todos os candidatos como imagens inválidas.

A política padrão não declara nenhuma cor como “sem eco”. Portanto, o resultado inicial é DIAGNÓSTICO: pode remover horas secas e selecionar somente pixels com eco.
Não use suas prevalências como climatologia natural nem libere treino com essa política sem revisar a semântica de observação.
Nem preto, branco, transparência, nem um frame inteiramente vazio devem ser automaticamente tratados como radar válido e seco.

Para documentar regras, gere uma cópia editável da política:

```bash
PYTHONPATH=pipelines/dataset_preparation/corrdiff_radar_qc python -c 'import json; from radar_qc import DEFAULT; print(json.dumps(DEFAULT, indent=2))' > radar_policy_reviewed.json
```

Só acrescente entradas RGBA em dry_rgba depois de confirmar sua semântica na fonte. Registre dry_evidence e source_contract_evidence.
Mesmo uma imagem inteiramente composta por cores secas requer revisão de aquisição válida; a política não permite confundir ausência de dados com ausência de chuva.
No CSV de revisão preencha timestamp com fuso, source_sha256, action (accept/exclude), reason e reviewer. Não altere os hashes.
Aceitar um frame não torna cores desconhecidas válidas: elas continuam mascaradas.

Rode novamente em OUTRA pasta, preservando a auditoria inicial:

```bash
QC_CACHE=data/radar_cache_qc_v2_reviewed QC_POLICY=radar_policy_reviewed.json QC_REVIEWS=radar_source_reviews.csv bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh scan
```

Não há retomada automática de execução interrompida; use uma pasta nova. Se mudar QC_CACHE, mantenha a mesma variável nas etapas seguintes.

## Reconstrução e métricas

Execute cada etapa apenas após a anterior terminar com sucesso:

```bash
bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh rebuild
bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh audit
bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh diagnostics
bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh baselines
```

Para execução longa, aplique o mesmo padrão nohup e um log distinto por etapa.
Variáveis configuráveis: PYTHON, RADAR_DIR, OLD_CACHE, QC_CACHE, OLD_DATASET, QC_DATASET e ANALYSIS.
Os padrões criam datasets/corrdiff_2011_2024_restored_qc_v2 e analysis_outputs_restored_qc_v2.
Não reutilize pastas de saída para versões diferentes.

14e usa minimum_radar_valid_ratio=0 para preservar slots espaciais e mantém a máscara; patches sem alvo válido devem ser ignorados na loss e nas métricas.
Horas sem qualquer observação decodificável não entram no cache. Compare a cobertura por ano/estação antes de interpretar diferenças.
14f grava índices temporais próprios e sua impressão digital. O baseline confere esses índices antes de ler alvos.
Os diagnósticos 13–15 dependem dos scripts existentes no repositório. Eles não foram executados aqui no dataset real.
O acesso descritivo às fontes de 2023/2024 serve ao QC; não autoriza usar esses anos para selecionar modelos.

## Alcance dos baselines

16_masked_baselines.py é uma implementação focal, não uma substituição de todos os arquivos da fase 16 original.
Exporta erros pontuais, contagens categóricas, CSI/POD/FAR e FSS em 20/30/40/45 dBZ, além de climatologias e normalização.
Não exporta todo o relatório original de máximos por patch, ETS e diagnósticos sazonais.
A comparação entre baselines usa suporte válido comum; persistência exige t−1 h exato na mesma posição e dentro de validação.
FSS utiliza frações normalizadas pelo suporte válido da janela. A convenção difere da avaliação antiga sem máscara; não compare os valores diretamente sem recalcular ambos com o mesmo protocolo.
As janelas 1/2/4/8 pixels pressupõem resolução de 2 km.

## Antes da fase 18

1. Resolver e documentar a distinção entre ausência de eco e ausência de observação.
2. Executar scan/rebuild/audit e revisar a cobertura e a distribuição após QC.
3. Reexecutar os diagnósticos e baselines; comparar os índices dos splits antes do handoff.
4. Preservar a matriz da fase 17, mas atualizar hashes/caminhos/normalização e integrar a máscara na loss, na amostragem e nas métricas.
5. Manter 2023 e 2024 bloqueados para seleção. Não inferir que o excesso de extremos de 2024 é meteorológico enquanto a contaminação não for quantificada após QC.

production_ready permanece false deliberadamente. Este pacote entrega a correção e os instrumentos de revisão; não certifica automaticamente a qualidade científica do dataset.

## Validação realizada nesta entrega

11 testes sintéticos passaram: paleta, transparência, cores não reconhecidas, quarentena, revisão vinculada à fonte, proteção contra sobrescrita, purge, limiares float32, métricas mascaradas e proteção dos testes temporais.
O teste de baselines envenena os alvos de 2023/2024 com NaN e verifica que a avaliação de validação continua correta.
Não houve reconstrução do dataset real nem execução de treino no servidor CEFET.
