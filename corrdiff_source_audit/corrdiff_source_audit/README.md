# Auditoria 14c — imagem original → cache → dataset

Somente leitura das fontes. Não modifica cache, Zarr, scripts do projeto nem splits.
Executar no ambiente Python do projeto, a partir da raiz do repositório.
Requer os CSVs da auditoria 14b e o builder local que contém RadarCacheSourceDataset.

## Execução

```bash
python -m pip install -r corrdiff_source_audit/requirements.txt
mkdir -p logs
nohup python -u corrdiff_source_audit/scripts/14c_audit_radar_sources.py \
  --dataset-dir datasets/corrdiff_2011_2024_restored \
  --audit14b-dir analysis_outputs_restored/14b_extreme_episodes \
  --radar-dir /home/vsantos/data/radar_sumare \
  --radar-cache-dir /home/vsantos/repositories/rionowcastdf/data/radar_cache \
  --builder-file src/corrdiff/builder/dataset_builder.py \
  --output-dir analysis_outputs_restored/14c_source_audit \
  > logs/14c_source_audit.log 2>&1 &
```

Ajustar `--audit14b-dir` para a pasta que contém timestamp_metrics.csv e episodes_ranked.csv.
A saída deve ser nova; não há overwrite. Para repetir use outro nome de saída.
Acompanhar: `tail -f logs/14c_source_audit.log`.

Geometria padrão reproduz o builder examinado: latitude [-23.5,-22.25], longitude
[-44,-42.5], resolução 2 km, imagem esperada 654 linhas × 656 colunas.
Se a configuração usada para gerar o cache for diferente, informar os parâmetros
--lat-range, --lon-range, --resolution-km, --image-height e --image-width.
A igualdade PNG/cache só testa o algoritmo e geometria informados; não é validação
independente da georreferência. Conferir se a imagem inteira corresponde à bbox,
se existem legendas/bordas e se esses parâmetros correspondem ao produto original.
O script extrai a classe do builder local, registra seu SHA256 e nunca chama
process_time (que reutilizaria/escreveria o cache).

## Seleção e custo

Analisa todas as horas com fração unique >=45 dBZ de pelo menos 10%, em todos os
anos; os picos dos top 10 globais e top 10 de 2024; dois controles por ano com
ocorrência >=45 abaixo desse corte; e vizinhos disponíveis ±1h.
10% é critério de triagem, NÃO limiar de exclusão. `--candidate-fraction` ajusta-o.
Os controles são os maiores abaixo do corte, não uma amostra climatológica aleatória.
Hashes de campos repetidos em toda a série vêm do CSV 14b; as comparações de
PNG/RGB/máscaras são restritas aos horários selecionados.
Verifica índices e hashes 14b contra o Zarr atual para rejeitar resultados obsoletos.
Gera até 40 painéis por padrão (`--max-panels`). Os CSVs cobrem todos os selecionados.

## Produtos e decisão

- source_checks.csv: disponibilidade, dimensões, comparação PNG/cache,
  cache/target/máscara, hashes, transparência e distância RGB à legenda.
- extreme_source_colors.csv: contagem das cores RGBA nos pixels que o dataset
  classifica como >=45. Permite identificar a cor responsável pelo padrão.
- repeated_fields_all_years.csv: campos completos repetidos com extremos, em toda
  a série (hash 14b inclui coordenadas, target e máscara).
- *_hash_repeats.csv: repetições de máscara extrema, RGB integral e RGB amostrado.
- PNGs: fonte original, amostragem, conversão, target, máscara extrema e distância RGB.
  Eixos são índices, não mapas geográficos independentes.
- manual_review.csv: registro humano, inicialmente pending para todos.
- summary.json: parâmetros, hash do builder, faltas e erros.

Uma máscara igual NÃO implica campo completo igual. Cor distante da legenda pode
ser interpolação/antialiasing; transparência pode esconder RGB arbitrário. São
indícios para revisão, nunca confirmação automática de corrupção.
PNG ausente deixa a origem inconclusiva mesmo quando cache e dataset coincidem.
Dimensões diferentes são registradas: o algoritmo atual recorta coordenadas por
clipping, que também é reproduzido para diagnosticar a origem da diferença.

Preencher manual_review.csv com decision: keep, confirmed_artifact ou inconclusive;
registrar reason, evidence (arquivo/painel/observação) e reviewer.
Critérios para concluir:
1. PNG meteorológico plausível e conversão fiel: manter; extremidade não é motivo de exclusão.
2. Texto, fundo, transparência ou erro de produto convertido em eco: documentar
   a assinatura visual e a regra técnica; aplicar a mesma regra a todos os anos.
3. PNG correto mas cache divergente: investigar versão do conversor, geometria,
   arquivo fonte e seleção temporal; não sobrescrever a referência.
4. Sem fonte ou evidência suficiente: manter inconclusivo e reportar sensibilidade.

Depois de confirmar a causa, implementar a correção em uma nova versão do cache
/dataset; preservar originais. Se só parte espacial estiver inválida, avaliar
máscara de validade em vez de eliminar automaticamente a hora inteira. Reavaliar
cobertura, estatísticas e baselines após a correção. Nunca escolher a regra para
melhorar métricas de 2023/2024. Este pacote produz evidências; não executa limpeza.

Enviar para interpretação: summary.json, source_checks.csv,
extreme_source_colors.csv e painéis de 2011-09-08, 2012-02-25,
2024-05-14/15 e 2024-08-30. Se faltarem PNGs, localizar a fonte antes de concluir.
