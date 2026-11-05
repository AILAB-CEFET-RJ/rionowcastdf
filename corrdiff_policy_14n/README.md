# 14n — Política do produtor sobre o cache intermediário

Esta entrega aplica as confirmações do Alerta Rio recebidas em 04–05/11/2026.
Ela não reconstrói o acervo, não altera o 14j, não substitui o inventário ERA5,
não modifica os splits e não interrompe a auditoria integral 14m.

## Decisões e alcance

1. **Base radar:** `data/radar_intermediate_14j_v1`, com fuso civil histórico
   `America/Sao_Paulo`, geometria de referência e classes preservadas.
2. **Base temporal/ERA5:** `analysis_outputs_restored/14k_pairing_v2` (o pareamento
   concluído), que já contém snapshots do inventário e manifesto radar.
3. **Nova camada:** `analysis_outputs_restored/14n_policy_v1`. Não usar o cache
   legado nem o dataset antigo como alvos desta política.
4. **Auditoria 14m:** manter rodando. Ela lê os PNGs em todos os minutos;
   este programa lê objetos decodificados do 14j, que são horários exatos.
   Podem concorrer por disco; se necessário, executar 14n após 14m por desempenho,
   mas não há dependência semântica de esperar a auditoria completa.
5. **Início do acervo:** dia local `2011-01-17`. O limite de inclusão por dia é
   `2011-01-17T02:00:00Z`, considerando o horário de verão histórico.
   O primeiro minuto efetivamente disponível NÃO foi informado. Portanto os
   horários ausentes dentro desse primeiro dia continuam ausentes.
6. Com o manifesto completo anterior de 122.736 horas, a exclusão do período
   anterior remove 386 horários e deixa **122.350 horários no período**. O fim
   permanece o do pareamento existente: `2024-12-31T23:00:00Z`, não meia-noite
   local. Não são acrescentadas as duas primeiras horas UTC de 2025.
   Caso existam objetos PNG antes do início declarado, o programa interrompe
   para revisão, em vez de excluí-los silenciosamente.

O manifesto novo mantém as linhas anteriores ao acervo com
`outside_archive_period`; os relatórios principais usam somente o período
válido como denominador. Isso torna a diferença com os relatórios antigos
reconciliável. Não é necessário repetir a extração do ERA5.

## Política por pixel e por imagem

| Entrada | Resultado |
|---|---|
| PNG transparente, legível, formato esperado | Frame sem eco exibido; pixels transparentes entram no suporte espacial declarado |
| Imagem com paleta e fundo transparente | Paleta decodificada + fundo sem eco; outros pixels continuam desconhecidos |
| Paleta no PNG, mas nenhuma cor selecionada na grade | Pontos transparentes da grade entram como sem eco; não significa que todo o PNG estava sem ecos |
| Opaque/alpha parcial não reconhecido | Desconhecido, fora da perda; não é convertido em seco |
| PNG ausente | Dado ausente; não se atribui automaticamente manutenção como causa |
| Erro de leitura, dimensão inesperada, múltiplos frames | Excluído/revisão |
| Sem paleta exata, mas não totalmente transparente | Continua em revisão; pode ser outro produto/formato |
| Cartografia/tela | Revisão por hash; não há liberação genérica |
| Horário local ambíguo por DST | Preservado no inventário, mas excluído dos candidatos pareados |

Nos NPZ derivados há:

- `observation_class`: 0 desconhecido; 1 sem eco exibido; 2 eco exibido.
- `observation_mask`, `echo_mask`, `no_echo_mask`: máscaras booleanas separadas.
- `displayed_dbz`: 20, 25, 30, 35, 40, 45 ou 50 somente nos pixels da paleta;
  NaN no fundo e nos desconhecidos.
- `target_code`: 0 para a categoria sem eco; classe dBZ para eco; NaN desconhecido.
- `source_sha256` e descrição da codificação.

**Zero é um código de categoria, não uma medição de 0 dBZ ou de chuva zero.**
A transformação final do alvo ainda será definida antes do treinamento científico.
Não se introduz `log1p`, conversão Z–R, média temporal ou média RGB nesta entrega.
O smoke test usa `target_code/50` apenas para verificar a execução técnica.

## Cobertura espacial: hipótese explícita

A resposta do produtor condiciona alfa zero à região efetivamente observada.
A `geometric_mask` do 14j delimita a geometria amostrada, não uma máscara oficial
de bloqueios, sombras ou validade por varredura.

Sem máscara externa, o programa funciona e gera resultados com
`coverage_state=provisional_geometric_region`. Usa a geometria como domínio
provisório e mantém `spatial_validity_confirmed=false` e `production_ready=false`.
Isso permite atualizar disponibilidade e testar o fluxo sem declarar que toda
transparência dentro do círculo é cobertura observacional certificada.

Se houver máscara de cobertura revisada, fornecer:

- NPZ com `observed_region_mask` booleano 70x84, `lat` e `lon` exatamente iguais
  aos do 14j. Máscara não vazia e contida na geometria.
- JSON com `semantics: effective_observed_region`, `evidence`, `reviewer` e
  `reviewed_at`. Documentar a fonte e o período de aplicabilidade.

Exemplo de argumentos adicionais:

```bash
--coverage-mask configs/sumare_observed_region.npz \
--coverage-metadata configs/sumare_observed_region.json
```

A revisão estática passa a ser registrada como `reviewed_static_region`; ela não
é uma avaliação da qualidade meteorológica de cada varredura. Alterar cobertura
ou regras exige uma nova saída; nunca modificar uma saída concluída em uso.

## Imagens com cartografia

A resposta do produtor confirmou ausência de refletividade nos exemplos de
25/02/2012 às 03h e 04h. A imagem das 04h é reconhecida pelo SHA256 no arquivo
`map_review_example.csv`, mas a decisão inicial é `pending`: o trecho recebido
não estabelece que toda tela cartográfica ou com mensagem seja uma varredura
válida sem eco.

O programa produz `map_review_queue.csv`, com um exemplo por hash, quantidade de
horas e a evidência já registrada. `map_decisions_template.csv` tem o formato
exato aceito na próxima execução. Decisões:

- `pending`: aguardar revisão; não usar como alvo.
- `reject`: não usar como alvo.
- `accept_no_echo`: revisão documenta imagem válida sem eco. Exige evidência,
  responsável e data. O programa ainda exige ZERO pixels de paleta meteorológica
  no PNG e aceita somente `map_screen_review` com dimensão e frame únicos esperados.

A aceitação vale para **todas as ocorrências dos mesmos bytes (SHA256)** no
manifesto. A evidência deve sustentar esse alcance. Uma aprovação apenas de um
horário não deve ser ampliada para todos os horários sem revisão.

Não se usa OCR ou cor bege para provar validade. A heurística antiga permanece
somente um mecanismo de seleção para revisão. Se houver uma cor exata da paleta
numa imagem cartográfica, a aprovação como sem eco é recusada: é necessário
esclarecer esse caso em vez de apagar o pixel.

Para aplicar decisões revisadas, copiar/editar o CSV e executar novamente em
`14n_policy_v2`, passando `--map-review caminho/do/csv`. Não editar os rótulos
originais do 14j e não mudar arquivos dentro de uma execução já concluída.

## Execução no servidor CEFET

No terminal Linux, dentro do repositório, após copiar e extrair o ZIP desta entrega:

```bash
cd /home/vsantos/repositories/rionowcastdf
conda activate dataset
unzip corrdiff_policy_14n.zip
python -m pip install -r corrdiff_policy_14n/requirements.txt
python -m unittest discover -s corrdiff_policy_14n/tests -v
mkdir -p logs
```

Aplicar a política (não precisa reler PNGs nem aguardar o fim do 14m):

```bash
nohup python -u corrdiff_policy_14n/apply_policy.py \
  --radar-cache-dir data/radar_intermediate_14j_v1 \
  --pairing-dir analysis_outputs_restored/14k_pairing_v2 \
  --archive-start-local 2011-01-17 \
  --map-review corrdiff_policy_14n/map_review_example.csv \
  --output-dir analysis_outputs_restored/14n_policy_v1 \
  > logs/14n_policy_v1.log 2>&1 &
echo "PID da política: $!"
```

É indispensável apontar o diretório 14k que contém o pareamento concluído.
Se no servidor ele tiver outro nome, trocar somente esse argumento.

Monitorar:

```bash
tail -n 30 logs/14n_policy_v1.log
cat analysis_outputs_restored/14n_policy_v1/progress.json
```

`progress.json` acompanha processamento de objetos, não a conclusão dos relatórios.
A conclusão está em `summary.json`. Saída 0: execução concluída sem erros de
objetos; 2: relatórios concluídos, mas houve erros de objetos; 1: erro de contrato
ou execução; 130: interrupção. Não avançar para smoke com erros de objetos.

Retomada, somente depois que a execução anterior parar: repetir o comando com
os mesmos argumentos e `--resume`, preferindo `>> logs/14n_policy_v1.log`.
Não usar `--resume` quando mudou cobertura, decisões, código ou arquivos de entrada;
nesses casos usar `14n_policy_v2` ou outro diretório novo.

### Novo smoke técnico

Depois de `complete=true` e `object_error_hours=0`:

```bash
nohup python -u corrdiff_policy_14n/smoke_policy.py \
  --policy-dir analysis_outputs_restored/14n_policy_v1 \
  --output-dir analysis_outputs_restored/14n_smoke_v1 \
  --max-timestamps 8 --steps 20 --batch-size 4 --device cpu \
  > logs/14n_smoke_v1.log 2>&1 &
echo "PID do smoke: $!"
```

A CPU é deliberada, diante do aviso anterior de incompatibilidade do driver CUDA.
A amostra técnica procura incluir horas sem eco na grade e horas com eco, sem
usar validação/teste. A seleção estratificada NÃO é o sampler do treino final.

O teste lê os 12 canais ERA5 dos arquivos já inventariados, verifica fingerprints,
interpola bilinearmente na grade, normaliza apenas as horas de treino selecionadas,
extrai patches e executa uma TinyRegressor. Preserva o layout antigo de 12 patches
32x32/stride16: últimas 6 linhas e 4 colunas ficam fora dos patches; não altera
esse protocolo simultaneamente à mudança de semântica.

Patches inteiramente sem eco agora são úteis. Patches sem qualquer pixel
supervisionável continuam excluídos. A perda ignora desconhecidos por indexação
antes da aritmética; não multiplica NaN por zero.

`technical_smoke_passed` verifica loader, máscaras, gradientes e atualização de
pesos. Não prova habilidade meteorológica, convergência, execução nativa CorrDiff
ou adequação da transformação final do alvo. Perdas inicial/final usam minibatches
que podem diferir. Não são métricas de validação.

## Produtos esperados

| Arquivo | Uso |
|---|---|
| `summary.json` | Conclusão, contagens e limitações explícitas |
| `policy_config.json` | Evidência do produtor, data inicial, máscara, revisões e hashes das entradas/código |
| `policy_manifest.parquet` / `.csv.gz` | Linhagem hora a hora e novos candidatos; colunas legadas permanecem como histórico |
| `split_coverage.csv` | Disponibilidade por split no período corrigido |
| `split_policy_status.csv` | Quantidade e percentual por tipo de imagem/política |
| `annual_policy_status.csv` | Distribuição anual |
| `period_reconciliation.csv` | Horas anteriores ao acervo separadas das ausências no acervo |
| `map_review_queue.csv` | Amostra por hash e quantidade de horas de cartografia |
| `map_decisions_template.csv` | Planilha de decisões para execução futura |
| `errors.csv` | Erros encontrados ao ler os objetos decodificados |
| `coverage.npz` | Geometria e domínio espacial efetivamente usado nesta execução |
| `targets/` | Objetos derivados por hash, sem duplicar os PNGs originais |
| `policy_objects.sqlite` | Checkpoint para retomada |

Usar `policy_pair_candidate` e `policy_train_candidate` para a nova seleção.
Os indicadores originais `paired_palette_candidate` e `train_smoke_candidate`
continuam no manifesto para rastreabilidade e NÃO incluem as recuperações.
`policy_pair_candidate` inclui disponibilidade em purge/teste para relatórios;
o loader usa somente `policy_train_candidate` e respeita os bloqueios.

As 17.325 horas transparentes anteriormente calculadas são uma referência de
recuperação no treino, não um total prometido da nova execução. Também podem
ser recuperadas horas com paleta presente no PNG, mas sem cor na grade, e pixels
transparentes nos frames com ecos. Os números novos virão dos relatórios.

## Organização do repositório

Não apagar nem mover entradas enquanto o 14m roda ou enquanto 14n/smoke leem
arquivos. O 14n aponta para o 14k original e para os objetos 14j; mover essas
pastas quebra a linhagem e as verificações. Manter dados grandes no servidor,
fora do versionamento, e scripts/relatórios pequenos no Git.

Gerar diagnóstico e acrescentar regras de ignore (sem apagar ou desversionar):

```bash
python corrdiff_policy_14n/repository_report.py \
  --repo-root /home/vsantos/repositories/rionowcastdf \
  --output-dir analysis_outputs_restored/14n_repository_review_v1 \
  --apply-gitignore
```

Esse comando só acrescenta regras ausentes ao `.gitignore`, produz a lista de
arquivos rastreados grandes/ignorados/Zone.Identifier e salva `git status`.
Não executa `git rm`, `git reset`, `git clean`, commit ou push.

Consultar `ORGANIZACAO.md` para os conjuntos ativos e históricos. `.gitignore`
não remove blobs já commitados. Se o push ainda falhar por histórico antigo,
tratar os commits locais separadamente, após examinar o relatório; não apagar
arquivos físicos nem repetir um reset genérico.

## Próxima etapa após a auditoria completa

O 14m continuará fornecendo a distribuição de todos os PNGs e sequências/gaps.
O 14n é exclusivamente horário exato; ele não contém todos os frames de 2 minutos.
Depois do inventário integral, comparar no treino: hora exata, frame elegível mais
próximo com tolerância documentada, janela causal curta e, se fizer sentido para
o alvo, janela centrada. Medir disponibilidade, proporção de ecos, classes e
sensibilidade; definir mínimos de cobertura de janela. Transparência válida conta
como observação; arquivo ausente não conta. Não calcular médias RGB ou tratar
somente os pixels com eco como denominador de uma média de chuva.

A agregação física depende da definição do alvo e do tratamento da categoria
abaixo do limiar. Manter teste bloqueado e impedir uso de futuro como condicionante
quando a tarefa exigir previsão causal. Implementar essa comparação após o 14m,
em entrega própria, sem mudar simultaneamente fuso, geometria e interpretação.

## O que enviar ao terminar

1. `14n_policy_v1/summary.json`.
2. `split_coverage.csv` e `split_policy_status.csv`.
3. `period_reconciliation.csv`.
4. `map_review_queue.csv` (ou apenas primeiras linhas e quantidade, se grande).
5. `14n_smoke_v1/summary.json` e o final do log.
6. Quando o 14m acabar: `summary.json`, `counts_by_year_status.csv`,
   `timestamp_quality.csv`, `dimensions.csv` e o pacote de exemplos.

Não enviar `targets/`, SQLite, Parquet ou tensores pelo GitHub.
