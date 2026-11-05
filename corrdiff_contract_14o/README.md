# 14o — contrato espacial e numérico — revisão passo 8

Esta entrega usa o snapshot **14n** como fonte dos alvos. Não reconstrói os PNGs,
não altera 14j/14k/14n e não aplica conversão Z–R para chuva. A auditoria 14m pode
continuar: ela não é uma dependência desta preparação.

## Decisão numérica

| Estado | Alvo científico `target_code` | Tensor da rede | Máscara |
|---|---:|---:|---:|
| Sem eco, segundo a política do produtor | 0 | 0 | verdadeira |
| Cor meteorológica de 20 dBZ | 20 | 0,4 | verdadeira |
| Classes 25, 30, 35, 40, 45 e 50 dBZ | classe correspondente | classe / 50 | verdadeira |
| Pixel desconhecido ou fora do suporte adotado | NaN | 0, somente preenchimento | falsa |

**Zero é um código de ausência de eco, não uma medição de 0 dBZ nem de chuva
igual a zero no solo.** O produto tem classes de exibição; não recuperamos a
refletividade contínua original, os valores abaixo de 20 dBZ ou valores acima
da última classe. O limite superior de uma classe não é inventado.

A regressão contínua penaliza erros no espaço de códigos: o salto de 0 para 20
é maior que o de 20 para 25. Isso é uma escolha explícita do baseline. Uma
alternativa futura é separar ocorrência e intensidade; ela não foi introduzida
agora porque alteraria a arquitetura e o objetivo do experimento.

As previsões contínuas permanecem sem recorte antes da perda. Multiplicar por
50 devolve **escores na escala do alvo codificado**. Valores previstos entre 0 e
20 não passam a ser medições físicas de refletividade sub-limiar. Para uma
visualização categórica derivada, `nearest_category` escolhe o código mais
próximo; empates vão para o menor. Por exemplo, escores >10 e <22,5 são atribuídos
a 20. Essa fronteira é uma regra de decodificação do modelo, não o limiar físico
do radar. Não aplicar essa quantização às métricas contínuas sem identificá-la.

## Contrato espacial

1. Herda exatamente as matrizes latitude/longitude e os checksums de 14j via
   14n. Não usa uma nova posição presumida para a antena. A grade é 70×84.
2. Mantém a amostragem já feita no PNG: pixel nativo mais próximo, sem média,
   máximo ou interpolação entre classes. Portanto, `valid_no_echo_on_sampled_grid`
   significa ausência de eco **nos pontos amostrados**, não ausência em toda a
   área de cada célula ou em toda a ROI. Ecos pequenos entre os pontos continuam
   fora desta representação, como no cache existente.
3. A referência 14j adotada no projeto usa projeção azimutal equidistante
   esférica, antena em −22,955139°, −43,248278°, centro de pixel (327,5;326,5),
   PNG de 656×654 e alcance de referência de 138,9 km. O código desta entrega
   **não recalcula** esses parâmetros: verifica e herda a geometria armazenada.
4. A grade de destino usa incremento angular nominal 2/111 grau; não deve ser
   descrita como células metricamente exatas de 2 km × 2 km. O espaçamento
   leste–oeste depende da latitude. As matrizes salvas são a definição operacional.
5. ERA5 é interpolado linearmente em latitude/longitude para cada ponto da
   grade, sem extrapolação. Coordenadas duplicadas, valores não finitos e grades
   incompletas interrompem a preparação.
6. Herda a máscara de suporte 14n. Se `coverage_state` for
   `provisional_geometric_region`, o resultado continua condicionado a essa
   hipótese. Estar dentro da geometria não certifica observação física de todos
   os pixels. A resposta do produtor sobre alfa zero resolve a semântica em
   região observada; não desenha por si só os limites dessa região.
7. Patches de 32×32, passo nominal 8, incluindo bordas: inícios de linha
   0,8,16,24,32,38 e de coluna 0,8,16,24,32,40,48,52. São até 48 patches
   por hora e cobertura geométrica completa de 70×84, sem padding. O último
   deslocamento é menor que 8 para alinhar a borda. Isso não certifica validade
   observacional: a máscara permanece obrigatória. A estatística ERA5 usa cada
   ponto da grade inteira uma vez por hora, antes dos patches.
8. Cada patch com ao menos um pixel suportado entra no índice. Não se descarta
   patch sem eco nem se faz oversampling de chuva. `fully_supported` permite
   comparar apenas patches integralmente suportados.

O horário continua sendo a hora UTC exata pareada na 14n, com nomes locais do
radar convertidos segundo `America/Sao_Paulo`, inclusive horário de verão.
Não há agregação temporal nesta versão. As 18 horas locais ambíguas continuam
submetidas à exclusão registrada no pareamento. O início do acervo é herdado
da política: 17/01/2011 local, limite operacional 02:00 UTC; não se afirma que
00:00 local seja o primeiro minuto efetivamente disponível.

## Cartografia e outros casos

A 14o respeita `policy_pair_candidate`. As 166 horas `map_review_pending` e as
1.207 `excluded_no_exact_palette_review` não são liberadas automaticamente.
Um mapa específico aprovado por hash na 14n será herdado por uma nova execução
14o baseada naquele novo snapshot. A informação sobre o exemplo de 25/02/2012
não é extrapolada a todas as imagens cartográficas. Ausentes, erros de leitura,
pixels opacos desconhecidos e alfa parcial não recebem o rótulo sem eco.

## O que cada programa faz

| Programa | Atividade | Resultado |
|---|---|---|
| `prepare.py` | Verifica snapshots, materializa ERA5 mensal, lê alvos 14n, calcula estatísticas e indexa patches | Novo dataset 14o imutável |
| `check_data.py` | Percorre uma amostra determinística de patches reais de treino | Relatório do loader, sem alegar teste nativo |
| `dataset.py` | Lê campos por memória mapeada e aplica normalização | 12 canais ERA5, alvo, máscara, coordenadas |
| `native_smoke.py` | Treina brevemente modelos nativos de regressão e difusão residual em sequência | Checkpoints diagnósticos e relatório de compatibilidade |
| `tests/test_contract.py` | Testes sintéticos com Parquet/SQLite e máscaras | Evidência local de integridade do contrato |

Os Parquets são materializados por arquivo e conjunto de horas do mês no DuckDB,
em vez de reabertos para cada canal de cada hora. A memória do DuckDB é limitada
a 1 GB, com possível uso de disco temporário; isso não é um limite global de RAM
do processo. Os tensores mensais `.npy` não são comprimidos e são mapeáveis em
memória. A retomada valida os meses completos e refaz somente meses incompletos.
Se mudarem política, configuração, código ou arquivos-fonte, usar outro diretório.

Cada hora de treino e ponto da grade entra uma vez nas estatísticas ERA5;
validação e testes não entram. São média e desvio populacional, por canal, com
acumulação em float64. Canal com desvio <1e−8 usa escala 1 e é registrado como
constante. Não se força uma distribuição de chuva/sem eco na amostragem.

## Instalação no servidor

Extraia o ZIP na raiz de `/home/vsantos/repositories/rionowcastdf`, resultando
na pasta `corrdiff_contract_14o`. Use o ambiente `dataset` já utilizado.

```bash
cd /home/vsantos/repositories/rionowcastdf
python -m pip install -r corrdiff_contract_14o/requirements.txt
python -m unittest discover -s corrdiff_contract_14o/tests -v
mkdir -p logs
```

O ambiente deve ter PyTorch para os testes/loader; a preparação por si só não
importa PyTorch. Não atualize automaticamente PyTorch, CUDA ou o driver para
executar esta etapa. O aviso de incompatibilidade CUDA anterior não exige
alterar o ambiente usado para preparar os dados.

## Primeiro: piloto com dados reais

```bash
nohup python -u corrdiff_contract_14o/prepare.py \
  --policy-dir analysis_outputs_restored/14n_policy_v1 \
  --output-dir data/corrdiff_14o_pilot_s8_v1 \
  --pilot-hours 8 \
  > logs/14o_pilot_s8_v1.log 2>&1 &
```

São até oito horas de treino distribuídas pelo período. Não é amostragem
estratificada chuva/sem eco. As estatísticas do piloto são diagnósticas e ficam
marcadas `pilot_only=true`. Não reutilizá-las no experimento completo.

Após `complete=true`:

```bash
python corrdiff_contract_14o/check_data.py \
  --data-dir data/corrdiff_14o_pilot_s8_v1 \
  --output-dir analysis_outputs_restored/14o_loader_pilot_s8_v1
```

## Depois: preparação completa

```bash
nohup python -u corrdiff_contract_14o/prepare.py \
  --policy-dir analysis_outputs_restored/14n_policy_v1 \
  --output-dir data/corrdiff_14o_s8_v1 \
  > logs/14o_prepare_s8_v1.log 2>&1 &
```

Não rodar piloto e preparação completa simultaneamente se disputarem disco.
Para retomar uma execução interrompida, repita os mesmos argumentos acrescentando
`--resume` e usando `>>` no log. Uma saída já completa é imutável e recusará
nova execução. Não apague o snapshot para resolver esse aviso.

Com os totais que você enviou, esperamos **90.962 horas**, excluindo as 186
candidatas da faixa de purga: treino 70.271; validação 13.912; teste principal
4.433; teste OOD 2.346. O programa calcula os totais reais, sem fixá-los no código.
São até 4.366.176 patches; patches sem nenhum suporte são omitidos.

Os campos principais ocupariam aproximadamente **28,35 GB decimais** para
90.962 horas (12 canais float32, alvo float32, máscara bool, grade 70×84), além
dos índices, arquivos temporários e relatórios. Não é necessário armazenar
uma cópia física dos patches sobrepostos.

## Teste com os modelos nativos

O código nativo consultado foi `NVIDIA/physicsnemo`, commit
`42fa19c3fa4c80d76a621b2454511326cde8a177`:

- `examples/weather/corrdiff/train.py`;
- `physicsnemo/models/diffusion_unets/unet.py`;
- `physicsnemo/diffusion/preconditioners/legacy.py`;
- `physicsnemo/diffusion/metrics/legacy_losses.py`.

A perda padrão inspecionada não recebe nossa máscara. Por isso, esta entrega
usa **os modelos nativos em um loop próprio**, não modifica silenciosamente o
`train.py` da NVIDIA. Não registra este teste como integração do entrypoint
padrão. Usa `CorrDiffRegressionUNet` e `EDMPrecondSuperResolution`, com SongUNet
pequena e FP32; não há fallback para TinyRegressor se o import falhar.

O condicionamento desse teste tem **15 canais**: 12 ERA5 normalizados, duas
coordenadas estáticas escaladas e uma máscara de suporte. A máscara informa o
domínio onde o alvo é observado; não fornece valores de refletividade. Para
patches parciais, o resíduo e o ruído ficam em zero fora dela; somente os pixels
suportados participam da perda. Isso define um modelo condicionado também ao
suporte: não se deve ignorar esse canal na futura geração. A representatividade
do suporte e seu possível viés de seleção ainda devem constar da metodologia.

O teste roda primeiro regressão e depois difusão do resíduo usando a regressão
congelada. `sigma_data=0.5` é hiperparâmetro explícito do teste, não estatística
estimada dos resíduos finais. O número reduzido de passos não treina um modelo
cientificamente utilizável nem verifica um sampler de geração.

Em um ambiente que já tenha essa API PhysicsNeMo disponível:

```bash
python -u corrdiff_contract_14o/native_smoke.py \
  --data-dir data/corrdiff_14o_pilot_s8_v1 \
  --output-dir analysis_outputs_restored/14o_native_pilot_s8_v1 \
  --device cpu --steps 20 --batch-size 2
```

Esse comando pode ser usado com `nohup` e redirecionamento para
`logs/14o_native_pilot_s8_v1.log`. CPU é o padrão por causa do aviso CUDA anterior.
Se o import nativo falhar, envie o erro e a versão/checkout instalado; não
instale a versão mais recente sobre o ambiente de produção às cegas. Uma
execução opcional com `--fully-supported-only` isola a compatibilidade dos
modelos dos efeitos de borda das máscaras.

O teste nativo **não foi executado neste ambiente de entrega**, onde não há
PhysicsNeMo instalado. A compatibilidade foi implementada contra o código
inspecionado e deve ser comprovada no ambiente de execução. `production_ready`
continua falso mesmo depois de passar: cobertura, protocolo de treino completo
e avaliação da geração ainda não estão certificados.

## Resultados para enviar

Depois do piloto, envie:

1. `data/corrdiff_14o_pilot_s8_v1/summary.json`;
2. `analysis_outputs_restored/14o_loader_pilot_s8_v1/summary.json`;
3. relatório ou traceback do teste nativo, se executado.

Depois do build completo, envie:

1. `data/corrdiff_14o_s8_v1/summary.json`;
2. `data/corrdiff_14o_s8_v1/normalization.json`;
3. `data/corrdiff_14o_s8_v1/patch_summary.csv`.

Não envie os `.npy`, SQLite ou Parquets grandes ao GitHub. Preserve os relatórios
pequenos em uma pasta de análises versionável. Para evitar repetir o bloqueio
de push, acrescente ao `.gitignore` (sem excluir arquivos do disco):

```gitignore
/data/corrdiff_14o*/
/analysis_outputs_restored/14o_native_*/**/*.pt
```

Arquivos já rastreados não deixam de ser rastreados apenas com `.gitignore`.
Esta entrega não reescreve histórico Git nem remove as auditorias anteriores.

## Atualização de uma execução anterior com passo 16

Se o build anterior ainda estiver rodando, deixe-o terminar e não substitua os
scripts durante a execução. Os campos horários completos continuam utilizáveis.
Não use `--resume` com o código novo sobre um build criado com o código antigo:
o contrato imutável foi alterado.

Depois que o build antigo terminar, extraia esta revisão e crie um novo índice:

```bash
nohup python -u corrdiff_contract_14o/reindex.py \
  --prepared-dir data/corrdiff_14o_v1 \
  --output-dir data/corrdiff_14o_s8_v1 \
  > logs/14o_reindex_s8_v1.log 2>&1 &
```

Ajuste `--prepared-dir` se o diretório anterior tiver outro nome. Para o piloto,
use os respectivos caminhos de piloto. Execute **ou** a reindexação de um build
concluído **ou** a preparação nova; não ambos para o mesmo destino.

A reindexação não acessa PNGs ou os Parquets ERA5 originais: verifica os shards
preparados, refaz o índice e recalcula as mesmas estatísticas de treino, exigindo
igualdade com a normalização anterior. Portanto ainda há leitura de disco.
Os shards são referenciados por links simbólicos: **mantenha o diretório antigo**.
O destino reindexado depende dele e não deve ser copiado sozinho para outra máquina.
Não há duplicação física dos campos mensais. Se interrompida a reindexação, use
outro diretório novo para repetir; ela não possui retomada própria.

O loader lê as coordenadas do índice e aceita os novos recortes sem alterar
as dimensões dos tensores. A quantidade de recortes quadruplica em relação aos
12 da versão original, mas não a quantidade de observações independentes.
O índice Parquet e a RAM usados para construí-lo aumentam; os campos mensais
continuam com o mesmo tamanho. O limite DuckDB de 1 GB não limita a RAM do índice.

## Zarr e reavaliação das fases 0–14

Esta revisão continua produzindo NumPy + Parquet, não Zarr. Não execute scripts
que esperam Zarr diretamente nessa estrutura. A exportação com o esquema exato
esperado pelas auditorias anteriores e a adaptação de suas suposições numéricas
são etapas distintas, ainda não incluídas aqui.

As auditorias aplicáveis ao dataset final devem ser reexecutadas em nova pasta
versionada. Mudaram os horários pareados, a geometria, as máscaras, a semântica
de sem eco, os candidatos e a codificação do alvo. Relatórios anteriores não
certificam a nova versão. Preserve-os como histórico de diagnóstico.

Antes da reexecução, verificar nos scripts: ausência de inversão `expm1` do
antigo alvo log1p; limiares meteorológicos aplicados a `target_code` ou a 50 vezes
o tensor normalizado; exclusão dos pixels mascarados; normalização ajustada
somente no treino; splits e purga preservados; estatísticas por hora/pixel único
separadas das estatísticas ponderadas pelos patches sobrepostos.

As avaliações espaciais devem recompor a grade e combinar previsões
sobrepostas, usando uma contagem de contribuições por pixel. Não interpretar
cada ocorrência repetida de um pixel nos patches como observação independente.
O passo 8 dá cobertura completa, mas não resolve o contexto espacial do patch:
32×32 continua sendo uma escolha de baseline a verificar na validação.
