# Sequência de validação da referência e migração das fases 0–18

## Estado desta entrega

Executável agora: auditoria integral v2 (tensores, fontes selecionadas PNG e recálculo ERA5), lançador, rastreabilidade de execução e testes sintéticos. Os scripts científicos antigos e o adaptador CorrDiff **não foram migrados por esta entrega**. O plano abaixo explicita o trabalho necessário; não é uma declaração de fases aprovadas nem um script que execute automaticamente ferramentas incompatíveis.

A inspeção encontrou `np.expm1`/limiares `log1p` em várias fases; `adapter/conditioning.py` exige target log1p; `adapter/rionowcast_corrdiff_dataset.py` documenta máscara totalmente válida; e a geração da fase 18 aplica expm1. Essas hipóteses não são válidas para o novo Zarr. O TinyRegressor não testa a interface/loss do CorrDiff nativo.

## Etapa A — integridade (antes da fase 0)

Execute AUDITORIA_INTEGRAL.md. Critério: execução sem erro, integridade aprovada e modos completos confirmados. Em caso de erro, localizar a primeira divergência na cadeia fonte → cache/staging → patch; preservar a saída original e corrigir em versão nova se houver mudança de dados/contrato. Não excluir timestamps para apenas obter PASS.

## Etapa B — leitor comum e regressões numéricas

Implementar um leitor compartilhado das análises que:

- Abre os quatro splits explícitos; verifica transformação, canais, shapes e contrato.
- Expõe input em unidades físicas para diagnóstico e input padronizado somente quando solicitado para treino.
- Expõe alvo armazenado, código `target*50` e máscara separadamente; zero observado e zero inválido não se confundem.
- Mantém metadados de referência/fonte UTC, offset, hash e posição dos patches.
- Faz comparação de limiares no domínio armazenado float32, por exemplo `target >= np.float32(30/50)`, para evitar fronteiras alteradas por arredondamento.
- Suporta processamento em blocos, reconstrução de horas/campos e multiplicidade espacial. Um pixel coberto por vários patches não recebe vários votos na climatologia espacial deduplicada.
- Impede que treino use teste/purge e que normalização seja refeita em validação/teste.

Testes devem incluir classes 0/20/25/30/35/40/45/50, máscara parcial, imagens sem eco, gaps, offset de ±120 s, bordas e sobreposição dos patches, ausência de mistura entre splits e dupla normalização. Verificar contagens e resultados numéricos com um fixture independente.

## Etapa C — fases 0 a 4

| Fase | Migração necessária | Saídas a conferir |
|---|---|---|
| 0 | Leitor novo; quatro splits; target code/50; metadados temporais; máscara; compatibilidade com resultado da integridade | Shapes, canais, horas, patches, suporte, classes, gaps, hashes e contratos |
| 1 | Estatísticas em unidades físicas; alvo codificado e categorias; distinção entre distribuição de patches e horas únicas | Médias, desvios, quantis, histogramas e frequências sem eco/eco |
| 2 | Derivações a partir de canais físicos, níveis e unidades explícitos; valores inválidos mascarados | Campos derivados e controle de plausibilidade |
| 3 | Relações conjuntas com mesma população e suporte; sem inversão expm1 | Correlações, relações condicionais e contagens efetivas |
| 4 | Limiares do produto e quantis de treino; evitar classificar sem eco como 0 dBZ medido | Frequência de extremos, eventos, limiares, denominadores |

Executar 0 primeiro; 1–2 podem ser preparados após aprovação do leitor; 3–4 dependem das convenções estatísticas acima. Publicar relatórios novos em analysis_outputs_reference/, separados por versão e split.

## Etapa D — conferência antecipada da fase 15

A referência já tem splits congelados. Adaptar a fase 15 para validá-los e exportar manifests correspondentes à ordem real dos patches dos Zarrs, sem aplicar novamente o particionamento histórico. Conferir ranges temporais, purge, fonte selecionada, contagens por conjunto e ausência de sobreposição indevida.

Não embaralhar patches da mesma hora entre conjuntos. Duplicidade de conteúdo de imagens transparentes em datas diferentes, sozinha, não comprova vazamento; comparar a identidade temporal/de aquisição.

## Etapa E — fases 5 a 14

| Fase | Pré-requisito e regra | Resultado esperado |
|---|---|---|
| 5 Sazonalidade | 0/1; disponibilidade por mês/ano e denominadores válidos | Ciclos sazonais e efeito da cobertura desigual |
| 6 Ciclo diurno | 0/1; UTC e America/Sao_Paulo histórico, com DST | Frequências/estatísticas por hora e disponibilidade |
| 7 Defasagens | Leitor temporal; somente pares separados pelo intervalo exato definido | Curvas de dependência, pares válidos, gaps excluídos |
| 8 Persistência | 7; baseline sem acesso a observações futuras | MAE/RMSE e métricas de evento mascaradas por horizonte |
| 9 Estrutura espacial | Grade e máscara; evitar duplicar patches sobrepostos | Dependência espacial e anisotropia na geometria de referência |
| 10 ERA5–radar espacial | 2/9; escalas e interpolação explícitas | Relações espaciais condicionadas e suporte efetivo |
| 11 Escalas | 9; distância angular vs distância física e amostragem | Diagnóstico de escalas sem atribuir resolução instrumental de 2 km |
| 12 Multivariada | 1–3; ajustes somente no treino | Componentes/representações e projeção dos outros conjuntos |
| 13 Regimes | 12; agrupamento fit no treino, aplicação sem refit | Regimes, frequências, transições e composição dos conjuntos |
| 14 Shift | Splits da 15; diagnósticos 1–13 | Diferenças entre treino/validação/testes e cobertura |

Essas análises podem explorar os testes para caracterização previamente definida, mas resultados dos testes não devem orientar escolha de hiperparâmetros, limiares aprendidos ou seleção de modelos. Registrar a política de avaliação antes da comparação final.

## Etapa F — fase 16: baselines e handoff

Adaptar artefatos de normalização, índices e saída; nenhuma estrutura denominada target_log1p pode transportar silenciosamente code/50. Baselines climatológicos/estatísticos são ajustados só no treino. Persistência requer horizonte e pares exatos; para downscaling contemporâneo, definir e justificar os baselines apropriados ao problema, sem tratar automaticamente o experimento como previsão futura.

As perdas e métricas usam a máscara. Reportar avaliação por pixel único/hora além de métricas ponderadas por patches quando relevante. Documentar denominadores para eventos raros. Gerar o handoff com canais, normalização, transformação, geometrias, splits, hashes e versão do leitor.

## Etapa G — fase 17: ablações

Construir manifesto de casos versionado: identificador, canais, tipo de modelo (regressão/difusão/ambos), seeds, orçamento, conjunto de validação, normalização e checkpoints. Todos os casos comparáveis devem usar a mesma população de referência, ou registrar explicitamente sua diferença. A paralelização em máquinas diferentes exige identidade dos dados, ambiente e artefatos de cada caso; não deve produzir splits diferentes por máquina.

## Etapa H — fase 18: CorrDiff nativo

1. Inspecionar e fixar o commit NVIDIA/PhysicsNeMo usado, os entrypoints e assinaturas da API de dataset/loss.
2. Registrar driver, GPU, CUDA, PyTorch e dependências. Corrigir a incompatibilidade de driver já observada antes de iniciar jobs GPU.
3. Implementar adaptador do novo contrato com indexação por split, canais selecionados e normalização única.
4. Integrar máscara à perda real de regressão **e** de difusão. A API nativa pode não transportar máscara; nesse caso, adaptar explicitamente o lote/loss. Não substituir máscara parcial por all-valid nem simplesmente zerar pixels inválidos e usar MSE comum.
5. Executar carregamento e forward; verificar shapes, valores finitos e semântica do alvo. Isso ainda não é um teste de treinamento.
6. Treinar regressão curta em treino, validar em validação, salvar e recarregar checkpoint; comprovar gradientes, mudança de parâmetros e losses finitas.
7. Executar difusão residual usando regressão e condicionamento corretos; mesma máscara e normalização.
8. Gerar amostras, testar seeds, salvar resultados, recuperar escala por multiplicação por 50 e avaliar com máscara. Não usar expm1 nem impor arredondamento às classes sem uma regra de avaliação previamente definida.
9. Testar retomada e registrar resumo verificável. Só então iniciar os experimentos formais da fase 17 e avaliar testes bloqueados ao final.

## Entregas e evidências

Cada migração precisa de código, testes numéricos, CLI documentada e relatório de execução sobre a referência. Estados recomendados: implementado, testado sinteticamente, executado na referência, revisado. Não promover um estado automaticamente ao seguinte.

A auditoria integral é executável nesta entrega. A migração científica das fases 0–18 é trabalho separado ainda pendente, conforme os blocos B–H. O fechamento da geometria física não é uma consequência automática do PASS técnico.

## Inspeção nativa que pode ser feita enquanto a auditoria roda

Localize o diretório que contém o `train.py` do exemplo CorrDiff:

```bash
find /home/vsantos/repositories -type d -name .git -prune -o \
  -type f -path '*/corrdiff/train.py' -print
```

Use o diretório encontrado no lugar de `/caminho/do/exemplo/corrdiff`:

```bash
mkdir -p analysis_outputs_reference/native_context_v1
python scripts/18_inspect_nvidia_corrdiff_checkout.py \
  --corrdiff-dir /caminho/do/exemplo/corrdiff \
  --output analysis_outputs_reference/native_context_v1/checkout.json
nvidia-smi > analysis_outputs_reference/native_context_v1/nvidia-smi.txt
```

Esse inspetor preexistente compara também com o contrato antigo da fase 18. Seu status não certifica compatibilidade com o novo dataset; o objetivo agora é obter commit, caminhos e interfaces. Envie checkout.json e nvidia-smi.txt para orientar a migração nativa. Não execute treinamento ou geração antigos sobre o Zarr de referência.
