# Entrega 3 — 14l: teste técnico de dados, máscaras e otimização

## O que esta entrega comprova

Teste pequeno, restrito ao TREINO, para exercitar o caminho:
manifesto14k → fontes ERA5 → interpolação → cache radar14j → patches → tensores
PyTorch → forward → loss mascarada → backward → atualização de pesos.

Pode usar uma fotografia parcial do cache. A rede é uma CNN pequena de diagnóstico,
NÃO o CorrDiff nativo. O teste não valida a difusão, loss de denoising, arquitetura
NVIDIA, desempenho meteorológico, generalização ou convergência. O relatório deixa
`corrdiff_native_tested=false` e `production_ready=false`.

O suporte da loss é somente a paleta opaca reconhecida no cache. Isso não é uma
máscara completa de validade observacional. Nenhum pixel transparente é declarado
seco ou convertido em observação de chuva zero. O treino neste suporte é condicionado
à presença de paleta e é enviesado para os pixels coloridos; não serve como modelo
final de ocorrência ou ausência de precipitação.

## Dependências e instalação

No ambiente dataset, na raiz do repositório CEFET:

```bash
cd /home/vsantos/repositories/rionowcastdf
unzip corrdiff_smoke_14l.zip
python -c 'import torch; print(torch.__version__)'
```

Se torch ainda não estiver instalado, para este teste CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Se já existir uma instalação PyTorch funcional, preserve-a. CPU é o padrão; este
teste não requer GPU ou substituição da instalação CUDA existente.

```bash
python -m pip install -r corrdiff_smoke_14l/requirements.txt
mkdir -p logs
```

`pytz` está explicitamente incluído, corrigindo a omissão de dependência da entrega14k.
O pacote é autônomo; não precisa ser importado pelo builder original.

## 1. Criar uma fotografia parcial pelo14k

O inventário ERA5 deve ter terminado sem erros. O cache radar pode continuar ativo:

```bash
python -u corrdiff_pairing_14k/pair_inventory.py pair \
  --radar-cache-dir data/radar_intermediate_14j_v1 \
  --inventory-dir analysis_outputs_restored/14k_era5_inventory \
  --output-dir analysis_outputs_restored/14k_pairing_preview_01 \
  --allow-partial \
  > logs/14k_pairing_preview_01.log 2>&1
```

```bash
cat analysis_outputs_restored/14k_pairing_preview_01/summary.json
```

Verifique `train_smoke_candidate_hours`. Se zero, não haverá dados para o teste:
ainda podem faltar horas processadas do radar, canais ERA5 ou cobertura técnica.
Não altere as flags para contornar esse bloqueio. Se houver canais ausentes ou
sobreposição de arquivos, consulte channel_coverage.csv e era5_files.csv do14k.

O pareamento pode ser mais demorado que o smoke test, porque exporta o período
inteiro. Esta fotografia não se atualiza sozinha. Para outra fotografia, use um
novo diretório. As cópias SQLite são consistentes mesmo com o14j em execução.

## 2. Executar o teste técnico agora

```bash
python -u corrdiff_smoke_14l/smoke_test.py \
  --pairing-dir analysis_outputs_restored/14k_pairing_preview_01 \
  --radar-cache-dir data/radar_intermediate_14j_v1 \
  --output-dir analysis_outputs_restored/14l_smoke_preview_01 \
  --allow-provisional \
  --max-timestamps 8 \
  --steps 20 \
  --batch-size 4 \
  --device cpu \
  > logs/14l_smoke_preview_01.log 2>&1
```

```bash
tail -n 40 logs/14l_smoke_preview_01.log
cat analysis_outputs_restored/14l_smoke_preview_01/summary.json
cat analysis_outputs_restored/14l_smoke_preview_01/training_steps.csv
```

Há limite de64timestamps e1000passos para manter o escopo diagnóstico. O padrão
usa8horas e20passos. A seleção é determinística e distribuída cronologicamente
entre candidatos técnicos de treino disponíveis na fotografia; não seleciona casos
por intensidade nem consulta alvos de validação/teste. `--seed` controla o modelo
e a ordem de batches. CPU é suficiente para esta rede pequena.

`--device cuda` é opcional e exige PyTorch com CUDA funcional. Não é necessário
para validar o caminho de dados e loss.

## 3. Executar novamente após o cache completo

Crie o pareamento final14k em um diretório novo, sem --allow-partial, conforme a
segunda entrega. Então execute:

```bash
python -u corrdiff_smoke_14l/smoke_test.py \
  --pairing-dir analysis_outputs_restored/14k_pairing_v1 \
  --radar-cache-dir data/radar_intermediate_14j_v1 \
  --output-dir analysis_outputs_restored/14l_smoke_final_01 \
  --max-timestamps 8 --steps 20 --batch-size 4 --device cpu \
  > logs/14l_smoke_final_01.log 2>&1
```

Saídas são imutáveis: se o diretório já existir, use outro. Uma falha após a criação
da saída deixa summary.complete=false. Não há retomada dos20passos: execute novamente
em outra saída. Os arquivos de origem não são modificados.

## Atividades implementadas

### Seleção e rastreabilidade

- Lê o paired_manifest.parquet e recusa fotografia parcial sem autorização explícita.
- Exige inventário ERA5 sem erros.
- Seleciona somente split=train, locked=false, train_smoke_candidate=true,
  era5_technical_candidate=true, horário local não ambíguo e radar com paleta na grade.
- Revalida anos2011–2020 e a purga anterior à fronteira2021-01-01, mesmo se alguém
  editar indevidamente as flags do manifesto.
- Confere hashes do contrato14j, geometry.npz e snapshot de inventário ERA5.
- Localiza os arquivos ERA5 reais pela tabela metrics do snapshot14k, não por
  suposição de nome mensal. Exige uma única fonte por canal/horário.
- Confere tamanho e mtime_ns das fontes ERA5 antes/depois da leitura. Essa identidade
  é a do inventário14k; não é hash integral dos Parquets. Não modifique o acervo.
- Confere SHA256 de cada NPZ radar selecionado contra o manifesto, formatos e máscara
  geométrica. Valida correspondência entre classe de paleta e valor dBZ.

### Tempo e interpolação do condicionamento

- UTC exato: não busca hora vizinha, não aplica deslocamento adicional de fuso.
- Reutiliza nomes de colunas/aliases e unidade de pressão registrados pelo14k.
- Lê tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500 nessa ordem.
- Converte longitudes0–360 para−180–180, ordena eixos e reconstrói a matriz retangular.
- Recusa coordenadas duplicadas, malha incompleta e qualquer valor não finito.
- Aplica interpolação linear bidimensional nos eixos latitude/longitude do ERA5,
  avaliada nos pontos de geometry.npz. Em uma célula retangular, corresponde à
  interpolação bilinear dos quatro nós vizinhos.
- Não extrapola além das coordenadas ERA5 e não transforma unidades físicas.
- A grade radar já reamostrada pelo14j não é interpolada de novo. O teste exige70×84.

Não há média temporal. Interpolar o ERA5 torna seus valores consultáveis nos pontos
da grade radar; não acrescenta informação meteorológica em alta resolução.

### Normalização e patches

- Calcula média/desvio por canal usando apenas campos completos das horas de treino
  selecionadas, antes de criar patches sobrepostos.
- Canal praticamente constante recebe escala1 e é registrado.
- Essa normalização é EXCLUSIVAMENTE diagnóstica, ajustada em poucas horas; não deve
  ser promovida para a normalização do treinamento completo.
- Extrai patches32×32 com stride16:12posições possíveis na grade70×84.
- Como no recorte por janelas inteiras, as últimas6linhas e4colunas ficam fora dos
  patches. Se todo o suporte de um caso estiver nessas bordas, nenhum patch dele entra.
- Registra todas as posições em patch_audit.csv; descarta patches sem suporte para loss.
- Não impõe um limiar meteorológico mínimo de cobertura. Um pixel é suficiente para
  o teste mecânico; isso não significa que seja suficiente para treino científico.

### Target e loss

- target_raw_dbz preserva NaN fora do suporte.
- Suporte = classe de paleta reconhecida + alpha255 + geometria + dBZ finito.
- target_tensor contém dBZ/50 nos pixels com suporte e zero como placeholder nos demais.
- Zero fora do suporte NÃO é rótulo seco; a máscara exclui esses elementos antes da
  subtração da loss. Não se usa `(NaN * 0)`, que continuaria NaN.
- A loss é média do erro quadrático sobre todos os pixels com suporte no batch.
  Patches com mais suporte têm mais peso nessa média por pixel.
- A escala dBZ/50 é de diagnóstico. Não converte refletividade para chuva e não
  substitui a decisão sobre a transformação log1p do target de produção.

### Otimização

- CNN12→16→16→1, com convoluções3×3, ativaçõesSiLU e saída1×1.
- DataLoader de patches de treino; Adam, learning rate1e-3.
- Verifica forward finito, loss finita, gradientes finitos e parâmetros finitos.
- Confirma que pelo menos um parâmetro mudou após os passos.
- Salva state_dict apenas para rastrear o teste, sem promover o modelo a produção.
- Não exige loss monotonicamente decrescente: batches mudam entre passos. O valor
  final não é métrica de validação e não deve ser interpretado como desempenho.

## Artefatos e resultados esperados

| Arquivo | Conteúdo |
|---|---|
| summary.json | Conclusão, shapes, horas, patches, suporte e situação do teste |
| selected_train_timestamps.csv | Horas escolhidas e sua proveniência no manifesto14k |
| patch_audit.csv | Todas as posições tentadas, com inclusão/exclusão por suporte |
| patch_manifest.csv | Correspondência dos índices dos tensores com timestamp/posição |
| smoke_tensors.npz | conditioning[N,12,32,32], target_raw_dbz[N,1,32,32], target_tensor, palette_support, lat/lon da grade inteira |
| diagnostic_normalization.json | Média/desvio/escala por canal, ajustados somente na amostra de treino |
| training_steps.csv | Loss, quantidade de pixels de suporte e norma dos gradientes por passo |
| diagnostic_model_state.pt | Pesos da CNN pequena ao final do teste |
| era5_source_provenance.json | Fontes ERA5, fingerprints de tamanho/mtime e metadata |
| run_config.json | Parâmetros, canais, hashes, interpolação e limites metodológicos |

Resultado de sucesso esperado:

```json
{
  "complete": true,
  "technical_smoke_passed": true,
  "production_ready": false,
  "dry_rule_documented": false,
  "corrdiff_native_tested": false
}
```

Isso demonstra que a amostra percorreu o caminho técnico sem valores inválidos
contaminarem a loss. Não estima a quantidade final de horas treináveis do dataset.
Para isso, use14k e a resposta do produtor sobre semântica/cobertura.

## Validação e continuidade

```bash
python -m unittest discover -s corrdiff_smoke_14l/tests -v
```

Os testes incluem interpolação de campo analítico, bloqueio de extrapolação,
separação treino/teste, layout dos patches, invariância da loss/gradiente a alterações
nos pixels excluídos, pipeline sintético integrado e corrupção de cache.

O próximo trabalho de integração nativa será adaptar e testar o caminho CorrDiff
real (regressão e difusão) com sua interface de máscara e loss. Esta CNN não verifica
a compatibilidade dessa interface. Não use a loss nativa sem máscara para alvos
parcialmente desconhecidos, nem preencha todos os NaNs como dados secos.
