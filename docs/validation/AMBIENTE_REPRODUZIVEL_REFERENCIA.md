# Passo 3 — preparação reproduzível e auditoria da máquina

## O que esta entrega faz

Um CLI `scripts/reference_environment.py` com três subcomandos:

- `doctor`: inventário e verificações, sem instalar pacotes ou treinar CorrDiff.
- `bundle`: congela o ambiente Python corrente e produz wheels locais, incluindo
  uma wheel construída do checkout PhysicsNeMo no commit indicado.
- `install`: cria um NOVO venv e instala somente as wheels congeladas, offline,
  com validação SHA-256 e sem alterar o ambiente `dataset`.

A implementação não instala driver NVIDIA, CUDA do sistema, Git ou Python.
Esses são pré-requisitos da máquina. Não há apt/sudo, exclusão de dados ou
modificação de arquivos do checkout. O estágio de empacotamento usa rede para
obter dependências e requisitos de build; a instalação do bundle não usa índices.
Não baixe nem instale bundles de origem desconhecida: hashes verificam integridade,
não autenticidade de um manifesto não assinado.

## Preparação inicial no CEFET

Extraia o ZIP fora do repositório e aplique o patch na raiz:

```bash
cd ~/repositories/rionowcastdf
conda activate dataset
git apply --check ~/Downloads/environment_step3/environment_step3.patch
git apply ~/Downloads/environment_step3/environment_step3.patch
python -m unittest discover -s tests -p test_reference_environment.py -v
```

Auditoria da máquina atual (as pastas de saída são criadas; relatórios existentes
não são sobrescritos):

```bash
python scripts/reference_environment.py doctor \
  --project-dir "$PWD" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --dataset-dir datasets/corrdiff_reference_v1 \
  --gpus 0,1 \
  --output "analysis_outputs_reference/environment/doctor_$(date +%Y%m%d_%H%M%S).json"
```

`doctor` registra Python, pacotes, CUDA do PyTorch, localização de importação do
PhysicsNeMo, commit e alterações locais, driver/UUID/memória das GPUs, CPU,
libc e espaço disponível. Executa pip check, importa as duas classes nativas,
executa forward/backward FP32 em uma pequena convolução em cada GPU escolhida,
verifica metadados de treino/validação pelo adaptador e executa as suítes dos
passos 1–2. Não acessa os splits de teste. O teste de convolução não é teste de
CorrDiff. `ready_for_training` permanece false, pois a integração ainda está pendente.

IDs de GPU são LÓGICOS após CUDA_VISIBLE_DEVICES. Por exemplo, se definir
CUDA_VISIBLE_DEVICES=1, use --gpus 0. `--cpu-only` é apenas auditoria parcial:
o relatório terá gpu_checked=false e não valida uma máquina de treinamento GPU.
Testes ignorados fazem a verificação correspondente falhar, para não serem
confundidos com testes aprovados.

## Congelar dependências

O ambiente `dataset` já foi usado nos testes; ele é a fonte do inventário.
Não há uma lista inventada de versões mínimas. A instalação atual é capturada
com pip freeze --all. Como isso captura TODOS os pacotes Python, o bundle pode
ser grande e incluir pacotes alheios ao treinamento. Isso privilegia reprodução
da instalação disponível; uma redução futura precisa de nova validação.

```bash
python scripts/reference_environment.py bundle \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --output "$HOME/corrdiff_environment_bundles/reference_cu129_v1"
```

Esse índice corresponde ao PyTorch 2.13.0+cu129 informado nesta máquina. O script
não instala uma versão diferente nem escolhe CUDA automaticamente. Se houver
pacotes de outros índices, forneça outros --extra-index-url. Não inclua tokens
ou senhas nas URLs de linha de comando.

O comando exige checkout limpo e commit esperado. Default:
`c15bd23eb56216c32bd3e30180593e5685be9bc2`. Se houver alterações, ele interrompe;
não as descarta. Verifique `git status`, preserve o trabalho em commit e use
`--expected-commit SHA` se esse passar a ser o baseline adotado. Arquivos não
rastreados também são sinalizados. Não use comandos de limpeza destrutiva.

Dependências editáveis, URLs diretas e versões não fixadas são recusadas com
mensagem explícita. É necessário resolver a proveniência delas antes de gerar
um bundle reproduzível. A distribuição nvidia-physicsnemo da lista congelada é
substituída por uma wheel construída do checkout. Portanto, essa instalação nova
pode diferir do pacote site-packages atual; sua auditoria deve ser executada de novo.
O simples `git rev-parse` nunca prova a origem do pacote importado.

Saídas:

- source-freeze.txt: inventário original.
- dependencies.txt: dependências exatas a empacotar.
- wheels/: artefatos instaláveis.
- requirements.lock: versões exatas e hashes das wheels finais.
- manifest.json: commit, plataforma, hashes e validated_install=false.

O build inicial pode compilar pacotes sem wheel disponível e exigir ferramentas
locais. Se falhar, não há bundle aprovado. Não reutilize o diretório parcial como
se estivesse completo. Escolha uma nova saída após resolver a causa. O script não
promete build binário idêntico a partir de fontes: a reprodução usa as wheels
já congeladas e verificadas.

## Instalar em ambiente separado

Na máquina original ou outra de mesma plataforma, com Python da mesma versão
menor e suporte a venv:

```bash
python3.12 scripts/reference_environment.py install \
  --bundle "$HOME/corrdiff_environment_bundles/reference_cu129_v1" \
  --venv "$HOME/venvs/corrdiff_reference_v1"
```

O destino deve ser novo. O comando valida hashes e plataforma, cria venv, usa
pip --no-index --require-hashes, executa pip check e importa as classes nativas.
Emite `reference_installation.json` no venv, com proveniência e inventário
instalado. Em falha, o diretório pode conter instalação parcial; não há recibo de
sucesso. O ambiente `dataset` original permanece intacto.

Em seguida, validar usando o Python NOVO:

```bash
"$HOME/venvs/corrdiff_reference_v1/bin/python" scripts/reference_environment.py doctor \
  --project-dir "$PWD" \
  --physicsnemo-dir "$HOME/repositories/physicsnemo" \
  --dataset-dir datasets/corrdiff_reference_v1 \
  --gpus 0,1 \
  --output "analysis_outputs_reference/environment/recreated_$(date +%Y%m%d_%H%M%S).json"
```

Para outra máquina, copie o bundle INTEIRO e disponha dos repositórios e dataset
nos caminhos passados. Não copie o venv, pois contém caminhos dependentes da máquina.
Guarde o código de rionowcastdf em um commit e use o mesmo commit no destino.
O bundle não inclui código do projeto, dataset, dados brutos, bibliotecas do sistema,
driver ou ferramentas Conda. Mesmo OS/arquitetura não garante mesma libc/driver:
pip, imports e doctor verificam a instalação efetiva; a validação nativa continua
necessária. Windows, ARM e outra versão menor de Python exigem bundle próprio.

Esta etapa pressupõe dataset de referência pronto. Transformar dados brutos em
Zarr continua responsabilidade do pipeline de construção e suas auditorias.

## Validação desta entrega

Nove testes unitários passaram localmente: leitura de wheel e manifesto, alteração
indevida de wheel/lock, incompatibilidade de plataforma, caminho inseguro,
congelamento de versões, rejeição de origens não reproduzíveis, preservação de
ambiente existente e rejeição de commit incorreto. O CLI --help foi verificado.

Não foi possível construir e recriar aqui o ambiente CUDA do CEFET: ele não está
neste ambiente de execução. Não se afirma que doctor/bundle/install já passaram
na máquina do usuário. Isso será estabelecido pelos relatórios reais. Instalação
reproduzida e verificação de GPU não substituem os passos 5–6 (CorrDiff e geração).
