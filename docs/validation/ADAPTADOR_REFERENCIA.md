# Adaptador do contrato reference-zarr-v1

Implementação aditiva: `adapter/reference_corrdiff_dataset.py`, classe
`ReferenceCorrDiffDataset`. O adaptador histórico continua intacto. Dependência:
leitor compartilhado `src/corrdiff/diagnostics/reference_reader.py` já existente.

## Aplicação

Na raiz do repositório, aplique o patch distribuído:

```bash
git apply --check /caminho/reference_adapter.patch
git apply /caminho/reference_adapter.patch
python -m unittest discover -s tests -p test_reference_adapter.py -v
python scripts/check_reference_adapter.py \
  --dataset-dir datasets/corrdiff_reference_v1 --samples 8 --workers 2
```

Use o ambiente `dataset` já validado. O adaptador requer NumPy, Zarr e PyTorch;
não exige reinstalar CUDA ou PhysicsNeMo. O teste amostral abre somente treino e
validação e imprime JSON. Não substitui auditoria integral dos arrays.

## API

```python
from torch.utils.data import DataLoader
from adapter.reference_corrdiff_dataset import ReferenceCorrDiffDataset

dataset = ReferenceCorrDiffDataset('datasets/corrdiff_reference_v1', 'train')
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))
x, y, mask = batch['input'], batch['target'], batch['mask']
```

`input`: float32 `[B,C,32,32]`, padronizado uma única vez pelo leitor usando
normalization.json. `target`: float32 `[B,1,32,32]`, código/50 sem transformação
adicional. `mask`: bool na mesma forma do alvo. `metadata`: índice no split,
split, timestamps, offset, posição do patch e SHA da fonte. O collation padrão
mantém SHA/split como listas de strings e números como tensores.

Seleção ordenada de canais brutos via `channels=['tcwv','r_850','r_500']` permite
o conjunto de C01; C00 usa os doze canais. Não há escolha automática por ID de
experimento: esse mapeamento pertence ao futuro launcher. Derivados C07–C10
são recusados até implementar cálculo físico e estatísticas ajustadas no treino.

A máscara distingue zero válido (categoria sem eco) de placeholder inválido.
Nenhum patch é descartado. Um patch sem pixels válidos é retornado normalmente;
o treinamento deverá definir e registrar sua política para esse caso. O
adaptador não atribui significado de 0 dBZ ao código zero.

Abertura somente leitura, validações herdadas do leitor, checks de posição e
finitude após normalização. Cada worker abre seu próprio leitor; pickle remove
handles. Fingerprints de summary/contract/normalization são conferidos ao
abrir/reabrir. Isso não é um hash integral dos arrays nem protege contra alteração
concorrente de dados: mantenha o dataset congelado durante o treinamento.

## Fronteira com o treinamento

O resultado é um dicionário, deliberadamente incompatível com o unpacking do
`train.py` legado, que interpreta o terceiro elemento como lead time. Não
substitua a classe na configuração antiga esperando treinar sem modificar o loop.
O caminho novo não importa `datasets.base`, evitando colisão com outros pacotes
chamados datasets. Execute com a raiz deste repositório no PYTHONPATH.

A implementação não altera `native.masked_loss_verified`, não executa fase 18
e não emite evidência de treinamento. Próximo passo: conectar esse lote a um loop
que transporte máscara em treino e validação, trate residuais inválidos antes da
entrada no denoiser, reduza a loss por suporte válido e teste gradientes,
atualização de parâmetros e recarga de checkpoint. Depois executar C00 inicial.

Não há grade geográfica única por dataset de patches: posição e timestamp
identificam cada patch. Não se inventa espaçamento em quilômetros nem se fornece
uma latitude/longitude fixa incorreta ao modelo.
