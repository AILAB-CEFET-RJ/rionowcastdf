# Leitor compartilhado do Zarr de referência

Implementado contra o commit fd85d9e. Não altera builder, cache, contrato,
normalização ou tensores auditados. Não requer CUDA ou PyTorch. Usa NumPy e
Zarr 2 do requirements_reference_zarr.txt. Os diagnósticos antigos ainda precisam
ser migrados explicitamente para este leitor; o adaptador CorrDiff é etapa posterior.

## Convenções

- Um split por instância; padrão train. Nunca concatena treino e testes.
- Entradas em unidades físicas por padrão. input_mode='standardized' aplica
  uma única vez a média/desvio congelados do treino, sem recalcular estatísticas.
- target preserva o tensor code/50. target_code retorna as classes 0,20,...,50
  e NaN nos pontos inválidos. Zero representa ausência de eco exibido, não
  refletividade medida de 0 dBZ. Não se aplica log1p.
- mask acompanha sempre o alvo. no_echo significa mask & (target == 0).
- echo_at_least usa limiares das classes exibidas e respeita a máscara.
- Metadados incluem timestamps UTC em segundos Unix, horário UTC do PNG,
  offset, posição do patch e hash da fonte. Não reconverte nomes locais.
- batch e iter_batches limitam a leitura dos tensores por lote. hours carrega
  somente o vetor de timestamps dos patches para indexar horas.
- hour reconstrói 70x84, exige igualdade das sobreposições e devolve patch_count.
  Estatísticas calculadas sobre hour contam cada posição uma vez. Não ponderar
  novamente por patch_count. Pixels não cobertos permanecem input=NaN, mask=false.
  Na reconstrução, target usa zero como placeholder e deve ser lido com mask.
- A leitura verifica hashes de metadados, ordem de canais, formato e tipos dos
  tensores. Os valores são verificados à medida que forem lidos; não se trata
  de uma nova auditoria integral. Não confirma a cobertura física do radar.

## API

```python
from src.corrdiff.diagnostics import ReferenceReader
r = ReferenceReader('datasets/corrdiff_reference_v1', split='train')
b = r.batch(0, 48)  # input [N,C,32,32], target/mask [N,1,32,32]
h = r.hour(int(r.hours[0]))  # input [C,70,84], target/mask [1,70,84]
valid = h['mask']
no_echo_fraction = h['no_echo'].sum() / valid.sum()
echo = r.echo_at_least(h, 20)
standardized = ReferenceReader('datasets/corrdiff_reference_v1',
    input_mode='standardized', channels=['tcwv','t2m'])
```

## Testar e verificar no servidor

Execute a partir da raiz, no mesmo ambiente da auditoria aprovada:

```bash
python -m unittest discover -s tests -p test_shared_reference_reader.py -v
mkdir -p logs
nohup python -u scripts/check_reference_reader.py \
  --dataset-dir datasets/corrdiff_reference_v1 \
  --output-dir analysis_outputs_reference/shared_reader_v1 \
  --hours-per-split 3 > logs/shared_reader_v1.log 2>&1 &
echo "PID: $!"
tail -f logs/shared_reader_v1.log
```

O diretório de saída deve ser novo. summary.json deve indicar complete=true
 e reader_check_passed=true. O padrão verifica primeira, intermediária e última
hora de cada split, todos os patches dessas horas, sobreposições e padronização.
Não substitui a auditoria integral já aprovada. production_ready continua false.

## Validação da implementação

Nove testes sintéticos executados: unidades físicas/padronizadas e seleção de
canais; zero versus máscara; limiar de 20 dBZ; reconstrução incluindo bordas;
conflitos de sobreposição; placeholder inválido; offsets; hashes de metadados;
lotes/limites e CLI dos quatro splits. Sem acesso ao Zarr completo do CEFET:
a verificação acima deve ser executada naquele servidor.

## Próxima etapa

Adaptar os diagnósticos de estrutura e separação temporal para esta API e depois
as demais fases conforme suas dependências. Resultados de integridade técnica,
diagnósticos científicos e testes nativos CorrDiff devem permanecer distintos.
