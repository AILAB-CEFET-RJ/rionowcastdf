# Entrega 1 — cache intermediário Sumaré 14j

Versão: `14j-intermediate-cache-v1`. Python 3.10+ em Linux/WSL.

Implementação isolada: não modifica builder, dataset, PNGs nem cache antigos.
Preserva evidências e reamostra a geometria do piloto 14i; não decide se transparência
é tempo seco. Não é ainda um dataset pronto para treinamento. Entregas 2 (ERA5/splits)
e 3 (treino técnico) não fazem parte deste pacote.

## Instalação e primeiro lote

Extraia o ZIP na raiz de `/home/vsantos/repositories/rionowcastdf`.

```bash
cd /home/vsantos/repositories/rionowcastdf
python -m pip install -r corrdiff_cache_14j/requirements.txt
python -m unittest discover -s corrdiff_cache_14j/tests -v
mkdir -p logs
python -u corrdiff_cache_14j/build_cache.py \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir data/radar_intermediate_14j_check \
  --start-utc 2012-02-25T00:00:00Z \
  --end-utc 2012-02-25T23:00:00Z
```

O primeiro lote testa 24 timestamps, incluindo os casos transparente e cartográfico.
A saída deve ser nova. Não há acesso ao ERA5 neste programa.

## Acervo horário de 2011–2024

```bash
nohup python -u corrdiff_cache_14j/build_cache.py \
  --radar-dir /home/vsantos/data/radar_sumare \
  --output-dir data/radar_intermediate_14j_v1 \
  --start-utc 2011-01-01T00:00:00Z \
  --end-utc 2024-12-31T23:00:00Z \
  --cadence-minutes 60 \
  > logs/14j_intermediate_cache.log 2>&1 &
```

Intervalo inclusivo: 122.736 timestamps UTC. Os primeiros horários UTC de 2011
podem procurar PNGs na data local de 2010; ausência é registrada, sem deslocamento
artificial. A janela cobre horas UTC, não dias locais completos.

Acompanhar:
```bash
tail -n 30 logs/14j_intermediate_cache.log
cat data/radar_intermediate_14j_v1/summary.json
```

Para retomar, execute EXATAMENTE o mesmo comando acrescentando `--resume`.
Não execute dois escritores ao mesmo tempo: o lock POSIX impede isso.
A retomada relê os PNGs para verificar o hash; reutiliza decodificações íntegras,
reavalia ausentes/erros e registra alterações da fonte. Logo, retomada não é custo zero,
mas evita refazer a decodificação e compressão de objetos já processados.
Uma mudança de código, geometria, intervalo ou cadência exige outra saída.

`--cadence-minutes 2` é opcional: usa cada instante da sequência, não médias.
Para cobrir todos os minutos pares do último dia, use `--end-utc 2024-12-31T23:58:00Z`.
O custo é muito maior; a entrega padrão usa apenas horas exatas para o ERA5.

## Arquivos gerados

- `config.json`: contrato completo, paleta, parâmetros e hashes do código.
- `geometry.npz`: lat/lon 70×84, índices nativos, máscara geométrica e referência legada.
- `manifest.sqlite`: tabela `frames` por timestamp UTC, `sources` por hash e `changes`.
- `objects/xx/HASH.png`: bytes EXATOS do original, inclusive para arquivo ilegível.
- `decoded/xx/HASH.npz`: RGBA nativo, classes nativas e campos reamostrados.
- `summary.json`: progresso, contagens e limitações.

Objetos são deduplicados pelo SHA-256 dos bytes. Arquivos transparentes idênticos têm
um único objeto, mas todos os seus timestamps permanecem no manifesto. Originais
antigos continuam preservados caso a fonte mude durante uma retomada.

Cada objeto NPZ decodificado contém:

| Campo | Conteúdo |
|---|---|
| `native_rgba` | RGBA uint8 completo; inclui RGB oculto por alfa zero |
| `native_class` | Códigos descritivos, não máscara observacional |
| `metadata_json` | Modo original, dimensões, info do Pillow, contagens, status |
| `grid_class` | Classe no ponto nativo mais próximo da grade |
| `grid_alpha` | Alfa amostrado; deve ser lido junto de `geometric_mask` |
| `grid_dbz_palette` | dBZ nominal da paleta adotada, NaN para não reconhecido |
| `grid_dbz_review` | Mesmo campo, inteiramente NaN se tela cartográfica suspeita |
| `geometric_mask` | Dentro do círculo/geometria; NÃO valida observação |

Os campos de grade só são criados para PNG de quadro único com dimensões esperadas.
Dimensão inesperada preserva RGBA/classes mas não força um redimensionamento.
PNG multiquadro mantém o original integral; RGBA é apenas do primeiro quadro e
não é reamostrado. Falha de decodificação preserva apenas os bytes e o erro.

Classes: 0 transparente; 1..7 paleta opaca na ordem [50,45,40,35,30,25,20] dBZ;
254 alfa parcial; 255 opaco não reconhecido; 253 fora da geometria (apenas grade).
Alfa parcial não é descartado: fica preservado em RGBA, mas não recebe dBZ automático.

**Não existe uma máscara de validade observacional inferida.** Não usar
`isfinite(grid_dbz_palette)` como máscara de toda a área observada. Nenhum NaN é
substituído por zero. Classificação cartográfica é heurística, não prova operacional.
`production_ready=false` e `dry_rule_documented=false` são esperados.

## Tempo e geografia

Cada solicitação parte de UTC e busca o nome local por `America/Sao_Paulo`, incluindo
horário de verão. Se um horário local tiver dois instantes UTC possíveis, ambas as
linhas esperadas são registradas com `local_filename_ambiguous=1`; o mesmo PNG não
comprova duas observações distintas. Entregas seguintes deverão resolver ou mascarar
essas linhas. O programa não inventa uma escolha de fold.

AEQD esférica do piloto 14i: centro (-22.955139,-43.248278), raio de imagem 138,9 km,
656×654. Parâmetros da referência acadêmica, sujeitos à confirmação histórica pelo
produtor. Grade lat/lon [-23.5,-22.25) × [-44,-42.5), passo 2/111 graus: **não é uma
grade métrica de exatamente 2 km em ambas as direções**. Amostragem pontual por vizinho
mais próximo. Pequenos ecos podem ficar entre pontos; nenhuma média é aplicada.
A geometria é fixa na execução; revisões exigem outro contrato de saída. O RGBA nativo
permite reamostrar sem reler/decodificar o PNG, embora essa ferramenta ainda não inclua
um comando específico para regenerar somente a grade.

## Inspeção e exportação

Use SQLite via Python padrão. Exemplo, a partir da raiz do repositório:

```python
import sqlite3
import numpy as np
from pathlib import Path

root = Path('data/radar_intermediate_14j_v1')
db = sqlite3.connect(root / 'manifest.sqlite')
for row in db.execute('SELECT status,COUNT(*) FROM frames GROUP BY status'):
    print(row)
row = db.execute('''SELECT s.decoded_path FROM frames f
JOIN sources s ON f.source_sha256=s.sha256
WHERE f.timestamp_utc=?''', ('2012-12-11T19:00:00+00:00',)).fetchone()
if row and row[0]:
    with np.load(root / row[0], allow_pickle=False) as data:
        print(data.files)
        print(data['metadata_json'].item())
db.close()
```

Compartilhe inicialmente apenas `summary.json` e `config.json`. O cache é grande e
não deve ser commitado no GitHub. O manifesto contém caminhos locais do acervo.
Não copie apenas o arquivo SQLite durante uma execução ativa: existem arquivos WAL.
Pare o escritor ou use a API SQLite `backup` para obter snapshot consistente.

## Integridade, interrupções e espaço

Gravações de objetos usam arquivo temporário e rename atômico. Manifesto é commitado
por timestamp. Interrupções podem deixar temporários/objetos órfãos, mas não uma linha
commitada apontando para escrita parcial. Não há coleta automática de órfãos.
Erros de disco/contrato interrompem o processo; dados ausentes ou PNGs ilegíveis são
registrados e a execução continua. Exit 0: inventário concluído sem erros de leitura;
2: concluído com erros de leitura/decodificação; 1: falha fatal; 130: interrupção.
`complete=true` não significa que todos os arquivos existam ou sejam válidos.

A saída ocupa espaço adicional: originais únicos + NPZ nativos/grade + SQLite.
Não há estimativa universal segura. Meça o lote inicial com `du -sh` e confira `df -h`
antes da execução completa. Cada frame é processado isoladamente; não se carrega o
acervo inteiro em memória. A execução é sequencial para limitar memória e I/O.

## Verificação entregue

Testes cobrem preservação RGBA/alpha, ponto geográfico conhecido, separação de mapa,
dimensões inesperadas, DST, deduplicação, ausentes, retomada, alteração da fonte,
corrupção do cache e rejeição de contrato incompatível. Testes usam dados sintéticos;
o arquivo VALIDACAO.md registra separadamente as verificações com PNGs reais.
