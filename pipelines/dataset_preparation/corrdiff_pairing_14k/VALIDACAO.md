# Validação da entrega14k

Implementação validada localmente em 02/11/2026 com Python, DuckDB2.0.0 e
PyArrow26.0.0. Sete testes automatizados passaram:

1. Timestamp com offset−02 convertido para UTC sem deslocamento duplicado; longitudes315/318 convertidas para−45/−42.
2. Duplicatas de coordenadas e infinito são contados; níveis85000/50000Pa reconhecidos com opçãoPa.
3. Duas colunas temporais candidatas causam erro explícito.
4. Purga em fronteira mantém intervalo semiaberto de96horas e respeita ambos os limites.
5. Inventário→retomada→pareamento dos12canais; sobreposição de arquivos bloqueia candidatura, sem duplicar timestamps; training_ready continua falso.
6. Canal ausente não é preenchido; Parquet corrompido registrado como erro.
7. Linha temporal2011–2024 contém122736horas, com288horas purgadas e87624horas nominais remanescentes em treino.

Também verificada a interface --help. O teste integrado usa Parquets e manifesto
SQLite sintéticos. Nenhum arquivo ERA5 do servidor CEFET foi lido nesta validação,
e não foi executado o inventário completo remoto. Disponibilidade e tempo real de
execução dependem do acervo. Os testes não validam a semântica meteorológica dos PNGs.

Referências de implementação consultadas no repositório rionowcastdf:
- src/corrdiff/builder/dataset_builder.py: raízes single/pressure, aliases, níveis e canais.
- scripts/15_compute_formal_splits.py: purga [boundary−48h,boundary+48h), protocolo temporal.
- Contrato local da entrega14j: schema frames/sources, geometry.npz, classificação descritiva.

Nenhuma dependência em importações do repositório: o pacote é autônomo.
