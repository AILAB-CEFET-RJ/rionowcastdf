# Validação da entrega 14n — 05/11/2026

Seis testes automatizados passaram (4,767 s na execução registrada):

1. Classes por pixel: sem eco, paleta, alpha parcial, opaco desconhecido e fora
   da cobertura; NaN preservado em dBZ para sem eco e em alvo para desconhecidos.
2. Política ponta a ponta com manifesto 14j/14k sintético, início local/UTC,
   exclusão anterior ao acervo, preservação dos arquivos de entrada, retomada e
   liberação cartográfica somente por decisão explícita com evidência.
3. Máscara espacial externa, verificação de coordenadas e detecção de objeto
   decodificado corrompido, que fica fora dos candidatos.
4. Horário local ambíguo preserva o diagnóstico espacial, mas não entra nos
   candidatos pareados nem de treino.
5. Smoke ponta a ponta com Parquet e SQLite de teste, leitura dos 12 canais,
   interpolação, seleção de horas com e sem eco, 36 patches usados (nenhum patch
   seco descartado), três atualizações finitas de pesos; adulteração posterior
   do arquivo de cobertura é detectada. Nenhum teste/validação usado no treino.
6. Organização Git em repositório temporário: .gitignore aditivo, relatório de
   arquivo rastreado/ignorado, sem apagar arquivo nem modificar o índice.

O log está em `test_results.txt`.

## Compatibilidade com exemplos reais já disponíveis

Dois PNGs reais de exemplos anteriores foram decodificados com o código original
14j recuperado para esta implementação; a política foi aplicada às suas classes.
Os PNGs não são incluídos neste pacote. Hashes e resultados estão em
`real_examples_validation.json`.

- `2012_02_25_04_00.png`: cartografia, zero pixels de paleta; permanece em revisão
  por padrão. O SHA coincide com o exemplo conhecido.
- `2012_12_11_17_00.png`: 17.578 pixels de paleta nativos; na grade geométrica,
  714 pontos com eco, 5.042 pontos sem eco e 124 desconhecidos. Suporte total:
  5.756 pontos de 5.880. Esses números usam a geometria provisória, não uma nova
  certificação da cobertura observacional.

## Limitações da validação

O programa não foi executado por este assistente no servidor CEFET nem sobre
seus 122.736 horários reais. Os testes locais validam comportamento e contratos,
não antecipam a contagem final do acervo nem o tempo de execução no servidor.
Não foi executado CorrDiff nativo ou treino científico; o modelo do smoke é
TinyRegressor. Não foi criada máscara de bloqueio topográfico por inferência.
Não foram alterados o cache original, o inventário ERA5, os splits, a agregação
ou a auditoria 14m em execução. Não houve alteração ou push no GitHub.
