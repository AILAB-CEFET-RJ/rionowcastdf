# Validação desta entrega

Foram executados oito testes sintéticos ponta a ponta, com SQLite, Parquet e
alvos NPZ, cobrindo:

1. Preparação mensal, estatísticas somente de treino e leitura dos patches.
   A validação contém um deslocamento de 10.000 nas variáveis: ele não altera
   a média calculada no treino.
2. Retomada após interrupção entre os meses completos e o relatório final,
   recusa de saída imutável concluída e detecção de mudança no ERA5.
3. Recusa de shard adulterado ao carregar dados.
4. Piloto restrito ao treino, sem incluir validação.
5. Perda sem contribuição de pixels desconhecidos e ausência de ruído/resíduo
   fora do suporte, inclusive quando o alvo científico desconhecido é NaN.
6. Acumulação de média/variância e decodificação categórica documentada.

Comando reproduzível:

```bash
python -m unittest discover -s corrdiff_contract_14o/tests -v
```

Isso não equivale a executar o acervo do CEFET. A entrega não possui os
Parquets mensais nem os snapshots 14n reais no ambiente local. O piloto no
servidor verificará esses arquivos e a execução completa emitirá seus próprios
relatórios. Os testes sintéticos não confirmam cobertura física do radar.

O código-fonte dos modelos e perdas nativas foi inspecionado no commit NVIDIA
registrado no README. O teste nativo está implementado, mas não foi executado
aqui; PhysicsNeMo não está instalado. Não há alegação de treino CorrDiff
validado, geração validada ou qualidade preditiva demonstrada.

Nenhum arquivo de 14j, 14k ou 14n foi modificado por esta entrega.

7. Passo 8 com bordas: 48 recortes e todos os 5.880 pontos cobertos.
8. Reindexação preservando os campos e a normalização, com leitura pelo loader.
