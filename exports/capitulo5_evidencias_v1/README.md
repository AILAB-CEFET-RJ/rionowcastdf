# Evidências para o capítulo 5

Comece por export_report.json e inventory.csv. O pacote preserva todas as saídas referenciadas pelos summaries das fases 0–17. O diretório phases contém originais; tables_csv contém conversões, quando disponíveis. NPZs preservam mapas, amostras e ajustes.

Este pacote não executa análises, não acessa os PNGs/ERA5 brutos e não exporta o Zarr. Complete/passed significa conclusão das verificações implementadas, não validade meteorológica. Fase 17 contém planos, não resultados de treinamento.

Para compartilhar via Git, priorize CSV/JSON e código. Não presuma que o leitor remoto consegue abrir ZIP, NPZ ou Parquet. Verifique tamanhos antes de versionar.
