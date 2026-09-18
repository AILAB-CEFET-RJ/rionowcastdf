# CorrDiff — Fase 4 v3: extremos com amostragem estratificada global

A v3 corrige a limitação remanescente da v2.

## O que mudou

A v2 estratificava corretamente os pixels **dentro de 256 blocos selecionados**,
mas esses blocos ainda sub-representavam a cauda do radar.

A v3 executa:

1. **PASSO A — varredura global do target + mask**
   - lê todos os `1,132,464` patches do target/mask;
   - conta exatamente os estratos no dataset inteiro;
   - mantém reservoirs globais de endereços `(patch, pixel)`.

2. **PASSO B — recuperação dos preditores**
   - agrupa os endereços selecionados pelo chunk do `input`;
   - lê os chunks ERA5 necessários;
   - extrai os 12 canais brutos e calcula as 8 variáveis derivadas apenas
     nos pixels selecionados.

3. **Pesos globais**
   - `peso = N_global_estrato / n_amostra_estrato`.

## Execução recomendada

```bash
python scripts/04_compute_extreme_statistics.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --phase0-dir analysis_outputs/00_quality \
  --output-dir analysis_outputs/04_extremes \
  --stratum-size 20000 \
  --zero-stratum-size 20000 \
  --overwrite
```

Se o PASSO B ficar pesado em I/O, a primeira redução segura é diminuir apenas
o reservoir de zeros:

```bash
--zero-stratum-size 10000
```

Não reduza os estratos de evento antes de avaliar o resultado.

## Saídas novas/importantes

- `global_scan_blocks.parquet`
- `stratum_sampling.parquet`
- `input_blocks_read.parquet`
- `event_prevalence.parquet`
- `effect_sizes.parquet`
- `analysis_summary.json`

## Critérios para fechar a Fase 4

O relatório deve mostrar:

- `phase4-extremes-v3-global-stratified-weighted`;
- todos os patches do target/mask varridos;
- `exact_global_event_rate_v3` praticamente igual à Fase 0;
- `max_abs_fixed_threshold_rate_diff_v3_vs_phase0` próximo de zero;
- amostra útil em `>=40` e `>=45`;
- `>=50` tratado apenas como caso ultra-raro;
- `effective_n_event` interpretado como ESS dos pesos, não como número de
  eventos meteorológicos independentes.

## Observação científica

As frequências continuam sendo **patch-overlap weighted**, isto é, representam
a distribuição de treinamento armazenada no Zarr. A análise espacial/climatológica
deduplicada será tratada nas fases posteriores.
