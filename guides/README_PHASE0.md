# CorrDiff — Fase 0 v3: auditoria, cobertura, qualidade e desbalanceamento do alvo

Esta versão encerra a Fase 0 com quatro blocos de auditoria:

1. **estrutura e qualidade numérica** do `train.zarr`;
2. **cobertura temporal**, incluindo padrões bidimensionais de missingness;
3. **semântica do radar**, do PNG/cache até o `target` armazenado;
4. **desbalanceamento do target**, usando limiares no domínio numérico da legenda do radar.

O script foi pensado para o dataset CorrDiff 2011–2024 e lê o Zarr em batches, evitando carregar o conjunto completo em RAM.

## Arquivos

```text
scripts/
└── 00_audit_dataset.py

notebooks/
└── 00_dataset_quality.ipynb
```

## Execução recomendada no CEFET

A partir da raiz do projeto:

```bash
export CORRDIFF_RADAR_CACHE_DIR=/caminho/para/o/cache_do_radar

python scripts/00_audit_dataset.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --radar-cache-dir "$CORRDIFF_RADAR_CACHE_DIR" \
  --output-dir analysis_outputs/00_quality \
  --batch-samples 512 \
  --quantile-sample-values 200000 \
  --radar-cache-file-sample 128 \
  --radar-target-validation-samples 256 \
  --overwrite
```

A seguir, abra no JupyterLab:

```text
notebooks/00_dataset_quality.ipynb
```

## Smoke test

```bash
python scripts/00_audit_dataset.py \
  --dataset-dir datasets/corrdiff_2011_2024 \
  --radar-cache-dir "$CORRDIFF_RADAR_CACHE_DIR" \
  --output-dir analysis_outputs/00_quality_quick \
  --radar-cache-file-sample 32 \
  --radar-target-validation-samples 64 \
  --quick \
  --overwrite
```

O modo `--quick` **não calcula as taxas de eventos do target**, pois elas exigem a varredura completa dos pixels válidos.

## Principais artefatos

```text
analysis_outputs/00_quality/
├── dataset_summary.json
├── audit_warnings.json
├── channel_summary.parquet
├── missing_data.parquet
├── temporal_coverage.parquet
├── yearly_coverage.parquet
├── monthly_coverage.parquet
├── hourly_coverage.parquet
├── year_month_coverage.parquet
├── month_hour_coverage.parquet
├── year_hour_coverage.parquet
├── patch_position_counts.parquet
├── expected_patch_spatial_coverage.npy
├── actual_patch_spatial_coverage.npy
├── radar_legend_mapping.parquet
├── radar_cache_sample_statistics.parquet
├── radar_target_mapping_validation.parquet
├── radar_target_semantics.json
├── target_event_thresholds.parquet
├── target_event_rates_global.parquet
├── target_event_rates_yearly.parquet
├── target_event_rates_monthly.parquet
├── target_event_rates_hourly.parquet
└── audit.log
```

## Heatmaps de disponibilidade

O notebook produz três visualizações complementares:

- **ano × mês**: identifica períodos extensos de baixa cobertura;
- **mês × hora UTC**: verifica se a disponibilidade horária muda sazonalmente;
- **ano × hora UTC**: verifica se o padrão horário de missingness muda entre anos.

Esses gráficos devem ser consultados antes de qualquer interpretação de sazonalidade ou ciclo diurno da precipitação.

## Semântica do radar

A implementação auditada segue o caminho:

```text
PNG RGB
  ↓
valores numéricos da legenda
  ↓
grade radar float32 em cache .npy
  ↓
patch
  ↓
mask = isfinite(cache_patch)
  ↓
nan_to_num
  ↓
clip(min=0)
  ↓
log1p
  ↓
target do Zarr
```

A validação direta cache → Zarr reproduz a transformação para uma amostra de patches. A unidade física da legenda, porém, **não é declarada no builder**. Até existir documentação externa, os valores devem ser chamados de **valores numéricos da legenda de refletividade**, e não automaticamente de dBZ.

## Diagnóstico de desbalanceamento do target

A versão v3 calcula os seguintes limiares no domínio numérico da legenda:

```text
> 0
>= 20
>= 25
>= 30
>= 35
>= 40
>= 45
>= 50
```

Para cada limiar são produzidas duas métricas:

- `event_pixel_ratio`: fração dos **pixels válidos dos patches** que satisfazem o limiar;
- `event_patch_ratio`: fração dos **patches de treinamento** que possuem pelo menos um pixel válido no limiar.

As taxas são produzidas globalmente e por:

```text
ano
mês
hora UTC
```

### Atenção ao overlap dos patches

Os patches `32×32` com `stride=16` se sobrepõem. Portanto, `event_pixel_ratio` é deliberadamente uma estatística da **distribuição de treinamento em patches**, e não uma climatologia espacial de campo completo sem duplicação. Pixels centrais podem contribuir mais de uma vez.

Esse diagnóstico é útil para decidir posteriormente sobre:

- amostragem de patches com eventos;
- ponderação de perdas;
- métricas específicas para eventos intensos;
- baselines que não sejam favorecidos por prever predominantemente zero.

## Resultado esperado ao fechar a Fase 0

Ao final, devemos ter documentados:

- integridade estrutural do Zarr;
- ordem e qualidade dos 12 canais ERA5;
- cobertura temporal total e seus padrões ano/mês/hora;
- geometria e cobertura espacial dos patches;
- semântica cache → target;
- faixa e distribuição básica do target;
- grau de desbalanceamento do radar por limiar;
- limitações ainda abertas, em especial a unidade física do produto radar.

A Fase 1 pode então começar com análise univariada detalhada dos 12 canais e do radar, mantendo separados o `target` armazenado e `expm1(target)` no domínio pós-clip do cache.
