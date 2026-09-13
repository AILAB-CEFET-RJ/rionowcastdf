# CorrDiff — Fase 0: auditoria, qualidade e semântica do alvo radar

Esta versão da Fase 0 mantém a auditoria estrutural/temporal/espacial do dataset e acrescenta uma validação explícita do caminho do radar:

`PNG RGB -> valores numéricos da legenda -> cache .npy float32 -> patch -> mask/clip/log1p -> target Zarr`.

## Arquivos

- `scripts/00_audit_dataset.py`
- `notebooks/00_dataset_quality.ipynb`

## Execução recomendada

A partir da raiz do projeto no CEFET:

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

O cache esperado usa nomes `YYYYMMDD_HH_MM.npy`, exatamente como no builder. Se você ainda não souber o caminho, consulte o `RADAR_CACHE_DIR` usado pelo projeto/builder.

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

## Novos artefatos da auditoria do radar

Além dos arquivos da primeira versão, esta versão produz:

```text
analysis_outputs/00_quality/
├── radar_legend_mapping.parquet
├── radar_cache_sample_statistics.parquet       # se o cache for informado
├── radar_target_mapping_validation.parquet     # se cache + patch_row/patch_col estiverem disponíveis
└── radar_target_semantics.json
```

`radar_target_semantics.json` separa três coisas:

1. **comportamento comprovado pelo código do builder**;
2. **estatísticas observadas diretamente em uma amostra dos `.npy`**;
3. **validação empírica cache -> Zarr**.

A unidade física da legenda **não é declarada pelo builder**. O código chama o campo de `reflectivity`, mas a confirmação de `dBZ` deve vir da documentação/origem do produto radar.

## Validação cache -> Zarr

Para uma amostra de patches, o auditor reproduz exatamente:

```python
expected_mask = np.isfinite(cache_patch)
filled = np.nan_to_num(cache_patch, nan=0.0, posinf=0.0, neginf=0.0)
clipped = np.clip(filled, 0.0, None)
expected_target = np.log1p(clipped)
```

E compara com `train.zarr/target` e `train.zarr/mask`. Também verifica:

```python
np.expm1(stored_target) ~= clipped_cache_patch
```

Isso demonstra o domínio numérico efetivamente armazenado e evidencia a perda irreversível de qualquer valor negativo após `clip(min=0)`.
