#!/usr/bin/env bash
set -euo pipefail
# Execute from the rionowcastdf repository root. Each stage is explicit.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
RADAR_DIR="${RADAR_DIR:-/home/vsantos/data/radar_sumare}"
OLD_CACHE="${OLD_CACHE:-/home/vsantos/repositories/rionowcastdf/data/radar_cache}"
QC_CACHE="${QC_CACHE:-data/radar_cache_qc_v2}"
OLD_DATASET="${OLD_DATASET:-datasets/corrdiff_2011_2024_restored}"
QC_DATASET="${QC_DATASET:-datasets/corrdiff_2011_2024_restored_qc_v2}"
ANALYSIS="${ANALYSIS:-analysis_outputs_restored_qc_v2}"
case "${1:-help}" in
  scan)
    extra=()
    if [[ -n "${QC_POLICY:-}" ]]; then extra+=(--policy "$QC_POLICY"); fi
    if [[ -n "${QC_REVIEWS:-}" ]]; then extra+=(--reviews "$QC_REVIEWS"); fi
    "$PYTHON" -u "$HERE/14d_rebuild_radar_cache.py" --radar-dir "$RADAR_DIR" --old-cache-dir "$OLD_CACHE" --output-dir "$QC_CACHE" "${extra[@]}"
    ;;
  rebuild)
    "$PYTHON" -u "$HERE/14e_rebuild_dataset.py" --original-report "$OLD_DATASET/stats/execution_report.json" --qc-cache "$QC_CACHE" --output-dir "$QC_DATASET" --execute
    ;;
  audit)
    "$PYTHON" -u "$HERE/14f_masked_audit.py" --dataset-dir "$QC_DATASET" --reference-dataset-dir "$OLD_DATASET" --output-dir "$ANALYSIS/14f_masked_audit"
    ;;
  diagnostics)
    "$PYTHON" -u scripts/06_compute_diurnal_cycle.py --dataset-dir "$QC_DATASET" --output-dir "$ANALYSIS/06_diurnal"
    "$PYTHON" -u "$HERE/run_phase_with_qc_events.py" --phase 13 -- --phase6-dir "$ANALYSIS/06_diurnal" --output-dir "$ANALYSIS/13_regimes"
    "$PYTHON" -u "$HERE/run_phase_with_qc_events.py" --phase 14 -- --phase6-dir "$ANALYSIS/06_diurnal" --phase13-dir "$ANALYSIS/13_regimes" --output-dir "$ANALYSIS/14_distribution_shift"
    "$PYTHON" -u "$HERE/run_phase_with_qc_events.py" --phase 15 -- --dataset-dir "$QC_DATASET" --phase6-dir "$ANALYSIS/06_diurnal" --phase13-dir "$ANALYSIS/13_regimes" --phase14-dir "$ANALYSIS/14_distribution_shift" --output-dir "$ANALYSIS/15_formal_splits"
    ;;
  baselines)
    "$PYTHON" -u "$HERE/16_masked_baselines.py" --dataset-dir "$QC_DATASET" --splits-dir "$ANALYSIS/14f_masked_audit" --output-dir "$ANALYSIS/16_masked_baselines"
    ;;
  test)
    PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m unittest discover -s "$HERE/tests" -v
    ;;
  *) echo "Uso: bash pipelines/dataset_preparation/corrdiff_radar_qc/run_qc.sh {scan|rebuild|audit|diagnostics|baselines|test}"; exit 2 ;;
esac
