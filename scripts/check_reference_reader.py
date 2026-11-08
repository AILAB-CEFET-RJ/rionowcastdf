"""Sampled reader check; does not replace the full source integrity audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corrdiff.diagnostics.reference_reader import ReferenceReader, SPLITS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--hours-per-split', type=int, default=3)
    a = p.parse_args()
    if a.hours_per_split < 1:
        p.error('--hours-per-split must be positive')
    out, root = Path(a.output_dir).resolve(), Path(a.dataset_dir).resolve()
    if out == root or root in out.parents:
        p.error('Output must be outside the dataset')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(version='shared-reference-reader-v1', complete=False,
                  reader_check_passed=False, production_ready=False,
                  scope='sampled hours; all overlapping patches of selected hours', splits={})
    try:
        for split in SPLITS:
            physical = ReferenceReader(root, split)
            normalized = ReferenceReader(root, split, input_mode='standardized')
            hours = physical.hours
            selected = hours[np.unique(np.linspace(0, len(hours)-1,
                       min(a.hours_per_split, len(hours)), dtype=int))] if len(hours) else []
            results = []
            for timestamp in selected:
                h = physical.hour(int(timestamp))
                z = normalized.hour(int(timestamp))
                np.testing.assert_allclose(z['input'], (h['input']-physical.mean)/physical.std,
                                           rtol=1e-6, atol=1e-6, equal_nan=True)
                results.append(dict(timestamp_utc=int(timestamp),
                    valid_pixels=int(h['mask'].sum()), no_echo_pixels=int(h['no_echo'].sum()),
                    covered_grid_points=int((h['patch_count'] > 0).sum()),
                    max_patch_overlap=int(h['patch_count'].max())))
            report['splits'][split] = dict(patches=len(physical), hours=len(hours), checked=results)
            print(json.dumps({'split':split, 'hours_checked':len(results)}), flush=True)
        report.update(complete=True, reader_check_passed=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
