#!/usr/bin/env python3
"""Missing or blocked evidence never closes Phase 18 as PASS."""
import argparse
import json
from pathlib import Path
VERSION = 'phase18-nvidia-corrdiff-native-integration-v2'


def evaluate(directory):
    checks = {}
    def read(name):
        path = directory / name
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            return {}
        return data if data.get('phase_version') == VERSION else {}
    checkout = read('phase18_nvidia_checkout_audit.json')
    checks['checkout'] = checkout.get('status') == 'PASS'
    stats = read('conditioning_stats/conditioning_stats_audit.json')
    checks['train_only_statistics'] = (stats.get('status') == 'PASS' and stats.get('fit_split') == 'train'
        and stats.get('all_means_finite') is True and stats.get('all_stds_positive_finite') is True
        and {e['experiment_id'] for e in stats.get('experiments', [])} == {f'C{i:02}' for i in range(11)})
    data = read('dataset_smoke_audit.json')
    checks['dataset_and_masks'] = data.get('status') == 'PASS' and data.get('all_train_validation_masks_valid') is True
    checks['locked_splits'] = all(data.get('test_lock_checks', {}).get(k, {}).get('refused') is True
                                 for k in ('test_primary', 'test_stress_ood'))
    for exp in ('C00', 'C07'):
        checks[f'{exp}_dataset_splits'] = all(data.get('experiments', {}).get(exp, {}).get(s, {}).get('length', 0) > 0
                                             for s in ('train', 'validation'))
        for mode in ('compose', 'regression', 'diffusion'):
            native = read(f'native_smoke_audit_{exp}_{mode}.json')
            valid = (native.get('status') == 'PASS' and native.get('mode') == mode
                     and native.get('experiment_id') == exp
                     and native.get('nvidia_commit') == checkout.get('git', {}).get('commit'))
            if mode == 'compose':
                valid = valid and set(native.get('resolved_configs', {})) == {'regression', 'diffusion'}
            else:
                valid = (valid and native.get('fingerprint') == data.get('fingerprint')
                    and bool(native.get('fingerprint')) and bool(native.get('loss_evidence'))
                    and bool(native.get('checkpoints'))
                    and native.get('seed_evidence', {}).get('model_seeds') == [native.get('seed')]
                    and native.get('seed_evidence', {}).get('sampler_seeds') == [native.get('seed')])
            checks[f'{exp}_{mode}'] = bool(valid)
        generation = read(f'native_generation_audit_{exp}.json')
        checks[f'{exp}_generation'] = (generation.get('status') == 'PASS'
            and generation.get('experiment_id') == exp and bool(generation.get('fingerprint'))
            and generation.get('fingerprint') == data.get('fingerprint')
            and generation.get('nvidia_commit') == checkout.get('git', {}).get('commit'))
    integration = 'PASS' if all(checks.values()) else 'BLOCKED'
    return dict(phase_version=VERSION, integration_status=integration, checks=checks,
                pending=[k for k, v in checks.items() if not v], phase18_status=integration,
                stage_r_ready=integration == 'PASS', design_blockers=[],
                scope='No full training. Only train/validation may be used.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase18-dir', type=Path, default=Path('analysis_outputs/18_nvidia_integration'))
    p.add_argument('--output', type=Path, default=Path('analysis_outputs/summary_phase18.txt'))
    args = p.parse_args()
    result = evaluate(args.phase18_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output.read_text())
    return 0 if result['phase18_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
