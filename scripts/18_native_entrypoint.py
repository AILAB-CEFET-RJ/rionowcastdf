#!/usr/bin/env python3
"""Seed bridge for pinned native train.py; no NVIDIA source files edited."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def install_seed_hooks(helpers, datasets, seed, audit_path):
    original_seed = helpers.set_seed
    original_datasets = datasets.init_train_valid_datasets_from_config
    evidence = {'experiment_seed': seed, 'model_seeds': [], 'sampler_seeds': []}
    def save():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(evidence, indent=2))
    def seeded_model(offset):
        effective = seed + offset
        original_seed(effective)
        evidence['model_seeds'].append(effective)
        save()
    def seeded_datasets(*args, **kwargs):
        kwargs['seed'] = seed
        value = original_datasets(*args, **kwargs)
        evidence['sampler_seeds'].append(seed)
        save()
        return value
    helpers.set_seed = seeded_model
    datasets.init_train_valid_datasets_from_config = seeded_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrdiff-dir', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int, choices=[42, 43, 44])
    parser.add_argument('--seed-audit', required=True, type=Path)
    args, native_args = parser.parse_known_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Phase-18 seed bridge supports single-process smoke only.')
    corrdiff = args.corrdiff_dir.resolve()
    sys.path.insert(0, str(corrdiff))
    from helpers import train_helpers
    from datasets import dataset
    install_seed_hooks(train_helpers, dataset, args.seed, args.seed_audit.resolve())
    sys.argv = [str(corrdiff / 'train.py'), *native_args]
    runpy.run_path(str(corrdiff / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
