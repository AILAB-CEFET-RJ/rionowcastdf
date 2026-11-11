"""Sampled train/validation adapter check; never runs training or opens tests."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from adapter.reference_corrdiff_dataset import ReferenceCorrDiffDataset
from src.corrdiff.diagnostics.reference_reader import ReferenceReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if args.samples < 1 or args.workers < 0:
        p.error('samples must be positive and workers nonnegative')
    reports = []
    for split in ('train', 'validation'):
        ds = ReferenceCorrDiffDataset(args.dataset_dir, split)
        if not len(ds):
            raise ValueError(f'Empty {split}')
        indices = np.unique(np.linspace(0, len(ds)-1, min(args.samples, len(ds)), dtype=int)).tolist()
        raw = ReferenceReader(args.dataset_dir, split, input_mode='physical')
        for i in indices:
            sample, physical = ds[i], raw[i]
            expected = (physical['input'] - raw.mean) / raw.std
            np.testing.assert_array_equal(sample['input'].numpy(), expected)
            np.testing.assert_array_equal(sample['target'].numpy(), physical['target'])
            np.testing.assert_array_equal(sample['mask'].numpy(), physical['mask'])
        kwargs = {'multiprocessing_context': 'spawn', 'timeout': 60} if args.workers else {}
        seen = 0
        for batch in DataLoader(Subset(ds, indices), batch_size=2, num_workers=args.workers, **kwargs):
            assert batch['input'].dtype == torch.float32
            assert batch['target'].dtype == torch.float32
            assert batch['mask'].dtype == torch.bool
            seen += len(batch['input'])
        if seen != len(indices):
            raise ValueError('Incomplete DataLoader evaluation')
        reports.append(dict(split=split, samples_checked=seen, info=ds.info()))
    print(json.dumps(dict(complete=True, adapter_passed=True, native_training_tested=False,
                          test_splits_untouched=True, results=reports), indent=2))


if __name__ == '__main__':
    main()
