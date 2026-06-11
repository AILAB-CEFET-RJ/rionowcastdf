"""
create_split.py

Create train/validation splits for CorrDiff datasets.

This script generates:

    train_index.npy
    valid_index.npy

Expected dataset structure:

datasets/
└── corrdiff/
    ├── train.zarr/
    ├── train_index.npy
    └── valid_index.npy

Recommended split:

    train: 95%
    valid: 5%

Author:
    Vinicius / CorrDiff adaptation

"""

from pathlib import Path
import numpy as np
import zarr


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "datasets/corrdiff/train.zarr"

TRAIN_OUTPUT = "datasets/corrdiff/train_index.npy"
VALID_OUTPUT = "datasets/corrdiff/valid_index.npy"

TRAIN_RATIO = 0.95

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CorrDiff Train/Validation Split Generator")
    print("=" * 80)

    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    print(f"Opening dataset: {dataset_path}")

    z = zarr.open(
        str(dataset_path),
        mode="r",
    )

    n_samples = z["input"].shape[0]

    print(f"Total samples: {n_samples}")

    # ============================================================
    # SHUFFLE
    # ============================================================

    rng = np.random.default_rng(SEED)

    indices = np.arange(n_samples)

    rng.shuffle(indices)

    # ============================================================
    # SPLIT
    # ============================================================

    train_size = int(n_samples * TRAIN_RATIO)

    train_idx = indices[:train_size]
    valid_idx = indices[train_size:]

    print(f"Train samples: {len(train_idx)}")
    print(f"Valid samples: {len(valid_idx)}")

    # ============================================================
    # SAVE
    # ============================================================

    np.save(
        TRAIN_OUTPUT,
        train_idx,
    )

    np.save(
        VALID_OUTPUT,
        valid_idx,
    )

    print("\nSaved files:")
    print(TRAIN_OUTPUT)
    print(VALID_OUTPUT)

    print("\nDone.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()