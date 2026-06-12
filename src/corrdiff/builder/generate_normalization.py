"""
CorrDiff Normalization Statistics Generator
===========================================

This script computes normalization statistics for a CorrDiff
training dataset stored in Zarr format.

The generated statistics are used during:

- Regression stage training
- Diffusion stage training
- Validation
- Inference

Overview
--------
The script reads the complete training dataset and computes:

    input_mean
    input_std

for each ERA5 predictor channel and

    target_mean
    target_std

for the precipitation target.

These statistics are stored in a single file:

    normalization.npz

which is later consumed by the
ZarrCorrDiffDataset loader.

Dataset Format
--------------
Expected Zarr structure:

train.zarr
│
├── input   (N, C, H, W)
└── target  (N, 1, H, W)

Where:

- N = number of samples
- C = number of ERA5 variables
- H,W = patch dimensions

Normalization Formula
---------------------

Input normalization:

    x_norm = (x - input_mean) / input_std

Target normalization:

    y_norm = (y - target_mean) / target_std

The resulting normalized tensors have approximately:

    mean ≈ 0
    std  ≈ 1

which improves neural network training stability.

Important
---------
The target stored in train.zarr is already transformed
using:

    target = log1p(precipitation)

during dataset generation.

Therefore, this script MUST NOT apply log1p again.

Applying log1p twice would corrupt the target
distribution and produce incorrect normalization
statistics.

Generated File
--------------

normalization.npz

Contents:

    input_mean
    input_std
    target_mean
    target_std

Example
-------

python generate_normalization.py

Output:

datasets/corrdiff/normalization.npz

Usage in Dataset Loader
-----------------------

The resulting file is consumed by:

    ZarrCorrDiffDataset

which performs:

    x = (x - input_mean) / input_std

    y = (y - target_mean) / target_std

during sample loading.


The target statistics are computed on the log-transformed precipitation field because the 
dataset builder stores log1p(precipitation) directly into the Zarr dataset.

Project
-------
Radar precipitation downscaling using NVIDIA CorrDiff.
"""

import numpy as np
import zarr
from pathlib import Path

DATASET_PATH = "datasets/corrdiff/train.zarr"
OUTPUT_PATH = "datasets/corrdiff/normalization.npz"

print("Opening dataset...")

z = zarr.open(DATASET_PATH, mode="r")

x = z["input"]
y = z["target"]

print("Computing input statistics...")

# ============================================================================
# COMPUTE INPUT STATISTICS
# ============================================================================
#
# Input shape:
#
#     (N, C, H, W)
#
# Statistics are computed per channel over:
#
#     samples
#     height
#     width
#
# preserving one mean/std value per predictor variable.
#

input_mean = np.mean(x, axis=(0, 2, 3))
input_std = np.std(x, axis=(0, 2, 3))


# ============================================================================
#
# Target shape:
#
#     (N, 1, H, W)
#
# The target already contains:
#
#     log1p(precipitation)
#
# generated during dataset creation.
#
# Therefore we directly compute mean and standard deviation.
#
target = y[:]

target_mean = np.mean(
    target,
    axis=(0,2,3)
)

target_std = np.std(
    target,
    axis=(0,2,3)
)

# avoid division by zero
input_std[input_std == 0] = 1.0
target_std[target_std == 0] = 1.0

# ============================================================================
# SAVE NORMALIZATION FILE
# ============================================================================
#
# The resulting file is consumed by:
#
#     ZarrCorrDiffDataset
#
# during training and inference.
#
print("Saving normalization file...")

np.savez(
    OUTPUT_PATH,
    input_mean=input_mean.astype(np.float32),
    input_std=input_std.astype(np.float32),
    target_mean=target_mean.astype(np.float32),
    target_std=target_std.astype(np.float32),
)

print("Done.")