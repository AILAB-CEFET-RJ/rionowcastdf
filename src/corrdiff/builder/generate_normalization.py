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

# shape:
# input  -> (N, C, H, W)
# target -> (N, 1, H, W)

input_mean = np.mean(x, axis=(0, 2, 3))
input_std = np.std(x, axis=(0, 2, 3))

#target = np.log1p(y[:])
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

print("Saving normalization file...")

np.savez(
    OUTPUT_PATH,
    input_mean=input_mean.astype(np.float32),
    input_std=input_std.astype(np.float32),
    target_mean=target_mean.astype(np.float32),
    target_std=target_std.astype(np.float32),
)

print("Done.")