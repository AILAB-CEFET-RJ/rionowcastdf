import zarr
import numpy as np

dataset_dir = "./datasets/corrdiff/"
root = zarr.open(dataset_dir + "train.zarr", mode="r")

target = root["target"]

print("GLOBAL MIN:", target[:].min())
print("GLOBAL MAX:", target[:].max())
print("GLOBAL MEAN:", target[:].mean())

norm = np.load(
    "datasets/corrdiff/normalization.npz"
)

for k in norm.files:
    print(k, norm[k])
    
import zarr
import numpy as np

z = zarr.open(
    "datasets/corrdiff/train.zarr",
    mode="r"
)

y = z["target"]

print("mean =", np.mean(y))
print("std  =", np.std(y))

print("nonzero ratio =",
      np.mean(y[:] > 0))

print("pixels > 5 dBZ =",
      np.mean(y[:] > 5))

print("pixels > 20 dBZ =",
      np.mean(y[:] > 20))