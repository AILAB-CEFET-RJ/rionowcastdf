"""
===============================================================================
CorrDiffDatasetBuilder.py
===============================================================================

Author:
    Vinicius / OpenAI Refactor

Description:
    This module builds a complete Zarr-based dataset pipeline compatible with
    NVIDIA CorrDiff / PhysicsNeMo training workflows.

    The builder performs:

    1. ERA5 loading from partitioned parquet datasets
    2. Temporal filtering
    3. Spatial interpolation of ERA5 variables into radar domain
    4. Radar image decoding into reflectivity fields
    5. Patch extraction
    6. NaN masking
    7. Train statistics computation
    8. Dataset normalization metadata export
    9. Zarr dataset generation optimized for ML training

Dataset Structure:
    datasets/corrdiff/
    ├── train.zarr/
    │   ├── input
    │   ├── target
    │   ├── mask
    │   └── timestamps
    │
    ├── stats/
    │   ├── input_mean.npy
    │   ├── input_std.npy
    │   ├── target_mean.npy
    │   └── target_std.npy
    │
    └── metadata.json

Zarr Arrays:
    input:
        Shape:
            (N, C, H, W)

        Example:
            (500000, 8, 32, 32)

        Meaning:
            N = samples
            C = ERA5 variables
            H/W = patch size

    target:
        Shape:
            (N, 1, H, W)

        Meaning:
            Radar reflectivity patches

    mask:
        Shape:
            (N, 1, H, W)

        Meaning:
            Valid radar pixels

    timestamps:
        Shape:
            (N,)

        Meaning:
            Unix timestamp for each patch

Compatible With:
    - CorrDiff Regression Training
    - CorrDiff Diffusion Training
    - PhysicsNeMo
    - Multi-GPU DDP
    - Torch DataLoader
    - Distributed training

Recommended Workflow:
    Step 1:
        Generate dataset

    Step 2:
        Compute statistics

    Step 3:
        Train regression model

    Step 4:
        Train diffusion model

Requirements:
    - zarr
    - numcodecs
    - pyarrow
    - scipy
    - pillow
    - pandas
    - numpy

Example:
    python3 -m src.spatiotemporal_builder.CorrdiffDatasetBuilder \
        -b 2024-01-01 \
        -e 2024-01-31 \
        --era5_variables u,v,t,q \
        --radar_res 2

===============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import zarr

from numcodecs import Blosc
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from src.config.paths import (
    ERA5_DIR,
    RADAR_CACHE_DIR,
    RADAR_DIR,
)

# =============================================================================
# LOGGER
# =============================================================================


def setup_logger():

    logger = logging.getLogger("corrdiff")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# =============================================================================
# ERA5 DATASET
# =============================================================================


class ERA5Dataset:

    def __init__(
        self,
        path,
        variables,
        start_date,
        end_date,
    ):

        self.logger = setup_logger()

        self.path = Path(path)

        self.variables = variables

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.df = None

        self.lat = None
        self.lon = None

    def load(self):

        self.logger.info(f"Loading ERA5 dataset from {self.path}")

        dataset = ds.dataset(
            self.path,
            format="parquet",
            partitioning="hive",
        )

        time_col = None

        for col in dataset.schema.names:

            if col in ["time", "valid_time"]:

                time_col = col
                break

        if time_col is None:
            raise ValueError("No time column found")

        columns = [
            "latitude",
            "longitude",
            time_col,
        ] + self.variables

        filter_expr = (
            (pc.field(time_col) >= pc.scalar(self.start_date))
            &
            (pc.field(time_col) <= pc.scalar(self.end_date))
        )

        table = dataset.to_table(
            columns=columns,
            filter=filter_expr,
        )

        df = table.to_pandas()

        df["time"] = pd.to_datetime(df[time_col])

        df = df.sort_values(
            ["time", "latitude", "longitude"]
        )

        self.df = df

        self.lat = np.sort(df["latitude"].unique())
        self.lon = np.sort(df["longitude"].unique())

        self.logger.info(f"ERA5 loaded: {df.shape}")

    def interpolate(
        self,
        era5_t,
        target_lat,
        target_lon,
    ):

        grids = []

        for var in self.variables:

            grid = era5_t.pivot(
                index="latitude",
                columns="longitude",
                values=var,
            ).values

            interp = RegularGridInterpolator(
                (self.lat, self.lon),
                grid,
                bounds_error=False,
                fill_value=np.nan,
            )

            pts = np.stack(
                [
                    target_lat.ravel(),
                    target_lon.ravel(),
                ],
                axis=-1,
            )

            vals = interp(pts).reshape(target_lat.shape)

            grids.append(vals)

        return np.stack(grids, axis=0)


# =============================================================================
# RADAR DATASET
# =============================================================================


class RadarDataset:

    def __init__(
        self,
        radar_path,
        cache_dir,
        resolution_km,
        lat_range,
        lon_range,
        start_date,
        end_date,
    ):

        self.logger = setup_logger()

        self.radar_path = Path(radar_path)

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        self.res_km = resolution_km

        self.lat_range = lat_range
        self.lon_range = lon_range

        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)

        self.pos_sumare = (
            -22.955139,
            -43.248278,
        )

        self.legend_values = np.array(
            [50, 45, 40, 35, 30, 25, 20, 0]
        )

        self.legend_colors = np.array(
            [
                (197, 0, 197),
                (227, 6, 5),
                (255, 112, 0),
                (195, 230, 0),
                (4, 85, 4),
                (19, 122, 19),
                (0, 167, 12),
                (0, 0, 0),
            ]
        )

    def build_grid(self):

        deg = self.res_km / 111.0

        self.lat = np.arange(
            self.lat_range[0],
            self.lat_range[1],
            deg,
        )

        self.lon = np.arange(
            self.lon_range[0],
            self.lon_range[1],
            deg,
        )

        self.Lon, self.Lat = np.meshgrid(
            self.lon,
            self.lat,
        )

    def precompute_pixel_map(self):

        lat0, lon0 = self.pos_sumare

        self.px = (
            (self.Lon - lon0) * -32.5 + lon0
        ).astype(int)

        self.py = (
            (self.Lat - lat0) * 19.5 + lat0
        ).astype(int)

    def filepath(self, t):

        return self.radar_path / t.strftime(
            "%Y/%m/%d/%Y_%m_%d_%H_%M.png"
        )

    def cache_path(self, t):

        return self.cache_dir / (
            f"{t.strftime('%Y%m%d_%H')}.npy"
        )

    def rgb_to_reflectivity(self, rgb):

        flat = rgb.reshape(-1, 3)

        dists = np.sqrt(
            (
                (
                    flat[:, None, :]
                    - self.legend_colors[None]
                )
                ** 2
            ).sum(axis=2)
        )

        idx = np.argsort(dists, axis=1)[:, :2]

        c1 = self.legend_colors[idx[:, 0]]
        c2 = self.legend_colors[idx[:, 1]]

        v1 = self.legend_values[idx[:, 0]]
        v2 = self.legend_values[idx[:, 1]]

        dist = np.linalg.norm(c1 - c2, axis=1)

        t = np.divide(
            np.linalg.norm(flat - c1, axis=1),
            dist,
            out=np.zeros_like(dist),
            where=dist != 0,
        )

        val = v1 + t * (v2 - v1)

        return val.reshape(rgb.shape[:2])

    def process_time(self, t):

        cache = self.cache_path(t)

        if cache.exists():
            return np.load(cache)

        path = self.filepath(t)

        if not path.exists():
            return None

        img = np.array(
            Image.open(path).convert("RGB")
        )

        reflect = self.rgb_to_reflectivity(img)

        H, W = reflect.shape

        px = np.clip(self.px, 0, W - 1)
        py = np.clip(self.py, 0, H - 1)

        grid = reflect[py, px]

        np.save(
            cache,
            grid.astype(np.float32),
        )

        return grid

    def get_grid(self, t):

        return self.process_time(
            pd.to_datetime(t)
        )


# =============================================================================
# CORRDIFF DATASET BUILDER
# =============================================================================


class CorrDiffDatasetBuilder:

    def __init__(
        self,
        era5,
        radar,
        output_dir,
        patch_size=32,
        stride=16,
        chunk_size=256,
    ):

        self.logger = setup_logger()

        self.era5 = era5
        self.radar = radar

        self.output_dir = Path(output_dir)

        self.patch_size = patch_size
        self.stride = stride
        self.chunk_size = chunk_size

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sample_id = 0

        self.zarr_path = (
            self.output_dir / "train.zarr"
        )

        compressor = Blosc(
            cname="zstd",
            clevel=3,
            shuffle=2,
        )

        self.root = zarr.open(
            str(self.zarr_path),
            mode="w",
            zarr_version=2,
)

        self.compressor = compressor

        self.input_ds = None
        self.target_ds = None
        self.mask_ds = None
        self.timestamps_ds = None

        self.running_sum = None
        self.running_sq_sum = None
        self.running_count = 0

        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.target_count = 0

    def initialize_zarr(self, input_channels):

        self.logger.info("Initializing Zarr arrays...")

        self.root = zarr.open(
            str(self.zarr_path),
            mode="w",
            zarr_version=2,
        )

        self.input_ds = self.root.create_dataset(
            name="input",
            shape=(1, input_channels,
                self.patch_size,
                self.patch_size),
            chunks=(self.chunk_size,
                    input_channels,
                    self.patch_size,
                    self.patch_size),
            dtype="float32",
            compressor=self.compressor,
        )

        self.target_ds = self.root.create_dataset(
            name="target",
            shape=(1, 1,
                self.patch_size,
                self.patch_size),
            chunks=(self.chunk_size,
                    1,
                    self.patch_size,
                    self.patch_size),
            dtype="float32",
            compressor=self.compressor,
        )

        self.mask_ds = self.root.create_dataset(
            name="mask",
            shape=(1, 1,
                self.patch_size,
                self.patch_size),
            chunks=(self.chunk_size,
                    1,
                    self.patch_size,
                    self.patch_size),
            dtype="float32",
            compressor=self.compressor,
        )

        self.timestamps_ds = self.root.create_dataset(
            name="timestamps",
            shape=(1,),
            chunks=(self.chunk_size,),
            dtype="int64",
            compressor=self.compressor,
        )

    def append_sample(
        self,
        xp,
        yp,
        mask,
        timestamp,
    ):

        idx = self.sample_id

        new_shape_input = (
            idx + 1,
            xp.shape[0],
            self.patch_size,
            self.patch_size,
        )

        new_shape_target = (
            idx + 1,
            1,
            self.patch_size,
            self.patch_size,
        )

        self.input_ds.resize(new_shape_input)
        self.target_ds.resize(new_shape_target)
        self.mask_ds.resize(new_shape_target)

        self.timestamps_ds.resize((idx + 1,))

        self.input_ds[idx] = xp.astype(np.float32)
        self.target_ds[idx] = yp.astype(np.float32)
        self.mask_ds[idx] = mask.astype(np.float32)

        self.timestamps_ds[idx] = int(
            pd.Timestamp(timestamp).timestamp()
        )

        if self.running_sum is None:

            self.running_sum = xp.sum(
                axis=(1, 2)
            )

            self.running_sq_sum = (
                xp ** 2
            ).sum(axis=(1, 2))

        else:

            self.running_sum += xp.sum(
                axis=(1, 2)
            )

            self.running_sq_sum += (
                xp ** 2
            ).sum(axis=(1, 2))

        self.running_count += (
            xp.shape[1] * xp.shape[2]
        )

        self.target_sum += yp.sum()

        self.target_sq_sum += (
            yp ** 2
        ).sum()

        self.target_count += yp.size

        self.sample_id += 1

    def save_stats(self):

        stats_dir = self.output_dir / "stats"

        stats_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        input_mean = (
            self.running_sum
            / self.running_count
        )

        input_std = np.sqrt(
            (
                self.running_sq_sum
                / self.running_count
            )
            - input_mean**2
        )

        target_mean = (
            self.target_sum
            / self.target_count
        )

        target_std = np.sqrt(
            (
                self.target_sq_sum
                / self.target_count
            )
            - target_mean**2
        )

        np.save(
            stats_dir / "input_mean.npy",
            input_mean.astype(np.float32),
        )

        np.save(
            stats_dir / "input_std.npy",
            input_std.astype(np.float32),
        )

        np.save(
            stats_dir / "target_mean.npy",
            np.array(
                [target_mean],
                dtype=np.float32,
            ),
        )

        np.save(
            stats_dir / "target_std.npy",
            np.array(
                [target_std],
                dtype=np.float32,
            ),
        )

        metadata = {
            "num_samples": int(self.sample_id),
            "patch_size": self.patch_size,
            "stride": self.stride,
            "input_channels": int(
                len(self.era5.variables)
            ),
            "variables": self.era5.variables,
        }

        with open(
            self.output_dir / "metadata.json",
            "w",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

    def build(self):

        times = sorted(
            self.era5.df["time"].unique()
        )

        self.logger.info(
            f"Total timesteps: {len(times)}"
        )

        for t in times:

            self.logger.info(
                f"Processing timestep: {t}"
            )

            era5_t = self.era5.df[
                self.era5.df["time"] == t
            ]

            if len(era5_t) == 0:
                continue

            X = self.era5.interpolate(
                era5_t,
                self.radar.Lat,
                self.radar.Lon,
            )

            Y = self.radar.get_grid(t)

            if Y is None:
                continue

            if np.isnan(Y).all():
                continue

            if self.input_ds is None:

                self.initialize_zarr(
                    input_channels=X.shape[0]
                )

            H, W = Y.shape

            patches_created = 0

            for i in range(
                0,
                H - self.patch_size + 1,
                self.stride,
            ):

                for j in range(
                    0,
                    W - self.patch_size + 1,
                    self.stride,
                ):

                    xp = X[
                        :,
                        i:i+self.patch_size,
                        j:j+self.patch_size,
                    ]

                    yp = Y[
                        i:i+self.patch_size,
                        j:j+self.patch_size,
                    ]

                    valid_ratio = np.mean(
                        ~np.isnan(yp)
                    )

                    if valid_ratio < 0.05:
                        continue

                    mask = ~np.isnan(yp)

                    yp = np.nan_to_num(
                        yp,
                        nan=0.0,
                    )

                    yp = yp[None, ...]
                    mask = mask[None, ...]

                    self.append_sample(
                        xp,
                        yp,
                        mask,
                        t,
                    )

                    patches_created += 1

            self.logger.info(
                f"Patches created: {patches_created}"
            )

        self.save_stats()

        self.logger.info(
            f"Dataset size: {self.sample_id}"
        )


# =============================================================================
# ARGUMENT PARSER
# =============================================================================


def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--begin",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--end",
        required=True,
    )

    parser.add_argument(
        "--era5_variables",
        type=str,
        default="u,v",
    )

    parser.add_argument(
        "--lat_range",
        nargs=2,
        type=float,
        default=[-23.5, -22.25],
    )

    parser.add_argument(
        "--lon_range",
        nargs=2,
        type=float,
        default=[-44.0, -42.5],
    )

    parser.add_argument(
        "--radar_res",
        type=int,
        default=2,
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main(args):

    era5 = ERA5Dataset(
        path=ERA5_DIR,
        variables=args.era5_variables.split(","),
        start_date=args.begin + " 00:00:00",
        end_date=args.end + " 23:00:00",
    )

    era5.load()

    radar = RadarDataset(
        radar_path=RADAR_DIR,
        cache_dir=RADAR_CACHE_DIR,
        resolution_km=args.radar_res,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        start_date=args.begin + " 00:00:00",
        end_date=args.end + " 23:59:00",
    )

    radar.build_grid()

    radar.precompute_pixel_map()

    builder = CorrDiffDatasetBuilder(
        era5=era5,
        radar=radar,
        output_dir="datasets/corrdiff",
        patch_size=32,
        stride=16,
        chunk_size=256,
    )

    builder.build()


if __name__ == "__main__":

    args = parameter_parser()

    main(args)