"""
CorrDiff dataset builder with automatic radar-cache preparation.

OVERVIEW
========
This module builds the patch-based Zarr dataset consumed by the CorrDiff
training pipeline. It combines monthly ERA5 predictors (single-level and
pressure-level) with radar reflectivity from the Sumare radar.

The complete execution flow is:

1. Parse configuration
   - Read the requested temporal interval, ERA5 variables, pressure levels,
     geographic domain, radar resolution, patch geometry and output options.

2. Ensure the radar cache is available
   - The training builder consumes radar grids from ``YYYYMMDD_HH_MM.npy``
     files, not directly from PNG images.
   - In ``--radar_cache_policy auto`` mode (default), the cache is reused when
     the cache directory exists and contains at least one ``.npy`` file.
   - If the directory is absent or empty, the radar-cache generation pipeline
     is executed automatically before the CorrDiff dataset is built.
   - ``--radar_cache_policy always`` runs the restart-safe cache generator even
     when cache files already exist; existing files are skipped and missing
     files are generated.
   - ``--radar_cache_policy never`` disables automatic generation and requires
     a pre-existing cache.

3. Build the radar cache when required
   Raw radar PNG layout expected by the cache generator::

       RADAR_DIR/YYYY/MM/DD/YYYY_MM_DD_HH_MM.png

   Cache pipeline::

       PNG image
         -> RGB pixels
         -> reflectivity values from the radar colour legend
         -> geographic 2-D target grid
         -> float32 NumPy array
         -> RADAR_CACHE_DIR/YYYYMMDD_HH_MM.npy

   Detailed cache steps:
   a. Generate the expected radar timestamps at the configured raw-radar
      cadence (2 minutes by default).
   b. Keep only timestamps for which the source PNG exists.
   c. Build the geographic grid from ``lat_range``, ``lon_range`` and
      ``radar_res``.
   d. Precompute the geographic-grid-to-image-pixel mapping once per worker.
   e. Process timestamps in parallel with ``ProcessPoolExecutor``.
   f. Skip an output file if it already exists, making the operation safe to
      restart after an interrupted run.
   g. Save each processed grid as ``float32`` ``.npy``.

4. Validate radar-cache coverage for the CorrDiff temporal sequence
   - CorrDiff normally samples hourly timestamps (``--time_frequency 1h``),
     although the raw radar/cache may be available every two minutes.
   - The script reports how many exact CorrDiff timestamps have a cache file.
   - Missing radar timestamps remain valid gaps and are counted during the
     dataset build rather than being silently substituted with nearest times.

5. Read and prepare ERA5 monthly data
   - Surface and pressure Parquet datasets are discovered month by month.
   - Pressure data are expected in long format: variables remain columns while
     the atmospheric level is stored in a separate ``pressure_level`` column.
   - Variable aliases are resolved to canonical CorrDiff channel names.
   - Auxiliary ERA5 dimensions that produce duplicate logical coordinates are
     coalesced only when their values do not conflict.
   - Requested pressure levels are validated explicitly.

6. Reconstruct ERA5 fields and interpolate to the radar grid
   - For each requested timestamp, surface fields are reconstructed as regular
     latitude/longitude grids.
   - Pressure fields are reconstructed independently for each requested level.
   - ``RegularGridInterpolator`` interpolates the multivariable cubes onto the
     exact radar grid.
   - The resulting predictor tensor has shape ``(C, H, W)``.
   - Pressure-level channel labels are generated logically, for example
     ``t_850``, ``r_850``, ``u_500`` and ``v_500``.

7. Synchronize ERA5 and radar
   For every CorrDiff timestamp:
   - require the exact radar cache file;
   - load and validate the radar grid;
   - build the ERA5 tensor for the same timestamp;
   - verify that ERA5 and radar have identical spatial shapes.

8. Extract spatial patches
   - Slide a ``patch_size x patch_size`` window with the configured ``stride``.
   - Reject patches whose radar-valid fraction is below
     ``minimum_radar_valid_ratio``.
   - Reject patches whose ERA5 finite-value fraction is below
     ``minimum_input_valid_ratio``.
   - Store patch row/column offsets so overlapping patches can later be traced
     back to their location in the full radar grid.

9. Prepare the radar target
   - Invalid radar pixels are represented through a separate mask.
   - The current implementation fills invalid values with zero, clips negative
     values to zero and applies ``log1p`` before storing the target.
   - This transformation is recorded in Zarr attributes and ``metadata.json``.

10. Write Zarr batches
    The output ``train.zarr`` contains::

        input       (N, C, P, P) float32
        target      (N, 1, P, P) float32
        mask        (N, 1, P, P) float32
        timestamps  (N,)         int64
        patch_row   (N,)         int16
        patch_col   (N,)         int16

    Patches are buffered in memory and flushed in batches to reduce Zarr resize
    and write overhead.

11. Compute and persist normalization statistics
    - Per-input-channel mean and standard deviation are calculated over finite
      input values.
    - Target statistics use only pixels marked valid by the radar mask.
    - Results are written to ``stats/*.npy`` and ``normalization.npz``.
    - ``metadata.json`` records channels, period, grid, patch configuration,
      target transformation and build counters.

RADAR CACHE POLICY EXAMPLES
===========================
Default: automatically build only when the cache is absent/empty::

    python -m src.corrdiff.builder.build_corrdiff_dataset_refactored \
      --begin "2011-01-01 00:00:00" \
      --end "2011-01-31 23:00:00" \
      --radar_cache_policy auto

Repair/fill a partial cache before building the dataset::

    ... --radar_cache_policy always

Require a pre-generated cache and never build it automatically::

    ... --radar_cache_policy never

IMPORTANT ASSUMPTIONS
=====================
- Raw radar images are expected at the path pattern documented above.
- The cache generator preserves the existing radar colour-to-reflectivity and
  pixel-mapping algorithm used by the project.
- The raw radar image dimensions used by that mapping default to 654x656 and
  are configurable through CLI options.
- Exact timestamp matching is intentional; no nearest-neighbour temporal
  substitution or temporal aggregation is performed by this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import gc
import json
import logging
import os
import platform
import shutil
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from tqdm import tqdm

from src.config.paths import ERA5_DIR, RADAR_CACHE_DIR, RADAR_DIR


# =============================================================================
# Logging
# =============================================================================


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("corrdiff.builder")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


LOGGER = setup_logger()


# =============================================================================
# Utilities
# =============================================================================


def _csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_int_list(value: str | None) -> list[int]:
    return [int(item) for item in _csv_list(value)]


def _normalize_timestamp(value: object) -> pd.Timestamp:
    """Return a timezone-naive UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _parse_end_timestamp(value: str) -> pd.Timestamp:
    """Interpret a date-only end value as the end of that day."""
    timestamp = pd.Timestamp(value)
    if len(value.strip()) <= 10:
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return timestamp


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for execution reports."""
    return datetime.now(timezone.utc).isoformat()


def _directory_size_bytes(path: str | Path) -> int:
    """Return the recursive on-disk size of a file/directory when available."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                pass
    return total


def _system_snapshot() -> dict[str, object]:
    """Collect lightweight, dependency-free execution environment metadata."""
    snapshot: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count_logical": os.cpu_count(),
        "pid": os.getpid(),
    }
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        snapshot["process_max_rss_raw"] = usage.ru_maxrss
        snapshot["process_user_cpu_seconds"] = usage.ru_utime
        snapshot["process_system_cpu_seconds"] = usage.ru_stime
    except Exception:
        pass
    return snapshot


# =============================================================================
# ERA5 dataset
# =============================================================================


class ERA5Dataset:
    """
    Monthly ERA5 reader that combines surface variables and pressure levels.

    Pressure data are read from one monthly Parquet partition containing a
    pressure-level column. Surface defaults mirror the current ERA5 single-level
    ETL. For a configuration such as::

        surface_variables = ["cape", "cin", "tcwv", "tcc", "z", "tp", "kx", "totalx"]
        pressure_variables = ["u", "v", "r", "q", "t", "w", "crwc"]
        pressure_levels = [850, 500]

    the returned tensor has channels in this order::

        cape, cin, tcwv, tcc, z, tp, kx, totalx,
        u_850, v_850, r_850, q_850, t_850, w_850, crwc_850,
        u_500, v_500, r_500, q_500, t_500, w_500, crwc_500

    The public method :meth:`get_tensor` returns an array with shape
    ``(C, H, W)`` interpolated onto the radar grid.
    """

    TIME_COLUMNS = ("valid_time", "time", "datetime", "date")
    LATITUDE_COLUMNS = ("latitude", "lat")
    LONGITUDE_COLUMNS = ("longitude", "lon")
    PRESSURE_COLUMNS = (
        "pressure_level",
        "level",
        "isobaricinhpa",
        "isobaricInhPa",
        "pressure",
    )

    VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
        # Current ERA5 single-level ETL
        "cape": ("cape", "convective_available_potential_energy"),
        "cin": ("cin", "convective_inhibition"),
        "tcwv": ("tcwv", "total_column_water_vapour"),
        "tcc": ("tcc", "total_cloud_cover"),
        "z": ("z", "geopotential"),
        "tp": ("tp", "total_precipitation"),
        "kx": ("kx", "k_index"),
        "totalx": ("totalx", "total_totals_index"),

        # Optional/future single-level variables kept for compatibility
        "t2m": ("t2m", "2m_temperature", "2_metre_temperature"),
        "u10": (
            "u10",
            "10m_u_component_of_wind",
            "10_metre_u_component_of_wind",
        ),
        "v10": (
            "v10",
            "10m_v_component_of_wind",
            "10_metre_v_component_of_wind",
        ),
        "sp": ("sp", "surface_pressure"),
        "msl": ("msl", "mean_sea_level_pressure"),
        "d2m": ("d2m", "2m_dewpoint_temperature", "2_metre_dewpoint_temperature"),

        # Pressure levels
        "u": ("u", "u_component_of_wind"),
        "v": ("v", "v_component_of_wind"),
        "r": ("r", "relative_humidity"),
        "q": ("q", "specific_humidity"),
        "t": ("t", "temperature"),
        "w": ("w", "vertical_velocity"),
        "crwc": ("crwc", "specific_rain_water_content"),
    }

    def __init__(
        self,
        era5_root: str | Path,
        surface_variables: Sequence[str],
        pressure_variables: Sequence[str],
        pressure_levels: Sequence[int],
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.root = Path(era5_root)
        self.surface_root = self.root / "single"
        self.pressure_root = self.root / "pressure"

        self.surface_variables = [
            self._canonical_variable(name) for name in surface_variables
        ]
        self.pressure_variables = [
            self._canonical_variable(name) for name in pressure_variables
        ]
        self.pressure_levels = sorted(
            {int(level) for level in pressure_levels}, reverse=True
        )

        self.target_lat = np.asarray(target_lat, dtype=np.float64)
        self.target_lon = np.asarray(target_lon, dtype=np.float64)
        if self.target_lat.shape != self.target_lon.shape:
            raise ValueError("target_lat and target_lon must have the same shape")
        if self.target_lat.ndim != 2:
            raise ValueError("target_lat and target_lon must be two-dimensional grids")

        self._target_points = np.column_stack(
            (self.target_lat.ravel(), self.target_lon.ravel())
        )

        self.channels = self._build_channels()

        self.loaded_month: str | None = None
        self.surface_df: pd.DataFrame | None = None
        self.pressure_df: pd.DataFrame | None = None
        self._surface_variable_columns: dict[str, str] = {}
        self._pressure_variable_columns: dict[str, str] = {}
        self._surface_lat = np.empty(0, dtype=np.float64)
        self._surface_lon = np.empty(0, dtype=np.float64)
        self._pressure_lat = np.empty(0, dtype=np.float64)
        self._pressure_lon = np.empty(0, dtype=np.float64)
        self._path_cache: dict[tuple[str, int, int], Path] = {}
        self.last_error: str | None = None
        self.performance: dict[str, object] = {
            "timings_seconds": {
                "surface_month_read": 0.0,
                "surface_month_prepare": 0.0,
                "pressure_month_read": 0.0,
                "pressure_month_prepare": 0.0,
                "interpolation": 0.0,
            },
            "operations": {
                "months_loaded": 0,
                "surface_month_reads": 0,
                "pressure_month_reads": 0,
                "interpolation_calls": 0,
            },
        }

        self.logger.info("ERA5 channels (%d): %s", self.n_channels, self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def _build_channels(self) -> list[str]:
        channels = list(self.surface_variables)
        channels.extend(
            f"{variable}_{level}"
            for level in self.pressure_levels
            for variable in self.pressure_variables
        )
        return channels

    @classmethod
    def _canonical_variable(cls, name: str) -> str:
        normalized = name.strip().lower().replace(" ", "_")
        for canonical, aliases in cls.VARIABLE_ALIASES.items():
            if normalized == canonical or normalized in {
                alias.lower() for alias in aliases
            }:
                return canonical
        return normalized

    @staticmethod
    def _detect_column(
        columns: Iterable[str], candidates: Sequence[str], label: str
    ) -> str:
        column_map = {str(column).lower(): str(column) for column in columns}
        for candidate in candidates:
            match = column_map.get(candidate.lower())
            if match is not None:
                return match
        raise ValueError(
            f"Could not find {label} column. Available columns: {list(columns)}"
        )

    @classmethod
    def _resolve_variable_columns(
        cls, columns: Iterable[str], requested: Sequence[str], dataset_name: str
    ) -> dict[str, str]:
        column_map = {str(column).lower(): str(column) for column in columns}
        resolved: dict[str, str] = {}

        for canonical in requested:
            aliases = cls.VARIABLE_ALIASES.get(canonical, (canonical,))
            candidates = (canonical, *aliases)
            actual = next(
                (column_map[candidate.lower()] for candidate in candidates if candidate.lower() in column_map),
                None,
            )
            if actual is None:
                raise ValueError(
                    f"Variable '{canonical}' was not found in {dataset_name}. "
                    f"Available columns: {list(columns)}"
                )
            resolved[canonical] = actual

        return resolved

    def _collapse_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
        key_columns: Sequence[str],
        value_columns: Sequence[str],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Collapse duplicate coordinates only when values are non-conflicting.

        ERA5 files may expose auxiliary dimensions such as ``expver`` or
        ``number``. After those columns are intentionally discarded, more than
        one row can remain for the same logical coordinate. We do not silently
        average conflicting values. Duplicate groups are accepted only when
        each requested variable has at most one distinct non-null value; in that
        case ``groupby.first`` coalesces complementary NaNs safely.
        """
        duplicate_mask = dataframe.duplicated(list(key_columns), keep=False)
        if not duplicate_mask.any():
            return dataframe

        duplicate_rows = dataframe.loc[duplicate_mask, [*key_columns, *value_columns]]
        grouped = duplicate_rows.groupby(list(key_columns), dropna=False, sort=False)

        conflicts: list[str] = []
        for variable in value_columns:
            distinct_counts = grouped[variable].nunique(dropna=True)
            conflicting = distinct_counts[distinct_counts > 1]
            if not conflicting.empty:
                conflicts.append(f"{variable}: {len(conflicting)} groups")

        if conflicts:
            raise ValueError(
                f"{dataset_name} contains conflicting duplicate coordinate rows "
                f"after dropping auxiliary ERA5 dimensions (for example expver/number). "
                f"Conflicts: {', '.join(conflicts)}"
            )

        duplicate_count = int(duplicate_mask.sum())
        self.logger.warning(
            "%s contains %d duplicate rows after removing auxiliary dimensions; "
            "coalescing non-conflicting values by logical coordinate",
            dataset_name,
            duplicate_count,
        )

        return (
            dataframe.groupby(list(key_columns), as_index=False, dropna=False, sort=False)[
                list(value_columns)
            ]
            .first()
        )

    @staticmethod
    def _month_key(timestamp: pd.Timestamp) -> str:
        return timestamp.strftime("%Y-%m")

    def clear_cache(self) -> None:
        self.loaded_month = None
        self.surface_df = None
        self.pressure_df = None
        self._surface_variable_columns = {}
        self._pressure_variable_columns = {}
        self._surface_lat = np.empty(0, dtype=np.float64)
        self._surface_lon = np.empty(0, dtype=np.float64)
        self._pressure_lat = np.empty(0, dtype=np.float64)
        self._pressure_lon = np.empty(0, dtype=np.float64)
        gc.collect()

    def _candidate_month_paths(
        self, root: Path, timestamp: pd.Timestamp, dataset_kind: str
    ) -> list[Path]:
        year = timestamp.year
        month = timestamp.month
        month_2 = f"{month:02d}"

        candidates = [
            root / f"year={year}" / f"month={month}",
            root / f"year={year}" / f"month={month_2}",
            root / str(year) / month_2,
            root / str(year) / str(month),
        ]

        if dataset_kind == "surface":
            candidates.extend(
                [
                    root / str(year) / month_2 / f"era5_single_{year}_{month_2}.parquet",
                    root / f"era5_single_{year}_{month_2}.parquet",
                ]
            )
        else:
            candidates.extend(
                [
                    root / str(year) / month_2 / f"era5_pressure_{year}_{month_2}.parquet",
                    root / f"era5_pressure_{year}_{month_2}.parquet",
                ]
            )

        return candidates

    @staticmethod
    def _contains_parquet(path: Path) -> bool:
        if path.is_file() and path.suffix.lower() == ".parquet":
            return True
        if path.is_dir():
            try:
                next(path.rglob("*.parquet"))
                return True
            except StopIteration:
                return False
        return False

    def _locate_month_path(
        self, root: Path, timestamp: pd.Timestamp, dataset_kind: str
    ) -> Path:
        key = (dataset_kind, timestamp.year, timestamp.month)
        cached = self._path_cache.get(key)
        if cached is not None and self._contains_parquet(cached):
            return cached

        tried: list[str] = []
        for candidate in self._candidate_month_paths(root, timestamp, dataset_kind):
            tried.append(str(candidate))
            if self._contains_parquet(candidate):
                self._path_cache[key] = candidate
                return candidate

        # Last-resort filename search for non-partitioned local copies.
        month_token = f"{timestamp.month:02d}"
        patterns = (
            f"*{timestamp.year}*{month_token}*.parquet",
            f"*{timestamp.year}*{timestamp.month}*.parquet",
        )
        for pattern in patterns:
            matches = sorted(root.rglob(pattern)) if root.exists() else []
            if matches:
                parent = matches[0].parent
                self._path_cache[key] = parent
                return parent

        raise FileNotFoundError(
            f"No {dataset_kind} Parquet partition found for "
            f"{timestamp:%Y-%m}. Tried: {tried}"
        )

    def _read_month_dataframe(
        self, root: Path, timestamp: pd.Timestamp, dataset_kind: str
    ) -> pd.DataFrame:
        try:
            path = self._locate_month_path(root, timestamp, dataset_kind)
            self.logger.info("Loading ERA5 %s month from %s", dataset_kind, path)
            return pd.read_parquet(path)
        except FileNotFoundError as direct_error:
            # Hive-partition fallback. This handles a dataset whose only stable
            # entry point is the root directory.
            try:
                self.logger.info(
                    "Trying partition-filtered read for ERA5 %s %s",
                    dataset_kind,
                    timestamp.strftime("%Y-%m"),
                )
                return pd.read_parquet(
                    root,
                    filters=[
                        ("year", "==", timestamp.year),
                        ("month", "==", timestamp.month),
                    ],
                )
            except Exception as filtered_error:
                raise FileNotFoundError(
                    f"Unable to read ERA5 {dataset_kind} for {timestamp:%Y-%m}. "
                    f"Direct discovery error: {direct_error}. "
                    f"Filtered read error: {filtered_error}"
                ) from filtered_error

    @staticmethod
    def _normalize_time_series(series: pd.Series) -> pd.Series:
        values = pd.to_datetime(series, errors="coerce", utc=True)
        if values.isna().any():
            raise ValueError("ERA5 time column contains invalid timestamps")
        return values.dt.tz_convert(None)

    def _prepare_surface_month(
        self, dataframe: pd.DataFrame, timestamp: pd.Timestamp
    ) -> None:
        time_column = self._detect_column(
            dataframe.columns, self.TIME_COLUMNS, "surface time"
        )
        latitude_column = self._detect_column(
            dataframe.columns, self.LATITUDE_COLUMNS, "surface latitude"
        )
        longitude_column = self._detect_column(
            dataframe.columns, self.LONGITUDE_COLUMNS, "surface longitude"
        )

        dataframe = dataframe.rename(
            columns={
                time_column: "_time",
                latitude_column: "_latitude",
                longitude_column: "_longitude",
            }
        )
        dataframe["_time"] = self._normalize_time_series(dataframe["_time"])
        dataframe["_latitude"] = pd.to_numeric(dataframe["_latitude"])
        dataframe["_longitude"] = pd.to_numeric(dataframe["_longitude"])

        month_start = timestamp.to_period("M").start_time
        month_end = (timestamp.to_period("M") + 1).start_time
        dataframe = dataframe[
            (dataframe["_time"] >= month_start)
            & (dataframe["_time"] < month_end)
        ]

        self._surface_variable_columns = self._resolve_variable_columns(
            dataframe.columns, self.surface_variables, "surface dataset"
        )
        value_columns = list(self._surface_variable_columns.values())
        keep_columns = [
            "_time",
            "_latitude",
            "_longitude",
            *value_columns,
        ]
        dataframe = dataframe[keep_columns]
        dataframe = self._collapse_duplicate_rows(
            dataframe,
            key_columns=("_time", "_latitude", "_longitude"),
            value_columns=value_columns,
            dataset_name="surface dataset",
        ).sort_values(
            ["_time", "_latitude", "_longitude"], kind="mergesort"
        )
        self._surface_lat = np.sort(dataframe["_latitude"].unique())
        self._surface_lon = np.sort(dataframe["_longitude"].unique())
        self.surface_df = dataframe.set_index("_time", drop=True)

    def _prepare_pressure_month(
        self, dataframe: pd.DataFrame, timestamp: pd.Timestamp
    ) -> None:
        time_column = self._detect_column(
            dataframe.columns, self.TIME_COLUMNS, "pressure time"
        )
        latitude_column = self._detect_column(
            dataframe.columns, self.LATITUDE_COLUMNS, "pressure latitude"
        )
        longitude_column = self._detect_column(
            dataframe.columns, self.LONGITUDE_COLUMNS, "pressure longitude"
        )
        pressure_column = self._detect_column(
            dataframe.columns, self.PRESSURE_COLUMNS, "pressure level"
        )

        dataframe = dataframe.rename(
            columns={
                time_column: "_time",
                latitude_column: "_latitude",
                longitude_column: "_longitude",
                pressure_column: "_pressure_level",
            }
        )
        dataframe["_time"] = self._normalize_time_series(dataframe["_time"])
        dataframe["_latitude"] = pd.to_numeric(dataframe["_latitude"])
        dataframe["_longitude"] = pd.to_numeric(dataframe["_longitude"])
        dataframe["_pressure_level"] = (
            pd.to_numeric(dataframe["_pressure_level"], errors="coerce")
            .round()
            .astype("Int16")
        )
        if dataframe["_pressure_level"].isna().any():
            raise ValueError("Pressure-level column contains invalid values")
        dataframe["_pressure_level"] = dataframe["_pressure_level"].astype(int)

        month_start = timestamp.to_period("M").start_time
        month_end = (timestamp.to_period("M") + 1).start_time
        dataframe = dataframe[
            (dataframe["_time"] >= month_start)
            & (dataframe["_time"] < month_end)
            & (dataframe["_pressure_level"].isin(self.pressure_levels))
        ]

        available_levels = sorted(dataframe["_pressure_level"].unique().tolist())
        missing_levels = sorted(set(self.pressure_levels) - set(available_levels))
        if missing_levels:
            raise ValueError(
                f"Requested pressure levels are missing from {timestamp:%Y-%m}: "
                f"{missing_levels}. Available levels: {available_levels}"
            )

        self._pressure_variable_columns = self._resolve_variable_columns(
            dataframe.columns, self.pressure_variables, "pressure dataset"
        )
        value_columns = list(self._pressure_variable_columns.values())
        keep_columns = [
            "_time",
            "_pressure_level",
            "_latitude",
            "_longitude",
            *value_columns,
        ]
        dataframe = dataframe[keep_columns]
        dataframe = self._collapse_duplicate_rows(
            dataframe,
            key_columns=("_time", "_pressure_level", "_latitude", "_longitude"),
            value_columns=value_columns,
            dataset_name="pressure dataset",
        ).sort_values(
            ["_time", "_pressure_level", "_latitude", "_longitude"],
            kind="mergesort",
        )
        self._pressure_lat = np.sort(dataframe["_latitude"].unique())
        self._pressure_lon = np.sort(dataframe["_longitude"].unique())
        self.pressure_df = dataframe.set_index(
            ["_time", "_pressure_level"], drop=True
        )

    def load_month(self, timestamp: object) -> None:
        timestamp = _normalize_timestamp(timestamp)
        month_key = self._month_key(timestamp)
        if month_key == self.loaded_month:
            return

        self.clear_cache()
        self.logger.info("Loading ERA5 month %s", month_key)

        if self.surface_variables:
            stage_started = time.perf_counter()
            surface = self._read_month_dataframe(
                self.surface_root, timestamp, "surface"
            )
            self.performance["timings_seconds"]["surface_month_read"] += (
                time.perf_counter() - stage_started
            )
            self.performance["operations"]["surface_month_reads"] += 1

            stage_started = time.perf_counter()
            self._prepare_surface_month(surface, timestamp)
            self.performance["timings_seconds"]["surface_month_prepare"] += (
                time.perf_counter() - stage_started
            )

        if self.pressure_variables and self.pressure_levels:
            stage_started = time.perf_counter()
            pressure = self._read_month_dataframe(
                self.pressure_root, timestamp, "pressure"
            )
            self.performance["timings_seconds"]["pressure_month_read"] += (
                time.perf_counter() - stage_started
            )
            self.performance["operations"]["pressure_month_reads"] += 1

            stage_started = time.perf_counter()
            self._prepare_pressure_month(pressure, timestamp)
            self.performance["timings_seconds"]["pressure_month_prepare"] += (
                time.perf_counter() - stage_started
            )

        self.loaded_month = month_key
        self.performance["operations"]["months_loaded"] += 1
        gc.collect()

    @staticmethod
    def _select_index_frame(
        dataframe: pd.DataFrame | None, key: object
    ) -> pd.DataFrame | None:
        if dataframe is None:
            return None
        try:
            selected = dataframe.loc[key]
        except KeyError:
            return None
        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T
        return selected

    @staticmethod
    def _select_pressure_frame(
        dataframe: pd.DataFrame | None, timestamp: pd.Timestamp, level: int
    ) -> pd.DataFrame | None:
        if dataframe is None:
            return None
        try:
            selected = dataframe.xs(
                (timestamp, level),
                level=("_time", "_pressure_level"),
                drop_level=True,
            )
        except KeyError:
            return None
        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T
        return selected

    @staticmethod
    def _frame_to_cube(
        frame: pd.DataFrame,
        variable_columns: Sequence[str],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        expected_size = len(latitudes) * len(longitudes)
        ordered = frame.sort_values(
            ["_latitude", "_longitude"], kind="mergesort"
        )

        coordinate_duplicates = ordered.duplicated(
            ["_latitude", "_longitude"]
        ).any()
        if len(ordered) == expected_size and not coordinate_duplicates:
            data = ordered[list(variable_columns)].to_numpy(dtype=np.float32)
            return data.reshape(len(latitudes), len(longitudes), len(variable_columns))

        # Robust fallback for duplicated/missing coordinate rows.
        fields = []
        for variable in variable_columns:
            grid = ordered.pivot_table(
                index="_latitude",
                columns="_longitude",
                values=variable,
                aggfunc="mean",
            ).reindex(index=latitudes, columns=longitudes)
            fields.append(grid.to_numpy(dtype=np.float32))
        return np.stack(fields, axis=-1)

    def _interpolate_cube(
        self,
        cube: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        interpolation_started = time.perf_counter()
        if len(latitudes) < 2 or len(longitudes) < 2:
            raise ValueError("ERA5 source grid must contain at least 2x2 points")

        interpolator = RegularGridInterpolator(
            (latitudes, longitudes),
            cube,
            bounds_error=False,
            fill_value=np.nan,
        )
        interpolated = interpolator(self._target_points)
        interpolated = interpolated.reshape(
            *self.target_lat.shape, cube.shape[-1]
        )
        result = np.moveaxis(interpolated, -1, 0).astype(np.float32, copy=False)
        self.performance["timings_seconds"]["interpolation"] += (
            time.perf_counter() - interpolation_started
        )
        self.performance["operations"]["interpolation_calls"] += 1
        return result

    def get_tensor(self, timestamp: object) -> np.ndarray | None:
        """Return all configured ERA5 channels as ``(C, H, W)``."""
        timestamp = _normalize_timestamp(timestamp)
        self.last_error = None

        try:
            self.load_month(timestamp)
            channel_groups: list[np.ndarray] = []

            if self.surface_variables:
                surface_frame = self._select_index_frame(self.surface_df, timestamp)
                if surface_frame is None or surface_frame.empty:
                    self.last_error = f"Missing surface timestamp {timestamp}"
                    return None

                surface_columns = [
                    self._surface_variable_columns[name]
                    for name in self.surface_variables
                ]
                surface_cube = self._frame_to_cube(
                    surface_frame,
                    surface_columns,
                    self._surface_lat,
                    self._surface_lon,
                )
                channel_groups.append(
                    self._interpolate_cube(
                        surface_cube, self._surface_lat, self._surface_lon
                    )
                )

            if self.pressure_variables and self.pressure_levels:
                pressure_columns = [
                    self._pressure_variable_columns[name]
                    for name in self.pressure_variables
                ]
                for level in self.pressure_levels:
                    pressure_frame = self._select_pressure_frame(
                        self.pressure_df, timestamp, level
                    )
                    if pressure_frame is None or pressure_frame.empty:
                        self.last_error = (
                            f"Missing pressure timestamp {timestamp} at {level} hPa"
                        )
                        return None

                    pressure_cube = self._frame_to_cube(
                        pressure_frame,
                        pressure_columns,
                        self._pressure_lat,
                        self._pressure_lon,
                    )
                    channel_groups.append(
                        self._interpolate_cube(
                            pressure_cube, self._pressure_lat, self._pressure_lon
                        )
                    )

            if not channel_groups:
                raise ValueError("No ERA5 variables were configured")

            tensor = np.concatenate(channel_groups, axis=0)
            if tensor.shape[0] != self.n_channels:
                raise RuntimeError(
                    f"Unexpected ERA5 channel count: {tensor.shape[0]} "
                    f"instead of {self.n_channels}"
                )
            return tensor.astype(np.float32, copy=False)

        except Exception as error:
            self.last_error = str(error)
            self.logger.warning("ERA5 failed for %s: %s", timestamp, error)
            return None

    def summary(self) -> None:
        self.logger.info("=" * 72)
        self.logger.info("ERA5 DATASET")
        self.logger.info("Root               : %s", self.root)
        self.logger.info("Surface variables  : %s", self.surface_variables)
        self.logger.info("Pressure variables : %s", self.pressure_variables)
        self.logger.info("Pressure levels    : %s", self.pressure_levels)
        self.logger.info("Channels           : %s", self.channels)
        self.logger.info("Target grid        : %s", self.target_lat.shape)
        self.logger.info("=" * 72)


# =============================================================================
# Radar cache generation
# =============================================================================


class RadarCacheSourceDataset:
    """Convert raw Sumare radar PNG images into cached geographic NPY grids.

    This class implements the same cache-generation logic used by the standalone
    high-performance radar-cache script. It is intentionally separate from
    :class:`RadarDataset`, which is the lightweight cache reader used during
    CorrDiff patch generation.

    The cache generator is restart-safe: :meth:`process_time` returns an
    existing cache file without rewriting it.
    """

    DEFAULT_LEGEND_VALUES = np.asarray(
        [50, 45, 40, 35, 30, 25, 20, 0], dtype=np.float32
    )
    DEFAULT_LEGEND_COLORS = np.asarray(
        [
            (197, 0, 197),
            (227, 6, 5),
            (255, 112, 0),
            (195, 230, 0),
            (4, 85, 4),
            (19, 122, 19),
            (0, 167, 12),
            (0, 0, 0),
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        radar_path: str | Path,
        cache_dir: str | Path,
        resolution_km: float,
        lat_range: Sequence[float],
        lon_range: Sequence[float],
        start_date: object,
        end_date: object,
        time_resolution_minutes: int = 2,
        image_height: int = 654,
        image_width: int = 656,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.radar_path = Path(radar_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.resolution_km = float(resolution_km)
        self.lat_range = tuple(float(value) for value in lat_range)
        self.lon_range = tuple(float(value) for value in lon_range)
        self.start = _normalize_timestamp(start_date)
        self.end = _normalize_timestamp(end_date)
        self.time_resolution_minutes = int(time_resolution_minutes)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

        if self.start > self.end:
            raise ValueError("Radar cache start_date must not be after end_date")
        if self.time_resolution_minutes <= 0:
            raise ValueError("Radar time resolution must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Radar source image dimensions must be positive")

        self.legend_values = self.DEFAULT_LEGEND_VALUES.copy()
        self.legend_colors = self.DEFAULT_LEGEND_COLORS.copy()

        self.lat: np.ndarray | None = None
        self.lon: np.ndarray | None = None
        self.Lat: np.ndarray | None = None
        self.Lon: np.ndarray | None = None
        self.px: np.ndarray | None = None
        self.py: np.ndarray | None = None

    def generate_timestamps(self) -> pd.DatetimeIndex:
        """Generate every raw-radar timestamp expected in the requested period."""
        return pd.date_range(
            start=self.start,
            end=self.end,
            freq=f"{self.time_resolution_minutes}min",
        )

    def filepath(self, timestamp: object) -> Path:
        """Return the raw PNG path for a radar timestamp."""
        timestamp = _normalize_timestamp(timestamp)
        return self.radar_path / timestamp.strftime(
            "%Y/%m/%d/%Y_%m_%d_%H_%M.png"
        )

    def cache_path(self, timestamp: object) -> Path:
        """Return the NPY cache path for a radar timestamp."""
        timestamp = _normalize_timestamp(timestamp)
        return self.cache_dir / f"{timestamp:%Y%m%d_%H_%M}.npy"

    def available_timestamps(self) -> list[pd.Timestamp]:
        """Return raw-radar timestamps for which a PNG exists on disk."""
        timestamps: list[pd.Timestamp] = []
        for timestamp in self.generate_timestamps():
            if self.filepath(timestamp).exists():
                timestamps.append(timestamp)

        self.logger.info(
            "Available raw radar timestamps: %d",
            len(timestamps),
        )
        return timestamps

    def build_grid(self) -> None:
        """Build the regular geographic grid used by the CorrDiff target."""
        degree_resolution = self.resolution_km / 111.0
        self.lat = np.arange(
            self.lat_range[0],
            self.lat_range[1],
            degree_resolution,
            dtype=np.float64,
        )
        self.lon = np.arange(
            self.lon_range[0],
            self.lon_range[1],
            degree_resolution,
            dtype=np.float64,
        )
        self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
        self.logger.info("Radar cache target grid shape: %s", self.Lat.shape)

    def precompute_pixel_map(self) -> None:
        """Precompute output-grid to source-image pixel coordinates."""
        if self.Lat is None or self.Lon is None:
            raise RuntimeError("build_grid() must be called before precompute_pixel_map()")

        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range

        self.px = (
            (self.Lon - lon_min)
            / (lon_max - lon_min)
            * (self.image_width - 1)
        ).astype(np.int32)

        self.py = (
            (lat_max - self.Lat)
            / (lat_max - lat_min)
            * (self.image_height - 1)
        ).astype(np.int32)

        self.logger.debug(
            "Radar source pixel map: px=%d..%d py=%d..%d",
            int(self.px.min()),
            int(self.px.max()),
            int(self.py.min()),
            int(self.py.max()),
        )

    def rgb_to_reflectivity(self, rgb: np.ndarray) -> np.ndarray:
        """Map RGB pixels to interpolated values of the configured radar legend."""
        flat = rgb.reshape(-1, 3).astype(np.float32, copy=False)

        distances = np.sqrt(
            ((flat[:, None, :] - self.legend_colors[None, :, :]) ** 2).sum(axis=2)
        )
        nearest = np.argsort(distances, axis=1)[:, :2]

        color1 = self.legend_colors[nearest[:, 0]]
        color2 = self.legend_colors[nearest[:, 1]]
        value1 = self.legend_values[nearest[:, 0]]
        value2 = self.legend_values[nearest[:, 1]]

        color_distance = np.linalg.norm(color1 - color2, axis=1)
        alpha = np.divide(
            np.linalg.norm(flat - color1, axis=1),
            color_distance,
            out=np.zeros_like(color_distance),
            where=color_distance != 0,
        )

        values = value1 + alpha * (value2 - value1)
        return values.reshape(rgb.shape[:2])

    def process_time(self, timestamp: object) -> np.ndarray | None:
        """Build or reuse the cached radar grid for one timestamp."""
        timestamp = _normalize_timestamp(timestamp)
        cache = self.cache_path(timestamp)

        if cache.exists():
            return np.load(cache, allow_pickle=False)

        path = self.filepath(timestamp)
        if not path.exists():
            return None

        try:
            image = np.asarray(Image.open(path).convert("RGB"))
        except Exception as error:
            self.logger.warning("Could not read radar image %s: %s", path, error)
            return None

        reflectivity = self.rgb_to_reflectivity(image)

        if self.px is None or self.py is None:
            raise RuntimeError(
                "precompute_pixel_map() must be called before process_time()"
            )

        height, width = reflectivity.shape
        px = np.clip(self.px, 0, width - 1)
        py = np.clip(self.py, 0, height - 1)
        grid = reflectivity[py, px].astype(np.float32, copy=False)

        np.save(cache, grid)
        return grid


_RADAR_CACHE_WORKER: RadarCacheSourceDataset | None = None


def _init_radar_cache_worker(
    radar_path: str,
    cache_dir: str,
    resolution_km: float,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    start_date: str,
    end_date: str,
    time_resolution_minutes: int,
    image_height: int,
    image_width: int,
) -> None:
    """Initialize one reusable radar-cache dataset per worker process."""
    global _RADAR_CACHE_WORKER

    _RADAR_CACHE_WORKER = RadarCacheSourceDataset(
        radar_path=radar_path,
        cache_dir=cache_dir,
        resolution_km=resolution_km,
        lat_range=lat_range,
        lon_range=lon_range,
        start_date=start_date,
        end_date=end_date,
        time_resolution_minutes=time_resolution_minutes,
        image_height=image_height,
        image_width=image_width,
    )
    _RADAR_CACHE_WORKER.build_grid()
    _RADAR_CACHE_WORKER.precompute_pixel_map()


def _process_radar_cache_timestamp(timestamp: pd.Timestamp) -> str:
    """Process one timestamp inside a radar-cache worker process."""
    if _RADAR_CACHE_WORKER is None:
        raise RuntimeError("Radar cache worker was not initialized")

    cache_file = _RADAR_CACHE_WORKER.cache_path(timestamp)
    if cache_file.exists():
        return "cached"

    grid = _RADAR_CACHE_WORKER.process_time(timestamp)
    if grid is None:
        return "missing"
    return "processed"


def radar_cache_has_files(cache_dir: str | Path) -> bool:
    """Return True when the cache directory contains at least one NPY file."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return False
    try:
        next(cache_dir.glob("*.npy"))
        return True
    except StopIteration:
        return False


def build_radar_cache(
    radar_path: str | Path,
    cache_dir: str | Path,
    start_date: object,
    end_date: object,
    lat_range: Sequence[float],
    lon_range: Sequence[float],
    resolution_km: float,
    max_workers: int,
    chunk_size: int,
    time_resolution_minutes: int = 2,
    image_height: int = 654,
    image_width: int = 656,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    """Generate the radar cache in parallel and return counters plus performance metrics."""
    logger = logger or LOGGER
    started = time.perf_counter()
    radar_path = Path(radar_path)
    cache_dir = Path(cache_dir)

    if not radar_path.exists():
        raise FileNotFoundError(
            f"Raw radar directory not found; cannot generate cache: {radar_path}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)

    source = RadarCacheSourceDataset(
        radar_path=radar_path,
        cache_dir=cache_dir,
        resolution_km=resolution_km,
        lat_range=lat_range,
        lon_range=lon_range,
        start_date=start_date,
        end_date=end_date,
        time_resolution_minutes=time_resolution_minutes,
        image_height=image_height,
        image_width=image_width,
        logger=logger,
    )

    timestamps = source.available_timestamps()
    if not timestamps:
        raise RuntimeError(
            f"No raw radar PNG files were found under {radar_path} for the "
            f"requested interval {start_date} to {end_date}"
        )

    workers = max(1, int(max_workers))
    chunksize = max(1, int(chunk_size))

    logger.info("=" * 72)
    logger.info("RADAR CACHE GENERATION")
    logger.info("Raw radar directory : %s", radar_path)
    logger.info("Cache directory     : %s", cache_dir)
    logger.info("Period              : %s to %s", start_date, end_date)
    logger.info("Raw cadence          : %d min", time_resolution_minutes)
    logger.info("Source PNGs found    : %d", len(timestamps))
    logger.info("Workers              : %d", workers)
    logger.info("Executor chunk size  : %d", chunksize)
    logger.info("=" * 72)

    counters = {"processed": 0, "cached": 0, "missing": 0}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_radar_cache_worker,
        initargs=(
            str(radar_path),
            str(cache_dir),
            float(resolution_km),
            tuple(float(value) for value in lat_range),
            tuple(float(value) for value in lon_range),
            str(_normalize_timestamp(start_date)),
            str(_normalize_timestamp(end_date)),
            int(time_resolution_minutes),
            int(image_height),
            int(image_width),
        ),
    ) as executor:
        results = executor.map(
            _process_radar_cache_timestamp,
            timestamps,
            chunksize=chunksize,
        )

        for result in tqdm(
            results,
            total=len(timestamps),
            desc="Building Radar Cache",
        ):
            counters[result] += 1

    logger.info("=" * 72)
    logger.info("RADAR CACHE SUMMARY")
    logger.info("Processed : %d", counters["processed"])
    logger.info("Cached    : %d", counters["cached"])
    logger.info("Missing   : %d", counters["missing"])
    logger.info("Directory : %s", cache_dir)
    logger.info("=" * 72)

    elapsed = time.perf_counter() - started
    metrics: dict[str, object] = {
        "mode": "generated",
        "build_triggered": True,
        "source_pngs": len(timestamps),
        "workers": workers,
        "executor_chunk_size": chunksize,
        "raw_time_resolution_minutes": int(time_resolution_minutes),
        "elapsed_seconds": elapsed,
        "timestamps_per_second": len(timestamps) / elapsed if elapsed > 0 else None,
        "counters": counters,
    }
    logger.info("Elapsed   : %.2f s", elapsed)
    if elapsed > 0:
        logger.info("Throughput: %.2f timestamps/s", len(timestamps) / elapsed)
    return metrics


def ensure_radar_cache(
    policy: str,
    radar_path: str | Path,
    cache_dir: str | Path,
    start_date: object,
    end_date: object,
    lat_range: Sequence[float],
    lon_range: Sequence[float],
    resolution_km: float,
    max_workers: int,
    chunk_size: int,
    time_resolution_minutes: int,
    image_height: int,
    image_width: int,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    """Ensure a usable radar cache exists and return cache-stage execution metrics."""
    logger = logger or LOGGER
    started = time.perf_counter()
    policy = policy.lower().strip()
    cache_dir = Path(cache_dir)
    has_cache = radar_cache_has_files(cache_dir)

    if policy not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported radar cache policy: {policy}")

    if policy == "never":
        if not has_cache:
            raise FileNotFoundError(
                f"Radar cache is required but missing/empty: {cache_dir}. "
                "Use --radar_cache_policy auto or always to generate it."
            )
        logger.info("Radar cache policy=never: using existing cache %s", cache_dir)
        elapsed = time.perf_counter() - started
        return {
            "mode": "reused",
            "build_triggered": False,
            "policy": policy,
            "elapsed_seconds": elapsed,
        }

    if policy == "auto" and has_cache:
        logger.info(
            "Radar cache found at %s; automatic generation is not required",
            cache_dir,
        )
        elapsed = time.perf_counter() - started
        return {
            "mode": "reused",
            "build_triggered": False,
            "policy": policy,
            "elapsed_seconds": elapsed,
        }

    reason = (
        "policy=always"
        if policy == "always"
        else "cache directory is missing or contains no NPY files"
    )
    logger.info("Radar cache generation required: %s", reason)

    build_metrics = build_radar_cache(
        radar_path=radar_path,
        cache_dir=cache_dir,
        start_date=start_date,
        end_date=end_date,
        lat_range=lat_range,
        lon_range=lon_range,
        resolution_km=resolution_km,
        max_workers=max_workers,
        chunk_size=chunk_size,
        time_resolution_minutes=time_resolution_minutes,
        image_height=image_height,
        image_width=image_width,
        logger=logger,
    )

    if not radar_cache_has_files(cache_dir):
        raise RuntimeError(
            f"Radar cache generation finished but no NPY files were produced in {cache_dir}"
        )

    total_elapsed = time.perf_counter() - started
    build_metrics["policy"] = policy
    build_metrics["ensure_elapsed_seconds"] = total_elapsed
    return build_metrics


def log_radar_cache_coverage(
    cache_dir: str | Path,
    start_date: object,
    end_date: object,
    frequency: str,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    """Log and return exact cache coverage for CorrDiff timestamps."""
    logger = logger or LOGGER
    started = time.perf_counter()
    cache_dir = Path(cache_dir)
    timestamps = pd.date_range(
        _normalize_timestamp(start_date),
        _normalize_timestamp(end_date),
        freq=frequency,
    )

    available = sum(
        (cache_dir / f"{timestamp:%Y%m%d_%H_%M}.npy").exists()
        for timestamp in timestamps
    )
    total = len(timestamps)
    missing = total - available

    logger.info("=" * 72)
    logger.info("RADAR CACHE COVERAGE FOR CORRDIFF")
    logger.info("Requested timestamps : %d", total)
    logger.info("Available cache      : %d", available)
    logger.info("Missing cache        : %d", missing)
    logger.info("Coverage             : %.2f%%", 100.0 * available / total if total else 0.0)
    logger.info("=" * 72)
    elapsed = time.perf_counter() - started
    return {
        "requested_timestamps": total,
        "available_timestamps": available,
        "missing_timestamps": missing,
        "coverage_ratio": available / total if total else 0.0,
        "coverage_percent": 100.0 * available / total if total else 0.0,
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# Radar cache dataset
# =============================================================================


class RadarDataset:
    """Read preprocessed radar grids from ``YYYYMMDD_HH_MM.npy`` files.

    Cache creation is handled before this reader is instantiated by
    :func:`ensure_radar_cache`. This class deliberately remains focused on
    deterministic cache reads and shape validation during CorrDiff generation.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        lat_range: Sequence[float],
        lon_range: Sequence[float],
        resolution_km: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.cache_dir = Path(cache_dir)
        self.lat_range = tuple(float(value) for value in lat_range)
        self.lon_range = tuple(float(value) for value in lon_range)
        self.resolution_km = float(resolution_km)

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Radar cache directory not found: {self.cache_dir}")

        degree_resolution = self.resolution_km / 111.0
        self.lat = np.arange(
            self.lat_range[0], self.lat_range[1], degree_resolution, dtype=np.float64
        )
        self.lon = np.arange(
            self.lon_range[0], self.lon_range[1], degree_resolution, dtype=np.float64
        )
        self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
        self.logger.info("Radar grid shape: %s", self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.Lat.shape

    def cache_path(self, timestamp: object) -> Path:
        timestamp = _normalize_timestamp(timestamp)
        return self.cache_dir / f"{timestamp:%Y%m%d_%H_%M}.npy"

    def exists(self, timestamp: object) -> bool:
        return self.cache_path(timestamp).exists()

    def get_grid(self, timestamp: object) -> np.ndarray | None:
        path = self.cache_path(timestamp)
        if not path.exists():
            return None
        try:
            grid = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        except Exception as error:
            self.logger.warning("Could not read radar cache %s: %s", path, error)
            return None
        if grid.shape != self.shape:
            self.logger.warning(
                "Radar shape mismatch for %s: %s instead of %s",
                path.name,
                grid.shape,
                self.shape,
            )
            return None
        return grid

    def summary(self) -> None:
        self.logger.info("=" * 72)
        self.logger.info("RADAR DATASET")
        self.logger.info("Cache directory : %s", self.cache_dir)
        self.logger.info("Grid shape      : %s", self.shape)
        self.logger.info("Latitude range  : %.4f to %.4f", self.lat.min(), self.lat.max())
        self.logger.info("Longitude range : %.4f to %.4f", self.lon.min(), self.lon.max())
        self.logger.info("Resolution      : %.2f km", self.resolution_km)
        self.logger.info("=" * 72)


# =============================================================================
# CorrDiff dataset builder
# =============================================================================


class CorrDiffDatasetBuilder:
    """Create patch-based Zarr data for CorrDiff regression and diffusion."""

    def __init__(
        self,
        era5: ERA5Dataset,
        radar: RadarDataset,
        output_dir: str | Path,
        start_date: object,
        end_date: object,
        time_frequency: str = "1h",
        patch_size: int = 32,
        stride: int = 16,
        chunk_size: int = 256,
        write_batch_size: int = 2048,
        minimum_radar_valid_ratio: float = 0.05,
        minimum_input_valid_ratio: float = 1.0,
        overwrite: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.era5 = era5
        self.radar = radar
        self.output_dir = Path(output_dir)
        self.start_date = _normalize_timestamp(start_date)
        self.end_date = _normalize_timestamp(end_date)
        self.time_frequency = time_frequency
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.chunk_size = int(chunk_size)
        self.write_batch_size = int(write_batch_size)
        self.minimum_radar_valid_ratio = float(minimum_radar_valid_ratio)
        self.minimum_input_valid_ratio = float(minimum_input_valid_ratio)
        self.overwrite = overwrite

        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if self.patch_size <= 0 or self.stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if self.patch_size > min(self.radar.shape):
            raise ValueError(
                f"patch_size {self.patch_size} exceeds radar grid {self.radar.shape}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = self.output_dir / "train.zarr"
        self.stats_dir = self.output_dir / "stats"

        self.root = None
        self.input_ds = None
        self.target_ds = None
        self.mask_ds = None
        self.timestamps_ds = None
        self.patch_rows_ds = None
        self.patch_cols_ds = None

        self.sample_count = 0
        self.written_count = 0
        self.input_sum = np.zeros(self.era5.n_channels, dtype=np.float64)
        self.input_sq_sum = np.zeros(self.era5.n_channels, dtype=np.float64)
        self.input_count = np.zeros(self.era5.n_channels, dtype=np.int64)
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.target_count = 0

        self._buffer_x: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self._buffer_mask: list[np.ndarray] = []
        self._buffer_timestamp: list[int] = []
        self._buffer_patch_row: list[int] = []
        self._buffer_patch_col: list[int] = []

        self.counters = {
            "timestamps_total": 0,
            "timestamps_processed": 0,
            "missing_radar": 0,
            "invalid_radar": 0,
            "missing_era5": 0,
            "shape_mismatch": 0,
            "patches_rejected_radar": 0,
            "patches_rejected_input": 0,
            "patches_written": 0,
        }

        # Cumulative performance instrumentation. Times use perf_counter() and
        # therefore represent wall-clock elapsed time for each measured stage.
        self.performance: dict[str, object] = {
            "timings_seconds": {
                "prepare_output": 0.0,
                "dataset_loop": 0.0,
                "radar_read": 0.0,
                "era5_tensor": 0.0,
                "patch_processing": 0.0,
                "zarr_flush": 0.0,
                "statistics": 0.0,
                "build_total": 0.0,
            },
            "operations": {
                "radar_read_attempts": 0,
                "era5_tensor_attempts": 0,
                "patch_process_calls": 0,
                "zarr_flush_count": 0,
                "zarr_uncompressed_bytes_submitted": 0,
            },
        }

        self.compressor = Blosc(cname="zstd", clevel=3, shuffle=2)

    def _timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.start_date, self.end_date, freq=self.time_frequency
        )

    def _prepare_output(self) -> None:
        if self.zarr_path.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output already exists: {self.zarr_path}. Use --overwrite."
                )
            shutil.rmtree(self.zarr_path)

        self.root = zarr.open(
            str(self.zarr_path), mode="w", zarr_version=2
        )
        spatial_shape = (self.patch_size, self.patch_size)

        self.input_ds = self.root.create_dataset(
            "input",
            shape=(0, self.era5.n_channels, *spatial_shape),
            chunks=(self.chunk_size, self.era5.n_channels, *spatial_shape),
            dtype="float32",
            compressor=self.compressor,
        )
        self.target_ds = self.root.create_dataset(
            "target",
            shape=(0, 1, *spatial_shape),
            chunks=(self.chunk_size, 1, *spatial_shape),
            dtype="float32",
            compressor=self.compressor,
        )
        self.mask_ds = self.root.create_dataset(
            "mask",
            shape=(0, 1, *spatial_shape),
            chunks=(self.chunk_size, 1, *spatial_shape),
            dtype="float32",
            compressor=self.compressor,
        )
        self.timestamps_ds = self.root.create_dataset(
            "timestamps",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="int64",
            compressor=self.compressor,
        )
        self.patch_rows_ds = self.root.create_dataset(
            "patch_row",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="int16",
            compressor=self.compressor,
        )
        self.patch_cols_ds = self.root.create_dataset(
            "patch_col",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="int16",
            compressor=self.compressor,
        )

        self.root.attrs.update(
            {
                "channels": self.era5.channels,
                "surface_variables": self.era5.surface_variables,
                "pressure_variables": self.era5.pressure_variables,
                "pressure_levels_hpa": self.era5.pressure_levels,
                "target_transform": "log1p",
                "patch_size": self.patch_size,
                "stride": self.stride,
                "radar_resolution_km": self.radar.resolution_km,
            }
        )

    def _append_patch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        timestamp: pd.Timestamp,
        patch_row: int,
        patch_col: int,
    ) -> None:
        self._buffer_x.append(x.astype(np.float32, copy=False))
        self._buffer_y.append(y.astype(np.float32, copy=False))
        self._buffer_mask.append(mask.astype(np.float32, copy=False))
        self._buffer_timestamp.append(int(timestamp.timestamp()))
        self._buffer_patch_row.append(patch_row)
        self._buffer_patch_col.append(patch_col)

        finite_input = np.isfinite(x)
        x64 = np.where(finite_input, x, 0.0).astype(np.float64, copy=False)
        self.input_sum += x64.sum(axis=(1, 2))
        self.input_sq_sum += np.square(x64).sum(axis=(1, 2))
        self.input_count += finite_input.sum(axis=(1, 2))

        valid_target = mask.astype(bool)
        target_values = y[valid_target]
        self.target_sum += float(target_values.sum(dtype=np.float64))
        self.target_sq_sum += float(
            np.square(target_values.astype(np.float64)).sum(dtype=np.float64)
        )
        self.target_count += int(target_values.size)

        self.sample_count += 1
        self.counters["patches_written"] += 1
        if len(self._buffer_x) >= self.write_batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer_x:
            return

        flush_started = time.perf_counter()
        x_batch = np.stack(self._buffer_x)
        y_batch = np.stack(self._buffer_y)
        mask_batch = np.stack(self._buffer_mask)
        timestamp_batch = np.asarray(self._buffer_timestamp, dtype=np.int64)
        row_batch = np.asarray(self._buffer_patch_row, dtype=np.int16)
        col_batch = np.asarray(self._buffer_patch_col, dtype=np.int16)

        submitted_bytes = (
            x_batch.nbytes
            + y_batch.nbytes
            + mask_batch.nbytes
            + timestamp_batch.nbytes
            + row_batch.nbytes
            + col_batch.nbytes
        )

        start = self.written_count
        end = start + len(x_batch)

        self.input_ds.resize((end, *self.input_ds.shape[1:]))
        self.target_ds.resize((end, *self.target_ds.shape[1:]))
        self.mask_ds.resize((end, *self.mask_ds.shape[1:]))
        self.timestamps_ds.resize((end,))
        self.patch_rows_ds.resize((end,))
        self.patch_cols_ds.resize((end,))

        self.input_ds[start:end] = x_batch
        self.target_ds[start:end] = y_batch
        self.mask_ds[start:end] = mask_batch
        self.timestamps_ds[start:end] = timestamp_batch
        self.patch_rows_ds[start:end] = row_batch
        self.patch_cols_ds[start:end] = col_batch

        self.written_count = end
        self._buffer_x.clear()
        self._buffer_y.clear()
        self._buffer_mask.clear()
        self._buffer_timestamp.clear()
        self._buffer_patch_row.clear()
        self._buffer_patch_col.clear()

        flush_elapsed = time.perf_counter() - flush_started
        timings = self.performance["timings_seconds"]
        operations = self.performance["operations"]
        timings["zarr_flush"] += flush_elapsed
        operations["zarr_flush_count"] += 1
        operations["zarr_uncompressed_bytes_submitted"] += int(submitted_bytes)

    def _process_patches(
        self, x: np.ndarray, radar_grid: np.ndarray, timestamp: pd.Timestamp
    ) -> int:
        height, width = radar_grid.shape
        created = 0

        for row in range(0, height - self.patch_size + 1, self.stride):
            for col in range(0, width - self.patch_size + 1, self.stride):
                x_patch = x[
                    :,
                    row : row + self.patch_size,
                    col : col + self.patch_size,
                ]
                radar_patch = radar_grid[
                    row : row + self.patch_size,
                    col : col + self.patch_size,
                ]

                radar_mask = np.isfinite(radar_patch)
                if float(radar_mask.mean()) < self.minimum_radar_valid_ratio:
                    self.counters["patches_rejected_radar"] += 1
                    continue

                input_valid_ratio = float(np.isfinite(x_patch).mean())
                if input_valid_ratio < self.minimum_input_valid_ratio:
                    self.counters["patches_rejected_input"] += 1
                    continue

                radar_filled = np.nan_to_num(
                    radar_patch, nan=0.0, posinf=0.0, neginf=0.0
                )
                radar_filled = np.clip(radar_filled, 0.0, None)
                target = np.log1p(radar_filled)[None, ...].astype(np.float32)
                mask = radar_mask[None, ...].astype(np.float32)

                self._append_patch(
                    x_patch,
                    target,
                    mask,
                    timestamp,
                    row,
                    col,
                )
                created += 1

        return created

    def _save_statistics(self) -> None:
        statistics_started = time.perf_counter()
        if self.sample_count == 0:
            raise RuntimeError("No samples were generated; statistics cannot be saved")
        if np.any(self.input_count == 0) or self.target_count == 0:
            raise RuntimeError("Normalization statistics contain empty channels")

        self.stats_dir.mkdir(parents=True, exist_ok=True)

        input_mean = self.input_sum / self.input_count
        input_variance = np.maximum(
            self.input_sq_sum / self.input_count - np.square(input_mean), 0.0
        )
        input_std = np.sqrt(input_variance)
        target_mean = self.target_sum / self.target_count
        target_variance = max(
            self.target_sq_sum / self.target_count - target_mean**2, 0.0
        )
        target_std = np.sqrt(target_variance)

        input_std[input_std == 0] = 1.0
        if target_std == 0:
            target_std = 1.0

        input_mean_f32 = input_mean.astype(np.float32)
        input_std_f32 = input_std.astype(np.float32)
        target_mean_f32 = np.asarray([target_mean], dtype=np.float32)
        target_std_f32 = np.asarray([target_std], dtype=np.float32)

        np.save(self.stats_dir / "input_mean.npy", input_mean_f32)
        np.save(self.stats_dir / "input_std.npy", input_std_f32)
        np.save(self.stats_dir / "target_mean.npy", target_mean_f32)
        np.save(self.stats_dir / "target_std.npy", target_std_f32)
        np.savez(
            self.output_dir / "normalization.npz",
            input_mean=input_mean_f32,
            input_std=input_std_f32,
            target_mean=target_mean_f32,
            target_std=target_std_f32,
        )

        # Human-readable channel statistics complement the compact NPY files.
        channel_rows = [
            {
                "kind": "input",
                "channel_index": index,
                "channel": channel,
                "mean": float(input_mean[index]),
                "std": float(input_std[index]),
                "finite_value_count": int(self.input_count[index]),
            }
            for index, channel in enumerate(self.era5.channels)
        ]
        channel_rows.append(
            {
                "kind": "target",
                "channel_index": 0,
                "channel": "radar_target",
                "mean": float(target_mean),
                "std": float(target_std),
                "finite_value_count": int(self.target_count),
            }
        )
        pd.DataFrame(channel_rows).to_csv(
            self.stats_dir / "channel_statistics.csv", index=False
        )

        metadata = {
            "num_samples": self.sample_count,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "time_frequency": self.time_frequency,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "input_channels": self.era5.n_channels,
            "channels": self.era5.channels,
            "surface_variables": self.era5.surface_variables,
            "pressure_variables": self.era5.pressure_variables,
            "pressure_levels_hpa": self.era5.pressure_levels,
            "radar_grid_shape": list(self.radar.shape),
            "radar_resolution_km": self.radar.resolution_km,
            "target_transform": "log1p",
            "counters": self.counters,
        }
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        self.performance["timings_seconds"]["statistics"] += (
            time.perf_counter() - statistics_started
        )

    def _finalize_performance_metrics(self) -> dict[str, object]:
        """Derive throughput/latency metrics after the dataset has been built."""
        timings = self.performance["timings_seconds"]
        operations = self.performance["operations"]

        dataset_loop_seconds = float(timings["dataset_loop"])
        build_total_seconds = float(timings["build_total"])
        zarr_seconds = float(timings["zarr_flush"])

        derived = {
            "timestamps_per_second_dataset_loop": (
                self.counters["timestamps_total"] / dataset_loop_seconds
                if dataset_loop_seconds > 0 else None
            ),
            "processed_timestamps_per_second": (
                self.counters["timestamps_processed"] / dataset_loop_seconds
                if dataset_loop_seconds > 0 else None
            ),
            "samples_per_second_total_build": (
                self.sample_count / build_total_seconds
                if build_total_seconds > 0 else None
            ),
            "samples_per_processed_timestamp": (
                self.sample_count / self.counters["timestamps_processed"]
                if self.counters["timestamps_processed"] else None
            ),
            "timestamp_acceptance_ratio": (
                self.counters["timestamps_processed"] / self.counters["timestamps_total"]
                if self.counters["timestamps_total"] else 0.0
            ),
            "average_radar_read_ms": (
                1000.0 * float(timings["radar_read"]) / operations["radar_read_attempts"]
                if operations["radar_read_attempts"] else None
            ),
            "average_era5_tensor_ms": (
                1000.0 * float(timings["era5_tensor"]) / operations["era5_tensor_attempts"]
                if operations["era5_tensor_attempts"] else None
            ),
            "average_patch_processing_ms": (
                1000.0 * float(timings["patch_processing"]) / operations["patch_process_calls"]
                if operations["patch_process_calls"] else None
            ),
            "zarr_submit_MB_per_second": (
                (operations["zarr_uncompressed_bytes_submitted"] / (1024 ** 2)) / zarr_seconds
                if zarr_seconds > 0 else None
            ),
        }

        self.performance["derived"] = derived
        self.performance["era5_internal"] = self.era5.performance
        self.performance["counters"] = dict(self.counters)
        self.performance["samples_generated"] = int(self.sample_count)
        self.performance["samples_written"] = int(self.written_count)
        return self.performance

    def build(self) -> dict[str, object]:
        build_started = time.perf_counter()

        stage_started = time.perf_counter()
        self._prepare_output()
        self.performance["timings_seconds"]["prepare_output"] += (
            time.perf_counter() - stage_started
        )

        timestamps = self._timestamps()
        self.counters["timestamps_total"] = len(timestamps)

        self.logger.info("=" * 72)
        self.logger.info("CORRDIFF DATASET BUILD")
        self.logger.info("Period      : %s to %s", self.start_date, self.end_date)
        self.logger.info("Timestamps  : %d", len(timestamps))
        self.logger.info("Channels    : %d", self.era5.n_channels)
        self.logger.info("Output      : %s", self.zarr_path)
        self.logger.info("=" * 72)

        loop_started = time.perf_counter()
        for timestamp in tqdm(timestamps, desc="Building CorrDiff dataset"):
            timestamp = _normalize_timestamp(timestamp)

            if not self.radar.exists(timestamp):
                self.counters["missing_radar"] += 1
                continue

            radar_started = time.perf_counter()
            radar_grid = self.radar.get_grid(timestamp)
            self.performance["timings_seconds"]["radar_read"] += (
                time.perf_counter() - radar_started
            )
            self.performance["operations"]["radar_read_attempts"] += 1

            if radar_grid is None or not np.isfinite(radar_grid).any():
                self.counters["invalid_radar"] += 1
                continue

            era5_started = time.perf_counter()
            era5_tensor = self.era5.get_tensor(timestamp)
            self.performance["timings_seconds"]["era5_tensor"] += (
                time.perf_counter() - era5_started
            )
            self.performance["operations"]["era5_tensor_attempts"] += 1

            if era5_tensor is None:
                self.counters["missing_era5"] += 1
                continue

            if era5_tensor.shape[1:] != radar_grid.shape:
                self.counters["shape_mismatch"] += 1
                self.logger.warning(
                    "Spatial mismatch at %s: ERA5 %s, radar %s",
                    timestamp,
                    era5_tensor.shape,
                    radar_grid.shape,
                )
                continue

            patch_started = time.perf_counter()
            created = self._process_patches(era5_tensor, radar_grid, timestamp)
            self.performance["timings_seconds"]["patch_processing"] += (
                time.perf_counter() - patch_started
            )
            self.performance["operations"]["patch_process_calls"] += 1

            if created > 0:
                self.counters["timestamps_processed"] += 1

        self.performance["timings_seconds"]["dataset_loop"] = (
            time.perf_counter() - loop_started
        )

        self._flush()
        self._save_statistics()

        self.performance["timings_seconds"]["build_total"] = (
            time.perf_counter() - build_started
        )
        metrics = self._finalize_performance_metrics()

        self.logger.info("=" * 72)
        self.logger.info("DATASET COMPLETED")
        self.logger.info("Samples written : %d", self.written_count)
        for name, value in self.counters.items():
            self.logger.info("%-26s: %d", name, value)
        self.logger.info("Build elapsed   : %.2f s", metrics["timings_seconds"]["build_total"])
        if metrics["derived"]["samples_per_second_total_build"] is not None:
            self.logger.info(
                "Sample throughput: %.2f samples/s",
                metrics["derived"]["samples_per_second_total_build"],
            )
        self.logger.info("=" * 72)
        return metrics


# =============================================================================
# CLI
# =============================================================================


def parameter_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CorrDiff Zarr dataset from ERA5 surface/pressure Parquet "
            "partitions and preprocessed radar NPY grids."
        )
    )
    parser.add_argument("-b", "--begin", required=True, help="Start date/time")
    parser.add_argument("-e", "--end", required=True, help="End date/time")
    parser.add_argument(
        "--era5_root", type=Path, default=Path(ERA5_DIR), help="ERA5 root directory"
    )
    parser.add_argument(
        "--radar_cache",
        type=Path,
        default=Path(RADAR_CACHE_DIR),
        help="Directory containing YYYYMMDD_HH_MM.npy radar grids",
    )
    parser.add_argument(
        "--radar_source",
        type=Path,
        default=Path(RADAR_DIR),
        help=(
            "Root directory containing raw radar PNG files in "
            "YYYY/MM/DD/YYYY_MM_DD_HH_MM.png layout"
        ),
    )
    parser.add_argument(
        "--radar_cache_policy",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Radar cache handling: auto builds only when cache is missing/empty; "
            "always runs the restart-safe cache builder; never requires an existing cache"
        ),
    )
    parser.add_argument(
        "--radar_cache_workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Worker processes used when generating radar cache",
    )
    parser.add_argument(
        "--radar_cache_chunk_size",
        type=int,
        default=200,
        help="ProcessPoolExecutor chunksize for radar-cache generation",
    )
    parser.add_argument(
        "--radar_time_resolution_minutes",
        type=int,
        default=2,
        help="Raw radar temporal cadence used to discover PNG files",
    )
    parser.add_argument(
        "--radar_image_height",
        type=int,
        default=654,
        help="Source radar PNG height used by the existing pixel-mapping algorithm",
    )
    parser.add_argument(
        "--radar_image_width",
        type=int,
        default=656,
        help="Source radar PNG width used by the existing pixel-mapping algorithm",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("datasets/corrdiff")
    )
    parser.add_argument(
        "--surface_variables",
        default="cape,cin,tcwv,tcc,z,tp,kx,totalx",
        help=(
            "Comma-separated ERA5 single-level variables. Default matches the "
            "current etl_era5_single pipeline."
        ),
    )
    parser.add_argument(
        "--pressure_variables",
        default="u,v,r,q,t,w,crwc",
        help="Comma-separated ERA5 pressure-level variables",
    )
    parser.add_argument("--pressure_levels", default="850,500")
    parser.add_argument(
        "--lat_range", nargs=2, type=float, default=[-23.5, -22.25]
    )
    parser.add_argument(
        "--lon_range", nargs=2, type=float, default=[-44.0, -42.5]
    )
    parser.add_argument("--radar_res", type=float, default=2.0)
    parser.add_argument("--time_frequency", default="1h")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--write_batch_size", type=int, default=2048)
    parser.add_argument("--minimum_radar_valid_ratio", type=float, default=0.05)
    parser.add_argument("--minimum_input_valid_ratio", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger = setup_logger(logging.DEBUG if args.debug else logging.INFO)

    run_id = uuid.uuid4().hex
    run_started_at = _utc_now_iso()
    pipeline_started = time.perf_counter()
    status = "running"
    error_info: dict[str, object] | None = None

    start_date = pd.Timestamp(args.begin)
    end_date = _parse_end_timestamp(args.end)
    output_dir = Path(args.output_dir)
    stats_dir = output_dir / "stats"

    cache_metrics: dict[str, object] = {}
    coverage_metrics: dict[str, object] = {}
    builder_metrics: dict[str, object] = {}
    initialization_elapsed = 0.0

    try:
        cache_metrics = ensure_radar_cache(
            policy=args.radar_cache_policy,
            radar_path=args.radar_source,
            cache_dir=args.radar_cache,
            start_date=start_date,
            end_date=end_date,
            lat_range=args.lat_range,
            lon_range=args.lon_range,
            resolution_km=args.radar_res,
            max_workers=args.radar_cache_workers,
            chunk_size=args.radar_cache_chunk_size,
            time_resolution_minutes=args.radar_time_resolution_minutes,
            image_height=args.radar_image_height,
            image_width=args.radar_image_width,
            logger=logger,
        )

        coverage_metrics = log_radar_cache_coverage(
            cache_dir=args.radar_cache,
            start_date=start_date,
            end_date=end_date,
            frequency=args.time_frequency,
            logger=logger,
        )

        initialization_started = time.perf_counter()
        radar = RadarDataset(
            cache_dir=args.radar_cache,
            lat_range=args.lat_range,
            lon_range=args.lon_range,
            resolution_km=args.radar_res,
            logger=logger,
        )

        era5 = ERA5Dataset(
            era5_root=args.era5_root,
            surface_variables=_csv_list(args.surface_variables),
            pressure_variables=_csv_list(args.pressure_variables),
            pressure_levels=_csv_int_list(args.pressure_levels),
            target_lat=radar.Lat,
            target_lon=radar.Lon,
            logger=logger,
        )
        initialization_elapsed = time.perf_counter() - initialization_started

        radar.summary()
        era5.summary()

        builder = CorrDiffDatasetBuilder(
            era5=era5,
            radar=radar,
            output_dir=args.output_dir,
            start_date=start_date,
            end_date=end_date,
            time_frequency=args.time_frequency,
            patch_size=args.patch_size,
            stride=args.stride,
            chunk_size=args.chunk_size,
            write_batch_size=args.write_batch_size,
            minimum_radar_valid_ratio=args.minimum_radar_valid_ratio,
            minimum_input_valid_ratio=args.minimum_input_valid_ratio,
            overwrite=args.overwrite,
            logger=logger,
        )

        builder_metrics = builder.build()
        status = "completed"

    except Exception as error:
        status = "failed"
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise

    finally:
        pipeline_elapsed = time.perf_counter() - pipeline_started
        run_ended_at = _utc_now_iso()

        report = {
            "schema_version": 1,
            "run": {
                "run_id": run_id,
                "status": status,
                "started_at_utc": run_started_at,
                "ended_at_utc": run_ended_at,
                "elapsed_seconds": pipeline_elapsed,
                "command": " ".join(sys.argv),
                "error": error_info,
            },
            "system": _system_snapshot(),
            "configuration": {
                "begin": str(start_date),
                "end": str(end_date),
                "era5_root": str(args.era5_root),
                "radar_source": str(args.radar_source),
                "radar_cache": str(args.radar_cache),
                "radar_cache_policy": args.radar_cache_policy,
                "radar_cache_workers": args.radar_cache_workers,
                "radar_cache_chunk_size": args.radar_cache_chunk_size,
                "radar_time_resolution_minutes": args.radar_time_resolution_minutes,
                "output_dir": str(args.output_dir),
                "surface_variables": _csv_list(args.surface_variables),
                "pressure_variables": _csv_list(args.pressure_variables),
                "pressure_levels_hpa": _csv_int_list(args.pressure_levels),
                "lat_range": list(args.lat_range),
                "lon_range": list(args.lon_range),
                "radar_resolution_km": args.radar_res,
                "time_frequency": args.time_frequency,
                "patch_size": args.patch_size,
                "stride": args.stride,
                "chunk_size": args.chunk_size,
                "write_batch_size": args.write_batch_size,
                "minimum_radar_valid_ratio": args.minimum_radar_valid_ratio,
                "minimum_input_valid_ratio": args.minimum_input_valid_ratio,
                "overwrite": args.overwrite,
            },
            "stages": {
                "radar_cache": cache_metrics,
                "radar_cache_coverage": coverage_metrics,
                "dataset_initialization": {
                    "elapsed_seconds": initialization_elapsed
                },
                "dataset_build": builder_metrics,
            },
            "artifacts": {
                "train_zarr_size_bytes": _directory_size_bytes(output_dir / "train.zarr"),
                "stats_size_bytes_before_report": _directory_size_bytes(stats_dir),
                "output_directory_size_bytes_before_report": _directory_size_bytes(output_dir),
            },
        }

        try:
            stats_dir.mkdir(parents=True, exist_ok=True)
            report_path = stats_dir / "execution_report.json"
            with report_path.open("w", encoding="utf-8") as file:
                json.dump(report, file, indent=2)

            stage_rows = []
            if builder_metrics:
                for name, seconds in builder_metrics.get("timings_seconds", {}).items():
                    stage_rows.append({
                        "scope": "dataset_build",
                        "stage": name,
                        "elapsed_seconds": float(seconds),
                    })
                era5_internal = builder_metrics.get("era5_internal", {})
                for name, seconds in era5_internal.get("timings_seconds", {}).items():
                    stage_rows.append({
                        "scope": "era5_internal",
                        "stage": name,
                        "elapsed_seconds": float(seconds),
                    })

            stage_rows.extend(
                [
                    {
                        "scope": "pipeline",
                        "stage": "radar_cache",
                        "elapsed_seconds": float(
                            cache_metrics.get(
                                "ensure_elapsed_seconds",
                                cache_metrics.get("elapsed_seconds", 0.0),
                            )
                        ),
                    },
                    {
                        "scope": "pipeline",
                        "stage": "radar_cache_coverage",
                        "elapsed_seconds": float(
                            coverage_metrics.get("elapsed_seconds", 0.0)
                        ),
                    },
                    {
                        "scope": "pipeline",
                        "stage": "dataset_initialization",
                        "elapsed_seconds": float(initialization_elapsed),
                    },
                    {
                        "scope": "pipeline",
                        "stage": "pipeline_total",
                        "elapsed_seconds": float(pipeline_elapsed),
                    },
                ]
            )
            pd.DataFrame(stage_rows).to_csv(
                stats_dir / "execution_stage_timings.csv", index=False
            )
            logger.info("Execution report: %s", report_path)
        except Exception as report_error:
            logger.error("Could not write execution report: %s", report_error)


if __name__ == "__main__":
    main(parameter_parser())
