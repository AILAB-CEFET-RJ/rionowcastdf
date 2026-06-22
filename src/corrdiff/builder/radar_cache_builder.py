#!/usr/bin/env python3

"""
High-performance radar cache builder.

Pipeline:

PNG
 -> RGB
 -> Reflectivity
 -> 70x84 grid
 -> NPY cache

Optimized version:

- RadarDataset initialized only once per worker
- Parallel processing
- Restart-safe
- Suitable for multi-million image archives
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from tqdm import tqdm

from src.config.paths import (
    RADAR_DIR,
    RADAR_CACHE_DIR,
)

from src.corrdiff.builder.CorrdiffDatasetBuilder import (
    RadarDataset,
)

# ============================================================
# CONFIG
# ============================================================

START_DATE = "2011-01-01"
END_DATE = "2024-12-31"

LAT_RANGE = [-23.5, -22.25]
LON_RANGE = [-44.0, -42.5]

RADAR_RES_KM = 2

MAX_WORKERS = max(
    1,
    mp.cpu_count() - 2
)

CHUNK_SIZE = 200

# ============================================================
# GLOBAL WORKER STATE
# ============================================================

RADAR = None

# ============================================================
# WORKER INITIALIZER
# ============================================================


def init_worker():

    global RADAR

    RADAR = RadarDataset(
        radar_path=RADAR_DIR,
        cache_dir=RADAR_CACHE_DIR,
        resolution_km=RADAR_RES_KM,
        lat_range=LAT_RANGE,
        lon_range=LON_RANGE,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    RADAR.build_grid()

    RADAR.precompute_pixel_map()

# ============================================================
# WORKER
# ============================================================


def process_timestamp(timestamp):

    global RADAR

    cache_file = RADAR.cache_path(timestamp)

    if cache_file.exists():
        return "cached"

    grid = RADAR.process_time(timestamp)

    if grid is None:
        return "missing"

    return "processed"

# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 60)
    print("RADAR CACHE GENERATION V2")
    print("=" * 60)

    radar = RadarDataset(
        radar_path=RADAR_DIR,
        cache_dir=RADAR_CACHE_DIR,
        resolution_km=RADAR_RES_KM,
        lat_range=LAT_RANGE,
        lon_range=LON_RANGE,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    timestamps = radar.available_timestamps()

    print(f"Workers         : {MAX_WORKERS}")
    print(f"Timestamps      : {len(timestamps):,}")
    print(f"Chunk Size      : {CHUNK_SIZE}")

    processed = 0
    cached = 0
    missing = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
    ) as executor:

        results = executor.map(
            process_timestamp,
            timestamps,
            chunksize=CHUNK_SIZE,
        )

        for result in tqdm(
            results,
            total=len(timestamps),
            desc="Building Radar Cache",
        ):

            if result == "processed":
                processed += 1

            elif result == "cached":
                cached += 1

            elif result == "missing":
                missing += 1

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Processed : {processed:,}")
    print(f"Cached    : {cached:,}")
    print(f"Missing   : {missing:,}")

    print()
    print(f"Cache directory:")
    print(RADAR_CACHE_DIR)


if __name__ == "__main__":
    main()