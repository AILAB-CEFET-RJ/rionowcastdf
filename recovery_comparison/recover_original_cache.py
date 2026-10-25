from pathlib import Path
from datetime import datetime
import json
import shutil

import numpy as np
import pandas as pd

from src.corrdiff.builder.dataset_builder import RadarCacheSourceDataset

original_file = Path(
    "recovery_comparison/temporal_coverage_original.parquet"
)
current_file = Path(
    "analysis_outputs/00_quality/temporal_coverage.parquet"
)
preserved = Path("/home/vsantos/data/radar_cache")
radar = Path("/home/vsantos/data/radar_sumare")

def available(path):
    frame = pd.read_parquet(path)
    return set(pd.to_datetime(
        frame.loc[frame["samples"].gt(0), "timestamp"],
        utc=True,
    ))

original = available(original_file)
current = available(current_file)

assert len(original) == 84684, len(original)
assert len(current) == 73843, len(current)
assert current <= original, "Há horários atuais fora da referência"
assert len(original - current) == 10841

run = Path("recovery_comparison") / datetime.now().strftime(
    "cache_original_%Y%m%d_%H%M%S_%f"
)
run.mkdir(parents=True, exist_ok=False)
cache = run / "cache"

source = RadarCacheSourceDataset(
    radar_path=radar,
    cache_dir=cache,
    resolution_km=2.0,
    lat_range=(-23.5, -22.25),
    lon_range=(-44.0, -42.5),
    start_date="2011-01-01",
    end_date="2024-12-31 23:00:00",
    image_height=654,
    image_width=656,
)
source.build_grid()
source.precompute_pixel_map()

def validate(array):
    if array.shape != (70, 84):
        raise ValueError(f"Shape inesperado: {array.shape}")
    if array.dtype != np.dtype("float32"):
        raise ValueError(f"Tipo inesperado: {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Grade contém valores não finitos")

copied = recovered = 0
failures = []

print("Cache de saída:", cache.resolve(), flush=True)

for index, t in enumerate(sorted(original), 1):
    name = t.strftime("%Y%m%d_%H_%M.npy")
    destination = cache / name
    try:
        if t in current:
            origin = preserved / name
            array = np.load(origin, allow_pickle=False)
            validate(array)

            shutil.copyfile(origin, destination)
            saved = np.load(destination, allow_pickle=False)
            validate(saved)
            if not np.array_equal(array, saved):
                raise ValueError("Cópia difere do arquivo preservado")
            copied += 1
        else:
            array = source.process_time(t)
            if array is None:
                raise ValueError("PNG ausente ou não legível")
            validate(array)

            saved = np.load(destination, allow_pickle=False)
            validate(saved)
            if not np.array_equal(array, saved):
                raise ValueError("Arquivo gravado difere da conversão")
            recovered += 1

    except Exception as exc:
        failures.append({
            "timestamp": t.isoformat(),
            "reason": str(exc),
        })
        print("FAIL:", failures[-1], flush=True)

    if index % 500 == 0 or index == len(original):
        print(
            f"{index}/{len(original)} | copiados={copied} "
            f"recuperados={recovered} falhas={len(failures)}",
            flush=True,
        )

expected_names = {
    t.strftime("%Y%m%d_%H_%M.npy") for t in original
}
actual_names = {p.name for p in cache.glob("*.npy")}
passed = (
    not failures
    and copied == 73843
    and recovered == 10841
    and actual_names == expected_names
)

report = {
    "status": "PASS" if passed else "FAIL",
    "cache_dir": str(cache.resolve()),
    "expected_timestamps": len(original),
    "copied": copied,
    "recovered": recovered,
    "failures": failures,
    "missing_files": sorted(expected_names - actual_names),
    "extra_files": sorted(actual_names - expected_names),
}
(run / "recovery_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)

if passed:
    (run / "READY.txt").write_text(str(cache.resolve()) + "\n")

print(json.dumps({
    k: v for k, v in report.items() if k != "failures"
}, indent=2), flush=True)
print("Relatório:", (run / "recovery_report.json").resolve(), flush=True)

raise SystemExit(0 if passed else 1)
