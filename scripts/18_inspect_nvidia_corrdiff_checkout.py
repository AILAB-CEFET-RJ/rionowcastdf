#!/usr/bin/env python3
"""
Phase 18 - inspect a real NVIDIA PhysicsNeMo CorrDiff checkout.

The script does not modify the NVIDIA repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PHASE_VERSION = "phase18-nvidia-corrdiff-native-integration-v1-smoke-only"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corrdiff-dir",
        type=Path,
        required=True,
        help=(
            "Path to examples/weather/corrdiff in the checked-out "
            "NVIDIA PhysicsNeMo repository."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_outputs/18_nvidia_integration/"
            "phase18_nvidia_checkout_audit.json"
        ),
    )
    return p.parse_args()


def git_value(cwd: Path, *args):
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def find_repo_root(path: Path) -> Path | None:
    cur = path.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def file_flags(corrdiff: Path):
    required = [
        "train.py",
        "generate.py",
        "score_samples.py",
        "datasets/base.py",
        "conf/config_training_custom.yaml",
        "conf/config_generate_custom.yaml",
        "conf/base/model/regression.yaml",
        "conf/base/model/diffusion.yaml",
        "conf/base/model_size/mini.yaml",
        "conf/base/training/regression.yaml",
        "conf/base/training/diffusion.yaml",
    ]
    return {
        rel: (corrdiff / rel).exists()
        for rel in required
    }


def scan_text(path: Path, needles: list[str]):
    if not path.exists():
        return {
            n: False for n in needles
        }
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    return {
        n: (n in text)
        for n in needles
    }


def main():
    args = parse_args()
    corrdiff = args.corrdiff_dir.expanduser().resolve()
    if not corrdiff.exists():
        raise FileNotFoundError(corrdiff)

    flags = file_flags(corrdiff)
    missing = [
        name
        for name, ok in flags.items()
        if not ok
    ]

    train_scan = scan_text(
        corrdiff / "train.py",
        [
            "register_dataset",
            "dataset.type",
            "validation",
            "regression_checkpoint_path",
        ],
    )
    base_scan = scan_text(
        corrdiff / "datasets/base.py",
        [
            "class DownscalingDataset",
            "def input_channels",
            "def output_channels",
            "def image_shape",
            "def __len__",
        ],
    )

    repo = find_repo_root(corrdiff)
    git = {
        "repo_root": str(repo) if repo else None,
        "commit": (
            git_value(repo, "rev-parse", "HEAD")
            if repo else None
        ),
        "branch": (
            git_value(
                repo,
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            )
            if repo else None
        ),
        "status_porcelain": (
            git_value(repo, "status", "--porcelain")
            if repo else None
        ),
    }

    status = (
        "PASS"
        if not missing
        and base_scan["class DownscalingDataset"]
        else "FAIL"
    )

    doc = {
        "phase_version": PHASE_VERSION,
        "status": status,
        "corrdiff_dir": str(corrdiff),
        "required_files": flags,
        "missing_files": missing,
        "train_py_feature_scan": train_scan,
        "base_dataset_interface_scan": base_scan,
        "git": git,
        "integration_strategy": (
            "Use the official custom-dataset hook "
            "(file.py::Class) and add only new Hydra configs; "
            "do not patch NVIDIA core files in Phase 18."
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            doc,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        doc,
        indent=2,
        ensure_ascii=False,
    ))

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
