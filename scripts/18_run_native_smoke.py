#!/usr/bin/env python3
"""
Phase 18 - native NVIDIA CorrDiff smoke-test orchestrator.

Modes:
- compose: Hydra config composition only (no GPU/training).
- regression: tiny native regression training smoke.
- diffusion: tiny native diffusion training smoke, requires regression ckpt.

The generated configs only reference Phase-15 train/validation indices.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PHASE_VERSION = "phase18-nvidia-corrdiff-native-integration-v1-smoke-only"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corrdiff-dir", type=Path, required=True)
    p.add_argument(
        "--experiment-id",
        choices=["C00", "C07"],
        default="C00",
    )
    p.add_argument(
        "--mode",
        choices=["compose", "regression", "diffusion"],
        default="compose",
    )
    p.add_argument(
        "--regression-checkpoint",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_outputs/18_nvidia_integration/"
            "native_smoke_audit.json"
        ),
    )
    return p.parse_args()


def config_name(experiment_id: str, stage: str):
    return (
        f"config_training_rionowcast_"
        f"{experiment_id}_{stage}_smoke.yaml"
    )


def run_command(cmd, cwd, env):
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - t0,
        "stdout_tail": "\n".join(
            proc.stdout.splitlines()[-200:]
        ),
    }


def main():
    args = parse_args()
    corrdiff = args.corrdiff_dir.expanduser().resolve()

    if not (corrdiff / "train.py").exists():
        raise FileNotFoundError(corrdiff / "train.py")

    stage = (
        "regression"
        if args.mode in {"compose", "regression"}
        else "diffusion"
    )
    cfg = config_name(
        args.experiment_id,
        stage,
    )
    cfg_path = corrdiff / "conf" / cfg
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    env = os.environ.copy()

    if stage == "diffusion":
        if args.regression_checkpoint is None:
            raise RuntimeError(
                "--regression-checkpoint is required for diffusion smoke."
            )
        ckpt = args.regression_checkpoint.expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        env["CORRDIFF_REGRESSION_CKPT"] = str(ckpt)

    if args.mode == "compose":
        # Hydra composes the actual native configuration but exits before
        # model construction/training.
        cmd = [
            sys.executable,
            "train.py",
            f"--config-name={cfg}",
            "--cfg",
            "job",
        ]
    else:
        cmd = [
            sys.executable,
            "train.py",
            f"--config-name={cfg}",
        ]

    result = run_command(
        cmd,
        cwd=corrdiff,
        env=env,
    )

    audit = {
        "phase_version": PHASE_VERSION,
        "mode": args.mode,
        "experiment_id": args.experiment_id,
        "stage": stage,
        "corrdiff_dir": str(corrdiff),
        "config": str(cfg_path),
        "result": result,
        "tests_accessed": False,
        "status": (
            "PASS"
            if result["returncode"] == 0
            else "FAIL"
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ))

    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
