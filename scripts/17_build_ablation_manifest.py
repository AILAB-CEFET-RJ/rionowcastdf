#!/usr/bin/env python3
"""
CorrDiff - Fase 17
Materializa o desenho formal das ablações de condicionamento.

Este script NÃO treina modelos.
Ele:
- valida o contrato dos canais;
- expande cada experimento para uma lista explícita de canais;
- opcionalmente valida o handoff da Fase 16;
- gera manifestos de runs para execução posterior no ambiente NVIDIA CorrDiff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PHASE_VERSION = "phase17-conditioning-ablations-v1-nvidia-corrdiff-design"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--package-config-dir",
        type=Path,
        default=Path("configs"),
    )
    p.add_argument(
        "--phase16-handoff-dir",
        type=Path,
        default=Path(
            "analysis_outputs/16_baselines/nvidia_handoff"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis_outputs/17_conditioning_ablations"
        ),
    )
    p.add_argument(
        "--skip-phase16-handoff-check",
        action="store_true",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expand_experiment(exp, catalog):
    raw_selected = {
        channel
        for fam in exp["raw_families"]
        for channel in catalog["families"][fam]
    }
    raw = [
        channel
        for channel in catalog["raw_channels_canonical_order"]
        if channel in raw_selected
    ]

    derived_selected = {
        channel
        for fam in exp["derived_families"]
        for channel in catalog["derived_families"][fam]
    }
    derived = [
        channel
        for channel in catalog["derived_channels_canonical_order"]
        if channel in derived_selected
    ]

    channels = raw + derived
    if len(channels) != len(set(channels)):
        raise RuntimeError(
            f"Duplicate channels in {exp['experiment_id']}: {channels}"
        )

    return {
        **exp,
        "raw_channels": raw,
        "derived_channels": derived,
        "channels": channels,
        "n_raw_channels": len(raw),
        "n_derived_channels": len(derived),
        "n_total_channels": len(channels),
        "requires_derived_computation": bool(derived),
        "normalization_policy": (
            "fit mean/std on Phase-15 train only for exactly these channels"
        ),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in out.items():
                if isinstance(value, list):
                    out[key] = "|".join(map(str, value))
            w.writerow({k: out.get(k) for k in fields})


def main():
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; use --overwrite"
            )
        for p in args.output_dir.iterdir():
            if p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_json(
        args.package_config_dir / "phase17_channel_catalog.json"
    )
    experiments_doc = load_json(
        args.package_config_dir / "phase17_experiments.json"
    )
    contract = load_json(
        args.package_config_dir / "phase17_ablation_contract.json"
    )

    experiments = [
        expand_experiment(exp, catalog)
        for exp in experiments_doc["experiments"]
    ]

    raw_expected = catalog["raw_channels_canonical_order"]

    handoff_info = {
        "checked": False,
        "valid": None,
        "data_contract": None,
    }

    if not args.skip_phase16_handoff_check:
        contract_path = (
            args.phase16_handoff_dir
            / "corrdiff_data_contract.json"
        )
        if not contract_path.exists():
            raise FileNotFoundError(contract_path)

        phase16_contract = load_json(contract_path)
        actual = phase16_contract["channels_in_order"]

        if actual != raw_expected:
            raise RuntimeError(
                "Phase 16 raw channel order differs from Phase 17 "
                f"catalog.\nPhase16={actual}\nPhase17={raw_expected}"
            )

        handoff_info = {
            "checked": True,
            "valid": True,
            "data_contract": str(contract_path),
            "data_contract_sha256": sha256(contract_path),
        }

    # Explicit experiment table.
    fields = [
        "experiment_id",
        "name",
        "tier",
        "interpretation_role",
        "n_raw_channels",
        "n_derived_channels",
        "n_total_channels",
        "raw_families",
        "derived_families",
        "raw_channels",
        "derived_channels",
        "channels",
        "requires_derived_computation",
        "purpose",
    ]
    write_csv(
        args.output_dir / "ablation_matrix.csv",
        experiments,
        fields,
    )

    (args.output_dir / "ablation_matrix.json").write_text(
        json.dumps(
            {
                "phase_version": PHASE_VERSION,
                "experiments": experiments,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exp_lookup = {
        x["experiment_id"]: x
        for x in experiments
    }

    # Stage R: all experiments, all seeds.
    stage_r_cfg = contract["two_stage_execution"][
        "stage_R_regression_screen"
    ]
    stage_r_rows = []
    run_id = 0
    for exp_id in stage_r_cfg["experiments"]:
        exp = exp_lookup[exp_id]
        for seed in stage_r_cfg["seeds"]:
            run_id += 1
            stage_r_rows.append({
                "run_id": f"R{run_id:03d}",
                "stage": "regression_screen",
                "experiment_id": exp_id,
                "experiment_name": exp["name"],
                "seed": seed,
                "split_train": "train",
                "split_selection": "validation",
                "test_primary_access": False,
                "test_stress_ood_access": False,
                "n_channels": exp["n_total_channels"],
                "channels": exp["channels"],
                "status": "planned",
            })

    write_csv(
        args.output_dir / "planned_runs_stage_R.csv",
        stage_r_rows,
        [
            "run_id", "stage", "experiment_id",
            "experiment_name", "seed",
            "split_train", "split_selection",
            "test_primary_access", "test_stress_ood_access",
            "n_channels", "channels", "status",
        ],
    )

    # Stage D mandatory confirmation runs.
    stage_d_cfg = contract["two_stage_execution"][
        "stage_D_corrdiff_confirmation"
    ]
    stage_d_rows = []
    run_id = 0
    for exp_id in stage_d_cfg["mandatory_experiments"]:
        exp = exp_lookup[exp_id]
        for seed in stage_d_cfg["seeds"]:
            run_id += 1
            stage_d_rows.append({
                "run_id": f"D{run_id:03d}",
                "stage": "corrdiff_confirmation_mandatory",
                "experiment_id": exp_id,
                "experiment_name": exp["name"],
                "seed": seed,
                "split_train": "train",
                "split_selection": "validation",
                "test_primary_access": False,
                "test_stress_ood_access": False,
                "n_channels": exp["n_total_channels"],
                "channels": exp["channels"],
                "status": "planned",
            })

    write_csv(
        args.output_dir / "planned_runs_stage_D_mandatory.csv",
        stage_d_rows,
        [
            "run_id", "stage", "experiment_id",
            "experiment_name", "seed",
            "split_train", "split_selection",
            "test_primary_access", "test_stress_ood_access",
            "n_channels", "channels", "status",
        ],
    )

    optional_ids = (
        stage_d_cfg["optional_sufficiency_experiments"]
        + stage_d_cfg["optional_derived_family_experiments"]
    )
    optional_rows = []
    for exp_id in optional_ids:
        exp = exp_lookup[exp_id]
        optional_rows.append({
            "experiment_id": exp_id,
            "experiment_name": exp["name"],
            "tier": exp["tier"],
            "n_channels": exp["n_total_channels"],
            "channels": exp["channels"],
            "activation_rule": (
                "execute only if compute budget permits or if required "
                "to resolve an ambiguity from Stage R; document before run"
            ),
        })

    write_csv(
        args.output_dir / "optional_corrdiff_experiments.csv",
        optional_rows,
        [
            "experiment_id", "experiment_name", "tier",
            "n_channels", "channels", "activation_rule",
        ],
    )

    # Machine-readable adapter contract for Phase 18.
    adapter_contract = {
        "phase_version": PHASE_VERSION,
        "purpose": (
            "Input contract to be consumed by the native NVIDIA CorrDiff "
            "dataset/config adapter in Phase 18."
        ),
        "raw_channel_order": raw_expected,
        "derived_channel_order": (
            catalog["derived_channels_canonical_order"]
        ),
        "experiments": {
            exp["experiment_id"]: {
                "name": exp["name"],
                "channels": exp["channels"],
                "n_channels": exp["n_total_channels"],
                "derived_channels": exp["derived_channels"],
                "normalization": (
                    "train-only, fit separately per experiment channel set"
                ),
            }
            for exp in experiments
        },
        "split_indices_source": (
            "Phase 16 NVIDIA handoff / Phase 15 split arrays"
        ),
        "target_is_unchanged": True,
        "test_splits_must_remain_locked": True,
        "implementation_note": (
            "Do not assume a particular Hydra config path or NVIDIA "
            "class name here; Phase 18 maps this contract to the actual "
            "checked-out CorrDiff repository."
        ),
    }

    (
        args.output_dir / "nvidia_ablation_adapter_contract.json"
    ).write_text(
        json.dumps(
            adapter_contract,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "phase_version": PHASE_VERSION,
        "n_experiments": len(experiments),
        "n_stage_R_runs": len(stage_r_rows),
        "n_stage_D_mandatory_runs": len(stage_d_rows),
        "n_stage_D_optional_experiments": len(optional_rows),
        "stage_R_seeds": stage_r_cfg["seeds"],
        "stage_D_seeds": stage_d_cfg["seeds"],
        "tests_locked": True,
        "phase16_handoff": handoff_info,
        "next_phase": (
            "Phase 18: map this contract to the actual NVIDIA CorrDiff "
            "repository/configs and smoke-test native regression + diffusion "
            "data pipelines."
        ),
    }
    (
        args.output_dir / "phase17_manifest.json"
    ).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
