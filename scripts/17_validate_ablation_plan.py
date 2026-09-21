#!/usr/bin/env python3
"""Validate CorrDiff Phase 17 conditioning-ablation design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    return p.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()

    catalog = load_json(
        args.config_dir / "phase17_channel_catalog.json"
    )
    exp_doc = load_json(
        args.config_dir / "phase17_experiments.json"
    )
    contract = load_json(
        args.config_dir / "phase17_ablation_contract.json"
    )

    raw_order = catalog["raw_channels_canonical_order"]
    families = catalog["families"]
    derived_order = catalog["derived_channels_canonical_order"]
    derived_families = catalog["derived_families"]

    errors = []
    warnings = []

    # Raw families must be disjoint and exactly cover raw12.
    flat_raw = [
        c
        for fam in families.values()
        for c in fam
    ]
    if len(flat_raw) != len(set(flat_raw)):
        errors.append("Raw channel families overlap.")
    if set(flat_raw) != set(raw_order):
        errors.append(
            "Raw channel families do not exactly cover raw12."
        )

    # Derived families must be disjoint and exactly cover the catalog.
    flat_derived = [
        c
        for fam in derived_families.values()
        for c in fam
    ]
    if len(flat_derived) != len(set(flat_derived)):
        errors.append("Derived channel families overlap.")
    if set(flat_derived) != set(derived_order):
        errors.append(
            "Derived families do not exactly cover derived catalog."
        )

    experiments = exp_doc["experiments"]
    ids = [e["experiment_id"] for e in experiments]
    names = [e["name"] for e in experiments]

    if len(ids) != len(set(ids)):
        errors.append("Duplicate experiment_id.")
    if len(names) != len(set(names)):
        errors.append("Duplicate experiment name.")

    expanded_sets = {}
    for e in experiments:
        raw_selected = set()
        for fam in e["raw_families"]:
            if fam not in families:
                errors.append(
                    f"{e['experiment_id']}: unknown raw family {fam}"
                )
                continue
            raw_selected.update(families[fam])
        raw = [
            c for c in raw_order
            if c in raw_selected
        ]

        derived_selected = set()
        for fam in e["derived_families"]:
            if fam not in derived_families:
                errors.append(
                    f"{e['experiment_id']}: unknown derived family {fam}"
                )
                continue
            derived_selected.update(derived_families[fam])
        derived = [
            c for c in derived_order
            if c in derived_selected
        ]

        channels = raw + derived
        if len(channels) != len(set(channels)):
            errors.append(
                f"{e['experiment_id']}: duplicate channel."
            )
        expanded_sets[e["experiment_id"]] = channels

    # Reference must be raw12, in canonical raw order.
    if expanded_sets.get("C00") != raw_order:
        errors.append(
            "C00 must exactly equal canonical raw12 channel order."
        )

    # The three single-family experiments.
    expected = {
        "C01": families["moisture"],
        "C02": families["thermal"],
        "C03": families["dynamics"],
    }
    for exp_id, channels in expected.items():
        if expanded_sets.get(exp_id) != channels:
            errors.append(
                f"{exp_id} does not match its intended family."
            )

    # Leave-one-family-out identities.
    checks = {
        "C04": [
            c for c in raw_order
            if c in set(families["moisture"] + families["thermal"])
        ],
        "C05": [
            c for c in raw_order
            if c in set(families["moisture"] + families["dynamics"])
        ],
        "C06": [
            c for c in raw_order
            if c in set(families["thermal"] + families["dynamics"])
        ],
    }
    for exp_id, channels in checks.items():
        if expanded_sets.get(exp_id) != channels:
            errors.append(
                f"{exp_id} is not the intended leave-one-family-out set."
            )

    # All-derived representation.
    if expanded_sets.get("C07") != raw_order + derived_order:
        errors.append(
            "C07 must be raw12 followed by all derived channels."
        )

    # No test access in design.
    frozen = contract["frozen_from_phase15_16"]
    if not frozen.get("validation_only_for_ablation_selection", False):
        errors.append(
            "Ablation selection must be validation-only."
        )

    for stage_name, stage in contract[
        "two_stage_execution"
    ].items():
        if not stage.get("tests_locked", False):
            errors.append(
                f"{stage_name}: tests_locked must be true."
            )

    # Scientific wording guards.
    semantics = catalog["derived_semantics"]
    if "not height-normalized" not in semantics["bulk_wind_note"]:
        warnings.append(
            "Bulk wind difference semantics should explicitly state "
            "it is not height-normalized shear."
        )
    if "do not label" not in semantics["delta_t_note"].lower():
        warnings.append(
            "Temperature-difference semantics should guard against "
            "calling delta_t instability."
        )

    print("=" * 90)
    print("CORRDIFF PHASE 17 - DESIGN VALIDATION")
    print("=" * 90)
    print("experiments:", len(experiments))
    print("raw channels:", len(raw_order))
    print("derived channels:", len(derived_order))
    print("errors:", len(errors))
    print("warnings:", len(warnings))

    for x in errors:
        print("ERROR:", x)
    for x in warnings:
        print("WARNING:", x)

    if errors:
        raise SystemExit(1)

    print("STATUS: PASS")


if __name__ == "__main__":
    main()
