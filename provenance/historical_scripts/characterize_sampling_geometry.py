#!/usr/bin/env python3
"""Build the sampling schedule and characterize temporal geometry.

This script reads only the eligible participant sample and Boolean cell masks.
It never opens raw EEG and never computes a spectral quantity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import mne
import numpy as np
import scipy

from sampling_design import NDRAWS, STRATA10, candidates_c, candidates_f5, draw_pair

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "DS005385_PARTICIPANT_CENSUS.csv"
SCHEDULE = REPO / "structural" / "ds005385_sampling_schedule.npz"
CSV_OUT = REPO / "docs" / "preregistration" / "SAMPLING_GEOMETRY_CHARACTERIZATION.csv"
ENV_OUT = REPO / "structural" / "software_environment.json"
DATASET = "ds005385"
REGIMES = ("C", "F5", "F10")
SUPPORT_START_S = 8.0
CELL_S = 4.0

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def quantiles(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q025": float(np.quantile(x, 0.025)),
        "q975": float(np.quantile(x, 0.975)),
    }

def centers(indices: np.ndarray) -> np.ndarray:
    return SUPPORT_START_S + (indices.astype(float) + 0.5) * CELL_S

def arm_span(indices: np.ndarray) -> float:
    return float((int(indices.max()) - int(indices.min()) + 1) * CELL_S)

def symmetric_nearest_separation(a: np.ndarray, b: np.ndarray) -> float:
    ac = centers(a)
    bc = centers(b)
    distances = np.abs(ac[:, None] - bc[None, :])
    return float(np.concatenate((distances.min(axis=1), distances.min(axis=0))).mean())

def has_f10_cross_stratum_adjacency(indices: np.ndarray) -> bool:
    stratum = {cell: i for i, cells in enumerate(STRATA10) for cell in cells}
    ordered = sorted(int(x) for x in indices)
    return any(
        right - left == 1 and stratum[left] != stratum[right]
        for left, right in zip(ordered, ordered[1:])
    )

def load_roster() -> tuple[list[str], np.ndarray]:
    with CENSUS.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["final_eligible"] == "True"]
    participant_ids = [row["participant_id"] for row in rows]
    masks = np.asarray(
        [[value == "1" for value in row["qc_eligible_mask"]] for row in rows],
        dtype=bool,
    )
    if len(participant_ids) != 531 or masks.shape != (531, 42):
        raise RuntimeError(f"Unexpected eligible-sample shape: {len(participant_ids)}, {masks.shape}")
    if len(set(participant_ids)) != len(participant_ids):
        raise RuntimeError("Participant identifiers are not unique")
    return participant_ids, masks

def build_schedule(participant_ids: list[str], masks: np.ndarray) -> np.ndarray:
    schedule = np.empty((len(participant_ids), len(REGIMES), NDRAWS, 2, 10), dtype=np.uint8)
    for participant_index, (participant, mask) in enumerate(zip(participant_ids, masks)):
        for regime_index, regime in enumerate(REGIMES):
            prepared = candidates_c(mask) if regime == "C" else candidates_f5(mask) if regime == "F5" else None
            for replicate in range(NDRAWS):
                a, b = draw_pair(mask, DATASET, participant, regime, replicate, prepared)
                if set(a) & set(b):
                    raise RuntimeError(f"A/B overlap: {participant} {regime} {replicate}")
                schedule[participant_index, regime_index, replicate, 0] = a
                schedule[participant_index, regime_index, replicate, 1] = b
    return schedule

def summarize(schedule: np.ndarray) -> list[dict[str, str | int | float]]:
    output: list[dict[str, str | int | float]] = []
    for regime_index, regime in enumerate(REGIMES):
        centroid_differences: list[float] = []
        arm_spans: list[float] = []
        nearest_separations: list[float] = []
        f10_draw_adjacency = 0
        f10_arm_adjacency = 0
        for participant_schedule in schedule[:, regime_index]:
            for a, b in participant_schedule:
                ac = centers(a)
                bc = centers(b)
                centroid_differences.append(float(abs(ac.mean() - bc.mean())))
                arm_spans.extend((arm_span(a), arm_span(b)))
                nearest_separations.append(symmetric_nearest_separation(a, b))
                if regime == "F10":
                    adjacent_a = has_f10_cross_stratum_adjacency(a)
                    adjacent_b = has_f10_cross_stratum_adjacency(b)
                    f10_arm_adjacency += int(adjacent_a) + int(adjacent_b)
                    f10_draw_adjacency += int(adjacent_a or adjacent_b)
        centroid = quantiles(centroid_differences)
        span = quantiles(arm_spans)
        nearest = quantiles(nearest_separations)
        n_pairs = schedule.shape[0] * schedule.shape[2]
        row: dict[str, str | int | float] = {
            "dataset": DATASET,
            "regime": regime,
            "participants": schedule.shape[0],
            "paired_draws_per_participant": schedule.shape[2],
            "participant_draw_pairs": n_pairs,
            "cells_per_arm": schedule.shape[-1],
            "retained_seconds_per_arm": schedule.shape[-1] * CELL_S,
            "ab_overlap_failures": 0,
        }
        for prefix, values in (
            ("abs_centroid_difference_s", centroid),
            ("arm_span_s", span),
            ("symmetric_nearest_separation_s", nearest),
        ):
            for suffix, value in values.items():
                row[f"{prefix}_{suffix}"] = value
        if regime == "F10":
            row["fraction_paired_draws_with_cross_stratum_adjacency"] = f10_draw_adjacency / n_pairs
            row["fraction_arms_with_cross_stratum_adjacency"] = f10_arm_adjacency / (2 * n_pairs)
        else:
            row["fraction_paired_draws_with_cross_stratum_adjacency"] = "NOT_APPLICABLE"
            row["fraction_arms_with_cross_stratum_adjacency"] = "NOT_APPLICABLE"
        output.append(row)
    return output

def main() -> None:
    participant_ids, masks = load_roster()
    schedule = build_schedule(participant_ids, masks)
    SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCHEDULE,
        participant_ids=np.asarray(participant_ids),
        regimes=np.asarray(REGIMES),
        eligible_masks=masks,
        schedule=schedule,
        support_start_seconds=np.asarray(SUPPORT_START_S),
        cell_seconds=np.asarray(CELL_S),
    )
    rows = summarize(schedule)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    environment = {
        "validation_date": "2026-08-29",
        "python": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mne": mne.__version__,
        "schedule_shape": list(schedule.shape),
        "schedule_dtype": str(schedule.dtype),
        "schedule_file": str(SCHEDULE.relative_to(REPO)).replace("\\", "/"),
        "schedule_sha256": sha256(SCHEDULE),
        "scientific_spectral_outcomes": False,
    }
    ENV_OUT.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"environment": environment, "geometry": rows}, indent=2))

if __name__ == "__main__":
    main()
