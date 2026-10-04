#!/usr/bin/env python3
"""Production entry point for the registered temporal-sampling analysis.

Importing this module or invoking it without ``--run`` cannot calculate an EEG
spectral outcome. The default action validates the registered inputs only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from preregistered_analysis import (
    BOOTSTRAP_RESAMPLES,
    REGIMES,
    REQUIRED_CHANNELS,
    NumericalFailure,
    absolute_displacement_summary,
    analyze_primary_contrast,
    analyze_rank_reproducibility,
    generate_bootstrap_indices,
    participant_markers,
    subset_registered_schedule,
)
from sampling_design import draw_pair


REPO = Path(__file__).resolve().parents[1]
REGISTERED = REPO / "osf_upload_package"
REGISTERED_CENSUS = REGISTERED / "DS005385_PARTICIPANT_CENSUS.csv"
REGISTERED_QC = REGISTERED / "structural" / "ds005385_qc_results.csv"
REGISTERED_SCHEDULE = REGISTERED / "structural" / "ds005385_sampling_schedule.npz"
REPLICATION_CENSUS = REPO / "structural" / "ds004148_session1_census.csv"

EXPECTED_HASHES = {
    "analysis_specification.yaml": "b01b98a0c03f6c36b1230d13ce88b1fe41bea897460f30e0f123cd38d60b5455",
    "DS005385_PARTICIPANT_CENSUS.csv": "90f425ba446daac8db3daa73e23035d4a6e9f6efc3a232671efad16cdfd04c82",
    "structural/ds005385_qc_results.csv": "4b9b8ac7e7d9622247e8502ff82fb80f1a6eda7d9a2a8c269231c5e73503d797",
    "structural/ds005385_sampling_schedule.npz": "95c0da1f105ff9f92a64010d120b8ebecc760c75540b448f209dc391e91649c1",
    "scripts/sampling_design.py": "d55ebefd87d32fee3ed48cbd45fabc3991041d1985792f60d7bcd0a6f9d0f445",
    "tests/test_psd_equivalence.py": "cd6c87250b08a647814867e67f800d9c208ec66f78ae53b35d260d0352a3287a",
}
EXPECTED_REPLICATION_CENSUS_SHA256 = (
    "ac76412b073040859ac382ec978a0835e40860f2a2bfce1dbab54e7e2efd5971"
)

ABS_UV = 500.0
P2P_UV = 1000.0
STEP_UV = 200.0
FLAT_SD_UV = 0.1
IDENTICAL_RUN_MS = 100.0


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def true_value(value: str) -> bool:
    return value.strip().lower() == "true"


def validate_registered_inputs() -> dict[str, object]:
    """Validate byte identity, rosters, and schedule without reading EEG."""
    for relative_path, expected in EXPECTED_HASHES.items():
        path = REGISTERED / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = file_digest(path)
        if observed != expected:
            raise RuntimeError(f"registered hash mismatch: {relative_path}: {observed} != {expected}")
    for relative_path in ("scripts/sampling_design.py", "tests/test_psd_equivalence.py"):
        observed = file_digest(REPO / relative_path)
        expected = EXPECTED_HASHES[relative_path]
        if observed != expected:
            raise RuntimeError(f"local implementation differs from registered file: {relative_path}")
    if file_digest(REPLICATION_CENSUS) != EXPECTED_REPLICATION_CENSUS_SHA256:
        raise RuntimeError("replication structural census differs from the post-registration input")

    census = read_csv(REGISTERED_CENSUS)
    qc = read_csv(REGISTERED_QC)
    primary_rows = [row for row in census if true_value(row["final_eligible"])]
    sensitivity_rows = [row for row in primary_rows if true_value(row["complete_support_sensitivity"])]
    if (len(census), len(primary_rows), len(sensitivity_rows)) != (608, 531, 516):
        raise RuntimeError("registered census counts do not equal 608/531/516")
    if len(qc) != 536 or sum(true_value(row["qc_eligible"]) for row in qc) != 531:
        raise RuntimeError("registered QC counts do not equal 536/531")

    with np.load(REGISTERED_SCHEDULE, allow_pickle=False) as archive:
        required_keys = {
            "participant_ids", "regimes", "eligible_masks", "schedule",
            "support_start_seconds", "cell_seconds",
        }
        if set(archive.files) != required_keys:
            raise RuntimeError(f"unexpected registered schedule keys: {archive.files}")
        participant_ids = archive["participant_ids"].astype(str)
        regimes = archive["regimes"].astype(str)
        masks = archive["eligible_masks"]
        schedule = archive["schedule"]
        support_start = float(archive["support_start_seconds"])
        cell_seconds = float(archive["cell_seconds"])

    expected_ids = np.asarray([row["participant_id"] for row in primary_rows])
    if schedule.shape != (531, 3, 200, 2, 10) or schedule.dtype != np.uint8:
        raise RuntimeError("registered schedule shape or dtype mismatch")
    if masks.shape != (531, 42) or masks.dtype != np.bool_:
        raise RuntimeError("registered eligible-mask shape or dtype mismatch")
    if not np.array_equal(regimes, np.asarray(REGIMES)):
        raise RuntimeError("registered regime ordering is not C/F5/F10")
    if not np.array_equal(participant_ids, expected_ids):
        raise RuntimeError("registered schedule participant ordering differs from the census")
    if support_start != 8.0 or cell_seconds != 4.0:
        raise RuntimeError("registered support metadata mismatch")
    if schedule.min() < 0 or schedule.max() > 41:
        raise RuntimeError("registered schedule contains an out-of-range cell")
    if np.any(np.diff(np.sort(schedule, axis=-1), axis=-1) == 0):
        raise RuntimeError("a registered schedule arm does not contain ten unique cells")
    overlap = (schedule[..., 0, :, None] == schedule[..., 1, None, :]).any()
    if overlap:
        raise RuntimeError("registered schedule contains within-draw A/B overlap")
    scheduled_masks = np.take_along_axis(
        np.broadcast_to(masks[:, None, None, None, :], (531, 3, 200, 2, 42)),
        schedule,
        axis=-1,
    )
    if not scheduled_masks.all():
        raise RuntimeError("registered schedule uses a cell outside a participant's eligible mask")

    return {
        "census": census,
        "primary_rows": primary_rows,
        "sensitivity_rows": sensitivity_rows,
        "participant_ids": participant_ids.tolist(),
        "eligible_masks": masks,
        "schedule": schedule,
        "schedule_sha256": EXPECTED_HASHES["structural/ds005385_sampling_schedule.npz"],
    }


def _longest_equal_run(samples: np.ndarray) -> int:
    if samples.size < 2:
        return int(samples.size)
    changes = np.flatnonzero(np.diff(samples) != 0)
    if changes.size == 0:
        return int(samples.size)
    boundaries = np.concatenate(([-1], changes, [samples.size - 1]))
    return int(np.diff(boundaries).max())


def time_domain_qc_mask(required_support_uvolts: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Apply the registered time-domain QC rules to the 42-cell support."""
    data = np.asarray(required_support_uvolts, dtype=float)
    samples_per_cell = int(round(4.0 * float(sampling_rate)))
    if data.shape != (5, 42 * samples_per_cell):
        raise ValueError("QC input must be five channels over the exact 168-second support")
    mastoid_mean = (data[3] + data[4]) / 2.0
    posterior = data[:3] - mastoid_mean
    identical_run_samples = int(round(IDENTICAL_RUN_MS * sampling_rate / 1000.0))
    mask = np.ones(42, dtype=bool)
    for cell in range(42):
        start = cell * samples_per_cell
        stop = start + samples_per_cell
        raw_window = data[:, start:stop]
        posterior_window = posterior[:, start:stop]
        centered = posterior_window - np.median(posterior_window, axis=1, keepdims=True)
        failures = (
            not np.isfinite(raw_window).all()
            or np.nanmax(np.abs(centered)) > ABS_UV
            or np.nanmax(np.ptp(posterior_window, axis=1)) > P2P_UV
            or np.nanmax(np.abs(np.diff(posterior_window, axis=1))) > STEP_UV
            or np.nanmin(np.std(raw_window, axis=1)) < FLAT_SD_UV
            or any(_longest_equal_run(channel) >= identical_run_samples for channel in raw_window)
        )
        mask[cell] = not failures
    return mask


def _load_required_support(raw_path: Path, reader: str) -> tuple[np.ndarray, float]:
    """Load only [8,176) s from five required channels; no filtering/resampling."""
    import mne

    if reader == "edf":
        raw = mne.io.read_raw_edf(raw_path, preload=False, verbose="ERROR")
    elif reader == "brainvision":
        raw = mne.io.read_raw_brainvision(raw_path, preload=False, verbose="ERROR")
    else:
        raise ValueError(reader)
    missing = [channel for channel in REQUIRED_CHANNELS if channel not in raw.ch_names]
    if missing:
        raise NumericalFailure(f"missing required channels: {','.join(missing)}")
    sampling_rate = float(raw.info["sfreq"])
    start = int(round(8.0 * sampling_rate))
    stop = int(round(176.0 * sampling_rate))
    if raw.n_times < int(round(180.0 * sampling_rate)):
        raise NumericalFailure("recording is shorter than the required 180 seconds")
    support_volts = raw.get_data(picks=list(REQUIRED_CHANNELS), start=start, stop=stop)
    expected = (5, int(round(168.0 * sampling_rate)))
    if support_volts.shape != expected:
        raise NumericalFailure(f"required support shape {support_volts.shape} != {expected}")
    if not np.isfinite(support_volts).all():
        raise NumericalFailure("nonfinite sample in required support")
    return support_volts, sampling_rate


def _primary_markers(
    dataset_root: Path,
    primary_rows: list[dict[str, str]],
    schedule: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(primary_rows), 3, 200, 2), dtype=float)
    for index, row in enumerate(primary_rows):
        raw_path = dataset_root / Path(row["target_recording"])
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if raw_path.stat().st_size != int(row["annex_bytes"]):
            raise RuntimeError(f"primary raw byte-count mismatch: {row['participant_id']}")
        if file_digest(raw_path) != row["annex_digest"]:
            raise RuntimeError(f"primary raw SHA-256 mismatch: {row['participant_id']}")
        support_volts, sampling_rate = _load_required_support(raw_path, "edf")
        if not np.isclose(sampling_rate, float(row["edf_eeg_sampling_hz"])):
            raise RuntimeError(f"primary sampling-rate mismatch: {row['participant_id']}")
        output[index] = participant_markers(support_volts, sampling_rate, schedule[index])
    return output


def _brainvision_references(header_path: Path) -> tuple[Path, Path]:
    data_file = None
    marker_file = None
    for line in header_path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("DataFile="):
            data_file = line.split("=", 1)[1].strip()
        elif line.startswith("MarkerFile="):
            marker_file = line.split("=", 1)[1].strip()
    if not data_file or not marker_file:
        raise RuntimeError(f"BrainVision references missing in {header_path}")
    return header_path.parent / data_file, header_path.parent / marker_file


def _validate_brainvision_continuity(marker_path: Path) -> None:
    """Permit only an optional initial New Segment; ignore synchronization triggers."""
    segment_positions: list[int] = []
    for line in marker_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        fields = line.split("=", 1)[1].split(",")
        if fields and fields[0].strip().lower() == "new segment":
            if len(fields) < 3:
                raise NumericalFailure("malformed BrainVision New Segment marker")
            segment_positions.append(int(fields[2]))
    if len(segment_positions) > 1 or (segment_positions and segment_positions[0] != 1):
        raise NumericalFailure("post-onset BrainVision segment break")


def _schedule_from_mask(mask: np.ndarray, dataset: str, participant: str) -> np.ndarray:
    schedule = np.empty((3, 200, 2, 10), dtype=np.uint8)
    for regime_index, regime in enumerate(REGIMES):
        for replicate in range(200):
            arm_a, arm_b = draw_pair(mask.tolist(), dataset, participant, regime, replicate)
            schedule[regime_index, replicate, 0] = arm_a
            schedule[regime_index, replicate, 1] = arm_b
    return schedule


def _replication_markers(dataset_root: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows = read_csv(REPLICATION_CENSUS)
    if len(rows) != 60 or not all(true_value(row["structurally_compatible"]) for row in rows):
        raise RuntimeError("replication structural census does not contain 60 compatible recordings")
    marker_rows: list[np.ndarray] = []
    participant_ids: list[str] = []
    schedules: list[np.ndarray] = []
    for row in rows:
        participant = row["participant_id"]
        stem = f"{participant}_ses-session1_task-eyesclosed_eeg"
        header_path = dataset_root / participant / "ses-session1" / "eeg" / f"{stem}.vhdr"
        if not header_path.is_file():
            raise FileNotFoundError(header_path)
        data_path, marker_path = _brainvision_references(header_path)
        if not data_path.is_file() or not marker_path.is_file():
            raise FileNotFoundError(f"BrainVision companion file missing for {participant}")
        if data_path.stat().st_size != int(row["annex_signal_bytes"]):
            raise RuntimeError(f"replication raw byte-count mismatch: {participant}")
        if file_digest(data_path, "md5") != row["annex_digest"]:
            raise RuntimeError(f"replication raw MD5 mismatch: {participant}")
        _validate_brainvision_continuity(marker_path)
        support_volts, sampling_rate = _load_required_support(header_path, "brainvision")
        if not np.isclose(sampling_rate, float(row["sampling_hz"])):
            raise RuntimeError(f"replication sampling-rate mismatch: {participant}")
        mask = time_domain_qc_mask(support_volts * 1e6, sampling_rate)
        try:
            participant_schedule = _schedule_from_mask(mask, "ds004148", participant)
        except (AssertionError, IndexError, ValueError):
            # Any failed required draw excludes the participant; partial schedules
            # and repair draws are forbidden.
            continue
        participant_ids.append(participant)
        schedules.append(participant_schedule)
        marker_rows.append(participant_markers(support_volts, sampling_rate, participant_schedule))
    if len(marker_rows) < 2:
        raise NumericalFailure("fewer than two replication participants passed QC and schedule construction")
    return np.asarray(marker_rows), participant_ids, np.asarray(schedules, dtype=np.uint8)


def _rank_summary(
    analysis: dict[str, object],
    regime_indices: Iterable[tuple[str, int]],
) -> dict[str, object]:
    point = analysis["point"]
    requested = list(regime_indices)
    return {
        "participant_rank_reproducibility": {
            name: float(point["back_transformed_rho"][index]) for name, index in requested
        },
        "mean_fisher_z": {name: float(point["mean_z"][index]) for name, index in requested},
        "delta_z_F10_minus_C": float(point["delta_z"]),
        "percentile_95_interval": [float(value) for value in analysis["interval"]],
        "monte_carlo_standard_error": float(point["mcse"]),
        "bootstrap_index_sha256": analysis["bootstrap_index_sha256"],
    }


def execute_registered_analysis(
    primary_root: Path,
    replication_root: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Execute only after the caller has passed the explicit production barrier."""
    registered = validate_registered_inputs()
    primary_rows = registered["primary_rows"]
    schedule = registered["schedule"]
    participant_ids = registered["participant_ids"]

    primary_markers = _primary_markers(primary_root, primary_rows, schedule)
    primary_bootstrap = generate_bootstrap_indices(531, "primary_ds005385")
    primary_rank = analyze_rank_reproducibility(primary_markers, primary_bootstrap)
    displacement = absolute_displacement_summary(primary_markers, primary_bootstrap)

    sensitivity_ids = [row["participant_id"] for row in registered["sensitivity_rows"]]
    sensitivity_schedule, selected_ids, selected_indices = subset_registered_schedule(
        schedule, participant_ids, sensitivity_ids
    )
    if selected_ids != sensitivity_ids or not np.array_equal(sensitivity_schedule, schedule[selected_indices]):
        raise RuntimeError("complete-support schedule was not a direct registered-schedule subset")
    sensitivity_markers = primary_markers[selected_indices][:, [0, 2], :, :]
    sensitivity_bootstrap = generate_bootstrap_indices(516, "complete_support_ds005385")
    sensitivity = analyze_primary_contrast(sensitivity_markers, sensitivity_bootstrap)

    replication_markers, replication_ids, replication_schedule = _replication_markers(replication_root)
    replication_bootstrap = generate_bootstrap_indices(len(replication_ids), "replication_ds004148")
    replication = analyze_rank_reproducibility(replication_markers, replication_bootstrap)

    primary_sign = int(np.sign(float(primary_rank["point"]["delta_z"])))
    replication_sign = int(np.sign(float(replication["point"]["delta_z"])))
    if primary_sign == 0 or replication_sign == 0:
        concordance = "indeterminate_zero_point_estimate"
    elif primary_sign == replication_sign:
        concordance = "descriptive_directional_concordance"
    else:
        concordance = "descriptive_directional_discordance"

    summary = {
        "registration": {"osf_id": "nh4d8", "date": "2026-08-30"},
        "PRIMARY": {
            "dataset": "ds005385_v1.0.3",
            "participant_n": 531,
            **_rank_summary(primary_rank, (("C", 0), ("F10", 2))),
        },
        "PRESPECIFIED_DESCRIPTIVE_SECONDARY": {
            "F5": {
                "role": "ordered_descriptive_middle_point",
                "participant_rank_reproducibility": float(
                    primary_rank["point"]["back_transformed_rho"][1]
                ),
                "mean_fisher_z": float(primary_rank["point"]["mean_z"][1]),
            },
            "absolute_displacement": {
                "participant_n": 531,
                "matched_differences_per_participant": displacement[
                    "matched_differences_per_participant"
                ],
                "cohort_mean": float(displacement["estimate"]),
                "percentile_95_interval": [float(value) for value in displacement["interval"]],
                "bootstrap_index_sha256": primary_rank["bootstrap_index_sha256"],
            },
        },
        "PRESPECIFIED_SENSITIVITY": {
            "dataset": "ds005385_v1.0.3",
            "participant_n": 516,
            "schedule_source": "direct_subset_of_registered_N531_schedule",
            **_rank_summary(sensitivity, (("C", 0), ("F10", 1))),
        },
        "DIRECTIONAL_REPLICATION": {
            "dataset": "ds004148_v1.0.0_session1_eyesclosed",
            "participant_n_after_time_domain_qc": len(replication_ids),
            "participant_ids": replication_ids,
            "schedule_shape": list(replication_schedule.shape),
            "pooling_with_primary": False,
            "directional_interpretation": concordance,
            **_rank_summary(replication, (("C", 0), ("F5", 1), ("F10", 2))),
        },
        "uncertainty": {
            "participant_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "interval": "two_sided_95_percentile_numpy_method_linear",
            "mcse_reported_separately": True,
        },
    }
    arrays = {
        "primary_participant_ids": np.asarray(participant_ids),
        "primary_markers": primary_markers,
        "primary_draw_rho": primary_rank["point"]["draw_rho"],
        "primary_draw_fisher_z": primary_rank["point"]["draw_z"],
        "primary_draw_delta_z": primary_rank["point"]["draw_differences"],
        "primary_bootstrap_delta_z": primary_rank["bootstrap_distribution"],
        "secondary_participant_displacement": displacement["participant_displacement"],
        "secondary_bootstrap_displacement": displacement["bootstrap_distribution"],
        "sensitivity_participant_ids": np.asarray(selected_ids),
        "sensitivity_schedule": sensitivity_schedule,
        "sensitivity_markers_C_F10": sensitivity_markers,
        "sensitivity_draw_rho_C_F10": sensitivity["point"]["draw_rho"],
        "sensitivity_draw_fisher_z_C_F10": sensitivity["point"]["draw_z"],
        "sensitivity_draw_delta_z": sensitivity["point"]["draw_differences"],
        "sensitivity_bootstrap_delta_z": sensitivity["bootstrap_distribution"],
        "replication_participant_ids": np.asarray(replication_ids),
        "replication_schedule": replication_schedule,
        "replication_markers": replication_markers,
        "replication_draw_rho": replication["point"]["draw_rho"],
        "replication_draw_fisher_z": replication["point"]["draw_z"],
        "replication_draw_delta_z": replication["point"]["draw_differences"],
        "replication_bootstrap_delta_z": replication["bootstrap_distribution"],
    }
    return summary, arrays


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly authorize the registered real-EEG production analysis",
    )
    parser.add_argument("--primary-root", type=Path, help="local root of OpenNeuro ds005385 v1.0.3")
    parser.add_argument("--replication-root", type=Path, help="local root of OpenNeuro ds004148 v1.0.0")
    parser.add_argument("--output", type=Path, help="new JSON path for the registered summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registered = validate_registered_inputs()
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "validation_only_no_scientific_outcome",
                    "primary_n": len(registered["primary_rows"]),
                    "complete_support_n": len(registered["sensitivity_rows"]),
                    "schedule_shape": list(registered["schedule"].shape),
                    "schedule_sha256": registered["schedule_sha256"],
                    "production_flag_executed": False,
                },
                indent=2,
            )
        )
        return 0
    if args.primary_root is None or args.replication_root is None or args.output is None:
        raise SystemExit("--run requires --primary-root, --replication-root, and --output")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite an existing result file: {args.output}")
    artifact_path = args.output.with_name("scientific_results.npz")
    if artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing result file: {artifact_path}")
    results, arrays = execute_registered_analysis(args.primary_root, args.replication_root)
    write_npz_atomic(artifact_path, arrays)
    results["result_artifact"] = {
        "path": artifact_path.name,
        "bytes": artifact_path.stat().st_size,
        "sha256": file_digest(artifact_path),
    }
    write_json_atomic(args.output, results)
    print(f"Registered analysis completed; results written to {args.output} and {artifact_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
