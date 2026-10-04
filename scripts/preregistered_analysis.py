#!/usr/bin/env python3
"""Pure functions for the preregistered temporal-sampling analysis.

This module is safe to import: it performs no file access and no analysis at
import time. Scientific outcomes can only be produced by an explicit caller.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
from scipy import signal
from scipy.stats import rankdata


REGIMES = ("C", "F5", "F10")
REGIME_INDEX = {name: index for index, name in enumerate(REGIMES)}
REQUIRED_CHANNELS = ("O1", "Oz", "O2", "TP9", "TP10")
ROI_CHANNELS = REQUIRED_CHANNELS[:3]
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_NAMESPACE = "OSF-nh4d8|participant_bootstrap|{analysis_label}"
BOOTSTRAP_LABELS = (
    "primary_ds005385",
    "complete_support_ds005385",
    "replication_ds004148",
)
EXPECTED_BOOTSTRAP_SEEDS = {
    "primary_ds005385": 15066513651140358874,
    "complete_support_ds005385": 6846413471351983530,
    "replication_ds004148": 7728558626112585725,
}


class NumericalFailure(RuntimeError):
    """A prespecified quantity is not estimable without an unregistered repair."""


def derive_bootstrap_seed(analysis_label: str) -> int:
    text = BOOTSTRAP_NAMESPACE.format(analysis_label=analysis_label)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def generate_bootstrap_indices(
    participant_count: int,
    analysis_label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> np.ndarray:
    if participant_count < 2:
        raise ValueError("participant_count must be at least two")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    seed = derive_bootstrap_seed(analysis_label)
    if analysis_label in EXPECTED_BOOTSTRAP_SEEDS:
        expected = EXPECTED_BOOTSTRAP_SEEDS[analysis_label]
        if seed != expected:
            raise RuntimeError(f"bootstrap seed mismatch for {analysis_label}: {seed} != {expected}")
    rng = np.random.default_rng(seed)
    return rng.integers(0, participant_count, size=(resamples, participant_count), dtype=np.int64)


def bootstrap_index_sha256(indices: np.ndarray) -> str:
    array = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def resample_participants(array: np.ndarray, participant_indices: np.ndarray) -> np.ndarray:
    """Apply one participant index vector to every trailing analysis dimension."""
    values = np.asarray(array)
    indices = np.asarray(participant_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("one bootstrap resample must be a one-dimensional index vector")
    return values[indices, ...]


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise NumericalFailure("percentile interval requires a finite one-dimensional distribution")
    lower, upper = np.percentile(values, [2.5, 97.5], method="linear")
    return float(lower), float(upper)


def elementary_periodogram(samples: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Four-second, native-rate, periodic-Hann, one-sided density periodogram."""
    x = np.asarray(samples, dtype=float)
    n = int(round(4.0 * float(sampling_rate)))
    if x.shape[-1] != n:
        raise ValueError("exactly four seconds are required; partial cells are forbidden")
    if not np.isfinite(x).all():
        raise NumericalFailure("nonfinite sample in elementary periodogram")
    detrended = signal.detrend(x, type="linear", axis=-1)
    taper = signal.windows.hann(n, sym=False)
    transformed = np.fft.rfft(detrended * taper, n=n, axis=-1)
    density = (np.abs(transformed) ** 2) / (float(sampling_rate) * np.sum(taper**2))
    density[..., 1:-1] *= 2.0
    return density


def compute_cell_periodograms(required_channel_data: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 42 elementary PSDs after the linked-mastoid reference.

    Input channel order must be O1, Oz, O2, TP9, TP10 and must contain the
    exact 168-second [8,176) support.
    """
    data = np.asarray(required_channel_data, dtype=float)
    samples_per_cell = int(round(4.0 * float(sampling_rate)))
    expected_samples = 42 * samples_per_cell
    if data.shape != (5, expected_samples):
        raise ValueError(f"required data shape is (5, {expected_samples}), observed {data.shape}")
    if not np.isfinite(data).all():
        raise NumericalFailure("nonfinite sample in required-channel support")
    mastoid_mean = (data[3] + data[4]) / 2.0
    posterior = data[:3] - mastoid_mean
    psds = []
    for channel in range(3):
        channel_psds = []
        for cell in range(42):
            start = cell * samples_per_cell
            stop = start + samples_per_cell
            channel_psds.append(elementary_periodogram(posterior[channel, start:stop], sampling_rate))
        psds.append(channel_psds)
    frequency = np.fft.rfftfreq(samples_per_cell, 1.0 / float(sampling_rate))
    if not np.isclose(frequency[1] - frequency[0], 0.25):
        raise RuntimeError("registered 0.25-Hz frequency grid was not obtained")
    return frequency, np.asarray(psds)


def relative_alpha_from_cell_psds(
    cell_psds: np.ndarray,
    frequency: np.ndarray,
    cell_indices: Sequence[int],
) -> float:
    psds = np.asarray(cell_psds, dtype=float)
    freq = np.asarray(frequency, dtype=float)
    indices = np.asarray(cell_indices, dtype=np.int64)
    if psds.ndim != 3 or psds.shape[:2] != (3, 42):
        raise ValueError("cell PSD array must have shape (3, 42, frequency)")
    if indices.shape != (10,) or len(np.unique(indices)) != 10:
        raise ValueError("each arm must contain ten unique cell indices")
    if np.any(indices < 0) or np.any(indices > 41):
        raise ValueError("cell indices must lie in [0, 41]")
    if psds.shape[-1] != freq.size:
        raise ValueError("PSD and frequency dimensions disagree")
    # Registered aggregation order: cells within channel, then ROI channels.
    aggregate = psds[:, indices, :].mean(axis=1).mean(axis=0)
    alpha = (freq >= 8.0) & (freq <= 13.0)
    denominator = (freq >= 1.0) & (freq <= 30.0)
    numerator_power = np.trapezoid(aggregate[alpha], freq[alpha])
    denominator_power = np.trapezoid(aggregate[denominator], freq[denominator])
    value = numerator_power / denominator_power
    if not np.isfinite(value):
        raise NumericalFailure("nonfinite posterior relative alpha")
    return float(value)


def participant_markers(
    required_channel_data: np.ndarray,
    sampling_rate: float,
    participant_schedule: np.ndarray,
    regime_indices: Sequence[int] | None = None,
) -> np.ndarray:
    schedule = np.asarray(participant_schedule)
    if schedule.shape != (3, 200, 2, 10):
        raise ValueError(f"participant schedule must have shape (3, 200, 2, 10), observed {schedule.shape}")
    selected_regimes = tuple(range(3)) if regime_indices is None else tuple(regime_indices)
    if not selected_regimes or any(index not in range(3) for index in selected_regimes):
        raise ValueError("regime_indices must select from C/F5/F10")
    frequency, cell_psds = compute_cell_periodograms(required_channel_data, sampling_rate)
    markers = np.empty((len(selected_regimes), 200, 2), dtype=float)
    for output_regime, regime in enumerate(selected_regimes):
        for draw in range(200):
            for arm in range(2):
                markers[output_regime, draw, arm] = relative_alpha_from_cell_psds(
                    cell_psds, frequency, schedule[regime, draw, arm]
                )
    return markers


def _validate_marker_array(markers: np.ndarray, expected_regimes: int) -> np.ndarray:
    values = np.asarray(markers, dtype=float)
    if values.ndim != 4 or values.shape[1] != expected_regimes or values.shape[3] != 2:
        raise ValueError(
            f"marker array must have shape (participants, {expected_regimes}, draws, 2)"
        )
    if values.shape[0] < 2 or values.shape[2] < 1:
        raise ValueError("at least two participants and one draw are required")
    return values


def spearman_midrank_columns(x: np.ndarray, y: np.ndarray, label: str = "Spearman") -> np.ndarray:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Spearman inputs must be matching participant-by-draw matrices")
    finite_columns = np.isfinite(left).all(axis=0) & np.isfinite(right).all(axis=0)
    ranks_left = rankdata(left, axis=0, method="average")
    ranks_right = rankdata(right, axis=0, method="average")
    centered_left = ranks_left - ranks_left.mean(axis=0, keepdims=True)
    centered_right = ranks_right - ranks_right.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(centered_left**2, axis=0) * np.sum(centered_right**2, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.sum(centered_left * centered_right, axis=0) / denominator
    valid = finite_columns & np.isfinite(rho)
    if not valid.all():
        raise NumericalFailure(f"{label}: {int(valid.sum())}/{valid.size} draw correlations estimable")
    return rho


def fisher_z_checked(rho: np.ndarray, label: str) -> np.ndarray:
    correlations = np.asarray(rho, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = np.arctanh(correlations)
    valid = np.isfinite(transformed)
    if not valid.all():
        raise NumericalFailure(f"{label}: {int(valid.sum())}/{valid.size} Fisher-z values estimable")
    return transformed


def point_statistics(markers: np.ndarray) -> dict[str, np.ndarray | float]:
    values = _validate_marker_array(markers, 3)
    draw_rho = np.empty((3, values.shape[2]), dtype=float)
    draw_z = np.empty_like(draw_rho)
    for regime_index, regime in enumerate(REGIMES):
        draw_rho[regime_index] = spearman_midrank_columns(
            values[:, regime_index, :, 0],
            values[:, regime_index, :, 1],
            label=f"{regime} Spearman",
        )
        draw_z[regime_index] = fisher_z_checked(draw_rho[regime_index], f"{regime} Fisher-z")
    mean_z = draw_z.mean(axis=1)
    back_transformed_rho = np.tanh(mean_z)
    draw_differences = draw_z[REGIME_INDEX["F10"]] - draw_z[REGIME_INDEX["C"]]
    delta_z = float(draw_differences.mean())
    mcse = mcse_from_draw_differences(draw_differences)
    return {
        "draw_rho": draw_rho,
        "draw_z": draw_z,
        "mean_z": mean_z,
        "back_transformed_rho": back_transformed_rho,
        "draw_differences": draw_differences,
        "delta_z": delta_z,
        "mcse": mcse,
    }


def primary_contrast_statistics(markers: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute only C, F10, and their preregistered contrast.

    This narrower function is used for complete-support sensitivity and the
    directional replication so that those paths do not create an F5 result.
    """
    values = _validate_marker_array(markers, 2)
    draw_rho = np.empty((2, values.shape[2]), dtype=float)
    draw_z = np.empty_like(draw_rho)
    for output_index, regime in enumerate(("C", "F10")):
        draw_rho[output_index] = spearman_midrank_columns(
            values[:, output_index, :, 0],
            values[:, output_index, :, 1],
            label=f"{regime} Spearman",
        )
        draw_z[output_index] = fisher_z_checked(draw_rho[output_index], f"{regime} Fisher-z")
    mean_z = draw_z.mean(axis=1)
    draw_differences = draw_z[1] - draw_z[0]
    return {
        "draw_rho": draw_rho,
        "draw_z": draw_z,
        "mean_z": mean_z,
        "back_transformed_rho": np.tanh(mean_z),
        "draw_differences": draw_differences,
        "delta_z": float(draw_differences.mean()),
        "mcse": mcse_from_draw_differences(draw_differences),
    }


def mcse_from_draw_differences(draw_differences: np.ndarray) -> float:
    differences = np.asarray(draw_differences, dtype=float)
    if differences.ndim != 1 or differences.size < 2 or not np.isfinite(differences).all():
        raise NumericalFailure("MCSE requires at least two finite draw differences")
    return float(np.std(differences, ddof=1) / math.sqrt(differences.size))


def primary_bootstrap_distribution(
    markers: np.ndarray,
    bootstrap_indices: np.ndarray,
    batch_size: int = 25,
) -> np.ndarray:
    values = _validate_marker_array(markers, 2)
    indices = np.asarray(bootstrap_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != values.shape[0]:
        raise ValueError("bootstrap matrix width must equal participant count")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = np.empty(indices.shape[0], dtype=float)
    draws = values.shape[2]
    for start in range(0, indices.shape[0], batch_size):
        stop = min(start + batch_size, indices.shape[0])
        # Advanced indexing yields batch x participant x regime x draw x arm.
        sample = values[indices[start:stop], ...]
        batch = stop - start

        def correlations(regime_index: int, regime: str) -> np.ndarray:
            left = sample[:, :, regime_index, :, 0].transpose(1, 0, 2).reshape(
                values.shape[0], batch * draws
            )
            right = sample[:, :, regime_index, :, 1].transpose(1, 0, 2).reshape(
                values.shape[0], batch * draws
            )
            rho = spearman_midrank_columns(left, right, f"bootstrap {regime}")
            return fisher_z_checked(rho, f"bootstrap {regime} Fisher-z").reshape(batch, draws)

        c_z = correlations(0, "C")
        f10_z = correlations(1, "F10")
        output[start:stop] = np.mean(f10_z - c_z, axis=1)
    return output


def logit_strict(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & (x > 0.0) & (x < 1.0)
    if not valid.all():
        raise NumericalFailure(
            f"logit requires values strictly inside (0,1): {int(valid.sum())}/{valid.size} valid"
        )
    return np.log(x) - np.log1p(-x)


def participant_absolute_displacement(markers: np.ndarray) -> np.ndarray:
    values = _validate_marker_array(markers, 3)
    c = logit_strict(values[:, REGIME_INDEX["C"], :, :])
    f10 = logit_strict(values[:, REGIME_INDEX["F10"], :, :])
    # Two arms x 200 matched draw labels = 400 differences per participant.
    return (f10 - c).mean(axis=(1, 2))


def absolute_displacement_summary(
    markers: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray | float | tuple[float, float] | int]:
    displacement = participant_absolute_displacement(markers)
    indices = np.asarray(bootstrap_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != displacement.size:
        raise ValueError("bootstrap matrix width must equal participant count")
    bootstrap_distribution = displacement[indices].mean(axis=1)
    return {
        "participant_displacement": displacement,
        "estimate": float(displacement.mean()),
        "bootstrap_distribution": bootstrap_distribution,
        "interval": percentile_interval(bootstrap_distribution),
        "matched_differences_per_participant": int(markers.shape[2] * markers.shape[3]),
    }


def analyze_rank_reproducibility(
    markers: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    point = point_statistics(markers)
    bootstrap_distribution = primary_bootstrap_distribution(
        np.asarray(markers)[:, [REGIME_INDEX["C"], REGIME_INDEX["F10"]], :, :],
        bootstrap_indices,
    )
    return {
        "point": point,
        "bootstrap_distribution": bootstrap_distribution,
        "interval": percentile_interval(bootstrap_distribution),
        "bootstrap_index_sha256": bootstrap_index_sha256(bootstrap_indices),
    }


def analyze_primary_contrast(
    markers: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    point = primary_contrast_statistics(markers)
    bootstrap_distribution = primary_bootstrap_distribution(markers, bootstrap_indices)
    return {
        "point": point,
        "bootstrap_distribution": bootstrap_distribution,
        "interval": percentile_interval(bootstrap_distribution),
        "bootstrap_index_sha256": bootstrap_index_sha256(bootstrap_indices),
    }


def subset_registered_schedule(
    schedule: np.ndarray,
    participant_ids: Sequence[str],
    subset_ids: Iterable[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    array = np.asarray(schedule)
    ids = list(participant_ids)
    wanted = set(subset_ids)
    if array.shape[0] != len(ids):
        raise ValueError("schedule participant dimension and identifier count disagree")
    indices = np.asarray([index for index, participant in enumerate(ids) if participant in wanted], dtype=np.int64)
    selected_ids = [ids[index] for index in indices]
    if set(selected_ids) != wanted or len(selected_ids) != len(wanted):
        raise ValueError("subset identifiers are missing, duplicated, or outside the registered roster")
    return array[indices], selected_ids, indices
