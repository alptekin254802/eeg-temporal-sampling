#!/usr/bin/env python3
"""Synthetic-only tests for the post-registration production implementation."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from preregistered_analysis import (  # noqa: E402
    EXPECTED_BOOTSTRAP_SEEDS,
    NumericalFailure,
    absolute_displacement_summary,
    bootstrap_index_sha256,
    derive_bootstrap_seed,
    compute_cell_periodograms,
    elementary_periodogram,
    fisher_z_checked,
    generate_bootstrap_indices,
    logit_strict,
    mcse_from_draw_differences,
    participant_absolute_displacement,
    percentile_interval,
    point_statistics,
    primary_bootstrap_distribution,
    relative_alpha_from_cell_psds,
    resample_participants,
    spearman_midrank_columns,
    subset_registered_schedule,
)
from run_preregistered_analysis import build_parser, write_npz_atomic  # noqa: E402


def synthetic_markers(participants: int = 12, draws: int = 7) -> np.ndarray:
    participant = np.arange(participants, dtype=float)[:, None]
    draw = np.arange(draws, dtype=float)[None, :]
    values = np.empty((participants, 3, draws, 2), dtype=float)
    for regime in range(3):
        values[:, regime, :, 0] = 0.1 + 0.006 * participant + 0.0002 * draw + 0.002 * regime
        permutation = np.roll(np.arange(participants), regime + 1)
        values[:, regime, :, 1] = (
            0.12
            + 0.005 * permutation[:, None]
            + 0.0003 * ((draw + regime) % draws)
            + 0.001 * regime
        )
    return values


class TestPostRegistrationConventions(unittest.TestCase):
    def test_production_periodogram_matches_scipy_reference_at_native_rates(self):
        rng = np.random.default_rng(44)
        for sampling_rate in (500.0, 1000.0):
            samples = rng.standard_normal(int(4 * sampling_rate))
            observed = elementary_periodogram(samples, sampling_rate)
            _, expected = signal.periodogram(
                samples,
                fs=sampling_rate,
                window=signal.windows.hann(samples.size, sym=False),
                detrend="linear",
                return_onesided=True,
                scaling="density",
                nfft=samples.size,
            )
            self.assertTrue(np.allclose(observed, expected, rtol=1e-13, atol=1e-15))

    def test_linked_mastoid_reference_and_aggregation_match_registered_reference(self):
        from tests.test_psd_equivalence import marker as registered_marker

        sampling_rate = 500.0
        samples_per_cell = int(4 * sampling_rate)
        rng = np.random.default_rng(77)
        posterior = rng.standard_normal((3, 42, samples_per_cell))
        mastoids = rng.standard_normal((2, 42, samples_per_cell))
        mastoid_mean = mastoids.mean(axis=0)
        required = np.concatenate(
            (posterior + mastoid_mean[None, :, :], mastoids), axis=0
        ).reshape(5, 42 * samples_per_cell)
        frequency, cell_psds = compute_cell_periodograms(required, sampling_rate)
        indices = np.arange(10)
        observed = relative_alpha_from_cell_psds(cell_psds, frequency, indices)
        expected, _ = registered_marker(posterior[:, indices, :], fs=sampling_rate)
        self.assertAlmostEqual(observed, expected, places=14)

    def test_bootstrap_seed_derivation(self):
        observed = {label: derive_bootstrap_seed(label) for label in EXPECTED_BOOTSTRAP_SEEDS}
        self.assertEqual(observed, EXPECTED_BOOTSTRAP_SEEDS)

    def test_bootstrap_indices_are_deterministic(self):
        first = generate_bootstrap_indices(19, "primary_ds005385", resamples=31)
        second = generate_bootstrap_indices(19, "primary_ds005385", resamples=31)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(bootstrap_index_sha256(first), bootstrap_index_sha256(second))

    def test_one_participant_vector_applies_to_all_trailing_dimensions(self):
        original = np.arange(5 * 3 * 4 * 2).reshape(5, 3, 4, 2)
        indices = np.asarray([4, 1, 1, 0, 3])
        observed = resample_participants(original, indices)
        self.assertTrue(np.array_equal(observed, original[indices, ...]))
        for row, source in enumerate(indices):
            self.assertTrue(np.array_equal(observed[row], original[source]))

    def test_batched_bootstrap_matches_literal_resample_formula(self):
        rng = np.random.default_rng(123)
        markers = rng.uniform(0.1, 0.9, size=(30, 3, 7, 2))
        indices = generate_bootstrap_indices(30, "primary_ds005385", resamples=9)
        observed = primary_bootstrap_distribution(markers[:, [0, 2], :, :], indices, batch_size=4)
        expected = []
        for participant_index in indices:
            sample = markers[participant_index]
            c = np.asarray([
                spearmanr(sample[:, 0, draw, 0], sample[:, 0, draw, 1]).statistic
                for draw in range(sample.shape[2])
            ])
            f10 = np.asarray([
                spearmanr(sample[:, 2, draw, 0], sample[:, 2, draw, 1]).statistic
                for draw in range(sample.shape[2])
            ])
            expected.append(np.mean(np.arctanh(f10) - np.arctanh(c)))
        self.assertTrue(np.allclose(observed, np.asarray(expected)))

    def test_spearman_fisher_mean_and_back_transform(self):
        markers = synthetic_markers()
        observed = point_statistics(markers)
        for regime in range(3):
            expected_rho = np.asarray([
                spearmanr(markers[:, regime, draw, 0], markers[:, regime, draw, 1]).statistic
                for draw in range(markers.shape[2])
            ])
            expected_z = np.arctanh(expected_rho)
            self.assertTrue(np.allclose(observed["draw_rho"][regime], expected_rho))
            self.assertTrue(np.allclose(observed["draw_z"][regime], expected_z))
            self.assertAlmostEqual(observed["back_transformed_rho"][regime], np.tanh(expected_z.mean()))

    def test_primary_delta_z_formula(self):
        observed = point_statistics(synthetic_markers())
        expected = np.mean(observed["draw_z"][2] - observed["draw_z"][0])
        self.assertAlmostEqual(observed["delta_z"], expected)

    def test_mcse_uses_sample_standard_deviation(self):
        differences = np.asarray([-0.4, -0.1, 0.2, 0.8, 1.0])
        expected = np.std(differences, ddof=1) / np.sqrt(differences.size)
        self.assertAlmostEqual(mcse_from_draw_differences(differences), expected)
        self.assertNotAlmostEqual(
            mcse_from_draw_differences(differences),
            np.std(differences, ddof=0) / np.sqrt(differences.size),
        )

    def test_percentile_interval_is_linear(self):
        distribution = np.asarray([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
        expected = np.percentile(distribution, [2.5, 97.5], method="linear")
        self.assertTrue(np.array_equal(np.asarray(percentile_interval(distribution)), expected))

    def test_absolute_displacement_uses_400_matched_differences(self):
        participants = 4
        markers = np.full((participants, 3, 200, 2), 0.25)
        c = 0.20 + np.arange(participants)[:, None, None] * 0.01
        f10 = 0.30 + np.arange(participants)[:, None, None] * 0.01
        markers[:, 0] = np.broadcast_to(c, (participants, 200, 2))
        markers[:, 2] = np.broadcast_to(f10, (participants, 200, 2))
        expected = (
            np.log(f10) - np.log1p(-f10) - np.log(c) + np.log1p(-c)
        ).reshape(participants)
        observed = participant_absolute_displacement(markers)
        self.assertTrue(np.allclose(observed, expected))
        indices = np.tile(np.arange(participants), (5, 1))
        summary = absolute_displacement_summary(markers, indices)
        self.assertEqual(summary["matched_differences_per_participant"], 400)

    def test_logit_has_no_epsilon_clipping(self):
        with self.assertRaises(NumericalFailure):
            logit_strict(np.asarray([0.0, 0.5]))
        with self.assertRaises(NumericalFailure):
            logit_strict(np.asarray([0.5, 1.0]))
        value = np.nextafter(0.0, 1.0)
        self.assertTrue(np.isfinite(logit_strict(np.asarray([value]))[0]))

    def test_complete_support_is_direct_schedule_subset(self):
        schedule = np.arange(6 * 3 * 2 * 2 * 10, dtype=np.int64).reshape(6, 3, 2, 2, 10)
        ids = [f"sub-{index}" for index in range(6)]
        subset, selected_ids, selected_indices = subset_registered_schedule(
            schedule, ids, ["sub-1", "sub-4"]
        )
        self.assertEqual(selected_ids, ["sub-1", "sub-4"])
        self.assertTrue(np.array_equal(selected_indices, np.asarray([1, 4])))
        self.assertTrue(np.array_equal(subset, schedule[[1, 4]]))
        self.assertFalse(np.shares_memory(subset, schedule))

    def test_numerical_failures_are_not_repaired(self):
        constant = np.ones((8, 3))
        with self.assertRaises(NumericalFailure):
            spearman_midrank_columns(constant, constant)
        with self.assertRaises(NumericalFailure):
            fisher_z_checked(np.asarray([1.0]), "perfect correlation")
        bad = synthetic_markers()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(NumericalFailure):
            point_statistics(bad)

    def test_default_cli_is_validation_only(self):
        completed = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "run_preregistered_analysis.py")],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('"mode": "validation_only_no_scientific_outcome"', completed.stdout)
        self.assertIn('"production_flag_executed": false', completed.stdout)

    def test_production_barrier_defaults_to_disabled(self):
        self.assertFalse(build_parser().parse_args([]).run)
        self.assertTrue(build_parser().parse_args(["--run"]).run)

    def test_lossless_result_artifact_writer(self):
        arrays = {
            "markers": np.arange(24, dtype=float).reshape(2, 3, 2, 2),
            "participant_ids": np.asarray(["sub-001", "sub-002"]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scientific_results.npz"
            write_npz_atomic(path, arrays)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".npz.tmp").exists())
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(arrays))
                for key, expected in arrays.items():
                    self.assertTrue(np.array_equal(archive[key], expected))


if __name__ == "__main__":
    unittest.main()
