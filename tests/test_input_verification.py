"""Test how input checks handle missing files, units, ordering and tolerances."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce_inputs import parse_mask, rebuild_schedule, require_files, verify_replication_schedule
from verification_tolerances import assert_cell_powers_close
from verify_structural_census import event_diagnostics, historical_event_class, pointer_info


class TestPowerVerification(unittest.TestCase):
    def test_roundoff_across_power_scales_passes(self):
        expected = np.array([0., 1e-30, 4.64e-13, 3.54e-11, 1.98e-9])
        assert_cell_powers_close(expected*(1+1e-13), expected)

    def test_shared_scale_error_is_caught_even_when_ratio_is_unchanged(self):
        expected = np.array([[4.64e-13, 3.54e-11], [1e-10, 1.98e-9]])
        mutated = expected*1.0005
        self.assertLess(np.max(np.abs(mutated-expected)), 1e-12)
        np.testing.assert_allclose(mutated[:, 0]/mutated[:, 1], expected[:, 0]/expected[:, 1], rtol=0, atol=1e-16)
        with self.assertRaises(AssertionError):
            assert_cell_powers_close(mutated, expected)

    def test_small_power_relative_fault_is_caught(self):
        with self.assertRaises(AssertionError):
            assert_cell_powers_close([6.96e-13], [4.64e-13])

    def test_zero_and_near_zero_do_not_have_an_absolute_escape(self):
        for actual, expected in [([1e-30], [0.]), ([0.], [1e-30]), ([2e-30], [1e-30])]:
            with self.subTest(actual=actual, expected=expected), self.assertRaises(AssertionError):
                assert_cell_powers_close(actual, expected)

    def test_nonfinite_negative_and_shape_faults_are_rejected(self):
        for actual, expected in [([np.nan], [np.nan]), ([np.inf], [np.inf]), ([-1e-12], [-1e-12]), ([1.], [[1.]])]:
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                assert_cell_powers_close(actual, expected)


class TestStructuralDiagnostics(unittest.TestCase):
    def test_type_only_replay_and_strict_diagnostic_are_separate(self):
        events = [dict(type="boundary", onset="1"), dict(type="boundary", onset="25041")]
        self.assertEqual(historical_event_class(events), "ONLY-INITIAL-BOUNDARY")
        diagnostic = event_diagnostics(events, {"onset": {"Units": "ms"}}, 1000)
        self.assertFalse(diagnostic["single_initial_boundary_diagnostic"])
        np.testing.assert_allclose(diagnostic["boundary_onsets_seconds"], [.001, 25.041], atol=1e-14, rtol=0)
        np.testing.assert_allclose(diagnostic["boundary_onsets_in_support_seconds"], [25.041], atol=1e-14, rtol=0)

    def test_initial_millisecond_and_late_second_are_distinguished(self):
        event = [dict(type="boundary", onset="1")]
        self.assertTrue(event_diagnostics(event, {"onset": {"Units": "ms"}}, 1000)["single_initial_boundary_diagnostic"])
        self.assertFalse(event_diagnostics(event, {"onset": {"Units": "s"}}, 1000)["single_initial_boundary_diagnostic"])

    def test_unknown_units_are_not_guessed(self):
        result = event_diagnostics([dict(type="boundary", onset="1")], {}, 1000)
        self.assertIsNone(result["single_initial_boundary_diagnostic"])
        self.assertIsNone(result["boundary_onsets_seconds"])

    def test_signal_payload_is_not_read_as_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"payload.edf"
            path.write_bytes(b"x"*8193)
            with self.assertRaisesRegex(ValueError, "small annex pointer"):
                pointer_info(path)


class TestScheduleAndScopeVerification(unittest.TestCase):
    def test_preflight_lists_all_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/name for name in ("sub-026.edf", "sub-552.edf")]
            with self.assertRaises(FileNotFoundError) as caught:
                require_files(paths, "536-record QC")
            message = str(caught.exception)
            self.assertIn("2 required", message)
            self.assertTrue(all(str(path) in message for path in paths))
            self.assertIn("No files were skipped", message)

    def test_malformed_masks_are_rejected(self):
        for value in ("1"*41, "1"*43, "1"*41+"x"):
            with self.assertRaises(ValueError):
                parse_mask(value)

    def test_sub552_exclusion_is_not_repaired_to_valid_pairs(self):
        mask = parse_mask("111111111011111101111111111111111111111111")
        schedule, counts = rebuild_schedule(mask, "ds005385", "sub-552")
        self.assertIsNone(schedule)
        self.assertEqual(counts, [147, 200, 200])

    def test_replication_comparison_rejects_order_value_and_roster_count_faults(self):
        rows = [dict(participant_id=pid, eligible_mask="1"*42, C_valid_draws="200",
                     F5_valid_draws="200", F10_valid_draws="200", qc_schedule_eligible="True", exclusion_reason="")
                for pid in ("sub-01", "sub-02")]
        one = np.zeros((3, 200, 2, 10), dtype=np.uint8)
        accepted = np.stack([one, one])
        with patch("reproduce_inputs.rebuild_schedule", return_value=(one, [200, 200, 200])):
            verify_replication_schedule(rows, ["sub-01", "sub-02"], accepted)
            with self.assertRaisesRegex(ValueError, "IDs/order"):
                verify_replication_schedule(rows, ["sub-02", "sub-01"], accepted)
            bad = accepted.copy(); bad[1, 0, 0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, "differs from accepted"):
                verify_replication_schedule(rows, ["sub-01", "sub-02"], bad)
            rows[1]["C_valid_draws"] = "199"
            with self.assertRaisesRegex(ValueError, "valid-draw counts"):
                verify_replication_schedule(rows, ["sub-01", "sub-02"], accepted)


if __name__ == "__main__":
    unittest.main()
