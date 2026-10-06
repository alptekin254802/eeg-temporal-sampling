"""Check summary comparisons at the boundary between numbers and metadata."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verification_tolerances import assert_summary_close


class TestSummaryComparison(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "n": 526, "seed": 7932202023600012997,
            "ids": ["sub-001", "sub-002"], "hash": "abc123",
            "changed": False, "delta": .29901113129106244,
            "interval": [.2414123367227983, .3495496512767094],
            "rho": {"C": .9326222092010044, "F10": .9623786103087777},
        }

    def test_adjacent_float_values_pass(self):
        actual = copy.deepcopy(self.expected)
        actual["delta"] = np.nextafter(actual["delta"], np.inf)
        actual["interval"][1] = np.nextafter(actual["interval"][1], -np.inf)
        actual["rho"]["F10"] += 5e-13
        assert_summary_close(actual, self.expected)

    def test_changed_numerical_results_fail_with_their_location(self):
        for field in ["delta", "interval", "rho"]:
            actual = copy.deepcopy(self.expected)
            if field == "delta":
                actual[field] += 1e-8
            elif field == "interval":
                actual[field][0] += 1e-8
            else:
                actual[field]["C"] += 1e-8
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, field):
                assert_summary_close(actual, self.expected)

    def test_identifiers_hashes_counts_and_large_seeds_remain_exact(self):
        changes = {"n": 527, "seed": self.expected["seed"] + 1,
                   "ids": ["sub-002", "sub-001"], "hash": "abc124", "changed": True}
        for field, value in changes.items():
            actual = copy.deepcopy(self.expected); actual[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, field):
                assert_summary_close(actual, self.expected)

    def test_nonfinite_values_fail_even_when_both_sides_match(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaisesRegex(AssertionError, "nonfinite"):
                assert_summary_close({"value": value}, {"value": value})

    def test_missing_extra_and_wrongly_shaped_fields_fail(self):
        for actual in [
            {"value": []}, {"value": [.2, .3]}, {"value": (.2,)},
            {"value": [.2], "extra": 1}, {},
        ]:
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                assert_summary_close(actual, {"value": [.2]})

    def test_metadata_types_are_not_coerced(self):
        for actual, expected in [(1, True), (True, 1), (526.0, 526), (0, 0.0)]:
            with self.subTest(actual=actual, expected=expected), self.assertRaises(AssertionError):
                assert_summary_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
