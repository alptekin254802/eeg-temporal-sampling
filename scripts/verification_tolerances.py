"""Compare cell band powers using a relative tolerance."""
import numpy as np

CELL_POWER_RTOL = 1e-10
# Powers are integrated densities in V^2. Use zero absolute tolerance so small
# powers still receive the relative check and expected zeros remain zero.
CELL_POWER_ATOL_V2 = 0.0


def assert_cell_powers_close(actual, expected):
    """Check that V^2 powers are finite, nonnegative and match the reference."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise AssertionError(f"cell-power shape differs: {actual.shape} != {expected.shape}")
    for label, values in (("actual", actual), ("expected", expected)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise AssertionError(f"{label} cell powers must be finite and nonnegative V^2")
    np.testing.assert_allclose(
        actual, expected, rtol=CELL_POWER_RTOL, atol=CELL_POWER_ATOL_V2,
        equal_nan=False, err_msg="raw cell powers differ (V^2; relative tolerance 1e-10, zero absolute tolerance)",
    )

def assert_summary_close(actual, expected, path="summary"):
    """Allow roundoff in float results while keeping metadata exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: summary fields differ")
        for key in expected:
            assert_summary_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length or type differs")
        for i, (value, reference) in enumerate(zip(actual, expected)):
            assert_summary_close(value, reference, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, np.floating)):
            raise AssertionError(f"{path}: expected a floating-point result")
        if not np.isfinite(actual) or not np.isfinite(expected):
            raise AssertionError(f"{path}: nonfinite result")
        np.testing.assert_allclose(
            actual, expected, atol=1e-12, rtol=0, equal_nan=False,
            err_msg=f"{path}: numerical result differs",
        )
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{path}: exact metadata differs")
