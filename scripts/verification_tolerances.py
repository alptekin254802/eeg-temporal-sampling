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
