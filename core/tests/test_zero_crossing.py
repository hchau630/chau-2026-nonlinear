import numpy as np

from niarb.zero_crossing import find_n_crossings


def test_find_n_crossings():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )
    expected = np.array(
        [
            [np.nan, np.nan, np.nan, 1.5, 1.5, 0.5, 0.5, 1.5],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.5, np.nan],
        ]
    )
    out = find_n_crossings(x, y, n=2)
    np.testing.assert_allclose(out, expected, equal_nan=True)
