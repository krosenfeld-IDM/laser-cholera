import unittest

import numpy as np

from laser_cholera.metapop.utils import calc_affine_normalization
from laser_cholera.metapop.utils import fourier_series_double


class TestMetapopUtils(unittest.TestCase):
    def test_fourier_series_double(self):
        # Test with known values
        t = np.arange(16)
        beta0 = 1.0
        a1 = 0.5
        b1 = 0.5
        a2 = 0.2
        b2 = 0.2
        p = 10

        expected_result = (
            beta0
            + a1 * np.cos(2 * np.pi * t / p)
            + b1 * np.sin(2 * np.pi * t / p)
            + a2 * np.cos(4 * np.pi * t / p)
            + b2 * np.sin(4 * np.pi * t / p)
        )

        result = fourier_series_double(t, beta0, a1, b1, a2, b2, p)

        assert np.allclose(result, expected_result)

    def test_calc_affine_normalization(self):
        x = np.array([1, 2, 3, np.nan])
        expected_result = np.array([-1, 0, 1, np.nan])
        result = calc_affine_normalization(x)

        inotnan = ~np.isnan(expected_result)
        assert np.allclose(result[inotnan], expected_result[inotnan])

    def test_calc_affine_normalization_all_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        result = calc_affine_normalization(x)
        assert np.all(np.isnan(result))

    def test_calc_affine_normalization_zero_division(self):
        x = np.array([1, 1, 1])
        result = calc_affine_normalization(x)
        assert np.all(np.isnan(result))


if __name__ == "__main__":
    unittest.main()
