import numpy as np


def calc_affine_normalization(x):
    if np.all(np.isnan(x)):
        return x

    m = x[~np.isnan(x)].mean()
    min_x = x[~np.isnan(x)].min()

    if np.abs(m - min_x) < np.finfo((m - min_x).dtype).eps:
        return np.full(x.shape, np.nan)

    scaled = (x - m) / (m - min_x)

    return scaled


def fourier_series_double(t, beta0, a1, b1, a2, b2, p):
    """
    Fourier series with 2 harmonics
    :param t: time
    :param beta0: constant term
    :param a1: first harmonic amplitude (cosine)
    :param b1: first harmonic amplitude (sine)
    :param a2: second harmonic amplitude (cosine)
    :param b2: second harmonic amplitude (sine)
    :param p: period
    :return: Fourier series value at time t
    """
    return (
        beta0
        + a1 * np.cos(2 * np.pi * t / p)
        + b1 * np.sin(2 * np.pi * t / p)
        + a2 * np.cos(4 * np.pi * t / p)
        + b2 * np.sin(4 * np.pi * t / p)
    )
