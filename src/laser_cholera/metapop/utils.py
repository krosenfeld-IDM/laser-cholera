import numpy as np
from laser_core.migration import distance


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
        + a1[None, :] * np.cos(2 * np.pi * t / p)[:, None]
        + b1[None, :] * np.sin(2 * np.pi * t / p)[:, None]
        + a2[None, :] * np.cos(4 * np.pi * t / p)[:, None]
        + b2[None, :] * np.sin(4 * np.pi * t / p)[:, None]
    )


def get_daily_seasonality(params):
    beta_j0 = params.beta_j0_hum
    a1 = params.a_1_j
    b1 = params.b_1_j
    a2 = params.a_2_j
    b2 = params.b_2_j
    p = params.p
    t = np.arange(0, p)

    seasonality = (
        beta_j0
        * (
            1.0
            + a1[None, :] * np.cos(2 * np.pi * t / p)[:, None]
            + b1[None, :] * np.sin(2 * np.pi * t / p)[:, None]
            + a2[None, :] * np.cos(4 * np.pi * t / p)[:, None]
            + b2[None, :] * np.sin(4 * np.pi * t / p)[:, None]
        )
    ).astype(np.float32)

    return seasonality


def get_pi_from_lat_long(params):
    """
    x <- D; x[,] <- NA
    for (i in 1:length(N_orig)) {
      for (j in 1:length(N_dest)) {

        x[i,j] <- (N_dest[j]^params[k,'omega']) * (D[i,j]+0.001)^(-params[k,'gamma'])

      }
    }

    for (i in 1:length(N_orig)) {
      for (j in 1:length(N_dest)) {

        # M_hat[i,j] <- params[k,'theta'] * N_orig[i] * (x[i,j]/sum(x[i,]))
        M_hat[i,j] <- x[i,j]/sum(x[i,])

      }
    }
    """

    d = distance(params.latitude, params.longitude, params.latitude, params.longitude)
    x = np.zeros_like(d, dtype=np.float32)
    omega = params.mobility_omega
    gamma = params.mobility_gamma
    N = params.S_j_initial + params.E_j_initial + params.I_j_initial + params.R_j_initial + params.V1_j_initial + params.V2_j_initial
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if j == i:
                continue
            # gravity model uses origin and destination populations
            # we'll incorporate the destination population now
            # and the effective origin population, including tau (migrating fraction) at runtime
            x[i, j] = np.power(N[j], omega) * np.power(d[i, j], -gamma)

    m_hat = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        row_sum = np.sum(x[i, :])
        for j in range(x.shape[1]):
            if j == i:
                continue
            m_hat[i, j] = x[i, j] / row_sum

    return m_hat
