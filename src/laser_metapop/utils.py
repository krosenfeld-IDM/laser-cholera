"""Utilities that are *independent* of any particular pathogen.

These helper functions are copied verbatim from
``laser_cholera.metapop.utils`` because they implement generic mathematical
building-blocks:

* Gravity-based mobility matrix construction from lat/long coordinates.
* Fourier seasonality helpers.
* Convenience routine that converts CLI ``str`` overrides into correctly typed
  Python objects so they can be merged into a :class:`laser_core.propertyset.PropertySet`.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial

import numpy as np
from laser_core.migration import distance

# ---------------------------------------------------------------------------
# Seasonality – simple Fourier expansion helpers
# ---------------------------------------------------------------------------


def fourier_series_double(t, beta0, a1, b1, a2, b2, p):
    """Evaluate a 2-harmonic Fourier series.

    ``t`` is broadcast against the shape of the coefficient vectors so that the
    caller can either pass scalar or *per-patch* arrays.
    """

    return (
        beta0
        + a1[None, :] * np.cos(2 * np.pi * t / p)[:, None]
        + b1[None, :] * np.sin(2 * np.pi * t / p)[:, None]
        + a2[None, :] * np.cos(4 * np.pi * t / p)[:, None]
        + b2[None, :] * np.sin(4 * np.pi * t / p)[:, None]
    )


def get_daily_seasonality(params):
    """Pre-compute daily transmission multiplier for an *entire* year."""

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


# ---------------------------------------------------------------------------
# Gravity model helper
# ---------------------------------------------------------------------------


def get_pi_from_lat_long(params):
    """Compute the *gravity* mobility matrix between all patches."""

    d = distance(params.latitude, params.longitude, params.latitude, params.longitude)
    omega = params.mobility_omega
    gamma = params.mobility_gamma

    N = (
        params.S_j_initial
        + params.E_j_initial
        + params.I_j_initial
        + params.R_j_initial
        + params.V1_j_initial
        + params.V2_j_initial
    )

    x = np.zeros_like(d, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if i == j:
                continue
            x[i, j] = np.power(N[j], omega) * np.power(d[i, j], -gamma)

    m_hat = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        row_sum = np.sum(x[i, :])
        for j in range(x.shape[1]):
            if i == j:
                continue
            m_hat[i, j] = x[i, j] / row_sum

    return m_hat


# ---------------------------------------------------------------------------
# CLI override helper – typed conversion
# ---------------------------------------------------------------------------


def override_helper(overrides) -> dict:  # noqa: ANN001 – mapping is free-form
    def bool_from_string(value):
        return str(value).lower() in {"true", "1", "yes", "y", "t", "on", "enabled"}

    mapping = {
        "seed": int,
        "date_start": partial(datetime.strptime, format="%Y-%m-%d"),
        "date_stop": partial(datetime.strptime, format="%Y-%m-%d"),
        # All remaining entries map to ``None`` which signals *no automatic cast*.
        # They are nevertheless enumerated here so that users have a convenient
        # reference of available keys.
        "location_name": None,
        "S_j_initial": None,
        "E_j_initial": None,
        "I_j_initial": None,
        "R_j_initial": None,
        "V1_j_initial": None,
        "V2_j_initial": None,
        "b_jt": None,
        "d_jt": None,
        "nu_1_jt": None,
        "nu_2_jt": None,
        "phi_1": float,
        "phi_2": float,
        "omega_1": float,
        "omega_2": float,
        "iota": float,
        "gamma_1": float,
        "gamma_2": float,
        "epsilon": float,
        "mu_jt": None,
        "rho": float,
        "simga": float,
        "longitude": None,
        "latitude": None,
        "mobility_omega": float,
        "mobility_gamma": float,
        "tau_i": None,
        "beta_j0_hum": None,
        "a_1_j": None,
        "b_1_j": None,
        "a_2_j": None,
        "b_2_j": None,
        "p": int,
        "alpha_1": float,
        "alpha_2": float,
        "beta_j0_env": None,
        "theta_j": None,
        "psi_jt": None,
        "zeta_1": float,
        "zeta_2": float,
        "kappa": float,
        "decay_days_short": float,
        "decay_days_long": float,
        "decay_shape_1": float,
        "decay_shape_2": float,
        "return": None,
        "visualize": bool_from_string,
        "pdf": bool_from_string,
        "hdf5_output": bool_from_string,
        "compress": bool_from_string,
        "quiet": bool_from_string,
    }

    typed: dict = {}
    for key, value in overrides.items():
        fn = mapping.get(key)
        if fn is not None:
            typed[key] = fn(value) if callable(fn) else value
        else:
            typed[key] = value  # passthrough for unknown keys

    return typed


__all__ = [
    "fourier_series_double",
    "get_daily_seasonality",
    "get_pi_from_lat_long",
    "override_helper",
]
