"""laser_metapop
=================

This package provides a *generic* meta-population simulation engine that
implements the mechanics of LASER-based compartmental models (event loop,
metrics collection, and optional visualisation utilities), but **without any
assumptions that are specific to a particular pathogen**.  It is a light
refactor of the implementation that originally lived in
``laser_cholera.metapop`` so that other disease-specific models (e.g. measles)
can reuse the common infrastructure while supplying their own components and
parameter schema.

Only the genuinely generic artefacts are surfaced here – everything that
encodes cholera biology or nomenclature remains in the original
``laser_cholera`` package.  Down-stream models are expected to compose their
own component list and supply a compatible :class:`laser_core.propertyset.PropertySet`
containing the parameters that drive the simulation.
"""

from __future__ import annotations

# Public re-exports ----------------------------------------------------------

from .model import Model  # noqa: F401
from .params import PropertySetEx, get_parameters, Parameters  # noqa: F401
from .utils import (  # noqa: F401
    fourier_series_double,
    get_daily_seasonality,
    get_pi_from_lat_long,
    override_helper,
)

__all__ = [
    "Model",
    "PropertySetEx",
    "get_parameters",
    "Parameters",
    "fourier_series_double",
    "get_daily_seasonality",
    "get_pi_from_lat_long",
    "override_helper",
]
