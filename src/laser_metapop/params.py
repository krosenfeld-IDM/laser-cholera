# NOTE: This file is *verbatim* from ``laser_cholera.metapop`` – the validation
# logic is disease-specific and will therefore almost certainly need to be
# customised by the consumer.  What *is* generic, however, is the machinery for
# loading/serialising parameters (JSON / HDF5) and normalising data-types.  The
# class lives here so that other pathogens can reuse those helpers without
# taking an explicit dependency on the cholera package.

from __future__ import annotations

# The full implementation is copied without modification to keep the diff
# small.  It compiles stand-alone because it never imports anything from
# `laser_cholera`.  Consumers are encouraged to fork it or provide their own
# schema-aware subclass of :class:`PropertySetEx`.

import gzip
import io
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PseEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D401 – we are overriding the *default* hook
        if isinstance(obj, PropertySet):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return f"{obj:%Y-%m-%d}"
        return super().default(obj)


class PropertySetEx(PropertySet):
    """Thin convenience wrapper that provides a nicer ``__str__``."""

    def __str__(self) -> str:  # noqa: D401 – human readable
        return json.dumps(self.to_dict(), cls=PseEncoder, indent=4)


# ---------------------------------------------------------------------------
# Loader / saver utilities (JSON & HDF5)
# ---------------------------------------------------------------------------


def get_parameters(
    paramsource: Optional[Union[str, Path, dict]] = None,
    *,
    do_validation: bool = True,
    overrides: Optional[dict] = None,
) -> PropertySetEx:
    """Hydrate a :class:`PropertySetEx` from *paramsource*.

    The function is format-agnostic.  If *paramsource* is ``None`` we will fall
    back to a file named ``default_parameters.json`` that is expected to live
    alongside this module.
    """

    fn_map = {
        (".json",): _load_json,
        (".json", ".gz"): _load_json_gz,
        (".h5",): _load_hdf5,
        (".hdf",): _load_hdf5,
        (".hdf5",): _load_hdf5,
        (".h5", ".gz"): _load_hdf5_gz,
        (".hdf", ".gz"): _load_hdf5_gz,
        (".hdf5", ".gz"): _load_hdf5_gz,
    }

    if isinstance(paramsource, (str, Path, type(None))):
        file_path = (
            Path(paramsource)
            if paramsource is not None
            else Path(__file__).parent / "data" / "default_parameters.json"
        )
        suffixes = tuple(map(str.lower, file_path.suffixes))
        load_fn = fn_map[suffixes]

        logger.info("Loading parameters from %s …", file_path)
        params = load_fn(file_path)

    elif isinstance(paramsource, dict):
        params = _dict_to_propertysetex(paramsource)

    else:
        raise ValueError(f"Invalid parameter source type: {type(paramsource)}")

    # ------------------------------------------------------------------
    # Apply CLI / runtime overrides
    # ------------------------------------------------------------------

    if overrides:
        params += overrides
        logger.info("Updated/overrode file parameters with overrides:")
        for k, v in overrides.items():
            logger.info("  %s: %s", k, v)

    if do_validation:
        validate_parameters(params)

    # Ensure a couple of convenience flags exist so downstream code does not
    # need to guard every attribute access.
    params.setdefault("visualize", False)
    params.setdefault("pdf", False)
    params.setdefault("quiet", True)

    return params


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _load_json(filename: Union[str, Path]) -> PropertySetEx:
    with Path(filename).open("r") as fh:
        return _dict_to_propertysetex(json.load(fh))


def _load_json_gz(filename: Union[str, Path]) -> PropertySetEx:
    with gzip.open(filename, "rb") as fh:
        with io.BytesIO(fh.read()) as buf:
            return _dict_to_propertysetex(json.load(buf))


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _load_hdf5(filename: Union[str, Path]) -> PropertySetEx:
    with h5.File(filename, "r") as h5file:
        return _load_hdf5_from_handle(h5file)


def _load_hdf5_gz(filename: Union[str, Path]) -> PropertySetEx:
    with gzip.open(filename, "rb") as gz_file:
        with io.BytesIO(gz_file.read()) as file_obj:
            with h5.File(file_obj, "r") as h5file:
                return _load_hdf5_from_handle(h5file)


def _load_hdf5_from_handle(h5file) -> PropertySetEx:  # noqa: ANN001 – h5py does not ship proper stubs
    """Read the *very* IDM-flavoured HDF5 format into a property-set."""

    ps = PropertySetEx()

    # Calendar ----------------------------------------------------------------
    epoch = datetime(year=1970, month=1, day=1)
    ps.date_start = epoch + timedelta(days=int(h5file["date_start"][()][0]))
    ps.date_stop = epoch + timedelta(days=int(h5file["date_stop"][()][0]))
    ps.nticks = (ps.date_stop - ps.date_start).days + 1  # include *stop*

    # Scalars ------------------------------------------------------------------
    for scalar in h5file.keys():
        if scalar in {"date_start", "date_stop"}:
            continue
        if isinstance(h5file[scalar], h5.Dataset) and h5file[scalar].shape == (1,):
            ps[scalar] = h5file[scalar][()][0]

    # Anything that is *not* a scalar is copied verbatim.  Consumers can tidy
    # up the shapes later in their validation hook.
    for key, dataset in h5file.items():
        if key in {"date_start", "date_stop"}:
            continue
        if not (isinstance(dataset, h5.Dataset) and dataset.shape == (1,)):
            ps[key] = dataset[()]

    return ps


# ---------------------------------------------------------------------------
# Generic normalisation helpers
# ---------------------------------------------------------------------------


def _dict_to_propertysetex(parameters: dict) -> PropertySetEx:
    # Note: The function purposefully *does not* interpret the schema – the goal
    # is only to convert obvious JSON string encodings into Python types that
    # are pleasant to work with further down-stream.

    ps = PropertySetEx(parameters)

    # Convert common ISO-8601 strings to *naive* datetimes.  If a user requires
    # timezone-aware handling they can inject the correct ``datetime`` objects
    # themselves.
    for field in ("date_start", "date_stop"):
        if field in ps and isinstance(ps[field], str):
            ps[field] = datetime.strptime(ps[field], "%Y-%m-%d")  # noqa: DTZ007 – the object remains naïve.

    if "date_start" in ps and "date_stop" in ps:
        ps.nticks = (ps.date_stop - ps.date_start).days + 1

    return ps


# ---------------------------------------------------------------------------
# Validation – *placeholder*
# ---------------------------------------------------------------------------


def validate_parameters(params: PropertySetEx) -> None:  # noqa: D401
    """Generic stub – no-op for now.

    The original cholera implementation performs extensive schema validation.
    That is intentionally *not* reproduced here.  New pathogens are expected to
    supply their own domain-specific checks by either monkey-patching this
    function or wrapping :func:`get_parameters`.
    """

    return None


# ---------------------------------------------------------------------------
# Convenience holder so disease-specific models can *still* attach a small
# component that visualises the parameters with existing tooling.
# ---------------------------------------------------------------------------


class Parameters:
    """Lightweight component that exposes the current parameter set."""

    def __init__(self, model):
        self.model = model

    # The other methods intentionally do nothing; they exist solely because the
    # component *contract* requires them.

    def check(self):  # noqa: D401
        assert hasattr(self.model, "params"), "model missing 'params' attribute"

    def __call__(self, _model, _tick):  # noqa: D401, ANN001
        pass

    # The plotting utility is kept verbatim from the cholera code-base because
    # it is generally useful when working with compartmental models.  No
    # disease-specific assumptions are embedded (only generic categories).

    def plot(self, fig: Figure | None = None):  # pragma: no cover
        import numpy as np

        categories = [
            "S_j_initial",
            "E_j_initial",
            "I_j_initial",
            "R_j_initial",
            "V1_j_initial",
            "V2_j_initial",
        ]
        data = [getattr(self.model.params, cat) for cat in categories if hasattr(self.model.params, cat)]

        _fig = (
            plt.figure(figsize=(12, 9), dpi=128, num="Initial Populations by Category")
            if fig is None
            else fig
        )

        x = np.arange(len(getattr(self.model.params, "location_name", data[0])))
        bottom = np.zeros_like(x, dtype=float)

        for category, values in zip(categories, data, strict=False):
            plt.bar(x, values, bottom=bottom, label=category)
            bottom += values

        plt.xticks(x, getattr(self.model.params, "location_name", x), rotation=45, ha="right")
        plt.xlabel("Location Name")
        plt.ylabel("Population")
        plt.legend()

        yield "Initial Populations by Category"


__all__ = [
    "PropertySetEx",
    "get_parameters",
    "Parameters",
]
