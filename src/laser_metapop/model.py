"""laser_metapop.model
=====================

Disease-agnostic simulation engine that executes a list of *components* over a
user-specified number of ticks.  Each component must obey the following
contract (identical to the one established in ``laser_cholera.metapop``):

* ``__init__(self, model)`` – register any state in :class:`laser_core.laserframe.LaserFrame` and
  keep a reference to the parent ``model`` if required.
* ``check(self)`` – runtime validation of dependencies, raising ``AssertionError``
  on mismatch.  Executed once the model has been fully composed.
* ``__call__(self, model, tick)`` – advance the component by **one** tick.  It
  will be invoked in the sequence the user provided via
  :pyattr:`Model.components`.
* ``plot(self) -> typing.Iterable[str]`` – (optional) generator that yields a
  title for every :pymod:`matplotlib` figure it creates.  The engine takes care
  of saving or showing those figures.

The implementation is a direct lift from the cholera model with all biology-
specific imports stripped out so that it can be reused by other pathogens.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Type

import pandas as pd
from laser_core.laserframe import LaserFrame
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Model:
    """Core simulation object.

    Parameters
    ----------
    parameters:
        A :class:`laser_core.propertyset.PropertySet` (or subclass) instance
        containing *all* scalar, vector and matrix data required by the model.
        No assumptions are made about its concrete schema – that is the
        responsibility of the disease-specific layer that consumes this
        package.
    name:
        Human-readable identifier that will appear in log messages and plot
        titles.
    """

    # ---------------------------------------------------------------------
    # Construction & initialisation
    # ---------------------------------------------------------------------

    def __init__(self, parameters, name: str | None = None):
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005  – strictly wall-clock timing, no TZ semantics required here
        self.name = name or "Metapopulation Model"

        logger.info("%s: Creating the %s …", self.tinit, self.name)

        self.params = parameters

        # PRNG ----------------------------------------------------------------
        seed_value = getattr(parameters, "seed", None)
        self.prng = seed_prng(seed_value or self.tinit.microsecond)

        # LaserFrames ---------------------------------------------------------
        npatches = len(parameters.location_name)
        logger.info("Initialising model with %d patches …", npatches)

        self.agents: LaserFrame = LaserFrame(npatches)  # compartment states
        self.patches: LaserFrame = LaserFrame(npatches)  # exogenous inputs / reporting helpers

        # Components are supplied later via the public attribute
        self._components: list = []
        self.instances: list = []
        self.phases: list = []

        # Optional – the disease-specific layer can overwrite this to embed
        # geometry / demographics so that :meth:`plot` has something useful to
        # draw.
        self.scenario = None

        return

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    @property
    def components(self) -> List[Type]:  # noqa: D401 (non-imperative doc-style rule)
        """Return the list of component *types* used by the model."""

        return self._components

    @components.setter
    def components(self, components: Sequence[Type]):
        """Instantiate and wire a list of component classes.

        The sequence is also the execution *order* for every tick.  Each class
        is instantiated once and may therefore hold references to shared state
        across the entire simulation.
        """

        self._components = list(components)
        self.instances = []  # concrete objects
        self.phases = []  # callables executed every tick

        for component_cls in components:
            instance = component_cls(self)
            self.instances.append(instance)

            if callable(instance):  # pragma: no branch – most components are
                logger.debug("Adding %s to the update loop …", type(instance).__name__)
                self.phases.append(instance)

        # Allow every component to validate that all its dependencies have been
        # wired correctly before the first tick starts.
        for instance in self.instances:
            if hasattr(instance, "check"):
                instance.check()

        return

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the configured components for *nticks* time-steps."""

        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        nticks = self.params.nticks
        logger.info("%s: Running %s for %d ticks …", self.tstart, self.name, nticks)

        self.metrics: list[list[int]] = []  # tick | µs per phase

        for tick in tqdm(range(nticks), desc="Running model", disable=getattr(self.params, "quiet", False)):
            timing = [tick]

            for phase in self.phases:
                t0 = datetime.now(tz=None)  # noqa: DTZ005
                phase(self, tick)
                t1 = datetime.now(tz=None)  # noqa: DTZ005

                delta = t1 - t0
                timing.append(delta.seconds * 1_000_000 + delta.microseconds)

            self.metrics.append(timing)

        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        logger.info("%s: Completed %s", self.tfinish, self.name)

        # ------------------------------------------------------------------
        # Pretty print runtime summary to the log
        # ------------------------------------------------------------------

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(p).__name__ for p in self.phases])
        sums = metrics.iloc[:, 1:].sum()

        width = max(map(len, sums.index))
        for key, value in sums.items():
            logger.info("%*s: %13, d µs", width, key, value)
        logger.info("%s", "=" * (width + 18))
        logger.info("%*s %13, d microseconds", width + 1, "Total:", sums.sum())

        return

    # ------------------------------------------------------------------
    # Visualisation helpers (optional)
    # ------------------------------------------------------------------

    def visualize(self, pdf: bool = True) -> Optional[str]:  # pragma: no cover
        """Iterate over ``plot`` generators of all components.

        Parameters
        ----------
        pdf:
            *True* → write every figure to a single PDF file and return the
            filename.  *False* → show each figure interactively and return
            ``None``.
        """

        filename: str | None = None

        # Only produce plots for instances whose type appears in this sentinel
        # list.  If the list is *None* every component is included.  Useful for
        # selective debugging without recompiling the project.
        _debugging: Optional[Sequence[type]] = None  # noqa: N806 (private sentinel)

        if not pdf:
            for instance in [self, *self.instances]:
                if (_debugging is None) or (type(instance) in _debugging):
                    if hasattr(instance, "plot"):
                        for _ in instance.plot():
                            plt.tight_layout()
                            plt.show()
                    else:
                        logger.warning("%s does not implement a plot() method", type(instance).__name__)
        else:
            logger.info("Generating PDF output …")
            filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"

            with PdfPages(filename) as pdf_pages:
                for instance in [self, *self.instances]:
                    if (_debugging is None) or (type(instance) in _debugging):
                        if hasattr(instance, "plot"):
                            for title in instance.plot():
                                plt.title(title)
                                plt.tight_layout()
                                pdf_pages.savefig()
                                plt.close()
                        else:
                            logger.warning("%s does not implement a plot() method", type(instance).__name__)

            logger.info("PDF output saved to %s", filename)

        return filename

    # ------------------------------------------------------------------
    # Model-level diagnostic plots (scenario geometry & runtime profile)
    # ------------------------------------------------------------------

    def plot(self, fig: Figure | None = None):  # pragma: no cover
        import numpy as np

        # ------------------------------------------------------------------
        # Scenario geometry & populations (if available)
        # ------------------------------------------------------------------

        if self.scenario is not None:
            _fig = plt.figure(figsize=(12, 9), dpi=128, num="Scenario Patches and Populations") if fig is None else fig

            if "geometry" in self.scenario.columns:
                ax = plt.gca()
                self.scenario.plot(ax=ax)

            scatter = plt.scatter(
                self.scenario.longitude,
                self.scenario.latitude,
                s=self.scenario.population / 100_000,
                c=self.scenario.population,
                cmap="inferno",
            )
            plt.colorbar(scatter, label="Population")

            yield "Scenario Patches and Populations"

        # ------------------------------------------------------------------
        # Runtime profile (µs spent in each phase)
        # ------------------------------------------------------------------

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(p).__name__ for p in self.phases])
        sums = metrics.iloc[:, 1:].sum()

        _fig = plt.figure(
            figsize=(12, 9),
            dpi=128,
            num=f"Update Phase Times (Total {sums.sum():,} µs)",
        ) if fig is None else fig

        plt.pie(sums, labels=sums.index, autopct="%1.1f%%", startangle=140)

        yield f"Update Phase Times (Total {sums.sum():,} µs)"

        return


__all__ = ["Model"]
