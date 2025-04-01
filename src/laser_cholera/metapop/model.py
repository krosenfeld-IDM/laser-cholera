from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from laser_cholera.metapop import Analyzer
from laser_cholera.metapop import Census
from laser_cholera.metapop import DerivedValues
from laser_cholera.metapop import Environmental
from laser_cholera.metapop import EnvToHuman
from laser_cholera.metapop import EnvToHumanVax
from laser_cholera.metapop import Exposed
from laser_cholera.metapop import HumanToHuman
from laser_cholera.metapop import HumanToHumanVax
from laser_cholera.metapop import Infectious
from laser_cholera.metapop import Parameters
from laser_cholera.metapop import Recorder
from laser_cholera.metapop import Recovered
from laser_cholera.metapop import Susceptible
from laser_cholera.metapop import Vaccinated
from laser_cholera.metapop import get_parameters
from laser_cholera.metapop import scenario


class Model:
    def __init__(self, parameters: PropertySet, name: str = "Cholera Metapop"):
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tinit}: Creating the {name} model…")
        self.params = parameters
        self.name = name

        self.scenario = scenario

        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        click.echo(f"Initializing the {name} model with {len(parameters.location_id)} patches…")

        # https://gilesjohnr.github.io/MOSAIC-docs/model-description.html

        # setup the LaserFrame for agents/population (states and dynamics)
        # setup the LaserFrame for patches (inputs and reporting)
        npatches = len(parameters.location_id)
        self.agents = LaserFrame(npatches)
        self.patches = LaserFrame(npatches)

        return

    def verbose(self, *args, **kwargs) -> None:
        """
        Print verbose output if the verbose flag is set in the parameters.

        Args:
            *args: Positional arguments to be printed.
            **kwargs: Keyword arguments to be printed.

        Returns:
            None
        """
        if self.params.verbose:
            click.echo(*args, **kwargs)
        return

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.

        Returns:

            list: A list containing the components.
        """

        return self._components

    @components.setter
    def components(self, components: list) -> None:
        """
        Sets up the components of the model and initializes instances and phases.

        This function takes a list of component types, creates an instance of each, and adds each callable component to the phase list.
        It also registers any components with an `on_birth` function with the `Births` component.

        Args:

            components (list): A list of component classes to be initialized and integrated into the model.

        Returns:

            None
        """

        self._components = components
        self.instances = []  # instantiated instances of components
        self.phases = []  # callable phases of the model
        for component in components:
            instance = component(self, self.params.verbose)
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.verbose(f"Adding {type(instance).__name__} to the model…")
                self.phases.append(instance)

        _ = [instance.check() for instance in self.instances]

        return

    def run(self) -> None:
        """
        Execute the model for a specified number of ticks, recording the time taken for each phase.

        This method initializes the start time, iterates over the number of ticks specified in the model parameters,
        and for each tick, it executes each phase of the model while recording the time taken for each phase.

        The metrics for each tick are stored in a list. After completing all ticks, it records the finish time and,
        if verbose mode is enabled, prints a summary of the timing metrics.

        Attributes:

            tstart (datetime): The start time of the model execution.
            tfinish (datetime): The finish time of the model execution.
            metrics (list): A list of timing metrics for each tick and phase.

        Returns:

            None
        """

        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tstart}: Running the {self.name} model for {self.params.nticks} ticks…")

        self.metrics = []
        for tick in tqdm(range(self.params.nticks)):
            timing = [tick]
            for phase in self.phases:
                tstart = datetime.now(tz=None)  # noqa: DTZ005
                phase(self, tick)
                tfinish = datetime.now(tz=None)  # noqa: DTZ005
                delta = tfinish - tstart
                timing.append(delta.seconds * 1_000_000 + delta.microseconds)
            self.metrics.append(timing)

        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        print(f"Completed the {self.name} model at {self.tfinish:%Y-%m-%d %H-%M-%S}…")

        if self.params.verbose:
            metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width + 1}} {sum_columns.sum():13,} microseconds")

        return

    def visualize(self, pdf: bool = True) -> Optional[str]:
        """
        Visualize each compoonent instances either by displaying plots or saving them to a PDF file.

        Parameters:

            pdf (bool): If True, save the plots to a PDF file. If False, display the plots interactively. Default is True.

        Returns:

            None
        """

        filename = None

        _debugging = [DerivedValues]

        if not pdf:
            for instance in [self, *self.instances]:
                if (_debugging is None) or (type(instance) in _debugging):
                    if hasattr(instance, "plot"):
                        for _plot in instance.plot():
                            self.verbose(f"Plotting {type(instance).__name__}…")
                            plt.show()
                    else:
                        click.echo(f"Warning: {type(instance).__name__} does not have a plot method.")
                else:
                    self.verbose(f"Skipping {type(instance).__name__} visualization…")

        else:
            click.echo("Generating PDF output…")
            pdf_filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf:
                for instance in [self, *self.instances]:
                    if (_debugging is None) or (type(instance) in _debugging):
                        if hasattr(instance, "plot"):
                            for _plot in instance.plot():
                                self.verbose(f"Plotting {type(instance).__name__}…")
                                pdf.savefig()
                                plt.close()
                        else:
                            click.echo(f"Warning: {type(instance).__name__} does not have a plot method.")
                    else:
                        self.verbose(f"Skipping {type(instance).__name__} visualization…")

            click.echo(f"PDF output saved to '{pdf_filename}'.")
            filename = pdf_filename

        return filename

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        _fig.suptitle("Scenario Patches and Populations")
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

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(
            sum_columns,
            labels=sum_columns.index,  # [name for name in sum_columns.index],
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Update Phase Times")

        yield
        return


@click.command()
# @click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--no-viz", is_flag=True, default=False, help="Suppress displaying visualizations")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
@click.option("--outdir", default=Path.cwd(), help="Output file for results")
@click.option("--paramfile", default=None, help="JSON file with parameters")
@click.option("--param", "-p", multiple=True, help="Additional parameter overrides (param:value or param=value)")
def cli_run(**kwargs):
    """
    Run the cholera model simulation with the given parameters.

    This function initializes the model with the specified parameters, sets up the
    components of the model, seeds initial infections, runs the simulation, and
    optionally visualizes the results.

    Parameters:

        **kwargs: Arbitrary keyword arguments containing the parameters for the simulation.

            Expected keys include:

                - "verbose": (bool) Whether to print verbose output.
                - "no_viz": (bool) Whether to skip visualizations.
                - "pdf": (str) The file path to save the visualization as a PDF.

    Returns:

        None
    """

    run_model(**kwargs)

    return


def run_model(**kwargs):
    parameters = get_parameters(overrides=kwargs)
    model = Model(parameters)

    model.components = [
        Susceptible,
        Exposed,
        Recovered,
        Infectious,
        Vaccinated,
        Census,
        HumanToHuman,
        HumanToHumanVax,
        EnvToHuman,
        EnvToHumanVax,
        Environmental,
        DerivedValues,
        Analyzer,
        Recorder,
        Parameters,
    ]

    model.run()

    if not parameters.no_viz:
        model.visualize(pdf=parameters.pdf)

    return model


if __name__ == "__main__":
    ctx = click.Context(cli_run)
    # ctx.invoke(run, nticks=5 * 365, seed=20241107, verbose=True, no_viz=False, pdf=False)
    ctx.invoke(cli_run, seed=20241107, verbose=True, no_viz=False, pdf=False)
