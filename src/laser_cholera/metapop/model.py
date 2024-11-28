from datetime import datetime

import click
import numpy as np
import pandas as pd
from laser_core.laserframe import LaserFrame
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from laser_core.propertyset import PropertySet
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

# from laser_cholera import PropagatePopulation
from laser_cholera.metapop import Births
from laser_cholera.metapop import Environmental
from laser_cholera.metapop import Infected
from laser_cholera.metapop import Lambda_
from laser_cholera.metapop import Population
from laser_cholera.metapop import Psi
from laser_cholera.metapop import Recovered
from laser_cholera.metapop import Susceptibles
from laser_cholera.metapop import Transmission
from laser_cholera.metapop import Vaccinated
from laser_cholera.metapop import Wash
from laser_cholera.metapop import get_parameters
from laser_cholera.metapop import scenario
from laser_cholera.utils import calc_distances


class Model:
    def __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str = "Cholera Metapop"):
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tinit}: Creating the {name} model…")
        self.scenario = scenario
        self.params = parameters
        self.name = name

        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        click.echo(f"Initializing the {name} model with {len(scenario)} patches…")

        # https://gilesjohnr.github.io/MOSAIC-docs/model-description.html

        # setup the LaserFrame for patches (inputs and reporting)
        npatches = len(scenario)
        self.patches = LaserFrame(npatches)
        _istart, _iend = self.patches.add(npatches)
        # self.patches.add_vector_property("population", length=parameters.nticks, dtype=np.uint32, default=np.uint32(0))
        self.patches.add_scalar_property("initpop", dtype=np.uint32, default=np.uint32(0))
        self.patches.initpop[:] = scenario.population.values
        self.patches.add_vector_property("network", length=npatches, dtype=np.float32, default=np.float32(0.0))
        distances = calc_distances(scenario.latitude.values, scenario.longitude.values, parameters.verbose)
        self.patches.network[:, :] = row_normalizer(
            gravity(scenario.population.values, distances, parameters.k, parameters.a, parameters.b, parameters.c), parameters.max_frac
        )

        self.patches.add_vector_property("W", length=parameters.nticks + 1, dtype=np.float32, default=np.float32(0.0))
        self.patches.add_vector_property("beta_hum", length=parameters.nticks + 1, dtype=np.float32, default=np.float32(0.0))
        self.patches.add_vector_property("beta_env", length=parameters.nticks + 1, dtype=np.float32, default=np.float32(0.0))
        self.patches.add_vector_property("LAMBDA", length=parameters.nticks + 1, dtype=np.float32, default=np.float32(0.0))
        self.patches.add_vector_property("PSI", length=parameters.nticks + 1, dtype=np.float32, default=np.float32(0.0))
        self.patches.add_vector_property("nu", length=parameters.nticks + 1, dtype=np.float32, default=np.float32(0.0))

        self.patches.add_scalar_property("birthrate", dtype=np.float32, default=np.float32(0.0))
        Warning("TODO - initialize birth rate from JG data")
        self.patches.add_scalar_property("mortrate", dtype=np.float32, default=np.float32(0.0))

        # setup the LaserFrame for agents/population (states and dynamics)
        self.population = LaserFrame(npatches)

        # S, I, R, V (vaccinated), W (environmental)
        self.population.add_vector_property("S", length=parameters.nticks + 1, dtype=np.uint32, default=0)
        self.population.add_vector_property("I", length=parameters.nticks + 1, dtype=np.uint32, default=0)
        self.population.add_vector_property("R", length=parameters.nticks + 1, dtype=np.uint32, default=0)
        self.population.add_vector_property("V", length=parameters.nticks + 1, dtype=np.uint32, default=0)
        self.population.add_vector_property("N", length=parameters.nticks + 1, dtype=np.uint32, default=0)

        # initialize the "agent" states

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
                self.phases.append(instance)

        # TODO - check for 0, 1, or 2+ Births components
        # if 0 and there are components with an on_birth function, warn
        # if 1, add the component instances with an on_birth() function to the Births component
        # if 2+, raise an exception

        births = list(filter(lambda object: isinstance(object, Births), self.instances))
        if len(births) == 0:
            for instance in self.instances:
                if "on_birth" in dir(instance):
                    click.echo(
                        f"Warning: Component {type(instance).__name__} has an on_birth function but there is no Births component with which to register it."
                    )
        elif len(births) == 1:
            births = births[0]  # get the single Births component
            births.initializers = [instance for instance in self.instances if "on_birth" in dir(instance)]
        else:
            raise RuntimeError(f"Error: Multiple Births components found in {self.name} model.")

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
        print(f"Completed the {self.name} model at {self.tfinish}…")

        if self.params.verbose:
            metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width+1}} {sum_columns.sum():13,} microseconds")

        return

    def visualize(self, pdf: bool = True) -> None:
        """
        Visualize each compoonent instances either by displaying plots or saving them to a PDF file.

        Parameters:

            pdf (bool): If True, save the plots to a PDF file. If False, display the plots interactively. Default is True.

        Returns:

            None
        """

        if not pdf:
            for instance in [self, *self.instances]:
                if hasattr(instance, "plot"):
                    for _plot in instance.plot():
                        plt.show()
                else:
                    click.echo(f"Warning: {type(instance).__name__} does not have a plot method.")

        else:
            click.echo("Generating PDF output…")
            pdf_filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf:
                for instance in [self, *self.instances]:
                    if hasattr(instance, "plot"):
                        for _plot in instance.plot():
                            pdf.savefig()
                            plt.close()
                    else:
                        click.echo(f"Warning: {type(instance).__name__} does not have a plot method.")

            click.echo(f"PDF output saved to '{pdf_filename}'.")

        return

    def plot(self, fig: Figure = None):
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
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--no-viz", is_flag=True, default=False, help="Suppress displaying visualizations")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
@click.option("--output", default=None, help="Output file for results")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--param", "-p", multiple=True, help="Additional parameter overrides (param:value or param=value)")
def run(**kwargs):
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

    parameters = get_parameters(kwargs)
    # scenario = get_scenario(parameters, parameters["verbose"])
    model = Model(scenario, parameters)

    # infection dynamics come _before_ incubation dynamics so newly set itimers
    # don't immediately expire
    model.components = [
        #        PropagatePopulation,
        #        Births,
        Lambda_,
        Wash,
        Psi,
        Susceptibles,
        Vaccinated,
        Infected,
        Environmental,
        Recovered,
        Population,
        # NonDiseaseDeaths,
        # Susceptibility,
        # MaternalAntibodies,
        # RoutineImmunization,
        # Infection,
        # Incubation,
        Transmission,
    ]

    # seed_infections_randomly(model, ninfections=100)
    # ipatch = model.patches.population[0, :].argmax()
    ipatch = model.patches.initpop.argmax()
    seed_infections_in_patch(model, ipatch=ipatch, ninfections=100)

    model.run()

    if not parameters.no_viz:
        model.visualize(pdf=parameters.pdf)

    return


def seed_infections_in_patch(model, ipatch, ninfections):
    model.population.S[0, ipatch] -= ninfections
    model.population.I[0, ipatch] = ninfections
    model.patches.W[0, ipatch] = model.params.zeta * ninfections

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, no_viz=False, pdf=False)
