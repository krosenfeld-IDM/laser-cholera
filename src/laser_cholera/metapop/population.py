import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Population:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "population"), "Population: model needs to have a 'population' attribute."
        assert hasattr(model.population, "S"), "Population: model population needs to have a 'S' (susceptible) attribute."
        assert hasattr(model.population, "I"), "Population: model population needs to have a 'I' (infectious) attribute."
        assert hasattr(model.population, "R"), "Population: model population needs to have a 'R' (recovered) attribute."
        assert hasattr(model.population, "V"), "Population: model population needs to have a 'v' (vaccinated) attribute."
        assert hasattr(model.population, "N"), "Population: model population needs to have a 'N' (current population) attribute."
        assert hasattr(model, "patches"), "Population: model needs to have a 'patches' attribute."
        assert hasattr(model.patches, "initpop"), "Population: model patches needs to have a 'initpop' (initial population) attribute."

        model.population.N[0] = model.patches.initpop

        return

    def __call__(self, model, tick: int) -> None:
        Sprime = model.population.S[tick + 1]
        Vprime = model.population.V[tick + 1]
        Iprime = model.population.I[tick + 1]
        Rprime = model.population.R[tick + 1]
        Nprime = model.population.N[tick + 1]

        Nprime[:] = Sprime + Vprime + Iprime + Rprime

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Population in Patch {ipatch}")
        plt.plot(self.model.population.N[:, ipatch], color="purple", label="Total")

        plt.xlabel("Tick")

        yield
        return
