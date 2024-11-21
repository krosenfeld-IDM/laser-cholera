import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Transmission:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "population"), "Transmission: model needs to have a 'population' attribute."
        assert hasattr(model.population, "S"), "Transmission: model needs to have a 'S' (susceptible) attribute."
        assert hasattr(model.population, "I"), "Transmission: model needs to have a 'I' (infectious) attribute."
        assert hasattr(model.population, "R"), "Transmission: model needs to have a 'R' (recovered) attribute."
        assert hasattr(model.population, "V"), "Transmission: model needs to have a 'V' (vaccinated) attribute."
        assert hasattr(model, "patches"), "Transmission: model needs to have a 'patches' attribute."
        assert hasattr(model, "prng"), "Transmission: model needs to have a 'prng' pseudo-random number generator."

        model.patches.add_vector_property("incidence", length=model.params.nticks, dtype=np.uint32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        # new infections = Poisson(beta * S * I / N)

        susceptible = model.population.S[tick, :]
        infectious = model.population.I[tick, :]
        recovered = model.population.R[tick, :]
        vaccinated = model.population.V[tick, :]
        incidence = model.patches.incidence[tick, :]

        N = susceptible + infectious + recovered + vaccinated
        beta = model.params.beta
        incidence[:] = model.prng.poisson(beta * susceptible * infectious / N).astype(model.population.I.dtype)
        susceptible -= incidence
        infectious += incidence

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.population[:, 0].argmax()
        plt.plot(self.model.patches.incidence[ipatch, :], color="red", label="Cases")
        plt.xlabel("Tick")
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(self.model.patches.incidence[:, ipatch], color="purple", label="Incidence")
        plt.legend()

        yield

        return
