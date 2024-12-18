import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Transmission:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Transmission: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "S"), "Transmission: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(model.agents, "I"), "Transmission: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(model.agents, "R"), "Transmission: model agents needs to have a 'R' (recovered) attribute."
        assert hasattr(model.agents, "V"), "Transmission: model agents needs to have a 'V' (vaccinated) attribute."
        assert hasattr(model, "patches"), "Transmission: model needs to have a 'patches' attribute."
        assert hasattr(model, "prng"), "Transmission: model needs to have a 'prng' pseudo-random number generator."

        model.patches.add_vector_property("incidence", length=model.params.nticks, dtype=np.uint32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        # new infections = Poisson(beta * S * I / N)

        S = model.agents.S[tick, :]
        I = model.agents.I[tick, :]  # noqa: E741 ("I" is ambiguous)
        R = model.agents.R[tick, :]
        V = model.agents.V[tick, :]
        incidence = model.patches.incidence[tick, :]

        N = S + I + R + V
        beta = model.params.beta
        incidence[:] = model.prng.poisson(beta * S * I / N).astype(model.agents.I.dtype)
        Sprime = model.agents.S[tick + 1, :]
        Iprime = model.agents.I[tick + 1, :]
        Sprime[:] = S - incidence
        Iprime[:] = I + incidence

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Patch {ipatch} - Transmission")
        ax1 = plt.gca()
        ax1.plot(self.model.agents.I[:, ipatch], color="red", label="Cases")
        ax2 = ax1.twinx()
        ax2.plot(self.model.patches.incidence[:, ipatch], color="purple", label="Incidence")
        plt.xlabel("Tick")
        plt.legend()

        yield

        return
