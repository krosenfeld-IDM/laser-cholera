from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class Infectious:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Infectious: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("I", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Infectious: model needs to have a 'params' attribute."
        assert hasattr(
            self.model.params, "I_j_initial"
        ), "Infectious: model params needs to have a 'I_j_initial' (initial infectious population) parameter."
        model.agents.I[0] = model.params.I_j_initial

        return

    def check(self):
        assert hasattr(self.model.params, "d_j"), "Infectious: model params needs to have a 'd_j' (mortality rate) parameter."
        assert hasattr(self.model.params, "mu"), "Infectious: model params needs to have a 'mu' (disease mortality rate) parameter."
        assert hasattr(self.model.params, "sigma"), "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        I = model.agents.I[tick]  # noqa: E741
        Iprime = model.agents.I[tick + 1]

        # natural mortality
        # TODO - rate or probability?
        Imort = model.prng.poisson(model.params.d_j * I).astype(Iprime.dtype)
        Iprime[:] = I - Imort

        # TODO - rate or probability?
        disease_deaths = model.prng.binomial(I, model.params.mu * model.params.sigma).astype(Iprime.dtype)
        Iprime -= disease_deaths

        # human-to-human infection in humantohuman.py
        # environmental infection in envtohuman.py
        # recovery from infection in recovered.py

        assert np.all(Iprime >= 0), "I' should not go negative"

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, d_j: float = 0.0, mu: float = 0.0, sigma: float = 0.0):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.params = PropertySet(
                    {"d_j": d_j, "mu": mu, "sigma": sigma, "I_j_initial": [1_000, 10_000, 100_000, 1_000_000], "nticks": 8}
                )

        component = Infectious(model := Model(d_j=1.0 / 80.0))
        component.check()
        component(model, 0)
        assert np.all(model.agents.I[0] == model.params.I_j_initial), "Initial populations didn't match."
        assert np.all(
            model.agents.I[1] < model.agents.I[0]
        ), f"Some populations didn't shrink with natural mortality.\n\t{model.agents.I[0]}\n\t{model.agents.I[1]}"

        component = Infectious(model := Model(mu=0.015, sigma=0.24))
        component.check()
        component(model, 0)
        assert np.all(model.agents.I[0] == model.params.I_j_initial), "Initial populations didn't match."
        assert np.all(
            model.agents.I[1] < model.agents.I[0]
        ), f"Some populations didn't shrink with disease mortality.\n\t{model.agents.I[0]}\n\t{model.agents.I[1]}"

        print("PASSED Infectious.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Infectious")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.I[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
