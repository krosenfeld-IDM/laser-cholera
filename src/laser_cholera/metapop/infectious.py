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
        model.agents.add_vector_property("Isym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("Iasym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("deaths", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "params"), "Infectious: model needs to have a 'params' attribute."
        assert hasattr(model.params, "I_j_initial"), (
            "Infectious: model params needs to have a 'I_j_initial' (initial infectious population) parameter."
        )
        assert "sigma" in self.model.params, "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter."
        model.agents.Isym[0] = np.round(model.params.sigma * model.params.I_j_initial).astype(model.agents.Isym)
        model.agents.Iasym[0] = model.params.I_j_initial - model.agents.Isym[0]

        return

    def check(self):
        assert hasattr(self.model.agents, "R"), "Infectious: model.agents needs to have a 'S' attribute."
        assert "d_jt" in self.model.params, "Infectious: model params needs to have a 'd_jt' (mortality rate) parameter."
        assert "mu" in self.model.params, "Infectious: model params needs to have a 'mu' (disease mortality rate) parameter."
        assert "gamma_1" in self.model.params, "Infectious: model params needs to have a 'gamma_1' (recovery rate) parameter."
        assert "gamma_2" in self.model.params, "Infectious: model params needs to have a 'gamma_2' (recovery rate) parameter."
        assert "iota" in self.model.params, "Infectious: model params needs to have a 'iota' (progression rate) parameter."
        assert "sigma" in self.model.params, "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        # Symptomatic
        Isym = model.agents.I[tick]
        Isprime = model.agents.I[tick + 1]
        Isprime[:] = Isym

        ## natural deaths (d_jt)
        natural_deaths = np.binomial(Isprime, -np.expm1(-model.params.d_jt[:, tick])).astype(Isprime.dtype)
        Isprime -= natural_deaths

        ## disease deaths (mu)
        disease_deaths = np.binomial(Isprime, -np.expm1(-model.params.mu)).astype(Isprime.dtype)
        model.agents.deaths[tick] = disease_deaths
        Isprime -= disease_deaths

        ## recovery (gamma)
        recovered = np.binomial(Isprime, -np.expm1(-model.params.gamma_1)).astype(Isprime.dtype)
        Isprime -= recovered
        Rprime = model.agents.R[tick + 1]
        Rprime += recovered

        # Asymptomatic
        Iasym = model.agents.Iasym[tick]
        Iaprime = model.agents.Iasym[tick + 1]
        Iaprime[:] = Iasym

        ## natural deaths (d_jt)
        natural_deaths = np.binomial(Iaprime, -np.expm1(-model.params.d_jt[:, tick])).astype(Iaprime.dtype)
        Iaprime -= natural_deaths

        ## recovery
        recovered = np.binomial(Iaprime, -np.expm1(-model.params.gamma_2)).astype(Iaprime.dtype)
        Iaprime -= recovered
        # Rprime = model.agents.R[tick + 1]
        Rprime += recovered

        # Use Eprime here, can't progress deceased individuals
        Eprime = model.agents.E[tick + 1]
        progress = np.binomial(Eprime, -np.expm1(-model.params.iota)).astype(Eprime.dtype)
        Eprime -= progress

        ## new symptomatic infections
        new_symptomatic = np.round(model.params.sigma * progress).astype(Isprime.dtype)
        new_asymptomatic = progress - new_symptomatic
        Isprime += new_symptomatic
        Iaprime += new_asymptomatic

        # human-to-human infection in humantohuman.py
        # environmental infection in envtohuman.py
        # recovery from infection in recovered.py

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, d_jt: np.ndarray = None, mu: float = 0.0, sigma: float = 0.0):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                d_jt = d_jt if d_jt is not None else np.full((4, 1), 1.0 / 80.0)
                self.params = PropertySet(
                    {"d_jt": d_jt, "mu": mu, "sigma": sigma, "I_j_initial": [1_000, 10_000, 100_000, 1_000_000], "nticks": 8}
                )

        # TODO: Add more tests given Isym and Iasym split

        # component = Infectious(model := Model(d_jt=np.full((4, 1), 1.0 / 80.0)))
        # component.check()
        # component(model, 0)
        # assert np.all(model.agents.I[0] == model.params.I_j_initial), "Initial populations didn't match."
        # assert np.all(model.agents.I[1] < model.agents.I[0]), (
        #     f"Some populations didn't shrink with natural mortality.\n\t{model.agents.I[0]}\n\t{model.agents.I[1]}"
        # )

        # component = Infectious(model := Model(mu=0.015, sigma=0.24))
        # component.check()
        # component(model, 0)
        # assert np.all(model.agents.I[0] == model.params.I_j_initial), "Initial populations didn't match."
        # assert np.all(model.agents.I[1] < model.agents.I[0]), (
        #     f"Some populations didn't shrink with disease mortality.\n\t{model.agents.I[0]}\n\t{model.agents.I[1]}"
        # )

        print("PASSED Infectious.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Infectious (Symptomatic)")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.Isym[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield

        plt.title("Infectious (Asymptomatic)")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.Iasym[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield

        plt.title("Infectious (Total)")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.Isym[:, ipatch] + self.model.agents.Iasym[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
