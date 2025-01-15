from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class Recovered:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Recovered: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("R", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Recovered: model needs to have a 'params' attribute."
        assert hasattr(
            self.model.params, "R_j_initial"
        ), "Recovered: model params needs to have a 'R_j_initial' (initial recovered population) parameter."
        model.agents.R[0] = model.params.R_j_initial

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "Recovered: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "I"), "Recovered: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(self.model.params, "d_j"), "Recovered: model params needs to have a 'd_j' (mortality rate) parameter."
        assert hasattr(self.model.params, "epsilon"), "Recovered: model params needs to have a 'epsilon' (waning immunity rate) parameter."
        assert hasattr(self.model.params, "gamma"), "Recovered: model params needs to have a 'gamma' (infection recovery rate) parameter."
        return

    def __call__(self, model, tick: int) -> None:
        I = model.agents.I[tick]  # noqa: E741
        Iprime = model.agents.I[tick + 1]
        R = model.agents.R[tick]
        Rprime = model.agents.R[tick + 1]
        Sprime = model.agents.S[tick + 1]

        # natural mortality
        # TODO - rate or probability?
        Rmort = model.prng.poisson(model.params.d_j * R).astype(Rprime.dtype)
        Rprime[:] = R - Rmort

        # waning natural immunity
        # TODO - rate or probability?
        waned_natural_immunity = model.prng.binomial(Rprime, model.params.epsilon).astype(Sprime.dtype)
        Sprime += waned_natural_immunity
        Rprime -= waned_natural_immunity

        # recovery from infection
        # TODO - rate or probability?
        newly_recovered = model.prng.binomial(I, model.params.gamma).astype(Iprime.dtype)
        Iprime -= newly_recovered
        Rprime += newly_recovered

        assert np.all(Sprime >= 0), "S' should not go negative"
        assert np.all(Iprime >= 0), "I' should not go negative"
        assert np.all(Rprime >= 0), "R' should not go negative"

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, d_j: float = 0.0, epsilon: float = 0.0, gamma: float = 0.0):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("I", length=8, dtype=np.int32, default=0)
                self.agents.I[0] = [1_000, 10_000, 100_000, 1_000_000]
                self.params = PropertySet(
                    {"d_j": d_j, "epsilon": epsilon, "gamma": gamma, "R_j_initial": [1_000, 10_000, 100_000, 1_000_000], "nticks": 8}
                )

        component = Recovered(model := Model(d_j=1.0 / 80.0))
        component.check()
        component(model, 0)
        assert np.all(model.agents.R[0] == model.params.R_j_initial), "Initial populations didn't match."
        assert np.all(
            model.agents.R[1] < model.agents.R[0]
        ), f"Some populations didn't shrink with natural mortality.\n\t{model.agents.R[0]}\n\t{model.agents.R[1]}"

        component = Recovered(model := Model(epsilon=0.04))
        component.check()
        component(model, 0)
        assert np.all(model.agents.R[0] == model.params.R_j_initial), "Initial populations didn't match."
        assert np.all(
            model.agents.R[1] < model.agents.R[0]
        ), f"Some populations didn't shrink with waning natural immunity.\n\t{model.agents.R[0]}\n\t{model.agents.R[1]}"

        component = Recovered(model := Model(gamma=0.1))
        component.check()
        component(model, 0)
        assert np.all(model.agents.R[0] == model.params.R_j_initial), "Initial populations didn't match."
        assert np.all(
            model.agents.R[1] > model.agents.R[0]
        ), f"Some populations didn't increase with recovery from infection.\n\t{model.agents.R[0]}\n\t{model.agents.R[1]}"

        print("PASSED Recovered.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Recovered")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.R[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
