from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure

from laser_cholera.utils import printgreen


class Recovered:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Recovered: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("R", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "params"), "Recovered: model needs to have a 'params' attribute."
        assert "R_j_initial" in model.params, (
            "Recovered: model params needs to have a 'R_j_initial' (initial recovered population) parameter."
        )

        model.agents.R[0] = model.params.R_j_initial

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "Recovered: model agents needs to have a 'S' (susceptible) attribute."
        assert "d_jt" in self.model.params, "Recovered: model params needs to have a 'd_jt' (mortality rate) parameter."
        assert "epsilon" in self.model.params, "Recovered: model params needs to have a 'epsilon' (waning immunity rate) parameter."
        return

    def __call__(self, model, tick: int) -> None:
        R = model.agents.R[tick]
        R_next = model.agents.R[tick + 1]
        S_next = model.agents.S[tick + 1]

        R_next[:] = R

        # natural mortality
        deaths = model.prng.binomial(R_next, -np.expm1(-model.params.d_jt[tick])).astype(R_next.dtype)
        R_next -= deaths

        # waning natural immunity
        waned = model.prng.binomial(R_next, -np.expm1(-model.params.epsilon)).astype(R_next.dtype)
        R_next -= waned
        S_next += waned

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, d_jt: np.ndarray = None, epsilon: float = 0.0, gamma: float = 0.0):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("I", length=8, dtype=np.int32, default=0)
                self.agents.I[0] = [1_000, 10_000, 100_000, 1_000_000]
                d_jt = d_jt if d_jt is not None else np.full((4, 1), 1.0 / 80.0)
                self.params = PropertySet(
                    {"d_jt": d_jt, "epsilon": epsilon, "gamma": gamma, "R_j_initial": [1_000, 10_000, 100_000, 1_000_000], "nticks": 8}
                )

        component = Recovered(model := Model(d_jt=np.full((4, 1), 1.0 / 80.0)))
        component.check()
        component(model, 0)
        assert np.all(model.agents.R[0] == model.params.R_j_initial), "Initial populations didn't match."
        assert np.all(model.agents.R[1] < model.agents.R[0]), (
            f"Some populations didn't shrink with natural mortality.\n\t{model.agents.R[0]}\n\t{model.agents.R[1]}"
        )

        component = Recovered(model := Model(epsilon=0.04))
        component.check()
        component(model, 0)
        assert np.all(model.agents.R[0] == model.params.R_j_initial), "Initial populations didn't match."
        assert np.all(model.agents.R[1] < model.agents.R[0]), (
            f"Some populations didn't shrink with waning natural immunity.\n\t{model.agents.R[0]}\n\t{model.agents.R[1]}"
        )

        component = Recovered(model := Model(gamma=0.1))
        component.check()
        component(model, 0)
        assert np.all(model.agents.R[0] == model.params.R_j_initial), "Initial populations didn't match."
        assert np.all(model.agents.R[1] > model.agents.R[0]), (
            f"Some populations didn't increase with recovery from infection.\n\t{model.agents.R[0]}\n\t{model.agents.R[1]}"
        )

        printgreen("PASSED Recovered.test()")

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


if __name__ == "__main__":
    Recovered.test()
