from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class Component: ...


class Susceptible(Component):
    def __init__(self, mode_ell, verbose: bool = False):
        self.model = mode_ell
        self.verbose = verbose

        assert hasattr(mode_ell, "agents"), "Susceptible: model needs to have a 'agents' attribute."
        mode_ell.agents.add_vector_property("S", length=mode_ell.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Susceptible: model needs to have a 'params' attribute."
        assert "S_j_initial" in self.model.params, "Susceptible: model params needs to have a 'S_j_initial' parameter."
        mode_ell.agents.S[0] = mode_ell.params.S_j_initial

        return

    def check(self):
        assert hasattr(self.model.agents, "N"), "Susceptible: model.agents needs to have a 'N' attribute."
        assert hasattr(self.model.params, "b_jt"), "Susceptible: model.params needs to have a 'b_jt' attribute."
        assert hasattr(self.model.params, "d_jt"), "Susceptible: model.params needs to have a 'd_jt' attribute."

        return

    def __call__(self, model, tick: int) -> None:
        Sprime = model.agents.S[tick + 1]
        S = model.agents.S[tick]
        Sprime[:] = S

        # natural mortality
        deaths = np.binomial(S, -np.expm1(-model.params.d_jt[:, tick])).astype(Sprime.dtype)
        Sprime -= deaths

        # births
        N = model.agents.N[tick]
        births = np.binomial(N, -np.expm1(-model.params.b_jt[:, tick])).astype(Sprime.dtype)
        Sprime[:] += births

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, b_jt: np.ndarray = None, d_jt: np.ndarray = None):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("N", length=8, dtype=np.int32, default=0)
                self.agents.N[0] = [1_000, 10_000, 100_000, 1_000_000]
                b_jt = b_jt if b_jt is not None else np.full((4, 1), 3.0 / 80.0)
                d_jt = d_jt if d_jt is not None else np.full((4, 1), 1.0 / 80.0)
                self.params = PropertySet({"b_jt": b_jt, "d_jt": d_jt, "S_j_initial": [1_000, 10_000, 100_000, 1_000_000], "nticks": 8})

        component = Susceptible(model := Model(b_jt=np.full((4, 1), 3.0 / 80.0)))
        component.check()
        component(model, 0)
        assert np.all(model.agents.S[0] == model.params.S_j_initial), "Initial populations didn't match."
        assert np.all(model.agents.S[1] > model.agents.S[0]), (
            f"Some populations didn't grow with births.\n\t{model.agents.S[0]}\n\t{model.agents.S[1]}"
        )

        component = Susceptible(model := Model(d_jt=np.full((4, 1), 1.0 / 80.0)))
        component.check()
        component(model, 0)
        assert np.all(model.agents.S[0] == model.params.S_j_initial), "Initial populations didn't match."
        assert np.all(model.agents.S[1] < model.agents.S[0]), (
            f"Some populations didn't shrink with deaths.\n\t{model.agents.S[0]}\n\t{model.agents.S[1]}"
        )

        print("PASSED Susceptible.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Susceptibles")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.S[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
