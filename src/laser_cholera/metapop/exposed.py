import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser_cholera.utils import printgreen


class Exposed:
    def __init__(self, model, verbose: bool = False):
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Exposed: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("E", length=model.params.nticks + 1, dtype=np.int32, default=0)

        assert hasattr(self.model, "params"), "Exposed: model needs to have a 'params' attribute."
        assert "E_j_initial" in self.model.params, "Exposed: model params needs to have a 'E_j_initial' parameter."

        model.agents.E[0] = model.params.E_j_initial

        return

    def check(self):
        # Don't bother checking for model.params, we did that in __init__()
        assert "iota" in self.model.params, "Exposed: model params needs to have a 'iota' (progression rate) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        E_next = model.agents.E[tick + 1]
        E = model.agents.E[tick]
        E_next[:] = E

        # Do non-disease mortality first
        deaths = model.prng.binomial(E, -np.expm1(-model.params.d_jt[tick])).astype(E_next.dtype)
        E_next -= deaths

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self): ...

        component = Exposed(model := Model())
        component.check()
        component(model, 0)
        assert np.all(model.agents.E[0] == model.params.E_j_initial), "Initial populations didn't match."
        assert np.all(model.agents.E[1] < model.agents.E[0]), (
            f"Some populations didn't shrink with deaths.\n\t{model.agents.E[0]}\n\t{model.agents.E[1]}"
        )

        printgreen("PASSED Susceptible.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Exposed")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.E[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return


if __name__ == "__main__":
    Exposed.test()
