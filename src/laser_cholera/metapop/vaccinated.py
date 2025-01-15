from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class Vaccinated:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Vaccinated: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("V", length=model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "Vaccinated: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.params, "d_j"), "Susceptible: model.params needs to have a 'd_j' attribute."
        assert hasattr(self.model.params, "omega"), "Vaccinated: model.params needs to have a 'omega' (vaccination waning rate) parameter."
        assert hasattr(self.model.params, "phi"), "Vaccinated: model.params needs to have a 'phi' (vaccination efficacy) parameter."
        assert hasattr(self.model.params, "nu_jt"), "Vaccinated: model.params needs to have a 'nu_jt' (vaccination rate) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        V = model.agents.V[tick]
        Vprime = model.agents.V[tick + 1]
        S = model.agents.S[tick]
        N = model.agents.N[tick]
        Sprime = model.agents.S[tick + 1]

        # non-disease mortality
        # TODO - rate or probability?
        Vmort = model.prng.poisson(model.params.d_j * V).astype(Vprime.dtype)
        Vprime[:] = V - Vmort

        # waning vaccine immunity
        # TODO - rate or probability?
        waned_vaccinations = model.prng.binomial(Vprime, model.params.omega).astype(Sprime.dtype)
        Sprime += waned_vaccinations
        Vprime -= waned_vaccinations

        # vaccination
        # TODO - look up vaccination rates correctly based on tick/calendar date
        # TODO - rate or probability?
        # TODO - verify `tick%365` is correct (it isn't)
        # Only vaccinating S'es, not worrying about Is, Rs, or existing Vs.
        newly_vaccinated = model.prng.poisson(model.params.phi * model.params.nu_jt[:, tick % 365] * S / N).astype(Sprime.dtype)
        Sprime -= newly_vaccinated
        Vprime += newly_vaccinated

        assert np.all(Sprime >= 0), "S' should not go negative"
        assert np.all(Vprime >= 0), "V' should not go negative"

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, d_j: float = 0.0, omega: float = 0.0, phi: float = 0.0, nu_jt: np.ndarray = np.zeros((4, 8))):  # noqa: B008
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.S[0] = [1_000, 10_000, 100_000, 1_000_000]
                self.params = PropertySet({"d_j": d_j, "omega": omega, "phi": phi, "nu_jt": nu_jt, "nticks": 8})

        component = Vaccinated(model := Model(d_j=1.0 / 80.0))
        component.check()
        model.agents.V[0] = [1_000, 10_000, 100_000, 1_000_000]
        component(model, 0)
        assert np.all(model.agents.V[1] < model.agents.V[0]), (
            f"Some populations didn't shrink with natural mortality.\n\t{model.agents.V[0]}\n\t{model.agents.V[1]}"
        )

        component = Vaccinated(model := Model(omega=0.06))
        component.check()
        model.agents.V[0] = [1_000, 10_000, 100_000, 1_000_000]
        component(model, 0)
        assert np.all(model.agents.V[1] < model.agents.V[0]), (
            f"Some populations didn't shrink with waning vaccine immunity.\n\t{model.agents.V[0]}\n\t{model.agents.V[1]}"
        )

        component = Vaccinated(model := Model(phi=0.0, nu_jt=0.1 * np.ones((4, 8))))
        component.check()
        model.agents.V[0] = [1_000, 10_000, 100_000, 1_000_000]
        component(model, 0)
        assert np.all(model.agents.V[1] == model.agents.V[0]), (
            f"Some populations changed with vaccine take = 0.\n\t{model.agents.V[0]}\n\t{model.agents.V[1]}"
        )

        component = Vaccinated(model := Model(phi=0.64, nu_jt=np.zeros((4, 8))))
        component.check()
        model.agents.V[0] = [1_000, 10_000, 100_000, 1_000_000]
        component(model, 0)
        assert np.all(model.agents.V[1] == model.agents.V[0]), (
            f"Some populations changed with vaccination rate = 0.\n\t{model.agents.V[0]}\n\t{model.agents.V[1]}"
        )

        component = Vaccinated(model := Model(phi=0.64, nu_jt=0.1 * np.ones((4, 8))))
        component.check()
        model.agents.V[0] = [1_000, 10_000, 100_000, 1_000_000]
        component(model, 0)
        assert np.all(model.agents.V[1] > model.agents.V[0]), (
            f"Some populations didn't increase with vaccine take > 0 and vaccination rate > 0.\n\t{model.agents.V[0]}\n\t{model.agents.V[1]}"
        )

        print("PASSED Vaccinated.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Vaccinated")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.V[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
