from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class Census:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Census: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("N", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Census: model needs to have a 'params' attribute."
        assert "N_j_initial" in self.model.params, "Census: model params needs to have a 'N_j_initial' parameter."
        model.agents.N[0] = model.params.N_j_initial

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "Census: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "E"), "Census: model agents needs to have a 'E' (exposed) attribute."
        assert hasattr(self.model.agents, "Isym"), "Census: model agents needs to have a 'Isym' (infectious - symptomatic) attribute."
        assert hasattr(self.model.agents, "Iasym"), "Census: model agents needs to have a 'Iasym' (infectious - asymptomatic) attribute."
        assert hasattr(self.model.agents, "R"), "Census: model agents needs to have a 'R' (recovered) attribute."
        assert hasattr(self.model.agents, "V1"), "Census: model agents needs to have a 'V1' (vaccinated - one dose) attribute."
        assert hasattr(self.model.agents, "V2"), "Census: model agents needs to have a 'V2' (vaccinated - two dose) attribute."

        assert np.all(
            self.model.agents.N[0]
            == self.model.agents.S[0]
            + self.model.agents.E[0]
            + self.model.agents.Isym[0]
            + self.model.agents.Iasym[0]
            + self.model.agents.R[0]
            + self.model.agents.V1[0]
            + self.model.agents.V2[0]
        )

        return

    def __call__(self, model, tick: int) -> None:
        S_next = model.agents.S[tick + 1]
        E_next = model.agents.E[tick + 1]
        Is_next = model.agents.Isym[tick + 1]
        Ia_next = model.agents.Iasym[tick + 1]
        R_next = model.agents.R[tick + 1]
        V1_next = model.agents.V1[tick + 1]
        V2_next = model.agents.V2[tick + 1]
        N_next = model.agents.N[tick + 1]

        N_next[:] = S_next + E_next + Is_next + Ia_next + R_next + V1_next + V2_next

        assert np.all(N_next >= 0), "N' should not go negative"

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("I", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("R", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("V", length=8, dtype=np.int32, default=0)
                self.agents.S[0] = self.agents.S[1] = [250, 2_500, 25_000, 250_000]
                self.agents.I[0] = self.agents.I[1] = [100, 1_000, 10_000, 100_000]
                self.agents.R[0] = self.agents.R[1] = [500, 5_000, 50_000, 500_000]
                self.agents.V[0] = self.agents.V[1] = [150, 1_500, 15_000, 150_000]
                self.params = PropertySet({"N_j_initial": [1_000, 10_000, 100_000, 1_000_000], "nticks": 8})

        component = Census(model := Model())
        component.check()
        component(model, 0)
        assert np.all(model.agents.N[0] == model.params.N_j_initial), "Initial populations didn't match."
        S1 = model.agents.S[1]
        I1 = model.agents.I[1]
        R1 = model.agents.R[1]
        V1 = model.agents.V[1]
        assert np.all(model.agents.N[1] == S1 + I1 + R1 + V1), (
            f"N[1] didn't match S[1] + I[1] + R[1] + V[1].\n\t{model.agents.N[1]}\n\t{S1 + I1 + R1 + V1}"
        )

        print("PASSED Census.test()")

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Census")
        # plt.plot(self.model.agents.N[0:-1, :], color="black", label="Total Population")
        # for ipatch in np.argsort(self.model.params.S_j_initial):
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.N[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
