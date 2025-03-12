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
        model.agents.add_vector_property("V1", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V2", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V1imm", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V1sus", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V1inf", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V2imm", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V2sus", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("V2inf", length=model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "Vaccinated: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "E"), "Vaccinated: model agents needs to have a 'e' (exposed) attribute."

        assert "phi_1" in self.model.params, "Vaccinated: model params needs to have a 'phi_1' parameter."
        assert "phi_2" in self.model.params, "Vaccinated: model params needs to have a 'phi_2' parameter."
        assert "omega_1" in self.model.params, "Vaccinated: model params needs to have a 'omega_1' parameter."
        assert "omega_2" in self.model.params, "Vaccinated: model params needs to have a 'omega_2' parameter."
        assert "nu_1jt" in self.model.params, "Vaccinated: model params needs to have a 'nu_1jt' parameter."
        assert "nu_2jt" in self.model.params, "Vaccinated: model params needs to have a 'nu_2jt' parameter."

        assert hasattr(self.model.params, "d_jt"), "Susceptible: model.params needs to have a 'd_jt' attribute."

        return

    def __call__(self, model, tick: int) -> None:
        V1_next = model.agents.V1[tick + 1]
        V2_next = model.agents.V2[tick + 1]
        V1imm = model.agents.V1imm[tick]
        V1sus = model.agents.V1sus[tick]
        V1inf = model.agents.V1inf[tick]
        V1imm_next = model.agents.V1imm[tick + 1]
        V1sus_next = model.agents.V1sus[tick + 1]
        V1inf_next = model.agents.V1inf[tick + 1]
        V2imm = model.agents.V2imm[tick]
        V2sus = model.agents.V2sus[tick]
        V2inf = model.agents.V2inf[tick]
        V2imm_next = model.agents.V2imm[tick + 1]
        V2sus_next = model.agents.V2sus[tick + 1]
        V2inf_next = model.agents.V2inf[tick + 1]
        S = model.agents.S[tick]
        E = model.agents.E[tick]

        V1imm_next[:] = V1imm
        V1sus_next[:] = V1sus
        V1inf_next[:] = V1inf
        V2imm_next[:] = V2imm
        V2sus_next[:] = V2sus
        V2inf_next[:] = V2inf

        # -natural mortality
        deaths = np.binomial(V1imm, -np.expm1(-model.params.d_jt[:, tick])).astype(V1imm.dtype)
        V1imm_next -= deaths
        deaths = np.binomial(V1sus, -np.expm1(-model.params.d_jt[:, tick])).astype(V1sus.dtype)
        V1sus_next -= deaths
        deaths = np.binomial(V1inf, -np.expm1(-model.params.d_jt[:, tick])).astype(V1inf.dtype)
        V1inf_next -= deaths

        deaths = np.binomial(V2imm, -np.expm1(-model.params.d_jt[:, tick])).astype(V2imm.dtype)
        V2imm_next -= deaths
        deaths = np.binomial(V2sus, -np.expm1(-model.params.d_jt[:, tick])).astype(V2sus.dtype)
        V2sus_next -= deaths
        deaths = np.binomial(V2inf, -np.expm1(-model.params.d_jt[:, tick])).astype(V2inf.dtype)
        V2inf_next -= deaths

        # -waning immunity
        waned = np.binomial(V1imm_next, -np.expm1(-model.params.omega_1)).astype(V1imm_next.dtype)
        V1imm_next -= waned
        V1sus_next += waned

        waned = np.binomial(V2imm_next, -np.expm1(-model.params.omega_2)).astype(V2imm_next.dtype)
        V2imm_next -= waned
        V2sus_next += waned

        # +newly vaccinated (successful take)
        new_one_doses = np.poisson(model.params.nu_1jt[tick] * S / (S + E)).astype(V1imm.dtype)
        S -= new_one_doses
        # effective doses
        new_immunized = np.round(model.params.phi_1 * new_one_doses).astype(V1imm.dtype)
        V1imm_next += new_immunized
        # infective doses
        new_ineffective = new_one_doses - new_immunized
        V1sus_next += new_ineffective

        # -second dose recipients
        V1 = (V1imm + V1sus + V1inf).astype(V1imm.dtype)
        new_two_doses = np.poisson(model.params.nu_2jt[tick] * V1).astype(V1imm.dtype)
        v1imm_contribution = np.round((V1imm / V1) * new_two_doses).astype(V1imm.dtype)
        V1imm_next -= v1imm_contribution
        v1sus_contribution = np.round((V1sus / V1) * new_two_doses).astype(V1sus.dtype)
        V1sus_next -= v1sus_contribution
        v1inf_contribution = new_two_doses - v1imm_contribution - v1sus_contribution
        V1inf_next -= v1inf_contribution
        # effective doses
        # TODO - use new_two_doses here or (v1imm_contribution + v1sus_contribution)?
        new_immunized = np.round(model.params.phi_2 * ((V1imm + V1sus) / V1) * new_two_doses).astype(V2imm.dtype)
        V2imm_next += new_immunized
        # infective doses
        new_infective = np.round((1 - model.params.phi_2) * ((V1imm + V1sus) / V1) * new_two_doses).astype(V2sus.dtype)
        V2sus_next += new_infective
        # doses applied to previously infected one dose recipients
        v2inf_delta = new_two_doses - new_immunized - new_infective
        V2inf_next += v2inf_delta

        # V1 total
        V1_next[:] = V1imm_next + V1sus_next + V1inf_next

        # V2 total
        V2_next[:] = V2imm_next + V2sus_next + V2inf_next

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, d_jt: np.ndarray = None, omega: float = 0.0, phi: float = 0.0, nu_jt: np.ndarray = np.zeros((4, 8))):  # noqa: B008
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.S[0] = [1_000, 10_000, 100_000, 1_000_000]
                d_jt = d_jt if d_jt is not None else np.full((4, 8), 1.0 / 80.0)
                self.params = PropertySet({"d_jt": d_jt, "omega": omega, "phi": phi, "nu_jt": nu_jt, "nticks": 8})

        component = Vaccinated(model := Model(d_jt=np.full((4, 1), 1.0 / 80.0)))
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
