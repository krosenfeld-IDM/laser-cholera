"""Human-to-human transmission rate."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class HumanToHuman:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "patches"), "HumanToHuman: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("LAMDA", length=model.params.nticks + 1, dtype=np.float32, default=0.0)

        return

    def check(self):
        assert hasattr(self.model, "agents"), "HumanToHuman: model needs to have a 'agents' attribute."
        assert hasattr(self.model.agents, "S"), "HumanToHuman: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "I"), "HumanToHuman: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(self.model.agents, "N"), "HumanToHuman: model agents needs to have a 'N' (current agents) attribute."
        assert hasattr(self.model, "params"), "HumanToHuman: model needs to have a 'params' attribute."
        assert "alpha" in self.model.params, "HumanToHuman: model params needs to have an 'alpha' parameter."
        assert hasattr(self.model.params, "tau_i"), (
            "HumanToHuman: model.params needs to have a 'tau_i' (emmigration probability) parameter."
        )
        assert hasattr(self.model.params, "pi_ij"), (
            "HumanToHuman: model.params needs to have a 'pi_ij' (connectivity probability) parameter."
        )
        assert hasattr(self.model.params, "alpha"), "HumanToHuman: model.params needs to have a 'alpha' (population mixing) parameter."
        assert hasattr(self.model.params, "beta_j0_hum"), (
            "HumanToHuman: model.params needs to have a 'beta_j0_hum' (baseline transmission rate) parameter."
        )
        assert hasattr(self.model.params, "beta_j_seasonality"), (
            "HumanToHuman: model.params needs to have a 'beta_j_seasonality' (seasonal transmission rate) parameter."
        )

        return

    def __call__(self, model, tick: int) -> None:
        r"""
        Calculate the current human-to-human transmission rate per patch.

        $\Lambda_{j,t+1} = \frac {\beta^{hum}_{jt}((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{\forall i \neq j (\pi_{ij} \tau_j I_{it})}))^{\alpha_1}} {N^{\alpha_2}_{jt}}$

        """

        Lprime = model.patches.LAMDA[tick + 1]
        I = model.agents.I[tick]  # noqa: E741
        S = model.agents.S[tick]
        Sprime = model.agents.S[tick + 1]
        Iprime = model.agents.I[tick + 1]
        N = model.agents.N[tick]

        local_i = (I * (1 - model.params.tau_i)).astype(Lprime.dtype)
        # TODO - sum over axis=0 or axis=1?
        immigrating_i = (model.params.pi_ij * (model.params.tau_i * I)).sum(axis=1).astype(Lprime.dtype)
        effective_i = local_i + immigrating_i
        power_adjusted = np.power(effective_i, model.params.alpha1).astype(Lprime.dtype)
        seasonality = (model.params.beta_j0_hum + model.params.beta_j_seasonality[:, tick]).astype(Lprime.dtype)
        adjusted = seasonality * power_adjusted
        denominator = np.power(N, model.params.alpha2).astype(Lprime.dtype)
        rate = adjusted / denominator
        Lprime[:] = rate

        # TODO - still relevant?
        assert np.all(Lprime >= 0), "λ' should not go negative"

        local_s = np.round(S * (1 - model.params.tau_i)).astype(Lprime.dtype)
        probability = np.negative(np.expm1(-Lprime))
        newly_infected = model.prng.binomial(local_s, probability).astype(Sprime.dtype)

        psymptomatic = np.negative(np.expm1(model.params.sigma))
        newly_symptomatic = model.prng.binomial(newly_infected, psymptomatic).astype(Iprime.dtype)
        newly_asymptomatic = newly_infected - newly_symptomatic

        Sprime -= newly_infected
        Iprime += newly_infected

        return

    @staticmethod
    def test():
        class Model:
            def __init__(
                self, alpha: float = 0.0, tau_i: float = 0.0, pi_ij: float = 0.0, beta_j0_hum: float = 0.0, beta_j_seasonality: float = 0.0
            ):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("I", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("N", length=8, dtype=np.int32, default=0)
                self.agents.S[0] = self.agents.S[1] = [250, 2_500, 25_000, 250_000]  # 25%
                self.agents.I[0] = self.agents.I[1] = [100, 1_000, 10_000, 100_000]  # 10%
                self.agents.N[0] = self.agents.N[1] = [1_000, 10_000, 100_000, 1_000_000]
                self.patches = LaserFrame(4)  # required for HumanToHuman to add LAMDA
                self.params = PropertySet(
                    {
                        "alpha": alpha,
                        "tau_i": tau_i,
                        "pi_ij": pi_ij,
                        "beta_j0_hum": beta_j0_hum,
                        "beta_j_seasonality": beta_j_seasonality,
                        "nticks": 8,
                    }
                )

        component = HumanToHuman(
            model := Model(
                alpha=0.95,
                tau_i=np.array([1.2521719e-03, 4.1985715e-04, 2.0205112e-02, 9.2022895e-03]),  # 1000x some actual values
                pi_ij=np.array(
                    [
                        [0.0000, 0.2625, 0.1326, 0.1489],  # 10x some actual values
                        [0.2066, 0.0000, 0.0473, 0.0538],
                        [0.1093, 0.0495, 0.0000, 0.7596],
                        [0.1190, 0.0546, 0.7362, 0.0000],
                    ]
                ),
                beta_j0_hum=np.array([1.0, 1.0, 1.0, 1.0]),
                beta_j_seasonality=np.array(
                    [
                        [-0.0158, -0.0072, 0.0014, 0.0101, 0.0189, 0.0278, 0.0367, 0.0458],
                        [-0.8522, -0.8663, -0.8799, -0.8928, -0.9053, -0.9172, -0.9286, -0.9395],
                        [-0.0254, -0.0161, -0.0071, 0.0018, 0.0105, 0.0190, 0.0273, 0.0354],
                        [-0.0254, -0.0161, -0.0071, 0.0018, 0.0105, 0.0190, 0.0273, 0.0354],
                    ]
                ),
            )
        )
        component.check()
        component(model, 0)
        assert np.all(model.agents.S[1] < model.agents.S[0]), (
            f"Some susceptible populations didn't decline based on human-human transmission.\n\t{model.agents.S[0]}\n\t{model.agents.S[1]}"
        )
        assert np.all(model.agents.I[1] > model.agents.I[0]), (
            f"Some infected populations didn't increase based on human-human transmission.\n\t{model.agents.I[0]}\n\t{model.agents.I[1]}"
        )
        assert np.all(model.agents.S[0] - model.agents.S[1] == model.agents.I[1] - model.agents.I[0]), (
            f"S and I population changes don't match.\n\t{model.agents.S[0]=}\n\t{model.agents.S[1]=}\n\t{model.agents.I[0]=}\n\t{model.agents.I[1]=}"
        )

        print("PASSED HumanToHuman.test()")
        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Human-to-Human Transmission Rate")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.LAMDA[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return


if __name__ == "__main__":
    HumanToHuman.test()
    # plt.show()
