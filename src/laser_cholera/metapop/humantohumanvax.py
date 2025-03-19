"""Human-to-human transmission rate."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class HumanToHumanVax:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        return

    def check(self):
        assert hasattr(self.model, "patches"), "HumanToHuman: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "Lambda"), "HumanToHuman: model patches needs to have a 'Lambda' attribute."

        assert hasattr(self.model, "agents"), "HumanToHuman: model needs to have a 'agents' attribute."
        assert hasattr(self.model.agents, "E"), "HumanToHuman: model agents needs to have a 'E' (exposed) attribute."
        assert hasattr(self.model.agents, "V1sus"), (
            "HumanToHuman: model agents needs to have a 'V1sus' (vaccinated 1 susceptible) attribute."
        )
        assert hasattr(self.model.agents, "V1inf"), (
            "HumanToHuman: model agents needs to have a 'V1inf' (vaccinated 1 infectious) attribute."
        )
        assert hasattr(self.model.agents, "V2sus"), (
            "HumanToHuman: model agents needs to have a 'V2sus' (vaccinated 2 susceptible) attribute."
        )
        assert hasattr(self.model.agents, "V2inf"), (
            "HumanToHuman: model agents needs to have a 'V2inf' (vaccinated 2 infectious) attribute."
        )

        assert hasattr(self.model, "params"), "HumanToHuman: model needs to have a 'params' attribute."
        assert "tau_i" in self.model.params, "HumanToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        r"""
        Calculate the current human-to-human transmission rate per patch.

        $\Lambda_{j,t+1} = \frac {\beta^{hum}_{jt}((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{\forall i \neq j (\pi_{ij} \tau_j I_{it})}))^{\alpha_1}} {N^{\alpha_2}_{jt}}$

        """

        Lambda = model.patches.Lambda[tick + 1]
        Enext = model.agents.E[tick + 1]

        # LambdaV1
        V1sus = model.agents.V1sus[tick]
        local = np.round((1 - model.params.tau_i) * V1sus).astype(V1sus.dtype)
        infections = model.prng.binomial(local, -np.expm1(-Lambda)).astype(V1sus.dtype)
        V1sus_next = model.agents.V1sus[tick + 1]
        V1inf_next = model.agents.V1inf[tick + 1]
        V1sus_next -= infections
        V1inf_next += infections
        Enext += infections

        assert np.all(V1sus_next >= 0), f"V1sus' should not go negative ({tick=}\n\t{V1sus_next})"

        # LambdaV2
        V2sus = model.agents.V2sus[tick]
        local = np.round((1 - model.params.tau_i) * V2sus).astype(V2sus.dtype)
        infections = model.prng.binomial(local, -np.expm1(-Lambda)).astype(V2sus.dtype)
        V2sus_next = model.agents.V2sus[tick + 1]
        V2inf_next = model.agents.V2inf[tick + 1]
        V2sus_next -= infections
        V2inf_next += infections
        Enext += infections

        assert np.all(V2sus_next >= 0), f"V2sus' should not go negative ({tick=}\n\t{V2sus_next})"

        return

    @staticmethod
    def test():
        class Model:
            def __init__(
                self,
                alpha1: float = 1.0,
                alpha2: float = 1.0,
                tau_i: float = 0.0,
            ):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("I", length=8, dtype=np.int32, default=0)
                self.agents.add_vector_property("N", length=8, dtype=np.int32, default=0)
                self.agents.S[0] = self.agents.S[1] = [250, 2_500, 25_000, 250_000]  # 25%
                self.agents.I[0] = self.agents.I[1] = [100, 1_000, 10_000, 100_000]  # 10%
                self.agents.N[0] = self.agents.N[1] = [1_000, 10_000, 100_000, 1_000_000]
                self.patches = LaserFrame(4)  # required for HumanToHuman to add Lambda
                self.params = PropertySet(
                    {
                        "alpha1": alpha1,
                        "alpha2": alpha2,
                        "tau_i": tau_i,
                        "nticks": 8,
                    }
                )

        component = HumanToHumanVax(
            model := Model(
                alpha1=0.95,
                alpha2=0.95,
                tau_i=np.array([1.2521719e-03, 4.1985715e-04, 2.0205112e-02, 9.2022895e-03]),  # 1000x some actual values
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
            plt.plot(self.model.patches.Lambda[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return


if __name__ == "__main__":
    HumanToHuman.test()
    # plt.show()
