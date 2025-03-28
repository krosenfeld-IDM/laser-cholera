"""Human-to-human transmission rate."""

import matplotlib.pyplot as plt
import numpy as np
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

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        # plt.title("Human-to-Human Transmission Rate (Vax)")
        # for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
        #     plt.plot(self.model.patches.Lambda[:, ipatch], label=f"Patch {ipatch}")
        # plt.xlabel("Tick")
        # plt.legend()

        yield
        return


if __name__ == "__main__":
    HumanToHumanVax.test()
