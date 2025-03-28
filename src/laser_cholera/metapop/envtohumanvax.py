"""Environmental transmission rate."""

import numpy as np
from matplotlib.figure import Figure


class EnvToHumanVax:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        return

    def check(self):
        assert hasattr(self.model, "agents"), "EnvToHuman: model needs to have a 'agents' attribute."
        assert hasattr(self.model.agents, "E"), "EnvToHuman: model agents needs to have a 'E' (exposed) attribute."
        assert hasattr(self.model.agents, "V1sus"), "EnvToHuman: model agents needs to have a 'V1sus' (vaccinated 1 susceptible) attribute."
        assert hasattr(self.model.agents, "V1inf"), "EnvToHuman: model agents needs to have a 'V1inf' (vaccinated 1 infected) attribute."
        assert hasattr(self.model.agents, "V2sus"), "EnvToHuman: model agents needs to have a 'V2sus' (vaccinated 2 susceptible) attribute."
        assert hasattr(self.model.agents, "V2inf"), "EnvToHuman: model agents needs to have a 'V2inf' (vaccinated 2 infected) attribute."

        assert hasattr(self.model, "patches"), "EnvToHuman: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "Psi"), (
            "EnvToHuman: model patches needs to have a 'Psi' (environmental transmission rate) attribute."
        )

        assert hasattr(self.model, "params"), "EnvToHuman: model needs to have a 'params' attribute."
        assert "tau_i" in self.model.params, "EnvToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        Psi = model.patches.Psi[tick + 1]
        tau_i = model.params.tau_i
        local_frac = 1 - tau_i

        Enext = model.agents.E[tick + 1]

        # PsiV1
        V1sus = model.agents.V1sus[tick]
        V1sus_next = model.agents.V1sus[tick + 1]
        V1inf_next = model.agents.V1inf[tick + 1]
        local_v1 = np.round(local_frac * V1sus).astype(V1sus.dtype)
        infections_v1 = model.prng.binomial(local_v1, -np.expm1(-Psi)).astype(V1sus.dtype)
        V1sus_next -= infections_v1
        V1inf_next += infections_v1
        Enext += infections_v1

        assert np.all(V1sus_next >= 0), f"V1sus' should not go negative ({tick=}\n\t{V1sus_next})"

        # PsiV2
        V2sus = model.agents.V2sus[tick]
        V2sus_next = model.agents.V2sus[tick + 1]
        V2inf_next = model.agents.V2inf[tick + 1]
        local_v2 = np.round(local_frac * V2sus).astype(V2sus.dtype)
        infections_v2 = model.prng.binomial(local_v2, -np.expm1(-Psi)).astype(V2sus.dtype)
        V2sus_next -= infections_v2
        V2inf_next += infections_v2
        Enext += infections_v2

        assert np.all(V2sus_next >= 0), f"V2sus' should not go negative ({tick=}\n\t{V2sus_next})"

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        # plt.title("Environmental Transmission Rate (Vax)")
        # for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
        #     plt.plot(self.model.patches.Psi[:, ipatch], label=f"Patch {ipatch}")
        # plt.xlabel("Tick")
        # plt.legend()

        yield
        return
