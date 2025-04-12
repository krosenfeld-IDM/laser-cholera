"""Environmental transmission rate."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class EnvToHuman:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "EnvToHuman: model needs to have a 'agents' attribute."
        model.patches.add_vector_property("Psi", length=model.params.nticks + 1, dtype=np.float32, default=np.float32(0.0))

        assert hasattr(model, "params"), "EnvToHuman: model needs to have a 'params' attribute."
        assert hasattr(model.params, "psi_jt"), (
            "EnvToHuman: model params needs to have a 'psi_jt' (environmental contamination rate) parameter."
        )

        psi = model.params.psi_jt  # convenience
        # TODO - use newer laser_core with add_array_property and psi.shape
        model.patches.add_vector_property("beta_env", length=psi.shape[0], dtype=np.float32, default=0.0)
        assert model.patches.beta_env.shape == model.params.psi_jt.shape
        assert model.params.beta_j0_env.shape[0] == model.patches.beta_env.shape[1]
        psi_bar = psi.mean(axis=0, keepdims=True)
        model.patches.beta_env[:, :] = model.params.beta_j0_env.T * (1.0 + (psi - psi_bar) / psi_bar)

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "EnvToHuman: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "E"), "EnvToHuman: model agents needs to have a 'E' (exposed) attribute."

        assert hasattr(self.model, "patches"), "EnvToHuman: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "W"), "EnvToHuman: model patches needs to have a 'W' (environmental) attribute."

        assert "tau_i" in self.model.params, "EnvToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter."
        assert "theta_j" in self.model.params, (
            "EnvToHuman: model params needs to have a 'theta_j' (fraction of population with WASH) attribute."
        )
        assert "kappa" in self.model.params, "EnvToHuman: model params needs to have a 'kappa' (environmental transmission rate) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        Psi = model.patches.Psi[tick + 1]
        W = model.patches.W[tick]
        tau_i = model.params.tau_i
        local_frac = 1 - tau_i

        non_wash = (1 - model.params.theta_j) * W
        seasonal = model.patches.beta_env[tick] * non_wash
        denominator = model.params.kappa + W
        normalized = seasonal / denominator
        Psi[:] = normalized

        # PsiS
        S = model.agents.S[tick]
        S_next = model.agents.S[tick + 1]
        E_next = model.agents.E[tick + 1]
        # Use S_next here since some S will have been removed by natural mortality and by human-to-human transmission
        local_s = np.round(S_next * local_frac).astype(S.dtype)
        infections_s = model.prng.binomial(local_s, -np.expm1(-Psi)).astype(S_next.dtype)
        S_next -= infections_s
        E_next += infections_s

        assert np.all(S_next >= 0), f"Negative susceptible populations at tick {tick + 1}.\n\t{S_next=}"

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Environmental Transmission Rate")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.Psi[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
