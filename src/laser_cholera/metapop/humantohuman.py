"""Human-to-human transmission rate."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser_cholera.metapop.utils import get_daily_seasonality
from laser_cholera.metapop.utils import get_pi_from_lat_long
from laser_cholera.sc import printred


class HumanToHuman:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "patches"), "HumanToHuman: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("Lambda", length=model.params.nticks + 1, dtype=np.float32, default=0.0)

        assert "latitude" in self.model.params, "HumanToHuman: model params needs to have a 'latitude' (location latitude) parameter."
        assert "longitude" in self.model.params, "HumanToHuman: model params needs to have a 'longitude' (location longitude) parameter."
        assert "mobility_omega" in self.model.params, "HumanToHuman: model params needs to have a 'mobility_omega' (mobility) parameter."
        assert "mobility_gamma" in self.model.params, "HumanToHuman: model params needs to have a 'mobility_gamma' (mobility) parameter."

        model.patches.add_array_property("pi_ij", (model.patches.count, model.patches.count), dtype=np.float32, default=0.0)
        model.patches.pi_ij[:, :] = get_pi_from_lat_long(model.params)

        assert "a_1_j" in self.model.params, "HumanToHuman: model params needs to have a 'a_1_j' (seasonality) parameter."
        assert "b_1_j" in self.model.params, "HumanToHuman: model params needs to have a 'b_1_j' (seasonality) parameter."
        assert "a_2_j" in self.model.params, "HumanToHuman: model params needs to have a 'a_2_j' (seasonality) parameter."
        assert "b_2_j" in self.model.params, "HumanToHuman: model params needs to have a 'b_2_j' (seasonality) parameter."
        assert "p" in self.model.params, "HumanToHuman: model params needs to have a 'p' (seasonality pahse) parameter."

        model.patches.add_array_property("beta_j_seasonality", (model.params.p, model.patches.count), dtype=np.float32, default=0.0)
        model.patches.beta_j_seasonality[:, :] = get_daily_seasonality(model.params)

        return

    def check(self):
        assert hasattr(self.model, "agents"), "HumanToHuman: model needs to have a 'agents' attribute."
        assert hasattr(self.model.agents, "Isym"), "HumanToHuman: model agents needs to have a 'Isym' (symptomatic) attribute."
        assert hasattr(self.model.agents, "Iasym"), "HumanToHuman: model agents needs to have a 'Iasym' (asymptomatic) attribute."
        assert hasattr(self.model.agents, "S"), "HumanToHuman: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "E"), "HumanToHuman: model agents needs to have a 'E' (exposed) attribute."

        assert hasattr(self.model.patches, "N"), "HumanToHuman: model agents needs to have a 'N' (current agents) attribute."

        assert hasattr(self.model, "params"), "HumanToHuman: model needs to have a 'params' attribute."
        assert "tau_i" in self.model.params, "HumanToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter."
        assert "beta_j0_hum" in self.model.params, (
            "HumanToHuman: model params needs to have a 'beta_j0_hum' (baseline transmission rate) parameter."
        )

        assert "alpha_1" in self.model.params, "HumanToHuman: model params needs to have an 'alpha_1' (numerator power) parameter."
        assert "alpha_2" in self.model.params, "HumanToHuman: model params needs to have an 'alpha_2' (denominator power) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        r"""
        Calculate the current human-to-human transmission rate per patch.

        $\Lambda_{j,t+1} = \frac {\beta^{hum}_{jt}((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{\forall i \neq j (\pi_{ij} \tau_j I_{it})}))^{\alpha_1}} {N^{\alpha_2}_{jt}}$

        """

        # LambdaS
        Lambda = model.patches.Lambda[tick + 1]
        Isym = model.agents.Isym[tick]
        Iasym = model.agents.Iasym[tick]
        N = model.patches.N[tick]
        S = model.agents.S[tick]
        S_next = model.agents.S[tick + 1]
        E_next = model.agents.E[tick + 1]

        total_i = Isym + Iasym
        local_i = (total_i * (1 - model.params.tau_i)).astype(Lambda.dtype)
        # TODO - sum over axis=0 or axis=1?
        immigrating_i = (model.patches.pi_ij * (model.params.tau_i * total_i)).sum(axis=1).astype(Lambda.dtype)
        effective_i = local_i + immigrating_i
        power_adjusted = np.power(effective_i, model.params.alpha_1).astype(Lambda.dtype)
        seasonality = (model.params.beta_j0_hum * (1.0 + model.patches.beta_j_seasonality[tick % model.params.p, :])).astype(Lambda.dtype)
        adjusted = seasonality * power_adjusted
        denominator = np.power(N, model.params.alpha_2).astype(Lambda.dtype)
        rate = (adjusted / denominator).astype(Lambda.dtype)

        # TODO - check seasonality and power_adjusted for negative values so we don't have to do this
        if np.any(rate < 0.0):
            printred(f"Negative transmission rate at tick {tick + 1}.\n\t{rate=}")
            rate = np.maximum(rate, 0.0)

        Lambda[:] = rate
        local = np.round((1 - model.params.tau_i) * S).astype(S.dtype)
        if np.any(np.isnan(rate)):
            printred(f"NaN transmission rate at tick {tick + 1}.\n\t{rate=}")
        infections = model.prng.binomial(local, -np.expm1(-rate)).astype(S.dtype)
        S_next -= infections
        E_next += infections

        assert np.all(S_next >= 0), f"Negative susceptible populations at tick {tick + 1}.\n\t{S_next=}"

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Human-to-Human Transmission Rate") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.Lambda[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Transmission Rate")
        plt.legend()

        yield "Human-to-Human Transmission Rate"

        # Spatial connectivity matrix
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Spatial Connectivity Matrix (pi_ij)") if fig is None else fig

        plt.imshow(self.model.patches.pi_ij, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Connectivity")
        plt.xlabel("Destination Location Index")
        plt.xticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Source Location Index")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.yticks(rotation=45, ha="right")

        yield "Spatial Connectivity Matrix (pi_ij)"

        # Seasonality factor by location over time
        _fig = (
            plt.figure(figsize=(12, 9), dpi=128, num="Seasonal Human-Human Transmission Factor by Location Over Time")
            if fig is None
            else fig
        )

        indices = np.argsort(self.model.params.latitude)[::-1]
        plt.imshow(self.model.patches.beta_j_seasonality[:, indices].T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Seasonal Factor")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=[self.model.params.location_name[i] for i in indices])

        yield "Seasonal Human-Human Transmission Factor by Location Over Time"

        return
