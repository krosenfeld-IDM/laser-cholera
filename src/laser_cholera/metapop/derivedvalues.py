import matplotlib.pyplot as plt
import numpy as np


class DerivedValues:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "patches"), "DerivedValues: model needs to have a 'patches' attribute."
        assert hasattr(model, "params"), "DerivedValues: model needs to have a 'params' attribute."

        model.patches.add_vector_property("spatial_hazard", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        model.patches.add_array_property("coupling", shape=(model.patches.count, model.patches.count), dtype=np.float32, default=0.0)

        return

    def check(self):
        assert hasattr(self.model, "agents"), "DerivedValues: model needs to have an 'agents' attribute."
        assert hasattr(self.model.agents, "S"), "DerivedValues: model.agents needs to have 'S' attribute."
        assert hasattr(self.model.agents, "Isym"), "DerivedValues: model.agents needs to have 'Isym' attribute."
        assert hasattr(self.model.agents, "Iasym"), "DerivedValues: model.agents needs to have 'Iasym' attribute."
        assert hasattr(self.model.agents, "V1sus"), "DerivedValues: model.agents needs to have 'V1sus' attribute."
        assert hasattr(self.model.agents, "V2sus"), "DerivedValues: model.agents needs to have 'V2sus' attribute."

        assert hasattr(self.model, "patches"), "DerivedValues: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "N"), "DerivedValues: model.patches needs to have 'N' attribute."
        assert hasattr(self.model.patches, "beta_j_seasonality"), (
            "DerivedValues: model.patches needs to have 'beta_j_seasonality' attribute."
        )
        assert hasattr(self.model.patches, "pi_ij"), "DerivedValues: model.patches needs to have 'pi_ij' attribute."

        assert "beta_j0_hum" in self.model.params, "DerivedValues: model.params needs to have 'beta_j0_hum' attribute."
        assert "p" in self.model.params, "DerivedValues: model.params needs to have 'p' attribute."
        assert "tau_i" in self.model.params, "DerivedValues: model.params needs to have 'tau_i' attribute."

        return

    def __call__(self, model, tick: int) -> None:
        """Calculate derived values for the model.

        Spatial hazard and coupling are calculated at the end of the simulation.

        .. math::

            h(j,t) = \\frac {\\beta^{hum}_{jt} (1 - e^{-((1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}) / N_{jt}) \\sum_{\\forall i \\ne j} \\pi_{ij} \\tau_i ((I^{sym}_{it} + I^{asym}_{it}) / N_{it})})} {1/(1 + \\beta^{hum}_{jt}(1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}))}

        .. math::

            C_{ij} = \\frac {(y_{it} - \\bar y_{i}) (y_{jt} - \\bar y_{j})} { \\sqrt {\\text {var}(y_{i}) \\text {var}(y_{j})} }

        """
        if tick == model.params.nticks - 1:
            # spatial_hazard
            # https://www.mosaicmod.org/model-description.html#the-spatial-hazard
            for t in range(model.params.nticks):
                beta_jt = model.params.beta_j0_hum * (1.0 + model.patches.beta_j_seasonality[t % model.params.p, :])
                tau_j = model.params.tau_i
                S_j = model.agents.S[t] + model.agents.V1sus[t] + model.agents.V2sus[t]
                N_j = model.patches.N[t]
                pi_ij = model.patches.pi_ij
                tau_i = model.params.tau_i
                I_i = model.agents.Isym[t] + model.agents.Iasym[t]

                S_effective = (1 - tau_j) * S_j / N_j
                I_incoming = (pi_ij * tau_i * I_i / N_j).sum(axis=1)
                rate = beta_jt * S_effective * I_incoming
                probability = -np.expm1(-rate)
                denominator = 1 / (1 + beta_jt * (1 - tau_j) * S_j)
                hazard = (beta_jt * probability) / denominator
                model.patches.spatial_hazard[t] = hazard

            # coupling
            # https://www.mosaicmod.org/model-description.html#coupling-among-locations
            # y_t = I_i / N_j
            # y_bar_t = np.mean(y_t)
            # diff = y_t - y_bar_t
            # var_y_t = np.var(y_t)
            # model.patches.coupling[:, :] = diff[:, None] * diff[None, :] / np.sqrt(var_y_t[:, None] * var_y_t[None, :])

        return

    def plot(self, fig=None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Spatial Hazard by Location Over Time") if fig is None else fig

        plt.imshow(self.model.patches.spatial_hazard.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Spatial Hazard")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Spatial Hazard by Location Over Time"

        return
