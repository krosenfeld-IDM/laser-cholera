import numpy as np


class DerivedValues:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "patches"), "DerivedValues: model needs to have a 'patches' attribute."
        assert hasattr(model, "params"), "DerivedValues: model needs to have a 'params' attribute."

        nPatches = len(model.params.location_id)
        model.patches.add_array_property("p_ijt", shape=(model.params.nticks, nPatches, nPatches), dtype=np.float32, default=0.0)
        model.patches.add_vector_property("spatial_hazard", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        model.patches.add_vector_property("r_effective", length=model.params.nticks + 1, dtype=np.float32, default=0.0)

        return

    def check(self):
        assert hasattr(self.model, "agents"), "DerivedValues: model needs to have an 'agents' attribute."

        return

    def __call__(self, model, tick: int) -> None:
        if tick == model.params.nticks - 1:
            # p_ijt - probability of spatial transmission
            # https://www.mosaicmod.org/model-description.html#the-probability-of-spatial-transmission
            # $p(i,j,t) = 1 - e^{-\beta^{hum}_{jt}(((1 - \tau_j) S_{jt}) / N_{jt}) \pi_{ij} \tau_i I_{it}}$
            for t in range(model.params.nticks):
                beta_jt = model.params.beta_j0_hum * (1.0 + model.params.beta_j_seasonality[t % 366, :])
                tau_j = model.params.tau_i
                S_j = model.agents.S[t]
                N_j = model.patches.N[t]
                pi_ij = model.params.pi_ij
                I_i = model.agents.Isym[t] + model.agents.Iasym[t]
                tau_i = model.params.tau_i

                S_effective = (1 - tau_j) * S_j / N_j
                I_outgoing = tau_i * I_i
                I_incoming = pi_ij * I_outgoing[:, None]
                pressure = S_effective * I_incoming
                rate = beta_jt * pressure

                # rate = beta_jt * ((1 - tau_j) * S_j / N_j) * pi_ij * tau_i * I_i
                probability = -np.expm1(-rate)
                model.patches.p_ijt[t, :, :] = probability

            # spatial_hazard
            # https://www.mosaicmod.org/model-description.html#the-spatial-hazard
            # $h(j,t) = \frac {\beta^{hum}_{jt} (1 - e^{-((1 - \tau_j) S_{jt} / N_{jt}) \sum_{\forall i \ne j} \pi_{ij} \tau_i (I_{it} / N_{it})})} {1/(1 + \beta^{hum}_{jt}(1 - \tau_j) S_{jt})}$
            for t in range(model.params.nticks):
                beta_jt = model.params.beta_j0_hum * (1.0 + model.params.beta_j_seasonality[t % 366, :])
                tau_j = model.params.tau_i
                S_j = model.agents.S[t]
                N_j = model.patches.N[t]
                pi_ij = model.params.pi_ij
                tau_i = model.params.tau_i
                I_i = model.agents.Isym[t] + model.agents.Iasym[t]

                S_effective = (1 - tau_j) * S_j / N_j
                I_incoming = (pi_ij * tau_i * I_i / N_j).sum(axis=1)
                rate = beta_jt * S_effective * I_incoming
                probability = -np.expm1(-rate)
                denominator = 1 / (1 + beta_jt * (1 - tau_j) * S_j)
                hazard = (beta_jt * probability) / denominator
                model.patches.spatial_hazard[t] = hazard

            # r_effective
            # https://www.mosaicmod.org/model-description.html#the-reproductive-number
            # $R_{jt} = \frac {I_{jt}} {\sum^t_{\Delta t=1}g(\Delta t) I_{j,t-\Delta t}}$
            # TODO - need g(Delta t) function

        return

    # def plot(self, fig=None):  # pragma: no cover
    #     _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

    #     plt.title("Derived Values")
    #     plt.xlabel("Tick")
    #     plt.legend()

    #     yield
    #     return
