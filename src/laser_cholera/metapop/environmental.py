import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import beta


class Environmental:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "patches"), "Environmental: model needs to have a 'patches' attribute."

        model.patches.add_vector_property("W", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        assert hasattr(model, "params"), "Environmental: model needs to have a 'params' attribute."
        assert hasattr(model.params, "psi_jt"), (
            "Environmental: model params needs to have a 'psi_jt' (environmental contamination rate) parameter."
        )
        psi = model.params.psi_jt  # convenience
        # TODO - use newer laser_core with add_array_property and psi.shape
        model.patches.add_vector_property("delta_jt", length=psi.shape[0], dtype=np.float32, default=0.0)

        assert "decay_days_short" in model.params, (
            "Environmental: model params needs to have a 'decay_days_short' (maximum environmental decay) parameter."
        )
        assert "decay_days_long" in model.params, (
            "Environmental: model params needs to have a 'decay_days_long' (minimum environmental decay) parameter."
        )
        assert "decay_shape_1" in self.model.params, (
            "Environmental: model params needs to have a 'decay_shape_1' (beta function parameter 1) parameter."
        )
        assert "decay_shape_2" in self.model.params, (
            "Environmental: model params needs to have a 'decay_shape_2' (beta function parameter 2) parameter."
        )

        fast_decay = 1.0 / model.params.decay_days_short
        slow_decay = 1.0 / model.params.decay_days_long
        model.patches.delta_jt[:, :] = beta.cdf(
            fast_decay + model.params.psi_jt * (slow_decay - fast_decay), a=model.params.decay_shape_1, b=model.params.decay_shape_2
        )

        return

    def check(self):
        assert hasattr(self.model, "agents"), "Environmental: model needs to have a 'agents' attribute."
        assert hasattr(self.model.agents, "Isym"), "Environmental: model agents needs to have a 'Isym' (symptomatic) attribute."
        assert hasattr(self.model.agents, "Iasym"), "Environmental: model agents needs to have a 'Iasym' (asymptomatic) attribute."
        assert "zeta_1" in self.model.params, "Environmental: model params needs to have a 'zeta_1' (symptomatic shedding rate) parameter."
        assert "zeta_2" in self.model.params, "Environmental: model params needs to have a 'zeta_2' (asymptomatic shedding rate) parameter."
        assert "theta_j" in self.model.params, (
            "Environmental: model params needs to have a 'theta_j' (fraction of population with WASH) attribute."
        )

        return

    def __call__(self, model, tick: int) -> None:
        W = model.patches.W[tick]
        Wnext = model.patches.W[tick + 1]
        Wnext[:] = W

        Isym = model.agents.Isym[tick]
        Iasym = model.agents.Iasym[tick]

        # -decay
        # Use np.minimum() to make sure we don't go negative
        decay = np.minimum(model.prng.poisson(model.patches.delta_jt[tick] * W), W).astype(Wnext.dtype)
        Wnext -= decay

        # +shedding from Isymptomatic
        shedding_sym = model.prng.poisson(model.params.zeta_1 * Isym).astype(Wnext.dtype)
        Wnext += (1 - model.params.theta_j) * shedding_sym

        # +shedding from Iasymptomatic
        shedding_asym = model.prng.poisson(model.params.zeta_2 * Iasym).astype(Wnext.dtype)
        Wnext += (1 - model.params.theta_j) * shedding_asym

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Reservoir") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.W[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Environmental Reservoir")
        plt.legend()

        yield "Environmental Reservoir"
        return
