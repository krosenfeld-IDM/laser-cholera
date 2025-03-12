from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class Environmental:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "patches"), "Environmental: model needs to have a 'patches' attribute."

        model.patches.add_vector_property("W", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        raise AssertionError("Need to check model.params.psi_jt shape.")
        model.patches.add_vector_property("delta", length=365, dtype=np.float32, default=0.0)

        assert hasattr(model, "params"), "Environmental: model needs to have a 'params' attribute."
        assert hasattr(model.params, "psi_jt"), (
            "Environmental: model params needs to have a 'psi_jt' (environmental contamination rate) parameter."
        )
        assert hasattr(model.params, "delta_max"), "Environmental: model params needs to have a 'delta_max' (suitability decay) parameter."
        assert hasattr(model.params, "delta_min"), "Environmental: model params needs to have a 'delta_min' (suitability decay) parameter."

        model.patches.delta[:, :] = (model.params.delta_min + model.params.psi_jt * (model.params.delta_max - model.params.delta_min)).T

        return

    def check(self):
        assert hasattr(self.model, "agents"), "Environmental: model needs to have a 'agents' attribute."
        assert hasattr(self.model.agents, "Isym"), "Environmental: model agents needs to have a 'Isym' (symptomatic) attribute."
        assert hasattr(self.model.agents, "Iasym"), "Environmental: model agents needs to have a 'Iasym' (asymptomatic) attribute."
        assert "zeta_1" in self.model.params, "Environmental: model params needs to have a 'zeta_1' (symptomatic shedding rate) parameter."
        assert "zeta_2" in self.model.params, "Environmental: model params needs to have a 'zeta_2' (asymptomatic shedding rate) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        W = model.patches.W[tick]
        Wnext = model.patches.W[tick + 1]
        Wnext[:] = W

        Isym = model.agents.Isym[tick]
        Iasym = model.agents.Iasym[tick]

        # -decay
        decay = np.poisson(model.patches.delta[tick] * W).astype(Wnext.dtype)
        Wnext -= decay

        # +shedding from Isymptomatic
        shedding_sym = np.poisson(model.params.zeta_1 * Isym[tick]).astype(Wnext.dtype)
        Wnext += shedding_sym

        # +shedding from Iasymptomatic
        shedding_asym = np.poisson(model.params.zeta_2 * Iasym[tick]).astype(Wnext.dtype)
        Wnext += shedding_asym

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, psi_jt, delta_max, delta_min, zeta):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
                self.agents = LaserFrame(4)
                self.agents.add_vector_property("I", length=8, dtype=np.int32, default=0)
                self.agents.I[0] = [100, 1_000, 10_000, 100_000]
                self.patches = LaserFrame(4)
                self.params = PropertySet(
                    {
                        "psi_jt": psi_jt,
                        "delta_max": delta_max,
                        "delta_min": delta_min,
                        "zeta": zeta,
                        "nticks": 8,
                    }
                )

        psi_jt = np.zeros((4, 365))
        psi_jt[0, 0:8] = [0.5106762, 0.5108125, 0.5114274, 0.5124948, 0.5139883, 0.5158816, 0.5181482, 0.5207615]
        psi_jt[1, 0:8] = [0.0667797, 0.0667555, 0.0667324, 0.0667131, 0.0667009, 0.0666988, 0.0667097, 0.0667367]
        psi_jt[2, 0:8] = [0.6080956, 0.6040341, 0.5999001, 0.5956825, 0.5913696, 0.5869501, 0.5824124, 0.5777447]
        psi_jt[3, 0:8] = [0.6868838, 0.6866003, 0.6863385, 0.6860761, 0.6857905, 0.6854592, 0.6850596, 0.6845689]
        psi_jt[0, 8:] = psi_jt[0, 0:8].mean()
        psi_jt[1, 8:] = psi_jt[1, 0:8].mean()
        psi_jt[2, 8:] = psi_jt[2, 0:8].mean()
        psi_jt[3, 8:] = psi_jt[3, 0:8].mean()

        component = Environmental(baseline := Model(psi_jt=psi_jt, delta_max=0.0111111, delta_min=0.3333333, zeta=0.5))
        component.check()
        component(baseline, 0)

        component = Environmental(no_decay := Model(psi_jt=baseline.params.psi_jt, delta_max=0.0, delta_min=0.0, zeta=baseline.params.zeta))
        component.check()
        component(no_decay, 0)
        assert np.all(no_decay.patches.W[1] > baseline.patches.W[0]), (
            f"Expected more environmental contagion with no decay.\n\t{no_decay.patches.W[1]}\n\t{baseline.patches.W[0]}"
        )

        component = Environmental(
            increased_shedding := Model(
                psi_jt=baseline.params.psi_jt,
                delta_max=baseline.params.delta_max,
                delta_min=baseline.params.delta_min,
                zeta=baseline.params.zeta * 8.0,
            )
        )
        component.check()
        component(increased_shedding, 0)
        assert np.all(increased_shedding.patches.W[1] > baseline.patches.W[0]), (
            f"Expected more environmental contagion with increased shedding.\n\t{increased_shedding.patches.W[1]}\n\t{baseline.patches.W[0]}"
        )

        component = Environmental(
            more_infected := Model(
                psi_jt=baseline.params.psi_jt,
                delta_max=baseline.params.delta_max,
                delta_min=baseline.params.delta_min,
                zeta=baseline.params.zeta,
            )
        )
        more_infected.agents.I[0] *= 2
        component.check()
        component(more_infected, 0)
        assert np.all(more_infected.patches.W[1] > baseline.patches.W[0]), (
            f"Expected more environmental contagion with more infected agents.\n\t{more_infected.patches.W[1]}\n\t{baseline.patches.W[0]}"
        )

        print("PASSED Environmental.test()")
        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Environmental Reservoir")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.W[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return


if __name__ == "__main__":
    Environmental.test()
    # plt.show()
