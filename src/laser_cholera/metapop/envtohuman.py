"""Environmental transmission rate."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure


class EnvToHuman:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "EnvToHuman: model needs to have a 'agents' attribute."
        model.patches.add_vector_property("PSI", length=model.params.nticks + 1, dtype=np.float32, default=np.float32(0.0))

        assert hasattr(model, "params"), "EnvToHuman: model needs to have a 'params' attribute."
        assert hasattr(
            model.params, "psi_jt"
        ), "EnvToHuman: model params needs to have a 'psi_jt' (environmental contamination rate) parameter."

        model.patches.add_vector_property("beta_env", length=365, dtype=np.float32, default=0.0)
        assert model.patches.beta_env.shape[1] == model.params.psi_jt.shape[0]
        assert model.patches.beta_env.shape[1] == model.params.beta_j0_env.shape[0]
        dimension = model.patches.beta_env.shape[1]
        one_year = model.params.psi_jt[:, 0:365]
        psi_bar = one_year.mean(axis=1).reshape((dimension, -1))
        model.patches.beta_env[:, :] = (model.params.beta_j0_env.reshape((dimension, -1)) * (1 + (one_year - psi_bar) / psi_bar)).T

        return

    def check(self):
        assert hasattr(self.model.agents, "S"), "EnvToHuman: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "I"), "EnvToHuman: model agents needs to have a 'I' (infected) attribute."
        assert hasattr(self.model, "patches"), "EnvToHuman: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "W"), "EnvToHuman: model patches needs to have a 'W' (environmental) attribute."
        assert hasattr(
            self.model.params, "tau_i"
        ), "HumanToHuman: model.params needs to have a 'tau_i' (emmigration probability) parameter."
        assert hasattr(
            self.model.params, "theta_j"
        ), "EnvToHuman: model params needs to have a 'theta_j' (fraction of population with WASH) attribute."
        assert hasattr(
            self.model.params, "kappa"
        ), "EnvToHuman: model params needs to have a 'kappa' (environmental transmission rate) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        Pprime = model.patches.PSI[tick + 1]
        S = model.agents.S[tick]
        W = model.patches.W[tick]
        Sprime = model.agents.S[tick + 1]
        Iprime = model.agents.I[tick + 1]

        Pprime[:] = model.patches.beta_env[tick] * (S * (1 - model.params.tau_i)) * (1 - model.params.theta_j) * W
        Pprime /= model.params.kappa + W

        newly_infected = model.prng.poisson(Pprime).astype(Sprime.dtype)
        Sprime -= newly_infected
        Iprime += newly_infected

        return

    @staticmethod
    def test():
        class Model:
            def __init__(self, psi_jt, beta_j0_env, tau_i, theta_j, kappa, wfactor=1.0, beta_factor=1.0):
                self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005

                self.agents = LaserFrame(4)
                self.agents.add_vector_property("S", length=8, dtype=np.uint32, default=0)
                self.agents.add_vector_property("I", length=8, dtype=np.uint32, default=0)
                self.agents.S[0] = self.agents.S[1] = [250, 2_500, 25_000, 250_000]  # 25%
                self.agents.I[0] = self.agents.I[1] = [100, 1_000, 10_000, 100_000]  # 10%

                self.patches = LaserFrame(4)
                self.patches.add_vector_property("W", length=8, dtype=np.float32, default=0.0)
                self.patches.W[0, :] = 500_000.0 / 3.0
                self.patches.W[0] *= wfactor
                beta_j0_env *= beta_factor

                self.params = PropertySet(
                    {
                        "psi_jt": psi_jt,
                        "beta_j0_env": beta_j0_env,
                        "tau_i": tau_i,
                        "theta_j": theta_j,
                        "kappa": kappa,
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

        component = EnvToHuman(
            baseline := Model(
                psi_jt=psi_jt,
                beta_j0_env=np.array([0.4, 0.4, 0.4, 0.4]),
                tau_i=np.array([1.2521719e-03, 4.1985715e-04, 2.0205112e-02, 9.2022895e-03]),  # 1000x some actual values
                theta_j=np.array([0.5604815, 0.7254393, 0.6573358, 0.6017253]),
                kappa=500_000,
            )
        )
        component.check()
        component(baseline, 0)
        assert np.all(
            baseline.agents.S[1] < baseline.agents.S[0]
        ), f"Some susceptible populations didn't decrease with environmental transmission.\n\t{baseline.agents.S[0]}\n\t{baseline.agents.S[1]}"
        assert np.all(
            baseline.agents.I[1] > baseline.agents.I[0]
        ), f"Some infected populations didn't increase with environmental transmission.\n\t{baseline.agents.I[0]}\n\t{baseline.agents.I[1]}"

        component = EnvToHuman(
            increased_tau := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i * (0.95 / baseline.params.tau_i),
                theta_j=baseline.params.theta_j,
                kappa=baseline.params.kappa,
            )
        )
        component.check()
        component(increased_tau, 0)
        assert np.all(
            increased_tau.agents.S[1] > baseline.agents.S[1]
        ), f"Expected larger remaining susceptible populations with higher tau_i (migrating fraction).\n\t{baseline.agents.S[1]=}\n\t{increased_tau.agents.S[1]=}"
        assert np.all(
            increased_tau.agents.I[1] < baseline.agents.I[1]
        ), f"Expected smaller infected populations with higher tau_i (migrating fraction).\n\t{baseline.agents.I[1]=}\n\t{increased_tau.agents.I[1]=}"

        component = EnvToHuman(
            zero_wash := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=baseline.params.theta_j * 0,
                kappa=baseline.params.kappa,
            )
        )
        component.check()
        component(zero_wash, 0)
        assert np.all(
            zero_wash.agents.S[1] < baseline.agents.S[1]
        ), f"Expected smaller remaining susceptible populations with lower theta_j (WASH fraction).\n\t{baseline.agents.S[1]=}\n\t{zero_wash.agents.S[1]=}"
        assert np.all(
            zero_wash.agents.I[1] > baseline.agents.I[1]
        ), f"Expected larger infected populations with lower theta_j (WASH fraction).\n\t{baseline.agents.I[1]=}\n\t{zero_wash.agents.I[1]=}"

        component = EnvToHuman(
            high_wash := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=np.array([0.9, 0.9, 0.9, 0.9]),
                kappa=baseline.params.kappa,
            )
        )
        component.check()
        component(high_wash, 0)
        assert np.all(
            high_wash.agents.S[1] > baseline.agents.S[1]
        ), f"Expected larger remaining susceptible populations with higher theta_j (WASH fraction).\n\t{baseline.agents.S[1]=}\n\t{high_wash.agents.S[1]=}"
        assert np.all(
            high_wash.agents.I[1] < baseline.agents.I[1]
        ), f"Expected smaller infected populations with higher theta_j (WASH fraction).\n\t{baseline.agents.I[1]=}\n\t{high_wash.agents.I[1]=}"

        component = EnvToHuman(
            perfect_wash := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=np.array([1.0, 1.0, 1.0, 1.0]),
                kappa=baseline.params.kappa,
            )
        )
        component.check()
        component(perfect_wash, 0)
        assert np.all(
            perfect_wash.agents.S[1] == perfect_wash.agents.S[0]
        ), f"Expected no environmental transmission with theta_j (WASH fraction) = 1.0.\n\t{perfect_wash.agents.S[0]=}\n\t{perfect_wash.agents.S[1]=}"
        assert np.all(
            perfect_wash.agents.I[1] == perfect_wash.agents.I[0]
        ), f"Expected no environmental transmission with theta_j (WASH fraction) = 1.0.\n\t{perfect_wash.agents.I[0]=}\n\t{perfect_wash.agents.I[1]=}"

        component = EnvToHuman(
            smaller_kappa := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=baseline.params.theta_j,
                kappa=baseline.params.kappa / 8.0,
            )
        )
        component.check()
        component(smaller_kappa, 0)
        assert np.all(
            smaller_kappa.agents.S[1] < baseline.agents.S[1]
        ), f"Expected smaller remaining susceptible populations with lower kappa (50% probability concentration level).\n\t{baseline.agents.S[1]=}\n\t{smaller_kappa.agents.S[1]=}"
        assert np.all(
            smaller_kappa.agents.I[1] > baseline.agents.I[1]
        ), f"Expected larger infected populations with lower kappa (50% probability concentration level).\n\t{baseline.agents.I[1]=}\n\t{smaller_kappa.agents.I[1]=}"

        component = EnvToHuman(
            larger_kappa := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=baseline.params.theta_j,
                kappa=8.0 * baseline.params.kappa,
            )
        )
        component.check()
        component(larger_kappa, 0)
        assert np.all(
            larger_kappa.agents.S[1] > baseline.agents.S[1]
        ), f"Expected larger remaining susceptible populations with larger kappa (50% probability concentration level).\n\t{baseline.agents.S[1]=}\n\t{larger_kappa.agents.S[1]=}"
        assert np.all(
            larger_kappa.agents.I[1] < baseline.agents.I[1]
        ), f"Expected smaller infected populations with larger kappa (50% probability concentration level).\n\t{baseline.agents.I[1]=}\n\t{larger_kappa.agents.I[1]=}"

        component = EnvToHuman(
            smaller_w := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=baseline.params.theta_j,
                kappa=baseline.params.kappa,
                wfactor=0.125,
            )
        )
        component.check()
        component(smaller_w, 0)
        assert np.all(
            smaller_w.agents.S[1] > baseline.agents.S[1]
        ), f"Expected larger remaining susceptible populations with smaller W (environmental contagion).\n\t{baseline.agents.S[1]=}\n\t{smaller_w.agents.S[1]=}"
        assert np.all(
            smaller_w.agents.I[1] < baseline.agents.I[1]
        ), f"Expected smaller infected populations with smaller W (environmental contagion).\n\t{baseline.agents.I[1]=}\n\t{smaller_w.agents.I[1]=}"

        component = EnvToHuman(
            larger_w := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=baseline.params.theta_j,
                kappa=baseline.params.kappa,
                wfactor=8.0,
            )
        )
        component.check()
        component(larger_w, 0)
        assert np.all(
            larger_w.agents.S[1] < baseline.agents.S[1]
        ), f"Expected smaller remaining susceptible populations with larger W (environmental contagion).\n\t{baseline.agents.S[1]=}\n\t{larger_w.agents.S[1]=}"
        assert np.all(
            larger_w.agents.I[1] > baseline.agents.I[1]
        ), f"Expected larger infected populations with larger W (environmental contagion).\n\t{baseline.agents.I[1]=}\n\t{larger_w.agents.I[1]=}"

        component = EnvToHuman(
            smaller_beta := Model(
                psi_jt=baseline.params.psi_jt,
                beta_j0_env=baseline.params.beta_j0_env,
                tau_i=baseline.params.tau_i,
                theta_j=baseline.params.theta_j,
                kappa=baseline.params.kappa,
                beta_factor=0.03125,
            )
        )
        component.check()
        component(smaller_beta, 0)
        assert np.all(
            smaller_beta.agents.S[1] > baseline.agents.S[1]
        ), f"Expected larger remaining susceptible populations with smaller beta_env (seasonal factor).\n\t{baseline.agents.S[1]=}\n\t{smaller_beta.agents.S[1]=}"
        assert np.all(
            smaller_beta.agents.I[1] < baseline.agents.I[1]
        ), f"Expected smaller infected populations with smaller beta_env (seasonal factor).\n\t{baseline.agents.I[1]=}\n\t{smaller_beta.agents.I[1]=}"

        # component = EnvToHuman(
        #     larger_beta := Model(
        #         psi_jt=baseline.params.psi_jt,
        #         beta_j0_env=baseline.params.beta_j0_env,
        #         tau_i=baseline.params.tau_i,
        #         theta_j=baseline.params.theta_j,
        #         kappa=baseline.params.kappa,
        #         beta_factor=16.0,
        #     )
        # )
        # component.check()
        # component(larger_beta, 0)
        # assert np.all(
        #     larger_beta.agents.S[1] < baseline.agents.S[1]
        # ), f"Expected smaller remaining susceptible populations with larger beta_env (seasonal factor).\n\t{baseline.agents.S[1]=}\n\t{larger_beta.agents.S[1]=}"
        # assert np.all(
        #     larger_beta.agents.I[1] > baseline.agents.I[1]
        # ), f"Expected larger infected populations with larger beta_env (seasonal factor).\n\t{baseline.agents.I[1]=}\n\t{larger_beta.agents.I[1]=}"

        print("PASSED EnvToHuman.test()")
        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Environmental Transmission Rate in Patch {ipatch}")
        plt.plot(self.model.patches.PSI[:, ipatch], color="orange", label="Environmental Transmission Rate")
        plt.xlabel("Tick")

        yield
        return


if __name__ == "__main__":
    EnvToHuman.test()
    # plt.show()
