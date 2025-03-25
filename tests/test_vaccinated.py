import unittest

import numpy as np

from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.metapop.vaccinated import Vaccinated
from laser_cholera.utils import test_duration


class TestVaccinated(unittest.TestCase):
    # Test initial distribution of vaccinated agents - realistic phi_1 and phi_2
    def test_initial_distribution_realistic_phi(self):
        ps = get_parameters(overrides=test_duration())
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution
        assert np.all(model.agents.V1imm[0] == np.round(model.params.phi_1 * model.params.V1_j_initial)), (
            "V1_imm: initial distribution not correct."
        )
        assert np.all(model.agents.V1sus[0] == (model.params.V1_j_initial - model.agents.V1imm[0])), (
            "V1_sus: initial distribution not correct."
        )
        assert np.all(model.agents.V1inf[0] == 0), "V1_inf: initial distribution not correct."
        assert np.all(model.agents.V1[0] == model.params.V1_j_initial), "V1: initial distribution not correct."
        assert np.all(model.agents.V2imm[0] == np.round(model.params.phi_2 * model.params.V2_j_initial)), (
            "V2_imm: initial distribution not correct."
        )
        assert np.all(model.agents.V2sus[0] == (model.params.V2_j_initial - model.agents.V2imm[0])), (
            "V2_sus: initial distribution not correct."
        )
        assert np.all(model.agents.V2inf[0] == 0), "V2_inf: initial distribution not correct."
        assert np.all(model.agents.V2[0] == model.params.V2_j_initial), "V2: initial distribution not correct."

    # Test initial distribution of vaccinated agents - phi_1 and phi_2 = 0
    def test_initial_distribution_phi_zero(self):
        ps = get_parameters(overrides=test_duration())
        ps.phi_1 = 0
        ps.phi_2 = 0
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution
        assert np.all(model.agents.V1imm[0] == 0), "V1_imm: initial distribution not correct."
        assert np.all(model.agents.V1sus[0] == model.params.V1_j_initial), "V1_sus: initial distribution not correct."
        assert np.all(model.agents.V1inf[0] == 0), "V1_inf: initial distribution not correct."
        assert np.all(model.agents.V1[0] == model.params.V1_j_initial), "V1: initial distribution not correct."
        assert np.all(model.agents.V2imm[0] == 0), "V2_imm: initial distribution not correct."
        assert np.all(model.agents.V2sus[0] == model.params.V2_j_initial), "V2_sus: initial distribution not correct."
        assert np.all(model.agents.V2inf[0] == 0), "V2_inf: initial distribution not correct."
        assert np.all(model.agents.V2[0] == model.params.V2_j_initial), "V2: initial distribution not correct."

    # Test initial distribution of vaccinated agents - phi_1 and phi_2 = 1
    def test_initial_distribution_phi_one(self):
        ps = get_parameters(overrides=test_duration())
        ps.phi_1 = 1
        ps.phi_2 = 1
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution
        assert np.all(model.agents.V1imm[0] == model.params.V1_j_initial), "V1_imm: initial distribution not correct."
        assert np.all(model.agents.V1sus[0] == 0), "V1_sus: initial distribution not correct."
        assert np.all(model.agents.V1inf[0] == 0), "V1_inf: initial distribution not correct."
        assert np.all(model.agents.V1[0] == model.params.V1_j_initial), "V1: initial distribution not correct."
        assert np.all(model.agents.V2imm[0] == model.params.V2_j_initial), "V2_imm: initial distribution not correct."
        assert np.all(model.agents.V2sus[0] == 0), "V2_sus: initial distribution not correct."
        assert np.all(model.agents.V2inf[0] == 0), "V2_inf: initial distribution not correct."
        assert np.all(model.agents.V2[0] == model.params.V2_j_initial), "V2: initial distribution not correct."

    # Steady state
    def test_vaccinated_steadystate(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] == model.agents.V1imm[0]), "V1 immunized: steady state not held."
        assert np.all(model.agents.V1sus[-1] == model.agents.V1sus[0]), "V1 susceptible: steady state not held."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] == model.agents.V1[0]), "V1 total: steady state not held."
        assert np.all(model.agents.V2imm[-1] == model.agents.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.agents.V2sus[-1] == model.agents.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] == model.agents.V2[0]), "V2 total: steady state not held."

        return

    # Non-disease mortality
    def test_vaccinated_non_disease_deaths(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 10  # inflate non-disease death rate
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] < model.agents.V1imm[0]), "V1 immunized: non-disease deaths not occurring."
        assert np.all(model.agents.V1sus[-1] < model.agents.V1sus[0]), "V1 susceptible: non-disease deaths not occurring."
        # There is no infection, so V1inf == 0
        # assert np.all(model.agents.V1inf[-1] < model.agents.V1inf[0]), "V1 infected: non-disease deaths not occurring."
        assert np.all(model.agents.V1[-1] < model.agents.V1[0]), "V1 total: non-disease deaths not occurring."
        assert np.all(model.agents.V2imm[-1] < model.agents.V2imm[0]), "V2 immunized: non-disease deaths not occurring."
        assert np.all(model.agents.V2sus[-1] < model.agents.V2sus[0]), "V2 susceptible: non-disease deaths not occurring."
        # There is no infection, so V2inf == 0
        # assert np.all(model.agents.V2inf[-1] < model.agents.V2inf[0]), "V2 infected: non-disease deaths not occurring."
        assert np.all(model.agents.V2[-1] < model.agents.V2[0]), "V2 total: non-disease deaths not occurring."

        return

    # Waning immunity - one dose
    def test_vaccinated_waning_one_dose_immunity(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        # leave this on: ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] < model.agents.V1imm[0]), "V1 immunized: missing waning immunity."
        assert np.all(model.agents.V1sus[-1] > model.agents.V1sus[0]), "V1 susceptible: missing waning immunity."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: should not change."
        assert np.all(model.agents.V1[-1] == model.agents.V1[0]), "V1 total: should not change."
        assert np.all(model.agents.V2imm[-1] == model.agents.V2imm[0]), "V2 immunized: should not change."
        assert np.all(model.agents.V2sus[-1] == model.agents.V2sus[0]), "V2 susceptible: should not change."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: should not change."
        assert np.all(model.agents.V2[-1] == model.agents.V2[0]), "V2 total: should not change."

        return

    # Waning immunity - two doses
    def test_vaccinated_steadystate(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        # leave this on: ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        # Simulate a large vaccinated population
        ps.V1_j_initial += 10_000
        ps.V2_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] == model.agents.V1imm[0]), "V1 immunized: should not change."
        assert np.all(model.agents.V1sus[-1] == model.agents.V1sus[0]), "V1 susceptible: should not change."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: should not change."
        assert np.all(model.agents.V1[-1] == model.agents.V1[0]), "V1 total: should not change."
        assert np.all(model.agents.V2imm[-1] < model.agents.V2imm[0]), "V2 immunized: missing waning immunity."
        assert np.all(model.agents.V2sus[-1] > model.agents.V2sus[0]), "V2 susceptible: missing waning immunity."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: should not change."
        assert np.all(model.agents.V2[-1] == model.agents.V2[0]), "V2 total: should not change."

        return

    # Newly vaccinated - one dose, realistic efficacy
    def test_vaccinated_one_dose_vaccination_realistic_efficacy(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt[0:32] = 100
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        # Put any vaccinated folks back into the susceptible population
        ps.S_j_initial += ps.V1_j_initial + ps.V2_j_initial
        ps.V1_j_initial[:] = 0
        ps.V2_j_initial[:] = 0
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] > model.agents.V1imm[0]), "V1 immunized: missing newly vaccinated people."
        assert np.all(model.agents.V1sus[-1] > model.agents.V1sus[0]), "V1 susceptible: missing newly vaccinated people."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] > model.agents.V1[0]), "V1 total: missing newly vaccinated people."
        assert np.all(model.agents.V2imm[-1] == model.agents.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.agents.V2sus[-1] == model.agents.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] == model.agents.V2[0]), "V2 total: steady state not held."

        return

    # Newly vaccinated - one dose, perfect efficacy
    def test_vaccinated_one_dose_vaccination_perfect_efficacy(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt[0:32] = 100
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        ps.phi_1 = 1.0
        # Put any vaccinated folks back into the susceptible population
        ps.S_j_initial += ps.V1_j_initial + ps.V2_j_initial
        ps.V1_j_initial[:] = 0
        ps.V2_j_initial[:] = 0
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] > model.agents.V1imm[0]), "V1 immunized: missing newly vaccinated people."
        assert np.all(model.agents.V1sus[-1] == model.agents.V1sus[0]), "V1 susceptible: should not have newly vaccinated people."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] > model.agents.V1[0]), "V1 total: missing newly vaccinated people."
        assert np.all(model.agents.V2imm[-1] == model.agents.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.agents.V2sus[-1] == model.agents.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] == model.agents.V2[0]), "V2 total: steady state not held."

        return

    # Newly vaccinated - one dose, no efficacy
    def test_vaccinated_one_dose_vaccination_no_efficacy(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt[0:32] = 100
        ps.nu_2_jt *= 0  # turn off second dose vaccination
        ps.phi_1 = 0.0
        # Put any vaccinated folks back into the susceptible population
        ps.S_j_initial += ps.V1_j_initial + ps.V2_j_initial
        ps.V1_j_initial[:] = 0
        ps.V2_j_initial[:] = 0
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] == model.agents.V1imm[0]), "V1 immunized: should not have newly vaccinated people."
        assert np.all(model.agents.V1sus[-1] > model.agents.V1sus[0]), "V1 susceptible: missing newly vaccinated people."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] > model.agents.V1[0]), "V1 total: missing newly vaccinated people."
        assert np.all(model.agents.V2imm[-1] == model.agents.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.agents.V2sus[-1] == model.agents.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] == model.agents.V2[0]), "V2 total: steady state not held."

        return

    # Newly vaccinated - two doses, realistic efficacy
    def test_vaccinated_two_dose_vaccination_realistic_efficacy(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt[0:32] = 100
        # Put any vaccinated folks back into the susceptible population
        ps.S_j_initial += ps.V1_j_initial + ps.V2_j_initial
        ps.V1_j_initial[:] = 10_000
        ps.V2_j_initial[:] = 0
        ps.S_j_initial -= ps.V1_j_initial
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] < model.agents.V1imm[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.agents.V1sus[-1] < model.agents.V1sus[0]), "V1 susceptible: should be missing newly vaccinated people."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] < model.agents.V1[0]), "V1 total: should be missing newly vaccinated people."
        assert np.all(model.agents.V2imm[-1] > model.agents.V2imm[0]), "V2 immunized: missing newly vaccinated people."
        assert np.all(model.agents.V2sus[-1] > model.agents.V2sus[0]), "V2 susceptible: missing newly vaccinated people."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] > model.agents.V2[0]), "V2 total: missing newly vaccinated people."

        return

    # Newly vaccinated - two doses, perfect efficacy
    def test_vaccinated_two_dose_vaccination_perfect_efficacy(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt[0:32] = 100
        ps.phi_2 = 1.0
        # Put any vaccinated folks back into the susceptible population
        ps.S_j_initial += ps.V1_j_initial + ps.V2_j_initial
        ps.V1_j_initial[:] = 10_000
        ps.V2_j_initial[:] = 0
        ps.S_j_initial -= ps.V1_j_initial
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] < model.agents.V1imm[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.agents.V1sus[-1] < model.agents.V1sus[0]), "V1 susceptible: should be missing newly vaccinated people."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] < model.agents.V1[0]), "V1 total: should be missing newly vaccinated people."
        assert np.all(model.agents.V2imm[-1] > model.agents.V2imm[0]), "V2 immunized: missing newly vaccinated people."
        assert np.all(model.agents.V2sus[-1] == model.agents.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] > model.agents.V2[0]), "V2 total: missing newly vaccinated people."

        return

    # Newly vaccinated - two doses, no efficacy
    def test_vaccinated_two_dose_vaccination_no_efficacy(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.omega_1 = 0  # turn off waning immunity - one dose
        ps.omega_2 = 0  # turn off waning immunity - two dose
        ps.nu_1_jt *= 0  # turn off first dose vaccination
        ps.nu_2_jt[0:32] = 100
        ps.phi_2 = 0.0
        # Put any vaccinated folks back into the susceptible population
        ps.S_j_initial += ps.V1_j_initial + ps.V2_j_initial
        ps.V1_j_initial[:] = 10_000
        ps.V2_j_initial[:] = 0
        ps.S_j_initial -= ps.V1_j_initial
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Vaccinated]  # , Infectious, Recovered]
        model.run()
        assert np.all(model.agents.V1imm[-1] < model.agents.V1imm[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.agents.V1sus[-1] < model.agents.V1sus[0]), "V1 susceptible: should be missing newly vaccinated people."
        assert np.all(model.agents.V1inf[-1] == model.agents.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.agents.V1[-1] < model.agents.V1[0]), "V1 total: should be missing newly vaccinated people."
        assert np.all(model.agents.V2imm[-1] == model.agents.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.agents.V2sus[-1] > model.agents.V2sus[0]), "V2 susceptible: missing newly vaccinated people."
        assert np.all(model.agents.V2inf[-1] == model.agents.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.agents.V2[-1] > model.agents.V2[0]), "V2 total: missing newly vaccinated people."

        return


if __name__ == "__main__":
    unittest.main()
