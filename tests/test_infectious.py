import unittest

import numpy as np

from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.utils import test_duration


class TestInfectious(unittest.TestCase):
    # Test initial distribution of infectious agents - realistic sigma
    def test_initial_distribution_realistic_sigma(self):
        ps = get_parameters(overrides=test_duration())
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution
        assert np.all(model.agents.Isym[0] == np.round(model.params.sigma * model.params.I_j_initial)), (
            "I symptomatic: initial distribution not correct."
        )
        assert np.all(model.agents.Iasym[0] == (ps.I_j_initial - model.agents.Isym[0])), "I asymptomatic: initial distribution not correct."

    # Test initial distribution of infectious agents - sigma = 0
    def test_initial_distribution_sigma_zero(self):
        ps = get_parameters(overrides=test_duration())
        ps.sigma = 0
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution
        assert np.all(model.agents.Isym[0] == 0), "I symptomatic: initial distribution not correct."
        assert np.all(model.agents.Iasym[0] == ps.I_j_initial), "I asymptomatic: initial distribution not correct."

    # Test initial distribution of infectious agents - sigma = 1
    def test_initial_distribution_sigma_one(self):
        ps = get_parameters(overrides=test_duration())
        ps.sigma = 1
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution
        assert np.all(model.agents.Isym[0] == ps.I_j_initial), "I symptomatic: initial distribution not correct."
        assert np.all(model.agents.Iasym[0] == 0), "I asymptomatic: initial distribution not correct."

    # Steady state, d_jt = 0, mu = 0, gamma_1 = 0, gamma_2 = 0, iota = 0
    def test_infectous_steadystate(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.mu_jt *= 0  # turn off disease deaths
        ps.gamma_1 = 0  # turn off symptomatic recovery
        ps.gamma_2 = 0  # turn off asymptomatic recovery
        ps.iota = 0  # turn off progression from exposed to infectious
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        model.run()
        assert np.all(model.agents.Isym[-1] == model.agents.Isym[0]), "I symptomatic: steady state not held."
        assert np.all(model.agents.Iasym[-1] == model.agents.Iasym[0]), "I asymptomatic: steady state not held."

        return

    # Non-disease deaths
    def test_infectious_non_disease_deaths(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 10  # inflate non-disease death rate
        ps.mu_jt *= 0  # turn off disease deaths
        ps.gamma_1 = 0  # turn off symptomatic recovery
        ps.gamma_2 = 0  # turn off asymptomatic recovery
        ps.iota = 0  # turn off progression from exposed to infectious
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        model.run()
        assert np.all(model.agents.Isym[-1] < model.agents.Isym[0]), "I symptomatic: non-disease-deaths not occurring."
        assert np.all(model.agents.Iasym[-1] < model.agents.Iasym[0]), "I asymptomatic: non-disease-deaths not occurring."

        return

    # Disease deaths
    def test_infectious_disease_deaths(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off non-disease deaths
        ps.mu_jt *= 10  # inflate disease death rate
        ps.gamma_1 = 0  # turn off symptomatic recovery
        ps.gamma_2 = 0  # turn off asymptomatic recovery
        ps.iota = 0  # turn off progression from exposed to infectious
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        model.run()
        assert np.all(model.agents.Isym[-1] < model.agents.Isym[0]), "I symptomatic: disease deaths not occurring."
        assert np.all(model.agents.Iasym[-1] == model.agents.Iasym[0]), (
            "I asymptomatic: disease deaths occurring in the asymptomatic population."
        )

        return

    # Symptomatic Recovery
    def test_infectious_symptomatic_recovery(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off non-disease deaths
        ps.mu_jt *= 0  # turn off disease deaths
        # leave this on: ps.gamma_1 = 0 # turn off symptomatic recovery
        ps.gamma_2 = 0  # turn off asymptomatic recovery
        ps.iota = 0  # turn off progression from exposed to infectious
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        model.run()
        assert np.all(model.agents.Isym[-1] < model.agents.Isym[0]), "I symptomatic: recovery not occurring."
        assert np.all(model.agents.Iasym[-1] == model.agents.Iasym[0]), "I asymptomatic: recovery occurring in the asymptomatic population."

        return

    # Asymptomatic Recovery
    def test_infectious_asymptomatic_recovery(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off non-disease deaths
        ps.mu_jt *= 0  # turn off disease deaths
        ps.gamma_1 = 0  # turn off symptomatic recovery
        # leave this on: ps.gamma_2 = 0 # turn off asymptomatic recovery
        ps.iota = 0  # turn off progression from exposed to infectious
        # Simulate a large infectious population
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        model.run()
        assert np.all(model.agents.Isym[-1] == model.agents.Isym[0]), "I symptomatic: recovery occurring in the symptomatic population."
        assert np.all(model.agents.Iasym[-1] < model.agents.Iasym[0]), "I asymptomatic: recovery not occurring."

        return

    # Newly infectious
    def test_infectious_newly_infectious(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off non-disease deaths
        ps.mu_jt *= 0  # turn off disease deaths
        ps.gamma_1 = 0  # turn off symptomatic recovery
        ps.gamma_2 = 0  # turn off asymptomatic recovery
        # leave this on: ps.iota = 0 # turn off progression from exposed to infectious
        # Simulate large exposed and infectious populations
        ps.E_j_initial += 10_000
        ps.I_j_initial += 10_000
        ps.S_j_initial -= 20_000
        model = Model(parameters=ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered]
        model.run()
        assert np.all(model.agents.Isym[-1] > model.agents.Isym[0]), "I symptomatic: progression from exposed to infectious not occurring."
        assert np.all(model.agents.Iasym[-1] > model.agents.Iasym[0]), (
            "I asymptomatic: progression from exposed to infectious not occurring."
        )

        return


if __name__ == "__main__":
    unittest.main()
