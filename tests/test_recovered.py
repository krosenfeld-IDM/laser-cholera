import unittest

import numpy as np

from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible  # we need this to put waning recovereds back into susceptibles
from laser_cholera.utils import test_duration


class TestRecovered(unittest.TestCase):
    def test_recovered_steadystate(self):
        ps = get_parameters(overrides=test_duration())
        ps.d_jt *= 0  # turn off deaths
        ps.epsilon = 0  # turn off waning natural immunity
        model = Model(parameters=ps)
        model.components = [Susceptible, Recovered]
        model.run()
        assert np.all(model.agents.R[-1] == model.agents.R[0]), "Recovered: steady state not held."

        return

    def test_recovered_deaths(self):
        ps = get_parameters(overrides=test_duration())
        # Move some people to the recovered state so we have enough to see deaths
        ps.R_j_initial += 10_000
        ps.S_j_initial -= ps.R_j_initial
        model = Model(parameters=ps)
        model.components = [Susceptible, Recovered]
        model.run()
        assert np.all(model.agents.R[-1] < model.agents.R[0]), "Recovered: deaths not occurring."

        return

    def test_recovered_waning_immunity(self):
        ps = get_parameters(overrides=test_duration())
        # Move some people to the recovered state so we have enough to see waning immunity
        ps.R_j_initial += 10_000
        ps.S_j_initial -= ps.R_j_initial
        ps.d_jt *= 0  # turn off deaths
        model = Model(parameters=ps)
        model.components = [Susceptible, Recovered]
        model.run()
        assert np.all(model.agents.R[-1] < model.agents.R[0]), "Recovered: waning immunity not occurring."

        return


if __name__ == "__main__":
    unittest.main()
