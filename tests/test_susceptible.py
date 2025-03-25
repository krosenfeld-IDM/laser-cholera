import unittest
from datetime import datetime

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.utils import sim_duration


class TestSusceptible(unittest.TestCase):
    def test_susceptible_steadystate(self):
        ps = get_parameters(overrides=sim_duration())
        ps.b_jt *= 0  # turn off births
        ps.d_jt *= 0  # turn off deaths
        model = Model(parameters=ps)
        model.components = [Susceptible, Census]
        model.run()
        assert np.all(model.agents.S[-1] == model.agents.S[0]), "Susceptible: steady state not held."

        return

    def test_susceptible_deaths(self):
        ps = get_parameters(overrides=sim_duration())
        ps.b_jt *= 0  # turn off births
        model = Model(parameters=ps)
        model.components = [Susceptible, Census]
        model.run()
        assert np.all(model.agents.S[-1] < model.agents.S[0]), "Susceptible: deaths not occurring."

        return

    def test_susceptible_births(self):
        ps = get_parameters(overrides={"date_start": datetime(2025, 3, 24), "date_stop": datetime(2025, 4, 25), "nticks": 32})
        ps.d_jt *= 0  # turn off deaths
        model = Model(parameters=ps)
        model.components = [Susceptible, Census]
        model.run()
        assert np.all(model.agents.S[-1] > model.agents.S[0]), "Susceptible: births not occurring."

        return


if __name__ == "__main__":
    unittest.main()
