import unittest
from datetime import datetime

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.humantohuman import HumanToHuman
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.utils import sim_duration

_ONEYEAR = sim_duration(datetime(2024, 1, 1), datetime(2024, 12, 31))


class TestHumanToHuman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = get_parameters(overrides=_ONEYEAR)
        cls.model = Model(params)
        cls.model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        cls.model.run()

    # What if tau_i = 0? (no emmigration)
    def test_humantohuman_tau_i_zero(self):
        params = get_parameters(overrides=_ONEYEAR)
        iNigeria = params.location_name.index("NGA")
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        params.I_j_initial[iNigeria] = 100
        params.tau_i[:] = 0
        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model.run()
        for index in range(len(params.location_name)):
            if index != iNigeria:
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
            else:
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."

        return

    # What if tau_i = 1? (total emmigration, no local transmission)
    def test_humantohuman_tau_i_one(self):
        params = get_parameters(overrides=_ONEYEAR)
        iNigeria = params.location_name.index("NGA")
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        params.I_j_initial[iNigeria] = 100
        params.pi_ij[:, iNigeria] = 0.0  # No immigration into Nigeria
        params.tau_i[iNigeria] = 1.0  # No local transmission in Nigeria
        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model.run()
        assert np.all(model.patches.Lambda[:, iNigeria] == 0), "HumanToHuman: tau_i = 1, Lambda_jt not zero in Nigeria."

        return

    # Special cases for pi_ij? (no connectivity)
    def test_humantohuman_pi_ij_zero(self):
        params = get_parameters(overrides=_ONEYEAR)
        iNigeria = params.location_name.index("NGA")
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        params.I_j_initial[iNigeria] = 100
        params.pi_ij *= 0.0  # No migration at all
        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model.run()
        for index in range(len(params.location_name)):
            if index != iNigeria:
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
            else:
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."

        return

    # Special cases for alpha_1?
    # Special cases for beta_j0_hum?
    # Special cases for beta_j_seasonality?
    # Special cases for alpha_2?


if __name__ == "__main__":
    unittest.main()
