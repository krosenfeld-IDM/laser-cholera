import unittest

import numpy as np
import pytest

from laser_cholera.likelihood import calc_log_likelihood_beta
from laser_cholera.likelihood import calc_log_likelihood_binomial
from laser_cholera.likelihood import calc_log_likelihood_gamma
from laser_cholera.likelihood import calc_log_likelihood_negbin
from laser_cholera.likelihood import calc_log_likelihood_normal
from laser_cholera.likelihood import calc_log_likelihood_poisson
from laser_cholera.likelihood import get_model_likelihood
from laser_cholera.metapop.model import run_model


class LikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the model with default parameters
        cls.model = run_model(None)
        cls.obs_cases = cls.model.params.reported_cases
        cls.est_cases = cls.model.patches.incidence[1:].T  # ignore t=0 (initial conditions)
        cls.obs_deaths = cls.model.params.reported_deaths
        cls.est_deaths = cls.model.patches.disease_deaths[1:].T  # ignore t=0 (initial conditions)
        cls.n_locations, cls.n_steps = cls.obs_cases.shape

        cls.n_locations, cls.n_steps = cls.obs_cases.shape

    def test_get_model_likelihood(self):
        # test cases - poisson, deaths - poisson
        # test cases - poisson, deaths - normal (gaussian)
        # test cases - normal (gaussian), deaths - poisson
        # test cases - normal (gaussian), deaths - normal (gaussian)

        return

    def test_get_model_likelihood_invalid_inputs(self):
        # obs_cases isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=[list(row) for row in self.obs_cases],
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
            )

        # est_cases isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=[list(row) for row in self.est_cases],
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
            )

        # obs_deaths isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=[list(row) for row in self.obs_deaths],
                est_deaths=self.est_deaths,
            )

        # est_deaths isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=[list(row) for row in self.est_deaths],
            )

        # obs_cases and est_cases have different shapes
        with pytest.raises(ValueError, match=r"obs_\* and est_\* must have the same dimensions \(n_locations x n_time_steps\)\."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases[:, :-1],  # lop off last timestep
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
            )

        # obs_cases and obs_deaths have different shapes
        with pytest.raises(ValueError, match=r"obs_\* and est_\* must have the same dimensions \(n_locations x n_time_steps\)\."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths[:, :-1],  # lop off last timestep
                est_deaths=self.est_deaths,
            )

        # obs_cases and est_deaths have different shapes
        with pytest.raises(ValueError, match=r"obs_\* and est_\* must have the same dimensions \(n_locations x n_time_steps\)\."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths[:, :-1],  # lop off last timestep
            )

        # len(weights_location) != self.n_locations
        with pytest.raises(ValueError, match="weights_location must have length"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weights_location=np.ones(self.n_locations - 1),
            )

        # len(weights_time) != self.n_steps
        with pytest.raises(ValueError, match="weights_time must have length"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weights_time=np.ones(self.n_steps - 1),
            )

        # some weights_location are negative
        with pytest.raises(ValueError, match="weights_location must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weights_location=np.random.random(self.n_locations) - 1,
            )

        # some weights_time are negative
        with pytest.raises(ValueError, match="weights_time must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weights_time=np.random.random(self.n_steps) - 1,
            )

        # all weights_location are zero
        with pytest.raises(ValueError, match="weights_location and weights_time must not all be zero."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weights_location=np.zeros(self.n_locations),
            )

        # all weights_time are zero
        with pytest.raises(ValueError, match="weights_location and weights_time must not all be zero."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weights_time=np.zeros(self.n_steps),
            )

        # weight_cases < 0
        with pytest.raises(ValueError, match="weight_cases must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weight_cases=-0.8,
            )

        # weight_deaths < 0
        with pytest.raises(ValueError, match="weight_deaths must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                est_cases=self.est_cases,
                obs_deaths=self.obs_deaths,
                est_deaths=self.est_deaths,
                weight_deaths=-0.8,
            )

        return

    def test_calc_log_likelihood(self):
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23]
        # _beta = calc_log_likelihood_beta(obs_cases, est_cases, verbose=False)
        # _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), verbose=False)
        # _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, verbose=False)
        # _normal = calc_log_likelihood_normal(obs_cases, est_cases, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, verbose=False)

        return

    def test_calc_log_likelihood_no_weights(self):
        return

    def test_calc_log_likelihood_nan_observed(self):
        obs_cases = np.full_like(self.obs_cases[23], np.nan, dtype=np.float64)
        est_cases = self.est_cases[23]
        _beta = calc_log_likelihood_beta(obs_cases, est_cases, verbose=False)
        _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), verbose=False)
        _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, verbose=False)
        _normal = calc_log_likelihood_normal(obs_cases, est_cases, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, verbose=False)

        return

    def test_calc_log_likelihood_nan_estimated(self):
        obs_cases = self.obs_cases[23]
        est_cases = np.full_like(self.est_cases[23], np.nan, dtype=np.float64)
        _beta = calc_log_likelihood_beta(obs_cases, est_cases, verbose=False)
        _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), verbose=False)
        _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, verbose=False)
        _normal = calc_log_likelihood_normal(obs_cases, est_cases, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, verbose=False)

        return

    def test_calc_log_likelihood_nan_weights(self):
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23]
        weights = np.full_like(obs_cases, np.nan, dtype=np.float64)

        _beta = calc_log_likelihood_beta(obs_cases, est_cases, weights=weights, verbose=False)
        _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, weights=weights, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, weights=weights, verbose=False)
        _normal = calc_log_likelihood_normal(obs_cases, est_cases, weights=weights, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_nan_trials(self):
        # Just binomial
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23]
        _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.full_like(obs_cases, np.nan, dtype=np.float32), verbose=False)

        return

    def test_calc_log_likelihood_length_mismatches(self):
        # len(estimated) != len(observed)
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23][:-1]  # lop off last timestep

        with pytest.raises(ValueError, match="Lengths of observed "):
            _beta = calc_log_likelihood_beta(obs_cases, est_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _normal = calc_log_likelihood_normal(obs_cases, est_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, verbose=False)

        # len(weights) != len(observed)
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23]
        weights = np.full_like(obs_cases[:-1], np.nan, dtype=np.float64)  # lop off last timestep

        with pytest.raises(ValueError, match="Lengths of observed "):
            _beta = calc_log_likelihood_beta(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _normal = calc_log_likelihood_normal(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_negative_weights(self):
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23]
        weights = -np.random.rand(len(obs_cases))

        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _beta = calc_log_likelihood_beta(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _normal = calc_log_likelihood_normal(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_zero_weights(self):
        obs_cases = self.obs_cases[23]
        est_cases = self.est_cases[23]
        weights = np.zeros_like(obs_cases)

        with pytest.raises(ValueError, match="All weights are zero, cannot compute likelihood."):
            _beta = calc_log_likelihood_beta(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights are zero, cannot compute likelihood."):
            _binomial = calc_log_likelihood_binomial(obs_cases, est_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights are zero, cannot compute likelihood."):
            _gamma = calc_log_likelihood_gamma(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights are zero, cannot compute likelihood."):
            _negbin = calc_log_likelihood_negbin(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights are zero, cannot compute likelihood."):
            _normal = calc_log_likelihood_normal(obs_cases, est_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights are zero, cannot compute likelihood."):
            _poisson = calc_log_likelihood_poisson(obs_cases, est_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_invalid_beta_inputs(self):
        return

    def test_calc_log_likelihood_invalid_binomial_inputs(self):
        return

    def test_calc_log_likelihood_invalid_gamma_inputs(self):
        return

    def test_calc_log_likelihood_invalid_negbin_inputs(self):
        return

    def test_calc_log_likelihood_invalid_normal_inputs(self):
        return

    def test_calc_log_likelihood_invalid_poisson_inputs(self):
        return


if __name__ == "__main__":
    unittest.main()
