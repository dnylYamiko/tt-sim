"""Tests for ZeroSpinEstimator."""

import numpy as np
from numpy.testing import assert_array_equal

from tt_sim.baselines.spin import ZeroSpinEstimator
from tt_sim.interfaces import BallObservation


class TestZeroSpinEstimator:
    def setup_method(self):
        self.estimator = ZeroSpinEstimator()

    def test_returns_zero_omega_and_confidence(self):
        obs = [BallObservation(position=np.array([1.0, 2.0, 3.0]), sigma=0.01, timestamp=0.0)]
        result = self.estimator.estimate(obs)
        assert_array_equal(result.omega, np.zeros(3))
        assert_array_equal(result.confidence, np.zeros(3))

    def test_empty_observations(self):
        result = self.estimator.estimate([])
        assert_array_equal(result.omega, np.zeros(3))
        assert_array_equal(result.confidence, np.zeros(3))

    def test_multiple_observations(self):
        obs = [
            BallObservation(position=np.array([i, 0.0, 0.0]), sigma=0.01, timestamp=i * 0.01)
            for i in range(5)
        ]
        result = self.estimator.estimate(obs)
        assert_array_equal(result.omega, np.zeros(3))
        assert_array_equal(result.confidence, np.zeros(3))
