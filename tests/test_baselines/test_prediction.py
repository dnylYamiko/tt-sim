"""Tests for BallisticPredictor."""

import numpy as np
import pytest

from tt_sim.baselines.prediction import BallisticPredictor, GRAVITY
from tt_sim.interfaces import BallObservation


def _make_obs(x0, v0, t):
    """Generate a ballistic observation at time t."""
    pos = np.array([
        x0[0] + v0[0] * t,
        x0[1] + v0[1] * t,
        x0[2] + v0[2] * t - 0.5 * GRAVITY * t**2,
    ])
    return BallObservation(position=pos, sigma=0.001, timestamp=t)


class TestBallisticPredictor:
    def test_predict_known_trajectory(self):
        """Fit 3 ballistic points, predict a 4th."""
        x0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([5.0, -3.0, 10.0])

        obs = [_make_obs(x0, v0, t) for t in [0.0, 0.1, 0.2]]
        pred = BallisticPredictor().predict(obs, t_future=0.3)

        expected = _make_obs(x0, v0, 0.3)
        np.testing.assert_allclose(pred.position, expected.position, atol=1e-6)

        expected_vel = np.array([v0[0], v0[1], v0[2] - GRAVITY * 0.3])
        np.testing.assert_allclose(pred.velocity, expected_vel, atol=1e-6)

    def test_too_few_observations(self):
        obs = [BallObservation(position=np.zeros(3), sigma=0.001, timestamp=0.0)]
        with pytest.raises(ValueError):
            BallisticPredictor().predict(obs, t_future=1.0)

    def test_two_observations(self):
        """Minimum case: 2 observations."""
        x0 = np.array([0.0, 0.0, 5.0])
        v0 = np.array([1.0, 1.0, 0.0])

        obs = [_make_obs(x0, v0, t) for t in [0.0, 0.5]]
        pred = BallisticPredictor().predict(obs, t_future=1.0)

        expected = _make_obs(x0, v0, 1.0)
        np.testing.assert_allclose(pred.position, expected.position, atol=1e-6)
