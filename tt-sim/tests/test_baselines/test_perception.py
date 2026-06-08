"""Tests for SimPerceiver."""

import numpy as np
import pytest

from tt_sim.baselines.perception import SimPerceiver
from tt_sim.interfaces import BallObservation


@pytest.fixture
def perceiver():
    return SimPerceiver()


def test_observe_returns_ball_observation(perceiver):
    state = {"ball_pos": [1.0, 2.0, 3.0], "time": 0.5}
    obs = perceiver.observe(state)
    assert isinstance(obs, BallObservation)


def test_observe_position(perceiver):
    state = {"ball_pos": [1.0, 2.0, 3.0], "time": 0.5}
    obs = perceiver.observe(state)
    np.testing.assert_array_equal(obs.position, np.array([1.0, 2.0, 3.0]))
    assert obs.position.dtype == np.float64


def test_observe_sigma_zero(perceiver):
    state = {"ball_pos": [0.0, 0.0, 0.0], "time": 1.0}
    obs = perceiver.observe(state)
    assert obs.sigma == 0.0


def test_observe_timestamp(perceiver):
    state = {"ball_pos": [0.0, 0.0, 0.0], "time": 1.23}
    obs = perceiver.observe(state)
    assert obs.timestamp == 1.23


def test_observe_default_timestamp(perceiver):
    state = {"ball_pos": [0.0, 0.0, 0.0]}
    obs = perceiver.observe(state)
    assert obs.timestamp == 0.0
