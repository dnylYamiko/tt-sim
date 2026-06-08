"""Tests for FaceNetAimer baseline."""

import numpy as np
import pytest
from tt_sim.baselines.aiming import FaceNetAimer
from tt_sim.interfaces import BallState, SpinEstimate


def _make_ball(pos):
    return BallState(position=np.array(pos, dtype=float), velocity=np.zeros(3))


def _make_spin():
    return SpinEstimate(omega=np.zeros(3), confidence=np.zeros(3))


@pytest.fixture
def aimer():
    return FaceNetAimer()


class TestFaceNetAimer:
    @pytest.mark.parametrize("pos", [
        [-0.5, -1.0, 0.8],
        [0.0, -0.5, 1.0],
        [0.3, -1.2, 0.76],
        [-0.2, -0.3, 0.9],
    ])
    def test_normal_points_toward_net(self, aimer, pos):
        target = aimer.aim(_make_ball(pos), _make_spin())
        # Robot at -y, net at y=0, so normal should have positive y component
        assert target.normal[1] > 0

    @pytest.mark.parametrize("pos", [
        [0.0, -1.0, 0.8],
        [0.5, -0.5, 1.2],
        [-0.3, -1.37, 0.76],
    ])
    def test_normal_is_unit_vector(self, aimer, pos):
        target = aimer.aim(_make_ball(pos), _make_spin())
        np.testing.assert_allclose(np.linalg.norm(target.normal), 1.0, atol=1e-10)

    def test_velocity_is_zero(self, aimer):
        target = aimer.aim(_make_ball([0.0, -1.0, 0.8]), _make_spin())
        np.testing.assert_array_equal(target.velocity, np.zeros(3))

    def test_contact_position_matches_ball(self, aimer):
        pos = [0.1, -0.8, 0.9]
        target = aimer.aim(_make_ball(pos), _make_spin())
        np.testing.assert_array_equal(target.position, np.array(pos))

    def test_t_contact_is_positive(self, aimer):
        target = aimer.aim(_make_ball([0.0, -1.0, 0.8]), _make_spin())
        assert target.t_contact > 0.0
