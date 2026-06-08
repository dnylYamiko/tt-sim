"""Tests for tt_sim.interfaces dataclasses and ABCs."""

import numpy as np
from tt_sim.interfaces import (
    BallObservation, BallState, SpinEstimate, PaddleTarget, JointTrajectory,
)


def test_ball_observation():
    obs = BallObservation(position=np.zeros(3), sigma=0.01, timestamp=0.0)
    assert obs.position.shape == (3,)
    assert obs.sigma == 0.01


def test_ball_state():
    bs = BallState(position=np.zeros(3), velocity=np.ones(3))
    assert bs.covariance is None
    bs2 = BallState(position=np.zeros(3), velocity=np.ones(3), covariance=np.eye(6))
    assert bs2.covariance.shape == (6, 6)


def test_spin_estimate():
    se = SpinEstimate(omega=np.zeros(3), confidence=np.ones(3))
    assert se.omega.shape == (3,)


def test_paddle_target():
    pt = PaddleTarget(
        position=np.zeros(3), normal=np.array([0, 0, 1.0]),
        velocity=np.zeros(3), t_contact=1.5,
    )
    assert pt.t_contact == 1.5


def test_joint_trajectory():
    jt = JointTrajectory(times=np.linspace(0, 1, 10), positions=np.zeros((10, 7)))
    assert jt.velocities is None
    assert jt.positions.shape == (10, 7)
