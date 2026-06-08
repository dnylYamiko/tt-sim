"""Tests for LerpSwingPlanner."""

import numpy as np
import pytest

from tt_sim.baselines.swing import LerpSwingPlanner
from tt_sim.interfaces import PaddleTarget


def _make_target(t_contact: float = 1.0) -> PaddleTarget:
    return PaddleTarget(
        position=np.array([0.5, 0.0, 0.3]),
        normal=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([0.0, 0.0, -1.0]),
        t_contact=t_contact,
    )


class TestLerpSwingPlanner:
    def test_trajectory_shape(self):
        planner = LerpSwingPlanner(n_joints=7, n_steps=50)
        q_current = np.zeros(7)
        traj = planner.plan(_make_target(), q_current)
        assert traj.positions.shape == (50, 7)
        assert traj.velocities.shape == (50, 7)
        assert traj.times.shape == (50,)

    def test_starts_at_current_ends_at_target(self):
        q_target = np.array([1.0, 2.0, 3.0])
        planner = LerpSwingPlanner(
            n_joints=3,
            n_steps=20,
            ik_fn=lambda _: q_target,
        )
        q_current = np.array([0.1, 0.2, 0.3])
        traj = planner.plan(_make_target(), q_current)
        np.testing.assert_allclose(traj.positions[0], q_current)
        np.testing.assert_allclose(traj.positions[-1], q_target)

    def test_times_length(self):
        planner = LerpSwingPlanner(n_joints=4, n_steps=30)
        traj = planner.plan(_make_target(t_contact=2.0), np.zeros(4))
        assert len(traj.times) == 30
        assert traj.times[0] == pytest.approx(0.0)
        assert traj.times[-1] == pytest.approx(2.0)

    def test_custom_ik_fn(self):
        q_goal = np.ones(5) * 0.5
        planner = LerpSwingPlanner(n_joints=5, n_steps=10, ik_fn=lambda _: q_goal)
        traj = planner.plan(_make_target(), np.zeros(5))
        np.testing.assert_allclose(traj.positions[-1], q_goal)

    def test_default_ik_returns_zeros(self):
        planner = LerpSwingPlanner(n_joints=3, n_steps=10)
        q_current = np.array([1.0, 2.0, 3.0])
        traj = planner.plan(_make_target(), q_current)
        # default ik returns zeros, so trajectory ends at zeros
        np.testing.assert_allclose(traj.positions[-1], np.zeros(3))
