import pytest
import numpy as np

from tt_sim.interfaces import PaddleTarget, JointTrajectory, SwingPlanner
from tt_sim.registry import load


MODULE = "tt_sim.upgrades.swing"
CLASS_NAME = "QuinticSwingPlanner"
FLAG = "quintic"


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def _make_target(pos=None, vel=None, t_contact=0.5):
    return PaddleTarget(
        position=pos if pos is not None else np.array([0.5, 0.5, 1.0]),
        normal=np.array([0, 1, 0]),
        velocity=vel if vel is not None else np.zeros(3),
        t_contact=t_contact,
    )


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, SwingPlanner)


def test_registry():
    cls = load("swing", FLAG)
    assert cls is not None


class TestQuinticSwingPlanner:
    """Functional tests for the Stage 1 quintic swing planner."""

    def _make_planner(self, n_joints=6, ik_fn=None, jac_fn=None, n_steps=50):
        cls = _get_cls()
        return cls(n_joints=n_joints, n_steps=n_steps, ik_fn=ik_fn, jac_fn=jac_fn)

    def test_returns_joint_trajectory(self):
        planner = self._make_planner()
        q0 = np.zeros(6)
        result = planner.plan(_make_target(), q0)
        assert isinstance(result, JointTrajectory)

    def test_output_shapes(self):
        n_joints, n_steps = 6, 50
        planner = self._make_planner(n_joints=n_joints, n_steps=n_steps)
        q0 = np.zeros(n_joints)
        result = planner.plan(_make_target(), q0)
        assert result.times.shape == (n_steps,)
        assert result.positions.shape == (n_steps, n_joints)
        assert result.velocities.shape == (n_steps, n_joints)

    def test_starts_at_q_current(self):
        q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        planner = self._make_planner()
        result = planner.plan(_make_target(), q0)
        np.testing.assert_allclose(result.positions[0], q0, atol=1e-10)

    def test_ends_at_q_target(self):
        q_target = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.8])
        ik_fn = lambda target: q_target
        planner = self._make_planner(ik_fn=ik_fn)
        q0 = np.zeros(6)
        result = planner.plan(_make_target(), q0)
        np.testing.assert_allclose(result.positions[-1], q_target, atol=1e-10)

    def test_zero_velocity_at_start(self):
        q_target = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.8])
        ik_fn = lambda target: q_target
        planner = self._make_planner(ik_fn=ik_fn)
        q0 = np.zeros(6)
        result = planner.plan(_make_target(), q0)
        np.testing.assert_allclose(result.velocities[0], np.zeros(6), atol=1e-10)

    def test_zero_velocity_at_end_no_jac(self):
        """Without jac_fn, end velocity should be zero."""
        q_target = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.8])
        ik_fn = lambda target: q_target
        planner = self._make_planner(ik_fn=ik_fn)
        q0 = np.zeros(6)
        result = planner.plan(_make_target(vel=np.zeros(3)), q0)
        np.testing.assert_allclose(result.velocities[-1], np.zeros(6), atol=1e-10)

    def test_nonzero_end_velocity_with_jac(self):
        """With jac_fn and nonzero paddle velocity, end velocity should be nonzero."""
        q_target = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.8])
        ik_fn = lambda target: q_target
        # Simple identity-like Jacobian for testing
        jac_fn = lambda q: np.eye(3, 6)
        planner = self._make_planner(ik_fn=ik_fn, jac_fn=jac_fn)
        q0 = np.zeros(6)
        paddle_vel = np.array([0.0, 1.0, 0.0])
        result = planner.plan(_make_target(vel=paddle_vel), q0)
        # End velocity should not be zero
        assert np.linalg.norm(result.velocities[-1]) > 0.01

    def test_trajectory_is_smooth(self):
        """Positions should change smoothly (no discontinuities)."""
        q_target = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.8])
        ik_fn = lambda target: q_target
        planner = self._make_planner(ik_fn=ik_fn, n_steps=100)
        q0 = np.zeros(6)
        result = planner.plan(_make_target(t_contact=1.0), q0)
        # Check that position differences are small between consecutive steps
        diffs = np.diff(result.positions, axis=0)
        max_jump = np.max(np.abs(diffs))
        assert max_jump < 0.5, f"Max jump {max_jump} too large for smooth trajectory"

    def test_times_span_t_contact(self):
        planner = self._make_planner()
        q0 = np.zeros(6)
        t_contact = 0.75
        result = planner.plan(_make_target(t_contact=t_contact), q0)
        assert result.times[0] == 0.0
        np.testing.assert_allclose(result.times[-1], t_contact)

    def test_dof_mismatch_handled(self):
        """IK returning more DOFs than q_current should be truncated."""
        ik_fn = lambda target: np.ones(10)  # returns 10 joints
        planner = self._make_planner(n_joints=10, ik_fn=ik_fn)
        q0 = np.zeros(6)  # only 6 joints
        result = planner.plan(_make_target(), q0)
        assert result.positions.shape[1] == 6
