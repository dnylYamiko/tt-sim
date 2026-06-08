"""Tests for OpenLoopController."""

import numpy as np
from tt_sim.interfaces import (
    Perceiver, Predictor, SpinEstimator, Aimer, SwingPlanner,
    BallObservation, BallState, SpinEstimate, PaddleTarget, JointTrajectory,
)
from tt_sim.baselines.control import OpenLoopController


# ── Dummy implementations ────────────────────────────────────────────────────

class DummyPerceiver(Perceiver):
    def __init__(self):
        self._call_count = 0

    def observe(self, env_state: dict) -> BallObservation:
        self._call_count += 1
        return BallObservation(
            position=np.array([1.0, 2.0, 3.0]),
            sigma=0.01,
            timestamp=self._call_count * 0.1,
        )


class DummyPredictor(Predictor):
    def __init__(self):
        self.call_count = 0

    def predict(self, observations, t_future):
        self.call_count += 1
        return BallState(position=np.zeros(3), velocity=np.zeros(3))


class DummySpinEstimator(SpinEstimator):
    def __init__(self):
        self.call_count = 0

    def estimate(self, observations):
        self.call_count += 1
        return SpinEstimate(omega=np.zeros(3), confidence=np.ones(3))


class DummyAimer(Aimer):
    def __init__(self):
        self.call_count = 0

    def aim(self, ball, spin):
        self.call_count += 1
        return PaddleTarget(
            position=np.zeros(3), normal=np.array([0, 0, 1.0]),
            velocity=np.zeros(3), t_contact=0.5,
        )


class DummySwingPlanner(SwingPlanner):
    def __init__(self):
        self.call_count = 0

    def plan(self, target, q_current):
        self.call_count += 1
        times = np.array([0.0, 0.1, 0.2])
        positions = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        return JointTrajectory(times=times, positions=positions)


def _make_controller():
    p = DummyPerceiver()
    pred = DummyPredictor()
    spin = DummySpinEstimator()
    aim = DummyAimer()
    swing = DummySwingPlanner()
    ctrl = OpenLoopController(p, pred, spin, aim, swing)
    return ctrl, pred, spin, aim, swing


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_first_call_returns_q_current():
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    result = ctrl.step({}, q)
    np.testing.assert_array_equal(result, q)


def test_plans_after_three_observations():
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    ctrl.step({}, q)  # 1st obs
    ctrl.step({}, q)  # 2nd obs
    result = ctrl.step({}, q)  # 3rd obs, plans, returns traj[0]
    np.testing.assert_array_equal(result, np.array([1.0, 1.0]))


def test_replays_trajectory():
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    ctrl.step({}, q)  # 1st obs
    ctrl.step({}, q)  # 2nd obs
    ctrl.step({}, q)  # 3rd obs, plans, returns traj[0]
    r2 = ctrl.step({}, q)  # traj[1]
    r3 = ctrl.step({}, q)  # traj[2]
    np.testing.assert_array_equal(r2, np.array([2.0, 2.0]))
    np.testing.assert_array_equal(r3, np.array([3.0, 3.0]))


def test_holds_position_after_trajectory_ends():
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    for _ in range(6):
        ctrl.step({}, q)
    # Trajectory has 3 steps; calls 3-5 exhaust it, call 6 should hold
    result = ctrl.step({}, q)
    np.testing.assert_array_equal(result, q)


def test_plans_only_once():
    ctrl, pred, spin, aim, swing = _make_controller()
    q = np.array([0.5, 0.5])
    for _ in range(10):
        ctrl.step({}, q)
    assert pred.call_count == 1
    assert spin.call_count == 1
    assert aim.call_count == 1
    assert swing.call_count == 1


def test_reset_clears_state():
    ctrl, pred, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    ctrl.step({}, q)
    ctrl.step({}, q)
    ctrl.reset()
    assert ctrl._observations == []
    assert ctrl._trajectory is None
    assert ctrl._step_index == 0
    assert ctrl._planned is False
    # After reset, first call should hold again
    result = ctrl.step({}, q)
    np.testing.assert_array_equal(result, q)
