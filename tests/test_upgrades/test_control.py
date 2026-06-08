import pytest
import numpy as np

from tt_sim.interfaces import (
    HighLevelController, Perceiver, Predictor, SpinEstimator,
    Aimer, SwingPlanner, BallObservation, BallState, SpinEstimate,
    PaddleTarget, JointTrajectory,
)
from tt_sim.registry import load
from tt_sim.upgrades.control import ReplanController


MODULE = "tt_sim.upgrades.control"
CLASS_NAME = "ReplanController"
FLAG = "replan"


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, HighLevelController)


def test_registry():
    cls = load("control", FLAG)
    assert cls is not None


# ── Dummy implementations ────────────────────────────────────────────────────

class DummyPerceiver(Perceiver):
    def __init__(self):
        self._call_count = 0

    def observe(self, env_state):
        self._call_count += 1
        return BallObservation(
            position=np.array([1.0, 2.0, 3.0]),
            sigma=0.01,
            timestamp=self._call_count * 0.008,
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
    ctrl = ReplanController(p, pred, spin, aim, swing)
    return ctrl, pred, spin, aim, swing


# ── Functional tests ─────────────────────────────────────────────────────────

def test_holds_before_min_obs():
    ctrl, _, _, _, swing = _make_controller()
    q = np.array([0.5, 0.5])
    # First 3 steps (< MIN_OBS=4) should hold
    for _ in range(3):
        result = ctrl.step({}, q)
        np.testing.assert_array_equal(result, q)
    assert swing.call_count == 0


def test_replans_at_interval():
    """Swing planner is called at frame 4 (first replan), then 8, etc."""
    ctrl, _, _, _, swing = _make_controller()
    q = np.array([0.5, 0.5])
    for _ in range(4):
        ctrl.step({}, q)
    assert swing.call_count == 1  # frame 4 triggers first replan

    for _ in range(4):
        ctrl.step({}, q)
    assert swing.call_count == 2  # frame 8 triggers second replan


def test_holds_between_replans():
    """Between replans, the controller returns the same action."""
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    # Trigger first replan at frame 4
    for _ in range(4):
        ctrl.step({}, q)
    action_at_4 = ctrl._last_action.copy()

    # Frames 5, 6, 7 should return the same action
    for _ in range(3):
        result = ctrl.step({}, q)
        np.testing.assert_array_equal(result, action_at_4)


def test_returns_lookahead_not_start():
    """The returned action should NOT be traj[0] (q_current); it should be
    interpolated ahead into the trajectory."""
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    for _ in range(4):
        result = ctrl.step({}, q)
    # Trajectory: times=[0, 0.1, 0.2], positions=[[1,1],[2,2],[3,3]]
    # dt_ahead = max(2*0.032, 0.3*0.2) = max(0.064, 0.06) = 0.064
    # interp at 0.064 -> 1.0 + (2.0-1.0) * 0.064/0.1 = 1.64
    expected = np.array([1.64, 1.64])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_reset_clears_state():
    ctrl, *_ = _make_controller()
    q = np.array([0.5, 0.5])
    for _ in range(4):
        ctrl.step({}, q)
    ctrl.reset()
    assert ctrl._observations == []
    assert ctrl._frame_count == 0
    assert ctrl._last_action is None
    # After reset, should hold
    result = ctrl.step({}, q)
    np.testing.assert_array_equal(result, q)
