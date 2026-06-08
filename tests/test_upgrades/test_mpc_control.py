"""Tests for MPCController."""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")

from tt_sim.interfaces import HighLevelController
from tt_sim.registry import load
from tt_sim.upgrades.control import MPCController
from tt_sim.upgrades.control import TorqueMPCController

# Reuse Dummy implementations from test_control.py
from tests.test_upgrades.test_control import (
    DummyPerceiver, DummyPredictor, DummySpinEstimator,
    DummyAimer, DummySwingPlanner,
)


def test_import():
    assert MPCController is not None


def test_subclass():
    assert issubclass(MPCController, HighLevelController)


def test_registry():
    cls = load("control", "mpc")
    assert cls is MPCController


def test_holds_without_fk():
    """Without FK dict, controller holds position."""
    ctrl = MPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), fk_dict=None,
    )
    q = np.array([0.5, 0.5])
    for _ in range(10):
        result = ctrl.step({}, q)
        np.testing.assert_array_equal(result, q)


def test_reset_clears_state():
    ctrl = MPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), fk_dict=None,
    )
    q = np.array([0.5, 0.5])
    for _ in range(10):
        ctrl.step({}, q)
    ctrl.reset()
    assert ctrl._observations == []
    assert ctrl._frame_count == 0
    assert ctrl._prev_solution is None


# ── TorqueMPCController tests ────────────────────────────────────────────


def test_torque_mpc_import():
    assert TorqueMPCController is not None


def test_torque_mpc_subclass():
    assert issubclass(TorqueMPCController, HighLevelController)


def test_torque_mpc_registry():
    cls = load("control", "torque_mpc")
    assert cls is TorqueMPCController


def test_torque_mpc_holds_without_dynamics():
    """Without dynamics dict, controller returns zero torque."""
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    q = np.array([0.5, 0.5])
    qd = np.array([0.0, 0.0])
    for _ in range(10):
        result = ctrl.step({}, q, qd)
    np.testing.assert_array_equal(result, np.zeros(2))


def test_torque_mpc_returns_bounded():
    """Output is in ctrl-space, bounded to [-1, 1]."""
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    q = np.array([0.5, 0.5])
    qd = np.array([0.0, 0.0])
    result = ctrl.step({}, q, qd)
    assert np.all(np.abs(result) <= 1.0)


def test_torque_mpc_torque_mode_flag():
    """Controller has torque_mode = True."""
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    assert ctrl.torque_mode is True


def test_torque_mpc_reset():
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    q = np.array([0.5, 0.5])
    qd = np.array([0.0, 0.0])
    for _ in range(10):
        ctrl.step({}, q, qd)
    ctrl.reset()
    assert ctrl._observations == []
    assert ctrl._frame_count == 0
    assert ctrl._prev_x0 is None
