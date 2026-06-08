"""Tests for symbolic FK builder — validates against MuJoCo FK."""

import pytest
import numpy as np

# Skip all tests if casadi or mujoco unavailable
casadi = pytest.importorskip("casadi")
mujoco = pytest.importorskip("mujoco")

from tt_sim.upgrades.mpc_fk import build_fk_casadi


def _load_model(robot: str):
    """Load MuJoCo model for a robot."""
    import os
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'xml')
    if robot == 'fanuc':
        xml = os.path.join(base, 'table_tennis_fanuc.xml')
        ndof = 6
    else:
        xml = os.path.join(base, 'table_tennis_env.xml')
        ndof = 7
    model = mujoco.MjModel.from_xml_path(xml)
    return model, ndof


def _mujoco_fk(model, q, ndof):
    """Ground truth FK from MuJoCo."""
    d = mujoco.MjData(model)
    d.qpos[:ndof] = q
    mujoco.mj_forward(model, d)
    ee_pos = d.body("EE").xpos.copy()
    ee_rot = d.body("EE").xmat.reshape(3, 3).copy()
    return ee_pos, ee_rot


@pytest.mark.parametrize("robot", ["fanuc", "wam"])
def test_fk_matches_mujoco_at_home(robot):
    """Symbolic FK matches MuJoCo FK at home position."""
    model, ndof = _load_model(robot)
    fk = build_fk_casadi(model, ndof)

    q_home = np.zeros(ndof)
    p_sym = np.array(fk['fk_fn'](q_home)).flatten()
    p_mj, _ = _mujoco_fk(model, q_home, ndof)

    np.testing.assert_allclose(p_sym, p_mj, atol=0.02,
        err_msg=f"{robot} FK position mismatch at home")


@pytest.mark.parametrize("robot", ["fanuc", "wam"])
def test_fk_matches_mujoco_at_random(robot):
    """Symbolic FK matches MuJoCo FK at several random configs."""
    model, ndof = _load_model(robot)
    fk = build_fk_casadi(model, ndof)

    rng = np.random.default_rng(42)
    for _ in range(5):
        q = rng.uniform(fk['joint_ranges'][:, 0] * 0.3,
                        fk['joint_ranges'][:, 1] * 0.3)
        p_sym = np.array(fk['fk_fn'](q)).flatten()
        p_mj, _ = _mujoco_fk(model, q, ndof)

        np.testing.assert_allclose(p_sym, p_mj, atol=0.02,
            err_msg=f"{robot} FK mismatch at q={q.round(3)}")


@pytest.mark.parametrize("robot", ["fanuc", "wam"])
def test_fk_orientation_matches(robot):
    """Symbolic FK rotation matches MuJoCo at a test config."""
    model, ndof = _load_model(robot)
    fk = build_fk_casadi(model, ndof)

    q = np.zeros(ndof)
    q[0] = 0.5  # rotate base joint
    p_sym, R_flat = fk['fk_rot_fn'](q)
    R_sym = np.array(R_flat).reshape(3, 3)
    _, R_mj = _mujoco_fk(model, q, ndof)

    np.testing.assert_allclose(R_sym, R_mj, atol=0.05,
        err_msg=f"{robot} FK rotation mismatch")


def test_joint_ranges_extracted():
    """Joint ranges are correctly extracted from model."""
    model, ndof = _load_model("fanuc")
    fk = build_fk_casadi(model, ndof)
    ranges = fk['joint_ranges']

    assert ranges.shape == (6, 2)
    # All Fanuc joints have symmetric ranges >= pi
    for i in range(6):
        assert ranges[i, 0] < 0
        assert ranges[i, 1] > 0
        assert ranges[i, 1] >= np.pi * 0.9


def test_fk_is_differentiable():
    """CasADi function has valid Jacobian."""
    model, ndof = _load_model("fanuc")
    fk = build_fk_casadi(model, ndof)

    q_test = np.zeros(ndof)
    J = np.array(fk['fk_fn'].jacobian()(q_test, np.zeros(3)))
    # Jacobian should have nonzero entries
    assert np.any(np.abs(J) > 1e-6)
