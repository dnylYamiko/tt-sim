"""Tests for symbolic RNEA dynamics — validates against MuJoCo."""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")
mujoco = pytest.importorskip("mujoco")

from tt_sim.upgrades.mpc_dynamics import build_dynamics_casadi


def _load_model(robot: str):
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


def _mujoco_dynamics(model, q, qd, qdd, ndof):
    """Ground truth inverse dynamics from MuJoCo: tau = M*qdd + C*qd + g."""
    d = mujoco.MjData(model)
    d.qpos[:ndof] = q
    d.qvel[:ndof] = qd
    d.qacc[:ndof] = qdd
    mujoco.mj_inverse(model, d)
    return d.qfrc_inverse[:ndof].copy()


def _mujoco_gravity(model, q, ndof):
    """Gravity vector from MuJoCo (inverse dynamics with qd=0, qdd=0)."""
    return _mujoco_dynamics(model, q, np.zeros(ndof), np.zeros(ndof), ndof)


def _mujoco_mass_matrix(model, q, ndof):
    """Mass matrix from MuJoCo."""
    d = mujoco.MjData(model)
    d.qpos[:ndof] = q
    mujoco.mj_forward(model, d)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, d.qM)
    return M[:ndof, :ndof].copy()


class TestGravityVector:
    """Symbolic gravity vector matches MuJoCo."""

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_gravity_at_home(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        g_sym = np.array(dyn['gravity_fn'](q)).flatten()
        g_mj = _mujoco_gravity(model, q, ndof)
        np.testing.assert_allclose(g_sym, g_mj, atol=0.5,
            err_msg=f"{robot} gravity mismatch at home")

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_gravity_at_random(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        rng = np.random.default_rng(42)
        for _ in range(3):
            q = rng.uniform(-1, 1, ndof)
            g_sym = np.array(dyn['gravity_fn'](q)).flatten()
            g_mj = _mujoco_gravity(model, q, ndof)
            np.testing.assert_allclose(g_sym, g_mj, atol=0.5,
                err_msg=f"{robot} gravity mismatch at q={q.round(3)}")


class TestMassMatrix:
    """Symbolic mass matrix matches MuJoCo."""

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_mass_matrix_at_home(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        M_sym = np.array(dyn['mass_matrix_fn'](q))
        M_mj = _mujoco_mass_matrix(model, q, ndof)
        assert M_sym.shape == (ndof, ndof)
        # Symmetric
        np.testing.assert_allclose(M_sym, M_sym.T, atol=1e-10,
            err_msg="Mass matrix not symmetric")
        # Positive definite
        eigvals = np.linalg.eigvalsh(M_sym)
        assert np.all(eigvals > 0), f"Mass matrix not positive definite: {eigvals}"
        np.testing.assert_allclose(M_sym, M_mj, atol=1.0,
            err_msg=f"{robot} mass matrix mismatch at home")

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_mass_matrix_at_random(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        rng = np.random.default_rng(42)
        for _ in range(3):
            q = rng.uniform(-1, 1, ndof)
            M_sym = np.array(dyn['mass_matrix_fn'](q))
            M_mj = _mujoco_mass_matrix(model, q, ndof)
            np.testing.assert_allclose(M_sym, M_mj, atol=1.0,
                err_msg=f"{robot} mass matrix mismatch at q={q.round(3)}")


class TestInverseDynamics:
    """Full inverse dynamics tau = RNEA(q, qd, qdd) matches MuJoCo."""

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_id_at_home_zero_motion(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        qd = np.zeros(ndof)
        qdd = np.zeros(ndof)
        tau_sym = np.array(dyn['rnea_fn'](q, qd, qdd)).flatten()
        tau_mj = _mujoco_dynamics(model, q, qd, qdd, ndof)
        np.testing.assert_allclose(tau_sym, tau_mj, atol=0.5,
            err_msg=f"{robot} ID mismatch at rest")

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_id_with_motion(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        rng = np.random.default_rng(123)
        for _ in range(3):
            q = rng.uniform(-1, 1, ndof)
            qd = rng.uniform(-2, 2, ndof)
            qdd = rng.uniform(-5, 5, ndof)
            tau_sym = np.array(dyn['rnea_fn'](q, qd, qdd)).flatten()
            tau_mj = _mujoco_dynamics(model, q, qd, qdd, ndof)
            np.testing.assert_allclose(tau_sym, tau_mj, atol=2.0,
                err_msg=f"{robot} ID mismatch with motion")


class TestDifferentiability:
    """RNEA functions have valid CasADi derivatives."""

    def test_rnea_jacobian(self):
        model, ndof = _load_model("fanuc")
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        qd = np.zeros(ndof)
        qdd = np.zeros(ndof)
        J = dyn['rnea_fn'].jacobian()
        result = J(q, qd, qdd, np.zeros(ndof))
        # Jacobian should be computable without error and have nonzero entries
        J_arr = np.array(result[0])  # first output is the Jacobian matrix
        assert J_arr.shape[0] == ndof
        assert np.any(np.abs(J_arr) > 1e-6)
