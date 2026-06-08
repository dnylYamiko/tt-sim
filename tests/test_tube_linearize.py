"""Tests for dynamics linearization."""
import numpy as np
import pytest


def test_linearize_dimensions():
    """A and B should have correct shapes."""
    from tt_sim.upgrades.tube_linearize import build_linearization_fn
    import mujoco
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from run import ROBOT_CONFIGS
    
    rcfg = ROBOT_CONFIGS["fanuc"]
    nj = rcfg["ndof"]
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'xml', rcfg['xml'])
    
    from tt_sim.upgrades.mpc_dynamics import build_dynamics_casadi
    model = mujoco.MjModel.from_xml_path(xml_path)
    dynamics_dict = build_dynamics_casadi(model, nj)
    
    lin_fn = build_linearization_fn(dynamics_dict, nj, dt=0.008)
    
    q = np.zeros(nj)
    qd = np.zeros(nj)
    tau = np.zeros(nj)
    
    A, B = lin_fn(q, qd, tau)
    assert A.shape == (2*nj, 2*nj), f"A shape {A.shape} != {(2*nj, 2*nj)}"
    assert B.shape == (2*nj, nj), f"B shape {B.shape} != {(2*nj, nj)}"


def test_linearize_finite_diff():
    """Linearization should match finite differences."""
    from tt_sim.upgrades.tube_linearize import build_linearization_fn
    import mujoco
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from run import ROBOT_CONFIGS
    
    rcfg = ROBOT_CONFIGS["fanuc"]
    nj = rcfg["ndof"]
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'xml', rcfg['xml'])
    
    from tt_sim.upgrades.mpc_dynamics import build_dynamics_casadi
    model = mujoco.MjModel.from_xml_path(xml_path)
    dynamics_dict = build_dynamics_casadi(model, nj)
    dt = 0.008
    
    lin_fn = build_linearization_fn(dynamics_dict, nj, dt=dt)
    
    # Test at a non-trivial point
    rng = np.random.default_rng(42)
    q0 = rng.uniform(-1, 1, nj)
    qd0 = rng.uniform(-0.5, 0.5, nj)
    tau0 = rng.uniform(-10, 10, nj)
    
    A, B = lin_fn(q0, qd0, tau0)
    
    # Finite-diff A: perturb each state dimension
    eps = 1e-5
    nx = 2 * nj
    
    # We need the forward dynamics function
    rnea_fn = dynamics_dict['rnea_fn']
    M_fn = dynamics_dict['mass_matrix_fn']
    
    def simulate_step(q, qd, tau):
        M = np.array(M_fn(q))
        bias = np.array(rnea_fn(q, qd, np.zeros(nj))).flatten()
        qdd = np.linalg.solve(M, tau - bias)
        qd_next = qd + dt * qdd
        q_next = q + dt * qd_next
        return np.concatenate([q_next, qd_next])
    
    x0 = np.concatenate([q0, qd0])
    f0 = simulate_step(q0, qd0, tau0)
    
    A_fd = np.zeros((nx, nx))
    for i in range(nx):
        x_pert = x0.copy()
        x_pert[i] += eps
        f_pert = simulate_step(x_pert[:nj], x_pert[nj:], tau0)
        A_fd[:, i] = (f_pert - f0) / eps
    
    np.testing.assert_allclose(A, A_fd, atol=1e-4, rtol=1e-3,
                               err_msg="A doesn't match finite differences")
    
    # Finite-diff B
    B_fd = np.zeros((nx, nj))
    for i in range(nj):
        tau_pert = tau0.copy()
        tau_pert[i] += eps
        f_pert = simulate_step(q0, qd0, tau_pert)
        B_fd[:, i] = (f_pert - f0) / eps
    
    np.testing.assert_allclose(B, B_fd, atol=1e-4, rtol=1e-3,
                               err_msg="B doesn't match finite differences")
