"""Tests for constraint tightening."""
import numpy as np
import pytest


def test_tighten_box_basic():
    """Tightened bounds should be strictly inside original."""
    from tt_sim.upgrades.tube_tighten import tighten_box_constraints
    n = 6
    lb = -np.ones(n) * 3.14
    ub = np.ones(n) * 3.14
    Z = np.eye(n) * 0.01  # small tube
    
    lb_t, ub_t = tighten_box_constraints(lb, ub, Z)
    assert np.all(lb_t > lb)
    assert np.all(ub_t < ub)
    assert np.all(lb_t < ub_t)  # still feasible


def test_tighten_box_zero_tube():
    """Zero tube should not change bounds."""
    from tt_sim.upgrades.tube_tighten import tighten_box_constraints
    n = 6
    lb = -np.ones(n) * 2.0
    ub = np.ones(n) * 2.0
    Z = np.zeros((n, n))
    
    lb_t, ub_t = tighten_box_constraints(lb, ub, Z)
    np.testing.assert_array_equal(lb_t, lb)
    np.testing.assert_array_equal(ub_t, ub)


def test_tighten_box_large_tube_stays_feasible():
    """Even with large tube, bounds should not cross."""
    from tt_sim.upgrades.tube_tighten import tighten_box_constraints
    n = 6
    lb = -np.ones(n) * 0.1  # very tight original bounds
    ub = np.ones(n) * 0.1
    Z = np.eye(n) * 10.0  # huge tube (larger than bounds)
    
    lb_t, ub_t = tighten_box_constraints(lb, ub, Z)
    assert np.all(lb_t < ub_t), "Bounds crossed!"


def test_tighten_all_knots():
    """Full pipeline should produce correct number of per-knot bounds."""
    from tt_sim.upgrades.tube_tighten import tighten_all_knots
    nj = 6
    N = 15
    nx = 2 * nj
    
    q_lo = -np.ones(nj) * 3.14
    q_hi = np.ones(nj) * 3.14
    qd_max = np.ones(nj) * 5.0
    tau_max = np.array([500, 500, 300, 150, 80, 50], dtype=float)
    
    Zs = [np.eye(nx) * 0.001 * k for k in range(N + 1)]
    Ks = [np.random.randn(nj, nx) * 0.01 for _ in range(N)]
    
    q_lo_l, q_hi_l, qd_lo_l, qd_hi_l, tau_l = tighten_all_knots(
        q_lo, q_hi, qd_max, tau_max, Zs, Ks
    )
    
    assert len(q_lo_l) == N
    assert len(tau_l) == N
    # All tau should be positive and <= original
    for k in range(N):
        assert np.all(tau_l[k] > 0)
        assert np.all(tau_l[k] <= tau_max)
