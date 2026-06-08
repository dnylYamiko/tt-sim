"""Tests for disturbance characterization."""
import numpy as np
import pytest


def test_bounding_ellipsoid_shape():
    """W should be symmetric PSD with correct dimensions."""
    from tt_sim.upgrades.tube_disturbance import compute_disturbance_bounds
    rng = np.random.default_rng(42)
    n_state = 12
    samples = rng.standard_normal((200, n_state)) * 0.01
    
    W = compute_disturbance_bounds(samples, confidence=0.95)
    
    assert W.shape == (n_state, n_state)
    assert np.allclose(W, W.T), "W not symmetric"
    eigvals = np.linalg.eigvalsh(W)
    assert np.all(eigvals > 0), f"W not PD: min eig = {eigvals.min()}"


def test_bounding_ellipsoid_covers_data():
    """Most samples should lie inside the ellipsoid."""
    from tt_sim.upgrades.tube_disturbance import compute_disturbance_bounds
    rng = np.random.default_rng(42)
    n_state = 12
    samples = rng.standard_normal((1000, n_state)) * 0.01
    
    W = compute_disturbance_bounds(samples, confidence=0.95)
    W_inv = np.linalg.inv(W)
    
    # Check coverage: x' W^{-1} x <= 1 means inside
    inside = sum(1 for s in samples if s @ W_inv @ s <= 1.0)
    coverage = inside / len(samples)
    # Should be approximately 0.95
    assert coverage > 0.90, f"Coverage {coverage:.2f} < 0.90"


def test_tracking_disturbance():
    """Tracking disturbance should have correct shape."""
    from tt_sim.upgrades.tube_disturbance import compute_tracking_disturbance
    T, nj = 100, 6
    rng = np.random.default_rng(42)
    q_plan = rng.standard_normal((T, nj))
    qd_plan = rng.standard_normal((T, nj))
    q_act = q_plan + rng.standard_normal((T, nj)) * 0.001
    qd_act = qd_plan + rng.standard_normal((T, nj)) * 0.01
    
    samples = compute_tracking_disturbance(q_plan, qd_plan, q_act, qd_act)
    assert samples.shape == (T, 2*nj)
