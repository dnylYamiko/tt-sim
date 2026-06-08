"""Tests for LTV-LQR tube gains."""
import numpy as np
import pytest


def test_ltv_lqr_dimensions():
    """Gains should have correct shape."""
    from tt_sim.upgrades.tube_lqr import compute_ltv_gains
    nj = 6
    nx, nu = 2*nj, nj
    N = 15
    
    rng = np.random.default_rng(42)
    # Create stable-ish A matrices (near identity)
    As = [np.eye(nx) + 0.01 * rng.standard_normal((nx, nx)) for _ in range(N)]
    Bs = [rng.standard_normal((nx, nu)) * 0.1 for _ in range(N)]
    Q = np.eye(nx)
    R = np.eye(nu) * 0.1
    
    Ks = compute_ltv_gains(As, Bs, Q, R)
    assert len(Ks) == N
    for k, K in enumerate(Ks):
        assert K.shape == (nu, nx), f"K[{k}] shape {K.shape} != {(nu, nx)}"


def test_ltv_lqr_stabilizes():
    """Closed-loop eigenvalues should be inside unit circle for LTI case."""
    from tt_sim.upgrades.tube_lqr import compute_ltv_gains
    nj = 6
    nx, nu = 2*nj, nj
    N = 15
    
    # Use a controllable double-integrator-like system (LTI, repeated)
    dt = 0.02
    A = np.eye(nx)
    A[:nj, nj:] = dt * np.eye(nj)  # position += dt * velocity
    B = np.zeros((nx, nu))
    B[nj:, :] = dt * np.eye(nj)    # velocity += dt * accel
    
    As = [A.copy() for _ in range(N)]
    Bs = [B.copy() for _ in range(N)]
    Q = np.eye(nx) * 10.0
    R = np.eye(nu) * 0.01
    
    Ks = compute_ltv_gains(As, Bs, Q, R)
    
    # Check all but last knot (last has minimal terminal cost influence)
    for k in range(N - 1):
        Acl = As[k] - Bs[k] @ Ks[k]
        eigvals = np.abs(np.linalg.eigvals(Acl))
        assert np.all(eigvals < 1.0), \
            f"Unstable at knot {k}: max |eig| = {eigvals.max():.4f}"


def test_tube_cross_sections_grow():
    """Tube should grow monotonically from zero."""
    from tt_sim.upgrades.tube_lqr import compute_ltv_gains, compute_tube_cross_sections
    nj = 6
    nx, nu = 2*nj, nj
    N = 10
    
    As = [np.eye(nx) * 0.95 for _ in range(N)]  # stable
    Bs = [np.vstack([np.zeros((nj, nj)), np.eye(nj)]) * 0.1 for _ in range(N)]
    Q = np.eye(nx)
    R = np.eye(nu)
    W = np.eye(nx) * 0.001  # small disturbance
    
    Ks = compute_ltv_gains(As, Bs, Q, R)
    Zs = compute_tube_cross_sections(As, Bs, Ks, W)
    
    assert len(Zs) == N + 1
    assert np.allclose(Zs[0], 0)  # starts at zero
    
    # Trace should grow (more uncertainty over time)
    traces = [np.trace(Z) for Z in Zs]
    for k in range(1, len(traces)):
        assert traces[k] >= traces[k-1] - 1e-10, \
            f"Tube shrank at k={k}: {traces[k]} < {traces[k-1]}"


def test_tube_cross_sections_bounded():
    """With stable closed-loop, tube should converge (not explode)."""
    from tt_sim.upgrades.tube_lqr import compute_ltv_gains, compute_tube_cross_sections
    nj = 6
    nx, nu = 2*nj, nj
    N = 50  # longer horizon to check convergence
    
    As = [np.eye(nx) * 0.9 for _ in range(N)]  # clearly stable
    Bs = [np.vstack([np.zeros((nj, nj)), np.eye(nj)]) * 0.5 for _ in range(N)]
    Q = np.eye(nx) * 10.0
    R = np.eye(nu)
    W = np.eye(nx) * 0.01
    
    Ks = compute_ltv_gains(As, Bs, Q, R)
    Zs = compute_tube_cross_sections(As, Bs, Ks, W)
    
    # Should not explode
    max_trace = max(np.trace(Z) for Z in Zs)
    assert max_trace < 100.0, f"Tube exploded: max trace = {max_trace}"
