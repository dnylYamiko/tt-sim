"""LTV-LQR gain computation for Tube MPC.

Given time-varying linearized dynamics (A_k, B_k), computes stabilizing
feedback gains K_k via backward discrete-time Riccati recursion.
"""

import numpy as np
from typing import List, Tuple


def compute_ltv_gains(
    As: List[np.ndarray],
    Bs: List[np.ndarray],
    Q: np.ndarray,
    R: np.ndarray,
    Q_terminal: np.ndarray | None = None,
) -> List[np.ndarray]:
    """Backward Riccati recursion for time-varying LQR gains.
    
    Args:
        As: List of N state transition matrices (nx x nx)
        Bs: List of N input matrices (nx x nu)
        Q: State cost matrix (nx x nx), symmetric PSD
        R: Input cost matrix (nu x nu), symmetric PD
        Q_terminal: Terminal state cost (default: Q)
    
    Returns:
        List of N gain matrices K_k (nu x nx) such that
        u_k = -K_k @ (x_k - x_nom_k) stabilizes the system.
    """
    N = len(As)
    assert len(Bs) == N
    
    if Q_terminal is None:
        Q_terminal = Q.copy()
    
    Ks = [None] * N
    P = Q_terminal.copy()
    
    for k in range(N - 1, -1, -1):
        A, B = As[k], Bs[k]
        # K_k = (R + B' P B)^{-1} B' P A
        BtP = B.T @ P
        S = R + BtP @ B
        Ks[k] = np.linalg.solve(S, BtP @ A)
        # P_k = Q + A' P A - A' P B K_k  (simplified)
        # Equivalently: P = Q + (A - B K)' P (A - B K) + K' R K
        Acl = A - B @ Ks[k]
        P = Q + Acl.T @ P @ Acl + Ks[k].T @ R @ Ks[k]
    
    return Ks


def compute_tube_cross_sections(
    As: List[np.ndarray],
    Bs: List[np.ndarray],
    Ks: List[np.ndarray],
    W: np.ndarray,
) -> List[np.ndarray]:
    """Propagate disturbance set through closed-loop to get tube cross-sections.
    
    The tube cross-section Z_k represents the set of possible deviations
    from the nominal trajectory at time k, given bounded disturbance w ∈ W.
    
    For ellipsoidal sets (W = {x : x' W^{-1} x <= 1}):
        Z_{k+1} = (A_k - B_k K_k) Z_k (A_k - B_k K_k)' + W
    
    Args:
        As: State matrices
        Bs: Input matrices
        Ks: LQR gains
        W: Disturbance ellipsoid matrix (nx x nx), PSD
    
    Returns:
        List of N+1 tube cross-section matrices Z_k (nx x nx), PSD.
        Z_0 = 0 (start exactly on nominal), Z_k grows over time.
    """
    N = len(As)
    nx = As[0].shape[0]
    
    Zs = [np.zeros((nx, nx))]  # Z_0 = 0
    
    for k in range(N):
        Acl = As[k] - Bs[k] @ Ks[k]
        Z_next = Acl @ Zs[-1] @ Acl.T + W
        # Symmetrize to avoid numerical drift
        Z_next = 0.5 * (Z_next + Z_next.T)
        Zs.append(Z_next)
    
    return Zs
