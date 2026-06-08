"""Constraint tightening for Tube MPC.

Given tube cross-sections Z_k, tighten joint/torque bounds to ensure
the nominal trajectory stays feasible even when the actual state deviates
within the tube.

Tightening: X_tightened = X ⊖ Z  (Pontryagin difference)
For box constraints: subtract the maximum extent of Z along each axis.
"""

import numpy as np
from typing import Tuple, List


def tighten_box_constraints(
    lb: np.ndarray,
    ub: np.ndarray,
    Z: np.ndarray,
    margin_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tighten box constraints [lb, ub] by the extent of ellipsoid Z.
    
    For box constraints, the Pontryagin difference with an ellipsoid
    reduces each bound by the support function of Z in that direction:
        h_Z(e_i) = sqrt(e_i' Z e_i) = sqrt(Z_ii)
    
    Args:
        lb: Lower bounds (n,)
        ub: Upper bounds (n,)
        Z: Tube cross-section matrix (n x n), PSD
        margin_fraction: How much of the tube to subtract (0.8 = conservative)
    
    Returns:
        (lb_tight, ub_tight): Tightened bounds
    """
    n = len(lb)
    assert Z.shape == (n, n)
    
    # Support function of ellipsoid along coordinate axes = sqrt(diagonal)
    extent = np.sqrt(np.maximum(np.diag(Z), 0.0)) * margin_fraction
    
    lb_tight = lb + extent
    ub_tight = ub - extent
    
    # Ensure feasibility: if tightened bounds cross, use midpoint
    for i in range(n):
        if lb_tight[i] >= ub_tight[i]:
            mid = 0.5 * (lb[i] + ub[i])
            lb_tight[i] = mid - 1e-4
            ub_tight[i] = mid + 1e-4
    
    return lb_tight, ub_tight


def tighten_all_knots(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    qd_max: np.ndarray,
    tau_max: np.ndarray,
    Zs: List[np.ndarray],
    Ks: List[np.ndarray],
    margin_fraction: float = 0.8,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Tighten constraints at each knot based on tube cross-sections.
    
    Args:
        q_lo, q_hi: Joint position limits (nj,)
        qd_max: Joint velocity limits (nj,) — symmetric: [-qd_max, qd_max]
        tau_max: Torque limits (nj,) — symmetric: [-tau_max, tau_max]
        Zs: List of N+1 tube cross-sections (2*nj x 2*nj)
        Ks: List of N gain matrices (nj x 2*nj)
        margin_fraction: How aggressively to tighten (0.8 default)
    
    Returns:
        (q_lo_list, q_hi_list, qd_lo_list, qd_hi_list, tau_max_list):
        Per-knot tightened bounds.
    """
    N = len(Ks)
    nj = len(q_lo)
    
    q_lo_list = []
    q_hi_list = []
    qd_lo_list = []
    qd_hi_list = []
    tau_max_list = []
    
    for k in range(N):
        Z_k = Zs[k + 1]  # Z_0=0, use Z_{k+1} for knot k+1 constraints
        
        # State tightening: Z_k is (2*nj x 2*nj), extract q and qd blocks
        Z_q = Z_k[:nj, :nj]        # position block
        Z_qd = Z_k[nj:, nj:]       # velocity block
        
        # Tighten joint positions
        q_lo_k, q_hi_k = tighten_box_constraints(q_lo, q_hi, Z_q, margin_fraction)
        q_lo_list.append(q_lo_k)
        q_hi_list.append(q_hi_k)
        
        # Tighten joint velocities
        qd_lo_k, qd_hi_k = tighten_box_constraints(-qd_max, qd_max, Z_qd, margin_fraction)
        qd_lo_list.append(qd_lo_k)
        qd_hi_list.append(qd_hi_k)
        
        # Tighten torque limits: u_tube = K_k @ z, max |u_tube_i| = |K_k[i,:]| @ sqrt(diag(Z_k))
        # More precisely: support of K Z K' along e_i = sqrt((K Z K')[i,i])
        K_k = Ks[k]
        KZKt = K_k @ Z_k @ K_k.T  # (nj x nj)
        tau_extent = np.sqrt(np.maximum(np.diag(KZKt), 0.0)) * margin_fraction
        tau_max_k = np.maximum(tau_max - tau_extent, tau_max * 0.1)  # keep at least 10%
        tau_max_list.append(tau_max_k)
    
    return q_lo_list, q_hi_list, qd_lo_list, qd_hi_list, tau_max_list
