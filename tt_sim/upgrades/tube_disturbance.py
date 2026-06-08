"""Empirical disturbance characterization for Tube MPC.

Computes bounding ellipsoid W from logged trajectory disturbances.
Disturbance sources:
  1. Ball prediction error → target position shift between replans
  2. Tracking error → deviation of actual state from planned trajectory
"""

import numpy as np
from typing import Tuple


def compute_disturbance_bounds(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    """Compute ellipsoidal disturbance bound from samples.
    
    Returns W such that w ~ N(0, W) captures `confidence` fraction of samples.
    W is the scaled covariance: W = cov(samples) * chi2_scale.
    
    The disturbance set is: {w : w' W^{-1} w <= 1}
    
    Args:
        samples: (N_samples, n_state) array of disturbance observations
        confidence: Fraction of samples to cover (0, 1)
    
    Returns:
        W: (n_state, n_state) PSD matrix defining the ellipsoid
    """
    assert samples.ndim == 2
    n_samples, n_state = samples.shape
    assert n_samples > n_state, "Need more samples than state dimensions"
    
    # Covariance
    cov = np.cov(samples.T)
    
    # Scale by chi-squared quantile
    from scipy.stats import chi2
    scale = chi2.ppf(confidence, df=n_state)
    
    W = cov * scale
    
    # Ensure PSD (regularize if needed)
    eigvals = np.linalg.eigvalsh(W)
    if eigvals.min() < 1e-10:
        W += np.eye(n_state) * 1e-8
    
    return W


def compute_tracking_disturbance(
    q_planned: np.ndarray,
    qd_planned: np.ndarray,
    q_actual: np.ndarray,
    qd_actual: np.ndarray,
) -> np.ndarray:
    """Compute state-space tracking errors as disturbance samples.
    
    Args:
        q_planned: (T, nj) planned joint positions
        qd_planned: (T, nj) planned joint velocities
        q_actual: (T, nj) actual joint positions
        qd_actual: (T, nj) actual joint velocities
    
    Returns:
        samples: (T, 2*nj) array of [delta_q, delta_qd] per timestep
    """
    delta_q = q_actual - q_planned
    delta_qd = qd_actual - qd_planned
    return np.hstack([delta_q, delta_qd])


def load_and_compute_W(npz_dir: str, confidence: float = 0.95) -> np.ndarray:
    """Load logged episode .npz files and compute disturbance bound.
    
    Each .npz should contain:
        'q': (T, nj) joint positions
        'dq': (T, nj) joint velocities  
        'action': (T, nj) applied torques
    
    Disturbance = difference between consecutive-step predicted state
    and actual observed state (one-step prediction error).
    """
    import os
    import glob
    
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {npz_dir}")
    
    all_errors = []
    dt = 0.008
    
    for f in files:
        data = np.load(f)
        q = data['q']      # (T, nj)
        dq = data['dq']    # (T, nj)
        nj = q.shape[1]
        
        # One-step prediction error: predict x_{k+1} from x_k using simple Euler
        # This captures the "process noise" w_k = x_{k+1}_actual - f(x_k, u_k)
        # For simplicity, use velocity prediction: q_pred = q_k + dt*dq_k
        for k in range(len(q) - 1):
            q_pred = q[k] + dt * dq[k]
            dq_pred = dq[k]  # assume constant velocity (no accel info)
            
            error = np.concatenate([
                q[k+1] - q_pred,
                dq[k+1] - dq_pred,
            ])
            all_errors.append(error)
    
    samples = np.array(all_errors)
    return compute_disturbance_bounds(samples, confidence)
