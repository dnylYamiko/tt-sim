"""Dynamics linearization for Tube MPC.

Computes A_k = df/dx, B_k = df/du at a given (q, qd, tau) using CasADi autodiff.
"""

import casadi as ca
import numpy as np


def build_linearization_fn(dynamics_dict: dict, nj: int, dt: float = 0.008):
    """Build CasADi function that returns discrete-time A, B matrices.
    
    Linearizes the semi-implicit Euler dynamics:
        qdd = M(q)^{-1} (tau - C(q,qd)*qd - g(q))
        qd+ = qd + dt*qdd
        q+  = q + dt*qd+
    
    State x = [q; qd], Input u = tau.
    Returns A (2nj x 2nj), B (2nj x nj).
    """
    q_sym = ca.SX.sym('q', nj)
    qd_sym = ca.SX.sym('qd', nj)
    tau_sym = ca.SX.sym('tau', nj)
    
    # Get symbolic dynamics components
    M_fn = dynamics_dict['mass_matrix_fn']
    
    # Evaluate M, compute forward dynamics
    M = M_fn(q_sym)  # nj x nj
    
    # Coriolis + gravity via RNEA: rnea(q, qd, 0) = C*qd + g
    rnea_fn = dynamics_dict['rnea_fn']
    bias = rnea_fn(q_sym, qd_sym, ca.SX.zeros(nj))  # C*qd + g
    
    # Forward dynamics: qdd = M^{-1}(tau - bias)
    qdd = ca.solve(M, tau_sym - bias)
    
    # Semi-implicit Euler integration
    qd_next = qd_sym + dt * qdd
    q_next = q_sym + dt * qd_next
    
    # State transition: x_next = f(x, u)
    x = ca.vertcat(q_sym, qd_sym)
    x_next = ca.vertcat(q_next, qd_next)
    
    # Jacobians via CasADi autodiff
    A_sym = ca.jacobian(x_next, x)     # (2nj x 2nj)
    B_sym = ca.jacobian(x_next, tau_sym)  # (2nj x nj)
    
    # Create callable CasADi function
    _lin_fn = ca.Function('linearize', 
                          [q_sym, qd_sym, tau_sym], 
                          [A_sym, B_sym],
                          ['q', 'qd', 'tau'], ['A', 'B'])
    
    def linearize(q: np.ndarray, qd: np.ndarray, tau: np.ndarray):
        """Evaluate linearization at a point. Returns (A, B) as numpy arrays."""
        A_val, B_val = _lin_fn(q, qd, tau)
        return np.array(A_val, dtype=np.float64), np.array(B_val, dtype=np.float64)
    
    # Expose the CasADi function too for batch evaluation
    linearize.casadi_fn = _lin_fn
    return linearize
