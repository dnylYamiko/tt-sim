"""Swing-planning upgrades for tt-sim."""

from __future__ import annotations

from typing import Callable

import numpy as np

from tt_sim.interfaces import (
    JointTrajectory,
    PaddleTarget,
    SwingPlanner,
)


class QuinticSwingPlanner(SwingPlanner):
    """Stage 1 – Quintic polynomial swing planner.

    Algorithm
    ---------
    Generates a smooth joint-space trajectory using a **quintic (5th-order)
    polynomial** for each joint independently.  Six boundary conditions
    (position, velocity, acceleration at start and end) fully determine the
    six polynomial coefficients per joint.

    Mathematics
    -----------
    For each joint *j*, the trajectory is:

        q_j(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵

    Boundary conditions at t=0 (start) and t=T (end):

        q(0)   = q_start,    q(T)   = q_end
        dq(0)  = dq_start,   dq(T)  = dq_end
        ddq(0) = ddq_start,  ddq(T) = ddq_end

    These yield a **6×6 linear system** per joint, solved via
    ``numpy.linalg.solve``.

    Implementation
    --------------
    * Inverse kinematics maps ``PaddleTarget`` → desired joint angles.
    * Start state is ``q_current`` with zero velocity / acceleration.
    * End velocity from ``PaddleTarget.velocity`` via Jacobian (if
      ``jac_fn`` provided), otherwise zero.
    * End acceleration is zero.
    * Time horizon is ``target.t_contact``.

    References
    ----------
    * Craig, J. J. *Introduction to Robotics: Mechanics and Planning*,
      Chapter 7 – Trajectory Generation.

    Getting Started
    ---------------
    1. Instantiate (optionally supply IK solver or DH parameters).
    2. Call ``plan(target, q_current)`` → ``JointTrajectory``.
    3. Feed the trajectory to the robot's joint-level controller.
    """

    def __init__(
        self,
        n_joints: int = 7,
        n_steps: int = 50,
        ik_fn: Callable[[PaddleTarget], np.ndarray] | None = None,
        jac_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """Initialise the quintic swing planner.

        Parameters
        ----------
        n_joints : int
            Number of robot joints.
        n_steps : int
            Number of trajectory waypoints to generate.
        ik_fn : callable or None
            Inverse kinematics: ``ik_fn(target) -> q_target (n_joints,)``.
            Defaults to returning zeros (for unit testing).
        jac_fn : callable or None
            Jacobian function: ``jac_fn(q) -> J (3, n_joints)``.
            Used to convert Cartesian paddle velocity to joint velocity.
            If None, end velocity is set to zero.
        """
        self.n_joints = n_joints
        self.n_steps = n_steps
        self.ik_fn = ik_fn or (lambda target: np.zeros(n_joints))
        self.jac_fn = jac_fn

    @staticmethod
    def _quintic_coeffs(
        q0: float, qf: float,
        dq0: float, dqf: float,
        ddq0: float, ddqf: float,
        T: float,
    ) -> np.ndarray:
        """Solve for quintic polynomial coefficients [a0..a5].

        The 6x6 system from boundary conditions at t=0 and t=T:

            q(0)=q0, dq(0)=dq0, ddq(0)=ddq0
            q(T)=qf, dq(T)=dqf, ddq(T)=ddqf
        """
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        A = np.array([
            [1, 0,   0,    0,     0,     0    ],
            [0, 1,   0,    0,     0,     0    ],
            [0, 0,   2,    0,     0,     0    ],
            [1, T,   T2,   T3,    T4,    T5   ],
            [0, 1, 2*T, 3*T2,  4*T3,  5*T4   ],
            [0, 0,   2, 6*T,  12*T2, 20*T3   ],
        ])

        b = np.array([q0, dq0, ddq0, qf, dqf, ddqf])
        return np.linalg.solve(A, b)

    def plan(self, target: PaddleTarget, q_current: np.ndarray, dq_current: np.ndarray | None = None) -> JointTrajectory:
        # Pass q_current as seed to IK for solution continuity (avoids branch jumps)
        try:
            q_target = self.ik_fn(target, q_seed=q_current)
        except TypeError:
            q_target = self.ik_fn(target)

        # Make DOF-agnostic: match q_target length to q_current
        ndof = len(q_current)
        if len(q_target) > ndof:
            q_target = q_target[:ndof]
        elif len(q_target) < ndof:
            q_target = np.concatenate([q_target, np.zeros(ndof - len(q_target))])

        T = max(target.t_contact, 0.01)

        # Start boundary: use provided velocity or zero (at rest)
        dq_start = dq_current if dq_current is not None else np.zeros(ndof)
        ddq_start = np.zeros(ndof)

        # End boundary: convert Cartesian paddle velocity to joint velocity
        dq_end = np.zeros(ndof)
        if (
            self.jac_fn is not None
            and target.velocity is not None
            and np.linalg.norm(target.velocity) > 1e-8
        ):
            J = self.jac_fn(q_target)
            # Damped least-squares pseudoinverse: dq = J^T (J J^T + λ²I)^{-1} v
            damping = 0.01
            JJT = J @ J.T + damping**2 * np.eye(J.shape[0])
            dq_end = J.T @ np.linalg.solve(JJT, target.velocity)

        ddq_end = np.zeros(ndof)

        # Solve quintic coefficients for each joint
        coeffs = np.zeros((ndof, 6))
        for j in range(ndof):
            coeffs[j] = self._quintic_coeffs(
                q_current[j], q_target[j],
                dq_start[j], dq_end[j],
                ddq_start[j], ddq_end[j],
                T,
            )

        # Evaluate trajectory at n_steps evenly spaced times
        times = np.linspace(0.0, T, self.n_steps)
        positions = np.zeros((self.n_steps, ndof))
        velocities = np.zeros((self.n_steps, ndof))

        for i, t in enumerate(times):
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t

            # q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
            tvec = np.array([1, t, t2, t3, t4, t5])
            positions[i] = coeffs @ tvec

            # dq(t) = a1 + 2*a2*t + 3*a3*t² + 4*a4*t³ + 5*a5*t⁴
            dvec = np.array([0, 1, 2*t, 3*t2, 4*t3, 5*t4])
            velocities[i] = coeffs @ dvec

        return JointTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
        )
