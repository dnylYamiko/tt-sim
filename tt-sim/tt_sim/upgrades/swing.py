"""Swing-planning upgrades for tt-sim."""

from __future__ import annotations

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
    * End state targets zero velocity / acceleration at contact.
    * Time horizon is ``target.t_contact - t_now``.

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

    def plan(self, target: PaddleTarget, q_current: np.ndarray) -> JointTrajectory:
        raise NotImplementedError("QuinticSwingPlanner.plan")
