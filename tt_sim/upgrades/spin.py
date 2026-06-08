"""Spin-estimation upgrades for tt-sim."""

from __future__ import annotations

import numpy as np

from tt_sim.interfaces import (
    BallObservation,
    SpinEstimate,
    SpinEstimator,
)


# ── Physics constants ────────────────────────────────────────────────────────

C_D: float = 0.4
C_L: float = 0.55          # Magnus lift coefficient
RHO_AIR: float = 1.225     # kg/m^3
BALL_MASS: float = 0.0027  # kg
BALL_RADIUS: float = 0.02  # m
BALL_AREA: float = np.pi * BALL_RADIUS ** 2
GRAVITY: np.ndarray = np.array([0.0, 0.0, -9.81])


class MagnusSpinEstimator(SpinEstimator):
    """Stage 1 – Least-squares Magnus-inversion spin estimator.

    Algorithm
    ---------
    Estimates spin angular velocity **ω** by inverting the Magnus force
    equation.  Given observed trajectory accelerations (from finite
    differences of position data), subtract gravity and drag to isolate the
    Magnus contribution, then solve for ω via least squares.

    Mathematics
    -----------
    The equation of motion for a spinning ball is:

        m * a = m * g  -  (C_D ρ A / 2) |v| v  +  C_M (ω × v)

    where C_M = C_L · ρ · A · R / 2  (Magnus coefficient).

    Rearranging:

        a - g - drag_term = (C_M / m) (ω × v)

    The cross product ω × v is linear in ω, giving a linear system
    **A ω = b** solved via ``numpy.linalg.lstsq``.

    Accelerations are estimated from positions using **Savitzky–Golay**
    smoothing (``scipy.signal.savgol_filter``, window=5, polyorder=2).

    Implementation
    --------------
    * Savitzky–Golay filter for velocity and acceleration estimation.
    * Assemble over-determined system from multiple time steps.
    * ``numpy.linalg.lstsq`` for robust solution.
    * Confidence derived from residual norm.

    References
    ----------
    * Nguyen, T. et al. (2025). "Real-time spin estimation for table tennis
      robots." *IEEE RA-L*.
    * Zhang, Y. et al. (2014). "Spin estimation of ping-pong ball from
      trajectory analysis." *ROBIO*.

    Datasets
    --------
    Kienzle 50 k trajectory dataset with ground-truth spin labels.

    Getting Started
    ---------------
    1. Collect ≥ 8 observations spanning the flight phase.
    2. Call ``estimate(observations)`` → ``SpinEstimate(omega, confidence)``.
    3. Use the confidence vector to gate downstream spin-aware modules.
    """

    def estimate(self, observations: list[BallObservation]) -> SpinEstimate:
        raise NotImplementedError("MagnusSpinEstimator.estimate")
