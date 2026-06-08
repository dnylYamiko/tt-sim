"""Aiming upgrades for tt-sim."""

from __future__ import annotations

import numpy as np

from tt_sim.interfaces import (
    Aimer,
    BallState,
    PaddleTarget,
    SpinEstimate,
)


class SpecularAimer(Aimer):
    """Stage 1 – Specular-reflection aimer.

    Algorithm
    ---------
    Computes the paddle face normal required to redirect the incoming ball
    velocity toward a chosen target point on the opponent's side of the
    table, using the specular (mirror) reflection law.

    Assumes **zero spin** on the incoming ball so that the outgoing velocity
    lies in the plane of incidence.

    Mathematics
    -----------
    Given incoming ball velocity *v_in* and desired outgoing velocity
    *v_out* (directed toward the target point):

        n = normalize(v_out - v_in)

    The paddle velocity is set to zero (static contact assumption) and
    ``t_contact`` is taken from the predicted ball state.

    Implementation
    --------------
    * ``v_out`` direction is computed as the unit vector from the predicted
      contact position toward the target point on the opponent's half.
    * ``v_out`` magnitude is scaled by the coefficient of restitution
      (e = 0.91) to conserve energy approximately.

    References
    ----------
    * Standard geometric optics / specular reflection.

    Getting Started
    ---------------
    1. Set the desired landing target (default: centre of opponent's half).
    2. Call ``aim(ball, spin)`` with the predicted ``BallState`` at contact.
    3. Returns a ``PaddleTarget`` with position, normal, velocity, t_contact.
    """

    # Physics constants
    RESTITUTION: float = 0.91

    def aim(self, ball: BallState, spin: SpinEstimate) -> PaddleTarget:
        raise NotImplementedError("SpecularAimer.aim")
