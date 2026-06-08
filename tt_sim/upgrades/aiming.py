"""Aiming upgrades for tt-sim."""

from __future__ import annotations

import numpy as np

from tt_sim.interfaces import (
    Aimer,
    BallState,
    PaddleTarget,
    SpinEstimate,
)

# Standard table tennis table dimensions (meters)
TABLE_LENGTH = 2.74
TABLE_HEIGHT = 0.76


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
    FOLLOW_THROUGH: float = 0.3  # fraction of incoming speed for paddle velocity

    def __init__(self, target: np.ndarray | None = None) -> None:
        """Initialise with an optional landing target.

        Parameters
        ----------
        target : (3,) array or None
            Desired landing point on the opponent's side.  Defaults to the
            centre of the opponent's half of the table.
        """
        if target is not None:
            self.target = np.asarray(target, dtype=float)
        else:
            # Centre of opponent's half: x=-TABLE_LENGTH/4, y=0, z=TABLE_HEIGHT
            # Ball travels in +X; opponent's side is -X
            self.target = np.array([-TABLE_LENGTH / 4, 0.0, TABLE_HEIGHT])

    def aim(self, ball: BallState, spin: SpinEstimate) -> PaddleTarget:
        v_in = ball.velocity

        # Direction from contact position toward the target
        to_target = self.target - ball.position
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm < 1e-8:
            v_out_dir = np.array([0.0, 1.0, 0.0])
        else:
            v_out_dir = to_target / to_target_norm

        # Outgoing speed = restitution * incoming speed
        speed_in = np.linalg.norm(v_in)
        v_out = v_out_dir * (self.RESTITUTION * speed_in)

        # Paddle normal via specular reflection: n = normalize(v_out - v_in)
        diff = v_out - v_in
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-8:
            # v_in and v_out are (nearly) identical — face the net
            normal = np.array([0.0, 1.0, 0.0])
        else:
            normal = diff / diff_norm

        # Estimate time to contact (fallback; controllers should override this
        # with the actual remaining time to intercept)
        if speed_in > 0.1:
            # Use distance from ball to robot-side table edge as rough proxy
            dx = max(1.37 - ball.position[0], 0.1)  # table +X edge at 1.37
            t_contact = max(dx / speed_in, 0.1)
        else:
            t_contact = 0.5

        # Follow-through: paddle moves along its normal at contact
        paddle_speed = self.FOLLOW_THROUGH * speed_in
        paddle_velocity = normal * paddle_speed

        return PaddleTarget(
            position=ball.position.copy(),
            normal=normal,
            velocity=paddle_velocity,
            t_contact=t_contact,
        )
