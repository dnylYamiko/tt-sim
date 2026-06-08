"""Baseline perception module."""

import numpy as np

from tt_sim.interfaces import BallObservation, Perceiver


class SimPerceiver(Perceiver):
    """Stage 0 perception: reads ground truth ball position from sim."""

    def observe(self, env_state: dict) -> BallObservation:
        return BallObservation(
            position=np.array(env_state["ball_pos"], dtype=np.float64),
            sigma=0.0,
            timestamp=env_state.get("time", 0.0),
        )
