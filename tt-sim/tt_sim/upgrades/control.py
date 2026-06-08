"""High-level controller upgrades for tt-sim."""

from __future__ import annotations

import numpy as np

from tt_sim.interfaces import (
    Aimer,
    HighLevelController,
    Perceiver,
    Predictor,
    SpinEstimator,
    SwingPlanner,
)


class ReplanController(HighLevelController):
    """Stage 1 – Re-planning closed-loop controller.

    Algorithm
    ---------
    On **every control frame**, re-runs the full perception → prediction →
    spin estimation → aiming → swing planning pipeline with the latest
    sensor data.  This provides closed-loop correction for prediction errors
    and disturbances, at the cost of higher computation per frame.

    The constructor accepts the same component interfaces as an
    ``OpenLoopController``: perceiver, predictor, spin_estimator, aimer,
    and swing_planner.  A ``reset()`` method clears internal state between
    rallies.

    Implementation
    --------------
    * Stores a rolling window of ``BallObservation`` objects.
    * Each ``step()`` call:
      1. ``perceiver.observe(env_state)`` → new observation.
      2. ``predictor.predict(observations, t_future)`` → predicted state.
      3. ``spin_estimator.estimate(observations)`` → spin.
      4. ``aimer.aim(ball, spin)`` → paddle target.
      5. ``swing_planner.plan(target, q_current)`` → trajectory.
      6. Return first waypoint joint positions from the trajectory.
    * ``reset()`` clears observation history.

    Getting Started
    ---------------
    1. Instantiate with concrete implementations of each pipeline stage.
    2. Call ``step(env_state, q_current)`` in your control loop.
    3. Call ``reset()`` between rallies / episodes.
    """

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
    ) -> None:
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self._observations: list = []

    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ReplanController.step")

    def reset(self) -> None:
        """Clear observation history between rallies."""
        self._observations.clear()
