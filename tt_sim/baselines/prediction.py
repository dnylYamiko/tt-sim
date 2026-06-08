"""Baseline ball trajectory prediction."""

from tt_sim.interfaces import Predictor, BallObservation, BallState
import numpy as np

GRAVITY = 9.81  # m/s²


class BallisticPredictor(Predictor):
    """Stage 0 prediction: ballistic parabola (no drag, no spin)."""

    def predict(self, observations: list[BallObservation], t_future: float) -> BallState:
        if len(observations) < 2:
            raise ValueError("Need at least 2 observations for ballistic prediction")

        times = np.array([o.timestamp for o in observations])
        positions = np.array([o.position for o in observations])

        t0 = times[0]
        dt = times - t0

        A_linear = np.column_stack([np.ones_like(dt), dt])  # (N, 2)

        sol_x, _, _, _ = np.linalg.lstsq(A_linear, positions[:, 0], rcond=None)
        sol_y, _, _, _ = np.linalg.lstsq(A_linear, positions[:, 1], rcond=None)

        z_corrected = positions[:, 2] + 0.5 * GRAVITY * dt**2
        sol_z, _, _, _ = np.linalg.lstsq(A_linear, z_corrected, rcond=None)

        dt_pred = t_future - t0

        pos = np.array([
            sol_x[0] + sol_x[1] * dt_pred,
            sol_y[0] + sol_y[1] * dt_pred,
            sol_z[0] + sol_z[1] * dt_pred - 0.5 * GRAVITY * dt_pred**2,
        ])

        vel = np.array([
            sol_x[1],
            sol_y[1],
            sol_z[1] - GRAVITY * dt_pred,
        ])

        return BallState(position=pos, velocity=vel)
