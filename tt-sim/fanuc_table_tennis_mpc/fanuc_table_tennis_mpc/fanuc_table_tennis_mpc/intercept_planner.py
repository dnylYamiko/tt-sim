"""Intercept selection for a FANUC CRX-25iA table-tennis setup."""

import numpy as np


class InterceptPlanner:
    """Choose a reachable ball interception point."""

    def __init__(self):
        self.x_range = (0.55, 1.65)
        self.y_range = (-0.70, 0.70)
        self.z_range = (0.35, 1.30)
        self.workspace_center = np.array([1.15, 0.0, 0.80], dtype=float)

    def is_in_zone(self, p):
        x, y, z = np.asarray(p, dtype=float)
        return (
            self.x_range[0] <= x <= self.x_range[1]
            and self.y_range[0] <= y <= self.y_range[1]
            and self.z_range[0] <= z <= self.z_range[1]
        )

    def choose_intercept(self, ball_predictions):
        ball_predictions = np.asarray(ball_predictions, dtype=float)
        for k, state in enumerate(ball_predictions):
            p = state[:3]
            if self.is_in_zone(p):
                return p.copy(), k

        distances = np.linalg.norm(ball_predictions[:, :3] - self.workspace_center, axis=1)
        k_best = int(np.argmin(distances))
        return ball_predictions[k_best, :3].copy(), k_best
