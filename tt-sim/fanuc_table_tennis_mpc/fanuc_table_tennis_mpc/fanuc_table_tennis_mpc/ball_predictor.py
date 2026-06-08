"""Table-tennis ball trajectory prediction with gravity and drag."""

import numpy as np


class BallTrajectoryPredictor:
    """
    Ball state:
        [x, y, z, vx, vy, vz]
    """

    def __init__(self, dt=0.02, drag_coeff=0.05, gravity=9.81):
        self.dt = float(dt)
        self.drag_coeff = float(drag_coeff)
        self.g = float(gravity)

    def dynamics(self, state):
        state = np.asarray(state, dtype=float)
        v = state[3:]
        speed = np.linalg.norm(v) + 1e-9
        drag_acc = -self.drag_coeff * speed * v
        acc = np.array([drag_acc[0], drag_acc[1], drag_acc[2] - self.g])
        return np.hstack([v, acc])

    def step(self, state):
        state = np.asarray(state, dtype=float)
        dt = self.dt
        k1 = self.dynamics(state)
        k2 = self.dynamics(state + 0.5 * dt * k1)
        k3 = self.dynamics(state + 0.5 * dt * k2)
        k4 = self.dynamics(state + dt * k3)
        next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if next_state[2] < 0.0:
            next_state[2] = 0.0
            next_state[5] = 0.0
        return next_state

    def predict(self, initial_state, horizon_steps):
        initial_state = np.asarray(initial_state, dtype=float)
        states = np.zeros((int(horizon_steps) + 1, 6))
        states[0] = initial_state.copy()
        for k in range(int(horizon_steps)):
            states[k + 1] = self.step(states[k])
        return states
