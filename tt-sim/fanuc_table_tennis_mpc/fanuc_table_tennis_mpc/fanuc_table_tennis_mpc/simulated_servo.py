"""Simulated true FANUC-style position servo plant for offline testing and SINDy data generation."""

import numpy as np
from .crx25ia_kinematics import clamp


class TrueServoPlant:
    """
    Slightly nonlinear joint servo model.

    State:
        q, dq

    Input:
        q_cmd
    """

    def __init__(self, robot, dt=0.02):
        self.robot = robot
        self.dt = float(dt)
        self.q = robot.q_home.copy()
        self.dq = np.zeros(robot.n_joints)

        self.Kp = np.array([9.0, 8.5, 8.0, 10.0, 10.0, 12.0])
        self.D = np.array([1.2, 1.1, 1.1, 0.9, 0.8, 0.7])
        self.friction = np.array([0.10, 0.10, 0.08, 0.06, 0.05, 0.04])
        self.C = np.array([
            [0.00, 0.08, 0.04, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.06, 0.00, 0.00, 0.00],
            [0.03, 0.05, 0.00, 0.02, 0.00, 0.00],
            [0.00, 0.00, 0.02, 0.00, 0.04, 0.02],
            [0.00, 0.00, 0.00, 0.03, 0.00, 0.03],
            [0.00, 0.00, 0.00, 0.02, 0.02, 0.00],
        ])

    def reset(self, q0=None):
        if q0 is None:
            q0 = self.robot.q_home.copy()
        self.q = clamp(np.asarray(q0, dtype=float), self.robot.q_min, self.robot.q_max)
        self.dq = np.zeros(self.robot.n_joints)

    def dynamics(self, q, dq, q_cmd):
        q_cmd = clamp(q_cmd, self.robot.q_min, self.robot.q_max)
        error = q_cmd - q
        coupling = self.C @ error
        nonlinear_friction = self.friction * np.tanh(4.0 * dq)
        return self.Kp * error + coupling - self.D * dq - nonlinear_friction

    def step(self, q_cmd):
        ddq = self.dynamics(self.q, self.dq, q_cmd)
        self.dq = self.dq + self.dt * ddq
        self.dq = clamp(self.dq, -self.robot.dq_max, self.robot.dq_max)
        self.q = self.q + self.dt * self.dq
        self.q = clamp(self.q, self.robot.q_min, self.robot.q_max)
        return self.q.copy(), self.dq.copy()

    def get_state(self):
        return np.hstack([self.q, self.dq])
