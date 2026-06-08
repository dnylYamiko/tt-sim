"""
Approximate FANUC CRX-25iA kinematics for academic simulation.

This is not an official FANUC model. Replace this with an official URDF,
ROBOGUIDE export, or calibrated kinematic model before real robot deployment.
"""

import numpy as np


def clamp(x, lower, upper):
    return np.minimum(np.maximum(x, lower), upper)


class FANUCCRX25iA:
    """
    Approximate 6-axis FANUC CRX-25iA-style robot model.

    Units:
        - metres
        - radians
    """

    def __init__(self):
        self.name = "FANUC CRX-25iA approximate model"
        self.n_joints = 6
        self.nominal_reach_m = 1.889

        # Standard DH-style parameters: [a, alpha, d, theta_offset]
        # These are approximate, not official FANUC data.
        self.dh_params = [
            [0.000, np.pi / 2.0, 0.420, 0.0],
            [0.760, 0.0,         0.000, 0.0],
            [0.670, 0.0,         0.000, 0.0],
            [0.000, np.pi / 2.0, 0.260, 0.0],
            [0.000, -np.pi / 2.0, 0.000, 0.0],
            [0.000, 0.0,         0.180, 0.0],
        ]

        self.q_min = np.deg2rad(np.array([-180, -120, -150, -190, -125, -360], dtype=float))
        self.q_max = np.deg2rad(np.array([ 180,  120,  150,  190,  125,  360], dtype=float))
        self.dq_max = np.deg2rad(np.array([100, 100, 100, 160, 160, 220], dtype=float))

        self.q_home = np.deg2rad(np.array([0.0, 35.0, -45.0, 0.0, 55.0, 0.0]))

    @staticmethod
    def dh_transform(a, alpha, d, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,     sa,      ca,      d],
            [0.0,    0.0,     0.0,    1.0],
        ])

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            T = T @ self.dh_transform(a, alpha, d, q[i] + theta_offset)
        return T

    def end_effector_position(self, q):
        return self.forward_kinematics(q)[:3, 3]

    def end_effector_rotation(self, q):
        return self.forward_kinematics(q)[:3, :3]

    def joint_positions(self, q):
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        points = [T[:3, 3].copy()]
        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            T = T @ self.dh_transform(a, alpha, d, q[i] + theta_offset)
            points.append(T[:3, 3].copy())
        return np.array(points)

    def numerical_jacobian(self, q, eps=1e-6):
        q = np.asarray(q, dtype=float)
        J = np.zeros((3, self.n_joints))
        p0 = self.end_effector_position(q)
        for i in range(self.n_joints):
            qp = q.copy()
            qp[i] += eps
            J[:, i] = (self.end_effector_position(qp) - p0) / eps
        return J
