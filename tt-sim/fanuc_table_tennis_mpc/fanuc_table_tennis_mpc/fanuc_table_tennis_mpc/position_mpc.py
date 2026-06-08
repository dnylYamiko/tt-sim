"""Position-level MPC for FANUC CRX-25iA table-tennis interception."""

import numpy as np
from scipy.optimize import minimize


def clamp(x, lower, upper):
    return np.minimum(np.maximum(x, lower), upper)


class SINDyPositionMPC:
    """
    SINDy-enhanced position-level MPC.

    Decision variable:
        U = [q_cmd_0, ..., q_cmd_N-1]

    Output:
        q_cmd_first, a 6-DOF joint-position command.
    """

    def __init__(self, robot, sindy_model, dt=0.02, horizon=10):
        self.robot = robot
        self.sindy = sindy_model
        self.dt = float(dt)
        self.N = int(horizon)
        self.nq = 6

        self.w_position = 150.0
        self.w_terminal = 600.0
        self.w_command_smoothness = 0.4
        self.w_joint_motion = 0.03
        self.w_velocity = 0.02
        self.w_posture = 0.02

        self.q_preferred = robot.q_home.copy()
        self.last_solution = None

    def unpack(self, u_flat):
        return np.asarray(u_flat).reshape(self.N, self.nq)

    def initial_guess(self, q_current):
        if self.last_solution is not None:
            shifted = np.vstack([self.last_solution[1:], self.last_solution[-1]])
            return shifted.flatten()
        return np.tile(q_current, (self.N, 1)).flatten()

    def predict_step(self, x, q_cmd):
        q_cmd = clamp(q_cmd, self.robot.q_min, self.robot.q_max)
        x_next = self.sindy.step(x, q_cmd, self.dt)
        q_next = clamp(x_next[:6], self.robot.q_min, self.robot.q_max)
        dq_next = clamp(x_next[6:], -self.robot.dq_max, self.robot.dq_max)
        return np.hstack([q_next, dq_next])

    def cost(self, u_flat, x_current, target_position):
        U = self.unpack(u_flat)
        x_pred = np.asarray(x_current, dtype=float).copy()
        target_position = np.asarray(target_position, dtype=float)
        previous_cmd = x_pred[:6].copy()
        total_cost = 0.0

        for k in range(self.N):
            q_cmd = U[k]
            x_next = self.predict_step(x_pred, q_cmd)
            q_next = x_next[:6]
            dq_next = x_next[6:]
            ee = self.robot.end_effector_position(q_next)
            err = ee - target_position

            total_cost += self.w_position * np.dot(err, err)
            total_cost += self.w_command_smoothness * np.dot(q_cmd - previous_cmd, q_cmd - previous_cmd)
            total_cost += self.w_joint_motion * np.dot(q_next - x_pred[:6], q_next - x_pred[:6])
            total_cost += self.w_velocity * np.dot(dq_next, dq_next)
            total_cost += self.w_posture * np.dot(q_next - self.q_preferred, q_next - self.q_preferred)

            x_pred = x_next
            previous_cmd = q_cmd

        terminal_err = self.robot.end_effector_position(x_pred[:6]) - target_position
        total_cost += self.w_terminal * np.dot(terminal_err, terminal_err)
        return total_cost

    def solve(self, x_current, target_position):
        x_current = np.asarray(x_current, dtype=float)
        u0 = self.initial_guess(x_current[:6])
        bounds = []
        for _ in range(self.N):
            for j in range(self.nq):
                bounds.append((self.robot.q_min[j], self.robot.q_max[j]))

        result = minimize(
            fun=self.cost,
            x0=u0,
            args=(x_current, target_position),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 40, "ftol": 1e-4, "disp": False},
        )
        U_opt = self.unpack(result.x)
        self.last_solution = U_opt.copy()
        return U_opt[0].copy(), float(result.fun)


class BaselinePositionMPC:
    """Simpler MPC using first-order servo prediction instead of SINDy."""

    def __init__(self, robot, dt=0.02, horizon=10):
        self.robot = robot
        self.dt = float(dt)
        self.N = int(horizon)
        self.nq = 6
        self.K_servo = np.array([8.0, 8.0, 8.0, 10.0, 10.0, 12.0])
        self.w_position = 150.0
        self.w_terminal = 600.0
        self.w_command_smoothness = 0.4
        self.w_joint_motion = 0.03
        self.w_posture = 0.02
        self.q_preferred = robot.q_home.copy()
        self.last_solution = None

    def unpack(self, u_flat):
        return np.asarray(u_flat).reshape(self.N, self.nq)

    def initial_guess(self, q_current):
        if self.last_solution is not None:
            shifted = np.vstack([self.last_solution[1:], self.last_solution[-1]])
            return shifted.flatten()
        return np.tile(q_current, (self.N, 1)).flatten()

    def predict_step(self, q, q_cmd):
        q_cmd = clamp(q_cmd, self.robot.q_min, self.robot.q_max)
        dq = self.K_servo * (q_cmd - q)
        dq = clamp(dq, -self.robot.dq_max, self.robot.dq_max)
        q_next = q + self.dt * dq
        return clamp(q_next, self.robot.q_min, self.robot.q_max)

    def cost(self, u_flat, q_current, target_position):
        U = self.unpack(u_flat)
        q_pred = np.asarray(q_current, dtype=float).copy()
        previous_cmd = q_pred.copy()
        target_position = np.asarray(target_position, dtype=float)
        total_cost = 0.0

        for k in range(self.N):
            q_cmd = U[k]
            q_next = self.predict_step(q_pred, q_cmd)
            err = self.robot.end_effector_position(q_next) - target_position
            total_cost += self.w_position * np.dot(err, err)
            total_cost += self.w_command_smoothness * np.dot(q_cmd - previous_cmd, q_cmd - previous_cmd)
            total_cost += self.w_joint_motion * np.dot(q_next - q_pred, q_next - q_pred)
            total_cost += self.w_posture * np.dot(q_next - self.q_preferred, q_next - self.q_preferred)
            q_pred = q_next
            previous_cmd = q_cmd

        final_err = self.robot.end_effector_position(q_pred) - target_position
        total_cost += self.w_terminal * np.dot(final_err, final_err)
        return total_cost

    def solve(self, q_current, target_position):
        u0 = self.initial_guess(q_current)
        bounds = [(self.robot.q_min[j], self.robot.q_max[j]) for _ in range(self.N) for j in range(self.nq)]
        result = minimize(
            fun=self.cost,
            x0=u0,
            args=(q_current, target_position),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 40, "ftol": 1e-4, "disp": False},
        )
        U_opt = self.unpack(result.x)
        self.last_solution = U_opt.copy()
        return U_opt[0].copy(), float(result.fun)
