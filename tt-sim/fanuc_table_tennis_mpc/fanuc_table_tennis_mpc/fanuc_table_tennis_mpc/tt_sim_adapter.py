"""Adapters for converting tt_sim observations into MPC-ready state vectors."""

import numpy as np


class TTSimBallAdapter:
    """
    Converts tt_sim BallObservation objects into:
        [x, y, z, vx, vy, vz]

    Supported formats:
        - object with .position and optional .timestamp
        - dict with "position" and optional "timestamp"
        - array [x, y, z]
        - array [x, y, z, vx, vy, vz]
    """

    def __init__(self):
        self.prev_position = None
        self.prev_timestamp = None

    def reset(self):
        self.prev_position = None
        self.prev_timestamp = None

    def to_state(self, obs):
        if isinstance(obs, dict):
            if "state" in obs:
                arr = np.asarray(obs["state"], dtype=float)
                if arr.shape == (6,):
                    return arr
            position = np.asarray(obs["position"], dtype=float)
            timestamp = float(obs.get("timestamp", 0.0))
        elif hasattr(obs, "position"):
            position = np.asarray(obs.position, dtype=float)
            timestamp = float(getattr(obs, "timestamp", 0.0))
        else:
            arr = np.asarray(obs, dtype=float)
            if arr.shape == (6,):
                return arr
            if arr.shape == (3,):
                position = arr
                timestamp = 0.0
            else:
                raise ValueError(f"Unsupported ball observation shape: {arr.shape}")

        if position.shape != (3,):
            raise ValueError(f"Expected ball position shape (3,), got {position.shape}")

        if self.prev_position is None:
            velocity = np.zeros(3)
        else:
            dt = timestamp - self.prev_timestamp
            if dt <= 1e-8:
                velocity = np.zeros(3)
            else:
                velocity = (position - self.prev_position) / dt

        self.prev_position = position.copy()
        self.prev_timestamp = timestamp
        return np.hstack([position, velocity])


class GymStyleFANUCTTSimController:
    """
    Generic Gym/Gymnasium-style tt_sim wrapper.

    Expected observation dictionary:
        obs["robot_q"]
        obs["robot_dq"]
        obs["ball_observation"]

    The action returned to env.step(action) is q_cmd.
    """

    def __init__(self, env, robot, ball_predictor, intercept_planner, mpc):
        self.env = env
        self.robot = robot
        self.ball_predictor = ball_predictor
        self.intercept_planner = intercept_planner
        self.mpc = mpc
        self.ball_adapter = TTSimBallAdapter()

    @staticmethod
    def extract_robot_state(obs):
        q = np.asarray(obs["robot_q"], dtype=float)
        dq = np.asarray(obs["robot_dq"], dtype=float)
        return q, dq

    def extract_ball_state(self, obs):
        return self.ball_adapter.to_state(obs["ball_observation"])

    def compute_action(self, obs):
        q, dq = self.extract_robot_state(obs)
        ball_state = self.extract_ball_state(obs)
        ball_predictions = self.ball_predictor.predict(ball_state, horizon_steps=80)
        target_position, _ = self.intercept_planner.choose_intercept(ball_predictions)
        x_robot = np.hstack([q, dq])
        q_cmd, cost = self.mpc.solve(x_robot, target_position)
        q_cmd = np.clip(q_cmd, self.robot.q_min, self.robot.q_max)
        return q_cmd, target_position, cost

    def rollout(self, max_steps=500):
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.ball_adapter.reset()
        logs = []
        for step in range(max_steps):
            q_cmd, target_position, cost = self.compute_action(obs)
            step_out = self.env.step(q_cmd)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out
            logs.append({
                "step": step,
                "q_cmd": q_cmd,
                "target_position": target_position,
                "cost": cost,
                "reward": reward,
            })
            if done:
                break
        return logs
