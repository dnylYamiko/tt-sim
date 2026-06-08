"""Generate SINDy training data for FANUC-style position servo dynamics."""

import numpy as np
from .crx25ia_kinematics import clamp
from .simulated_servo import TrueServoPlant


def finite_difference(X, dt):
    X = np.asarray(X, dtype=float)
    X_dot = np.zeros_like(X)
    X_dot[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    X_dot[0] = (X[1] - X[0]) / dt
    X_dot[-1] = (X[-1] - X[-2]) / dt
    return X_dot


def generate_sindy_training_data(robot, dt=0.02, n_episodes=30, episode_steps=180, seed=2):
    rng = np.random.default_rng(seed)
    plant = TrueServoPlant(robot, dt=dt)
    X_all, U_all = [], []

    for _ in range(int(n_episodes)):
        q0 = robot.q_home + np.deg2rad(rng.uniform(-20, 20, size=robot.n_joints))
        q0 = clamp(q0, robot.q_min, robot.q_max)
        plant.reset(q0)
        q_cmd = robot.q_home.copy()

        for k in range(int(episode_steps)):
            if k % 20 == 0:
                q_cmd = robot.q_home + np.deg2rad(rng.uniform(-45, 45, size=robot.n_joints))
                q_cmd = clamp(q_cmd, robot.q_min, robot.q_max)

            X_all.append(plant.get_state().copy())
            U_all.append(q_cmd.copy())
            plant.step(q_cmd)

    X = np.array(X_all)
    U = np.array(U_all)
    X_dot = finite_difference(X, dt)
    return X, U, X_dot
