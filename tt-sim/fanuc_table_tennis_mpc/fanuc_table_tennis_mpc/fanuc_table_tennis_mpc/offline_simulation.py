"""Offline FANUC table-tennis simulation comparing baseline MPC and SINDy-MPC."""

import numpy as np
import matplotlib.pyplot as plt

from .crx25ia_kinematics import FANUCCRX25iA
from .ball_predictor import BallTrajectoryPredictor
from .intercept_planner import InterceptPlanner
from .simulated_servo import TrueServoPlant
from .sindy_model import SINDYcModel
from .position_mpc import BaselinePositionMPC, SINDyPositionMPC
from .data_generation import generate_sindy_training_data

from .evaluation import evaluate_run


def train_or_load_sindy(robot, model_path="sindy_fanuc_crx25ia_model.npz", dt=0.02, train=True):
    sindy = SINDYcModel()
    if not train:
        sindy.load(model_path)
        return sindy

    X, U, X_dot = generate_sindy_training_data(robot, dt=dt)
    sindy.fit(X, U, X_dot, ridge=1e-6, threshold=1e-4, n_refine=5)
    sindy.save(model_path)
    return sindy


def run_interception(controller_type="sindy", sindy_model=None, dt=0.02, sim_time=3.0):
    robot = FANUCCRX25iA()
    plant = TrueServoPlant(robot, dt=dt)
    plant.reset(robot.q_home.copy())

    ball_predictor = BallTrajectoryPredictor(dt=dt)
    planner = InterceptPlanner()

    if controller_type == "baseline":
        controller = BaselinePositionMPC(robot, dt=dt, horizon=10)
    elif controller_type == "sindy":
        if sindy_model is None:
            sindy_model = SINDYcModel()
            sindy_model.use_fallback_model()
        controller = SINDyPositionMPC(robot, sindy_model, dt=dt, horizon=10)
    else:
        raise ValueError("controller_type must be 'baseline' or 'sindy'")

    ball_state = np.array([2.20, -0.35, 1.15, -1.45, 0.25, 0.35], dtype=float)
    steps = int(sim_time / dt)

    logs = {"q": [], "dq": [], "q_cmd": [], "ee": [], "target": [], "ball": [], "error": [], "cost": []}

    for k in range(steps):
        ball_predictions = ball_predictor.predict(ball_state, horizon_steps=80)
        target, _ = planner.choose_intercept(ball_predictions)
        x_current = plant.get_state()

        if controller_type == "baseline":
            q_cmd, cost = controller.solve(x_current[:6], target)
        else:
            q_cmd, cost = controller.solve(x_current, target)

        plant.step(q_cmd)
        ball_state = ball_predictor.step(ball_state)
        x_new = plant.get_state()
        ee = robot.end_effector_position(x_new[:6])
        error = np.linalg.norm(ee - target)

        logs["q"].append(x_new[:6].copy())
        logs["dq"].append(x_new[6:].copy())
        logs["q_cmd"].append(q_cmd.copy())
        logs["ee"].append(ee.copy())
        logs["target"].append(target.copy())
        logs["ball"].append(ball_state[:3].copy())
        logs["error"].append(error)
        logs["cost"].append(cost)

        if k % 20 == 0:
            print(f"{controller_type.upper()} t={k*dt:.2f}s error={error:.3f} m target={target.round(3)} ee={ee.round(3)}")

    for key in logs:
        logs[key] = np.array(logs[key])
    logs["dt"] = dt
    logs["controller"] = controller_type
    return logs, robot


def plot_results(baseline_logs, sindy_logs, output_prefix=None):
    dt = baseline_logs["dt"]
    time = np.arange(len(baseline_logs["error"])) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(time, baseline_logs["error"], label="Baseline MPC")
    plt.plot(time, sindy_logs["error"], label="SINDy-MPC")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector tracking error [m]")
    plt.title("FANUC Table-Tennis Tracking Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_tracking_error.png", dpi=200)
    plt.show()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(baseline_logs["ee"][:, 0], baseline_logs["ee"][:, 1], baseline_logs["ee"][:, 2], label="Baseline EE")
    ax.plot(sindy_logs["ee"][:, 0], sindy_logs["ee"][:, 1], sindy_logs["ee"][:, 2], label="SINDy EE")
    ax.plot(sindy_logs["target"][:, 0], sindy_logs["target"][:, 1], sindy_logs["target"][:, 2], "--", label="Target")
    ax.plot(sindy_logs["ball"][:, 0], sindy_logs["ball"][:, 1], sindy_logs["ball"][:, 2], label="Ball")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D FANUC Table-Tennis Simulation")
    ax.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_3d.png", dpi=200)
    plt.show()

    print("\nController comparison")
    print(f"Baseline mean error: {np.mean(baseline_logs['error']):.4f} m")
    print(f"SINDy-MPC mean error: {np.mean(sindy_logs['error']):.4f} m")
    print(f"Baseline final error: {baseline_logs['error'][-1]:.4f} m")
    print(f"SINDy-MPC final error: {sindy_logs['error'][-1]:.4f} m")


"""def main():
    dt = 0.02
    robot = FANUCCRX25iA()
    print("Training SINDy model...")
    sindy = train_or_load_sindy(robot, dt=dt, train=True)
    print("Running baseline MPC...")
    baseline_logs, _ = run_interception("baseline", None, dt=dt, sim_time=3.0)
    print("Running SINDy-MPC...")
    sindy_logs, _ = run_interception("sindy", sindy, dt=dt, sim_time=3.0)
    plot_results(baseline_logs, sindy_logs, output_prefix="fanuc_tt")
"""
def main():
    dt = 0.02

    robot = FANUCCRX25iA()

    print("Training SINDy model...")
    sindy = train_or_load_sindy(robot, dt=dt, train=True)

    print("Running baseline MPC...")
    baseline_logs, _ = run_interception(
        "baseline",
        None,
        dt=dt,
        sim_time=3.0
    )

    print("Running SINDy-MPC...")
    sindy_logs, _ = run_interception(
        "sindy",
        sindy,
        dt=dt,
        sim_time=3.0
    )

    plot_results(
        baseline_logs,
        sindy_logs,
        output_prefix="fanuc_tt"
    )

    print("Generating evaluation plots...")

    evaluate_run(
        logs=sindy_logs,
        output_dir="results/plots"
    )

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
