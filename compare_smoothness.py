"""Compare swing smoothness between controllers.

Metrics:
- Joint acceleration (RMS of ddq)
- Joint jerk (RMS of dddq)
- Control effort (RMS of action)
- Torque rate (RMS of d(action)/dt)

Usage:
    python compare_smoothness.py [--episodes 5]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import click

from run import (
    ENV_ID, ROBOT_CONFIGS, extract_env_state, load_subsystems,
    monkey_patch_fancy_gym,
)


def run_episodes_with_logging(
    controller_name: str,
    swing_name: str,
    episodes: int = 5,
    robot: str = "fanuc",
    save_dir: str | None = None,
) -> dict:
    """Run episodes and log q, dq, actions per timestep."""
    import gymnasium as gym
    import fancy_gym

    rcfg = ROBOT_CONFIGS[robot]
    ndof = rcfg["ndof"]
    dt = 0.008

    # Monkey-patch for custom XML
    xml_path = os.path.join(os.path.dirname(__file__), "assets", "xml", rcfg["xml"])
    if os.path.exists(xml_path):
        monkey_patch_fancy_gym(xml_path, rcfg["init_qpos"], ndof)

    env = gym.make(ENV_ID, render_mode=None)

    # Load subsystems
    controller, _ = load_subsystems(
        perceiver="sim",
        predictor="drag_bounce",
        spin="zero",
        aimer="specular",
        swing=swing_name,
        control=controller_name,
        env=env,
        robot_x=rcfg.get("robot_x", 1.0),
        follow_through=0.0,
        lookahead_fraction=0.6,
    )

    gear = rcfg["gear"]
    kp = rcfg.get("kp_scale", 50.0) / gear
    kd = rcfg.get("kd_scale", 5.0) / gear
    use_gravity_comp = rcfg.get("gravity_comp", False)

    import inspect
    _ctrl_accepts_dq = 'dq_current' in inspect.signature(controller.step).parameters

    all_episodes = []

    for ep in range(episodes):
        obs, info = env.reset()
        controller.reset()
        done = False
        step_count = 0

        ep_data = {"q": [], "dq": [], "action": [], "ee_pos": []}

        while not done:
            env_state, q_current, qd_current = extract_env_state(obs, ndof, step_count, dt)

            if getattr(controller, 'torque_mode', False):
                action_raw = controller.step(env_state, q_current, qd_current)
            else:
                if _ctrl_accepts_dq:
                    q_desired = controller.step(env_state, q_current, dq_current=qd_current)
                else:
                    q_desired = controller.step(env_state, q_current)
                action_raw = kp * (q_desired - q_current) - kd * qd_current
                if use_gravity_comp:
                    qfrc_bias = env.unwrapped.data.qfrc_bias[:ndof].copy()
                    action_raw = action_raw + qfrc_bias / gear

            act_dim = env.action_space.shape[0]
            if len(action_raw) < act_dim:
                action = np.zeros(act_dim)
                action[:len(action_raw)] = action_raw
            else:
                action = action_raw
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Log
            ep_data["q"].append(q_current.copy())
            ep_data["dq"].append(qd_current.copy())
            ep_data["action"].append(action[:ndof].copy())
            ep_data["ee_pos"].append(env.unwrapped.data.body("EE").xpos.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        # Convert to arrays
        for k in ep_data:
            ep_data[k] = np.array(ep_data[k])
        ep_data["hit"] = info.get("hit_ball", False)
        all_episodes.append(ep_data)

        # Save to disk for offline analysis
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(
                os.path.join(save_dir, f"ep_{ep:03d}.npz"),
                q=ep_data["q"],
                dq=ep_data["dq"],
                action=ep_data["action"],
                ee_pos=ep_data["ee_pos"],
            )

    env.close()
    return all_episodes


def compute_metrics(episodes_data: list, dt: float = 0.008) -> dict:
    """Compute smoothness metrics across episodes."""
    all_ddq_rms = []
    all_jerk_rms = []
    all_ctrl_rms = []
    all_ctrl_rate_rms = []
    all_ee_jerk_rms = []

    for ep in episodes_data:
        q = ep["q"]      # (T, ndof)
        dq = ep["dq"]    # (T, ndof)
        action = ep["action"]  # (T, ndof)
        ee = ep["ee_pos"]  # (T, 3)

        # Acceleration: finite diff of dq
        ddq = np.diff(dq, axis=0) / dt
        # Jerk: finite diff of ddq
        jerk = np.diff(ddq, axis=0) / dt

        # Control rate
        ctrl_rate = np.diff(action, axis=0) / dt

        # EE jerk
        ee_vel = np.diff(ee, axis=0) / dt
        ee_acc = np.diff(ee_vel, axis=0) / dt
        ee_jerk = np.diff(ee_acc, axis=0) / dt

        all_ddq_rms.append(np.sqrt(np.mean(ddq**2)))
        all_jerk_rms.append(np.sqrt(np.mean(jerk**2)))
        all_ctrl_rms.append(np.sqrt(np.mean(action**2)))
        all_ctrl_rate_rms.append(np.sqrt(np.mean(ctrl_rate**2)))
        all_ee_jerk_rms.append(np.sqrt(np.mean(ee_jerk**2)))

    return {
        "ddq_rms": np.mean(all_ddq_rms),
        "jerk_rms": np.mean(all_jerk_rms),
        "ctrl_rms": np.mean(all_ctrl_rms),
        "ctrl_rate_rms": np.mean(all_ctrl_rate_rms),
        "ee_jerk_rms": np.mean(all_ee_jerk_rms),
        "ddq_std": np.std(all_ddq_rms),
        "jerk_std": np.std(all_jerk_rms),
        "ctrl_std": np.std(all_ctrl_rms),
        "ctrl_rate_std": np.std(all_ctrl_rate_rms),
        "ee_jerk_std": np.std(all_ee_jerk_rms),
    }


@click.command()
@click.option("--episodes", default=5, help="Episodes per controller")
def main(episodes):
    configs = [
        ("Quintic+OpenLoop", "open_loop", "quintic"),
        ("Torque MPC", "torque_mpc", "quintic"),
        ("Position MPC", "mpc", "quintic"),
    ]

    results = {}
    for label, ctrl, swing in configs:
        click.echo(f"\n{'='*60}")
        click.echo(f"Running {label} ({episodes} episodes)...")
        click.echo(f"{'='*60}")
        ep_data = run_episodes_with_logging(ctrl, swing, episodes=episodes, save_dir=f"results/trajectories/{label.replace(' ', '_').lower()}")
        hits = sum(1 for e in ep_data if e["hit"])
        click.echo(f"  Contact: {hits}/{episodes}")
        metrics = compute_metrics(ep_data)
        results[label] = metrics

    # Print comparison table
    click.echo(f"\n{'='*60}")
    click.echo("SMOOTHNESS COMPARISON (Fanuc, N={} episodes)".format(episodes))
    click.echo(f"{'='*60}")
    click.echo(f"{'Metric':<20} | {'Quintic+OL':>14} | {'Torque MPC':>14} | {'Position MPC':>14}")
    click.echo("-" * 70)

    metric_labels = [
        ("ddq_rms", "Accel RMS (rad/s²)"),
        ("jerk_rms", "Jerk RMS (rad/s³)"),
        ("ctrl_rms", "Ctrl Effort RMS"),
        ("ctrl_rate_rms", "Ctrl Rate RMS"),
        ("ee_jerk_rms", "EE Jerk RMS (m/s³)"),
    ]

    for key, label in metric_labels:
        vals = [results[c][key] for c in ["Quintic+OpenLoop", "Torque MPC", "Position MPC"]]
        click.echo(f"{label:<20} | {vals[0]:>14.2f} | {vals[1]:>14.2f} | {vals[2]:>14.2f}")

    click.echo(f"\nLower = smoother (except Ctrl Effort which is just magnitude)")


if __name__ == "__main__":
    main()
