"""Robustness evaluation: compare controllers under injected disturbances.

Tests how gracefully controllers degrade when:
1. Target prediction noise (simulating sensor error)
2. Torque noise (simulating actuator uncertainty)  
3. Observation delay (simulating perception latency)

Usage:
    python eval_robustness.py [--episodes 40]
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


def run_with_disturbance(
    controller_name: str,
    episodes: int = 40,
    torque_noise_std: float = 0.0,
    obs_delay_frames: int = 0,
    robot: str = "fanuc",
) -> dict:
    """Run episodes with injected disturbances."""
    import gymnasium as gym
    import fancy_gym

    rcfg = ROBOT_CONFIGS[robot]
    ndof = rcfg["ndof"]
    dt = 0.008

    xml_path = os.path.join(os.path.dirname(__file__), "assets", "xml", rcfg["xml"])
    if os.path.exists(xml_path):
        monkey_patch_fancy_gym(xml_path, rcfg["init_qpos"], ndof)

    env = gym.make(ENV_ID, render_mode=None)

    controller, _ = load_subsystems(
        perceiver="sim",
        predictor="drag_bounce",
        spin="zero",
        aimer="specular",
        swing="quintic",
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
    is_torque_mode = getattr(controller, 'torque_mode', False)

    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    
    contacts = 0
    landings = 0
    total_reward = 0.0

    for ep in range(episodes):
        obs, info = env.reset()
        controller.reset()
        done = False
        step_count = 0
        obs_buffer = []  # for delay simulation

        while not done:
            # Simulate observation delay
            obs_buffer.append(obs.copy())
            if obs_delay_frames > 0 and len(obs_buffer) > obs_delay_frames:
                delayed_obs = obs_buffer[-1 - obs_delay_frames]
            else:
                delayed_obs = obs

            env_state, q_current, qd_current = extract_env_state(delayed_obs, ndof, step_count, dt)

            if is_torque_mode:
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

            # Inject torque noise
            if torque_noise_std > 0:
                noise = rng.normal(0, torque_noise_std, size=ndof)
                action_raw = action_raw + noise

            act_dim = env.action_space.shape[0]
            if len(action_raw) < act_dim:
                action = np.zeros(act_dim)
                action[:len(action_raw)] = action_raw
            else:
                action = action_raw
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        if info.get("hit_ball", False):
            contacts += 1
        if info.get("ball_returned_success", False):
            landings += 1
        total_reward += float(reward)

    env.close()
    return {
        "contact_rate": contacts / episodes,
        "landing_rate": landings / episodes,
        "mean_reward": total_reward / episodes,
    }


@click.command()
@click.option("--episodes", default=40, help="Episodes per condition")
def main(episodes):
    controllers = [
        ("Quintic+OL", "open_loop"),
        ("Torque MPC", "torque_mpc"),
        ("Tube MPC", "tube_mpc"),
    ]

    # Disturbance levels
    torque_noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    delay_levels = [0, 1, 2, 3]

    click.echo("=" * 70)
    click.echo(f"ROBUSTNESS EVALUATION (Fanuc, N={episodes} per condition)")
    click.echo("=" * 70)

    # --- Torque noise sweep ---
    click.echo(f"\n{'='*70}")
    click.echo("TORQUE NOISE SWEEP (std of additive ctrl-space noise)")
    click.echo(f"{'='*70}")
    click.echo(f"{'Noise σ':<10} | {'Quintic+OL':>12} | {'Torque MPC':>12} | {'Tube MPC':>12}")
    click.echo("-" * 55)

    for noise in torque_noise_levels:
        results = {}
        for label, ctrl in controllers:
            r = run_with_disturbance(ctrl, episodes=episodes, torque_noise_std=noise)
            results[label] = r
        click.echo(
            f"{noise:<10.2f} | "
            f"{results['Quintic+OL']['contact_rate']*100:>10.1f}% | "
            f"{results['Torque MPC']['contact_rate']*100:>10.1f}% | "
            f"{results['Tube MPC']['contact_rate']*100:>10.1f}%"
        )

    # --- Observation delay sweep ---
    click.echo(f"\n{'='*70}")
    click.echo("OBSERVATION DELAY SWEEP (frames of delayed ball position)")
    click.echo(f"{'='*70}")
    click.echo(f"{'Delay':<10} | {'Quintic+OL':>12} | {'Torque MPC':>12} | {'Tube MPC':>12}")
    click.echo("-" * 55)

    for delay in delay_levels:
        results = {}
        for label, ctrl in controllers:
            r = run_with_disturbance(ctrl, episodes=episodes, obs_delay_frames=delay)
            results[label] = r
        click.echo(
            f"{delay} frames  | "
            f"{results['Quintic+OL']['contact_rate']*100:>10.1f}% | "
            f"{results['Torque MPC']['contact_rate']*100:>10.1f}% | "
            f"{results['Tube MPC']['contact_rate']*100:>10.1f}%"
        )


if __name__ == "__main__":
    main()
