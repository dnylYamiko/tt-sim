"""CLI entry point for tt-sim."""

import click
import numpy as np
import os

from tt_sim import registry

# fancy_gym TableTennis4D observation layout (19 dims for WAM, 15 for Fanuc):
#   obs[0:ndof]       = robot joint positions
#   obs[ndof:2*ndof]  = robot joint velocities
#   obs[2*ndof:2*ndof+3] = ball position (x, y, z)
#   obs[2*ndof+3:2*ndof+5] = goal position (2D target on table)
# dt = 0.008s per step, ~195 steps per episode

ENV_ID = "fancy/TableTennis4D-v0"

# Robot configurations
ROBOT_CONFIGS = {
    "wam": {
        "ndof": 7,
        "xml": "table_tennis_env.xml",
        "gear": np.array([600.0, 500.0, 160.0, 240.0, 20.0, 20.0, 8.0]),
        "init_qpos": np.array([1.907, 0.839, 1.214, 2.739, -0.534, -0.783, -1.392]),
        "joint_prefix": "wam/",
        "robot_x": 1.3,   # intercept plane: base(2.1) - 0.8
    },
    "fanuc": {
        "ndof": 6,
        "xml": "table_tennis_fanuc.xml",
        "gear": np.array([500.0, 500.0, 300.0, 150.0, 80.0, 50.0]),
        "init_qpos": np.array([-1.57, 0.8, -1.5, 0.0, -1.57, 0.0]),
        "joint_prefix": "fanuc/",
        "gravity_comp": True,
        "kp_scale": 800.0,     # aggressive tracking for heavy arm
        "kd_scale": 50.0,
        "reactive_kwargs": {"gain": 5.0, "damping": 0.01},
        "robot_x": 1.0,   # intercept plane: base(1.6) - 0.6 (further from base = more time)
    },
}


def extract_env_state(obs: np.ndarray, ndof: int, step_count: int, dt: float = 0.008) -> tuple[dict, np.ndarray, np.ndarray]:
    """Extract env_state dict, q_current, and qd_current from observation.

    fancy_gym always outputs 19D obs (hardcoded for 7-DOF WAM):
      obs[0:7]    -> qpos[0:7] (for 6-DOF, index 6 = tar_x ball joint)
      obs[7:14]   -> qvel[0:7]
      obs[14:17]  -> ball position (x, y, z)
      obs[17:19]  -> goal position
    We extract only the first ndof entries for robot joints.
    """
    q_current = obs[0:ndof].copy()
    qd_current = obs[7:7 + ndof].copy()  # always at offset 7 (fancy_gym hardcoded)
    ball_pos = obs[14:17].copy()

    env_state = {
        "ball_pos": np.asarray(ball_pos, dtype=np.float64),
        "time": float(step_count * dt),
    }

    return env_state, np.asarray(q_current, dtype=np.float64), np.asarray(qd_current, dtype=np.float64)


def load_subsystems(perceiver, predictor, spin, aimer, swing, control, env=None, reactive_kwargs=None, robot_x=None, follow_through=0.0, lookahead_fraction=0.6):
    """Load and instantiate all subsystems via the registry."""
    PerceiverCls = registry.load("perceiver", perceiver)
    PredictorCls = registry.load("predictor", predictor)
    SpinCls = registry.load("spin", spin)
    AimerCls = registry.load("aimer", aimer)
    SwingCls = registry.load("swing", swing)
    ControlCls = registry.load("control", control)

    perc = PerceiverCls()
    # Pass sim-matched physics to DragBouncePredictor:
    # MuJoCo has NO air drag and very elastic bounce (~0.97 COR)
    from tt_sim.upgrades.prediction import DragBouncePredictor
    if issubclass(PredictorCls, DragBouncePredictor):
        pred = PredictorCls(c_d=0.0, restitution=0.91)
    else:
        pred = PredictorCls()
    spin_est = SpinCls()
    aim = AimerCls()

    # Some swing planners need the env (e.g., MujocoIKSwingPlanner)
    from tt_sim.baselines.swing import MujocoIKSwingPlanner, LerpSwingPlanner
    from tt_sim.upgrades.swing import QuinticSwingPlanner
    if issubclass(SwingCls, MujocoIKSwingPlanner):
        if env is None:
            raise click.ClickException(f"--swing={swing} requires the sim env (not available in --dry-run)")
        swng = SwingCls(env=env)
    elif issubclass(SwingCls, (QuinticSwingPlanner, LerpSwingPlanner)) and env is not None:
        # Wire up MuJoCo IK (and Jacobian for quintic) for swing planners
        import mujoco
        model = env.unwrapped.model
        n_joints = model.nu

        def _mujoco_ik(target, q_seed=None, _model=model, _env=env, _nj=n_joints):
            """Damped least-squares IK with position + orientation."""
            d = mujoco.MjData(_model)
            if q_seed is not None:
                d.qpos[:_nj] = q_seed[:_nj].copy()
            else:
                d.qpos[:_nj] = _env.unwrapped.data.qpos[:_nj].copy()
            mujoco.mj_forward(_model, d)
            ee_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "EE")

            # Orientation weight relative to position
            w_ori = 0.5

            for _ in range(200):
                ee_pos = d.body("EE").xpos.copy()
                pos_err = target.position - ee_pos

                # Orientation error: align EE body Z-axis with target.normal
                ee_rot = d.body("EE").xmat.reshape(3, 3)
                paddle_z = ee_rot[:, 2]  # paddle face normal = body Z
                desired_n = target.normal / (np.linalg.norm(target.normal) + 1e-10)
                # Orientation error as cross product (small-angle approximation)
                ori_err = np.cross(paddle_z, desired_n) * w_ori

                # Stack into 6D error
                err = np.concatenate([pos_err, ori_err])
                if np.linalg.norm(pos_err) < 0.005 and np.linalg.norm(ori_err) < 0.02:
                    break

                # Full 6D Jacobian (position + rotation)
                jacp = np.zeros((3, _model.nv))
                jacr = np.zeros((3, _model.nv))
                mujoco.mj_jacBody(_model, d, jacp, jacr, ee_id)
                J = np.vstack([jacp[:, :_nj], jacr[:, :_nj]])

                # Damped least squares
                damping = 0.05
                JJT = J @ J.T + damping**2 * np.eye(6)
                dq = J.T @ np.linalg.solve(JJT, err)
                # Limit step size to prevent large jumps (important for 6-DOF)
                max_step = 0.2
                dq_norm = np.linalg.norm(dq)
                if dq_norm > max_step:
                    dq = dq * (max_step / dq_norm)
                d.qpos[:_nj] += dq
                for i in range(_nj):
                    lo, hi = _model.jnt_range[i]
                    if lo != hi:
                        d.qpos[i] = np.clip(d.qpos[i], lo, hi)
                mujoco.mj_forward(_model, d)
            return d.qpos[:_nj].copy()

        def _mujoco_jac(q, _model=model, _nj=n_joints):
            """Position Jacobian (3 x n_joints) at joint config q."""
            d = mujoco.MjData(_model)
            d.qpos[:_nj] = q[:_nj].copy()
            mujoco.mj_forward(_model, d)
            ee_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "EE")
            jacp = np.zeros((3, _model.nv))
            mujoco.mj_jacBody(_model, d, jacp, None, ee_id)
            return jacp[:, :_nj]  # 3 x n_joints for Cartesian velocity mapping

        if issubclass(SwingCls, QuinticSwingPlanner):
            swng = SwingCls(n_joints=n_joints, ik_fn=_mujoco_ik, jac_fn=_mujoco_jac)
        else:
            # LerpSwingPlanner — just needs IK
            swng = SwingCls(n_joints=n_joints, ik_fn=_mujoco_ik)
    else:
        swng = SwingCls()

    # Build controller kwargs
    from tt_sim.baselines.control import ReactiveController
    ctrl_kwargs = dict(
        perceiver=perc,
        predictor=pred,
        spin_estimator=spin_est,
        aimer=aim,
        swing_planner=swng,
    )
    if issubclass(ControlCls, ReactiveController):
        ctrl_kwargs["env"] = env
        if reactive_kwargs:
            ctrl_kwargs.update(reactive_kwargs)
    else:
        if robot_x is not None:
            ctrl_kwargs["robot_x"] = robot_x

    # MPC controller needs symbolic FK from MuJoCo model
    from tt_sim.upgrades.control import MPCController
    if issubclass(ControlCls, MPCController) and env is not None:
        from tt_sim.upgrades.mpc_fk import build_fk_casadi
        model = env.unwrapped.model
        n_joints = model.nu
        fk_dict = build_fk_casadi(model, n_joints)
        ctrl_kwargs["fk_dict"] = fk_dict
        ctrl_kwargs["follow_through"] = follow_through
        ctrl_kwargs["lookahead_fraction"] = lookahead_fraction

    # Torque MPC needs full dynamics (RNEA) from MuJoCo model
    from tt_sim.upgrades.control import TorqueMPCController
    if issubclass(ControlCls, TorqueMPCController) and env is not None:
        from tt_sim.upgrades.mpc_dynamics import build_dynamics_casadi
        model = env.unwrapped.model
        if 'n_joints' not in dir():
            n_joints = model.nu
        dynamics_dict = build_dynamics_casadi(model, n_joints)
        ctrl_kwargs["dynamics_dict"] = dynamics_dict
        ctrl_kwargs["follow_through"] = follow_through

    ctrl = ControlCls(**ctrl_kwargs)
    return ctrl, {
        "perceiver": perceiver,
        "predictor": predictor,
        "spin": spin,
        "aimer": aimer,
        "swing": swing,
        "control": control,
    }


def monkey_patch_fancy_gym(xml_path: str, init_qpos: np.ndarray, ndof: int):
    """Monkey-patch fancy_gym TableTennis to use custom XML and init config."""
    from fancy_gym.envs.mujoco.table_tennis import table_tennis_env as _tt_mod
    import mujoco as _mj

    _orig_init = _tt_mod.TableTennisEnv.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.model = _mj.MjModel.from_xml_path(xml_path)
        self.data = _mj.MjData(self.model)
        if hasattr(self, 'mujoco_renderer'):
            self.mujoco_renderer.model = self.model
            self.mujoco_renderer.data = self.data
            self.mujoco_renderer._camera_name = "wide_table"
    _tt_mod.TableTennisEnv.__init__ = _patched_init

    # Patch step to truncate actions to actual actuator count
    _orig_step = _tt_mod.TableTennisEnv.step
    _nu = ndof  # number of actuators in our model
    def _patched_step(self, action):
        return _orig_step(self, action[:_nu])
    _tt_mod.TableTennisEnv.step = _patched_step

    _orig_reset_model = _tt_mod.TableTennisEnv.reset_model
    def _patched_reset_model(self):
        result = _orig_reset_model(self)
        # _orig_reset_model does qpos[:7] = WAM init, which corrupts ball
        # joints for non-7-DOF robots. Restore robot and ball state properly.
        self.data.qpos[:ndof] = init_qpos
        self.data.qvel[:ndof] = np.zeros(ndof)
        self.data.ctrl[:] = 0  # zero torque; gravity comp applied in control loop
        # Re-apply ball state (corrupted by qpos[:7] assignment in orig reset)
        ball_state = self._init_ball_state
        self.data.joint("tar_x").qpos = ball_state[0]
        self.data.joint("tar_y").qpos = ball_state[1]
        self.data.joint("tar_z").qpos = ball_state[2]
        self.data.joint("tar_x").qvel = ball_state[3]
        self.data.joint("tar_y").qvel = ball_state[4]
        self.data.joint("tar_z").qvel = ball_state[5]
        _mj.mj_forward(self.model, self.data)
        return self._get_obs()
    _tt_mod.TableTennisEnv.reset_model = _patched_reset_model


@click.command()
@click.option("--perceiver", default="sim", help="Perception module.")
@click.option("--predictor", default="ballistic", help="Prediction module.")
@click.option("--spin", default="zero", help="Spin estimation module.")
@click.option("--aimer", default="face_net", help="Aiming module.")
@click.option("--swing", default="lerp", help="Swing generation module.")
@click.option("--control", default="open_loop", help="Control module.")
@click.option("--robot", default="wam", type=click.Choice(["wam", "fanuc"]), help="Robot platform.")
@click.option("--episodes", default=10, type=int, help="Number of episodes.")
@click.option("--log", default=None, type=str, help="Log CSV file path.")
@click.option("--list", "list_impls", is_flag=True, help="List available implementations and exit.")
@click.option("--dry-run", is_flag=True, help="Load subsystems and print types without running sim.")
@click.option("--render", is_flag=True, help="Open live MuJoCo viewer window.")
@click.option("--record", default=None, type=str, help="Path to save mp4 video (e.g., results/demo.mp4).")
@click.option("--original-scene", is_flag=True, help="Use original fancy_gym scene (ceiling-mounted WAM, no custom XML).")
@click.option("--follow-through", "follow_through", default=0.0, type=float, help="Follow-through distance (meters) for MPC target offset along swing direction.")
@click.option("--lookahead", "lookahead_fraction", default=0.6, type=float, help="MPC lookahead as fraction of remaining time (0.0-1.0).")
def main(perceiver, predictor, spin, aimer, swing, control, robot, episodes, log, list_impls, dry_run, render, record, original_scene, follow_through, lookahead_fraction):
    """Run the tt-sim table tennis simulation."""
    if list_impls:
        registry.list_available()
        return

    rcfg = ROBOT_CONFIGS[robot]
    ndof = rcfg["ndof"]
    click.echo(f"Config: {robot=}, {perceiver=}, {predictor=}, {spin=}, {aimer=}, {swing=}, {control=}, {episodes=}")

    if dry_run:
        controller, config = load_subsystems(perceiver, predictor, spin, aimer, swing, control, robot_x=rcfg.get("robot_x"), follow_through=follow_through, lookahead_fraction=lookahead_fraction)
        click.echo("\n-- Dry run: subsystems loaded --")
        click.echo(f"  robot:          {robot} ({ndof}-DOF)")
        click.echo(f"  perceiver:      {type(controller.perceiver).__qualname__}")
        click.echo(f"  predictor:      {type(controller.predictor).__qualname__}")
        click.echo(f"  spin_estimator: {type(controller.spin_estimator).__qualname__}")
        click.echo(f"  aimer:          {type(controller.aimer).__qualname__}")
        click.echo(f"  swing_planner:  {type(controller.swing_planner).__qualname__}")
        click.echo(f"  controller:     {type(controller).__qualname__}")
        return

    # --- Simulation loop ---
    import gymnasium as gym
    import fancy_gym  # noqa: F401 – registers fancy envs
    from tt_sim.eval import EvalLogger, EpisodeMetrics

    # Render mode
    render_mode = None
    use_cv2_viewer = False
    if render or record:
        render_mode = "rgb_array"
        if render:
            try:
                import cv2
                use_cv2_viewer = True
                click.echo("Live viewer: press 'q' in viewer window to quit early")
            except ImportError:
                click.echo("Warning: opencv-python not installed, --render disabled.")
                if not record:
                    render_mode = None

    # Monkey-patch fancy_gym to use our custom XML (unless --original-scene)
    if not original_scene:
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "xml", rcfg["xml"])
        if os.path.exists(xml_path):
            monkey_patch_fancy_gym(xml_path, rcfg["init_qpos"], ndof)
            click.echo(f"Scene: {rcfg['xml']} (floor-mounted {robot})")
        else:
            click.echo(f"Warning: {xml_path} not found, falling back to default scene")
    else:
        click.echo("Scene: original fancy_gym (ceiling-mounted WAM)")

    env = gym.make(ENV_ID, render_mode=render_mode)
    reactive_kwargs = rcfg.get("reactive_kwargs", None)
    controller, config = load_subsystems(perceiver, predictor, spin, aimer, swing, control, env=env, reactive_kwargs=reactive_kwargs, robot_x=rcfg.get("robot_x"), follow_through=follow_through, lookahead_fraction=lookahead_fraction)
    dt = env.unwrapped.dt
    logger = EvalLogger(log_path=log, config=config)

    # Video writer
    video_writer = None
    if record:
        import imageio
        os.makedirs(os.path.dirname(record) or ".", exist_ok=True)
        video_writer = imageio.get_writer(record, fps=int(1.0 / dt), quality=8)
        click.echo(f"Recording to {record}")

    # PD gains and gravity compensation flag
    use_gravity_comp = rcfg.get("gravity_comp", False)
    gear = rcfg["gear"]
    kp = rcfg.get("kp_scale", 50.0) / gear
    kd = rcfg.get("kd_scale", 5.0) / gear

    # Check if controller.step accepts dq_current (MPC needs velocity continuity)
    import inspect
    _ctrl_accepts_dq = 'dq_current' in inspect.signature(controller.step).parameters

    # Note: with gear ratios on all robots, ctrl range is [-1, 1] and
    # gravity comp (qfrc_bias/gear) stays within that range.

    for ep in range(episodes):
        obs, info = env.reset()
        controller.reset()
        done = False
        reward = 0.0
        step_count = 0

        while not done:
            env_state, q_current, qd_current = extract_env_state(obs, ndof, step_count, dt)

            # Torque-mode controllers output ctrl-space torques directly
            if getattr(controller, 'torque_mode', False):
                action_raw = controller.step(env_state, q_current, qd_current)
            else:
                if _ctrl_accepts_dq:
                    q_desired = controller.step(env_state, q_current, dq_current=qd_current)
                else:
                    q_desired = controller.step(env_state, q_current)

                # Convert q_desired to torque action (PD control + optional gravity comp)
                action_raw = kp * (q_desired - q_current) - kd * qd_current
                if use_gravity_comp:
                    # Feedforward gravity+Coriolis compensation from MuJoCo
                    # qfrc_bias is in torque space; divide by gear to get control space
                    qfrc_bias = env.unwrapped.data.qfrc_bias[:ndof].copy()
                    action_raw = action_raw + qfrc_bias / gear

            # Pad to env action space and clip
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

            # Track minimum paddle-ball distance for diagnostics
            ee_pos = env.unwrapped.data.body("EE").xpos.copy()
            ball_pos_now = obs[14:17]
            pb_dist = np.linalg.norm(ee_pos - ball_pos_now)
            if step_count == 1:
                min_pb_dist = pb_dist
            else:
                min_pb_dist = min(min_pb_dist, pb_dist)

            # Render / record
            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    if video_writer is not None:
                        video_writer.append_data(frame)
                    if use_cv2_viewer:
                        import cv2
                        cv2.imshow("tt-sim", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            done = True

        hit_ball = info.get("hit_ball", False)
        ball_returned = info.get("ball_returned_success", False)
        land_dist = info.get("land_dist_error", None)

        metrics = EpisodeMetrics(
            episode=ep,
            contact=hit_ball,
            landed=ball_returned,
            landing_x=None,
            landing_y=None,
            prediction_error=None,
            reward=float(reward),
            episode_time=step_count * dt,
            **config,
        )
        logger.log(metrics)

        status = "HIT+LAND" if ball_returned else ("HIT" if hit_ball else "miss")
        click.echo(
            f"Episode {ep + 1}/{episodes}: reward={reward:.3f}  "
            f"status={status}  land_dist_err={land_dist:.3f}  "
            f"min_pb_dist={min_pb_dist:.3f}  steps={step_count}"
        )

    logger.print_summary()

    if video_writer is not None:
        video_writer.close()
        click.echo(f"Video saved to {record}")

    if use_cv2_viewer:
        import cv2
        cv2.destroyAllWindows()

    env.close()


if __name__ == "__main__":
    main()
