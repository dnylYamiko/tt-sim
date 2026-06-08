"""Baseline robot control."""

from tt_sim.interfaces import (
    HighLevelController, Perceiver, Predictor, SpinEstimator,
    Aimer, SwingPlanner, BallObservation
)
import numpy as np


class OpenLoopController(HighLevelController):
    """Stage 0 control: plan once based on intercept prediction, execute open-loop."""

    ROBOT_X = 1.3  # default, overridden by constructor

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
    ):
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self.t_predict = t_predict
        self.ROBOT_X = robot_x

        self._observations: list[BallObservation] = []
        self._trajectory = None
        self._plan_start_time = 0.0
        self._planned = False

    def _estimate_intercept_time(self) -> float:
        """Estimate when the ball will reach ROBOT_X.

        Uses the predictor's trajectory (drag-aware) to find the first time
        the ball's x-coordinate crosses ROBOT_X, falling back to linear
        extrapolation if the predictor is unavailable or the ball never reaches.
        """
        if len(self._observations) < 2:
            return self._observations[-1].timestamp + self.t_predict

        o_last = self._observations[-1]
        t_now = o_last.timestamp

        # Try drag-aware: sample the predictor at fine time steps
        try:
            dt_sample = 0.004  # 250 Hz sampling
            n_samples = int(self.t_predict / dt_sample)
            for i in range(1, n_samples + 1):
                t_probe = t_now + i * dt_sample
                state = self.predictor.predict(self._observations, t_probe)
                if state.position[0] >= self.ROBOT_X:
                    return t_probe
        except Exception:
            pass

        # Fallback: linear extrapolation
        o0 = self._observations[0]
        dt = o_last.timestamp - o0.timestamp
        if dt < 1e-6:
            return t_now + self.t_predict
        vx = (o_last.position[0] - o0.position[0]) / dt
        if vx < 0.1:
            return t_now + self.t_predict
        dx = self.ROBOT_X - o_last.position[0]
        return t_now + dx / vx

    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray:
        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        current_time = obs.timestamp

        if not self._planned and len(self._observations) >= 8:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            # Override t_contact with actual remaining time to intercept
            paddle_target.t_contact = max(t_intercept - current_time, 0.05)
            self._trajectory = self.swing_planner.plan(paddle_target, q_current)
            self._planned = True
            self._plan_start_time = current_time

        if self._trajectory is not None:
            # Time-based interpolation: map current sim time to trajectory
            t_elapsed = current_time - self._plan_start_time
            traj_times = self._trajectory.times
            traj_duration = traj_times[-1] - traj_times[0]

            if t_elapsed <= traj_duration:
                t_query = traj_times[0] + t_elapsed
                n_joints = self._trajectory.positions.shape[1]
                action = np.array([
                    np.interp(t_query, traj_times, self._trajectory.positions[:, j])
                    for j in range(n_joints)
                ])
                return action

        return q_current

    def reset(self):
        self._observations = []
        self._trajectory = None
        self._plan_start_time = 0.0
        self._planned = False


class ReactiveController(HighLevelController):
    """Reactive control: continuously moves EE toward current ball position.
    
    Bypasses prediction/aiming/swing — uses MuJoCo Jacobian to compute
    joint velocities that move the EE toward the ball. Simple but effective
    for getting initial contacts.
    """

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
        env=None,
        gain: float = 5.0,
        damping: float = 0.01,
        **kwargs,
    ):
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self._env = env
        self._gain = gain
        self._damping = damping
        self._observations: list[BallObservation] = []

    def set_env(self, env):
        """Set the gym env (needed for MuJoCo model access)."""
        self._env = env

    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray:
        import mujoco
        
        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        ball_pos = obs.position.copy()

        if self._env is None:
            return q_current

        model = self._env.unwrapped.model
        data = self._env.unwrapped.data
        
        # Keep target above table surface so arm doesn't collide with table
        ball_pos[2] = max(ball_pos[2], 0.82)

        ee_pos = data.body("EE").xpos.copy()
        err = ball_pos - ee_pos

        # Jacobian
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE")
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, ee_id)
        ndof = len(q_current)
        J = jacp[:, :ndof]

        # Damped least squares for desired joint velocity
        damping = self._damping
        JJT = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Scale: move proportionally to error magnitude
        gain = self._gain
        q_desired = q_current + gain * dq

        # Clamp to joint limits
        for i in range(ndof):
            lo, hi = model.jnt_range[i]
            if lo != hi:
                q_desired[i] = np.clip(q_desired[i], lo, hi)

        return q_desired

    def reset(self):
        self._observations = []
