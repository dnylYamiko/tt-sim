"""High-level controller upgrades for tt-sim."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tt_sim.interfaces import (
    Aimer,
    BallObservation,
    HighLevelController,
    Perceiver,
    Predictor,
    SpinEstimator,
    SwingPlanner,
)

if TYPE_CHECKING:
    pass


class ReplanController(HighLevelController):
    """Stage 1 – Re-planning closed-loop controller.

    Replans the full pipeline (predict → aim → swing) every REPLAN_INTERVAL
    frames (~31 Hz at dt=0.008s).  Between replans, holds the last commanded
    position.  Each replan generates a fresh quintic trajectory from the
    current joint state and returns the position at t = replan_dt into that
    trajectory (Option A: fixed lookahead clamped to trajectory duration).

    This provides closed-loop correction for prediction errors: as more ball
    observations accumulate, the predicted intercept becomes more accurate and
    the swing trajectory adapts.
    """

    REPLAN_INTERVAL = 4   # frames between replans (~31 Hz)
    MIN_OBS = 4           # fewer than open_loop since we keep correcting
    DT = 0.008            # sim timestep

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
    ) -> None:
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self.t_predict = t_predict
        self.ROBOT_X = robot_x

        self._observations: list[BallObservation] = []
        self._frame_count = 0
        self._last_action: np.ndarray | None = None
        self._prev_q: np.ndarray | None = None

    def _estimate_intercept_time(self) -> float:
        """Estimate when the ball will reach ROBOT_X (drag-aware)."""
        if len(self._observations) < 2:
            return self._observations[-1].timestamp + self.t_predict

        o_last = self._observations[-1]
        t_now = o_last.timestamp

        # Drag-aware: sample predictor at fine time steps
        try:
            dt_sample = 0.004
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
        self._frame_count += 1

        should_replan = (
            len(self._observations) >= self.MIN_OBS
            and self._frame_count % self.REPLAN_INTERVAL == 0
        )

        if should_replan:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            # Override t_contact with actual remaining time to intercept
            paddle_target.t_contact = max(t_intercept - obs.timestamp, 0.05)

            # Estimate current joint velocity for warm-start
            dq_current = None
            if self._prev_q is not None:
                dq_current = (q_current - self._prev_q) / (self.REPLAN_INTERVAL * self.DT)

            # Pass dq_current if swing planner supports it (QuinticSwingPlanner)
            try:
                trajectory = self.swing_planner.plan(paddle_target, q_current, dq_current=dq_current)
            except TypeError:
                trajectory = self.swing_planner.plan(paddle_target, q_current)

            self._prev_q = q_current.copy()

            # Look ahead: use a fraction of time-to-contact, not just one
            # replan interval.  This commits to more of the swing while still
            # allowing course correction at the next replan.
            replan_dt = self.REPLAN_INTERVAL * self.DT
            traj_times = trajectory.times
            traj_duration = traj_times[-1] - traj_times[0]
            # Look ahead by 2x replan interval or 30% of trajectory, whichever is larger
            dt_ahead = min(max(2 * replan_dt, 0.3 * traj_duration), traj_duration)
            t_query = traj_times[0] + dt_ahead

            n_joints = trajectory.positions.shape[1]
            self._last_action = np.array([
                np.interp(t_query, traj_times, trajectory.positions[:, j])
                for j in range(n_joints)
            ])

        if self._last_action is not None:
            return self._last_action

        return q_current

    def reset(self) -> None:
        """Clear observation history between rallies."""
        self._observations.clear()
        self._frame_count = 0
        self._last_action = None
        self._prev_q = None


class MPCController(HighLevelController):
    """Stage 2a – Receding-horizon MPC controller.

    Uses CasADi to solve a joint-space NLP at each replan cycle.
    Warm-starts from previous solution to maintain trajectory continuity.
    Replaces both swing planner and replan controller.

    Key design choices for heavy arms (Fanuc):
    - Initial velocity constraint ensures smooth trajectory transitions
    - Quintic warm-start on first solve provides high-quality initial guess
    - Lookahead of 2*dt gives PD controller proper reference tracking
    - Configurable replan interval (default 8 = 64ms) balances reactivity vs. commitment
    """

    MIN_OBS = 4
    DT = 0.008            # sim timestep

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
        fk_dict: dict | None = None,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
        horizon_N: int = 15,
        dt_mpc: float = 0.032,
        max_iter: int = 80,
        w_pos: float = 1000.0,
        w_ori: float = 100.0,
        w_smooth: float = 0.01,
        w_effort: float = 0.0,
        dq_max: float = 5.0,
        follow_through: float = 0.0,
        replan_interval: int = 4,
        lookahead_fraction: float = 0.6,
    ) -> None:
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self.t_predict = t_predict
        self.ROBOT_X = robot_x
        self.follow_through = follow_through
        self.replan_interval = replan_interval
        self.lookahead_fraction = lookahead_fraction

        self._observations: list[BallObservation] = []
        self._frame_count = 0
        self._prev_q: np.ndarray | None = None
        self._dq_current: np.ndarray | None = None

        # MPC params
        self.N = horizon_N
        self.dt_mpc = dt_mpc
        self.max_iter = max_iter
        self.dq_max = dq_max
        self.weights = {
            'pos': w_pos, 'ori': w_ori,
            'smooth': w_smooth, 'effort': w_effort,
        }

        # FK and NLP (built lazily)
        self._fk = fk_dict
        self._nlp_solver = None
        self._n_joints: int | None = None
        self._joint_ranges: np.ndarray | None = None

        # Warm-start state
        self._prev_solution: np.ndarray | None = None
        self._solution_trajectory: tuple | None = None
        self._solution_start_time: float | None = None

    def _build_nlp(self) -> None:
        """Build the CasADi NLP (called once).
        
        NLP structure:
        - Decision vars: Q = [q_1, ..., q_N] (N*nj variables)
        - Parameters: [q0, dq0, target_pos, target_normal, dt]
        - Constraints: initial velocity match + velocity limits
        """
        import casadi as ca

        nj = self._n_joints
        N = self.N
        w = self.weights

        # Decision variables: q_1, ..., q_N (q_0 is parameter = current state)
        Q = ca.SX.sym('Q', N * nj)
        q0_param = ca.SX.sym('q0', nj)
        target_pos = ca.SX.sym('target_pos', 3)
        target_normal = ca.SX.sym('target_normal', 3)
        dt_param = ca.SX.sym('dt_param')  # adaptive timestep

        params = ca.vertcat(q0_param, target_pos, target_normal, dt_param)

        fk_fn = self._fk['fk_fn']
        fk_rot_fn = self._fk['fk_rot_fn']

        cost = 0
        g = []  # constraints
        lbg = []
        ubg = []

        def get_q(k):
            if k == 0:
                return q0_param
            return Q[(k - 1) * nj: k * nj]

        # Running costs over horizon
        for k in range(1, N + 1):
            qk = get_q(k)
            qk_prev = get_q(k - 1)
            dq = (qk - qk_prev) / dt_param

            # Effort: minimize velocity
            cost += w['effort'] * ca.dot(dq, dq)

            # Velocity limits
            g.append(dq)
            lbg += [-self.dq_max] * nj
            ubg += [self.dq_max] * nj

            # Smoothness (acceleration penalty) for k >= 2
            if k >= 2:
                qk_prev2 = get_q(k - 2)
                ddq = (qk - 2 * qk_prev + qk_prev2) / (dt_param * dt_param)
                cost += w['smooth'] * ca.dot(ddq, ddq)

        # Terminal costs
        q_end = get_q(N)
        p_ee = fk_fn(q_end)
        _, R_flat = fk_rot_fn(q_end)
        # R_flat is row-major [R00,R01,R02,R10,...,R22]
        # Body Z-axis (column 2 of R) = [R02, R12, R22] = indices 2,5,8
        z_ee = ca.vertcat(R_flat[2], R_flat[5], R_flat[8])

        # Position tracking
        pos_err = p_ee - target_pos
        cost += w['pos'] * ca.dot(pos_err, pos_err)

        # Orientation tracking (cross product error)
        ori_err = ca.cross(z_ee, target_normal)
        cost += w['ori'] * ca.dot(ori_err, ori_err)

        # Joint limits as bounds
        lbx = []
        ubx = []
        for k in range(N):
            for j in range(nj):
                lbx.append(float(self._joint_ranges[j, 0]))
                ubx.append(float(self._joint_ranges[j, 1]))

        # Build NLP
        nlp = {'x': Q, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
        opts = {
            'ipopt.max_iter': self.max_iter,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_init': 1e-3,
        }
        self._nlp_solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = lbg
        self._ubg = ubg
        self._n_params = params.shape[0]

    def _solve(
        self,
        q_current: np.ndarray,
        dq_current: np.ndarray,
        target_pos: np.ndarray,
        target_normal: np.ndarray,
        dt_mpc: float,
    ) -> np.ndarray:
        """Solve the NLP, return (N, n_joints) trajectory."""
        nj = self._n_joints
        N = self.N

        p_val = np.concatenate([
            q_current, target_pos, target_normal, [dt_mpc]
        ])

        # Warm-start: shift previous solution forward, or hold current position
        if self._prev_solution is not None:
            x0 = np.zeros(N * nj)
            prev = self._prev_solution
            for k in range(N):
                src = min(k + 1, N - 1)
                x0[k * nj: (k + 1) * nj] = prev[src]
        else:
            x0 = np.tile(q_current, N)

        sol = self._nlp_solver(
            x0=x0, lbx=self._lbx, ubx=self._ubx,
            lbg=self._lbg, ubg=self._ubg, p=p_val,
        )

        Q_opt = np.array(sol['x']).flatten()
        trajectory = Q_opt.reshape(N, nj)
        self._prev_solution = trajectory.copy()
        return trajectory

    def _estimate_intercept_time(self) -> float:
        """Estimate when the ball will reach ROBOT_X."""
        if len(self._observations) < 2:
            return self._observations[-1].timestamp + self.t_predict

        o_last = self._observations[-1]
        t_now = o_last.timestamp

        try:
            dt_sample = 0.004
            n_samples = int(self.t_predict / dt_sample)
            for i in range(1, n_samples + 1):
                t_probe = t_now + i * dt_sample
                state = self.predictor.predict(self._observations, t_probe)
                if state.position[0] >= self.ROBOT_X:
                    return t_probe
        except Exception:
            pass

        o0 = self._observations[0]
        dt = o_last.timestamp - o0.timestamp
        if dt < 1e-6:
            return t_now + self.t_predict
        vx = (o_last.position[0] - o0.position[0]) / dt
        if vx < 0.1:
            return t_now + self.t_predict
        dx = self.ROBOT_X - o_last.position[0]
        return t_now + dx / vx

    def step(self, env_state: dict, q_current: np.ndarray, dq_current: np.ndarray | None = None) -> np.ndarray:
        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        self._frame_count += 1

        if dq_current is None:
            dq_current = np.zeros_like(q_current)
        self._dq_current = dq_current

        if self._n_joints is None:
            self._n_joints = len(q_current)
            if self._fk is not None:
                self._joint_ranges = self._fk['joint_ranges']
                self._build_nlp()

        should_replan = (
            self._fk is not None
            and self._nlp_solver is not None
            and len(self._observations) >= self.MIN_OBS
            and (
                self._solution_trajectory is None  # first plan: always trigger
                or self._frame_count % self.replan_interval == 0
            )
        )

        if should_replan:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            t_remaining = max(t_intercept - obs.timestamp, 0.05)
            paddle_target.t_contact = t_remaining

            # Adaptive dt so horizon covers time to intercept
            dt_mpc = max(t_remaining / self.N, 0.008)

            # Follow-through: offset target position along swing direction
            # so the optimizer implicitly produces velocity at contact
            target_pos = paddle_target.position.copy()
            if self.follow_through > 0 and paddle_target.velocity is not None:
                v_norm = np.linalg.norm(paddle_target.velocity)
                if v_norm > 1e-8:
                    swing_dir = paddle_target.velocity / v_norm
                    target_pos = target_pos + self.follow_through * swing_dir

            trajectory = self._solve(
                q_current,
                dq_current,
                target_pos,
                paddle_target.normal,
                dt_mpc,
            )

            # Store trajectory with timestamps for interpolation
            # Prepend current state so interp works between replans
            times = np.array([obs.timestamp] + [
                obs.timestamp + (k + 1) * dt_mpc for k in range(self.N)
            ])
            full_traj = np.vstack([q_current.reshape(1, -1), trajectory])
            self._solution_trajectory = (times, full_traj)
            self._solution_dt_mpc = dt_mpc

        # PD lookahead: return position ahead on trajectory as PD reference.
        # With high PD gains (kp_scale=800), the arm needs a large position error
        # to generate sufficient torque for fast swings. Query 60% of remaining time.
        if self._solution_trajectory is not None:
            times, traj = self._solution_trajectory
            t_now = obs.timestamp
            t_end = times[-1]
            t_remain = max(t_end - t_now, 0.01)
            t_query = min(t_now + self.lookahead_fraction * t_remain, t_end)
            result = np.array([
                np.interp(t_query, times, traj[:, j])
                for j in range(self._n_joints)
            ])
            return result

        return q_current

    def reset(self) -> None:
        """Clear state between rallies."""
        self._observations.clear()
        self._frame_count = 0
        self._prev_q = None
        self._dq_current = None
        self._prev_solution = None
        self._solution_trajectory = None
        self._solution_start_time = None


class TorqueMPCController(HighLevelController):
    """Stage 2b – Torque-level receding-horizon MPC controller.

    Unlike MPCController (position-level → PD), this controller directly outputs
    torques in ctrl-space. The NLP includes full rigid-body dynamics constraints
    (RNEA) so the optimizer produces physically consistent trajectories.

    The high-level controller absorbs the low-level controller: no PD loop needed.
    """

    REPLAN_INTERVAL = 4
    MIN_OBS = 4
    DT = 0.008

    torque_mode = True  # signals run.py to bypass PD loop

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
        dynamics_dict: dict | None = None,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
        horizon_N: int = 15,
        dt_mpc: float = 0.032,
        max_iter: int = 150,
        w_pos: float = 1000.0,
        w_ori: float = 0.0,
        w_smooth: float = 0.01,
        w_effort: float = 0.0,
        dq_max: float = 5.0,
        follow_through: float = 0.0,
    ) -> None:
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self.t_predict = t_predict
        self.ROBOT_X = robot_x
        self.follow_through = follow_through

        self._observations: list[BallObservation] = []
        self._frame_count = 0

        # MPC params
        self.N = horizon_N
        self.dt_mpc = dt_mpc
        self.max_iter = max_iter
        self.dq_max = dq_max
        self.weights = {
            'pos': w_pos, 'ori': w_ori,
            'smooth': w_smooth, 'effort': w_effort,
        }

        # Dynamics and NLP
        self._dyn = dynamics_dict
        self._nlp_solver = None
        self._n_joints: int | None = None
        self._joint_ranges: np.ndarray | None = None
        self._gear: np.ndarray | None = None
        self._damping: np.ndarray | None = None

        # Warm-start
        self._prev_x0: np.ndarray | None = None
        # Torque trajectory for interpolation between replans
        self._solution_times: np.ndarray | None = None
        self._solution_tau: np.ndarray | None = None  # (N, nj) ctrl-space
        self._solution_q: np.ndarray | None = None    # (N+1, nj) positions
        self._solution_qd: np.ndarray | None = None   # (N+1, nj) velocities

    def _build_nlp(self) -> None:
        """Build torque-level CasADi NLP (called once)."""
        import casadi as ca

        nj = self._n_joints
        N = self.N
        w = self.weights

        # Decision variables:
        #   Q:   positions  q[1..N]    (N * nj)
        #   QD:  velocities qd[1..N]   (N * nj)
        #   TAU: torques    tau[0..N-1] (N * nj)
        n_dec = 3 * N * nj
        X = ca.SX.sym('X', n_dec)

        def get_q(k):
            """Position at knot k (k=0 is parameter)."""
            if k == 0:
                return q0_param
            return X[(k - 1) * nj: k * nj]

        def get_qd(k):
            """Velocity at knot k (k=0 is parameter)."""
            if k == 0:
                return qd0_param
            offset = N * nj
            return X[offset + (k - 1) * nj: offset + k * nj]

        def get_tau(k):
            """Torque at knot k (k=0..N-1)."""
            offset = 2 * N * nj
            return X[offset + k * nj: offset + (k + 1) * nj]

        # Parameters
        q0_param = ca.SX.sym('q0', nj)
        qd0_param = ca.SX.sym('qd0', nj)
        target_pos = ca.SX.sym('target_pos', 3)
        target_normal = ca.SX.sym('target_normal', 3)
        dt_param = ca.SX.sym('dt_param')

        params = ca.vertcat(q0_param, qd0_param, target_pos, target_normal, dt_param)

        rnea_fn = self._dyn['rnea_fn']
        fk_fn = self._dyn['fk_fn']
        fk_rot_fn = self._dyn['fk_rot_fn']
        damping = ca.SX(self._damping)

        cost = 0
        g = []
        lbg = []
        ubg = []

        # Dynamics constraints: for k = 0..N-1
        # qdd_k = (qd_{k+1} - qd_k) / dt
        # tau_k = rnea(q_k, qd_k, qdd_k)  [RNEA includes gravity + Coriolis + inertia + damping]
        for k in range(N):
            q_k = get_q(k)
            qd_k = get_qd(k)
            q_k1 = get_q(k + 1)
            qd_k1 = get_qd(k + 1)
            tau_k = get_tau(k)

            # Acceleration (finite difference)
            qdd_k = (qd_k1 - qd_k) / dt_param

            # Inverse dynamics: tau = RNEA(q, qd, qdd)
            tau_rnea = rnea_fn(q_k, qd_k, qdd_k)

            # Dynamics constraint: tau_k = tau_rnea
            g.append(tau_k - tau_rnea)
            lbg += [0.0] * nj
            ubg += [0.0] * nj

            # Position integration (semi-implicit Euler): q_{k+1} = q_k + dt * qd_{k+1}
            g.append(q_k1 - q_k - dt_param * qd_k1)
            lbg += [0.0] * nj
            ubg += [0.0] * nj

            # Running cost: torque effort
            cost += w['effort'] * ca.dot(tau_k, tau_k)

            # Torque smoothness (for k >= 1)
            if k >= 1:
                tau_prev = get_tau(k - 1)
                dtau = tau_k - tau_prev
                cost += w['smooth'] * ca.dot(dtau, dtau)

        # Terminal costs
        q_end = get_q(N)
        qd_end = get_qd(N)
        p_ee = fk_fn(q_end)
        _, R_flat = fk_rot_fn(q_end)
        z_ee = ca.vertcat(R_flat[2], R_flat[5], R_flat[8])

        pos_err = p_ee - target_pos
        cost += w['pos'] * ca.dot(pos_err, pos_err)

        ori_err = ca.cross(z_ee, target_normal)
        cost += w['ori'] * ca.dot(ori_err, ori_err)

        # Bounds on decision variables
        lbx = []
        ubx = []
        gear = self._gear

        # Q bounds (positions): N * nj
        for k in range(N):
            for j in range(nj):
                lbx.append(float(self._joint_ranges[j, 0]))
                ubx.append(float(self._joint_ranges[j, 1]))

        # QD bounds (velocities): N * nj
        for k in range(N):
            for j in range(nj):
                lbx.append(-self.dq_max)
                ubx.append(self.dq_max)

        # TAU bounds (torques): N * nj — bounded by actuator limits (gear ratios)
        for k in range(N):
            for j in range(nj):
                lbx.append(-float(gear[j]))
                ubx.append(float(gear[j]))

        nlp = {'x': X, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
        opts = {
            'ipopt.max_iter': self.max_iter,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_init': 1e-3,
            'ipopt.tol': 1e-4,
        }
        self._nlp_solver = ca.nlpsol('torque_mpc', 'ipopt', nlp, opts)
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = lbg
        self._ubg = ubg

    def _solve(
        self,
        q_current: np.ndarray,
        qd_current: np.ndarray,
        target_pos: np.ndarray,
        target_normal: np.ndarray,
        dt_mpc: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the NLP. Returns (q_traj (N,nj), qd_traj (N,nj), tau_traj (N,nj))."""
        nj = self._n_joints
        N = self.N

        p_val = np.concatenate([
            q_current, qd_current, target_pos, target_normal, [dt_mpc]
        ])

        # Initial guess
        if self._prev_x0 is not None:
            x0 = self._prev_x0.copy()
            # Shift warm-start forward by one step
            # Q block
            for k in range(N - 1):
                x0[k * nj:(k + 1) * nj] = self._prev_x0[(k + 1) * nj:(k + 2) * nj]
            x0[(N - 1) * nj:N * nj] = self._prev_x0[(N - 1) * nj:N * nj]
            # QD block
            off = N * nj
            for k in range(N - 1):
                x0[off + k * nj:off + (k + 1) * nj] = self._prev_x0[off + (k + 1) * nj:off + (k + 2) * nj]
            x0[off + (N - 1) * nj:off + N * nj] = self._prev_x0[off + (N - 1) * nj:off + N * nj]
            # TAU block
            off2 = 2 * N * nj
            for k in range(N - 1):
                x0[off2 + k * nj:off2 + (k + 1) * nj] = self._prev_x0[off2 + (k + 1) * nj:off2 + (k + 2) * nj]
            x0[off2 + (N - 1) * nj:off2 + N * nj] = self._prev_x0[off2 + (N - 1) * nj:off2 + N * nj]
        else:
            # Hold current position, zero velocity, gravity-comp torque
            x0 = np.zeros(3 * N * nj)
            # Q: tile current position
            for k in range(N):
                x0[k * nj:(k + 1) * nj] = q_current
            # QD: zero (already)
            # TAU: gravity compensation at current config
            g_tau = np.array(self._dyn['gravity_fn'](q_current)).flatten()
            off2 = 2 * N * nj
            for k in range(N):
                x0[off2 + k * nj:off2 + (k + 1) * nj] = g_tau

        sol = self._nlp_solver(
            x0=x0, lbx=self._lbx, ubx=self._ubx,
            lbg=self._lbg, ubg=self._ubg, p=p_val,
        )

        X_opt = np.array(sol['x']).flatten()
        self._prev_x0 = X_opt.copy()

        q_traj = X_opt[:N * nj].reshape(N, nj)
        qd_traj = X_opt[N * nj:2 * N * nj].reshape(N, nj)
        tau_traj = X_opt[2 * N * nj:].reshape(N, nj)

        return q_traj, qd_traj, tau_traj

    def _estimate_intercept_time(self) -> float:
        """Estimate when the ball will reach ROBOT_X."""
        if len(self._observations) < 2:
            return self._observations[-1].timestamp + self.t_predict

        o_last = self._observations[-1]
        t_now = o_last.timestamp

        try:
            dt_sample = 0.004
            n_samples = int(self.t_predict / dt_sample)
            for i in range(1, n_samples + 1):
                t_probe = t_now + i * dt_sample
                state = self.predictor.predict(self._observations, t_probe)
                if state.position[0] >= self.ROBOT_X:
                    return t_probe
        except Exception:
            pass

        o0 = self._observations[0]
        dt = o_last.timestamp - o0.timestamp
        if dt < 1e-6:
            return t_now + self.t_predict
        vx = (o_last.position[0] - o0.position[0]) / dt
        if vx < 0.1:
            return t_now + self.t_predict
        dx = self.ROBOT_X - o_last.position[0]
        return t_now + dx / vx

    def step(self, env_state: dict, q_current: np.ndarray, qd_current: np.ndarray | None = None) -> np.ndarray:
        """Return ctrl-space torques (torque / gear, clipped to [-1, 1])."""
        if qd_current is None:
            qd_current = np.zeros_like(q_current)

        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        self._frame_count += 1

        if self._n_joints is None:
            self._n_joints = len(q_current)
            if self._dyn is not None:
                self._joint_ranges = self._dyn['joint_ranges']
                self._gear = self._dyn['gear_ratios']
                self._damping = self._dyn['joint_damping']
                self._build_nlp()

        should_replan = (
            self._dyn is not None
            and self._nlp_solver is not None
            and len(self._observations) >= self.MIN_OBS
            and (
                self._solution_times is None  # first plan: always trigger
                or self._frame_count % self.REPLAN_INTERVAL == 0
            )
        )

        if should_replan:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            t_remaining = max(t_intercept - obs.timestamp, 0.05)
            paddle_target.t_contact = t_remaining

            dt_mpc = max(t_remaining / self.N, 0.008)

            # Follow-through: offset target position along swing direction
            target_pos = paddle_target.position.copy()
            if self.follow_through > 0 and paddle_target.velocity is not None:
                v_norm = np.linalg.norm(paddle_target.velocity)
                if v_norm > 1e-8:
                    swing_dir = paddle_target.velocity / v_norm
                    target_pos = target_pos + self.follow_through * swing_dir

            q_traj, qd_traj, tau_traj = self._solve(
                q_current, qd_current,
                target_pos, paddle_target.normal,
                dt_mpc,
            )

            # Store solution for interpolation
            times = np.array([obs.timestamp + (k + 1) * dt_mpc for k in range(self.N)])
            self._solution_times = times
            self._solution_tau = tau_traj  # (N, nj) in torque space (Nm)

        # Interpolate torque at current time
        if self._solution_times is not None and self._solution_tau is not None:
            t_now = obs.timestamp
            # Find the appropriate knot — use the first knot that's >= t_now
            # or the last knot if we've passed the horizon
            tau = np.array([
                np.interp(t_now, self._solution_times, self._solution_tau[:, j])
                for j in range(self._n_joints)
            ])
            # Convert to ctrl space: ctrl = tau / gear
            ctrl = tau / self._gear
            return np.clip(ctrl, -1.0, 1.0)

        # No solution yet: gravity compensation to hold position
        if self._dyn is not None and self._gear is not None:
            g_tau = np.array(self._dyn['gravity_fn'](q_current)).flatten()
            ctrl = g_tau / self._gear
            return np.clip(ctrl, -1.0, 1.0)
        return np.zeros_like(q_current)

    def reset(self) -> None:
        """Clear state between rallies."""
        self._observations.clear()
        self._frame_count = 0
        self._prev_x0 = None
        self._solution_times = None
        self._solution_tau = None
        self._solution_q = None
        self._solution_qd = None


class TubeTorqueMPCController(TorqueMPCController):
    """Stage 2c – Tube MPC: robust torque-level MPC with ancillary LQR.

    Wraps TorqueMPCController with:
    1. Tightened constraints based on disturbance set W
    2. Online LTV-LQR gain computation along nominal trajectory
    3. Ancillary feedback: tau = tau_nom + K_k @ (x - x_nom_k)
    """

    def __init__(
        self,
        *args,
        disturbance_W: np.ndarray | None = None,
        tube_margin: float = 0.8,
        lqr_Q_scale: float = 1.0,
        lqr_R_scale: float = 10.0,
        correction_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._disturbance_W = disturbance_W
        self._tube_margin = tube_margin
        self._lqr_Q_scale = lqr_Q_scale
        self._lqr_R_scale = lqr_R_scale
        self._correction_enabled = correction_enabled

        # Tube state
        self._lin_fn = None
        self._tube_gains: list[np.ndarray] | None = None
        self._nominal_q: np.ndarray | None = None
        self._nominal_qd: np.ndarray | None = None
        self._nominal_tau: np.ndarray | None = None
        # Low-pass filtered deviation (exponential moving average)
        self._dx_filtered: np.ndarray | None = None
        self._lpf_alpha: float = 0.3  # smoothing factor (0=ignore new, 1=no filter)

    def _build_linearization(self) -> None:
        """Build linearization function from dynamics_dict (called once)."""
        from tt_sim.upgrades.tube_linearize import build_linearization_fn
        self._lin_fn = build_linearization_fn(self._dyn, self._n_joints, self.DT)

    def _solve(self, *args, **kwargs):
        """Override to store full trajectory for tube gains."""
        q_traj, qd_traj, tau_traj = super()._solve(*args, **kwargs)
        # Store nominal trajectory with start state prepended
        q_start = args[0]  # q_current
        qd_start = args[1]  # qd_current
        self._nominal_q = np.vstack([q_start[None, :], q_traj])
        self._nominal_qd = np.vstack([qd_start[None, :], qd_traj])
        self._nominal_tau = tau_traj
        # Reset gains so they're recomputed
        self._tube_gains = None
        self._dx_filtered = None  # reset filter on replan (new nominal)
        return q_traj, qd_traj, tau_traj

    def _compute_tube_gains(
        self, q_traj: np.ndarray, qd_traj: np.ndarray, tau_traj: np.ndarray, dt_mpc: float
    ) -> list[np.ndarray]:
        """Linearize along nominal and compute LTV-LQR gains."""
        from tt_sim.upgrades.tube_linearize import build_linearization_fn
        from tt_sim.upgrades.tube_lqr import compute_ltv_gains

        nj = self._n_joints
        N = len(tau_traj)

        # Build linearization at actual dt_mpc (not self.DT)
        lin_fn = build_linearization_fn(self._dyn, nj, dt_mpc)

        # Linearize at each knot
        As, Bs = [], []
        for k in range(N):
            # q_traj[k] is q at knot k+1, tau_traj[k] is tau at knot k
            # Linearize at (q_k, qd_k, tau_k) — use nominal_q which has start prepended
            q_k = self._nominal_q[k]    # knot k (index 0 = start state)
            qd_k = self._nominal_qd[k]  # knot k
            A, B = lin_fn(q_k, qd_k, tau_traj[k])
            As.append(A)
            Bs.append(B)

        # LQR tuning
        nx = 2 * nj
        Q = np.eye(nx) * self._lqr_Q_scale
        R = np.eye(nj) * self._lqr_R_scale

        Ks = compute_ltv_gains(As, Bs, Q, R)
        return Ks

    def step(self, env_state: dict, q_current: np.ndarray, qd_current: np.ndarray | None = None) -> np.ndarray:
        """Return ctrl-space torques with ancillary tube correction."""
        if qd_current is None:
            qd_current = np.zeros_like(q_current)

        # Call parent step to get nominal control
        ctrl = super().step(env_state, q_current, qd_current)

        # After parent replans, compute tube gains from its solution
        if (self._solution_tau is not None and self._tube_gains is None
                and self._n_joints is not None and self._nominal_q is not None):
            try:
                dt_mpc = (self._solution_times[1] - self._solution_times[0]) if len(self._solution_times) > 1 else 0.032
                self._tube_gains = self._compute_tube_gains(
                    self._nominal_q[1:], self._nominal_qd[1:],
                    self._nominal_tau, dt_mpc
                )
                self._tube_dt_mpc = dt_mpc
            except Exception:
                self._tube_gains = None

        # Apply ancillary correction (only if enabled)
        if (self._correction_enabled and self._tube_gains is not None 
                and self._solution_times is not None):
            nj = self._n_joints
            t_now = self._observations[-1].timestamp
            k_idx = np.searchsorted(self._solution_times, t_now)
            k_idx = min(k_idx, len(self._tube_gains) - 1)

            K_k = self._tube_gains[k_idx]
            q_nom = self._nominal_q[k_idx + 1]
            qd_nom = self._nominal_qd[k_idx + 1]
            dx_raw = np.concatenate([q_current - q_nom, qd_current - qd_nom])

            # Zero out velocity deviation — structural interpolation mismatch
            dx_raw[nj:] = 0.0

            # Low-pass filter deviation to reject transient noise
            # Only respond to persistent tracking errors, not single-step noise
            if self._dx_filtered is None:
                self._dx_filtered = dx_raw.copy()
            else:
                self._dx_filtered = (self._lpf_alpha * dx_raw 
                                     + (1 - self._lpf_alpha) * self._dx_filtered)

            # Scale correction: gains at dt_mpc, applied at dt
            dt_ratio = self.DT / self._tube_dt_mpc
            correction_gain = 0.1
            
            tau_correction = -correction_gain * dt_ratio * K_k @ self._dx_filtered

            # Clip correction (max 5% of gear capacity per control step)
            max_correction = self._gear * 0.05
            tau_correction = np.clip(tau_correction, -max_correction, max_correction)

            # Add correction in ctrl space
            ctrl_correction = tau_correction / self._gear
            ctrl = np.clip(ctrl + ctrl_correction, -1.0, 1.0)

        return ctrl
