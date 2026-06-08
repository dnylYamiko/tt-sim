# MPC Controller (Stage 2a) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace replan controller with receding-horizon MPC using CasADi to eliminate IK discontinuity on Fanuc while adding closed-loop correction.

**Architecture:** A new `MPCController` in `tt_sim/upgrades/control.py` solves a CasADi NLP at each replan cycle. Symbolic forward kinematics are built from MuJoCo XML body transforms. Warm-starting from the previous shifted solution keeps solutions continuous across replans. The controller replaces both the swing planner and replan controller — it directly outputs `q_desired` by optimizing a joint trajectory over a receding horizon toward the predicted ball intercept.

**Tech Stack:** CasADi 3.7.2 (IPOPT solver), NumPy, MuJoCo (for extracting kinematic chain at init)

**Verification command:** `cd /Users/baigm2/Documents/uw/ME595/hw1/tt-sim && python -m pytest tests/ -x -q`

**Evaluation commands:**
- Fanuc: `python run.py --robot=fanuc --predictor=drag_bounce --aimer=specular --swing=quintic --control=mpc --episodes=40`
- WAM: `python run.py --robot=wam --predictor=drag_bounce --aimer=specular --swing=quintic --control=mpc --episodes=40`

**Target:** Fanuc contact ≥ 45% (matching open_loop quintic), WAM contact ≥ 20% (matching replan)

---

## Task 1: Symbolic FK builder

Build a function that extracts the kinematic chain from a MuJoCo model and returns a CasADi symbolic FK function.

**Files:**
- Create: `tt_sim/upgrades/mpc_fk.py`
- Test: `tests/test_upgrades/test_mpc_fk.py`

### Step 1: Write the FK builder

Create `tt_sim/upgrades/mpc_fk.py`:

```python
"""Symbolic forward kinematics from MuJoCo model for CasADi NLP."""

from __future__ import annotations

import casadi as ca
import mujoco
import numpy as np


def _rotation_matrix(axis: np.ndarray, theta: ca.SX) -> ca.SX:
    """Rodrigues rotation: 3x3 rotation matrix for angle theta about unit axis."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    c = ca.cos(theta)
    s = ca.sin(theta)
    C = 1 - c
    return ca.vertcat(
        ca.horzcat(x*x*C + c,   x*y*C - z*s, x*z*C + y*s),
        ca.horzcat(y*x*C + z*s, y*y*C + c,   y*z*C - x*s),
        ca.horzcat(z*x*C - y*s, z*y*C + x*s, z*z*C + c),
    )


def build_fk_casadi(model: mujoco.MjModel, n_joints: int) -> dict:
    """Build symbolic FK from MuJoCo model.

    Returns dict with:
        'q': CasADi SX symbol (n_joints,)
        'p_ee': CasADi SX expression (3,) — EE position in world frame
        'R_ee': CasADi SX expression (3,3) — EE rotation in world frame
        'fk_fn': CasADi Function  q -> p_ee
        'fk_rot_fn': CasADi Function  q -> (p_ee, R_ee_flat)
        'joint_ranges': (n_joints, 2) numpy array of (lo, hi)
    """
    # Extract kinematic chain from MuJoCo model
    # Walk the body tree from worldbody to EE, collecting joint axes and body offsets
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE")

    # Build chain from EE back to world
    chain = []
    body_id = ee_id
    while body_id > 0:
        chain.append(body_id)
        body_id = model.body_parentid[body_id]
    chain.reverse()  # world -> ... -> EE

    # For each body in chain, get:
    #   - body offset from parent: model.body_pos[body_id]
    #   - body quaternion from parent: model.body_quat[body_id]
    #   - joint (if any): axis, range

    q = ca.SX.sym('q', n_joints)
    joint_ranges = np.zeros((n_joints, 2))

    # Start with identity transform
    R = ca.SX.eye(3)
    p = ca.SX.zeros(3)

    joint_idx = 0
    for body_id in chain:
        # Body frame offset from parent
        pos = model.body_pos[body_id].copy()
        quat = model.body_quat[body_id].copy()  # (w, x, y, z)

        # Convert body quaternion to rotation matrix (constant)
        w, x, y, z = quat
        body_R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])

        # Apply body transform: p = p + R @ pos, R = R @ body_R
        p = p + R @ pos
        if not np.allclose(quat, [1, 0, 0, 0]):
            R = R @ body_R

        # Check if this body has a joint
        # Find joints belonging to this body
        for jnt_id in range(model.njnt):
            if model.jnt_bodyid[jnt_id] == body_id and joint_idx < n_joints:
                axis = model.jnt_axis[jnt_id].copy()
                joint_ranges[joint_idx] = model.jnt_range[jnt_id].copy()

                # Apply joint rotation
                R_jnt = _rotation_matrix(axis, q[joint_idx])
                R = R @ R_jnt
                joint_idx += 1

    p_ee = p
    R_ee = R

    fk_fn = ca.Function('fk_pos', [q], [p_ee], ['q'], ['p_ee'])
    R_flat = ca.reshape(R_ee, 9, 1)
    fk_rot_fn = ca.Function('fk_rot', [q], [p_ee, R_flat], ['q'], ['p_ee', 'R_flat'])

    return {
        'q': q,
        'p_ee': p_ee,
        'R_ee': R_ee,
        'fk_fn': fk_fn,
        'fk_rot_fn': fk_rot_fn,
        'joint_ranges': joint_ranges,
    }
```

### Step 2: Write tests

Create `tests/test_upgrades/test_mpc_fk.py`:

```python
"""Tests for symbolic FK builder — validates against MuJoCo FK."""

import pytest
import numpy as np

# Skip all tests if casadi or mujoco unavailable
casadi = pytest.importorskip("casadi")
mujoco = pytest.importorskip("mujoco")

from tt_sim.upgrades.mpc_fk import build_fk_casadi


def _load_model(robot: str):
    """Load MuJoCo model for a robot."""
    import os
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'xml')
    if robot == 'fanuc':
        xml = os.path.join(base, 'table_tennis_fanuc.xml')
        ndof = 6
    else:
        xml = os.path.join(base, 'table_tennis_env.xml')
        ndof = 7
    model = mujoco.MjModel.from_xml_path(xml)
    return model, ndof


def _mujoco_fk(model, q, ndof):
    """Ground truth FK from MuJoCo."""
    d = mujoco.MjData(model)
    d.qpos[:ndof] = q
    mujoco.mj_forward(model, d)
    ee_pos = d.body("EE").xpos.copy()
    ee_rot = d.body("EE").xmat.reshape(3, 3).copy()
    return ee_pos, ee_rot


@pytest.mark.parametrize("robot", ["fanuc", "wam"])
def test_fk_matches_mujoco_at_home(robot):
    """Symbolic FK matches MuJoCo FK at home position."""
    model, ndof = _load_model(robot)
    fk = build_fk_casadi(model, ndof)

    q_home = np.zeros(ndof)
    p_sym = np.array(fk['fk_fn'](q_home)).flatten()
    p_mj, _ = _mujoco_fk(model, q_home, ndof)

    np.testing.assert_allclose(p_sym, p_mj, atol=0.02,
        err_msg=f"{robot} FK position mismatch at home")


@pytest.mark.parametrize("robot", ["fanuc", "wam"])
def test_fk_matches_mujoco_at_random(robot):
    """Symbolic FK matches MuJoCo FK at several random configs."""
    model, ndof = _load_model(robot)
    fk = build_fk_casadi(model, ndof)

    rng = np.random.default_rng(42)
    for _ in range(5):
        q = rng.uniform(fk['joint_ranges'][:, 0] * 0.3,
                        fk['joint_ranges'][:, 1] * 0.3)
        p_sym = np.array(fk['fk_fn'](q)).flatten()
        p_mj, _ = _mujoco_fk(model, q, ndof)

        np.testing.assert_allclose(p_sym, p_mj, atol=0.02,
            err_msg=f"{robot} FK mismatch at q={q.round(3)}")


@pytest.mark.parametrize("robot", ["fanuc", "wam"])
def test_fk_orientation_matches(robot):
    """Symbolic FK rotation matches MuJoCo at a test config."""
    model, ndof = _load_model(robot)
    fk = build_fk_casadi(model, ndof)

    q = np.zeros(ndof)
    q[0] = 0.5  # rotate base joint
    p_sym, R_flat = fk['fk_rot_fn'](q)
    R_sym = np.array(R_flat).reshape(3, 3)
    _, R_mj = _mujoco_fk(model, q, ndof)

    np.testing.assert_allclose(R_sym, R_mj, atol=0.05,
        err_msg=f"{robot} FK rotation mismatch")


def test_joint_ranges_extracted():
    """Joint ranges are correctly extracted from model."""
    model, ndof = _load_model("fanuc")
    fk = build_fk_casadi(model, ndof)
    ranges = fk['joint_ranges']

    assert ranges.shape == (6, 2)
    # All Fanuc joints have symmetric ranges >= pi
    for i in range(6):
        assert ranges[i, 0] < 0
        assert ranges[i, 1] > 0
        assert ranges[i, 1] >= np.pi * 0.9


def test_fk_is_differentiable():
    """CasADi function has valid Jacobian."""
    model, ndof = _load_model("fanuc")
    fk = build_fk_casadi(model, ndof)

    q_test = np.zeros(ndof)
    J = np.array(fk['fk_fn'].jacobian()(q_test, np.zeros(3)))
    # Jacobian should be (3, 6) and have nonzero entries
    assert J.shape[0] == 3 or J.shape[1] == 6  # CasADi returns augmented
    assert np.any(np.abs(J) > 1e-6)
```

### Step 3: Run tests, iterate until FK matches MuJoCo

Run: `python -m pytest tests/test_upgrades/test_mpc_fk.py -v`

The FK builder may need iteration — MuJoCo's body tree includes non-joint bodies (pedestal, base_link) that have position offsets but no joints. The builder must correctly walk the chain and only consume `q[joint_idx]` when a joint is present.

**Common failure modes:**
- Body quaternion not applied correctly (WAM uses non-identity quaternions at many bodies)
- Joint axis in body-local frame vs world frame confusion
- Off-by-one in joint indexing (MuJoCo may have free joints for the ball)
- EE body has an offset but no joint

**Debug approach:** Compare `p_sym` vs `p_mj` at q=0 first. If those match, test random q. Print intermediate transforms if needed.

### Step 4: Commit

```
git add tt_sim/upgrades/mpc_fk.py tests/test_upgrades/test_mpc_fk.py
git commit -m "feat: symbolic FK builder from MuJoCo XML for CasADi MPC"
```

---

## Task 2: MPCController — NLP formulation and solver

The core MPC controller class that builds and solves the CasADi NLP.

**Files:**
- Modify: `tt_sim/upgrades/control.py` — add `MPCController` class
- Test: `tests/test_upgrades/test_mpc_control.py`

### Step 1: Implement MPCController

Add to `tt_sim/upgrades/control.py`:

```python
class MPCController(HighLevelController):
    """Stage 2a – Receding-horizon MPC controller.

    Uses CasADi to solve a joint-space NLP at each replan cycle.
    Warm-starts from previous solution to maintain trajectory continuity.
    Replaces both swing planner and replan controller.
    """

    REPLAN_INTERVAL = 4   # frames between NLP re-solves
    MIN_OBS = 4
    DT = 0.008            # sim timestep

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,  # kept for interface compat, not used in MPC
        fk_dict: dict | None = None,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
        horizon_N: int = 10,
        dt_mpc: float = 0.032,
        max_iter: int = 20,
        w_pos: float = 100.0,
        w_ori: float = 10.0,
        w_vel: float = 5.0,
        w_smooth: float = 1.0,
        w_effort: float = 0.1,
        dq_max: float = 3.0,
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
        self._prev_q: np.ndarray | None = None

        # MPC params
        self.N = horizon_N
        self.dt_mpc = dt_mpc
        self.max_iter = max_iter
        self.dq_max = dq_max
        self.weights = {
            'pos': w_pos, 'ori': w_ori, 'vel': w_vel,
            'smooth': w_smooth, 'effort': w_effort,
        }

        # FK and NLP (built lazily or at init)
        self._fk = fk_dict
        self._nlp_solver = None
        self._n_joints = None
        self._joint_ranges = None

        # Warm-start state
        self._prev_solution = None  # (N, n_joints) previous optimal trajectory
        self._solution_trajectory = None  # current full trajectory for interpolation
        self._solution_start_time = None

    def _build_nlp(self):
        """Build the CasADi NLP (called once)."""
        import casadi as ca

        nj = self._n_joints
        N = self.N
        dt = self.dt_mpc
        w = self.weights

        # Decision variables: q_1, ..., q_N (q_0 is parameter = current state)
        Q = ca.SX.sym('Q', N * nj)
        q0_param = ca.SX.sym('q0', nj)  # current joint state
        target_pos = ca.SX.sym('target_pos', 3)  # desired EE position at end
        target_normal = ca.SX.sym('target_normal', 3)  # desired paddle normal
        target_dq = ca.SX.sym('target_dq', nj)  # desired joint velocity at end

        params = ca.vertcat(q0_param, target_pos, target_normal, target_dq)

        fk_fn = self._fk['fk_fn']
        fk_rot_fn = self._fk['fk_rot_fn']

        cost = 0
        g = []  # constraints
        lbg = []
        ubg = []

        def get_q(k):
            if k == 0:
                return q0_param
            return Q[(k-1)*nj : k*nj]

        # Smoothness and effort over horizon
        for k in range(1, N + 1):
            qk = get_q(k)
            qk_prev = get_q(k - 1)
            dq = (qk - qk_prev) / dt

            # Effort: minimize velocity
            cost += w['effort'] * ca.dot(dq, dq)

            # Velocity limits
            g.append(dq)
            lbg += [-self.dq_max] * nj
            ubg += [self.dq_max] * nj

            # Smoothness (acceleration penalty) for k >= 2
            if k >= 2:
                qk_prev2 = get_q(k - 2)
                ddq = (qk - 2 * qk_prev + qk_prev2) / (dt * dt)
                cost += w['smooth'] * ca.dot(ddq, ddq)

        # Terminal costs
        q_end = get_q(N)
        p_ee = fk_fn(q_end)
        _, R_flat = fk_rot_fn(q_end)
        R_ee = ca.reshape(R_flat, 3, 3)
        z_ee = R_ee[:, 2]  # paddle face normal = EE body Z-axis

        # Position tracking
        pos_err = p_ee - target_pos
        cost += w['pos'] * ca.dot(pos_err, pos_err)

        # Orientation tracking (cross product error)
        ori_err = ca.cross(z_ee, target_normal)
        cost += w['ori'] * ca.dot(ori_err, ori_err)

        # Velocity at end
        q_end_prev = get_q(N - 1)
        dq_end = (q_end - q_end_prev) / dt
        vel_err = dq_end - target_dq
        cost += w['vel'] * ca.dot(vel_err, vel_err)

        # Joint limits as bounds (not constraints)
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

    def _solve(self, q_current, target_pos, target_normal, target_dq):
        """Solve the NLP, return (N, n_joints) trajectory."""
        import casadi as ca

        nj = self._n_joints
        N = self.N

        # Build parameter vector
        p_val = np.concatenate([q_current, target_pos, target_normal, target_dq])

        # Initial guess
        if self._prev_solution is not None:
            # Warm-start: shift previous solution by one step
            x0 = np.zeros(N * nj)
            prev = self._prev_solution  # (N, nj)
            # Shift: old[1] -> new[0], old[2] -> new[1], ..., duplicate last
            for k in range(N):
                src = min(k + 1, N - 1)
                x0[k*nj:(k+1)*nj] = prev[src]
        else:
            # Cold start: linear interp from q_current to q_current (hold position)
            # Or use IK if available — but for simplicity just hold
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

    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray:
        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        self._frame_count += 1

        if self._n_joints is None:
            self._n_joints = len(q_current)
            if self._fk is not None and self._nlp_solver is None:
                self._build_nlp()

        should_replan = (
            self._fk is not None
            and len(self._observations) >= self.MIN_OBS
            and self._frame_count % self.REPLAN_INTERVAL == 0
        )

        if should_replan and self._nlp_solver is None:
            self._build_nlp()

        if should_replan:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            paddle_target.t_contact = max(t_intercept - obs.timestamp, 0.05)

            # Compute desired joint velocity from Cartesian paddle velocity
            target_dq = np.zeros(self._n_joints)
            if paddle_target.velocity is not None and np.linalg.norm(paddle_target.velocity) > 1e-8:
                # Use FK Jacobian at current config for approximate mapping
                J_val = np.array(self._fk['fk_fn'].jacobian()(q_current, np.zeros(3)))
                # Extract the (3, n_joints) block
                J = J_val[:3, :self._n_joints]
                damping = 0.01
                JJT = J @ J.T + damping**2 * np.eye(3)
                target_dq = J.T @ np.linalg.solve(JJT, paddle_target.velocity)

            trajectory = self._solve(
                q_current,
                paddle_target.position,
                paddle_target.normal,
                target_dq,
            )

            # Store trajectory with timestamps for interpolation
            times = np.array([obs.timestamp + (k+1) * self.dt_mpc for k in range(self.N)])
            self._solution_trajectory = (times, trajectory)
            self._solution_start_time = obs.timestamp

        # Interpolate from current solution
        if self._solution_trajectory is not None:
            times, traj = self._solution_trajectory
            t_now = obs.timestamp
            # Find position at current time
            result = np.array([
                np.interp(t_now, times, traj[:, j])
                for j in range(self._n_joints)
            ])
            return result

        return q_current

    def reset(self) -> None:
        self._observations.clear()
        self._frame_count = 0
        self._prev_q = None
        self._prev_solution = None
        self._solution_trajectory = None
        self._solution_start_time = None
```

### Step 2: Write tests

Create `tests/test_upgrades/test_mpc_control.py`:

```python
"""Tests for MPCController."""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")

from tt_sim.interfaces import (
    HighLevelController, Perceiver, Predictor, SpinEstimator,
    Aimer, SwingPlanner, BallObservation, BallState, SpinEstimate,
    PaddleTarget, JointTrajectory,
)
from tt_sim.registry import load


# Reuse DummyPerceiver etc from test_control.py
from tests.test_upgrades.test_control import (
    DummyPerceiver, DummyPredictor, DummySpinEstimator,
    DummyAimer, DummySwingPlanner,
)

from tt_sim.upgrades.control import MPCController


def test_import():
    assert MPCController is not None


def test_subclass():
    assert issubclass(MPCController, HighLevelController)


def test_registry():
    cls = load("control", "mpc")
    assert cls is MPCController


def test_holds_without_fk():
    """Without FK dict, controller holds position."""
    ctrl = MPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), fk_dict=None,
    )
    q = np.array([0.5, 0.5])
    for _ in range(10):
        result = ctrl.step({}, q)
        np.testing.assert_array_equal(result, q)


def test_reset_clears_state():
    ctrl = MPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), fk_dict=None,
    )
    q = np.array([0.5, 0.5])
    for _ in range(10):
        ctrl.step({}, q)
    ctrl.reset()
    assert ctrl._observations == []
    assert ctrl._frame_count == 0
    assert ctrl._prev_solution is None
```

Note: Full integration tests with FK require MuJoCo models and will be tested via the evaluation commands.

### Step 3: Run tests

```
python -m pytest tests/test_upgrades/test_mpc_control.py -v
python -m pytest tests/ -x -q  # verify no regressions
```

### Step 4: Commit

```
git add tt_sim/upgrades/control.py tests/test_upgrades/test_mpc_control.py
git commit -m "feat: add MPCController with CasADi receding-horizon NLP"
```

---

## Task 3: Registry and run.py wiring

Wire the MPC controller into the CLI and subsystem loading.

**Files:**
- Modify: `tt_sim/registry.py` — add `"mpc"` entry
- Modify: `run.py` — build FK dict and pass to MPCController in `load_subsystems()`

### Step 1: Add registry entry

In `tt_sim/registry.py`, add to the `"control"` dict:

```python
"mpc": "tt_sim.upgrades.control.MPCController",
```

### Step 2: Wire in run.py

In `load_subsystems()`, after the swing planner setup and before building `ctrl_kwargs`, add:

```python
from tt_sim.upgrades.control import MPCController
if issubclass(ControlCls, MPCController) and env is not None:
    import mujoco
    from tt_sim.upgrades.mpc_fk import build_fk_casadi
    model = env.unwrapped.model
    fk_dict = build_fk_casadi(model, n_joints if 'n_joints' in dir() else model.nu)
    ctrl_kwargs["fk_dict"] = fk_dict
```

This goes in the block that builds `ctrl_kwargs`, after the ReactiveController check. The MPC controller also needs `robot_x`, which is already handled by the existing `else` branch.

### Step 3: Test CLI

```bash
python run.py --robot=fanuc --control=mpc --dry-run
python run.py --list  # verify "mpc" appears
```

### Step 4: Commit

```
git add tt_sim/registry.py run.py
git commit -m "feat: wire MPC controller into registry and CLI"
```

---

## Task 4: Integration testing and tuning

Run full evaluation on both robots and tune MPC parameters.

**Files:**
- No new files — parameter tuning in existing `MPCController`

### Step 1: Fanuc evaluation (target: ≥45% contact)

```bash
python run.py --robot=fanuc --predictor=drag_bounce --aimer=specular --swing=quintic --control=mpc --episodes=40
```

Compare to Fanuc open_loop baseline: 45% contact, 10% landing.

If MPC is worse, debug:
1. Check if NLP converges: add temporary `print(sol['f'])` in `_solve()`
2. Check FK accuracy: verify `p_ee` at q_current matches MuJoCo
3. Tune weights: increase `w_pos` if not reaching target, decrease `w_smooth` if too conservative
4. Increase `max_iter` if solver not converging (check `sol['stats']['return_status']`)
5. Check warm-start: does `self._prev_solution` stay continuous?

### Step 2: WAM evaluation (target: ≥20% contact)

```bash
python run.py --robot=wam --predictor=drag_bounce --aimer=specular --swing=quintic --control=mpc --episodes=40
```

### Step 3: Record best video

```bash
python run.py --robot=fanuc --predictor=drag_bounce --aimer=specular --swing=quintic --control=mpc --episodes=10 --record=results/fanuc_mpc.mp4
```

### Step 4: Document results

Add MPC results to README ablation table. Compare against replan and open_loop.

### Step 5: Commit

```
git add tt_sim/upgrades/control.py README.md
git commit -m "feat: tune MPC controller, document results"
```

---

## Task 5: Full test suite verification

### Step 1: Run all tests

```bash
python -m pytest tests/ -x -q
```

All 117+ tests must pass (existing + new MPC tests).

### Step 2: Commit any fixes

```bash
git add -A && git commit -m "fix: address test failures from MPC integration"
```
