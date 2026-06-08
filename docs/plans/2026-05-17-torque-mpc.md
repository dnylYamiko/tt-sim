# Torque-Level MPC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade MPCController from position-level (outputs q_desired → PD converts to torque) to torque-level (outputs torques directly), eliminating the PD bottleneck and 60% lookahead hack. The high-level controller absorbs the low-level controller.

**Architecture:** Build symbolic RNEA (Recursive Newton-Euler) in CasADi from MuJoCo XML to get M(q), C(q,q̇), g(q). Reformulate the NLP with torque decision variables and dynamics constraints. The controller's `step()` returns torques in ctrl-space (÷gear), bypassing the PD loop in `run.py`.

**Tech Stack:** CasADi 3.7.2, MuJoCo, NumPy, Python 3.14

---

### Task 1: Symbolic RNEA — `mpc_dynamics.py`

**Files:**
- Create: `tt_sim/upgrades/mpc_dynamics.py`
- Reference: `tt_sim/upgrades/mpc_fk.py` (kinematic chain extraction pattern)
- Test: `tests/test_upgrades/test_mpc_dynamics.py`

**Step 1: Write failing tests**

```python
# tests/test_upgrades/test_mpc_dynamics.py
"""Tests for symbolic RNEA dynamics — validates against MuJoCo."""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")
mujoco = pytest.importorskip("mujoco")

from tt_sim.upgrades.mpc_dynamics import build_dynamics_casadi


def _load_model(robot: str):
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


def _mujoco_dynamics(model, q, qd, qdd, ndof):
    """Ground truth inverse dynamics from MuJoCo: tau = M*qdd + C*qd + g."""
    d = mujoco.MjData(model)
    d.qpos[:ndof] = q
    d.qvel[:ndof] = qd
    d.qacc[:ndof] = qdd
    mujoco.mj_inverse(model, d)
    return d.qfrc_inverse[:ndof].copy()


def _mujoco_gravity(model, q, ndof):
    """Gravity vector from MuJoCo (inverse dynamics with qd=0, qdd=0)."""
    return _mujoco_dynamics(model, q, np.zeros(ndof), np.zeros(ndof), ndof)


def _mujoco_mass_matrix(model, q, ndof):
    """Mass matrix from MuJoCo."""
    d = mujoco.MjData(model)
    d.qpos[:ndof] = q
    mujoco.mj_forward(model, d)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, d.qM)
    return M[:ndof, :ndof].copy()


class TestGravityVector:
    """Symbolic gravity vector matches MuJoCo."""

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_gravity_at_home(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        g_sym = np.array(dyn['gravity_fn'](q)).flatten()
        g_mj = _mujoco_gravity(model, q, ndof)
        np.testing.assert_allclose(g_sym, g_mj, atol=0.5,
            err_msg=f"{robot} gravity mismatch at home")

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_gravity_at_random(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        rng = np.random.default_rng(42)
        for _ in range(3):
            q = rng.uniform(-1, 1, ndof)
            g_sym = np.array(dyn['gravity_fn'](q)).flatten()
            g_mj = _mujoco_gravity(model, q, ndof)
            np.testing.assert_allclose(g_sym, g_mj, atol=0.5,
                err_msg=f"{robot} gravity mismatch at q={q.round(3)}")


class TestMassMatrix:
    """Symbolic mass matrix matches MuJoCo."""

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_mass_matrix_at_home(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        M_sym = np.array(dyn['mass_matrix_fn'](q))
        M_mj = _mujoco_mass_matrix(model, q, ndof)
        # Mass matrix should be symmetric positive definite
        assert M_sym.shape == (ndof, ndof)
        np.testing.assert_allclose(M_sym, M_sym.T, atol=1e-10,
            err_msg="Mass matrix not symmetric")
        eigvals = np.linalg.eigvalsh(M_sym)
        assert np.all(eigvals > 0), f"Mass matrix not positive definite: {eigvals}"
        np.testing.assert_allclose(M_sym, M_mj, atol=1.0,
            err_msg=f"{robot} mass matrix mismatch at home")

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_mass_matrix_at_random(self, robot):
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        rng = np.random.default_rng(42)
        for _ in range(3):
            q = rng.uniform(-1, 1, ndof)
            M_sym = np.array(dyn['mass_matrix_fn'](q))
            M_mj = _mujoco_mass_matrix(model, q, ndof)
            np.testing.assert_allclose(M_sym, M_mj, atol=1.0,
                err_msg=f"{robot} mass matrix mismatch at q={q.round(3)}")


class TestInverseDynamics:
    """Full inverse dynamics tau = RNEA(q, qd, qdd) matches MuJoCo."""

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_id_at_home_zero_motion(self, robot):
        """At home with zero velocity/acceleration, ID should equal gravity."""
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        qd = np.zeros(ndof)
        qdd = np.zeros(ndof)
        tau_sym = np.array(dyn['rnea_fn'](q, qd, qdd)).flatten()
        tau_mj = _mujoco_dynamics(model, q, qd, qdd, ndof)
        np.testing.assert_allclose(tau_sym, tau_mj, atol=0.5,
            err_msg=f"{robot} ID mismatch at rest")

    @pytest.mark.parametrize("robot", ["fanuc"])
    def test_id_with_motion(self, robot):
        """Full ID with nonzero qd, qdd."""
        model, ndof = _load_model(robot)
        dyn = build_dynamics_casadi(model, ndof)
        rng = np.random.default_rng(123)
        for _ in range(3):
            q = rng.uniform(-1, 1, ndof)
            qd = rng.uniform(-2, 2, ndof)
            qdd = rng.uniform(-5, 5, ndof)
            tau_sym = np.array(dyn['rnea_fn'](q, qd, qdd)).flatten()
            tau_mj = _mujoco_dynamics(model, q, qd, qdd, ndof)
            np.testing.assert_allclose(tau_sym, tau_mj, atol=2.0,
                err_msg=f"{robot} ID mismatch with motion")


class TestDifferentiability:
    """RNEA functions have valid CasADi derivatives."""

    def test_rnea_jacobian(self):
        model, ndof = _load_model("fanuc")
        dyn = build_dynamics_casadi(model, ndof)
        q = np.zeros(ndof)
        qd = np.zeros(ndof)
        qdd = np.zeros(ndof)
        # Should not raise
        J = dyn['rnea_fn'].jacobian()
        result = J(q, qd, qdd, np.zeros(ndof))
        assert np.array(result).shape[1] == 3 * ndof  # d(tau)/d(q,qd,qdd)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_upgrades/test_mpc_dynamics.py -v`
Expected: FAIL with ImportError (module doesn't exist yet)

**Step 3: Implement `mpc_dynamics.py`**

Build symbolic RNEA from MuJoCo model parameters. The algorithm:

1. **Extract from MuJoCo model:** For each body in the kinematic chain (same traversal as `mpc_fk.py`), extract: `body_mass`, `body_ipos` (CoM in body frame), `body_inertia` (diagonal principal inertia), `body_iquat` (principal frame orientation), `dof_damping`, `model.opt.gravity`.

2. **Forward pass** (base to tip): For each link i with joint angle q_i, compute:
   - `v_i` = spatial velocity of link i (6D: [omega; v_linear])
   - `a_i` = spatial acceleration of link i
   Using the recursive formulas with the joint axis and parent transforms.

3. **Backward pass** (tip to base): For each link i, compute:
   - `f_i` = spatial force on link i (from link inertia, velocities, child forces)
   - `tau_i` = projection of f_i onto joint axis

4. **Derived functions:**
   - `gravity_fn(q)` = rnea(q, 0, 0)
   - `mass_matrix_fn(q)`: call rnea(q, 0, e_i) for each unit vector e_i, subtract gravity. Returns (ndof, ndof) matrix.
   - `coriolis_fn(q, qd)` = rnea(q, qd, 0) - gravity_fn(q)

Return dict:
```python
{
    'rnea_fn': ca.Function,      # (q, qd, qdd) -> tau
    'gravity_fn': ca.Function,   # (q,) -> tau_gravity
    'mass_matrix_fn': ca.Function,  # (q,) -> M (ndof x ndof)
    'coriolis_fn': ca.Function,  # (q, qd) -> C(q,qd)*qd
    'n_joints': int,
    'joint_damping': np.ndarray,  # (ndof,) damping coefficients
    'gear_ratios': np.ndarray,    # (ndof,) actuator gear ratios
}
```

**Implementation note:** Use 3x3 rotation matrices (not spatial algebra 6x6) for clarity. Compute angular and linear quantities separately. This avoids the complexity of Plücker coordinates while CasADi handles the symbolic differentiation.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_upgrades/test_mpc_dynamics.py -v`
Expected: All PASS (tolerances are generous: 0.5 Nm for gravity, 1.0 for mass matrix, 2.0 for full ID — these account for potential differences in how MuJoCo handles composite bodies vs our body-by-body RNEA)

**Step 5: Commit**

```bash
git add tt_sim/upgrades/mpc_dynamics.py tests/test_upgrades/test_mpc_dynamics.py
git commit -m "feat: symbolic RNEA dynamics builder for torque-level MPC"
```

---

### Task 2: Torque-Level MPC Controller — `TorqueMPCController`

**Files:**
- Modify: `tt_sim/upgrades/control.py` (add new class after MPCController)
- Test: `tests/test_upgrades/test_mpc_control.py` (add tests)

**Step 1: Write failing tests**

Add to `tests/test_upgrades/test_mpc_control.py`:

```python
from tt_sim.upgrades.control import TorqueMPCController


def test_torque_mpc_import():
    assert TorqueMPCController is not None


def test_torque_mpc_subclass():
    assert issubclass(TorqueMPCController, HighLevelController)


def test_torque_mpc_registry():
    cls = load("control", "torque_mpc")
    assert cls is TorqueMPCController


def test_torque_mpc_holds_without_dynamics():
    """Without dynamics dict, controller holds zero torque."""
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    q = np.array([0.5, 0.5])
    qd = np.array([0.0, 0.0])
    result = ctrl.step({}, q, qd)
    np.testing.assert_array_equal(result, np.zeros(2))


def test_torque_mpc_returns_ctrl_space():
    """Output is in ctrl-space (divided by gear), clipped to [-1, 1]."""
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    q = np.array([0.5, 0.5])
    qd = np.array([0.0, 0.0])
    result = ctrl.step({}, q, qd)
    assert np.all(np.abs(result) <= 1.0)


def test_torque_mpc_torque_mode_flag():
    """Controller has torque_mode = True."""
    ctrl = TorqueMPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), dynamics_dict=None,
    )
    assert ctrl.torque_mode is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_upgrades/test_mpc_control.py -v -k torque`
Expected: FAIL with ImportError

**Step 3: Implement `TorqueMPCController`**

Key differences from `MPCController`:

1. **`__init__`** takes `dynamics_dict` (from `build_dynamics_casadi`) instead of `fk_dict`. The dynamics dict includes FK functions plus RNEA.

2. **`step(env_state, q_current, qd_current)`** — note the extra `qd_current` argument. This is the interface change: the controller needs velocities to apply dynamics.

3. **NLP formulation:**
   - Decision vars: `Q` (positions, N×nj), `QD` (velocities, N×nj), `TAU` (torques, N×nj)
   - Dynamics constraint at each knot:
     ```
     qdd_k = (qd_{k+1} - qd_k) / dt
     M(q_k) * qdd_k + C(q_k, qd_k) + g(q_k) + damping * qd_k = tau_k
     ```
     Rearranged: `tau_k - M(q_k) * qdd_k - coriolis(q_k, qd_k) - gravity(q_k) - damping * qd_k = 0`
   - Position integration: `q_{k+1} = q_k + dt * qd_{k+1}` (implicit Euler)
   - Bounds: `|tau_i| ≤ gear_i`, joint limits on q, velocity limits on qd

4. **Cost:** Same terminal position/orientation/velocity tracking as current MPC, plus running torque smoothness and effort.

5. **Output:** `tau_applied / gear` → ctrl-space vector, clipped to [-1, 1]. No PD loop.

6. **`torque_mode = True`** flag for `run.py` to detect.

**Step 4: Run tests**

Run: `pytest tests/test_upgrades/test_mpc_control.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tt_sim/upgrades/control.py tests/test_upgrades/test_mpc_control.py
git commit -m "feat: torque-level MPC controller with dynamics constraints"
```

---

### Task 3: Registry + `run.py` wiring

**Files:**
- Modify: `tt_sim/registry.py:36` (add `"torque_mpc"` entry)
- Modify: `run.py:326-364` (add torque_mode bypass for PD loop)
- Modify: `run.py:181-187` (wire dynamics_dict for TorqueMPCController)
- Modify: `run.py:43-62` (extract `qd_current` from obs)
- Modify: `run.py:344` (pass `qd_current` to step for torque controllers)

**Step 1: Registry entry**

Add to `REGISTRY["control"]`:
```python
"torque_mpc": "tt_sim.upgrades.control.TorqueMPCController",
```

**Step 2: Modify `extract_env_state` to also return `qd_current`**

```python
def extract_env_state(obs, ndof, step_count, dt=0.008):
    q_current = obs[0:ndof].copy()
    qd_current = obs[7:7 + ndof].copy()
    ball_pos = obs[14:17].copy()
    env_state = {
        "ball_pos": np.asarray(ball_pos, dtype=np.float64),
        "time": float(step_count * dt),
    }
    return env_state, np.asarray(q_current, dtype=np.float64), np.asarray(qd_current, dtype=np.float64)
```

**Step 3: Modify `load_subsystems` to wire dynamics_dict**

After the existing MPC FK wiring block:
```python
from tt_sim.upgrades.control import TorqueMPCController
if issubclass(ControlCls, TorqueMPCController) and env is not None:
    from tt_sim.upgrades.mpc_dynamics import build_dynamics_casadi
    model = env.unwrapped.model
    dynamics_dict = build_dynamics_casadi(model, n_joints)
    ctrl_kwargs["dynamics_dict"] = dynamics_dict
```

**Step 4: Modify main loop to handle torque_mode**

```python
env_state, q_current, qd_current = extract_env_state(obs, ndof, step_count, dt)

# Check if controller operates in torque mode
if getattr(controller, 'torque_mode', False):
    action_raw = controller.step(env_state, q_current, qd_current)
else:
    q_desired = controller.step(env_state, q_current)
    q_vel = obs[7:7 + ndof]
    action_raw = kp * (q_desired - q_current) - kd * q_vel
    if use_gravity_comp:
        qfrc_bias = env.unwrapped.data.qfrc_bias[:ndof].copy()
        action_raw = action_raw + qfrc_bias / gear

# Pad and clip (same as before)
```

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 130+ tests PASS (existing tests unaffected, new tests pass)

**Step 6: Commit**

```bash
git add tt_sim/registry.py run.py
git commit -m "feat: wire torque MPC into run.py, bypassing PD loop"
```

---

### Task 4: Integration test — 40 episode eval on Fanuc

**Step 1: Run eval**

```bash
python run.py --robot fanuc --control torque_mpc --predictor drag_bounce --aimer specular --swing quintic --episodes 40
```

**Step 2: Compare against position-level MPC baseline**

Previous best (position-level MPC): 37.5% contact, 2.5% landing
Target: ≥ 37.5% contact (torque-level should be at least as good)

**Step 3: Tune if needed**

Likely tuning parameters:
- `w_pos`, `w_ori`, `w_vel` (terminal tracking weights)
- `w_smooth`, `w_effort` (running costs)
- `horizon_N`, `max_iter` (solver params)
- Joint damping compensation accuracy

**Step 4: Document results in README upgrade log**

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: torque-level MPC eval results for Fanuc"
```

---

### Task 5: Full test suite verification

**Step 1:** Run `pytest tests/ -v` — all tests pass
**Step 2:** Run `python run.py --list` — verify `torque_mpc` appears
**Step 3:** Run `python run.py --dry-run --control torque_mpc --robot fanuc` — verify subsystems load

**Step 4: Final commit and push**

```bash
git add -A
git commit -m "feat: Stage 2b torque-level MPC — RNEA dynamics, direct torque control"
gh auth switch --user Farjad1
git config --global --unset http.proxy; git config --global --unset https.proxy
git push farjad1 main
```
