<<<<<<< HEAD
<h1 align="center">
  <br>
  <img src='https://raw.githubusercontent.com/ALRhub/fancy_gym/master/icon.svg' width="250px">
  <br><br>
  <b>Fancy Gym</b>
  <br><br>
</h1>

Built upon the foundation of [Gymnasium](https://gymnasium.farama.org) (a maintained fork of OpenAI’s renowned Gym library) `fancy_gym` offers a comprehensive collection of reinforcement learning environments.

**Key Features**:

- **New Challenging Environments**: `fancy_gym` includes several new environments ([Panda Box Pushing](https://alrhub.github.io/fancy_gym/envs/fancy/mujoco.html#box-pushing), [Table Tennis](https://alrhub.github.io/fancy_gym/envs/fancy/mujoco.html#table-tennis), [etc.](https://alrhub.github.io/fancy_gym/envs/fancy/index.html)) that present a higher degree of difficulty, pushing the boundaries of reinforcement learning research.
- **Support for Movement Primitives**: `fancy_gym` supports a range of movement primitives (MPs), including Dynamic Movement Primitives (DMPs), Probabilistic Movement Primitives (ProMP), and Probabilistic Dynamic Movement Primitives (ProDMP).
- **Upgrade to Movement Primitives**: With our framework, it’s straightforward to transform standard Gymnasium environments into environments that support movement primitives.
- **Benchmark Suite Compatibility**: `fancy_gym` makes it easy to access renowned benchmark suites such as [DeepMind Control](https://alrhub.github.io/fancy_gym/envs/dmc.html)
  and [Metaworld](https://alrhub.github.io/fancy_gym/envs/meta.html), whether you want to use them in the regular step-based setting or using MPs.
- **Contribute Your Own Environments**: If you’re inspired to create custom gym environments, both step-based and with movement primitives, this [guide](https://alrhub.github.io/fancy_gym/guide/upgrading_envs.html) will assist you. We encourage and highly appreciate submissions via PRs to integrate these environments into `fancy_gym`.

## Quickstart Guide

| &#x26A0; We recommend installing `fancy_gym` into a virtual environment as provided by [venv](https://docs.python.org/3/library/venv.html), [Poetry](https://python-poetry.org/) or [Conda](https://docs.conda.io/en/latest/). |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

Install via pip [or use an alternative installation method](https://alrhub.github.io/fancy_gym/guide/installation.html)

```bash
    pip install 'fancy_gym[all]'
```

Try out one of our step-based environments [or explore our other envs](https://alrhub.github.io/fancy_gym/envs/fancy/index.html)

```python
   import gymnasium as gym
   import fancy_gym
   import time

   env = gym.make('fancy/BoxPushingDense-v0', render_mode='human')
   observation = env.reset()
   env.render()

   for i in range(1000):
      action = env.action_space.sample() # Randomly sample an action
      observation, reward, terminated, truncated, info = env.step(action)
      time.sleep(1/env.metadata['render_fps'])

      if terminated or truncated:
            observation, info = env.reset()
```

Explore the MP-based variant [or learn more about Movement Primitives (MPs)](https://alrhub.github.io/fancy_gym/guide/episodic_rl.html)

```python
   import gymnasium as gym
   import fancy_gym

   env = gym.make('fancy_ProMP/BoxPushingDense-v0', render_mode='human')
   env.reset()
   env.render()

   for i in range(10):
      action = env.action_space.sample() # Randomly sample MP parameters
      observation, reward, terminated, truncated, info = env.step(action) # Will execute full trajectory, based on MP
      observation = env.reset()
```

## Documentation

Documentation for `fancy_gym` can be found [here](https://alrhub.github.io/fancy_gym/); Usage Examples can be found [here](https://alrhub.github.io/fancy_gym/examples/general.html).

## Citing the Project

To cite this repository in publications:

```bibtex
@software{fancy_gym,
	title = {Fancy Gym},
	author = {Otto, Fabian and Celik, Onur and Roth, Dominik and Zhou, Hongyi},
	abstract = {Fancy Gym: Unifying interface for various RL benchmarks with support for Black Box approaches.},
	url = {https://github.com/ALRhub/fancy_gym},
	organization = {Autonomous Learning Robots Lab (ALR) at KIT},
}
```

## Icon Attribution

The icon is based on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) icon as can be found [here](https://gymnasium.farama.org/_static/img/gymnasium_black.svg).
=======
# tt-sim

Modular table tennis robot simulation for ME 595. Plug-and-play subsystems (perception, prediction, spin, aiming, swing, control) where each can be independently upgraded from analytic baselines to data-driven methods.

**Team**: Pramit Chalise, Daniel Musundire, Farjad Baig

## Quick Start

```bash
# No install needed for base sim — just run from the repo root
python run.py --robot=fanuc --control=reactive --episodes=20 --render

# Record a video
python run.py --robot=fanuc --control=reactive --episodes=10 --record=results/demo.mp4

# List all available subsystem implementations
python run.py --list

# Dry-run (load subsystems, print types, don't run sim)
python run.py --dry-run
```

### Dependencies

```bash
pip install mujoco numpy scipy click imageio opencv-python
pip install --no-deps fancy_gym
pip install "mp-pytorch<=0.1.3" "gymnasium==0.29.1"
```

> On Apple Silicon, `fancy_gym[mujoco]` won't build. Install `mujoco>=3.8.0` separately, then `pip install --no-deps fancy_gym`.

## Development Workflow

The project is designed so each team member can independently upgrade one subsystem at a time, measure it against the eval harness, and iterate.

### 1. Pick a Subsystem

Each subsystem has a baseline (analytic, works out of the box) and one or more upgrade stubs (data-driven, need training). Choose one:

| Subsystem | Baseline | What to Upgrade | Difficulty |
|-----------|----------|----------------|------------|
| **Predictor** | `ballistic` (constant velocity) | Add drag, bounce, spin physics or learn from data | Medium |
| **Spin** | `zero` (ignores spin) | Estimate spin from trajectory curvature | Medium |
| **Aimer** | `face_net` (aim toward net center) | Angle-of-incidence aiming for target placement | Easy |
| **Swing** | `lerp` (linear interp) | Min-jerk quintic trajectory for smooth swings | Easy |
| **Perceiver** | `sim` (ground truth from sim) | HSV color tracking from camera frames | Hard |
| **Control** | `reactive` (Jacobian tracking) | Continuous replanning with prediction updates | Hard |

### 2. Establish a Baseline

Run the current system and log results to compare against later:

```bash
# Baseline: all defaults
python run.py --robot=fanuc --control=reactive --episodes=50 \
  --log=results/baseline.csv --record=results/baseline.mp4

# The eval logger prints contact rate, landing rate, mean reward
```

### 3. Get Data

Some upgrades need training data. Download external datasets or generate sim data:

```bash
# Download external datasets
python data/scripts/download_kienzle_50k.py    # 50k ball trajectories
python data/scripts/download_kienzle_120k.py   # 120k ball trajectories
python data/scripts/download_aimy.py           # AIMy robot dataset
python data/scripts/download_roboflow_tt.py    # Table tennis detection images
python data/scripts/download_blurball.py       # Motion blur ball images

# Generate sim trajectories (for drag_bounce, etc.)
python -m tt_sim.generate --type trajectories --episodes=1000 --output=data/sim_trajectories/
```

### 4. Train Your Upgrade

Each upgrade has a training script in `tt_sim/upgrades/train/`. These are standalone CLIs:

```bash
# Prediction upgrades
python -m tt_sim.upgrades.train.train_drag_bounce   --data-dir data/sim_trajectories/
python -m tt_sim.upgrades.train.train_residual_mlp  --data data/kienzle_50k.npz
python -m tt_sim.upgrades.train.train_neural_ode    --data data/kienzle_120k.npz
python -m tt_sim.upgrades.train.train_gp            --data data/kienzle_50k.npz

# Spin upgrade
python -m tt_sim.upgrades.train.train_magnus_spin   --data data/kienzle_50k_spin.npz
```

Trained weights are saved to `models/<subsystem>/<name>.pt` (gitignored — regenerate by re-running the train script).

### 5. Plug In and Evaluate

Swap in your upgrade with a CLI flag and compare against the baseline:

```bash
# Test your upgrade
python run.py --robot=fanuc --control=reactive --predictor=drag_bounce \
  --episodes=50 --log=results/drag_bounce.csv --record=results/drag_bounce.mp4

# Compare: look at contact rate, landing rate, mean reward vs baseline
# Results are in the CSV files and printed as summary stats
```

### 6. Iterate

The upgrade stubs in `tt_sim/upgrades/` have full docstrings describing the expected interface, training data format, and model loading convention. The development loop is:

1. **Read the stub** — each upgrade class has a detailed docstring explaining what to implement
2. **Implement** — fill in the stub methods, following the ABC interface from `tt_sim/interfaces.py`
3. **Test** — run `python -m pytest tests/test_upgrades/` to verify your implementation matches the interface
4. **Train** — run the corresponding training script to produce model weights
5. **Evaluate** — plug in via CLI flag and compare CSV results against the baseline
6. **Repeat** — tune hyperparameters, add more data, refine the model

Only change one subsystem at a time so you can isolate the effect of each upgrade.

## Robots

### Fanuc CRX-25iA (`--robot=fanuc`)

6-DOF collaborative robot. Kinematics built from real controller DCS data (serial 6064500003), FK-validated to **0.5mm** accuracy against the real robot.

- Floor-mounted on 0.6m pedestal behind table
- Torque actuators with gravity compensation (`qfrc_bias` feedforward)
- L-shaped arm geometry: vertical upper arm (0.95m) + horizontal forearm (0.73m) with green spherical joint housings
- URDF-accurate masses and inertias (total ~88kg)
- Racquet mounted 3 inches (76.2mm) normal to J6 flange
- Tool frame defined at racquet center (`fanuc/tool_frame` site)

**Kinematic parameters:**

| Parameter | Value |
|-----------|-------|
| J1 height | 405mm |
| Upper arm (J2→J3) | 950mm along Z, -183mm Y-offset |
| Forearm (J4→J5) | 730mm along X |
| Wrist (J5→J6) | 140mm along X |
| J6→racquet | 76.2mm (3 in) along tool Z |
| Joint axes | Z, Y, -Y, -X, -Y, -X |

### Barrett WAM 7-DOF (`--robot=wam`)

Floor-mounted 2x-scale WAM from fancy_gym, modified with custom gear ratios. Use `--original-scene` to revert to the default ceiling-mounted WAM.

## Architecture

```
run.py                      CLI entry point
tt_sim/
  interfaces.py             5 dataclasses + 6 ABCs
  registry.py               18 registered implementations
  eval.py                   EpisodeMetrics + EvalLogger (CSV + summary)
  generate.py               Data generation CLI
  baselines/                Built-in analytic implementations
    perception.py           SimPerceiver
    prediction.py           BallisticPredictor
    spin.py                 ZeroSpinEstimator
    aiming.py               FaceNetAimer
    swing.py                LerpSwingPlanner, MujocoIKSwingPlanner
    control.py              OpenLoopController, ReactiveController
  upgrades/                 Data-driven upgrades (stubs)
    perception.py           HSVPerceiver
    prediction.py           DragBounce, ResidualMLP, NeuralODE, GP, Ensemble
    spin.py                 MagnusSpinEstimator
    aiming.py               SpecularAimer
    swing.py                QuinticSwingPlanner
    control.py              ReplanController
    train/                  Training scripts
assets/
  xml/                      MuJoCo scene files
    table_tennis_fanuc.xml  Fanuc scene
    table_tennis_env.xml    WAM scene
    include_fanuc_crx25ia.xml   Robot body (L-shape, URDF masses)
    include_fanuc_actuator.xml  Torque actuators
  fanuc_crx25ia/
    crx25ia.urdf            FK-validated URDF
    validate_fk.py          FK validation script
data/scripts/               Dataset download scripts
models/                     Trained model weights (gitignored)
tests/                      88 unit tests
results/                    Videos and evaluation logs
```

## CLI Reference

```
python run.py [OPTIONS]

Options:
  --perceiver TEXT    Perception module       [default: sim]
  --predictor TEXT    Prediction module       [default: ballistic]
  --spin TEXT         Spin estimation module  [default: zero]
  --aimer TEXT        Aiming module           [default: face_net]
  --swing TEXT        Swing generation module [default: lerp]
  --control TEXT      Control module          [default: open_loop]
  --robot [wam|fanuc] Robot platform          [default: wam]
  --episodes INT      Number of episodes      [default: 10]
  --log TEXT          Log CSV file path
  --list              List available implementations and exit
  --dry-run           Load subsystems without running sim
  --render            Live viewer window (requires opencv)
  --record TEXT       Save mp4 video to path
  --original-scene    Use default fancy_gym scene (ceiling WAM)
```

### Examples

```bash
# Fanuc reactive controller, 20 episodes, with video
python run.py --robot=fanuc --control=reactive --episodes=20 --record=results/fanuc.mp4

# WAM with ballistic predictor + open-loop control, log results
python run.py --robot=wam --control=open_loop --predictor=ballistic --log=results/wam_openloop.csv

# Mix and match subsystems
python run.py --robot=fanuc --control=reactive --predictor=drag_bounce --swing=quintic
```

## Subsystem Implementations

| Subsystem | Baseline | Upgrades |
|-----------|----------|----------|
| Perceiver | `sim` — direct sim state | `hsv` — HSV color tracking |
| Predictor | `ballistic` — constant velocity | `drag_bounce`, `residual_mlp`, `neural_ode`, `gp`, `ensemble` |
| Spin | `zero` — no spin | `magnus` — Magnus effect estimation |
| Aimer | `face_net` — face ball toward net | `specular` — angle-of-incidence aiming |
| Swing | `lerp` — linear interpolation | `quintic` — min-jerk trajectory |
| Control | `open_loop` — plan once, execute | `reactive` — Jacobian tracking, `replan` — continuous replanning |

## Baseline Performance

| Robot | Controller | Contact Rate | Notes |
|-------|-----------|-------------|-------|
| WAM (floor) | reactive | ~20-40% | 2x scale, floor-mounted |
| WAM (ceiling) | reactive | ~46-70% | Original fancy_gym scene |
| Fanuc CRX-25iA | reactive | ~10-40% | Gravity compensated, table collision |

## Ablation Study: Cumulative Upgrade Impact

40 episodes per configuration. Upgrades are added cumulatively; the **quintic swing planner is the foundation** — predictor and aimer upgrades have zero effect without it (see note below).

### WAM 7-DOF

| # | Swing | Predictor | Aimer | Control | Contact | Landing | Reward | Delta |
|:-:|:---|:---|:---|:---|:-:|:-:|:-:|:---|
| 0 | lerp | ballistic | face_net | open_loop | 7.5% | 0.0% | 0.314 | — baseline |
| 1 | **quintic** | ballistic | face_net | open_loop | 0.0% | 0.0% | 0.142 | −7.5% (open-loop quintic overshoots) |
| 2 | quintic | **drag_bounce** | face_net | open_loop | 2.5% | 0.0% | 0.182 | +2.5% |
| 3 | quintic | drag_bounce | **specular** | open_loop | 2.5% | 0.0% | 0.205 | ±0 contact, +0.02 reward |
| 4 | quintic | drag_bounce | specular | **replan** | **20.0%** | 0.0% | **0.514** | **+17.5%** contact |

### Fanuc CRX-25iA 6-DOF

| # | Swing | Predictor | Aimer | Control | Contact | Landing | Reward | Delta |
|:-:|:---|:---|:---|:---|:-:|:-:|:-:|:---|
| 0 | lerp | ballistic | face_net | open_loop | 0.0% | 0.0% | 0.068 | — baseline |
| 1 | **quintic** | ballistic | face_net | open_loop | 0.0% | 0.0% | 0.168 | +0.10 reward, no contact |
| 2 | quintic | **drag_bounce** | face_net | open_loop | **42.5%** | 2.5% | **1.108** | **+42.5%** contact |
| 3 | quintic | drag_bounce | **specular** | open_loop | **45.0%** | **10.0%** | **1.499** | +2.5% contact, **+7.5% landing** |
| 4 | quintic | drag_bounce | specular | **replan** | 17.5% | 0.0% | 0.605 | −27.5% (regression) |

### Analysis

**Quintic swing is the prerequisite upgrade.** Predictor and aimer upgrades have zero effect with the baseline lerp swing planner — lerp produces linear joint interpolations that can't reach *any* target in time, so improving target accuracy is meaningless. With quintic (C2-continuous trajectories, proper IK, nonzero endpoint velocity), the arm can actually execute the planned motion, and upstream accuracy starts to matter.

**The predictor (drag_bounce) is the highest-impact accuracy upgrade.** The ball always bounces on the table before reaching the robot (bounce at t≈0.5s, intercept at t≈0.6-0.7s). BallisticPredictor has no bounce model — it predicts the ball at z=0.06m (below table, clipped to z=0.76) with velocity 5.8 m/s downward. DragBouncePredictor correctly predicts the post-bounce trajectory at z≈1.22m with velocity 2.7 m/s upward. The 0.46m vertical error and flipped velocity sign make ballistic predictions useless. On Fanuc this produces the largest single improvement: 0% → 42.5%.

**The aimer (specular) improves landing accuracy, not contact rate.** SpecularAimer computes a geometrically correct paddle normal for redirecting the ball toward the opponent's half, plus follow-through velocity. FaceNetAimer just points at the net with zero velocity. On Fanuc, adding specular raises landing rate from 2.5% → 10.0% with only marginal contact improvement (+2.5%).

**Replan control is robot-dependent.** WAM (7-DOF, 14 kg) benefits greatly from replanning (+17.5% contact) — its light arm can adapt mid-swing. Fanuc (6-DOF, 118 kg) is hurt by replanning (−27.5%) — the 6-DOF IK produces discontinuous solution jumps between replans, and the heavy arm can't track the resulting jerky trajectory.

**Best configuration per robot:**
- **Fanuc:** quintic + drag_bounce + specular + open_loop → **45% contact, 10% landing**, 1.499 reward
- **WAM:** quintic + drag_bounce + specular + replan → **20% contact**, 0.514 reward

## Upgrade Log

### Aiming: `face_net` → `specular` (Stage 0 → Stage 1)

**What changed:** Implemented `SpecularAimer` in `tt_sim/upgrades/aiming.py`. Uses the specular (mirror) reflection law to compute the paddle normal that redirects the incoming ball toward a configurable target landing point (default: center of opponent's half). Formula: `n = normalize(v_out - v_in)`, outgoing speed scaled by coefficient of restitution (e=0.91).

**Key design choices:**
- Configurable target point via `__init__(target=...)` for flexibility
- Zero paddle velocity (static contact assumption)
- Graceful fallback for zero-velocity balls and degenerate normals

**Comparison (20 episodes, `open_loop` control, `ballistic` predictor, `lerp` swing):**

| Aimer | Contact Rate | Landing Rate | Mean Reward | Mean Land Dist Err |
|-------|-------------|-------------|-------------|-------------------|
| `face_net` | 0% | 0% | 0.138 | 3.60 m |
| `specular` | 0% | 0% | 0.131 | 3.39 m |

**Analysis:** Neither aimer achieves contact under `open_loop` control — the bottleneck is the upstream predictor (`ballistic`, constant-velocity, no drag) and swing planner (`lerp`, linear interpolation). The arm cannot reach the predicted contact point in time. The specular aimer shows slightly lower landing distance error (3.39m vs 3.60m), suggesting better paddle orientation, but the effect is masked by the swing/prediction limitations.

**With `reactive` control (20 episodes):**

| Aimer | Contact Rate | Mean Reward |
|-------|-------------|-------------|
| `face_net` | ~10-20% | ~0.18 |
| `specular` | ~10-20% | ~0.18 |

Note: `ReactiveController` bypasses the aimer entirely (uses Jacobian tracking to chase ball position), so aimer choice has no effect. The specular aimer's value will be realized when paired with `open_loop` or `replan` control + an upgraded predictor and swing planner.

**Next steps to unlock aimer impact:**
1. ~~Upgrade predictor to `drag_bounce` (Stage 1) — more accurate ball trajectory~~ Done
2. Upgrade swing to `quintic` (Stage 1) — smooth, feasible joint trajectories
3. Re-run comparison with `open_loop` control to isolate aimer effect

### Predictor: `ballistic` → `drag_bounce` (Stage 0 → Stage 1)

**What changed:** Implemented `DragBouncePredictor` in `tt_sim/upgrades/prediction.py`. Integrates the gravity + quadratic aerodynamic drag ODE using `scipy.integrate.solve_ivp` (RK45), with table bounce detection and resolution via coefficient of restitution.

**Algorithm:**
```
dv/dt = g - (C_D * rho_air * A / 2m) * |v| * v
```
- `C_D = 0.4`, `rho_air = 1.225 kg/m^3`, ball area = pi * 0.02^2 m^2, mass = 2.7g
- Bounce at `z = TABLE_HEIGHT (0.76m)`: `v_z *= -0.91` (restitution)
- Up to 5 bounces handled per prediction
- Initial state bootstrapped from observations via least-squares (same as ballistic baseline)

**Key design choices:**
- `max_step=0.01` for integration accuracy near bounce events
- Terminal event with direction=-1 (only triggers when ball is descending through table plane)
- Snap z to TABLE_HEIGHT on bounce to prevent accumulation of floating-point drift

**Comparison (20 episodes, `open_loop` control, `specular` aimer, `lerp` swing):**

| Predictor | Contact Rate | Landing Rate | Mean Reward | Mean Land Dist Err |
|-----------|-------------|-------------|-------------|-------------------|
| `ballistic` | 0% | 0% | 0.127 | 3.93 m |
| `drag_bounce` | 0% | 0% | 0.130 | 3.55 m |

**With `face_net` aimer (20 episodes, `open_loop`, `lerp`):**

| Predictor | Contact Rate | Landing Rate | Mean Reward | Mean Land Dist Err |
|-----------|-------------|-------------|-------------|-------------------|
| `ballistic` | 0% | 0% | 0.138 | 3.60 m |
| `drag_bounce` | 0% | 0% | 0.125 | 3.88 m |

**Analysis:** The drag_bounce predictor shows improved landing distance error with specular aiming (3.55m vs 3.93m), confirming that drag modeling produces more accurate trajectory predictions. However, neither configuration achieves contact — the `lerp` swing planner remains the primary bottleneck. The arm cannot execute smooth, timely trajectories to reach the predicted contact point.

**Next steps:**
1. ~~Upgrade swing to `quintic` (Stage 1) — smooth C2-continuous trajectories with proper boundary conditions~~ Done
2. ~~This is the last Stage 0 component in the open_loop pipeline; upgrading it should unlock the first contacts under open_loop control~~ Confirmed

### Swing: `lerp` → `quintic` (Stage 0 → Stage 1)

**What changed:** Implemented `QuinticSwingPlanner` in `tt_sim/upgrades/swing.py`. Generates C2-continuous joint trajectories using 5th-order polynomials with 6 boundary conditions per joint. Updated `run.py` to wire MuJoCo-based IK and Jacobian functions when the sim env is available.

**Algorithm:** For each joint, solve the 6x6 linear system:
```
q_j(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
```
Boundary conditions: `q(0) = q_current`, `dq(0) = 0`, `ddq(0) = 0` (start at rest); `q(T) = q_target` (from IK), `dq(T) = J^+ * v_paddle` (from Jacobian if available), `ddq(T) = 0`.

**Key design choices:**
- Injectable `ik_fn` and `jac_fn` — testable without MuJoCo, wired to sim at runtime
- Nonzero end velocity supported via Jacobian pseudoinverse of `PaddleTarget.velocity`
- DOF mismatch handling (IK output truncated/padded to match `q_current`)
- Damped least-squares for both IK (100 iterations, 5mm convergence) and Jacobian inversion

**Comparison (20 episodes, `open_loop` control, all with `specular` aimer):**

| Predictor | Swing | Contact Rate | Mean Reward | Mean Land Dist Err |
|-----------|-------|-------------|-------------|-------------------|
| `ballistic` | `lerp` | 0% | 0.127 | 3.93 m |
| `ballistic` | `quintic` | 0% | 0.148 | 3.25 m |
| `drag_bounce` | `lerp` | 0% | 0.130 | 3.55 m |
| `drag_bounce` | `quintic` | 0% | 0.151 | 3.12 m |

**With `face_net` aimer + `drag_bounce` + `quintic` (20 episodes):**

| Config | Contact Rate | Mean Reward | Mean Land Dist Err |
|--------|-------------|-------------|-------------------|
| `drag_bounce` + `face_net` + `quintic` | **5%** | **0.256** | 3.07 m |

**Analysis:** The quintic planner consistently improves mean reward and landing distance error across all configurations. The combination of `drag_bounce` + `quintic` + `face_net` achieved the **first contact under `open_loop` control** (5% contact rate, 1 HIT in 20 episodes). The quintic's smooth acceleration profile allows the arm to reach the contact point more reliably than LERP's constant-velocity interpolation.

The specular aimer configurations didn't achieve contact in this run, but the improved trajectory quality (lower landing distance errors) suggests contacts are close — likely dependent on favorable ball trajectories in stochastic episodes.

**Next steps:**
1. Upgrade control to `replan` (Stage 1) — continuous replanning as new observations arrive
2. Increase episode count for more statistically significant comparisons
3. ~~Investigate why `face_net` + quintic outperformed `specular` + quintic — may indicate the specular normal is rotating the paddle away from optimal orientation for the IK solution~~ Resolved — see below

### Pipeline fix: Orientation-aware IK + Follow-through velocity

**Investigation:** The specular aimer's computed `PaddleTarget.normal` was being **completely ignored** by the IK solver. The MuJoCo IK in `run.py` only minimized positional error — the rotational Jacobian was not used (`jacr=None`). Both aimers produced identical joint trajectories because they targeted the same position with zero velocity. Additionally, both aimers set `PaddleTarget.velocity = zeros(3)`, meaning the arm decelerated to a stop at the contact point — no follow-through.

**Fix A — Orientation-aware IK** (`run.py`):
- Extended `_mujoco_ik` to use the full 6D Jacobian (3 position + 3 rotation)
- Added orientation error: `cross(paddle_z, target.normal)` aligns the EE body's Z-axis (paddle face normal) with the desired normal
- Weighted orientation at 0.5x position priority to avoid sacrificing reach for angle
- Convergence criteria: position < 5mm AND orientation error < 0.02 rad
- Increased iterations to 200 (from 100) for the additional constraints

**Fix B — Follow-through velocity** (`tt_sim/upgrades/aiming.py`):
- `SpecularAimer` now computes `PaddleTarget.velocity = normal * 0.3 * |v_in|`
- The paddle moves along its face normal at 30% of incoming ball speed at contact
- This activates the `QuinticSwingPlanner`'s `jac_fn` path, giving the arm swing-through motion
- For a stationary ball, velocity remains zero (graceful fallback)

**Before vs After (20 episodes, `open_loop`, `drag_bounce`, `quintic`):**

| Aimer | Contact (before) | Contact (after) | Reward (before) | Reward (after) |
|-------|-----------------|----------------|----------------|---------------|
| `specular` | 0% | **10%** | 0.151 | **0.365** |
| `face_net` | 5% | 0% | 0.256 | 0.162 |

**Analysis:** The fixes completely reversed the ranking. `specular` now achieves **10% contact rate** (2 HITs, reward 2.297 and 2.429) while `face_net` dropped to 0%. This makes physical sense:
- The specular normal correctly angles the paddle to redirect the ball toward the target
- Follow-through velocity gives the arm momentum through the contact, rather than stopping at the ball
- `face_net` always points toward the net center regardless of incoming ball direction, which is no longer a useful heuristic once the IK respects orientation — the paddle is angled wrong for most incoming trajectories

**Next steps:**
1. Upgrade control to `replan` (Stage 1) — continuous replanning as new observations arrive
2. Increase episode count for statistical significance
3. Tune `FOLLOW_THROUGH` fraction (currently 0.3) and `w_ori` (currently 0.5)

### Analysis: Why 10% Contact Rate Under Open-Loop

Investigation into bottlenecks limiting the best open_loop configuration (`specular` + `drag_bounce` + `quintic`) to 10% contact rate.

**Root causes (ranked by severity):**

| # | Factor | Severity | Details |
|---|--------|----------|---------|
| 1 | Trajectory timing mismatch | CRITICAL | `OpenLoopController` steps through waypoints at one-per-`env.step()` (index-based, not time-based). With 50 waypoints and dt=0.008s, every trajectory takes exactly 0.4s regardless of the quintic's designed `t_contact`. If predicted contact is 0.25s, the arm moves at 62% speed; if 0.6s, the arm outruns the waypoints. |
| 2 | Intercept time underestimate | HIGH | `_estimate_intercept_time()` uses linear `dx/vx`, ignoring drag. Ball decelerates, so real arrival is later than estimated — robot arrives early. |
| 3 | Plan-once, no replanning | HIGH | Only 3 observations (~16ms at dt=0.008s) bootstrap the velocity estimate. Tiny time window gives noisy least-squares fit. After planning once, the controller never updates. |
| 4 | Noisy velocity bootstrap | MEDIUM | 3 points over 16ms — ball moves ~5mm, so numerical precision dominates the velocity fit. |
| 5 | Fixed intercept x-plane | MEDIUM | `ROBOT_X=1.3` regardless of ball y/z position. Some balls may be more reachable at different x. |
| 6 | PD tracking lag | LOW | kp=400, kd=40 with heavy links (30kg upper arm) may have settling time comparable to trajectory duration. |

**Robot is NOT too slow:** No hard velocity limits in the XML. Torque limits (±500Nm J1, ±300Nm J3) with 30kg links allow ~15 rad/s² acceleration — adequate for 0.4s swings. Reach (~1.9m from shoulder) covers most of the robot's table half.

**Proposed fixes (within open_loop, no Stage 1 upgrade):**
1. **Time-based trajectory interpolation** — use `np.interp(t_now, traj.times, traj.positions)` instead of step index. Fixes the critical timing mismatch.
2. **More observations before planning** — change `>= 3` to `>= 8` (~64ms). Much better velocity bootstrap with minimal time cost.
3. **Drag-aware intercept time** — use `DragBouncePredictor`'s trajectory to find the actual x-crossing, not linear `dx/vx`.

### Control fixes: Time-based interp + more obs + drag-aware intercept

**What changed:** Applied all three proposed fixes to `OpenLoopController` in `tt_sim/baselines/control.py`:

1. **Time-based trajectory interpolation:** Replaced index-based stepping (`self._step_index += 1`) with `np.interp(current_time - plan_start_time, traj.times, traj.positions)`. The arm now tracks the quintic polynomial's designed timing exactly — if the trajectory says "be at joint angles X at time 0.15s", the arm is there at 0.15s of elapsed time, not after 19 env steps.

2. **8 observations before planning:** Changed threshold from `>= 3` (~24ms) to `>= 8` (~64ms). More data points give a more stable least-squares velocity fit for the predictor bootstrap.

3. **Drag-aware intercept time:** `_estimate_intercept_time()` now samples the predictor at 250Hz to find the first time `ball.x >= ROBOT_X`, accounting for drag deceleration. Falls back to linear `dx/vx` if the predictor raises an exception or the ball never reaches.

**Comparison (20 episodes, `open_loop`, `specular` aimer):**

| Config | Contact Rate | Mean Reward | Mean Land Dist Err |
|--------|-------------|-------------|-------------------|
| `drag_bounce` + `quintic` (before fixes) | 10% | 0.365 | 3.47 m |
| `drag_bounce` + `quintic` (after fixes) | 5% | 0.255 | 3.69 m |
| `drag_bounce` + `lerp` (after fixes) | 10% | 0.418 | 3.94 m |

**Analysis:** The fixes did not improve contact rate as hoped. The `lerp` swing actually outperforms `quintic` after the timing fix (10% vs 5%), suggesting the quintic polynomial's boundary conditions (zero start acceleration, follow-through end velocity) may not be well-suited to the time-based interpolation — the arm may spend too long accelerating from rest while lerp immediately starts moving. The stochastic variance in 20 episodes is high (5% = 1 HIT difference), so these results are not conclusive.

**Key insight:** The open-loop plan-once architecture has a fundamental ceiling. Even with perfect timing, the controller commits to a trajectory planned from early noisy observations and never corrects. The next major improvement requires **replanning** (`replan` controller, Stage 1) — continuously updating the predicted intercept and re-planning the swing as new observations arrive.

**Next steps:**
1. ~~Upgrade control to `replan` (Stage 1) — continuous replanning with prediction updates~~ Done
2. Run larger episode counts (50-100) for statistical significance
3. ~~Consider tuning quintic boundary conditions (nonzero start velocity, different follow-through fraction)~~ Done (warm-start dq)

### Control: `open_loop` → `replan` (Stage 0 → Stage 1)

**What changed:** Implemented `ReplanController` in `tt_sim/upgrades/control.py`. Instead of planning once and executing open-loop, this controller re-runs the full pipeline (predict → aim → swing) every 4 frames (~31 Hz). Each replan generates a fresh trajectory from the current joint state and returns a position looked-ahead into that trajectory.

**Algorithm:**
- Collect observations every frame, replan every `REPLAN_INTERVAL=4` frames
- Minimum 4 observations before first plan (vs 8 for open_loop — fewer needed since we keep correcting)
- Same drag-aware intercept time estimation as open_loop
- Lookahead: interpolate trajectory at `min(max(2 * replan_dt, 0.3 * t_contact), t_contact)` — commits to a meaningful portion of the swing while still allowing course correction
- Warm-start: estimates current joint velocity `dq = (q - q_prev) / dt` and passes to `QuinticSwingPlanner` as `dq_current`, so the quintic starts from the arm's actual velocity rather than rest

**Key design choices:**
- **Aggressive lookahead** — naive "use first waypoint" or "look ahead by one replan interval" caused the quintic to barely move (zero-acceleration start means tiny displacement in 0.032s). Looking ahead 30% of trajectory duration gives the arm meaningful displacement targets.
- **Velocity warm-start** — without this, each replan creates a quintic that decelerates to zero then re-accelerates, fighting the arm's existing motion. Passing `dq_current` maintains momentum.
- **`try/except TypeError` for `dq_current`** — gracefully falls back to `plan(target, q_current)` for swing planners that don't accept the extra parameter (e.g., `LerpSwingPlanner`).

**Also changed:** `QuinticSwingPlanner.plan()` now accepts optional `dq_current: np.ndarray | None` parameter for initial joint velocity (default: zeros, preserving backward compatibility).

**Comparison (20 episodes, `drag_bounce` predictor, `specular` aimer):**

| Control | Swing | Contact Rate | Mean Reward | Mean Land Dist Err |
|---------|-------|-------------|-------------|-------------------|
| `open_loop` | `quintic` | 5% | 0.255 | 3.69 m |
| `open_loop` | `lerp` | 10% | 0.418 | 3.94 m |
| `replan` | `quintic` | **10%** | **0.376** | 4.57 m |
| `replan` | `lerp` | 5% | 0.188 | 3.47 m |

**Analysis:** The replan controller with quintic swing achieves 10% contact rate, matching the best open_loop result. The replanning correctly adapts the swing target as more observations refine the ball trajectory prediction. Quintic now outperforms lerp under replanning (10% vs 5%) — the warm-start velocity and aggressive lookahead give the quintic meaningful displacement per replan cycle, while lerp's instantaneous jump to the target is less effective when the target keeps moving.

Contact rates remain in the 5-10% range across all configurations. The stochastic variance is high at 20 episodes (1 HIT = 5%), so differences may not be statistically significant. The main bottleneck is likely the prediction accuracy and IK solution quality rather than the control architecture.

**Next steps:**
1. ~~Run larger episode counts (50-100) for statistical significance~~ Done (40 episodes)
2. Tune `REPLAN_INTERVAL` (try 2, 8) and lookahead fraction
3. ~~Investigate whether the IK converges to good solutions — log position/orientation error at contact time~~ Investigated — found 3 bugs below
4. Consider a hybrid: replan until committed, then switch to open-loop execution for the final approach

### Investigation: Three pipeline bugs causing low contact rate

Deep investigation into why all configurations plateaued at 5-10% contact. Found three bugs:

**Bug 1 — Target landing point axes were swapped (CRITICAL):**
`SpecularAimer` default target was `[0.0, TABLE_LENGTH/4, TABLE_HEIGHT]` = `[0, 0.685, 0.76]`. In the sim, ball travels in +X, opponent's side is -X, table extends in X. The target was at the **net** (x=0) offset **laterally** (y=0.685). The paddle normal was computed to redirect the ball sideways rather than back over the net.

**Fix:** Changed to `[-TABLE_LENGTH/4, 0.0, TABLE_HEIGHT]` = `[-0.685, 0.0, 0.76]` — center of opponent's half in the correct axis.

**Bug 2 — `t_contact` was nonsensical (CRITICAL):**
`SpecularAimer` computed `t_contact = norm(ball.position) / speed_in` — distance from **origin** divided by ball speed. This has no physical meaning. A ball at `(1.3, 0.3, 0.9)` gives `t_contact = 0.65s`, but the actual time remaining may be 0.2s or 0.8s depending on when the controller plans.

The `QuinticSwingPlanner` uses `t_contact` as the trajectory duration (T). Wrong T means the quintic's acceleration profile is mismatched — the arm arrives too early or too late.

**Fix:** Both controllers now override `paddle_target.t_contact = max(t_intercept - t_now, 0.05)` with the actual remaining time to intercept, computed from the drag-aware predictor. The aimer's fallback was also improved to use `dx / speed_in` (distance in X to table edge) instead of `norm(position) / speed`.

**Bug 3 — WAM had no gravity compensation (HIGH):**
WAM used `kp=50, kd=5` with no `qfrc_bias` feedforward. The PD controller was fighting gravity with the same gains used for trajectory tracking, causing large steady-state errors and sluggish response. Fanuc had `kp=400, kd=40` plus gravity comp — much stiffer.

**Fix:** Enabled `gravity_comp=True` for WAM, increased gains to `kp_scale=100, kd_scale=10`. The gravity compensation feedforward is now divided by gear ratio to convert from torque space to control space (needed because WAM actuators have large gear ratios: 600, 500, 160, ...).

**Comparison after all three fixes (20 episodes, `drag_bounce`, `specular`):**

| Control | Swing | Contact (before) | Contact (after) | Reward (after) |
|---------|-------|:-:|:-:|:-:|
| `open_loop` | `quintic` | 5% | 5% | 0.261 |
| `replan` | `quintic` | 10% | **15%** | **0.393** |
| `replan` | `lerp` | 5% | 0% | 0.087 |

**40-episode eval of best config (`replan` + `quintic`):** **12.5% contact rate**, mean reward 0.389.

**Analysis:** The fixes improved `replan+quintic` from 10% → 12-15%. The correct target axis means the specular normal now redirects the ball back over the net. The correct `t_contact` means the quintic matches the actual time budget. Gravity comp gives the WAM stiffer tracking. `replan+lerp` degraded because lerp doesn't benefit from the corrected timing — it always interpolates linearly regardless of T.

**Remaining bottlenecks:**
1. **Ball lateral variation** — ball y-position is random in [-0.65, 0.65], requiring the arm to cover a wide workspace
2. **Single IK solution** — the IK may converge to configurations that are kinematically reachable but dynamically infeasible in the time budget
3. **No velocity-space planning** — the quintic plans in joint position space but the PD servo may not track it fast enough for aggressive swings
4. **20-episode variance** — 5% = 1 HIT difference; larger evals needed for confident comparisons

**Next steps:**
1. ~~Investigate prediction accuracy — does the predictor match the sim physics?~~ Done — found phantom drag bug
2. Investigate IK solution quality — does the arm actually reach the predicted contact point?
3. Try Fanuc robot (stiffer tracking, gravity comp tuned) for comparison

### Investigation: Phantom drag — predictor vs sim physics mismatch

Deep investigation into prediction accuracy revealed two physics mismatches between the `DragBouncePredictor` and the MuJoCo simulation:

**Bug 1 — Phantom air drag (CRITICAL):**
The predictor applied quadratic aerodynamic drag (`C_D=0.4, K_DRAG=0.114`), but the MuJoCo simulation has **no air resistance at all**. The XML scenes have `density=0` and `viscosity=0` (defaults), and the ball joint has `damping=0.0`. The ball is purely ballistic in the sim.

Effect: The predictor predicted the ball arriving ~14% slower and ~0.05-0.1s later than reality. At v=2.5 m/s, this means ~5-10cm position error in X at the predicted intercept time.

**Bug 2 — Bounce restitution (investigated, kept as-is):**
The predictor used `e=0.91`. MuJoCo uses a spring-damper contact model (`solref=[0.1, 0.03]`), which we initially estimated at e≈0.97. However, testing showed that `e=0.91` gives better results than `e=0.97`, suggesting the effective COR in MuJoCo is closer to 0.91 after accounting for the full contact dynamics.

**Fix:** Made `DragBouncePredictor` configurable with `c_d` and `restitution` constructor parameters (default to original values for backward compatibility). In `run.py`, the predictor is now instantiated with `c_d=0.0, restitution=0.91` to match the sim.

**Comparison (40 episodes, `replan` + `quintic` + `specular`):**

| Predictor config | Contact Rate | Mean Reward |
|:---|:-:|:-:|
| `c_d=0.4, e=0.91` (original) | 12.5% | 0.389 |
| `c_d=0.0, e=0.97` (overcorrected) | 0% | 0.161 |
| `c_d=0.0, e=0.91` (sim-matched) | **17.5%** | **0.471** |

**Also tested:** WAM gravity compensation (`qfrc_bias` feedforward, kp=100, kd=10) — this made results **worse** (0% contact in 40 episodes). The WAM's existing PD gains (kp=50, kd=5 effective) work better without gravity comp, possibly because the gravity compensation interacts poorly with the gear-ratio-scaled control space. Reverted.

**Best overall result to date:** **20% contact rate** (4/20 episodes) with `c_d=0.0, e=0.91`, confirmed at 17.5% over 40 episodes. This is a significant improvement from the initial 0% contact with baseline components.

**Next steps:**
1. Investigate remaining 80% miss rate — are misses near-misses or complete whiffs?
2. ~~Try the Fanuc robot for comparison~~ Done
3. Consider the `ballistic` predictor (also no drag) with bounce handling

### Fanuc home position fix

**Problem:** The Fanuc's original `init_qpos = [-2.938, 1.339, -2.621, -2.009, -2.92, -2.819]` placed J1 at -168°, pointing the entire arm into the ball's flight path (-X direction). The elbow and upper arm occupied x ≈ 0.5-1.2 — the ball frequently collided with the arm links instead of the paddle.

**Investigation:** Analyzed the Fanuc kinematic chain (J1=Z, J2=Y, J3=-Y, J4=-X, J5=-Y, J6=-X) and the scene geometry (base at x=1.6, ball travels in +X). Tested two alternative home positions:

- **Option A (arm to side):** `[-1.57, 0.8, -1.5, 0.0, -1.57, 0.0]` — J1=-90° rotates the arm to +Y, completely out of the ball's XZ flight plane
- **Option B (arm folded back):** `[0.0, -0.5, -2.0, 0.0, -1.0, 0.0]` — J1=0° tucks the arm behind the robot in +X

**Results (20 episodes each, `replan` + `quintic` + `specular` + `drag_bounce`):**

| Home Position | Contact Rate | Mean Reward | Notes |
|:---|:-:|:-:|:---|
| Original (J1=-168°) | 5% | 0.243 | Elbow blocks ball, sim instability |
| Option A (J1=-90°, side) | 0% | 0.178 | No blocking, stable, near-misses ~0.2m |
| Option B (J1=0°, back) | 0% | 0.154 | No blocking, some very close near-misses |

**With reactive controller (Option A):** 5% contact — confirms the new home position is viable and the arm can reach the ball. The `replan` controller needs further tuning for the Fanuc's 6-DOF geometry and different base position (x=1.6 vs WAM x=2.1).

**Decision:** Adopted **Option A** as the new default Fanuc `init_qpos`. Eliminates the elbow-blocking problem entirely. The contact rate under `replan` will improve with further tuning (the Fanuc's ROBOT_X=1.3 intercept plane may need adjustment given its base at x=1.6).

### Robot-specific intercept plane (`robot_x` configurable)

**Problem:** Both `OpenLoopController` and `ReplanController` hardcoded `ROBOT_X = 1.3` as the intercept plane. For WAM (base at x=2.1), this gives a comfortable 0.8m reach. For Fanuc (base at x=1.6), x=1.3 is only 0.3m from the base — near-singular IK that often fails to converge, causing 0% contact with `replan`.

**Fix:** Made `robot_x` a configurable constructor parameter in both controllers (default 1.3). `ROBOT_CONFIGS` now specifies per-robot values: WAM=1.3, Fanuc=1.0. `load_subsystems()` passes this through to the controller.

### Fanuc actuator model overhaul (gear ratios + PD tuning + IK fix)

**Problem:** Fanuc was stuck at 0% contact despite correct intercept geometry. Three root causes identified:

1. **Direct torque control (gear=1)**: Fanuc actuators used `gear=1` with `ctrlrange=[-500, 500]`, while WAM used high gear ratios (8-600) with normalized `ctrlrange=[-1, 1]`. The gear=1 setup meant no mass matrix regularization from the actuator model, causing QACC instability and poor numerical conditioning.

2. **Missing joint limits**: Fanuc joints had `range` attributes but no `limited="true"`, so MuJoCo didn't enforce joint limits. The IK solver could find solutions in unreachable configurations.

3. **IK solution discontinuity**: The damped least-squares IK (damping=0.01) could jump between different solution branches on consecutive replans, producing discontinuous `q_desired` trajectories that the PD controller couldn't track.

**Fixes applied:**

| Change | Before | After |
|:---|:---|:---|
| Actuator gear ratios | `gear=1`, `ctrlrange=[-500,500]` | `gear=[500,500,300,150,80,50]`, `ctrlrange=[-1,1]` |
| Joint limits | `range` only (not enforced) | Added `limited="true"` to all 6 joints |
| IK damping | 0.01 | 0.05 (prevents large jumps) |
| IK step limit | None | max 0.2 rad/iteration |
| IK seeding | From env qpos | From `q_current` (solution continuity) |
| PD gains (kp/kd) | 400/40 with gear=1 | 800/50 with gear ratios |
| Gravity comp | On (raw Nm) | On (normalized to ctrl space via `/gear`) |

**Results (40 episodes, `replan` + `quintic` + `specular` + `drag_bounce(c_d=0)`):**

| Robot | Contact Rate | Landing Rate | Mean Reward | Notes |
|:---|:-:|:-:|:-:|:---|
| **Fanuc** | **17.5%** | **5.0%** | 0.659 | 2 successful landings! |
| WAM | 20.0% | 0.0% | 0.514 | No regression |

**Key insight:** The critical improvement came from the combination of gear ratios (normalized control space, mass matrix regularization) and high PD gains (kp=800, kd=50 in gear-normalized space → effective 800/500 * 500 = 800 Nm/rad stiffness). The Fanuc is ~8x heavier than WAM (118kg vs 14kg) and needs proportionally stiffer control. Min paddle-ball distance analysis showed the arm was consistently within 0.1-0.2m even at 0% contact — the gear ratio + PD gain fix closed that gap.

**Files modified:**
- `assets/xml/include_fanuc_actuator.xml` — gear ratios, normalized ctrlrange
- `assets/xml/include_fanuc_crx25ia.xml` — `limited="true"` on all joints
- `run.py` — Fanuc config (gear, kp_scale, kd_scale), IK (seeding, damping, step limit), `load_subsystems()` (robot_x wiring)
- `tt_sim/upgrades/swing.py` — pass `q_current` as IK seed

### Upgrade 6: MPC Controller (Stage 2a)

**What changed:** Added `MPCController` in `tt_sim/upgrades/control.py` using CasADi to solve a receding-horizon NLP at each replan cycle. A symbolic FK builder (`tt_sim/upgrades/mpc_fk.py`) extracts the kinematic chain from MuJoCo XML and builds CasADi symbolic FK functions with Rodrigues rotation per joint. The MPC replaces both the swing planner and replan controller — it directly optimizes a joint trajectory (N=15 knot points) toward the predicted ball intercept with terminal position/orientation/velocity costs and smoothness/effort regularization.

**Key design choices:**
- **Adaptive horizon:** `dt_mpc = t_remaining / N` so the horizon always stretches to the intercept time
- **Warm-starting:** Previous solution shifted by one step provides continuity across replans
- **60% lookahead interpolation:** PD controller needs `q_desired` ahead of `q_current` for torque generation. Pure trajectory following (interpolate at `t_now`) only produces ~0.2 rad PD error vs quintic's ~1.1 rad. 60% of remaining time was the empirically optimal balance between PD drive and overshoot avoidance.
- **CasADi NLP with IPOPT:** 80 max iterations, velocity constraints (5 rad/s), joint limits as bounds
- **Row-major FK flattening:** Fixed CasADi's column-major `reshape` to match NumPy convention

**Results (40 episodes, drag_bounce + specular):**

| Robot | Controller | Contact | Landing | Mean Reward | Notes |
|-------|-----------|---------|---------|-------------|-------|
| **Fanuc** | open_loop+quintic | 45-65% | 10% | ~2.0 | Best baseline |
| **Fanuc** | replan+quintic | 17.5% | 5% | 0.659 | IK discontinuity problem |
| **Fanuc** | **mpc** | **20-27%** | **0-2.5%** | **0.65-1.0** | **Eliminates IK jumps** |
| WAM | replan+quintic | 20-25% | 0% | 0.51 | Replan works for WAM |
| WAM | **mpc** | **0%** | **0%** | **0.15** | **7-DOF NLP harder** |

**Analysis:**
- MPC **improves over replan for Fanuc** (20-27% vs 17.5%) by eliminating IK solution discontinuities. The NLP produces smooth trajectories without jumps between replans.
- MPC **does not match open_loop quintic** (45-65%) because the position-commanded PD interface limits tracking accuracy. The quintic polynomial provides smooth time-parameterized motion that naturally ramps PD error. The MPC trajectory, delivered through lookahead interpolation, caps PD drive.
- Most Fanuc misses have `min_pb_dist` of 0.08-0.15m (paddle radius 0.075m) — the arm gets very close but can't quite close the gap.
- WAM's 7-DOF makes the NLP harder to solve within 80 iterations, and the different PD gains/dynamics aren't tuned for MPC.
- **Limitation:** The architecture mismatch between trajectory-level MPC and position-commanded PD low-level control is the fundamental bottleneck. A torque-level MPC (outputting joint torques directly) would bypass this, but requires reformulating the problem with full dynamics.

**Files created/modified:**
- `tt_sim/upgrades/mpc_fk.py` — Symbolic FK builder from MuJoCo XML for CasADi
- `tt_sim/upgrades/control.py` — `MPCController` class
- `tt_sim/registry.py` — `"mpc"` entry
- `run.py` — FK dict construction for MPC in `load_subsystems()`
- `tests/test_upgrades/test_mpc_fk.py` — 8 FK validation tests
- `tests/test_upgrades/test_mpc_control.py` — 5 MPC controller tests

## Tests

```bash
python -m pytest tests/ -q
```

130 tests covering all baselines, upgrade stubs, FK validation, and MPC controller.
>>>>>>> b3893566535958413651b04af1f1c4097e1cf268
