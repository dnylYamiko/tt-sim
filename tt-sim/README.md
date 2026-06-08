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

## Tests

```bash
python -m pytest tests/ -q
```

88 tests covering all baselines and upgrade stubs.
