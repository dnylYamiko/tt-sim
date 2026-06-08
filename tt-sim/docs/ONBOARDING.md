## tt-sim Onboarding — Pramit

### 1. Clone the repo

```bash
git clone https://github.com/Farjad1/tt-sim.git
cd tt-sim
```

### 2. Python environment

We use conda base with Python 3.12. Key dependencies:

```bash
pip install mujoco==3.8.0
pip install --no-deps fancy_gym
pip install mp-pytorch<=0.1.3 gymnasium==0.29.1
pip install click imageio imageio-ffmpeg opencv-python numpy
pip install pytest  # for running tests
```

> **Note:** `fancy_gym 0.3.0` tries to install `mujoco==2.3.3` which won't build on Apple Silicon. The `--no-deps` flag avoids that conflict.

### 3. Verify setup

```bash
# Run all 88 tests
pytest tests/ -v

# Dry run (loads subsystems without sim)
python run.py --dry-run

# List all available implementations
python run.py --list
```

### 4. Run the sim

```bash
# WAM robot (7-DOF), reactive controller, live viewer
python run.py --robot=wam --control=reactive --episodes=10 --render

# Fanuc CRX-25iA (6-DOF), reactive controller, record video
python run.py --robot=fanuc --control=reactive --episodes=10 --record=results/my_test.mp4

# Open-loop controller with ballistic predictor
python run.py --robot=wam --control=open_loop --predictor=ballistic --episodes=10 --render
```

Key CLI flags:
- `--robot=wam|fanuc` — select robot
- `--control=reactive|open_loop` — controller type
- `--perceiver=sim` / `--predictor=ballistic` / `--spin=zero` / `--aimer=face_net` / `--swing=lerp` — subsystem selection
- `--render` — live cv2 viewer (press 'q' to quit)
- `--record=path.mp4` — save video
- `--log=path.csv` — save metrics CSV
- `--original-scene` — use ceiling-mounted WAM (fancy_gym default)

### 5. Project structure

```
tt-sim/
├── run.py                  # CLI entry point
├── tt_sim/
│   ├── interfaces.py       # ABCs + dataclasses (BallState, PaddleTarget, etc.)
│   ├── registry.py         # Maps string names → classes
│   ├── eval.py             # EvalLogger, EpisodeMetrics
│   ├── generate.py         # Data generation CLI
│   ├── baselines/          # Stage 0: working implementations
│   │   ├── perception.py   # SimPerceiver
│   │   ├── prediction.py   # BallisticPredictor
│   │   ├── spin.py         # ZeroSpinEstimator
│   │   ├── aiming.py       # FaceNetAimer
│   │   ├── swing.py        # LerpSwingPlanner, MujocoIKSwingPlanner
│   │   └── control.py      # OpenLoopController, ReactiveController
│   └── upgrades/           # Stage 1+: stubs to fill in
│       ├── prediction.py   # DragBouncePredictor, ResidualMLPPredictor, etc.
│       ├── aiming.py       # SpecularAimer
│       ├── swing.py        # QuinticSwingPlanner
│       ├── spin.py         # MagnusSpinEstimator
│       ├── control.py      # ReplanController
│       ├── perception.py   # HSVPerceiver
│       └── train/          # Training scripts for ML upgrades
├── assets/xml/             # MuJoCo scene files
├── tests/                  # 88 tests (baselines + upgrade stubs)
└── models/                 # Trained model weights (gitignored)
```

### 6. How upgrades work

Each upgrade stub in `tt_sim/upgrades/` has:
- A class inheriting from the corresponding ABC in `interfaces.py`
- A detailed docstring explaining the algorithm to implement
- A registration in `registry.py` so it can be selected via CLI flags
- Matching tests in `tests/test_upgrades/`

**Workflow to implement an upgrade:**
1. Read the stub's docstring — it describes the algorithm
2. Fill in the methods (replace `raise NotImplementedError`)
3. Run its tests: `pytest tests/test_upgrades/test_prediction.py -v` (example)
4. Run the sim with it: `python run.py --predictor=drag_bounce --episodes=10 --render`
5. Compare against baseline: check contact rate, reward, landing distance

**Example:** To implement `DragBouncePredictor`:
```bash
# 1. Read the stub
cat tt_sim/upgrades/prediction.py

# 2. Edit and implement

# 3. Test
pytest tests/test_upgrades/test_prediction.py -v

# 4. Run
python run.py --predictor=drag_bounce --control=open_loop --episodes=20 --log=results/drag_bounce.csv
```

### 7. Current performance baselines

| Robot | Controller | Contact Rate |
|-------|-----------|-------------|
| WAM (floor) | reactive | ~25% |
| WAM (ceiling) | reactive | ~46-70% |
| Fanuc CRX-25iA | reactive | ~40% |

### 8. Git workflow

```bash
git pull farjad1 main          # get latest
# ... make changes ...
git add -A && git commit -m "descriptive message"
git push farjad1 main
```
