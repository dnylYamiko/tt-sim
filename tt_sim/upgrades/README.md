# Upgrade System

## Selecting an upgrade

Every pipeline subsystem (perceiver, predictor, spin, aimer, swing, control)
can be swapped at the command line with a `--<subsystem>` flag:

```bash
python -m tt_sim.run --predictor drag_bounce --spin magnus
```

Run `python -m tt_sim.registry` (or call `registry.list_available()`) to see
all registered implementations and their defaults.

## Training models

Learned upgrades require trained weights before use.  Training scripts live in
`tt_sim/upgrades/train/` and are standalone CLIs:

```bash
python -m tt_sim.upgrades.train.train_residual_mlp --data data/kienzle_50k.npz
python -m tt_sim.upgrades.train.train_neural_ode   --data data/kienzle_120k.npz
python -m tt_sim.upgrades.train.train_gp            --data data/kienzle_50k.npz
python -m tt_sim.upgrades.train.train_magnus_spin   --data data/kienzle_50k_spin.npz
python -m tt_sim.upgrades.train.train_drag_bounce   --data-dir data/sim_trajectories/
```

Each script prints its expected arguments with `--help`.

## Model storage convention

Trained weights are saved under `models/<subsystem>/<name>.<ext>`:

```
models/
  prediction/
    residual_mlp.pt
    neural_ode.pt
    gp.pt
    drag_bounce.json
  spin/
    magnus.pt
```

The `models/` directory is git-ignored (weights are large).  Regenerate them
by re-running the corresponding train script.
