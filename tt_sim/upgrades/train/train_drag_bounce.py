"""Train drag-bounce predictor: fit aerodynamic drag coefficient C_D.

What to train
-------------
The DragBouncePredictor extends the ballistic baseline by adding quadratic air
drag (parameterised by a single scalar drag coefficient C_D) and an inelastic
bounce model (coefficient of restitution e).  This script fits C_D and e from
simulated trajectory data using non-linear least-squares optimisation.

Data
----
Simulated trajectories exported as NumPy arrays.  Each file contains an
(N, 7) array: columns [t, x, y, z, vx, vy, vz].  Pass the directory of
.npy files via ``--data-dir``.

Expected output
---------------
A JSON file at ``models/prediction/drag_bounce.json`` containing the fitted
parameters ``{"C_D": float, "e": float}``.
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import curve_fit


def ballistic_with_drag(t, x0, v0, C_D, e, g=9.81, rho=1.225, r=0.02, m=0.0027):
    """Forward-simulate a trajectory with drag for curve_fit."""
    # TODO: implement RK4 integration with drag force
    # F_drag = -0.5 * rho * C_D * pi * r^2 * |v| * v
    raise NotImplementedError("Implement forward model for curve_fit target")


def load_trajectories(data_dir: pathlib.Path):
    """Load all .npy trajectory files from *data_dir*."""
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    return [np.load(f) for f in files]


def fit(trajectories):
    """Fit C_D and e across all trajectories."""
    # TODO: stack observations, define residual, call curve_fit
    p0 = [0.5, 0.85]  # initial guesses for C_D, e
    # popt, pcov = curve_fit(model, t_all, pos_all, p0=p0)
    raise NotImplementedError("Implement fitting loop")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True, help="Directory of .npy trajectory files")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("models/prediction/drag_bounce.json"))
    args = parser.parse_args()

    trajectories = load_trajectories(args.data_dir)
    params = fit(trajectories)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved drag-bounce params to {args.output}")


if __name__ == "__main__":
    main()
