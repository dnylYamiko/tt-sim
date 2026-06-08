"""Train Magnus spin estimator from Kienzle 50k spin ground truth.

What to train
-------------
The MagnusSpinEstimator uses the Magnus effect equation to relate observed
lateral acceleration deviations to ball spin.  The Magnus force is:
    F_M = C_M * (omega x v)
where C_M is a scalar Magnus coefficient that depends on ball properties and
Reynolds number.  This script fits C_M (and optionally a spin-decay constant
lambda) from ground-truth spin measurements paired with trajectory data.

Data
----
Kienzle 50k dataset with spin annotations: .npz with keys ``states`` (N, T, 6),
``spin`` (N, T, 3) giving angular velocity in rad/s, and ``dt`` (scalar).
Pass the path via ``--data``.

Expected output
---------------
PyTorch checkpoint (dict with keys ``C_M``, ``lambda_decay``) saved to
``models/spin/magnus.pt``.
"""

import argparse
import pathlib

import numpy as np
import torch
from scipy.optimize import curve_fit


def magnus_force(v, omega, C_M):
    """Compute Magnus force: C_M * (omega x v)."""
    return C_M * np.cross(omega, v)


def load_data(path: pathlib.Path):
    """Load spin-annotated Kienzle 50k dataset."""
    data = np.load(path)
    # TODO: extract velocity, spin, and observed lateral acceleration
    raise NotImplementedError("Implement data loading")


def fit(velocities, spins, accels_lateral):
    """Fit C_M and lambda_decay from paired data."""
    # TODO: set up residual function and call curve_fit or torch optimiser
    p0 = [1.0, 0.01]  # C_M, lambda_decay
    raise NotImplementedError("Implement fitting")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to Kienzle 50k spin .npz")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("models/spin/magnus.pt"))
    args = parser.parse_args()

    velocities, spins, accels = load_data(args.data)
    params = fit(velocities, spins, accels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, args.output)
    print(f"Saved Magnus params to {args.output}")


if __name__ == "__main__":
    main()
