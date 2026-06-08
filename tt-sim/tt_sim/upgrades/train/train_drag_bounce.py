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

"""Train drag-bounce predictor: fit aerodynamic drag coefficient C_D and restitution e."""

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares


def simulate_drag_bounce(
    t,
    x0,
    v0,
    C_D,
    e,
    g=9.81,
    rho=1.225,
    r=0.02,
    m=0.0027,
):
    """
    Forward-simulate ball motion with quadratic drag and ground bounce.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time values.
    x0 : array-like, shape (3,)
        Initial position [x, y, z].
    v0 : array-like, shape (3,)
        Initial velocity [vx, vy, vz].
    C_D : float
        Drag coefficient.
    e : float
        Coefficient of restitution for z bounce.

    Returns
    -------
    states : ndarray, shape (N, 6)
        Simulated [x, y, z, vx, vy, vz].
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()

    area = np.pi * r**2
    k = 0.5 * rho * C_D * area / m

    states = np.zeros((len(t), 6), dtype=float)
    states[0, :3] = x
    states[0, 3:] = v

    def acceleration(vel):
        speed = np.linalg.norm(vel)
        drag_acc = -k * speed * vel
        gravity_acc = np.array([0.0, 0.0, -g])
        return drag_acc + gravity_acc

    def rk4_step(pos, vel, dt):
        def deriv(p, vv):
            return vv, acceleration(vv)

        k1_x, k1_v = deriv(pos, vel)
        k2_x, k2_v = deriv(pos + 0.5 * dt * k1_x, vel + 0.5 * dt * k1_v)
        k3_x, k3_v = deriv(pos + 0.5 * dt * k2_x, vel + 0.5 * dt * k2_v)
        k4_x, k4_v = deriv(pos + dt * k3_x, vel + dt * k3_v)

        pos_next = pos + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        vel_next = vel + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return pos_next, vel_next

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])

        if dt <= 0:
            raise ValueError("Time values must be strictly increasing.")

        x, v = rk4_step(x, v, dt)

        # Ground bounce at z = 0
        if x[2] < 0:
            x[2] = 0.0

            if v[2] < 0:
                v[2] = -e * v[2]

        states[i, :3] = x
        states[i, 3:] = v

    return states


def ballistic_with_drag(t, x0, v0, C_D, e, g=9.81, rho=1.225, r=0.02, m=0.0027):
    """
    Compatibility wrapper for forward simulation.

    Returns only position [x, y, z].
    """
    states = simulate_drag_bounce(
        t=t,
        x0=x0,
        v0=v0,
        C_D=C_D,
        e=e,
        g=g,
        rho=rho,
        r=r,
        m=m,
    )
    return states[:, :3]


def load_trajectories(data_dir: pathlib.Path):
    """Load all .npy trajectory files from data_dir."""

    files = sorted(data_dir.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    trajectories = []

    for file in files:
        arr = np.load(file)

        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(
                f"{file} has shape {arr.shape}, but expected shape (N, 7): "
                "[t, x, y, z, vx, vy, vz]"
            )

        trajectories.append(arr)

    return trajectories


def fit(trajectories):
    """
    Fit C_D and e across all trajectories using nonlinear least squares.

    Each trajectory must have columns:
    [t, x, y, z, vx, vy, vz]
    """

    def residual(params):
        C_D, e = params

        all_residuals = []

        for traj in trajectories:
            t = traj[:, 0]
            observed_pos = traj[:, 1:4]

            x0 = traj[0, 1:4]
            v0 = traj[0, 4:7]

            predicted_pos = ballistic_with_drag(
                t=t,
                x0=x0,
                v0=v0,
                C_D=C_D,
                e=e,
            )

            err = predicted_pos - observed_pos
            all_residuals.append(err.ravel())

        return np.concatenate(all_residuals)

    p0 = np.array([0.5, 0.85])

    lower_bounds = np.array([0.01, 0.1])
    upper_bounds = np.array([2.0, 1.0])

    result = least_squares(
        residual,
        x0=p0,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        max_nfev=300,
        verbose=1,
    )

    C_D, e = result.x

    return {
        "C_D": float(C_D),
        "e": float(e),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": result.message,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Directory of .npy trajectory files",
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("models/prediction/drag_bounce.json"),
    )

    args = parser.parse_args()

    trajectories = load_trajectories(args.data_dir)
    params = fit(trajectories)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved drag-bounce params to {args.output}")
    print(params)


if __name__ == "__main__":
    main()