"""Train Neural ODE predictor on Kienzle 50k dataset.

What to train
-------------
The NeuralODEPredictor models ball dynamics as a learned ODE:
    ds/dt = f_theta(s)
where s = [x, y, z, vx, vy, vz] and f_theta is a neural network.  The ODE is
integrated with an adaptive solver (dopri5) via torchdiffeq.  Training
minimises the MSE between solver-predicted and ground-truth trajectories.

Data
----
Kienzle 120k dataset: 120 000 trajectories stored as NumPy .npz with keys
``states`` (N, T, 6) and ``times`` (N, T).  Pass the path via ``--data``.

Expected output
---------------
PyTorch checkpoint saved to ``models/prediction/neural_ode.pt``.
"""
"""Train Neural ODE predictor on Kienzle trajectory dataset.

The NeuralODEPredictor models ball dynamics as a learned ODE:

    ds/dt = f_theta(s)

where:

    s = [x, y, z, vx, vy, vz]

The model is trained by integrating the learned ODE and minimising the MSE
between predicted and ground-truth trajectories.

Supported data formats
----------------------
.npz:
    states: (N, T, 6)
    times:  (N, T) or (T,)
    OR
    dt: scalar

.npy:
    (T, 7)    -> [t, x, y, z, vx, vy, vz]
    (T, 6)    -> [x, y, z, vx, vy, vz], assumes dt=0.01
    (N, T, 6) -> multiple trajectories, assumes dt=0.01

Outputs
-------
Best model:
    models/prediction/neural_ode.pt

Training history:
    models/prediction/neural_ode_history.json

Plots:
    models/prediction/plots/neural_ode_loss_plot.png
    models/prediction/plots/neural_ode_mse_plot.png
    models/prediction/plots/ground_truth_vs_prediction_position.png
    models/prediction/plots/ground_truth_vs_prediction_velocity.png
    models/prediction/plots/prediction_mse_over_time.png
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class ODEFunc(nn.Module):
    """Neural network that defines ds/dt = f(s)."""

    def __init__(self, hidden: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6),
        )

    def forward(self, t, s):
        return self.net(s)


def load_data(path: pathlib.Path, min_timesteps: int = 10):
    """Load trajectory data and return s0, times, targets."""

    path = pathlib.Path(path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(list(path.rglob("*.npz")) + list(path.rglob("*.npy")))
    else:
        raise FileNotFoundError(f"{path} does not exist")

    if not files:
        raise FileNotFoundError(f"No .npz or .npy files found under {path}")

    all_states = []
    all_times = []

    for file in files:
        print(f"Loading {file}")

        if file.suffix == ".npz":
            data = np.load(file)

            if "states" not in data:
                print(f"Skipping {file}: missing key 'states'")
                continue

            states = data["states"].astype(np.float32)

            if "times" in data:
                times = data["times"].astype(np.float32)

            elif "dt" in data:
                dt = float(data["dt"])
                times = np.arange(states.shape[1], dtype=np.float32) * dt

            else:
                print(f"Skipping {file}: missing 'times' or 'dt'")
                continue

        elif file.suffix == ".npy":
            arr = np.load(file, allow_pickle=True)

            if arr.ndim == 2 and arr.shape[1] == 7:
                # [t, x, y, z, vx, vy, vz]
                times = arr[:, 0].astype(np.float32)
                states = arr[:, 1:7].astype(np.float32)

                states = states[None, :, :]
                times = times[None, :]

            elif arr.ndim == 2 and arr.shape[1] == 6:
                # [x, y, z, vx, vy, vz]
                states = arr.astype(np.float32)
                times = np.arange(states.shape[0], dtype=np.float32) * 0.01

                states = states[None, :, :]
                times = times[None, :]

            elif arr.ndim == 3 and arr.shape[2] == 6:
                # (N, T, 6)
                states = arr.astype(np.float32)
                times = np.arange(states.shape[1], dtype=np.float32) * 0.01

            else:
                print(f"Skipping {file}: unsupported .npy shape {arr.shape}")
                continue

        else:
            continue

        states = np.asarray(states, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)

        if states.ndim != 3 or states.shape[2] != 6:
            print(
                f"Skipping {file}: expected states shape (N, T, 6), "
                f"got {states.shape}"
            )
            continue

        if times.ndim == 1:
            times = np.tile(times[None, :], (states.shape[0], 1))

        if times.ndim != 2:
            print(
                f"Skipping {file}: expected times shape (N, T) or (T,), "
                f"got {times.shape}"
            )
            continue

        if times.shape[0] != states.shape[0] or times.shape[1] != states.shape[1]:
            print(
                f"Skipping {file}: states shape {states.shape} and "
                f"times shape {times.shape} are incompatible"
            )
            continue

        if states.shape[1] < min_timesteps:
            print(f"Skipping {file}: too few timesteps T={states.shape[1]}")
            continue

        if not np.isfinite(states).all() or not np.isfinite(times).all():
            print(f"Skipping {file}: contains NaN or Inf")
            continue

        all_states.append(states)
        all_times.append(times)

        print(f"  accepted states {states.shape}, times {times.shape}")

    if not all_states:
        raise ValueError(
            f"No valid Neural ODE training data found under {path}. "
            "Expected .npz with states/times or .npy shaped "
            "(N,T,6), (T,6), or (T,7)."
        )

    # Crop all trajectories to the same length.
    min_T = min(s.shape[1] for s in all_states)

    print(f"\nCropping all trajectories to T = {min_T}")

    all_states = [s[:, :min_T, :] for s in all_states]
    all_times = [t[:, :min_T] for t in all_times]

    states = np.concatenate(all_states, axis=0)
    times = np.concatenate(all_times, axis=0)

    s0 = states[:, 0, :]
    targets = states

    s0 = torch.tensor(s0, dtype=torch.float32)
    times = torch.tensor(times, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    print("\nFinal dataset:")
    print(f"s0 shape:      {s0.shape}")
    print(f"times shape:   {times.shape}")
    print(f"targets shape: {targets.shape}")

    return s0, times, targets


def make_train_val_loaders(
    s0,
    times,
    targets,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
):
    """Create train and validation DataLoaders."""

    dataset = TensorDataset(s0, times, targets)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Invalid train/validation split: "
            f"total={total_size}, train={train_size}, val={val_size}. "
            "Use more data or reduce --val-split."
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print("\nData split:")
    print(f"Training samples:   {train_size}")
    print(f"Validation samples: {val_size}")

    return train_loader, val_loader


def run_epoch(func, loader, criterion, device, optimiser=None):
    """Run one training or validation epoch."""

    is_training = optimiser is not None

    if is_training:
        func.train()
    else:
        func.eval()

    total_loss = 0.0
    total_samples = 0

    for s0, times, s_target in loader:
        s0 = s0.to(device)
        times = times.to(device)
        s_target = s_target.to(device)

        # torchdiffeq expects one time vector for the whole batch.
        t = times[0]

        if not torch.allclose(times, t.unsqueeze(0).expand_as(times), atol=1e-5):
            raise ValueError(
                "All trajectories in a batch must share the same time grid. "
                "Use --batch-size 1 if your trajectories have different time grids."
            )

        if is_training:
            optimiser.zero_grad()

            pred = odeint(
                func,
                s0,
                t,
                method="dopri5",
                rtol=1e-5,
                atol=1e-6,
            )

            # odeint returns (T, B, 6), convert to (B, T, 6)
            pred = pred.permute(1, 0, 2)

            loss = criterion(pred, s_target)

            loss.backward()
            optimiser.step()

        else:
            with torch.no_grad():
                pred = odeint(
                    func,
                    s0,
                    t,
                    method="dopri5",
                    rtol=1e-5,
                    atol=1e-6,
                )

                pred = pred.permute(1, 0, 2)
                loss = criterion(pred, s_target)

        batch_size = s0.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train(
    func,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    output_path,
):
    """Train Neural ODE with validation and best-checkpoint saving."""

    if odeint is None:
        raise ImportError("Install torchdiffeq first: pip install torchdiffeq")

    func.to(device)

    optimiser = torch.optim.Adam(func.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = run_epoch(
            func=func,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimiser=optimiser,
        )

        val_loss = run_epoch(
            func=func,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimiser=None,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        # MSELoss is used, so loss and MSE are the same value here.
        history["train_mse"].append(float(train_loss))
        history["val_mse"].append(float(val_loss))

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": func.state_dict(),
                    "best_val_loss": float(best_val_loss),
                    "epoch": epoch + 1,
                    "history": history,
                },
                output_path,
            )

            print(f"  saved best checkpoint to {output_path}")

    return history


def plot_loss_curves(history, output_dir: pathlib.Path):
    """Plot train and validation loss."""

    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural ODE Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = output_dir / "neural_ode_loss_plot.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved loss plot to {path}")


def plot_mse_curves(history, output_dir: pathlib.Path):
    """Plot train and validation MSE."""

    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_mse"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mse"], label="Train MSE")
    plt.plot(epochs, history["val_mse"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Neural ODE Train/Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = output_dir / "neural_ode_mse_plot.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved MSE plot to {path}")


def plot_ground_truth_vs_prediction(
    func,
    loader,
    device,
    output_dir: pathlib.Path,
):
    """Plot ground truth vs predicted positions and velocities."""

    if odeint is None:
        raise ImportError("Install torchdiffeq first: pip install torchdiffeq")

    output_dir.mkdir(parents=True, exist_ok=True)

    func.eval()

    s0, times, s_target = next(iter(loader))

    s0 = s0.to(device)
    times = times.to(device)
    s_target = s_target.to(device)

    # Use one validation trajectory.
    s0_one = s0[0:1]
    t = times[0]
    target_one = s_target[0].detach().cpu().numpy()

    with torch.no_grad():
        pred = odeint(
            func,
            s0_one,
            t,
            method="dopri5",
            rtol=1e-5,
            atol=1e-6,
        )

    # pred shape: (T, 1, 6) -> (T, 6)
    pred = pred[:, 0, :].detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()

    # Position comparison plot
    plt.figure(figsize=(10, 6))

    plt.plot(t_np, target_one[:, 0], label="Ground truth x")
    plt.plot(t_np, pred[:, 0], "--", label="Predicted x")

    plt.plot(t_np, target_one[:, 1], label="Ground truth y")
    plt.plot(t_np, pred[:, 1], "--", label="Predicted y")

    plt.plot(t_np, target_one[:, 2], label="Ground truth z")
    plt.plot(t_np, pred[:, 2], "--", label="Predicted z")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Ground Truth vs Predicted Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = output_dir / "ground_truth_vs_prediction_position.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved position comparison plot to {path}")

    # Velocity comparison plot
    plt.figure(figsize=(10, 6))

    plt.plot(t_np, target_one[:, 3], label="Ground truth vx")
    plt.plot(t_np, pred[:, 3], "--", label="Predicted vx")

    plt.plot(t_np, target_one[:, 4], label="Ground truth vy")
    plt.plot(t_np, pred[:, 4], "--", label="Predicted vy")

    plt.plot(t_np, target_one[:, 5], label="Ground truth vz")
    plt.plot(t_np, pred[:, 5], "--", label="Predicted vz")

    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Ground Truth vs Predicted Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = output_dir / "ground_truth_vs_prediction_velocity.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved velocity comparison plot to {path}")

    # MSE over time plot
    mse_t = np.mean((pred - target_one) ** 2, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(t_np, mse_t, label="Trajectory MSE")
    plt.xlabel("Time [s]")
    plt.ylabel("MSE")
    plt.title("Prediction MSE Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = output_dir / "prediction_mse_over_time.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved prediction MSE-over-time plot to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=pathlib.Path,
        required=True,
        help="Path to Kienzle .npz/.npy file or folder",
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("models/prediction/neural_ode.pt"),
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-timesteps", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument(
        "--plots-dir",
        type=pathlib.Path,
        default=pathlib.Path("models/prediction/plots"),
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    if not (0.0 < args.val_split < 1.0):
        raise ValueError("--val-split must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    s0, times, targets = load_data(
        args.data,
        min_timesteps=args.min_timesteps,
    )

    train_loader, val_loader = make_train_val_loaders(
        s0=s0,
        times=times,
        targets=targets,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    func = ODEFunc(hidden=args.hidden)

    history = train(
        func=func,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_path=args.output,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    history_path = args.output.parent / "neural_ode_history.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")

    plot_loss_curves(
        history=history,
        output_dir=args.plots_dir,
    )

    plot_mse_curves(
        history=history,
        output_dir=args.plots_dir,
    )

    plot_ground_truth_vs_prediction(
        func=func,
        loader=val_loader,
        device=args.device,
        output_dir=args.plots_dir,
    )

    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
