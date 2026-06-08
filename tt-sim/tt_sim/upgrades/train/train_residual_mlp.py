"""Train residual MLP predictor on Kienzle 50k dataset.

What to train
-------------
The ResidualMLPPredictor learns a residual correction on top of the ballistic
baseline.  The MLP takes the current state (position + velocity, 6-D) and
outputs a 3-D acceleration correction that is added to gravity before
integration.  Training minimises the MSE between predicted and ground-truth
future positions over a short horizon.

Data
----
Kienzle 50k dataset: 50 000 recorded table-tennis trajectories stored as
NumPy .npz with keys ``states`` (N, T, 6) and ``dt`` (scalar).
Pass the path via ``--data``.

Expected output
---------------
PyTorch checkpoint saved to ``models/prediction/residual_mlp.pt``.
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualMLP(nn.Module):
    """Small MLP that predicts residual acceleration."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, state):
        return self.net(state)

def load_data(path: pathlib.Path):
    """Load .npz Kienzle files or .npy trajectory files from a file/folder/subfolders."""

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(list(path.rglob("*.npz")) + list(path.rglob("*.npy")))
        print(f"Searching under: {path}")
        print(f"Found {len(files)} candidate .npz/.npy files")
        for f in files[:20]:
            print("  ", f)
    else:
        raise FileNotFoundError(f"{path} does not exist")

    if not files:
        raise FileNotFoundError(f"No .npz or .npy files found under {path}")

    all_inputs = []
    all_targets = []

    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

    for file in files:
        print(f"Loading {file}")

        if file.suffix == ".npz":
            data = np.load(file)

            if "states" not in data or "dt" not in data:
                print(f"Skipping {file}: missing 'states' or 'dt'")
                continue

            states = data["states"]
            dt = float(data["dt"])

            if states.ndim != 3 or states.shape[2] != 6:
                print(f"Skipping {file}: expected states shape (N, T, 6), got {states.shape}")
                continue

            state_t = states[:, :-1, :]
            v_t = states[:, :-1, 3:6]
            v_next = states[:, 1:, 3:6]

        elif file.suffix == ".npy":
            traj = np.load(file, allow_pickle=True)
            print(f"  npy shape: {traj.shape}")

    # Case 1: shape (N, 7)
    # [t, x, y, z, vx, vy, vz]
            if traj.ndim == 2 and traj.shape[1] == 7:
                t = traj[:, 0]
                states = traj[:, 1:7]

    # Case 2: shape (N, 6)
    # [x, y, z, vx, vy, vz], assume fixed dt
            elif traj.ndim == 2 and traj.shape[1] == 6:
                states = traj
                t = np.arange(len(states), dtype=float) * 0.01

    # Case 3: shape (N, T, 6)
    # multiple trajectories already in state format
            elif traj.ndim == 3 and traj.shape[2] == 6:
                dt = 0.01

                state_t = traj[:, :-1, :]
                v_t = traj[:, :-1, 3:6]
                v_next = traj[:, 1:, 3:6]

                a_observed = (v_next - v_t) / dt
                accel_residual = a_observed - gravity

                inputs = state_t.reshape(-1, 6)
                targets = accel_residual.reshape(-1, 3)

                valid = (
                    np.isfinite(inputs).all(axis=1)
                    & np.isfinite(targets).all(axis=1)
                )

                inputs = inputs[valid]
                targets = targets[valid]

                if len(inputs) == 0:
                    print(f"Skipping {file}: no valid samples")
                    continue

                all_inputs.append(inputs.astype(np.float32))
                all_targets.append(targets.astype(np.float32))

                print(f"  used {inputs.shape[0]} samples")
                continue

            else:
                print(
                    f"Skipping {file}: unsupported .npy shape {traj.shape}. "
                        "Expected (N,7), (N,6), or (N,T,6)."
                    )
                continue

            dt_values = np.diff(t)

            if len(dt_values) == 0:
                print(f"Skipping {file}: not enough time steps")
                continue

            dt = float(np.median(dt_values))

            if dt <= 0:
                print(f"Skipping {file}: invalid dt={dt}")
                continue

            state_t = states[:-1, :]
            v_t = states[:-1, 3:6]
            v_next = states[1:, 3:6]

        else:
            continue

        a_observed = (v_next - v_t) / dt
        accel_residual = a_observed - gravity

        inputs = state_t.reshape(-1, 6)
        targets = accel_residual.reshape(-1, 3)

        valid = (
            np.isfinite(inputs).all(axis=1)
            & np.isfinite(targets).all(axis=1)
        )

        inputs = inputs[valid]
        targets = targets[valid]

        if len(inputs) == 0:
            print(f"Skipping {file}: no valid samples")
            continue

        all_inputs.append(inputs.astype(np.float32))
        all_targets.append(targets.astype(np.float32))

        print(f"  used {inputs.shape[0]} samples")

    #if not all_inputs:
    #    raise ValueError(f"No valid training data found under {path}")
    if not all_inputs:
        raise ValueError(
        f"No valid training data found under {path}. "
        "Your .npy files were found, but their shapes were not accepted. "
        "Expected (N,7), (N,6), or (N,T,6)."
        )
    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    print(f"\nTotal files used: {len(all_inputs)}")
    print(f"Total training pairs: {inputs.shape[0]}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    return inputs, targets
#def load_data(path: pathlib.Path):
#    """Load Kienzle 50k and return input/target tensors."""
#    #data = np.load(path)
#    # TODO: extract (state_t, accel_residual) pairs from trajectories
#    #raise NotImplementedError("Implement data loading and pair extraction")
    


def train(model, loader, epochs, lr, device):
    """Standard PyTorch training loop."""
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={avg:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to Kienzle 50k .npz/.npy")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("models/prediction/residual_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
 
    print(f"Using data path: {args.data}")
    print(f"Resolved path: {args.data.resolve()}")
    print(f"Exists: {args.data.exists()}")
    print(f"Is file: {args.data.is_file()}")
    print(f"Is dir: {args.data.is_dir()}")


    inputs, targets = load_data(args.data)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ResidualMLP()
    train(model, loader, args.epochs, args.lr, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
