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
    """Load Kienzle 50k and return input/target tensors."""
    data = np.load(path)
    # TODO: extract (state_t, accel_residual) pairs from trajectories
    raise NotImplementedError("Implement data loading and pair extraction")


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
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to Kienzle 50k .npz")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("models/prediction/residual_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
