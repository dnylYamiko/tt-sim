"""Train Neural ODE predictor on Kienzle 120k dataset.

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

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None  # will fail at runtime with a clear message


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


def load_data(path: pathlib.Path):
    """Load Kienzle 120k and return (initial_states, times, target_states)."""
    data = np.load(path)
    # TODO: extract initial conditions and full trajectories
    raise NotImplementedError("Implement data loading")


def train(func, loader, epochs, lr, device):
    """Training loop using torchdiffeq odeint."""
    if odeint is None:
        raise ImportError("Install torchdiffeq: pip install torchdiffeq")

    func.to(device)
    optimiser = torch.optim.Adam(func.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for s0, times, s_target in loader:
            s0, times, s_target = s0.to(device), times.to(device), s_target.to(device)
            # Integrate ODE from s0 over times
            # pred = odeint(func, s0, times)  # shape depends on batching strategy
            # loss = criterion(pred, s_target)
            raise NotImplementedError("Implement ODE integration and loss")
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * s0.size(0)
        avg = total_loss / len(loader.dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={avg:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to Kienzle 120k .npz")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("models/prediction/neural_ode.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    s0, times, targets = load_data(args.data)
    dataset = TensorDataset(s0, times, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    func = ODEFunc()
    train(func, loader, args.epochs, args.lr, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(func.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
