"""Train Gaussian Process predictor on Kienzle 50k dataset.

What to train
-------------
The GPPredictor uses a Gaussian Process to model the residual acceleration
(correction over ballistic baseline).  We use an exact GP with an RBF kernel
from gpytorch.  Because exact GPs are O(N^3), training uses a subset of
inducing points or a variational approximation for scalability.

Data
----
Kienzle 50k dataset: 50 000 trajectories stored as NumPy .npz with keys
``states`` (N, T, 6) and ``dt`` (scalar).  Pass the path via ``--data``.
State-acceleration residual pairs are extracted the same way as the residual
MLP trainer.

Expected output
---------------
PyTorch checkpoint (gpytorch model state dict) saved to
``models/prediction/gp.pt``.
"""

import argparse
import pathlib

import numpy as np
import torch

try:
    import gpytorch
except ImportError:
    gpytorch = None


class ExactGPModel(gpytorch.models.ExactGP if gpytorch else object):
    """Exact GP with RBF kernel for residual acceleration prediction."""

    def __init__(self, train_x, train_y, likelihood):
        if gpytorch is None:
            raise ImportError("Install gpytorch: pip install gpytorch")
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def load_data(path: pathlib.Path):
    """Load Kienzle 50k and return (inputs, targets) tensors."""
    data = np.load(path)
    # TODO: extract (state, residual_accel) pairs
    raise NotImplementedError("Implement data loading and pair extraction")


def train(model, likelihood, train_x, train_y, epochs, lr):
    """gpytorch exact GP training loop."""
    if gpytorch is None:
        raise ImportError("Install gpytorch: pip install gpytorch")

    model.train()
    likelihood.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(epochs):
        optimiser.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimiser.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  mll={-loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to Kienzle 50k .npz")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("models/prediction/gp.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-points", type=int, default=2000, help="Subsample for exact GP tractability")
    args = parser.parse_args()

    if gpytorch is None:
        raise ImportError("Install gpytorch: pip install gpytorch")

    train_x, train_y = load_data(args.data)

    # Subsample if needed
    if train_x.size(0) > args.max_points:
        idx = torch.randperm(train_x.size(0))[: args.max_points]
        train_x, train_y = train_x[idx], train_y[idx]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    train(model, likelihood, train_x, train_y, args.epochs, args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved GP model to {args.output}")


if __name__ == "__main__":
    main()
