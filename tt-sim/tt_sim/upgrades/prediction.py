"""Trajectory prediction upgrades for tt-sim."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

from tt_sim.interfaces import (
    BallObservation,
    BallState,
    Predictor,
)


# ── Physics constants ────────────────────────────────────────────────────────

C_D: float = 0.4          # drag coefficient (sphere in turbulent regime)
RHO_AIR: float = 1.225    # kg/m^3, air density at sea level
BALL_MASS: float = 0.0027 # kg, ITTF regulation ball mass
BALL_RADIUS: float = 0.02 # m, ITTF regulation ball radius
BALL_AREA: float = np.pi * BALL_RADIUS ** 2  # m^2, cross-sectional area
RESTITUTION: float = 0.91 # coefficient of restitution (ball–table)
C_L: float = 0.55         # lift (Magnus) coefficient
GRAVITY: np.ndarray = np.array([0.0, 0.0, -9.81])


class DragBouncePredictor(Predictor):
    """Stage 1 – Analytic gravity + aerodynamic drag predictor.

    Algorithm
    ---------
    Integrates the ODE for a sphere under gravity and quadratic drag::

        dv/dt = g - (C_D * rho_air * A) / (2 * m) * |v| * v

    where *A* is the cross-sectional area of the ball.  Table bounces are
    detected via event handling (z ≤ 0) and resolved with the coefficient
    of restitution (e = 0.91).

    The drag coefficient C_D can optionally be fit from trajectory data
    using least-squares minimisation of position residuals.

    Mathematics
    -----------
    State vector y = [x, v] ∈ ℝ⁶.

    dy/dt = [v,  g - (C_D ρ A / 2m) |v| v]

    Bounce condition:  z(t) = 0  →  v_z ← -e * v_z

    Implementation
    --------------
    * ``scipy.integrate.solve_ivp`` with RK45 and dense output.
    * Terminal event for table-plane crossing.

    References
    ----------
    * Huang, Y. et al. (2011). "Trajectory prediction of spinning ball for
      ping-pong player robot." *IEEE/RSJ IROS*.

    Datasets
    --------
    Fit / validate with any recorded ball-trajectory dataset (≥ 50 samples).

    Getting Started
    ---------------
    1. Instantiate with default constants or supply measured C_D.
    2. Call ``predict(observations, t_future)`` with ≥ 2 observations.
    3. Internally performs linear velocity bootstrap then integrates forward.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("DragBouncePredictor.predict")


class ResidualMLPPredictor(Predictor):
    """Stage 2a – Physics-informed residual MLP predictor.

    Algorithm
    ---------
    Augments the analytic drag ODE with a learned residual from a small
    multi-layer perceptron (MLP)::

        dv/dt = physics(x, v) + f_θ(x, v)

    where *physics* is the gravity + drag term and *f_θ* is a 2-layer MLP
    (hidden dims 64) with ReLU activations trained to minimise position
    prediction error over recorded trajectories.

    Mathematics
    -----------
    dy/dt = [v,  g - k_drag |v| v + f_θ(x, v)]

    Loss = Σ_i || x̂(t_i) - x(t_i) ||²

    Implementation
    --------------
    * PyTorch for the MLP.
    * ``torchdiffeq.odeint`` for differentiable ODE integration during
      training.

    References
    ----------
    * Achterhold, J. et al. (2023). "Physics-informed residual learning for
      table tennis trajectory prediction." *L4DC*.

    Datasets
    --------
    Kienzle 50 k trajectory dataset (position + timestamp).

    Getting Started
    ---------------
    1. Pre-train by loading a ``DragBouncePredictor`` as the physics prior.
    2. Train the MLP residual on recorded data via ``train()`` method.
    3. Call ``predict()`` for combined physics + learned prediction.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("ResidualMLPPredictor.predict")

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
    
class NeuralODEPredictor(Predictor):
    """Stage 2b – Fully learned Neural ODE predictor.

    Algorithm
    ---------
    Learns the full dynamics from data without an explicit physics prior::

        dx/dt = f_θ(x)

    where *f_θ* is a 3-layer MLP with 64 hidden units per layer and
    SiLU/Swish activations.  Training uses the adjoint sensitivity method
    for memory-efficient backpropagation through the ODE solver.

    Mathematics
    -----------
    State y = [x, v] ∈ ℝ⁶.

    dy/dt = f_θ(y),   f_θ : ℝ⁶ → ℝ⁶

    Loss = Σ_i || ŷ(t_i) - y(t_i) ||²

    Implementation
    --------------
    * ``torchdiffeq.odeint_adjoint`` for training.
    * 3-layer MLP, 64 hidden units, SiLU activation.

    References
    ----------
    * Chen, R. T. Q. et al. (2018). "Neural Ordinary Differential
      Equations." *NeurIPS*.
    * Rubanova, Y. et al. (2019). "Latent ODEs for Irregularly-Sampled
      Time Series." *NeurIPS*.

    Datasets
    --------
    Kienzle 120 k trajectory dataset (full state: position + velocity).

    Getting Started
    ---------------
    1. Prepare dataset of (t, y) pairs.
    2. Train with ``odeint_adjoint`` and Adam optimiser, lr=1e-3.
    3. Call ``predict()`` — internally integrates the learned ODE.
    
    """


class NeuralODEPredictor(Predictor):
    """Fully learned Neural ODE predictor."""

    def __init__(
        self,
        checkpoint_path: str = "models/prediction/neural_ode.pt",
        device: str | None = None,
    ):
        if odeint is None:
            raise ImportError("Install torchdiffeq first: pip install torchdiffeq")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.func = ODEFunc(hidden=128).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            self.func.load_state_dict(state_dict["model_state_dict"])
        else:
            self.func.load_state_dict(state_dict)
        self.func.eval()

    def _observations_to_state(observations):
        """
        Convert position-only BallObservation objects into:
            [x, y, z, vx, vy, vz]

        BallObservation fields:
            position: np.ndarray shape (3,)
            timestamp: float
        """

        if len(observations) < 2:
            raise ValueError(
                "NeuralODEPredictor requires at least two observations because "
                "BallObservation contains position only, not velocity."
            )

        obs_prev = observations[-2]
        obs_curr = observations[-1]

        p_prev = np.asarray(obs_prev.position, dtype=np.float32)
        p_curr = np.asarray(obs_curr.position, dtype=np.float32)

        t_prev = float(obs_prev.timestamp)
        t_curr = float(obs_curr.timestamp)

        dt = t_curr - t_prev

        if dt <= 0:
            raise ValueError(
                f"Observation timestamps must increase. Got dt={dt}."
            )

        v_curr = (p_curr - p_prev) / dt

        state = np.concatenate([p_curr, v_curr]).astype(np.float32)

        return state
    def predict(
    self,
    observations: list[BallObservation],
    t_future: float,
) -> BallState:

        if not observations:
            raise ValueError("At least one observation is required.")

        if len(observations) < 2:
            raise ValueError(
                "NeuralODEPredictor requires at least two observations because "
                "BallObservation only contains position and timestamp."
            )

        if t_future < 0:
            raise ValueError("t_future must be non-negative.")

        obs_prev = observations[-2]
        obs_curr = observations[-1]

        p_prev = np.asarray(obs_prev.position, dtype=np.float32)
        p_curr = np.asarray(obs_curr.position, dtype=np.float32)

        t_prev = float(obs_prev.timestamp)
        t_curr = float(obs_curr.timestamp)

        dt = t_curr - t_prev

        if dt <= 0:
            raise ValueError(f"Observation timestamps must increase. Got dt={dt}.")

        v_curr = (p_curr - p_prev) / dt

        state = np.concatenate([p_curr, v_curr]).astype(np.float32)

        y0 = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        t_eval = torch.tensor(
            [0.0, float(t_future)],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            pred = odeint(
                self.func,
                y0,
                t_eval,
                method="dopri5",
                rtol=1e-5,
                atol=1e-6,
            )

        final_state = pred[-1, 0].detach().cpu().numpy()

        if not np.isfinite(final_state).all():
            raise RuntimeError(f"Neural ODE produced invalid state: {final_state}")

        final_state[:3] = np.clip(final_state[:3], [-5.0, -5.0, 0.0], [5.0, 5.0, 5.0])
        final_state[3:] = np.clip(final_state[3:], [-20.0, -20.0, -20.0], [20.0, 20.0, 20.0])

        x, y, z, vx, vy, vz = final_state

        return BallState(
            position=np.array([x, y, z], dtype=np.float32),
            velocity=np.array([vx, vy, vz], dtype=np.float32),
            timestamp=float(obs_curr.timestamp + t_future),
        ) 
    """def predict(
    self,
    observations: list[BallObservation],
    t_future: float,
) -> BallState:

        if not observations:
            raise ValueError("At least one observation is required.")

        if t_future < 0:
            raise ValueError("t_future must be non-negative.")

        obs = observations[-1]

        def _obs_to_state(obs):
            #Convert BallObservation to [x, y, z, vx, vy, vz]. Supports several common layouts.

            # Show useful debug info if conversion fails
            def fail():
                raise AttributeError(
                    "Could not convert BallObservation to [x, y, z, vx, vy, vz]. "
                    f"type={type(obs)}, "
                    f"dict={getattr(obs, '__dict__', None)}, "
                    f"fields={[a for a in dir(obs) if not a.startswith('_')]}"
                )

            # Case 1: direct scalar fields
            direct_names = ["x", "y", "z", "vx", "vy", "vz"]
            if all(hasattr(obs, name) for name in direct_names):
                return [
                    getattr(obs, "x"),
                    getattr(obs, "y"),
                    getattr(obs, "z"),
                    getattr(obs, "vx"),
                    getattr(obs, "vy"),
                    getattr(obs, "vz"),
                ]

            # Case 2: position/velocity vectors
            if hasattr(obs, "position") and hasattr(obs, "velocity"):
                pos = getattr(obs, "position")
                vel = getattr(obs, "velocity")
                return [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]

            # Case 3: pos/vel vectors
            if hasattr(obs, "pos") and hasattr(obs, "vel"):
                pos = getattr(obs, "pos")
                vel = getattr(obs, "vel")
                return [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]

            # Case 4: p/v vectors
            if hasattr(obs, "p") and hasattr(obs, "v"):
                pos = getattr(obs, "p")
                vel = getattr(obs, "v")
                return [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]

            # Case 5: state vector already exists
            if hasattr(obs, "state"):
                state = getattr(obs, "state")
                if len(state) >= 6:
                    return [state[0], state[1], state[2], state[3], state[4], state[5]]

            # Case 6: observation vector already exists
            if hasattr(obs, "observation"):
                state = getattr(obs, "observation")
                if len(state) >= 6:
                    return [state[0], state[1], state[2], state[3], state[4], state[5]]

            # Case 7: numpy-like/list/tuple observation
            try:
                if len(obs) >= 6:
                    return [obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]]
            except TypeError:
                pass

            # Case 8: dictionary observation
            if isinstance(obs, dict):
                if all(k in obs for k in direct_names):
                    return [obs["x"], obs["y"], obs["z"], obs["vx"], obs["vy"], obs["vz"]]

                if "position" in obs and "velocity" in obs:
                    pos = obs["position"]
                    vel = obs["velocity"]
                    return [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]

                if "state" in obs:
                    state = obs["state"]
                    return [state[0], state[1], state[2], state[3], state[4], state[5]]

            fail()
   

        state = _obs_to_state(obs)

        y0 = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        t_eval = torch.tensor(
            [0.0, float(t_future)],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            pred = odeint(
                self.func,
                y0,
                t_eval,
                method="dopri5",
                rtol=1e-5,
                atol=1e-6,
            )

        final_state = pred[-1, 0].detach().cpu().numpy()

        x, y, z, vx, vy, vz = final_state

        return BallState(
            position=[float(x), float(y), float(z)],
            velocity=[float(vx), float(vy), float(vz)],
        )"""

class GPPredictor(Predictor):
    """Stage 2c – Gaussian Process predictor with physics mean function.

    Algorithm
    ---------
    Places a GP prior over trajectory residuals with respect to a physics
    mean function (gravity + drag).  The kernel is a scaled RBF
    (squared-exponential) with automatic relevance determination (ARD).

    Posterior predictions yield both a mean trajectory and calibrated
    uncertainty (covariance) over future ball states.

    Mathematics
    -----------
    f(t) ~ GP( m(t), k(t, t') )

    m(t) = physics_prediction(t)   (drag + gravity)

    k(t, t') = σ² exp( -||t - t'||² / (2 ℓ²) )

    Posterior:  p(f* | X, y) = N( μ*, Σ* )

    Implementation
    --------------
    * ``gpytorch`` exact GP with ``ScaleKernel(RBFKernel(ard_num_dims=…))``.
    * Physics mean via ``gpytorch.means.ConstantMean`` replaced with custom
      ``PhysicsMean``.

    References
    ----------
    * Deisenroth, M. P. & Rasmussen, C. E. (2011). "PILCO: A Model-Based
      and Data-Efficient Approach to Policy Search." *ICML*.

    Datasets
    --------
    Kienzle 50 k trajectory dataset.

    Getting Started
    ---------------
    1. Fit hyperparameters on training trajectories via marginal likelihood.
    2. Call ``predict()`` — returns ``BallState`` with covariance populated.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("GPPredictor.predict")


class EnsemblePredictor(Predictor):
    """Stage 3 – Deep ensemble predictor.

    Algorithm
    ---------
    Maintains 3–5 ``ResidualMLPPredictor`` instances, each trained with a
    different random seed and/or data shuffle.  At inference time the
    ensemble mean serves as the point prediction and the ensemble variance
    provides a calibrated uncertainty estimate.

    Mathematics
    -----------
    μ_ens = (1/K) Σ_k μ_k

    σ²_ens = (1/K) Σ_k (σ²_k + μ²_k) - μ²_ens

    Implementation
    --------------
    * Wraps a list of ``ResidualMLPPredictor`` members.
    * Aggregates predictions via NumPy mean / variance.

    References
    ----------
    * Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive
      Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

    Getting Started
    ---------------
    1. Train K=5 ``ResidualMLPPredictor`` instances with different seeds.
    2. Pass them to the constructor.
    3. ``predict()`` returns mean state with covariance from ensemble spread.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("EnsemblePredictor.predict")
