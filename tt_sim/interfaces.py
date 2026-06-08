"""Abstract interfaces for tt-sim pipeline stages."""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class BallObservation:
    position: np.ndarray   # (3,) xyz in meters
    sigma: float           # position uncertainty in meters
    timestamp: float       # seconds


@dataclass
class BallState:
    position: np.ndarray   # (3,)
    velocity: np.ndarray   # (3,)
    covariance: np.ndarray | None = None  # (6,6) optional


@dataclass
class SpinEstimate:
    omega: np.ndarray      # (3,) angular velocity rad/s
    confidence: np.ndarray  # (3,) per-axis confidence [0,1]


@dataclass
class PaddleTarget:
    position: np.ndarray   # (3,) desired paddle center
    normal: np.ndarray     # (3,) desired paddle face normal
    velocity: np.ndarray   # (3,) desired paddle velocity at contact
    t_contact: float       # time of contact (seconds)


@dataclass
class JointTrajectory:
    times: np.ndarray      # (N,) timestamps
    positions: np.ndarray  # (N, n_joints) joint angles
    velocities: np.ndarray | None = None  # (N, n_joints) optional


# ── ABCs ─────────────────────────────────────────────────────────────────────

class Perceiver(ABC):
    """Extracts ball observations from environment state."""

    @abstractmethod
    def observe(self, env_state: dict) -> BallObservation: ...


class Predictor(ABC):
    """Predicts future ball state from observations."""

    @abstractmethod
    def predict(self, observations: list[BallObservation], t_future: float) -> BallState: ...


class SpinEstimator(ABC):
    """Estimates ball spin from observations."""

    @abstractmethod
    def estimate(self, observations: list[BallObservation]) -> SpinEstimate: ...


class Aimer(ABC):
    """Computes desired paddle state to return the ball."""

    @abstractmethod
    def aim(self, ball: BallState, spin: SpinEstimate) -> PaddleTarget: ...


class SwingPlanner(ABC):
    """Plans joint trajectory to achieve paddle target."""

    @abstractmethod
    def plan(self, target: PaddleTarget, q_current: np.ndarray) -> JointTrajectory: ...


class HighLevelController(ABC):
    """Orchestrates perception->prediction->spin->aim->swing pipeline."""

    @abstractmethod
    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray: ...
