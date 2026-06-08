"""Baseline spin estimation."""

from tt_sim.interfaces import SpinEstimator, BallObservation, SpinEstimate
import numpy as np
from typing import List


class ZeroSpinEstimator(SpinEstimator):
    """Stage 0 spin estimation: assumes no spin."""

    def estimate(self, observations: List[BallObservation]) -> SpinEstimate:
        return SpinEstimate(
            omega=np.zeros(3),
            confidence=np.zeros(3),  # zero confidence = we know nothing
        )
