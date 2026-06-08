"""Perception upgrades for tt-sim."""

from __future__ import annotations

import numpy as np

from tt_sim.interfaces import (
    BallObservation,
    Perceiver,
)


# ── Physics constants ────────────────────────────────────────────────────────

BALL_RADIUS: float = 0.02  # m, known ball size for mono depth estimation


class HSVPerceiver(Perceiver):
    """Stage 1 – HSV colour-threshold ball perceiver.

    Algorithm
    ---------
    Detects the orange table-tennis ball in an RGB image by converting to
    **HSV colour space** and applying fixed thresholds.  The largest
    contour in the resulting binary mask is taken as the ball; its centroid
    gives the 2-D pixel location.  Depth is estimated monocularly from the
    **known physical ball diameter** and the apparent pixel diameter.

    Mathematics
    -----------
    HSV thresholds (OpenCV 0-180 H scale):

        H ∈ [5, 25],  S ∈ [100, 255],  V ∈ [100, 255]

    Mono depth from known ball radius *R* and apparent pixel radius *r*:

        Z = (f * R) / r

    where *f* is the camera focal length in pixels.

    No uncertainty is reported (``sigma = 0.0``).

    Implementation
    --------------
    * ``cv2.cvtColor`` → ``cv2.inRange`` → ``cv2.findContours``.
    * ``cv2.moments`` for sub-pixel centroid.
    * Depth from pinhole camera model with known ball size.

    References
    ----------
    * Standard OpenCV colour segmentation pipeline.

    Getting Started
    ---------------
    1. Instantiate with camera intrinsics (focal length, principal point).
    2. Pass ``env_state`` dict containing an ``"image"`` key (H×W×3 uint8).
    3. Returns ``BallObservation`` with 3-D position and ``sigma=0.0``.
    """

    # HSV thresholds (OpenCV convention: H 0-180)
    HSV_LOW: np.ndarray = np.array([5, 100, 100], dtype=np.uint8)
    HSV_HIGH: np.ndarray = np.array([25, 255, 255], dtype=np.uint8)

    def observe(self, env_state: dict) -> BallObservation:
        raise NotImplementedError("HSVPerceiver.observe")
