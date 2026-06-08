"""Baseline aiming strategy."""

from tt_sim.interfaces import Aimer, BallState, SpinEstimate, PaddleTarget
import numpy as np

# Standard table tennis table dimensions (meters)
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
TABLE_HEIGHT = 0.76
NET_Y = 0.0  # net is at center of table along y-axis


class FaceNetAimer(Aimer):
    """Stage 0 aiming: place paddle at contact point, face the net."""

    def __init__(self, robot_side_y: float = -TABLE_LENGTH / 2):
        """robot_side_y: y-coordinate of the robot's side of the table."""
        self.robot_side_y = robot_side_y

    def aim(self, ball: BallState, spin: SpinEstimate) -> PaddleTarget:
        contact_pos = ball.position.copy()

        # Normal points toward net center (~16cm above table)
        net_center = np.array([0.0, NET_Y, TABLE_HEIGHT + 0.16])
        direction = net_center - contact_pos
        norm = np.linalg.norm(direction)
        normal = direction / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])

        velocity = np.zeros(3)

        # Estimate time to contact from ball velocity
        # If ball is moving toward robot, estimate when it arrives
        if ball.velocity is not None and np.linalg.norm(ball.velocity) > 0.1:
            # Rough estimate: distance / speed
            dist = np.linalg.norm(contact_pos)
            speed = np.linalg.norm(ball.velocity)
            t_contact = max(dist / speed, 0.1)
        else:
            t_contact = 0.5  # default 500ms

        return PaddleTarget(
            position=contact_pos,
            normal=normal,
            velocity=velocity,
            t_contact=t_contact,
        )
