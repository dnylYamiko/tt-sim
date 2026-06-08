import pytest
import numpy as np

from tt_sim.interfaces import BallState, SpinEstimate, Aimer
from tt_sim.registry import load


MODULE = "tt_sim.upgrades.aiming"
CLASS_NAME = "SpecularAimer"
FLAG = "specular"


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def _make_spin():
    return SpinEstimate(omega=np.zeros(3), confidence=np.zeros(3))


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, Aimer)


def test_registry():
    cls = load("aimer", FLAG)
    assert cls is not None


def test_normal_is_unit_vector():
    aimer = _get_cls()()
    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, -3.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    assert abs(np.linalg.norm(result.normal) - 1.0) < 1e-6


def test_position_matches_ball():
    aimer = _get_cls()()
    pos = np.array([0.1, -0.5, 0.9])
    ball = BallState(position=pos, velocity=np.array([0.0, -2.0, 0.5]))
    result = aimer.aim(ball, _make_spin())
    np.testing.assert_allclose(result.position, pos)


def test_velocity_has_follow_through():
    """Paddle velocity should be nonzero along the normal for a moving ball."""
    aimer = _get_cls()()
    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, -3.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    # Velocity should be along the normal direction
    assert np.linalg.norm(result.velocity) > 0.1, "Paddle should have follow-through velocity"
    # Velocity direction should align with normal
    v_dir = result.velocity / np.linalg.norm(result.velocity)
    cos_angle = np.dot(v_dir, result.normal)
    assert cos_angle > 0.99, f"Velocity not aligned with normal: cos={cos_angle}"


def test_velocity_zero_for_stationary_ball():
    """Zero-velocity ball should produce zero paddle velocity."""
    aimer = _get_cls()()
    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, 0.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    np.testing.assert_allclose(result.velocity, np.zeros(3), atol=1e-10)


def test_t_contact_positive():
    aimer = _get_cls()()
    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, -3.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    assert result.t_contact > 0


def test_reflection_geometry():
    """The normal should bisect v_in and v_out (reflection law)."""
    aimer = _get_cls()()
    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, -3.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    n = result.normal

    # Reflect v_in about n: v_ref = v_in - 2*(v_in . n)*n
    v_in = ball.velocity
    v_ref = v_in - 2 * np.dot(v_in, n) * n

    # v_ref should point in the same direction as v_out
    to_target = aimer.target - ball.position
    v_out_dir = to_target / np.linalg.norm(to_target)

    # Reflected direction should align with v_out direction
    v_ref_dir = v_ref / np.linalg.norm(v_ref)
    cos_angle = np.dot(v_ref_dir, v_out_dir)
    assert cos_angle > 0.99, f"Reflected direction misaligned: cos={cos_angle}"


def test_custom_target():
    custom = np.array([0.5, 0.5, 0.76])
    aimer = _get_cls()(target=custom)
    np.testing.assert_allclose(aimer.target, custom)

    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, -3.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    assert abs(np.linalg.norm(result.normal) - 1.0) < 1e-6


def test_slow_ball_fallback():
    """Near-zero velocity ball should not crash and should return valid output."""
    aimer = _get_cls()()
    ball = BallState(
        position=np.array([0.0, -1.0, 0.8]),
        velocity=np.array([0.0, 0.0, 0.0]),
    )
    result = aimer.aim(ball, _make_spin())
    assert result.t_contact == 0.5
    assert abs(np.linalg.norm(result.normal) - 1.0) < 1e-6
