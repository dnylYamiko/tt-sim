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


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, Aimer)


def test_not_implemented():
    cls = _get_cls()
    instance = cls()
    ball = BallState(position=np.zeros(3), velocity=np.zeros(3))
    spin = SpinEstimate(omega=np.zeros(3), confidence=np.zeros(3))
    with pytest.raises(NotImplementedError):
        instance.aim(ball, spin)


def test_registry():
    cls = load("aimer", FLAG)
    assert cls is not None
