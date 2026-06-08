import pytest
import numpy as np

from tt_sim.interfaces import Perceiver
from tt_sim.registry import load


MODULE = "tt_sim.upgrades.perception"
CLASS_NAME = "HSVPerceiver"
FLAG = "hsv"


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, Perceiver)


def test_not_implemented():
    cls = _get_cls()
    instance = cls()
    env_state = {"ball_pos": np.zeros(3), "timestamp": 0.0}
    with pytest.raises(NotImplementedError):
        instance.observe(env_state)


def test_registry():
    cls = load("perceiver", FLAG)
    assert cls is not None
