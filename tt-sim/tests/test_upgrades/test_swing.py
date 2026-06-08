import pytest
import numpy as np

from tt_sim.interfaces import PaddleTarget, SwingPlanner
from tt_sim.registry import load


MODULE = "tt_sim.upgrades.swing"
CLASS_NAME = "QuinticSwingPlanner"
FLAG = "quintic"


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, SwingPlanner)


def test_not_implemented():
    cls = _get_cls()
    instance = cls()
    target = PaddleTarget(
        position=np.zeros(3),
        normal=np.array([1, 0, 0]),
        velocity=np.zeros(3),
        t_contact=0.5,
    )
    with pytest.raises(NotImplementedError):
        instance.plan(target, q_current=np.zeros(7))


def test_registry():
    cls = load("swing", FLAG)
    assert cls is not None
