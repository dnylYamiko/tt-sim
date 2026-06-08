import pytest
import numpy as np

from tt_sim.interfaces import BallObservation, SpinEstimator
from tt_sim.registry import load


MODULE = "tt_sim.upgrades.spin"
CLASS_NAME = "MagnusSpinEstimator"
FLAG = "magnus"


def _dummy_observations():
    return [BallObservation(position=np.zeros(3), sigma=0.01, timestamp=0.0)]


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, SpinEstimator)


def test_not_implemented():
    cls = _get_cls()
    instance = cls()
    with pytest.raises(NotImplementedError):
        instance.estimate(_dummy_observations())


def test_registry():
    cls = load("spin", FLAG)
    assert cls is not None
