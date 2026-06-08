import pytest
import numpy as np

from tt_sim.interfaces import HighLevelController
from tt_sim.registry import load


MODULE = "tt_sim.upgrades.control"
CLASS_NAME = "ReplanController"
FLAG = "replan"


def _get_cls():
    import importlib
    mod = importlib.import_module(MODULE)
    return getattr(mod, CLASS_NAME)


def test_import():
    cls = _get_cls()
    assert cls is not None


def test_subclass():
    cls = _get_cls()
    assert issubclass(cls, HighLevelController)


def test_registry():
    cls = load("control", FLAG)
    assert cls is not None
