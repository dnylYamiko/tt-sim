"""Tests for tt_sim.registry."""

import pytest
from tt_sim.registry import load, list_available, REGISTRY, DEFAULTS


def test_load_ballistic_succeeds():
    """Baselines are implemented, so loading should return the class."""
    cls = load("predictor", "ballistic")
    assert cls.__name__ == "BallisticPredictor"


def test_list_available_runs(capsys):
    list_available()
    out = capsys.readouterr().out
    assert "predictor" in out
    assert "ballistic" in out


def test_unknown_subsystem():
    with pytest.raises(KeyError, match="Unknown subsystem"):
        load("nonexistent", "foo")


def test_unknown_name():
    with pytest.raises(KeyError, match="Unknown implementation"):
        load("predictor", "drag")


def test_defaults_match_registry():
    for subsystem, default_name in DEFAULTS.items():
        assert subsystem in REGISTRY
        assert default_name in REGISTRY[subsystem]
