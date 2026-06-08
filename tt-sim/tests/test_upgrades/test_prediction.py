import pytest
import numpy as np

from tt_sim.interfaces import BallObservation, BallState
from tt_sim.registry import load


PREDICTORS = [
    ("tt_sim.upgrades.prediction", "DragBouncePredictor", "drag_bounce"),
    ("tt_sim.upgrades.prediction", "ResidualMLPPredictor", "residual_mlp"),
    ("tt_sim.upgrades.prediction", "NeuralODEPredictor", "neural_ode"),
    ("tt_sim.upgrades.prediction", "GPPredictor", "gp"),
    ("tt_sim.upgrades.prediction", "EnsemblePredictor", "ensemble"),
]


def _dummy_observations():
    return [BallObservation(position=np.zeros(3), sigma=0.01, timestamp=0.0)]


@pytest.mark.parametrize("module_path,class_name,flag", PREDICTORS)
def test_import(module_path, class_name, flag):
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None


@pytest.mark.parametrize("module_path,class_name,flag", PREDICTORS)
def test_subclass(module_path, class_name, flag):
    import importlib
    from tt_sim.interfaces import Predictor
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert issubclass(cls, Predictor)


@pytest.mark.parametrize("module_path,class_name,flag", PREDICTORS)
def test_not_implemented(module_path, class_name, flag):
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    instance = cls()
    with pytest.raises(NotImplementedError):
        instance.predict(_dummy_observations(), t_future=0.5)


@pytest.mark.parametrize("module_path,class_name,flag", PREDICTORS)
def test_registry(module_path, class_name, flag):
    cls = load("predictor", flag)
    assert cls is not None
