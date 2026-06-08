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

# Predictors that are still stubs (NotImplementedError)
STUB_PREDICTORS = [p for p in PREDICTORS if p[1] != "DragBouncePredictor"]


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


@pytest.mark.parametrize("module_path,class_name,flag", STUB_PREDICTORS)
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


# ── DragBouncePredictor functional tests ─────────────────────────────────────

def _make_obs(pos, t):
    return BallObservation(position=np.array(pos, dtype=float), sigma=0.01, timestamp=t)


class TestDragBouncePredictor:
    """Functional tests for the Stage 1 drag+bounce predictor."""

    def _get_predictor(self):
        from tt_sim.upgrades.prediction import DragBouncePredictor
        return DragBouncePredictor()

    def test_returns_ball_state(self):
        pred = self._get_predictor()
        obs = [_make_obs([0, 0, 1.0], 0.0), _make_obs([0.1, 0, 0.99], 0.033)]
        result = pred.predict(obs, t_future=0.1)
        assert isinstance(result, BallState)
        assert result.position.shape == (3,)
        assert result.velocity.shape == (3,)

    def test_requires_two_observations(self):
        pred = self._get_predictor()
        with pytest.raises(ValueError):
            pred.predict([_make_obs([0, 0, 1], 0.0)], t_future=0.5)

    def test_gravity_pulls_down(self):
        """A ball released from rest should fall due to gravity."""
        pred = self._get_predictor()
        z0 = 1.5
        obs = [_make_obs([0, 0, z0], 0.0), _make_obs([0, 0, z0 - 0.001], 0.01)]
        result = pred.predict(obs, t_future=0.3)
        assert result.position[2] < z0, "Ball should have fallen"
        assert result.velocity[2] < 0, "Ball should have downward velocity"

    def test_slow_ball_approx_ballistic(self):
        """For a slow ball, drag is negligible — should be close to ballistic."""
        pred = self._get_predictor()
        # Ball moving slowly in +y at z=1.0
        obs = [
            _make_obs([0, 0, 1.0], 0.0),
            _make_obs([0, 0.01, 1.0 - 0.5 * 9.81 * 0.01**2], 0.01),
        ]
        result = pred.predict(obs, t_future=0.1)
        # Ballistic: z = 1.0 - 0.5*g*0.1^2 = 1.0 - 0.049 = 0.951
        assert abs(result.position[2] - 0.951) < 0.01

    def test_drag_slows_fast_ball(self):
        """A fast ball should arrive slower than ballistic prediction."""
        from tt_sim.baselines.prediction import BallisticPredictor

        # Fast ball moving in +y at 10 m/s
        obs = [
            _make_obs([0, 0, 1.0], 0.0),
            _make_obs([0, 0.1, 1.0 - 0.5 * 9.81 * 0.01**2], 0.01),
        ]
        t_future = 0.3

        drag_pred = self._get_predictor()
        ball_pred = BallisticPredictor()

        drag_result = drag_pred.predict(obs, t_future)
        ball_result = ball_pred.predict(obs, t_future)

        # Drag should reduce y-displacement compared to ballistic
        assert drag_result.position[1] < ball_result.position[1], \
            "Drag predictor should predict shorter y-travel than ballistic"

    def test_bounce_reverses_vz(self):
        """Ball heading down at table height should bounce up."""
        pred = self._get_predictor()
        TABLE_H = 0.76
        # Ball just above table, moving downward fast
        obs = [
            _make_obs([0, 0, TABLE_H + 0.1], 0.0),
            _make_obs([0, 0, TABLE_H + 0.05], 0.01),
        ]
        # Predict far enough that it hits the table and bounces
        result = pred.predict(obs, t_future=0.2)
        # After bounce, ball should be at or above table height
        assert result.position[2] >= TABLE_H - 0.01, \
            "Ball should have bounced off table"

    def test_multiple_observations(self):
        """More observations should produce a valid prediction."""
        pred = self._get_predictor()
        g = 9.81
        obs = []
        for i in range(10):
            t = i * 0.033
            obs.append(_make_obs([0, t * 3.0, 1.0 - 0.5 * g * t**2], t))
        result = pred.predict(obs, t_future=0.5)
        assert result.position.shape == (3,)
        # Ball should have moved forward in y
        assert result.position[1] > 0.5

    def test_configurable_no_drag(self):
        """With c_d=0, DragBouncePredictor should match ballistic (no drag)."""
        from tt_sim.baselines.prediction import BallisticPredictor
        from tt_sim.upgrades.prediction import DragBouncePredictor

        # Fast ball in +y at 10 m/s
        obs = [
            _make_obs([0, 0, 1.0], 0.0),
            _make_obs([0, 0.1, 1.0 - 0.5 * 9.81 * 0.01**2], 0.01),
        ]
        t_future = 0.2

        no_drag = DragBouncePredictor(c_d=0.0)
        ballistic = BallisticPredictor()

        nd_result = no_drag.predict(obs, t_future)
        b_result = ballistic.predict(obs, t_future)

        # y-displacement should match closely (both no drag, no bounce)
        np.testing.assert_allclose(nd_result.position[1], b_result.position[1], atol=0.01)

    def test_configurable_restitution(self):
        """Custom restitution should affect post-bounce velocity."""
        from tt_sim.upgrades.prediction import DragBouncePredictor

        TABLE_H = 0.76
        obs = [
            _make_obs([0, 0, TABLE_H + 0.1], 0.0),
            _make_obs([0, 0, TABLE_H + 0.05], 0.01),
        ]
        elastic = DragBouncePredictor(c_d=0.0, restitution=1.0)
        inelastic = DragBouncePredictor(c_d=0.0, restitution=0.5)

        r_elastic = elastic.predict(obs, t_future=0.3)
        r_inelastic = inelastic.predict(obs, t_future=0.3)

        # Elastic bounce should give higher z than inelastic
        assert r_elastic.position[2] > r_inelastic.position[2]
