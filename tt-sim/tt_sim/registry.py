"""Registry for discovering and retrieving pipeline implementations."""

import importlib

REGISTRY = {
    "perceiver": {
        "sim": "tt_sim.baselines.perception.SimPerceiver",
        "hsv": "tt_sim.upgrades.perception.HSVPerceiver",
    },
    "predictor": {
        "ballistic": "tt_sim.baselines.prediction.BallisticPredictor",
        "drag_bounce": "tt_sim.upgrades.prediction.DragBouncePredictor",
        "residual_mlp": "tt_sim.upgrades.prediction.ResidualMLPPredictor",
        "neural_ode": "tt_sim.upgrades.prediction.NeuralODEPredictor",
        "gp": "tt_sim.upgrades.prediction.GPPredictor",
        "ensemble": "tt_sim.upgrades.prediction.EnsemblePredictor",
    },
    "spin": {
        "zero": "tt_sim.baselines.spin.ZeroSpinEstimator",
        "magnus": "tt_sim.upgrades.spin.MagnusSpinEstimator",
    },
    "aimer": {
        "face_net": "tt_sim.baselines.aiming.FaceNetAimer",
        "specular": "tt_sim.upgrades.aiming.SpecularAimer",
    },
    "swing": {
        "lerp": "tt_sim.baselines.swing.LerpSwingPlanner",
        "mujoco_ik": "tt_sim.baselines.swing.MujocoIKSwingPlanner",
        "quintic": "tt_sim.upgrades.swing.QuinticSwingPlanner",
    },
    "control": {
        "open_loop": "tt_sim.baselines.control.OpenLoopController",
        "reactive": "tt_sim.baselines.control.ReactiveController",
        "replan": "tt_sim.upgrades.control.ReplanController",
    },
}

DEFAULTS = {
    "perceiver": "sim",
    "predictor": "ballistic",
    "spin": "zero",
    "aimer": "face_net",
    "swing": "lerp",
    "control": "open_loop",
}


def load(subsystem: str, name: str) -> type:
    """Load a class from the registry by subsystem and flag name."""
    if subsystem not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown subsystem '{subsystem}'. Available: {available}"
        )

    implementations = REGISTRY[subsystem]
    if name not in implementations:
        available = ", ".join(sorted(implementations))
        raise KeyError(
            f"Unknown implementation '{name}' for {subsystem}. Available: {available}"
        )

    dotted_path = implementations[name]
    module_path, class_name = dotted_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Implementation '{name}' for {subsystem} not yet available. "
            f"Install or implement {dotted_path}"
        ) from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Implementation '{name}' for {subsystem} not yet available. "
            f"Install or implement {dotted_path}"
        ) from e

    return cls


def list_available() -> None:
    """Print all subsystems and their available implementations."""
    for subsystem, implementations in REGISTRY.items():
        default = DEFAULTS.get(subsystem)
        print(f"\n{subsystem}:")
        for name in sorted(implementations):
            marker = " (default)" if name == default else ""
            print(f"  {name}{marker} -> {implementations[name]}")
