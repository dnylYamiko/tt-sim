"""Data generation utilities for tt-sim.

CLI tool for generating and managing simulation data.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np

DATA_ROOT = Path("data/generated")

GENERATORS = ["trajectories", "swings", "laser_sim", "rl_episodes"]


def _output_dir(name: str) -> Path:
    d = DATA_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_manifest(directory: Path, generator: str, count: int, config: dict, files: list[str]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": generator,
        "count": count,
        "config": config,
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _stub_generate(name: str, n: int, config: dict) -> None:
    """Stub generator: creates dummy .npz files and a manifest."""
    d = _output_dir(name)
    batch_size = max(1, n)
    files: list[str] = []
    # Create a single dummy batch file
    fname = "batch_000.npz"
    np.savez(d / fname, dummy=np.zeros(batch_size))
    files.append(fname)
    click.echo(f"[stub] Would generate {n} {name} items. Created dummy {fname} in {d}/")
    _write_manifest(d, name, n, config, files)


@click.group()
def cli() -> None:
    """Generate and manage tt-sim data."""


@cli.command()
@click.option("--n", default=1000, help="Number of trajectories to generate.")
def trajectories(n: int) -> None:
    """Log sim trajectories."""
    _stub_generate("trajectories", n, {"env": "fancy/TableTennis-v1", "predictor": "ballistic"})


@cli.command()
@click.option("--n", default=1000, help="Number of swings to generate.")
def swings(n: int) -> None:
    """Solve & store optimized swings."""
    _stub_generate("swings", n, {"optimizer": "scipy.minimize", "method": "SLSQP"})


@cli.command()
@click.option("--n", default=1000, help="Number of laser readings to generate.")
def laser(n: int) -> None:
    """Simulated laser readings."""
    _stub_generate("laser_sim", n, {"sensor": "laser", "noise_std": 0.01})


@cli.command("rl-episodes")
@click.option("--n", default=10000, help="Number of RL episodes to generate.")
def rl_episodes(n: int) -> None:
    """RL training episodes."""
    _stub_generate("rl_episodes", n, {"env": "fancy/TableTennis-v1", "policy": "random"})


@cli.command()
def status() -> None:
    """Show what data has been generated."""
    click.echo("Generated Data Status:")
    for name in GENERATORS:
        manifest_path = DATA_ROOT / name / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            date = m["generated_at"][:10]
            click.echo(f"  {name + '/':20s} {m['count']} items ({date})")
        else:
            click.echo(f"  {name + '/':20s} (not generated)")
