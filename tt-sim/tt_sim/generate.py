import json
import click
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

GENERATORS = ["trajectories", "swings", "laser_sim", "rl_episodes"]

def _write_manifest(directory: Path, generator: str, count: int, config: dict, files: List[str]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": generator,
        "count": count,
        "config": config,
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))

def _get_config(gen_type: str) -> Dict:
    configs = {
        "trajectories": {"env": "fancy/TableTennis-v1", "predictor": "ballistic"},
        "swings": {"optimizer": "scipy.minimize", "method": "SLSQP"},
        "laser_sim": {"sensor": "laser", "noise_std": 0.01},
        "rl_episodes": {"env": "fancy/TableTennis-v1", "policy": "random"},
    }
    return configs.get(gen_type, {})

@click.group()
def cli():
    """Manage tt-sim data generation."""
    pass

@cli.command()
@click.option(
    "--type", 
    type=click.Choice(GENERATORS), 
    required=True, 
    help="The type of data to generate."
)
@click.option(
    "--episodes", 
    default=1000, 
    type=int, 
    help="Number of items/episodes to generate."
)
@click.option(
    "--output", 
    type=click.Path(), 
    default="data/generated", 
    help="Base directory for output."
)
def generate(type: str, episodes: int, output: str):
    """Generate simulation data based on type."""
    output_path = Path(output)

    if output_path.name != type:
        output_path = output_path / type

    output_path.mkdir(parents=True, exist_ok=True)

    config = _get_config(type)
    file_name = "batch_000.npy"

    # Create trajectory-shaped dummy data: [t, x, y, z, vx, vy, vz]
    t = np.linspace(0, 2.0, max(2, episodes))

    x0, y0, z0 = 0.0, 0.0, 1.0
    vx0, vy0, vz0 = 2.0, 0.5, 4.0
    g = 9.81

    x = x0 + vx0 * t
    y = y0 + vy0 * t
    z = z0 + vz0 * t - 0.5 * g * t**2

    vx = np.full_like(t, vx0)
    vy = np.full_like(t, vy0)
    vz = vz0 - g * t

    z = np.maximum(z, 0.0)

    data = np.column_stack([
        t,
        x,
        y,
        z,
        vx,
        vy,
        vz
    ])

    np.save(output_path / file_name, data)

    _write_manifest(output_path, type, episodes, config, [file_name])

    click.echo(f"Success: Generated trajectory data with shape {data.shape}.")
    click.echo(f"Output saved to: {output_path / file_name}")
    
@cli.command()
@click.option("--dir", "base_dir", default="data/generated", help="Directory to scan.")
def status(base_dir: str):
    """Show status of generated data."""
    root = Path(base_dir)
    click.echo(f"Data Status (Root: {root}):")
    
    for name in GENERATORS:
        manifest_path = root / name / "manifest.json"
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                date = m["generated_at"][:10]
                click.echo(f"  {name + '/':20s} {m['count']} items ({date})")
            except Exception:
                click.echo(f"  {name + '/':20s} (error reading manifest)")
        else:
            click.echo(f"  {name + '/':20s} (not generated)")

if __name__ == "__main__":
    cli()