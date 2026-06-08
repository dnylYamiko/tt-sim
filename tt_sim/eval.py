"""Evaluation utilities for tt-sim."""

import csv
import os
from dataclasses import dataclass, asdict


@dataclass
class EpisodeMetrics:
    episode: int
    contact: bool              # did paddle touch ball?
    landed: bool               # did ball land on opponent's side?
    landing_x: float | None    # landing position
    landing_y: float | None
    prediction_error: float | None  # meters, at contact time
    reward: float
    episode_time: float        # seconds
    # subsystem config
    perceiver: str
    predictor: str
    spin: str
    aimer: str
    swing: str
    control: str


class EvalLogger:
    """Logs per-episode metrics to CSV."""

    def __init__(self, log_path: str | None = None, config: dict | None = None):
        """
        Args:
            log_path: path to CSV file. If None, metrics are only stored in memory.
            config: dict of subsystem flag names (e.g., {"perceiver": "sim", ...})
        """
        self.log_path = log_path
        self.config = config or {}
        self.episodes: list[EpisodeMetrics] = []

        # Create CSV file with header if path given
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=EpisodeMetrics.__dataclass_fields__.keys())
                writer.writeheader()

    def log(self, metrics: EpisodeMetrics):
        """Log one episode's metrics."""
        self.episodes.append(metrics)
        if self.log_path:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=EpisodeMetrics.__dataclass_fields__.keys())
                writer.writerow(asdict(metrics))

    def summary(self) -> dict:
        """Return aggregate statistics."""
        if not self.episodes:
            return {}
        n = len(self.episodes)
        return {
            "total_episodes": n,
            "contact_rate": sum(e.contact for e in self.episodes) / n,
            "landing_rate": sum(e.landed for e in self.episodes) / n,
            "mean_reward": sum(e.reward for e in self.episodes) / n,
            "mean_prediction_error": (
                sum(e.prediction_error for e in self.episodes if e.prediction_error is not None)
                / max(1, sum(1 for e in self.episodes if e.prediction_error is not None))
            ),
        }

    def print_summary(self):
        """Print summary to console."""
        s = self.summary()
        if not s:
            print("No episodes logged.")
            return
        print(f"\n{'='*40}")
        print(f"Results: {s['total_episodes']} episodes")
        print(f"  Contact rate:     {s['contact_rate']:.1%}")
        print(f"  Landing rate:     {s['landing_rate']:.1%}")
        print(f"  Mean reward:      {s['mean_reward']:.3f}")
        print(f"  Mean pred error:  {s['mean_prediction_error']:.4f} m")
        print(f"{'='*40}\n")
