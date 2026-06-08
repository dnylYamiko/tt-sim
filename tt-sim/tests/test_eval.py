"""Tests for tt_sim.eval module."""

import csv
import tempfile
import os

import pytest

from tt_sim.eval import EpisodeMetrics, EvalLogger


def _make_metrics(episode=0, contact=True, landed=True, reward=1.0, prediction_error=0.05):
    return EpisodeMetrics(
        episode=episode,
        contact=contact,
        landed=landed,
        landing_x=1.0,
        landing_y=0.5,
        prediction_error=prediction_error,
        reward=reward,
        episode_time=2.0,
        perceiver="sim",
        predictor="sim",
        spin="none",
        aimer="center",
        swing="pid",
        control="pid",
    )


class TestEvalLoggerCSV:
    def test_csv_header_and_rows(self, tmp_path):
        path = str(tmp_path / "metrics.csv")
        logger = EvalLogger(log_path=path)
        for i in range(3):
            logger.log(_make_metrics(episode=i))

        with open(path) as f:
            reader = list(csv.reader(f))
        assert len(reader) == 4  # header + 3 rows
        assert reader[0][0] == "episode"


class TestEvalLoggerMemory:
    def test_summary_correct_aggregates(self):
        logger = EvalLogger()
        logger.log(_make_metrics(reward=1.0, prediction_error=0.1))
        logger.log(_make_metrics(reward=3.0, prediction_error=0.3))
        s = logger.summary()
        assert s["total_episodes"] == 2
        assert s["mean_reward"] == pytest.approx(2.0)
        assert s["mean_prediction_error"] == pytest.approx(0.2)

    def test_summary_empty(self):
        logger = EvalLogger()
        assert logger.summary() == {}


class TestRates:
    def test_contact_and_landing_rates(self):
        logger = EvalLogger()
        logger.log(_make_metrics(contact=True, landed=True))
        logger.log(_make_metrics(contact=True, landed=False))
        logger.log(_make_metrics(contact=False, landed=False))
        s = logger.summary()
        assert s["contact_rate"] == pytest.approx(2 / 3)
        assert s["landing_rate"] == pytest.approx(1 / 3)
