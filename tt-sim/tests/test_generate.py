"""Tests for tt_sim.generate CLI."""

import json

from click.testing import CliRunner

from tt_sim.generate import cli


def test_status_runs():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Generated Data Status:" in result.output


def test_trajectories_creates_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["trajectories", "--n=5"])
    assert result.exit_code == 0

    manifest_path = tmp_path / "data" / "generated" / "trajectories" / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["generator"] == "trajectories"
    assert manifest["count"] == 5
    assert "generated_at" in manifest
    assert "config" in manifest
    assert "files" in manifest
    assert len(manifest["files"]) > 0


def test_manifest_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["swings", "--n=3"])

    manifest = json.loads((tmp_path / "data" / "generated" / "swings" / "manifest.json").read_text())
    assert set(manifest.keys()) == {"generated_at", "generator", "count", "config", "files"}
    assert manifest["generator"] == "swings"
    assert manifest["count"] == 3
