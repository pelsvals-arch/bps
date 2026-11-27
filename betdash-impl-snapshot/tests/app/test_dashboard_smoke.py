import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from betdash.app.dashboard import StreamlitRuntime, _reset_runtime_for_tests, run_dashboard


@pytest.fixture(autouse=True)
def _reset_runtime():
    _reset_runtime_for_tests()
    yield
    _reset_runtime_for_tests()


def test_dashboard_starts_background_workers_once():
    with patch("betdash.app.dashboard.StreamlitRuntime") as rt:
        run_dashboard()
        assert rt.return_value.start_workers.call_count == 1


def test_streamlit_runtime_start_workers_is_idempotent(tmp_path):
    settings = MagicMock()
    settings.db_path = str(tmp_path / "b.sqlite")
    settings.poll_seconds = 90
    settings.max_daily_loss = 100.0
    settings.checkpoint_dir = str(tmp_path / "ck")
    settings.enable_catboost = False
    settings.betboom_line_path = "/siteapi/line/line"
    settings.bookmaker_hosts = {}

    collector = MagicMock()
    trainer = MagicMock()
    trainer.last_run_id = None
    trainer._state = {"features_version": "baseline-v1", "processed_count": 0}
    repo = MagicMock()
    repo.adapter_health.return_value = []

    with patch("betdash.app.dashboard.default_adapters", return_value=[]), patch(
        "betdash.app.dashboard.Repository", return_value=repo
    ), patch("betdash.app.dashboard.Collector", return_value=collector), patch(
        "betdash.app.dashboard.Trainer", return_value=trainer
    ), patch("betdash.app.dashboard.RiskEngine"):
        rt = StreamlitRuntime(settings=settings)
        rt.start_workers()
        rt.start_workers()
        assert collector.start.call_count == 1
        assert trainer.start.call_count == 1
        rt.stop_workers()
        assert collector.stop.call_count == 1
        assert trainer.stop.call_count == 1


def test_app_check_exits_zero_and_prints_status(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = "src"
    env["DB_PATH"] = str(tmp_path / "check.sqlite")
    env["CHECKPOINT_DIR"] = str(tmp_path / "ck")
    proc = subprocess.run(
        [sys.executable, "-m", "betdash.app", "--check"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert "adapters" in payload
    assert "training" in payload


# tray controller tests moved to tests/app/test_tray_controller.py


def test_run_dashboard_reuses_singleton_runtime():
    rt = MagicMock()

    def build():
        rt.start_workers()
        return rt

    with patch("betdash.app.dashboard._build_runtime", side_effect=build) as build_mock:
        run_dashboard()
        run_dashboard()
        assert build_mock.call_count == 1
        assert rt.start_workers.call_count == 1
