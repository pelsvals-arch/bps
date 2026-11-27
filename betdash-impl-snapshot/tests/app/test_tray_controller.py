from unittest.mock import MagicMock, patch

from betdash.app.tray import TrayController, _status_text
from betdash.config import Settings


def _settings(tmp_path):
    return Settings(db_path=str(tmp_path / "t.sqlite"), checkpoint_dir=str(tmp_path / "ck"))


def test_tray_start_stop_open(tmp_path):
    tc = TrayController(
        settings=_settings(tmp_path),
        dashboard_command=("python", "-c", "import time; time.sleep(5)"),
        workdir=str(tmp_path),
    )
    with patch("betdash.app.tray.webbrowser.open") as open_browser, patch(
        "betdash.app.tray.subprocess.Popen"
    ) as popen:
        proc = MagicMock()
        proc.poll.return_value = None
        popen.return_value = proc
        tc.start_dashboard()
        popen.assert_called_once()
        assert popen.call_args.kwargs.get("cwd") == str(tmp_path)
        tc.open_browser()
        open_browser.assert_called_once_with(tc.dashboard_url)
        status = tc.status()
        assert status["dashboard_running"] is True
        tc.stop_dashboard()
        proc.terminate.assert_called_once()


def test_tray_does_not_start_workers_in_process(tmp_path):
    tc = TrayController(settings=_settings(tmp_path), dashboard_command=("python", "-V"))
    with patch("betdash.app.tray.subprocess.Popen") as popen, patch(
        "betdash.app.dashboard.StreamlitRuntime.start_workers"
    ) as start_workers:
        proc = MagicMock()
        proc.poll.return_value = None
        popen.return_value = proc
        tc.start_dashboard()
        start_workers.assert_not_called()


def test_status_text_reflects_running_state():
    base = {"adapters": [{"status": "ok"}, {"status": "degraded"}], "training": {"last_run_id": "r1"}}
    assert "RUNNING" in _status_text({**base, "dashboard_running": True})
    assert "STOPPED" in _status_text({**base, "dashboard_running": False})


def test_build_icon_image_changes_by_state(tmp_path):
    tc = TrayController(settings=_settings(tmp_path))
    stopped = tc._build_icon_image(running=False)
    running = tc._build_icon_image(running=True)
    assert stopped.tobytes() != running.tobytes()
