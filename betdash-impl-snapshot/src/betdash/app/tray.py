"""Tray controller for dashboard process.

Small controller only: Start/Stop dashboard, Open in browser, adapter/training
status. Not a second UI. It controls the dashboard subprocess; worker ownership
stays in the dashboard process/runtime to avoid duplicate collectors/trainers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from betdash.config import Settings
from betdash.db.repository import Repository


def _load_pystray():
    import pystray  # type: ignore
    from PIL import Image, ImageDraw  # type: ignore

    return pystray, Image, ImageDraw


def _status_text(status: dict[str, Any]) -> str:
    adapters = status.get("adapters", [])
    training = status.get("training", {})
    ok = sum(1 for a in adapters if a.get("status") == "ok")
    degraded = sum(1 for a in adapters if a.get("status") != "ok")
    running = "RUNNING" if status.get("dashboard_running") else "STOPPED"
    return (
        f"Dashboard {running}; adapters ok={ok} degraded={degraded}; "
        f"trainer_run={training.get('last_run_id')}"
    )


@dataclass
class TrayController:
    settings: Settings = field(default_factory=Settings)
    dashboard_url: str = "http://localhost:8501"
    dashboard_command: tuple[str, ...] = (
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/betdash/app/dashboard_page.py",
    )
    workdir: str | None = None
    autostart_dashboard: bool = False
    status_poll_seconds: float = 5.0
    _icon: Any = None
    _dashboard_process: subprocess.Popen[str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _monitor_thread: threading.Thread | None = None
    _on_status: Callable[[str], None] | None = None

    def _repo(self) -> Repository:
        return Repository(self.settings.db_path)

    def dashboard_running(self) -> bool:
        with self._lock:
            proc = self._dashboard_process
        return proc is not None and proc.poll() is None

    def start_dashboard(self) -> None:
        with self._lock:
            if self._dashboard_process is not None and self._dashboard_process.poll() is None:
                self._refresh_icon_locked()
                return
            try:
                self._dashboard_process = subprocess.Popen(
                    list(self.dashboard_command),
                    cwd=self.workdir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except Exception:
                self._dashboard_process = None
            self._refresh_icon_locked()

    def stop_dashboard(self) -> None:
        with self._lock:
            proc = self._dashboard_process
            self._dashboard_process = None
            self._refresh_icon_locked()
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def open_browser(self) -> None:
        webbrowser.open(self.dashboard_url)

    def status(self) -> dict[str, Any]:
        adapters: list[dict[str, Any]] = []
        training: dict[str, Any] = {"started": False, "last_run_id": None}
        try:
            repo = self._repo()
            health_rows = repo.list_adapter_health() if hasattr(repo, "list_adapter_health") else repo.adapter_health()
            adapters = [
                {
                    "bookmaker": r.bookmaker,
                    "status": r.status,
                    "last_success": r.last_success.isoformat() if r.last_success else None,
                    "error_count": r.error_count,
                    "adapter_version": r.adapter_version,
                }
                for r in health_rows
            ]
            training = self._read_training_status()
        except Exception:
            pass
        running = self.dashboard_running()
        return {
            "dashboard_running": running,
            "workers_started": running,
            "adapters": adapters,
            "training": training,
        }

    def _read_training_status(self) -> dict[str, Any]:
        # Best-effort status from trainer checkpoint; avoids starting a runtime.
        ck = Path(self.settings.checkpoint_dir) / "trainer_checkpoint.json"
        running = self.dashboard_running()
        if not ck.exists():
            return {"started": running, "last_run_id": None}
        try:
            data = json.loads(ck.read_text(encoding="utf-8"))
        except Exception:
            return {"started": running, "last_run_id": None}
        return {
            "started": running,
            "last_run_id": data.get("last_run_id"),
            "features_version": data.get("features_version"),
            "processed_count": data.get("processed_count", 0),
        }

    def _build_icon_image(self, running: bool | None = None):
        _, Image, ImageDraw = _load_pystray()
        if running is None:
            running = self.dashboard_running()
        bg = (20, 24, 28)
        accent = (80, 200, 120) if running else (220, 96, 96)
        img = Image.new("RGB", (64, 64), color=bg)
        draw = ImageDraw.Draw(img)
        draw.rectangle((6, 6, 58, 58), outline=accent, width=4)
        draw.text((18, 16), "BD", fill=accent)
        # Small state indicator in corner.
        draw.rectangle((44, 44, 56, 56), fill=accent)
        return img

    def _icon_title(self) -> str:
        running = self.dashboard_running()
        state = "RUNNING" if running else "STOPPED"
        return f"BetDash: dashboard {state} (click menu to stop)"

    def _refresh_icon_locked(self) -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = self._build_icon_image(self.dashboard_running())
            icon.title = self._icon_title()
        except Exception:
            pass

    def refresh_icon(self) -> None:
        with self._lock:
            self._refresh_icon_locked()

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self.status_poll_seconds):
            self.refresh_icon()

    def start(self, detached: bool = True) -> None:
        pystray, _, _ = _load_pystray()
        menu = pystray.Menu(
            pystray.MenuItem(
                lambda *_: "Stop dashboard" if self.dashboard_running() else "Start dashboard",
                lambda *_: self.stop_dashboard() if self.dashboard_running() else self.start_dashboard(),
                default=True,
            ),
            pystray.MenuItem("Open in browser", lambda *_: self.open_browser()),
            pystray.MenuItem("Show status", lambda *_: self._emit_status()),
            pystray.MenuItem("Quit tray (stops dashboard)", lambda *_: self.stop()),
        )
        self._icon = pystray.Icon("betdash", self._build_icon_image(), self._icon_title(), menu)
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="betdash-tray-monitor", daemon=True)
        self._monitor_thread.start()
        if self.autostart_dashboard:
            self.start_dashboard()
        if detached:
            self._icon.run_detached()
        else:
            self._icon.run()

    def run(self) -> None:
        self.start(detached=False)

    def _emit_status(self) -> str:
        text = _status_text(self.status())
        self.refresh_icon()
        if self._on_status is not None:
            self._on_status(text)
        icon = self._icon
        if icon is not None:
            try:
                notify = getattr(icon, "notify", None)
                if callable(notify):
                    notify(text, "BetDash status")
            except Exception:
                pass
        return text

    def stop(self) -> None:
        self._stop_event.set()
        self.stop_dashboard()
        icon = self._icon
        self._icon = None
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        monitor = self._monitor_thread
        self._monitor_thread = None
        if monitor is not None and monitor.is_alive() and monitor is not threading.current_thread():
            monitor.join(timeout=1)

