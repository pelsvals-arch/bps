"""Resilient polling collector for bookmaker adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from betdash.config import Settings
from betdash.db.repository import AdapterHealth, Repository, Snapshot
from betdash.normalize import MVP_SPORTS, EntityRegistry, canon_market, canon_sport, normalize_event
from betdash.normalize.models import RawOdd


class Collector:
    """Poll adapters, normalize quotes, and persist immutable snapshots."""

    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        adapters: list[Any],
        registry: EntityRegistry,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.adapters = adapters
        self.registry = registry
        self._scheduler: BackgroundScheduler | None = None
        self._lock = Lock()
        self._error_counts: dict[str, int] = {}
        self._last_success: dict[str, datetime | None] = {}
        self._adapter_versions: dict[str, str] = {}
        self._seed_health_from_repository()

    def poll_once(self) -> int:
        """Poll all adapters once. Never raises to caller."""
        with self._lock:
            return self._poll_once_locked()

    def _seed_health_from_repository(self) -> None:
        try:
            rows = self.repository.list_adapter_health() if hasattr(self.repository, "list_adapter_health") else self.repository.adapter_health()
        except Exception:
            return
        if not rows:
            return
        for row in rows:
            self._error_counts[row.bookmaker] = row.error_count
            self._last_success[row.bookmaker] = row.last_success
            self._adapter_versions[row.bookmaker] = row.adapter_version

    def _ensure_bookmaker_state(self, adapter: Any, bookmaker: str) -> None:
        if bookmaker in self._error_counts:
            return
        try:
            rows = self.repository.list_adapter_health() if hasattr(self.repository, "list_adapter_health") else self.repository.adapter_health()
        except Exception:
            rows = []
        for row in rows or []:
            if row.bookmaker != bookmaker:
                continue
            self._error_counts[bookmaker] = row.error_count
            self._last_success[bookmaker] = row.last_success
            self._adapter_versions[bookmaker] = row.adapter_version
            return
        self._error_counts.setdefault(bookmaker, 0)
        self._last_success.setdefault(bookmaker, None)
        try:
            self._adapter_versions.setdefault(bookmaker, self._adapter_version(adapter))
        except Exception:
            self._adapter_versions.setdefault(bookmaker, "unknown")

    def _poll_once_locked(self) -> int:
        total_saved = 0
        for adapter in self.adapters:
            bookmaker = self._bookmaker(adapter)
            self._ensure_bookmaker_state(adapter, bookmaker)
            try:
                raw_odds = adapter.fetch_snapshot()
            except Exception:
                self._mark_degraded(adapter, bookmaker)
                continue

            if not raw_odds:
                self._mark_degraded(adapter, bookmaker)
                continue

            try:
                saved, had_persistable = self._ingest_raw_odds(adapter, bookmaker, raw_odds)
            except Exception:
                self._mark_degraded(adapter, bookmaker)
                continue

            total_saved += saved
            if had_persistable:
                self._mark_ok(adapter, bookmaker)
            else:
                # Fetched data but nothing persistable (all live/low confidence/bad).
                self._mark_degraded(adapter, bookmaker)
        return total_saved

    def _ingest_raw_odds(
        self, adapter: Any, bookmaker: str, raw_odds: list[RawOdd]
    ) -> tuple[int, bool]:
        snapshots: list[Snapshot] = []
        had_persistable = False
        for raw in raw_odds:
            try:
                if raw.is_live:
                    continue
                sport = canon_sport(raw.sport)
                market = canon_market(raw.market, raw.bookmaker)
                if sport not in MVP_SPORTS:
                    continue
                if market.startswith("unknown_") and len(tuple(getattr(raw, "participants", ()) or ())) < 2:
                    # Stat/prop noise often has no two-sided event identity.
                    continue
                canonical = normalize_event(raw, self.registry)
                if canonical.mapping_confidence < 0.5:
                    continue
                snapshot = Snapshot(
                    event_uid=canonical.event_uid,
                    bookmaker=raw.bookmaker,
                    market=market,
                    outcome=raw.outcome,
                    line=getattr(raw, "line", None),
                    odds_decimal=raw.odds_decimal,
                    observed_at=raw.observed_at,
                    source_event_id=raw.source_event_id,
                    adapter_version=self._adapter_version(adapter),
                    is_live=raw.is_live,
                    source_url=getattr(raw, "source_url", None),
                )
                snapshots.append(snapshot)
                self.repository.upsert_canonical_event(canonical)
                had_persistable = True
            except Exception:
                continue
        if not snapshots:
            return 0, had_persistable
        return self.repository.save_snapshots(snapshots), had_persistable

    def start(self) -> None:
        """Register periodic polling using Settings.poll_seconds."""
        with self._lock:
            if self._scheduler is not None:
                return
            scheduler = BackgroundScheduler()
            scheduler.add_job(
                self.poll_once,
                "interval",
                seconds=self.settings.poll_seconds,
                max_instances=1,
                coalesce=True,
            )
            scheduler.start()
            self._scheduler = scheduler

    def stop(self) -> None:
        """Shut scheduler down cleanly."""
        with self._lock:
            scheduler = self._scheduler
            self._scheduler = None
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    def health(self) -> list[AdapterHealth]:
        """Return current adapter health known to collector/repository."""
        if hasattr(self.repository, "list_adapter_health"):
            return self.repository.list_adapter_health()
        return self.repository.adapter_health()

    def _mark_ok(self, adapter: Any, bookmaker: str) -> None:
        now = datetime.now(timezone.utc)
        self._error_counts[bookmaker] = 0
        self._last_success[bookmaker] = now
        version = self._adapter_version(adapter)
        self._adapter_versions[bookmaker] = version
        self._upsert_health(bookmaker, "ok", now, 0, version)

    def _mark_degraded(self, adapter: Any, bookmaker: str) -> None:
        self._error_counts[bookmaker] = self._error_counts.get(bookmaker, 0) + 1
        version = self._adapter_version(adapter)
        self._adapter_versions[bookmaker] = version
        last_success = self._last_success.get(bookmaker)
        # If adapter reports last_success in its own health, preserve it if we don't have one.
        try:
            adapter_health = adapter.health()
            if last_success is None:
                last_success = adapter_health.last_success
        except Exception:
            pass
        self._last_success[bookmaker] = last_success
        self._upsert_health(
            bookmaker,
            "degraded",
            last_success,
            self._error_counts[bookmaker],
            version,
        )

    def _upsert_health(
        self,
        bookmaker: str,
        status: str,
        last_success: datetime | None,
        error_count: int,
        adapter_version: str,
    ) -> None:
        health = AdapterHealth(
            bookmaker=bookmaker,
            status=status,
            last_success=last_success,
            error_count=error_count,
            adapter_version=adapter_version,
        )
        # Support brief interface and existing repository.
        if hasattr(self.repository, "adapter_health_upsert"):
            self.repository.adapter_health_upsert(health)
            return
        if hasattr(self.repository, "upsert_adapter_health"):
            self.repository.upsert_adapter_health(health)
            return
        # brief-named method form
        if callable(getattr(self.repository, "adapter_health", None)):
            try:
                # Try brief signature: adapter_health(bookmaker, status, last_success, error_count, adapter_version)
                self.repository.adapter_health(bookmaker, status, last_success, error_count, adapter_version)  # type: ignore[misc]
                return
            except TypeError:
                pass
        self.repository.upsert_adapter_health(health)

    def _bookmaker(self, adapter: Any) -> str:
        try:
            health = adapter.health()
            if health.bookmaker:
                return health.bookmaker
        except Exception:
            pass
        return str(getattr(adapter, "bookmaker", "unknown"))

    def _adapter_version(self, adapter: Any) -> str:
        try:
            version = adapter.health().adapter_version
            if version:
                return version
        except Exception:
            pass
        return str(getattr(adapter, "adapter_version", "unknown"))
