from __future__ import annotations

from datetime import datetime, timezone

from betdash.config import Settings
from betdash.db.repository import AdapterHealth, Repository, Snapshot
from betdash.ingest.collector import Collector
from betdash.normalize import EntityRegistry


NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


class FakeEmptyAdapter:
    bookmaker = "pari"
    adapter_version = "fake/1"

    def fetch_snapshot(self):
        return []

    def health(self):
        return AdapterHealth(
            bookmaker="pari",
            status="degraded",
            last_success=None,
            error_count=1,
            adapter_version="fake/1",
        )


def seed_snapshot(repo: Repository) -> None:
    repo.save_snapshots(
        [
            Snapshot(
                event_uid="e1",
                bookmaker="pari",
                market="match_winner",
                outcome="p1",
                odds_decimal=2.0,
                observed_at=NOW,
                source_event_id="p1",
                adapter_version="1",
                is_live=False,
            )
        ]
    )


def test_empty_adapter_marks_degraded_and_keeps_last_snapshot(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    seed_snapshot(repo)
    registry = EntityRegistry({})
    collector = Collector(Settings(), repo, [FakeEmptyAdapter()], registry)
    assert collector.poll_once() == 0
    health = {h.bookmaker: h for h in collector.health()}
    assert health["pari"].status == "degraded"
    assert len(repo.last_snapshots("e1")) == 1


class FakeAdapter:
    def __init__(self, bookmaker="pari", rows=None, raises=False, health_status="ok"):
        self.bookmaker = bookmaker
        self.adapter_version = "fake/1"
        self.rows = rows or []
        self.raises = raises
        self.health_status = health_status
        self.calls = 0

    def fetch_snapshot(self):
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return list(self.rows)

    def health(self):
        return AdapterHealth(
            bookmaker=self.bookmaker,
            status=self.health_status,
            last_success=None,
            error_count=0,
            adapter_version=self.adapter_version,
        )


def make_raw(**overrides):
    from betdash.normalize.models import RawOdd

    values = dict(
        bookmaker="pari",
        source_event_id="ev1",
        sport="tennis",
        league="ATP",
        participants=("Djokovic N.", "Sinner J."),
        market="match_winner",
        outcome="p1",
        odds_decimal=2.0,
        observed_at=NOW,
        is_live=False,
        start_at=NOW,
    )
    values.update(overrides)
    return RawOdd(**values)


def test_poll_once_skips_live_rows_and_preserves_last_good(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    seed_snapshot(repo)
    registry = EntityRegistry({"Djokovic N.": "djokovic", "Sinner J.": "sinner"})
    adapter = FakeAdapter(rows=[make_raw(is_live=True)])
    collector = Collector(Settings(), repo, [adapter], registry)
    assert collector.poll_once() == 0
    assert len(repo.last_snapshots("e1")) == 1


def test_poll_once_persists_deterministic_name_mapping(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    registry = EntityRegistry({})
    adapter = FakeAdapter(rows=[make_raw()])
    collector = Collector(Settings(), repo, [adapter], registry)
    assert collector.poll_once() == 1
    health = {h.bookmaker: h for h in collector.health()}
    assert health["pari"].status == "ok"
    events = repo.events_for_resolution()
    assert len(events) == 1
    snaps = repo.last_snapshots(events[0].event_uid)
    assert len(snaps) == 1
    assert snaps[0].bookmaker == "pari"


def test_poll_once_isolates_adapter_exception(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    registry = EntityRegistry({"Djokovic N.": "djokovic", "Sinner J.": "sinner"})
    bad = FakeAdapter(bookmaker="pari", raises=True)
    good_rows = [make_raw(bookmaker="fonbet", source_event_id="e2")]
    good = FakeAdapter(bookmaker="fonbet", rows=good_rows)
    collector = Collector(Settings(), repo, [bad, good], registry)
    saved = collector.poll_once()
    assert saved == 1
    health = {h.bookmaker: h for h in collector.health()}
    assert health["pari"].status == "degraded"
    assert health["fonbet"].status == "ok"


def test_poll_once_saves_normalized_snapshot_and_marks_ok(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    registry = EntityRegistry({"Djokovic N.": "djokovic", "Sinner J.": "sinner"})
    rows = [make_raw()]
    collector = Collector(Settings(), repo, [FakeAdapter(rows=rows)], registry)
    saved = collector.poll_once()
    assert saved == 1
    stored = repo.last_snapshots(saved_event_uid(rows[0], registry))
    assert len(stored) == 1
    assert stored[0].bookmaker == "pari"
    assert stored[0].market == "match_winner"
    assert stored[0].outcome == "p1"
    health = {h.bookmaker: h for h in collector.health()}
    assert health["pari"].status == "ok"
    assert health["pari"].error_count == 0
    assert health["pari"].last_success is not None


def saved_event_uid(raw, registry):
    from betdash.normalize import normalize_event

    return normalize_event(raw, registry).event_uid


def test_start_and_stop_register_scheduler(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    registry = EntityRegistry({})
    collector = Collector(Settings(poll_seconds=90), repo, [], registry)
    collector.start()
    try:
        assert collector._scheduler is not None
        jobs = collector._scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].trigger.interval.total_seconds() == 90
    finally:
        collector.stop()
    assert collector._scheduler is None


def test_degraded_error_count_persists_across_collector_restart(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    registry = EntityRegistry({"Djokovic N.": "djokovic", "Sinner J.": "sinner"})
    last_success = datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc)
    repo.upsert_adapter_health(
        AdapterHealth(
            bookmaker="pari",
            status="degraded",
            last_success=last_success,
            error_count=3,
            adapter_version="fake/1",
        )
    )

    collector = Collector(Settings(), repo, [FakeAdapter(bookmaker="pari", raises=True)], registry)
    assert collector.poll_once() == 0

    health = {h.bookmaker: h for h in repo.adapter_health()}
    assert health["pari"].status == "degraded"
    assert health["pari"].error_count == 4
    assert health["pari"].last_success == last_success
