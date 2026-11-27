from __future__ import annotations

from datetime import datetime, timedelta, timezone

from betdash.backtest.engine import BacktestEngine
from betdash.db.repository import Repository, Snapshot
from betdash.normalize.models import CanonicalEvent
from betdash.results.ingest import ResultRow


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 31, tzinfo=timezone.utc)
T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
EVENT_START = datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc)
CLOSE = datetime(2026, 1, 10, 17, 55, tzinfo=timezone.utc)


class ConstantModel:
    def fit(self, rows):
        return self

    def predict_prob(self, row) -> float:
        return 0.55


def seed_repo(tmp_path, *, duplicate_close: bool = False) -> Repository:
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent(
        event_uid="e1",
        sport="tennis",
        league="ATP",
        participant_ids=("p1", "p2"),
        start_at=EVENT_START,
        mapping_confidence=0.99,
    )
    repo.upsert_canonical_event(event)
    # market_close_at lives in events table; set it for CLV.
    with repo._connect() as conn:
        conn.execute(
            """
            INSERT INTO events(event_uid, sport, league, participant_ids, start_at, market_close_at, mapping_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_uid) DO UPDATE SET market_close_at=excluded.market_close_at
            """,
            ("e1", "tennis", "ATP", '["p1","p2"]', EVENT_START.isoformat(), CLOSE.isoformat(), 0.99),
        )

    snaps = [
        # Decision-time snapshots, two books, both outcomes
        Snapshot("e1", "b1", "match_winner", "p1", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b2", "match_winner", "p1", 1.9, T0, "s2", "v1"),
        Snapshot("e1", "b2", "match_winner", "p2", 2.1, T0, "s2", "v1"),
        # Closing snapshots at/near market close
        Snapshot("e1", "b1", "match_winner", "p1", 1.95, CLOSE, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.05, CLOSE, "s1", "v1"),
        Snapshot("e1", "b2", "match_winner", "p1", 1.85, CLOSE, "s2", "v1"),
        Snapshot("e1", "b2", "match_winner", "p2", 2.15, CLOSE, "s2", "v1"),
    ]
    if duplicate_close:
        snaps.extend([
            Snapshot("e1", "b1", "match_winner", "p1", 1.95, CLOSE, "s1", "v1"),
            Snapshot("e1", "b1", "match_winner", "p2", 2.05, CLOSE, "s1", "v1"),
        ])
    repo.save_snapshots(snaps)
    repo.save_result("e1", ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="2-0", league="ATP"))
    return repo


def test_signal_uses_only_snapshots_before_decision_time(tmp_path):
    repo = seed_repo(tmp_path)
    engine = BacktestEngine(repo, ConstantModel())
    report = engine.run(START, END, "tennis")
    assert all(p.observed_at < p.event_start_at for p in report.predictions)


def test_duplicate_snapshot_counted_once_in_clv(tmp_path):
    repo = seed_repo(tmp_path, duplicate_close=True)
    engine = BacktestEngine(repo, ConstantModel())
    report = engine.run_with_duplicates(START, END, "tennis", duplicated=True)
    assert report.clv_sample_count == 1


def test_missing_close_snapshot_excludes_clv_but_keeps_signal(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent("e1", "tennis", "ATP", ("p1", "p2"), EVENT_START, 0.99)
    repo.upsert_canonical_event(event)
    # no market_close_at / no close snapshots
    with repo._connect() as conn:
        conn.execute(
            "INSERT INTO events(event_uid, sport, league, participant_ids, start_at, market_close_at, mapping_confidence) VALUES (?,?,?,?,?,?,?)",
            ("e1", "tennis", "ATP", '["p1","p2"]', EVENT_START.isoformat(), None, 0.99),
        )
    repo.save_snapshots([
        Snapshot("e1", "b1", "match_winner", "p1", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b2", "match_winner", "p1", 1.9, T0, "s2", "v1"),
        Snapshot("e1", "b2", "match_winner", "p2", 2.1, T0, "s2", "v1"),
    ])
    repo.save_result("e1", ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="2-0", league="ATP"))
    report = BacktestEngine(repo, ConstantModel()).run(START, END, "tennis")
    assert len(report.predictions) >= 1
    assert all(p.clv is None for p in report.predictions)
    assert report.clv_sample_count == 0
    assert all(p.event_start_at == EVENT_START for p in report.predictions)


def test_signal_requires_two_book_coverage(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent("e1", "tennis", "ATP", ("p1", "p2"), EVENT_START, 0.99)
    repo.upsert_canonical_event(event)
    repo.save_snapshots([
        Snapshot("e1", "b1", "match_winner", "p1", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.0, T0, "s1", "v1"),
        # second book missing p1 -> coverage for p1 < 2
        Snapshot("e1", "b2", "match_winner", "p2", 2.1, T0, "s2", "v1"),
    ])
    repo.save_result("e1", ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="2-0", league="ATP"))
    report = BacktestEngine(repo, ConstantModel()).run(START, END, "tennis")
    assert all(p.outcome != "p1" for p in report.predictions)


def test_actual_multidigit_score_not_misparsed(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent("e1", "basketball", "NBA", ("p1", "p2"), EVENT_START, 0.99)
    repo.upsert_canonical_event(event)
    with repo._connect() as conn:
        conn.execute(
            "INSERT INTO events(event_uid, sport, league, participant_ids, start_at, market_close_at, mapping_confidence) VALUES (?,?,?,?,?,?,?)",
            ("e1", "basketball", "NBA", '["p1","p2"]', EVENT_START.isoformat(), None, 0.99),
        )
    repo.save_snapshots([
        Snapshot("e1", "b1", "match_winner", "p1", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b2", "match_winner", "p1", 1.9, T0, "s2", "v1"),
        Snapshot("e1", "b2", "match_winner", "p2", 2.1, T0, "s2", "v1"),
    ])
    # p1 lost 101-110 but first-char parsing would wrongly read 1-1.
    repo.save_result("e1", ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="101-110", league="NBA"))
    report = BacktestEngine(repo, ConstantModel()).run(START, END, "basketball")
    p1 = [p for p in report.predictions if p.outcome == "p1"]
    p2 = [p for p in report.predictions if p.outcome == "p2"]
    assert p1 and p1[0].actual == 0
    assert p2 and p2[0].actual == 1


def test_actual_ambiguous_market_returns_none_not_guess(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent("e1", "tennis", "ATP", ("p1", "p2"), EVENT_START, 0.99)
    repo.upsert_canonical_event(event)
    repo.save_snapshots([
        Snapshot("e1", "b1", "total_over_2.5", "over", 1.9, T0, "s1", "v1"),
        Snapshot("e1", "b1", "total_over_2.5", "under", 1.9, T0, "s1", "v1"),
        Snapshot("e1", "b2", "total_over_2.5", "over", 1.95, T0, "s2", "v1"),
        Snapshot("e1", "b2", "total_over_2.5", "under", 1.85, T0, "s2", "v1"),
    ])
    repo.save_result("e1", ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="2-0", league="ATP"))
    report = BacktestEngine(repo, ConstantModel()).run(START, END, "tennis")
    # ambiguous/non-winner market: do not guess actual
    assert report.predictions == []


def test_clv_uses_persisted_canonical_market_close(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent("e1", "tennis", "ATP", ("p1", "p2"), EVENT_START, 0.99)
    repo.upsert_canonical_event(event)
    repo.set_market_close_at("e1", CLOSE)

    # Real persistence path: only canonical_events + snapshots/results persisted through repo.
    snaps = [
        Snapshot("e1", "b1", "match_winner", "p1", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.0, T0, "s1", "v1"),
        Snapshot("e1", "b2", "match_winner", "p1", 1.9, T0, "s2", "v1"),
        Snapshot("e1", "b2", "match_winner", "p2", 2.1, T0, "s2", "v1"),
        Snapshot("e1", "b1", "match_winner", "p1", 1.95, CLOSE, "s1", "v1"),
        Snapshot("e1", "b1", "match_winner", "p2", 2.05, CLOSE, "s1", "v1"),
        Snapshot("e1", "b2", "match_winner", "p1", 1.85, CLOSE, "s2", "v1"),
        Snapshot("e1", "b2", "match_winner", "p2", 2.15, CLOSE, "s2", "v1"),
    ]
    repo.save_snapshots(snaps)
    repo.save_result("e1", ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="2-0", league="ATP"))

    # Ensure no events.market_close_at row is required.
    with repo._connect() as conn:
        conn.execute("DELETE FROM events WHERE event_uid = ?", ("e1",))

    report = BacktestEngine(repo, ConstantModel()).run(START, END, "tennis")
    assert report.clv_sample_count == 1
    assert any(p.clv is not None for p in report.predictions)
