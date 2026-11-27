from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from betdash.db.repository import (
    AdapterHealth,
    Repository,
    Snapshot,
    init_db,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

EXPECTED_TABLES = {
    "events",
    "participants",
    "event_aliases",
    "market_types",
    "odds_snapshots",
    "results",
    "feature_values",
    "predictions",
    "model_runs",
    "adapter_health",
}


def make_snapshot(**overrides) -> Snapshot:
    values = {
        "event_uid": "e1",
        "bookmaker": "pari",
        "market": "match_winner",
        "outcome": "p1",
        "odds_decimal": 2.0,
        "observed_at": NOW,
        "source_event_id": "p1",
        "adapter_version": "1",
        "is_live": False,
    }
    values.update(overrides)
    return Snapshot(**values)


def test_init_db_creates_tables(tmp_path):
    db = tmp_path / "bet.sqlite"
    init_db(str(db))

    assert db.exists()
    with sqlite3.connect(db) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert EXPECTED_TABLES <= names


def test_init_db_enables_wal(tmp_path):
    db = tmp_path / "bet.sqlite"
    init_db(str(db))

    with sqlite3.connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_init_db_is_idempotent(tmp_path):
    db = tmp_path / "bet.sqlite"
    init_db(str(db))
    init_db(str(db))

    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'odds_snapshots'"
        ).fetchone()[0]
    assert count == 1


def test_save_snapshots_dedupes_same_key(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    s = Snapshot(
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

    assert repo.save_snapshots([s, s]) == 1
    assert len(repo.last_snapshots("e1")) == 1


def test_save_snapshots_never_updates_existing_rows(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    repo.save_snapshots([make_snapshot(odds_decimal=2.0)])

    assert repo.save_snapshots([make_snapshot(odds_decimal=3.5)]) == 0
    [stored] = repo.last_snapshots("e1")
    assert stored.odds_decimal == 2.0


def test_last_snapshots_returns_latest_row_per_selection(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    later = NOW + timedelta(seconds=90)
    repo.save_snapshots(
        [
            make_snapshot(observed_at=NOW, odds_decimal=2.0),
            make_snapshot(observed_at=later, odds_decimal=2.1),
            make_snapshot(outcome="p2", odds_decimal=1.8),
        ]
    )

    latest = {snap.outcome: snap for snap in repo.last_snapshots("e1")}

    assert latest["p1"].odds_decimal == 2.1
    assert latest["p1"].observed_at == later
    assert latest["p2"].odds_decimal == 1.8
    assert len(repo.last_snapshots("other")) == 0


def test_snapshot_round_trip_preserves_fields(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    snapshot = make_snapshot(source_url="https://example.test/line", is_live=True)
    repo.save_snapshots([snapshot])

    [stored] = repo.last_snapshots("e1")
    assert stored == snapshot


def test_adapter_health_round_trip(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    health = AdapterHealth(
        bookmaker="pari",
        status="ok",
        last_success=NOW,
        error_count=0,
        adapter_version="1",
    )
    repo.upsert_adapter_health(health)

    assert repo.adapter_health() == [health]


def test_adapter_health_upsert_keeps_one_row_per_bookmaker(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    repo.upsert_adapter_health(
        AdapterHealth("pari", "degraded", None, 3, "1")
    )
    repo.upsert_adapter_health(
        AdapterHealth("pari", "ok", NOW, 0, "1")
    )
    repo.upsert_adapter_health(
        AdapterHealth("fonbet", "ok", NOW, 0, "2")
    )

    health = {row.bookmaker: row for row in repo.adapter_health()}
    assert set(health) == {"pari", "fonbet"}
    assert health["pari"].status == "ok"
    assert health["pari"].error_count == 0
    assert health["fonbet"].adapter_version == "2"
