from __future__ import annotations

from datetime import datetime, timezone

from betdash.config import Settings
from betdash.db.repository import Repository, Snapshot
from betdash.models.baseline import EloRating, GlickoRating
from betdash.models.features import build_features
from betdash.normalize.models import CanonicalEvent
from betdash.results.ingest import ResultRow


NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
START = datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc)


def _seed_repo(tmp_path) -> Repository:
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent(
        event_uid="e1",
        sport="tennis",
        league="ATP",
        participant_ids=("p1", "p2"),
        start_at=START,
        mapping_confidence=0.99,
    )
    repo.upsert_canonical_event(event)
    repo.save_snapshots(
        [
            Snapshot("e1", "b1", "match_winner", "p1", 2.0, NOW, "s1", "v1"),
            Snapshot("e1", "b1", "match_winner", "p2", 2.0, NOW, "s1", "v1"),
            Snapshot("e1", "b2", "match_winner", "p1", 1.9, NOW, "s2", "v1"),
            Snapshot("e1", "b2", "match_winner", "p2", 2.1, NOW, "s2", "v1"),
        ]
    )
    repo.save_result(
        "e1",
        ResultRow(
            participant_a="p1",
            participant_b="p2",
            date="2026-01-10",
            score="2-0",
            league="ATP",
        ),
    )
    return repo


def test_elo_probability_range():
    elo = EloRating()
    p = elo.probability("a", "b")
    assert 0.0 < p < 1.0


def test_glicko_probability_range():
    g = GlickoRating()
    p = g.probability("a", "b")
    assert 0.0 < p < 1.0


def test_build_features_returns_float_dict(tmp_path):
    repo = _seed_repo(tmp_path)
    event = repo.events_for_resolution()[0]
    features = build_features(event, repo)
    assert isinstance(features, dict)
    assert features
    assert all(isinstance(k, str) for k in features)
    assert all(isinstance(v, float) for v in features.values())


def test_build_features_excludes_future_snapshots(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    start = datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc)
    event = CanonicalEvent("e1", "tennis", "ATP", ("p1", "p2"), start, 0.99)
    repo.upsert_canonical_event(event)
    before = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    after = datetime(2026, 1, 10, 19, 0, tzinfo=timezone.utc)
    repo.save_snapshots(
        [
            Snapshot("e1", "b1", "match_winner", "p1", 2.0, before, "s1", "v1"),
            Snapshot("e1", "b1", "match_winner", "p2", 2.0, before, "s1", "v1"),
            Snapshot("e1", "b2", "match_winner", "p1", 2.0, before, "s2", "v1"),
            Snapshot("e1", "b2", "match_winner", "p2", 2.0, before, "s2", "v1"),
            Snapshot("e1", "b1", "match_winner", "p1", 9.0, after, "s1", "v1"),
            Snapshot("e1", "b1", "match_winner", "p2", 1.1, after, "s1", "v1"),
            Snapshot("e1", "b2", "match_winner", "p1", 9.0, after, "s2", "v1"),
            Snapshot("e1", "b2", "match_winner", "p2", 1.1, after, "s2", "v1"),
        ]
    )
    f = build_features(event, repo)
    # implied from 2.0/2.0 odds before start, not post-start 9.0/1.1
    assert 0.45 < f["market_p1_consensus"] < 0.55


def test_build_features_ddmmyyyy_result_does_not_leak(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    event = CanonicalEvent(
        event_uid="e2",
        sport="tennis",
        league="ATP",
        participant_ids=("p1", "p2"),
        start_at=datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc),
        mapping_confidence=0.99,
    )
    repo.upsert_canonical_event(event)
    # Future result stored as DD/MM/YYYY - must not be used as history for event on 2026-01-10
    repo.save_result("e3", ResultRow(participant_a="p1", participant_b="p2", date="11/01/2026", score="2-0", league="ATP"))
    # Past result
    repo.save_result("e0", ResultRow(participant_a="p1", participant_b="p2", date="09/01/2026", score="0-2", league="ATP"))
    f = build_features(event, repo)
    # If future leaked, p1 form would be 1.0; with only past 0-2 loss, p1 form is 0.0
    assert f["p1_form"] == 0.0
    assert f["h2h_p1"] == 0.0
