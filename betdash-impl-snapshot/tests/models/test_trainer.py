from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from betdash.config import Settings
from betdash.db.repository import Repository, Snapshot
from betdash.models.trainer import Trainer
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


def test_trainer_resumes_after_restart(tmp_path):
    repo = _seed_repo(tmp_path)
    t1 = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    t1.step()
    t2 = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    assert t2.last_run_id == t1.last_run_id
    assert t2.step().features_version == t1.step().features_version


def test_trainer_writes_model_run(tmp_path):
    repo = _seed_repo(tmp_path)
    t = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    run = t.step()
    with repo._connect() as conn:
        row = conn.execute("SELECT run_id, features_version FROM model_runs WHERE run_id=?", (run.run_id,)).fetchone()
    assert row is not None
    assert row["features_version"] == run.features_version


def test_trainer_ratings_separate_per_sport(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    # seed two sports with same participant ids but different sports
    for sport, uid in [("tennis", "e1"), ("basketball", "e2")]:
        event = CanonicalEvent(uid, sport, "L", ("p1", "p2"), START, 0.99)
        repo.upsert_canonical_event(event)
        repo.save_snapshots([
            Snapshot(uid, "b1", "match_winner", "p1", 2.0, NOW, "s1", "v1"),
            Snapshot(uid, "b1", "match_winner", "p2", 2.0, NOW, "s1", "v1"),
            Snapshot(uid, "b2", "match_winner", "p1", 1.9, NOW, "s2", "v1"),
            Snapshot(uid, "b2", "match_winner", "p2", 2.1, NOW, "s2", "v1"),
        ])
        repo.save_result(uid, ResultRow(participant_a="p1", participant_b="p2", date="2026-01-10", score="2-0", league="L"))
    t = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    t.step()
    # internal state separation
    assert "tennis" in t._elo
    assert "basketball" in t._elo
    assert t._elo["tennis"] is not t._elo["basketball"]


def test_trainer_start_stop(tmp_path):
    repo = _seed_repo(tmp_path)
    t = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt"), poll_seconds=1))
    t.start()
    try:
        deadline = time.time() + 3
        while t.last_run_id is None and time.time() < deadline:
            time.sleep(0.05)
        assert t.last_run_id is not None
    finally:
        t.stop()
    # checkpoint should be valid JSON
    ckpt = (tmp_path / "ckpt" / "trainer_checkpoint.json").read_text(encoding="utf-8")
    data = json.loads(ckpt)
    assert data["last_run_id"] == t.last_run_id


def test_trainer_step_is_idempotent_no_new_data(tmp_path):
    repo = _seed_repo(tmp_path)
    t = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    r1 = t.step()
    state1 = json.loads((tmp_path / "ckpt" / "trainer_checkpoint.json").read_text(encoding="utf-8"))
    elo1 = {s: dict(r.ratings) for s, r in t._elo.items()}
    g1 = {s: {k: (v.rating, v.rd) for k, v in r.ratings.items()} for s, r in t._glicko.items()}

    r2 = t.step()
    state2 = json.loads((tmp_path / "ckpt" / "trainer_checkpoint.json").read_text(encoding="utf-8"))
    elo2 = {s: dict(r.ratings) for s, r in t._elo.items()}
    g2 = {s: {k: (v.rating, v.rd) for k, v in r.ratings.items()} for s, r in t._glicko.items()}

    assert elo1 == elo2
    assert g1 == g2
    assert state1.get("processed_event_uids") == state2.get("processed_event_uids")
    # model runs may differ in run_id/created_at, but no drift in processed state
    with repo._connect() as conn:
        m1 = conn.execute("SELECT metrics_json FROM model_runs WHERE run_id=?", (r1.run_id,)).fetchone()
        m2 = conn.execute("SELECT metrics_json FROM model_runs WHERE run_id=?", (r2.run_id,)).fetchone()
    assert json.loads(m1["metrics_json"])["samples"] == json.loads(m2["metrics_json"])["samples"]


def test_trainer_walk_forward_metrics_recorded(tmp_path):
    repo = Repository(str(tmp_path / "b.sqlite"))
    # two chronological events -> one walk-forward fold (train on first, test on second)
    for i, uid in enumerate(["e1", "e2"], start=1):
        start = datetime(2026, 1, 10 + i, 18, 0, tzinfo=timezone.utc)
        obs = datetime(2026, 1, 10 + i, 12, 0, tzinfo=timezone.utc)
        event = CanonicalEvent(uid, "tennis", "ATP", ("p1", "p2"), start, 0.99)
        repo.upsert_canonical_event(event)
        repo.save_snapshots([
            Snapshot(uid, "b1", "match_winner", "p1", 2.0, obs, "s1", "v1"),
            Snapshot(uid, "b1", "match_winner", "p2", 2.0, obs, "s1", "v1"),
            Snapshot(uid, "b2", "match_winner", "p1", 1.9, obs, "s2", "v1"),
            Snapshot(uid, "b2", "match_winner", "p2", 2.1, obs, "s2", "v1"),
        ])
        repo.save_result(uid, ResultRow(participant_a="p1", participant_b="p2", date=f"2026-01-{10+i}", score="2-0", league="ATP"))

    t = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    run = t.step()
    with repo._connect() as conn:
        row = conn.execute("SELECT metrics_json FROM model_runs WHERE run_id=?", (run.run_id,)).fetchone()
    metrics = json.loads(row["metrics_json"])
    wf = metrics["walk_forward"]
    assert wf["predictions"] == 1
    assert len(wf["folds"]) == 1
    assert wf["folds"][0]["train_size"] == 1
    assert wf["folds"][0]["test_event_uids"] == ["e2"]


def test_trainer_step_rolls_back_on_failure(tmp_path):
    repo = _seed_repo(tmp_path)
    t = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))

    # baseline clean run for expected ratings
    clean = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "clean")))
    clean.step()
    expected_elo = {s: dict(r.ratings) for s, r in clean._elo.items()}
    expected_glicko = {
        s: {k: (v.rating, v.rd) for k, v in r.ratings.items()}
        for s, r in clean._glicko.items()
    }

    original_write = t._write_model_run

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    t._write_model_run = boom  # type: ignore[assignment]
    try:
        t.step()
    except RuntimeError:
        pass
    finally:
        t._write_model_run = original_write  # type: ignore[assignment]

    # failure before commit must not leave mutated in-memory ratings/state
    assert {s: dict(r.ratings) for s, r in t._elo.items()} == {}
    assert {s: {k: (v.rating, v.rd) for k, v in r.ratings.items()} for s, r in t._glicko.items()} == {}
    assert t._state.get("processed_event_uids") == []

    # next step should apply exactly once
    t.step()
    assert {s: dict(r.ratings) for s, r in t._elo.items()} == expected_elo
    assert {
        s: {k: (v.rating, v.rd) for k, v in r.ratings.items()}
        for s, r in t._glicko.items()
    } == expected_glicko
    assert t._state.get("processed_event_uids") == ["e1"]


def test_trainer_restart_step_no_new_data_no_drift(tmp_path):
    repo = _seed_repo(tmp_path)
    t1 = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    t1.step()
    elo1 = {s: dict(r.ratings) for s, r in t1._elo.items()}
    g1 = {s: {k: (v.rating, v.rd) for k, v in r.ratings.items()} for s, r in t1._glicko.items()}

    t2 = Trainer(repo, Settings(checkpoint_dir=str(tmp_path / "ckpt")))
    t2.step()
    elo2 = {s: dict(r.ratings) for s, r in t2._elo.items()}
    g2 = {s: {k: (v.rating, v.rd) for k, v in r.ratings.items()} for s, r in t2._glicko.items()}

    assert elo1 == elo2
    assert g1 == g2
    assert t2._state.get("processed_event_uids") == ["e1"]
