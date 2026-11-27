from __future__ import annotations

from datetime import datetime, timezone

from betdash.normalize.models import CanonicalEvent
from betdash.results.ingest import ResultRow, ingest_tennis_results, parse_tennis_results
from betdash.results.resolve import resolve_result
from betdash.db.repository import Repository


def make_event(event_uid: str, p1: str, p2: str, date: str) -> CanonicalEvent:
    return CanonicalEvent(
        event_uid=event_uid,
        sport="tennis",
        league="ATP",
        participant_ids=(p1, p2),
        start_at=datetime.fromisoformat(f"{date}T12:00:00+00:00"),
        mapping_confidence=1.0,
    )


def make_result(p1: str, p2: str, date: str, score: str = "6-4 6-4", league: str = "ATP") -> ResultRow:
    return ResultRow(participant_a=p1, participant_b=p2, date=date, score=score, league=league)


def test_ambiguous_result_returns_none():
    events = [make_event("e1", "Djokovic", "Alcaraz", "2026-03-01"),
              make_event("e2", "Djokovic", "Alcaraz", "2026-03-01")]
    assert resolve_result(make_result("Djokovic", "Alcaraz", "2026-03-01"), events) is None


def test_no_match_returns_none():
    events = [make_event("e1", "Djokovic", "Alcaraz", "2026-03-01")]
    assert resolve_result(make_result("Djokovic", "Sinner", "2026-03-01"), events) is None
    assert resolve_result(make_result("Djokovic", "Alcaraz", "2026-03-02"), events) is None


def test_resolve_result_matches_normalized_names_and_date():
    events = [make_event("e1", "djokovic n", "alcaraz c", "2026-03-01")]
    result = make_result("Djokovic N.", "Alcaraz C.", "2026-03-01")
    assert resolve_result(result, events) == "e1"


def test_parse_tennis_results_reads_csv_fixture():
    rows = parse_tennis_results("tests/results/fixtures/tennis_results.csv")
    assert len(rows) == 2
    assert rows[0].participant_a == "Djokovic N."
    assert rows[0].participant_b == "Alcaraz C."
    assert rows[0].date == "2026-03-01"
    assert rows[0].score == "6-4 6-4"
    assert rows[0].league == "ATP Dubai"


def test_ingest_tennis_results_stores_resolved_rows(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    repo.upsert_canonical_event(make_event("e1", "Djokovic N.", "Alcaraz C.", "2026-03-01"))
    repo.upsert_canonical_event(make_event("e2", "Alcaraz C.", "Djokovic N.", "2026-03-02"))

    inserted = ingest_tennis_results("tests/results/fixtures/tennis_results.csv", repo)

    assert inserted == 2
    rows = repo.results()
    assert len(rows) == 2
    assert {r["event_uid"] for r in rows} == {"e1", "e2"}
    assert all(r["source"] == "tennis-data" for r in rows)

    # idempotent on same source/event
    assert ingest_tennis_results("tests/results/fixtures/tennis_results.csv", repo) == 0
    assert len(repo.results()) == 2
