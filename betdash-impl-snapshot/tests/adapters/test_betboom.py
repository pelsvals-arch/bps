"""Tests for the BetBoom public siteapi adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from betdash.adapters.base import Adapter
from betdash.adapters.betboom import ADAPTER_VERSION, BetBoomAdapter
from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize import canon_market

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/betboom_line.json").read_text(encoding="utf-8")
)


@pytest.fixture
def make_adapter():
    clients: list[httpx.Client] = []

    def factory(handler=None, host: str = "https://betboom.test") -> BetBoomAdapter:
        client = httpx.Client(transport=httpx.MockTransport(handler)) if handler else None
        if client is not None:
            clients.append(client)
        settings = Settings(bookmaker_hosts={"betboom": host})
        return BetBoomAdapter(settings, client=client)

    yield factory

    for client in clients:
        client.close()


def test_maps_cyber_and_tennis_counts():
    rows = BetBoomAdapter(Settings()).parse_fixture(FIXTURE)
    sports = {r.sport for r in rows}
    assert {"tennis", "basketball", "dota2", "cs2"} <= sports


def test_emits_canonical_match_winner_p1_p2():
    rows = BetBoomAdapter(Settings()).parse_fixture(FIXTURE)
    assert rows
    assert all(r.market == "match_winner" for r in rows)
    assert all(r.outcome in {"p1", "p2"} for r in rows)
    assert all(canon_market(r.market, r.bookmaker) == r.market for r in rows)
    by_event = {}
    for row in rows:
        by_event.setdefault(row.source_event_id, set()).add(row.outcome)
    assert all(outcomes == {"p1", "p2"} for outcomes in by_event.values())


def test_carries_metadata_and_start_at_when_present():
    before = datetime.now(timezone.utc)
    rows = BetBoomAdapter(Settings()).parse_fixture(FIXTURE)
    after = datetime.now(timezone.utc)

    tennis = [r for r in rows if r.source_event_id == "101"]
    assert len(tennis) == 2
    p1 = next(r for r in tennis if r.outcome == "p1")
    assert p1.bookmaker == "betboom"
    assert p1.sport == "tennis"
    assert p1.league == "ATP Tour"
    assert p1.participants == ("Player A", "Player B")
    assert p1.start_at == datetime(2025, 12, 1, 12, 0, tzinfo=timezone.utc)
    assert p1.is_live is False
    assert before <= p1.observed_at <= after

    missing_start = [r for r in rows if r.source_event_id in {"103", "104"}]
    assert missing_start
    assert all(r.start_at is None for r in missing_start)


def test_only_mvp_sports_and_normalized_names():
    payload = {
        "events": [
            {
                "id": 1,
                "sportName": "Football",
                "team1": "A",
                "team2": "B",
                "markets": [{"name": "Match Winner", "outcomes": [{"short": "P1", "odds": 2.0}, {"short": "P2", "odds": 1.8}]}],
            },
            {
                "id": 2,
                "sportName": "Counter-Strike 2",
                "team1": "A",
                "team2": "B",
                "markets": [{"name": "Winner", "outcomes": [{"short": "W1", "odds": 2.0}, {"short": "W2", "odds": 1.8}]}],
            },
        ]
    }
    rows = BetBoomAdapter(Settings()).parse_fixture(payload)
    assert [r.sport for r in rows] == ["cs2", "cs2"]


def test_health_starts_degraded_and_becomes_ok(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/siteapi/line/line"
        return httpx.Response(200, json=FIXTURE)

    adapter = make_adapter(handler=handler)
    assert isinstance(adapter, Adapter)
    assert adapter.health() == AdapterHealth(
        bookmaker="betboom",
        status="degraded",
        last_success=None,
        error_count=0,
        adapter_version=ADAPTER_VERSION,
    )

    rows = adapter.fetch_snapshot()
    assert len(rows) == 8
    health = adapter.health()
    assert health.status == "ok"
    assert health.error_count == 0
    assert health.last_success is not None
    assert health.adapter_version == ADAPTER_VERSION


def test_401_marks_degraded_and_returns_empty(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1
    assert health.last_success is None


def test_network_or_payload_failure_marks_degraded(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1


def test_fetch_uses_configured_host_and_path():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=FIXTURE)

    settings = Settings(
        bookmaker_hosts={"betboom": "https://custom-bb.test"},
        betboom_line_path="/custom/line",
    )
    adapter = BetBoomAdapter(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        rows = adapter.fetch_snapshot()
    finally:
        adapter._client.close()  # type: ignore[union-attr]

    assert rows
    assert seen == ["https://custom-bb.test/custom/line"]


def test_200_error_envelope_marks_degraded(make_adapter):
    """HTTP 200 with 404/NOT_FOUND envelope must not be treated as healthy empty feed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 404,
                "status": "NOT_FOUND",
                "error": {"message": "Not found"},
            },
        )

    adapter = make_adapter(handler=handler)
    rows = adapter.fetch_snapshot()
    assert rows == []
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1
