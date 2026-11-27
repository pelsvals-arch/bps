"""Tests for the Marathon SSR adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from betdash.adapters.base import Adapter
from betdash.adapters.marathon import ADAPTER_VERSION, MarathonAdapter
from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize import canon_market

FIXTURE_HTML = (
    Path(__file__).parent / "fixtures/marathon_ssr.html"
).read_text(encoding="utf-8")


@pytest.fixture
def make_adapter():
    clients: list[httpx.Client] = []

    def factory(handler=None, host: str = "https://marathon.test") -> MarathonAdapter:
        client = httpx.Client(transport=httpx.MockTransport(handler)) if handler else None
        if client is not None:
            clients.append(client)
        settings = Settings(bookmaker_hosts={"marathon": host})
        return MarathonAdapter(settings, client=client)

    yield factory

    for client in clients:
        client.close()


def test_extracts_odds_from_data_attributes():
    rows = MarathonAdapter(Settings()).parse_fixture(FIXTURE_HTML)
    assert any(r.outcome == "p1" and r.odds_decimal == 2.875 for r in rows)


def test_emits_canonical_match_winner_and_filters_sports():
    rows = MarathonAdapter(Settings()).parse_fixture(FIXTURE_HTML)
    assert rows
    assert all(r.market == "match_winner" for r in rows)
    assert all(canon_market(r.market, r.bookmaker) == r.market for r in rows)
    assert {r.sport for r in rows} == {"tennis", "basketball", "dota2", "cs2"}
    assert "football" not in {r.sport for r in rows}

    cs2 = [r for r in rows if r.source_event_id == "44440001"]
    assert [(r.outcome, r.odds_decimal) for r in cs2] == [("p1", 1.65), ("p2", 2.25)]
    assert all(r.sport == "cs2" and r.market == "match_winner" for r in cs2)


def test_carries_event_metadata_and_start_at():
    before = datetime.now(timezone.utc)
    rows = MarathonAdapter(Settings()).parse_fixture(FIXTURE_HTML)
    after = datetime.now(timezone.utc)

    tennis = [r for r in rows if r.source_event_id == "12345678"]
    assert len(tennis) == 2
    p1 = next(r for r in tennis if r.outcome == "p1")
    assert p1.bookmaker == "marathon"
    assert p1.source_event_id == "12345678"
    assert p1.participants == ("Djokovic N.", "Sinner J.")
    assert p1.league == "ATP Tour"
    assert p1.start_at == datetime(2026, 12, 1, 14, 0, tzinfo=timezone.utc)
    assert p1.is_live is False
    assert before <= p1.observed_at <= after


def test_health_starts_degraded_and_becomes_ok(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/su/"
        return httpx.Response(200, text=FIXTURE_HTML)

    adapter = make_adapter(handler=handler)
    assert isinstance(adapter, Adapter)
    assert adapter.health() == AdapterHealth(
        bookmaker="marathon",
        status="degraded",
        last_success=None,
        error_count=0,
        adapter_version=ADAPTER_VERSION,
    )

    rows = adapter.fetch_snapshot()
    assert len(rows) == 8
    health = adapter.health()
    assert health.status == "ok"
    assert health.last_success is not None
    assert health.error_count == 0


def test_blocked_or_failed_fetch_marks_degraded(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="blocked")

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1
    assert health.last_success is None


def test_network_failure_marks_degraded(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    assert adapter.health().status == "degraded"
    assert adapter.health().error_count == 1


def test_default_fetch_uses_new_marathon_host():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/sports/events"):
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, text=FIXTURE_HTML)

    adapter = MarathonAdapter(Settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        assert adapter.fetch_snapshot()
    finally:
        adapter._client.close()  # type: ignore[union-attr]
    assert seen == [
        "https://new.marathonbet.ru/eag/event-line/api/v1/main/sports/events",
        "https://new.marathonbet.ru/su/",
    ]


def test_unrecognized_interstitial_without_marker_is_degraded(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body><h1>Please wait</h1></body></html>")

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1
    assert health.last_success is None


def test_legitimate_empty_marathon_line_page_is_success(make_adapter):
    html = """<!doctype html><html><head><title>Marathon</title></head>
    <body data-page="line"><div class='line-page' data-event-treeId='root'></div></body></html>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "ok"
    assert health.error_count == 0
    assert health.last_success is not None
