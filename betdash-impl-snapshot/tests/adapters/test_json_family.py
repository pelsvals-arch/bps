"""Tests for the JSON-family (Pari, Fonbet, Bettery) adapters."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from betdash.adapters.base import Adapter
from betdash.adapters.json_family import (
    ADAPTER_VERSION,
    SCOPE_MARKETS,
    JsonFamilyAdapter,
    _line_endpoints,
)
from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize import canon_market

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/json_family_sample.json").read_text(
        encoding="utf-8"
    )
)
HOST = "https://pari.test"
LINE_HOST = "line-lb51.bk6bba-resources.com"
LINE_URL = f"https://{LINE_HOST}"
CLIENTSAPI_HOST = "clientsapi-lb51.bk6bba-resources.com"
DISCOVERY = [f"//{CLIENTSAPI_HOST}", f"//{LINE_HOST}"]


@pytest.fixture
def make_adapter():
    """Build an adapter over a deterministic in-memory HTTP transport."""
    clients: list[httpx.Client] = []

    def factory(
        bookmaker: str = "pari",
        handler=None,
        host: str = HOST,
    ) -> JsonFamilyAdapter:
        client = httpx.Client(transport=httpx.MockTransport(handler)) if handler else None
        if client is not None:
            clients.append(client)
        settings = Settings(bookmaker_hosts={bookmaker: host})
        return JsonFamilyAdapter(bookmaker, settings, client=client)

    yield factory

    for client in clients:
        client.close()


def line_handler(
    scope: int,
    discovery: list[str] | None = None,
    line_url: str = LINE_URL,
):
    """Serve the discovery list and fixture from one real shaped line host."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/urls.json":
            assert request.url.params["hash"] == ""
            return httpx.Response(200, json=DISCOVERY if discovery is None else discovery)
        if request.url.path == "/events/list":
            assert request.url.host == httpx.URL(line_url).host
            assert dict(request.url.params) == {
                "lang": "ru",
                "version": "0",
                "scopeMarket": str(scope),
            }
            return httpx.Response(200, json=FIXTURE)
        return httpx.Response(404)

    return handler


# --- fixture parsing ------------------------------------------------------


def test_parses_factor_921_923_as_p1_p2():
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    assert [(r.outcome, r.odds_decimal) for r in raw[:2]] == [("p1", 2.1), ("p2", 1.75)]


def test_emits_canonical_match_winner_market():
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    assert len(raw) == 8
    assert all(r.market == "match_winner" for r in raw)
    assert all(canon_market(r.market, r.bookmaker) == r.market for r in raw)


def test_carries_event_metadata_and_scheduled_start():
    before = datetime.now(timezone.utc)
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    after = datetime.now(timezone.utc)

    first, second = raw[:2]
    assert first.bookmaker == "pari"
    assert first.source_event_id == "1001"
    assert first.sport == "Теннис"
    assert first.league == "ATP Tour"
    assert first.participants == ("Djokovic N.", "Alcaraz C.")
    assert first.start_at == datetime(
        2026, 3, 1, 15, 0, tzinfo=timezone(timedelta(hours=3))
    )
    assert second.start_at == first.start_at
    assert first.is_live is False
    assert before <= first.observed_at <= after


def test_start_at_is_none_when_absent_or_unparseable():
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    assert {r.source_event_id: r.start_at for r in raw} == {
        "1001": datetime(2026, 3, 1, 15, 0, tzinfo=timezone(timedelta(hours=3))),
        "1002": None,
        "2001": None,
        "3001": None,
    }


def test_pairs_top_level_factors_with_top_level_events_in_order():
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    assert [(r.source_event_id, r.outcome) for r in raw] == [
        ("1001", "p1"),
        ("1001", "p2"),
        ("1002", "p1"),
        ("1002", "p2"),
        ("2001", "p1"),
        ("2001", "p2"),
        ("3001", "p1"),
        ("3001", "p2"),
    ]


def test_participants_prefer_team_fields_and_fall_back_to_name():
    by_id = {r.source_event_id: r for r in JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)}
    assert by_id["1001"].participants == ("Djokovic N.", "Alcaraz C.")
    assert by_id["3001"].participants == ("Fallback A", "Fallback B")


def test_resolves_sport_through_parents_and_league_from_segment_or_caption():
    by_id = {r.source_event_id: r for r in JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)}
    assert (by_id["1001"].sport, by_id["1001"].league) == ("Теннис", "ATP Tour")
    assert (by_id["2001"].sport, by_id["2001"].league) == ("Баскетбол", "NBA")


def test_event_with_only_one_of_the_two_factors_is_skipped():
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    assert "4001" not in {r.source_event_id for r in raw}


def test_skips_malformed_factor_rows_and_values():
    payload = {
        "events": [
            {"id": 7, "team1": "A", "team2": "B", "sportId": 77},
        ],
        "customFactors": [
            {"e": 8, "factors": [{"f": 921, "v": 2.0}, {"f": 923, "v": 3.0}]},
            {
                "e": 7,
                "factors": [
                    {"f": "bad", "v": 2.0},
                    {"f": 921, "v": "not-a-number"},
                    {"f": 921, "v": 2.0},
                    {"f": 923, "v": 1.5},
                ],
            },
        ],
    }
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(payload)
    assert [(r.outcome, r.odds_decimal) for r in raw] == [("p1", 2.0), ("p2", 1.5)]


def test_empty_payload_parses_to_no_odds():
    assert JsonFamilyAdapter("pari", Settings()).parse_fixture({}) == []


def test_epoch_start_time_is_converted_to_utc():
    expected = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    payload = {
        "events": [
            {
                "id": 7,
                "team1": "A",
                "team2": "B",
                "sportId": 77,
                "startTime": expected.timestamp(),
            }
        ],
        "customFactors": [
            {"e": 7, "factors": [{"f": 921, "v": 2.0}, {"f": 923, "v": 3.0}]}
        ],
    }
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(payload)
    assert raw[0].start_at == expected


def test_unparseable_start_time_is_none():
    payload = {
        "events": [
            {"id": 7, "team1": "A", "team2": "B", "startTime": "not-a-date"}
        ],
        "customFactors": [
            {"e": 7, "factors": [{"f": 921, "v": 2.0}, {"f": 923, "v": 3.0}]}
        ],
    }
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(payload)
    assert all(r.start_at is None for r in raw)


# --- discovery, fetching and health --------------------------------------


@pytest.mark.parametrize(
    "bookmaker,scope",
    [("pari", 2300), ("fonbet", 1600), ("bettery", 501)],
)
def test_all_bookmakers_request_their_scope_market(make_adapter, bookmaker, scope):
    adapter = make_adapter(bookmaker=bookmaker, handler=line_handler(scope))
    rows = adapter.fetch_snapshot()
    assert len(rows) == 8
    assert all(r.bookmaker == bookmaker for r in rows)


def test_line_endpoints_select_only_line_hosts():
    endpoints = _line_endpoints(
        [
            f"//{CLIENTSAPI_HOST}",
            f"//{LINE_HOST}",
            "https://example.test/line",
            "//line-alt.test/some/path",
        ],
        HOST,
    )
    assert endpoints == [LINE_URL, "https://line-alt.test"]


def test_discovery_ignores_clientsapi_and_fetches_from_line_host(make_adapter):
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host or "")
        if request.url.path == "/urls.json":
            return httpx.Response(200, json=DISCOVERY)
        if request.url.path == "/events/list":
            assert request.url.host == LINE_HOST
            return httpx.Response(200, json=FIXTURE)
        return httpx.Response(404)

    adapter = make_adapter(handler=handler)
    assert len(adapter.fetch_snapshot()) == 8
    assert CLIENTSAPI_HOST not in seen_hosts
    assert LINE_HOST in seen_hosts


def test_discovery_uses_first_responding_line_host(make_adapter):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/urls.json":
            return httpx.Response(
                200,
                json=[
                    f"//{CLIENTSAPI_HOST}",
                    "//line-first.test",
                    f"//{LINE_HOST}",
                    "//line-unused.test",
                ],
            )
        if request.url.host == "line-first.test":
            return httpx.Response(503, json={"error": "offline"})
        if request.url.host == LINE_HOST:
            assert request.url.path == "/events/list"
            return httpx.Response(200, json=FIXTURE)
        raise AssertionError(f"unexpected request: {request.url}")

    adapter = make_adapter(handler=handler)
    assert len(adapter.fetch_snapshot()) == 8
    assert not any(CLIENTSAPI_HOST in url for url in calls)
    assert not any("line-unused.test" in url for url in calls)


def test_discovery_uses_configured_bookmaker_host(make_adapter):
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host or "")
        if request.url.path == "/urls.json":
            return httpx.Response(200, json=DISCOVERY)
        return httpx.Response(200, json=FIXTURE)

    adapter = make_adapter(handler=handler, host="https://custom.pari.test")
    assert adapter.fetch_snapshot()
    assert seen_hosts[0] == "custom.pari.test"


def test_health_starts_degraded_and_becomes_ok_after_success(make_adapter):
    adapter = make_adapter(handler=line_handler(2300))
    assert isinstance(adapter, Adapter)
    assert adapter.health() == AdapterHealth(
        bookmaker="pari",
        status="degraded",
        last_success=None,
        error_count=0,
        adapter_version=ADAPTER_VERSION,
    )

    assert adapter.fetch_snapshot()
    health = adapter.health()
    assert health.status == "ok"
    assert health.last_success is not None
    assert health.error_count == 0
    assert health.adapter_version == ADAPTER_VERSION


def test_network_failure_degrades_without_raising(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    assert adapter.health().status == "degraded"
    assert adapter.health().error_count == 1
    assert adapter.health().last_success is None


def test_success_resets_previous_error_count(make_adapter):
    state = {"fail_next": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/urls.json":
            if state["fail_next"]:
                state["fail_next"] = False
                raise httpx.ConnectError("offline")
            return httpx.Response(200, json=DISCOVERY)
        return httpx.Response(200, json=FIXTURE)

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    assert adapter.health().error_count == 1
    assert adapter.fetch_snapshot()
    assert adapter.health().status == "ok"
    assert adapter.health().error_count == 0


def test_valid_empty_payload_is_success_not_degraded(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/urls.json":
            return httpx.Response(200, json=DISCOVERY)
        return httpx.Response(200, json={"sports": [], "events": [], "customFactors": []})

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "ok"
    assert health.error_count == 0
    assert health.last_success is not None


def test_valid_payload_without_supported_factors_is_success(make_adapter):
    payload = {
        "events": [{"id": 1, "team1": "A", "team2": "B"}],
        "customFactors": [{"e": 1, "factors": [{"f": 1234, "v": 9.0}]}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/urls.json":
            return httpx.Response(200, json=DISCOVERY)
        return httpx.Response(200, json=payload)

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    assert adapter.health().status == "ok"
    assert adapter.health().error_count == 0


def test_transport_failure_then_valid_empty_payload_is_success(make_adapter):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/urls.json":
            return httpx.Response(200, json=["//line-first.test", f"//{LINE_HOST}"])
        if request.url.host == "line-first.test":
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"sports": [], "events": [], "customFactors": []})

    adapter = make_adapter(handler=handler)
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "ok"
    assert health.error_count == 0
    assert health.last_success is not None


def test_unknown_bookmaker_is_rejected():
    with pytest.raises(ValueError, match="unsupported bookmaker"):
        JsonFamilyAdapter("marathon", Settings())
