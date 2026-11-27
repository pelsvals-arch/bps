from __future__ import annotations

import gzip
import io
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from betdash.adapters.base import Adapter
from betdash.adapters.winline import ADAPTER_VERSION, INIT_MESSAGES, WS_PATH, WinlineAdapter
from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize import canon_market

FIXTURE_BYTES = (Path(__file__).parent / "fixtures/winline_frame.bin").read_bytes()


def _build_frame(records: list[dict], *, compress: bool = True) -> bytes:
    buf = io.BytesIO()

    def put_str(s: str) -> None:
        b = s.encode("utf-8")
        buf.write(struct.pack("<I", len(b)))
        buf.write(b)

    buf.write(struct.pack("<H", len(records)))
    for r in records:
        put_str(r["event_id"])
        buf.write(struct.pack("<B", r["sport"]))
        put_str(r.get("league", ""))
        put_str(r.get("p1", ""))
        put_str(r.get("p2", ""))
        buf.write(struct.pack("<H", r["market"]))
        buf.write(struct.pack("<B", r["outcome"]))
        buf.write(struct.pack("<H", r["odds"]))
        buf.write(struct.pack("<B", r.get("live", 0)))
        buf.write(struct.pack("<q", r.get("start", 0)))
    payload = buf.getvalue()
    return gzip.compress(payload) if compress else payload


def test_decodes_market_0x0556_odds():
    rows = WinlineAdapter(Settings()).parse_frame(FIXTURE_BYTES)
    assert any(r.market == "match_winner" and r.odds_decimal == 2.5 for r in rows)


def test_parse_frame_returns_canonical_match_winner_rows():
    rows = WinlineAdapter(Settings()).parse_frame(FIXTURE_BYTES)
    assert len(rows) == 2
    assert all(r.bookmaker == "winline" for r in rows)
    assert all(r.market == "match_winner" for r in rows)
    assert all(canon_market(r.market, r.bookmaker) == r.market for r in rows)
    assert {r.outcome for r in rows} == {"p1", "p2"}
    by_outcome = {r.outcome: r for r in rows}
    assert by_outcome["p1"].odds_decimal == 2.5
    assert by_outcome["p2"].odds_decimal == 1.55


def test_parse_frame_populates_metadata_and_start_at():
    before = datetime.now(timezone.utc)
    rows = WinlineAdapter(Settings()).parse_frame(FIXTURE_BYTES)
    after = datetime.now(timezone.utc)

    row = rows[0]
    assert row.source_event_id == "1001"
    assert row.sport == "tennis"
    assert row.league == "ATP"
    assert row.participants == ("Player A", "Player B")
    assert row.is_live is False
    assert row.start_at == datetime(2024, 12, 1, 12, 0, tzinfo=timezone.utc)
    assert before <= row.observed_at <= after


def test_parse_frame_missing_start_at_is_none():
    frame = _build_frame(
        [
            dict(
                event_id="9",
                sport=2,
                league="NBA",
                p1="A",
                p2="B",
                market=0x0556,
                outcome=1,
                odds=180,
                live=0,
                start=0,
            )
        ]
    )
    rows = WinlineAdapter(Settings()).parse_frame(frame)
    assert len(rows) == 1
    assert rows[0].start_at is None
    assert rows[0].sport == "basketball"


def test_parse_frame_preserves_is_live_and_filters_mvp_sports():
    frame = _build_frame(
        [
            dict(event_id="1", sport=1, league="ATP", p1="A", p2="B", market=0x0556, outcome=1, odds=200, live=1, start=0),
            dict(event_id="1", sport=1, league="ATP", p1="A", p2="B", market=0x0556, outcome=2, odds=180, live=1, start=0),
            dict(event_id="2", sport=3, league="ESL", p1="T1", p2="T2", market=0x0556, outcome=1, odds=170, live=0, start=0),
            dict(event_id="2", sport=3, league="ESL", p1="T1", p2="T2", market=0x0556, outcome=2, odds=210, live=0, start=0),
            dict(event_id="3", sport=4, league="BLAST", p1="C1", p2="C2", market=0x0556, outcome=1, odds=160, live=0, start=0),
            dict(event_id="3", sport=4, league="BLAST", p1="C1", p2="C2", market=0x0556, outcome=2, odds=230, live=0, start=0),
            dict(event_id="4", sport=99, league="X", p1="X", p2="Y", market=0x0556, outcome=1, odds=150, live=0, start=0),
        ]
    )
    rows = WinlineAdapter(Settings()).parse_frame(frame)
    assert {r.sport for r in rows} == {"tennis", "dota2", "cs2"}
    assert all(r.is_live is (r.source_event_id == "1") for r in rows)
    assert not any(r.source_event_id == "4" for r in rows)


def test_parse_frame_ignores_unknown_market():
    frame = _build_frame(
        [
            dict(event_id="1", sport=1, league="ATP", p1="A", p2="B", market=0x0999, outcome=1, odds=200, live=0, start=0),
        ]
    )
    assert WinlineAdapter(Settings()).parse_frame(frame) == []


def test_parse_frame_accepts_uncompressed_payload():
    frame = _build_frame(
        [dict(event_id="1", sport=2, league="NBA", p1="A", p2="B", market=0x0556, outcome=1, odds=200, live=0, start=0)],
        compress=False,
    )
    rows = WinlineAdapter(Settings()).parse_frame(frame)
    assert len(rows) == 1
    assert rows[0].market == "match_winner"


def test_health_starts_degraded():
    adapter = WinlineAdapter(Settings())
    assert isinstance(adapter, Adapter)
    assert adapter.health() == AdapterHealth(
        bookmaker="winline",
        status="degraded",
        last_success=None,
        error_count=0,
        adapter_version=ADAPTER_VERSION,
    )


def test_fetch_snapshot_success_marks_ok(monkeypatch):
    sent: list[object] = []

    class FakeConn:
        def __init__(self, *args, **kwargs):
            self.url = args[0] if args else kwargs.get("uri")
            self.closed = False

        def send(self, msg):
            sent.append(msg)

        def recv(self, timeout=None):
            if not hasattr(self, "_done"):
                self._done = True
                return FIXTURE_BYTES
            raise TimeoutError()

        def close(self):
            self.closed = True

    import betdash.adapters.winline as winline_mod

    monkeypatch.setattr(
        winline_mod,
        "_connect_ws",
        lambda url, **kwargs: FakeConn(url, **kwargs),
    )

    adapter = WinlineAdapter(Settings())
    rows = adapter.fetch_snapshot()

    assert len(rows) == 2
    assert sent == list(INIT_MESSAGES)
    health = adapter.health()
    assert health.status == "ok"
    assert health.error_count == 0
    assert health.last_success is not None
    assert health.adapter_version == ADAPTER_VERSION


def test_fetch_snapshot_failure_marks_degraded(monkeypatch):
    import betdash.adapters.winline as winline_mod

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(winline_mod, "_connect_ws", boom)

    adapter = WinlineAdapter(Settings())
    assert adapter.fetch_snapshot() == []
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1
    assert health.last_success is None


def test_fetch_snapshot_uses_configured_winline_host(monkeypatch):
    seen = {}

    class FakeConn:
        def send(self, msg):
            pass

        def recv(self, timeout=None):
            if not hasattr(self, "_done"):
                self._done = True
                return FIXTURE_BYTES
            raise TimeoutError()

        def close(self):
            pass

    import betdash.adapters.winline as winline_mod

    def fake_connect(url, **kwargs):
        seen["url"] = url
        return FakeConn()

    monkeypatch.setattr(winline_mod, "_connect_ws", fake_connect)
    settings = Settings(bookmaker_hosts={"winline": "https://winline.test"})
    adapter = WinlineAdapter(settings)
    rows = adapter.fetch_snapshot()
    assert rows
    assert seen["url"] == "wss://winline.test" + WS_PATH


def test_fetch_snapshot_timeout_with_no_frames_marks_degraded(monkeypatch):
    sent: list[object] = []

    class FakeConn:
        def send(self, msg):
            sent.append(msg)

        def recv(self, timeout=None):
            assert timeout == 10
            raise TimeoutError()

        def close(self):
            pass

    import betdash.adapters.winline as winline_mod

    monkeypatch.setattr(winline_mod, "_connect_ws", lambda url, **kwargs: FakeConn())
    adapter = WinlineAdapter(Settings())
    assert adapter.fetch_snapshot() == []
    assert sent == list(INIT_MESSAGES)
    health = adapter.health()
    assert health.status == "degraded"
    assert health.error_count == 1
    assert health.last_success is None
