"""Winline public binary WebSocket adapter.

Connects to Winline's public data feed without authentication, sends the
documented init sequence, and decodes binary market frames. The binary parser
is isolated in ``parse_frame`` so protocol changes only touch that method and
the fixture.
"""

from __future__ import annotations

import gzip
import math
import struct
from datetime import datetime, timezone

from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize.models import RawOdd

ADAPTER_VERSION = "winline_ws/1"
WS_PATH = "/data_ng?client=newsite&nb=true"
INIT_MESSAGES = ("lang", "ru", "data", "WINLINE", "getdate")

_SUPPORTED_SPORTS = {1: "tennis", 2: "basketball", 3: "dota2", 4: "cs2"}
_MARKET_MATCH_WINNER = 0x0556
_OUTCOMES = {1: "p1", 2: "p2"}


class WinlineAdapter:
    """Fetch and parse Winline's public binary pre-match feed."""

    def __init__(self, settings: Settings) -> None:
        self.bookmaker = "winline"
        self.settings = settings
        self._health = AdapterHealth(
            bookmaker=self.bookmaker,
            status="degraded",
            last_success=None,
            error_count=0,
            adapter_version=ADAPTER_VERSION,
        )

    def parse_frame(self, data: bytes) -> list[RawOdd]:
        """Decode one binary Winline frame into canonical match-winner quotes.

        Frames may arrive gzip-compressed; the parser transparently decompresses
        them. The record layout is kept local to this method so fixture updates
        do not affect other adapters.
        """

        payload = _maybe_decompress(data)
        if len(payload) < 2:
            return []

        observed_at = datetime.now(timezone.utc)
        rows: list[RawOdd] = []
        try:
            (count,) = struct.unpack_from("<H", payload, 0)
            offset = 2
            for _ in range(count):
                event_id, offset = _read_str(payload, offset)
                if offset + 1 > len(payload):
                    break
                (sport_id,) = struct.unpack_from("<B", payload, offset)
                offset += 1
                league, offset = _read_str(payload, offset)
                p1, offset = _read_str(payload, offset)
                p2, offset = _read_str(payload, offset)
                if offset + 2 + 1 + 2 + 1 + 8 > len(payload):
                    break
                (market_id,) = struct.unpack_from("<H", payload, offset)
                offset += 2
                (outcome_id,) = struct.unpack_from("<B", payload, offset)
                offset += 1
                (odds_u16,) = struct.unpack_from("<H", payload, offset)
                offset += 2
                (is_live_raw,) = struct.unpack_from("<B", payload, offset)
                offset += 1
                (start_ts,) = struct.unpack_from("<q", payload, offset)
                offset += 8

                sport = _SUPPORTED_SPORTS.get(sport_id)
                if sport is None:
                    continue
                if market_id != _MARKET_MATCH_WINNER:
                    continue
                outcome = _OUTCOMES.get(outcome_id)
                if outcome is None:
                    continue
                odds_decimal = odds_u16 / 100
                if not math.isfinite(odds_decimal) or odds_decimal <= 1:
                    continue

                rows.append(
                    RawOdd(
                        bookmaker=self.bookmaker,
                        source_event_id=event_id,
                        sport=sport,
                        league=league,
                        participants=tuple(p for p in (p1, p2) if p),
                        market="match_winner",
                        outcome=outcome,
                        odds_decimal=odds_decimal,
                        observed_at=observed_at,
                        is_live=bool(is_live_raw),
                        start_at=_parse_start_ts(start_ts),
                    )
                )
        except (struct.error, UnicodeDecodeError, IndexError, ValueError):
            return rows
        return rows

    def fetch_snapshot(self) -> list[RawOdd]:
        """Connect to the public feed, read frames, and return parsed quotes.

        Network/protocol failures degrade health and return an empty list rather
        than raising into the collector.
        """

        client = None
        try:
            host = self.settings.bookmaker_hosts.get(self.bookmaker, "https://winline.ru")
            ws_url = _ws_url(host)
            client = _connect_ws(ws_url, open_timeout=10, close_timeout=5, max_size=2**20)
            for msg in INIT_MESSAGES:
                client.send(msg)
            rows: list[RawOdd] = []
            seen: set[tuple[str, str, float, bool]] = set()
            received_data = False
            # Read a bounded number of frames; callers control polling cadence.
            for _ in range(32):
                try:
                    message = client.recv(timeout=10)
                except TimeoutError:
                    break
                if not isinstance(message, (bytes, bytearray)):
                    continue
                parsed = self.parse_frame(bytes(message))
                if parsed:
                    received_data = True
                for row in parsed:
                    key = (row.source_event_id, row.outcome, row.odds_decimal, row.is_live)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
            if not received_data:
                self._mark_failure()
                return []
            self._mark_success()
            return rows
        except Exception:
            self._mark_failure()
            return []
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def health(self) -> AdapterHealth:
        return self._health

    def _mark_success(self) -> None:
        self._health = AdapterHealth(
            bookmaker=self.bookmaker,
            status="ok",
            last_success=datetime.now(timezone.utc),
            error_count=0,
            adapter_version=ADAPTER_VERSION,
        )

    def _mark_failure(self) -> None:
        self._health = AdapterHealth(
            bookmaker=self.bookmaker,
            status="degraded",
            last_success=self._health.last_success,
            error_count=self._health.error_count + 1,
            adapter_version=ADAPTER_VERSION,
        )


def _connect_ws(url: str, **kwargs):
    from websockets.sync.client import connect  # type: ignore[import-not-found]

    return connect(url, **kwargs)


def _ws_url(host: str) -> str:
    base = host.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif not base.startswith(("ws://", "wss://")):
        base = "wss://" + base
    # If caller passed a full wss host already including path, keep it.
    if base.endswith(WS_PATH):
        return base
    # default to wss.winline.ru for the configured winline host
    if base in {"wss://winline.ru", "ws://winline.ru"}:
        return "wss://wss.winline.ru" + WS_PATH
    return base + WS_PATH


def _maybe_decompress(data: bytes) -> bytes:
    # gzip magic header
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        try:
            return gzip.decompress(data)
        except OSError:
            return b""
    return data


def _read_str(payload: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    end = offset + length
    if end > len(payload):
        raise ValueError("string overruns payload")
    return payload[offset:end].decode("utf-8"), end


def _parse_start_ts(value: int) -> datetime | None:
    if value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
