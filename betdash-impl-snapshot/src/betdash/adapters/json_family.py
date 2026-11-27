"""Public JSON line adapters for Pari, Fonbet and Bettery.

All three books expose the same undocumented ``/events/list`` JSON feed, served
from a ``line-*`` host discovered through ``/urls.json?hash=``. They differ only
in their configured host and the ``scopeMarket`` id used when the line is
requested.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import httpx

from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize.models import RawOdd

ADAPTER_VERSION = "json_family/1"
TIMEOUT_SECONDS = 10.0

SCOPE_MARKETS: dict[str, int] = {"pari": 2300, "fonbet": 1600, "bettery": 501}
_FACTOR_P1 = 921
_FACTOR_P2 = 923
_START_KEYS = ("startTime", "start_time", "startAt", "start_at")


class JsonFamilyAdapter:
    """Fetch and parse one bookmaker's public pre-match JSON line."""

    def __init__(
        self,
        bookmaker: str,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        key = bookmaker.strip().casefold()
        if key not in SCOPE_MARKETS:
            raise ValueError(f"unsupported bookmaker: {bookmaker!r}")
        self.bookmaker = key
        self.settings = settings
        self.scope_market = SCOPE_MARKETS[key]
        self._client = client
        self._packet_versions: dict[str, int] = {}
        self._payload_cache: dict[str, dict] = {}
        self._health = AdapterHealth(
            bookmaker=key,
            status="degraded",
            last_success=None,
            error_count=0,
            adapter_version=ADAPTER_VERSION,
        )

    def parse_fixture(self, payload: dict) -> list[RawOdd]:
        """Convert a real ``events/list`` payload into canonical quotes.

        Factors live in a top-level table keyed by event id, and an event's
        ``sportId`` points at a tournament segment rather than a sport, so both
        are indexed before the events are walked.
        """

        observed_at = datetime.now(timezone.utc)
        events = payload.get("events")
        if not isinstance(events, list):
            return []

        sports = _sports_index(payload.get("sports"))
        captions = _tournament_captions(payload.get("tournamentInfos"))
        factors_by_event = _factors_by_event(payload.get("customFactors"))

        rows: list[RawOdd] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            source_event_id = _identifier(_first(event, ("id", "eventId", "event_id")))
            if not source_event_id:
                continue

            factor_rows = factors_by_event.get(source_event_id)
            if not factor_rows:
                continue

            sport, league = _resolve_sport_league(event, sports, captions)
            participants = _participants(event)
            start_at = _first_start(event)

            # Group by (market, line) to emit only complete two-sided markets.
            grouped: dict[tuple[str, float | None], dict[str, float]] = {}
            for factor in factor_rows:
                odds = _odds(factor.get("v"))
                if odds is None:
                    continue
                f_id = _factor_id(factor.get("f"))
                if f_id is None:
                    continue
                mapped = _map_factor(f_id, factor, sport)
                if mapped is None:
                    continue
                market, outcome, line = mapped
                grouped.setdefault((market, line), {})[outcome] = odds

            for (market, line), outcomes in grouped.items():
                required = {"p1", "p2"} if market in {"match_winner", "spread", "set_handicap"} else {"over", "under"}
                if not required.issubset(outcomes.keys()):
                    continue
                for outcome in sorted(required):
                    rows.append(
                        RawOdd(
                            bookmaker=self.bookmaker,
                            source_event_id=source_event_id,
                            sport=sport,
                            league=league,
                            participants=participants,
                            market=market,
                            outcome=outcome,
                            odds_decimal=outcomes[outcome],
                            observed_at=observed_at,
                            is_live=False,
                            start_at=start_at,
                            line=line,
                        )
                    )
        return rows

    def fetch_snapshot(self) -> list[RawOdd]:
        """Discover a working line host and return its current quotes.

        Network and payload failures mark the adapter degraded and yield an
        empty snapshot instead of raising into the collector. A valid parsed
        payload with no supported factors is a successful empty snapshot.
        """

        client = self._client or httpx.Client(timeout=TIMEOUT_SECONDS)
        owns_client = self._client is None
        try:
            host = self.settings.bookmaker_hosts[self.bookmaker].rstrip("/")
            response = client.get(f"{host}/urls.json?hash=")
            response.raise_for_status()
            endpoints = _line_endpoints(response.json(), host)
            if not endpoints:
                raise ValueError("no line host discovered")

            for endpoint in endpoints:
                rows = self._fetch_from_endpoint(client, endpoint)
                if rows is not None:
                    self._mark_success()
                    return rows
            raise ValueError("no responding line host")
        except Exception:
            self._mark_failure()
            return []
        finally:
            if owns_client:
                client.close()

    def health(self) -> AdapterHealth:
        """Return the adapter's last known state."""

        return self._health

    def _fetch_from_endpoint(
        self, client: httpx.Client, endpoint: str
    ) -> list[RawOdd] | None:
        endpoint = endpoint.rstrip("/")
        version = self._packet_versions.get(endpoint, 0)
        try:
            response = client.get(
                f"{endpoint}/events/list",
                params={
                    "lang": "ru",
                    "version": str(version),
                    "scopeMarket": self.scope_market,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            self._packet_versions.pop(endpoint, None)
            self._payload_cache.pop(endpoint, None)
            return None
        if not isinstance(payload, dict):
            self._packet_versions.pop(endpoint, None)
            self._payload_cache.pop(endpoint, None)
            return None

        # Delta response: merge into cached full payload when available.
        if version > 0 and self._payload_cache.get(endpoint):
            if payload.get("packetVersion") or payload.get("fromVersion"):
                payload = _merge_delta_payload(self._payload_cache[endpoint], payload)
            else:
                # Unexpected delta shape; force full refresh next call.
                self._packet_versions.pop(endpoint, None)
                self._payload_cache.pop(endpoint, None)

        packet = payload.get("packetVersion")
        if isinstance(packet, int):
            self._packet_versions[endpoint] = packet
            self._payload_cache[endpoint] = payload
        else:
            self._packet_versions.pop(endpoint, None)
            self._payload_cache.pop(endpoint, None)

        return self.parse_fixture(payload)

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


def _sports_index(value: object) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not isinstance(value, list):
        return index
    for row in value:
        if not isinstance(row, dict):
            continue
        row_id = _identifier(_first(row, ("id",)))
        if row_id:
            index[row_id] = row
    return index


def _tournament_captions(value: object) -> dict[str, str]:
    captions: dict[str, str] = {}
    if not isinstance(value, list):
        return captions
    for row in value:
        if not isinstance(row, dict):
            continue
        row_id = _identifier(_first(row, ("id", "tournamentId", "sportId")))
        caption = _text(_first(row, ("caption", "name", "title")))
        if row_id and caption:
            captions[row_id] = caption
    return captions


def _factors_by_event(value: object) -> dict[str, list[dict]]:
    """Build mapping: event_id -> [factor rows]."""

    result: dict[str, list[dict]] = {}
    if not isinstance(value, list):
        return result

    for row in value:
        if not isinstance(row, dict):
            continue
        event_id = _identifier(_first(row, ("e", "eventId", "event_id", "id")))
        factor_rows = row.get("factors")
        if not event_id or not isinstance(factor_rows, list):
            continue
        rows = [f for f in factor_rows if isinstance(f, dict)]
        if rows:
            result.setdefault(event_id, []).extend(rows)
    return result


_LINE_MARKET_MAP: dict[int, tuple[str, str]] = {
    930: ("total_points", "over"),
    931: ("total_points", "under"),
    1727: ("total_points", "over"),
    1728: ("total_points", "under"),
    1848: ("total_games", "over"),
    1849: ("total_games", "under"),
    927: ("spread", "p1"),
    928: ("spread", "p2"),
    937: ("spread", "p1"),
    938: ("spread", "p2"),
    1729: ("spread", "p1"),
    1730: ("spread", "p2"),
    1845: ("set_handicap", "p1"),
    1846: ("set_handicap", "p2"),
}


def _odds(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = float(value)
        return n if math.isfinite(n) and n > 0 else None
    if isinstance(value, str):
        txt = value.strip().replace(",", ".")
        try:
            n = float(txt)
        except ValueError:
            return None
        return n if math.isfinite(n) and n > 0 else None
    return None


def _factor_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        txt = value.strip()
        if txt.isdigit() or (txt.startswith("-") and txt[1:].isdigit()):
            return int(txt)
    return None


def _line_value(factor: dict) -> float | None:
    for key in ("p", "pt", "param", "line"):
        val = factor.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
        if isinstance(val, str):
            txt = val.strip().replace(",", ".")
            try:
                n = float(txt)
            except ValueError:
                continue
            if math.isfinite(n):
                return n
    return None


def _map_factor(f_id: int, factor: dict, sport: str) -> tuple[str, str, float | None] | None:
    if f_id == _FACTOR_P1:
        return "match_winner", "p1", None
    if f_id == _FACTOR_P2:
        return "match_winner", "p2", None
    mapped = _LINE_MARKET_MAP.get(f_id)
    if not mapped:
        return None
    market, outcome = mapped
    line = _line_value(factor)
    if line is None:
        return None
    sp = (sport or "").casefold()
    if "tennis" in sp or "теннис" in sp:
        if market == "total_points":
            market = "total_games"
        if market == "spread":
            market = "set_handicap"
    return market, outcome, line


def _resolve_sport_league(
    event: dict,
    sports: dict[str, dict],
    captions: dict[str, str],
) -> tuple[str, str]:
    """Return the ancestor sport name and the segment's display league."""

    segment_id = _identifier(_first(event, ("sportId", "sport_id")))
    segment = sports.get(segment_id)
    league = captions.get(segment_id) or (_text(segment.get("name")) if segment else "")

    sport = ""
    node = segment
    seen: set[str] = set()
    while isinstance(node, dict):
        node_id = _identifier(_first(node, ("id",)))
        if node_id:
            if node_id in seen:
                break
            seen.add(node_id)
        if _text(node.get("kind")).casefold() == "sport":
            sport = _text(node.get("name"))
            break

        parent_id = _identifier(_first(node, ("parentId", "parent_id")))
        if not parent_id:
            parent_ids = node.get("parentIds")
            if isinstance(parent_ids, (list, tuple)):
                for candidate in parent_ids:
                    parent_id = _identifier(candidate)
                    if parent_id:
                        break
            elif parent_ids is not None:
                parent_id = _identifier(parent_ids)
        if not parent_id:
            break
        node = sports.get(parent_id)
    return sport, league


def _first(mapping: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "title", "value"):
            found = value.get(key)
            if isinstance(found, str):
                return found.strip()
    return ""


def _identifier(value: object) -> str:
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    return str(value).strip()


def _participants(event: dict) -> tuple[str, ...]:
    team1 = _text(event.get("team1"))
    team2 = _text(event.get("team2"))
    if team1 or team2:
        return tuple(name for name in (team1, team2) if name)

    name = _text(event.get("name")) or _text(event.get("title"))
    for separator in (" - ", " vs ", " — "):
        if separator in name:
            parts = tuple(part.strip() for part in name.split(separator) if part.strip())
            if parts:
                return parts
    return (name,) if name else ()


def _first_start(event: dict) -> datetime | None:
    for key in _START_KEYS:
        if key in event:
            parsed = _parse_start(event[key])
            if parsed is not None:
                return parsed
    return None


def _parse_start(value: object) -> datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) > 1e11 else value
        try:
            return datetime.fromtimestamp(seconds, timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_start(int(text))
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _line_endpoints(value: object, base: str) -> list[str]:
    """Collect distinct ``line-*`` host roots, excluding ``clientsapi-*``."""

    endpoints: list[str] = []
    for item in _string_values(value):
        try:
            resolved = httpx.URL(base).join(item)
        except httpx.InvalidURL:
            continue
        host = resolved.host
        if not host:
            continue
        host_name = host.casefold()
        if "line" not in host_name or "clientsapi" in host_name:
            continue

        endpoint = f"https://{host}"
        if resolved.port and resolved.port != 443:
            endpoint = f"https://{host}:{resolved.port}"
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    return endpoints


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_string_values(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_string_values(item))
        return strings
    return []


def _merge_delta_payload(base: dict, delta: dict) -> dict:
    """Merge Fonbet-family delta payload into cached full payload.

    Delta responses may include partial lists (events/customFactors/sports/...)
    and deleted ids. We keep this conservative: merge by id, replace factor rows
    by event id, preserve unknown keys from base unless explicitly replaced.
    """

    merged = dict(base)
    for key, value in delta.items():
        if key in {"events", "customFactors", "sports", "tournamentInfos", "eventBlocks", "eventMiscs", "liveEventInfos"} and isinstance(value, list):
            merged[key] = _merge_rows_by_id(base.get(key), value, key=key)
        elif key == "deleted" and isinstance(value, dict):
            merged = _apply_deletions(merged, value)
        else:
            merged[key] = value
    return merged


def _apply_deletions(payload: dict, deleted: dict) -> dict:
    out = dict(payload)
    for table, ids in deleted.items():
        if not isinstance(ids, list):
            continue
        idset = {_identifier(i) for i in ids if _identifier(i)}
        if not idset:
            continue
        if table == "customFactors":
            rows = out.get("customFactors")
            if isinstance(rows, list):
                out["customFactors"] = [r for r in rows if _identifier(_first(r, ("e", "eventId", "event_id", "id"))) not in idset]
            continue
        rows = out.get(table)
        if isinstance(rows, list):
            out[table] = [r for r in rows if _identifier(_first(r, ("id", "eventId", "event_id"))) not in idset]
    return out


def _merge_rows_by_id(base_rows: object, delta_rows: list, *, key: str) -> list:
    if not isinstance(base_rows, list):
        base_rows = []
    if key == "customFactors":
        by_id: dict[str, dict] = {}
        for row in base_rows:
            if not isinstance(row, dict):
                continue
            rid = _identifier(_first(row, ("e", "eventId", "event_id", "id")))
            if rid:
                by_id[rid] = row
        for row in delta_rows:
            if not isinstance(row, dict):
                continue
            rid = _identifier(_first(row, ("e", "eventId", "event_id", "id")))
            if rid:
                by_id[rid] = row
        return list(by_id.values())

    by_id: dict[str, dict] = {}
    order: list[str] = []
    for row in base_rows:
        if not isinstance(row, dict):
            continue
        rid = _identifier(_first(row, ("id", "eventId", "event_id")))
        if not rid:
            continue
        by_id[rid] = row
        order.append(rid)
    for row in delta_rows:
        if not isinstance(row, dict):
            continue
        rid = _identifier(_first(row, ("id", "eventId", "event_id")))
        if not rid:
            continue
        if rid not in by_id:
            order.append(rid)
        by_id[rid] = row
    return [by_id[rid] for rid in order if rid in by_id]
