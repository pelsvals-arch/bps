"""BetBoom public siteapi line adapter.

Reads the public pre-match line from BetBoom's siteapi endpoint. The concrete
route is configurable via ``Settings.betboom_line_path``; the default was
captured from public browser/runtime evidence. No Qrator interaction, no
account session, no captcha bypass.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import httpx

from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize.models import RawOdd

ADAPTER_VERSION = "betboom_siteapi/1"
TIMEOUT_SECONDS = 10.0

_SUPPORTED_SPORTS = {"tennis", "basketball", "dota2", "cs2"}
_SPORT_ALIASES = {
    "tennis": "tennis",
    "теннис": "tennis",
    "basketball": "basketball",
    "баскетбол": "basketball",
    "dota 2": "dota2",
    "dota2": "dota2",
    "дота 2": "dota2",
    "cs2": "cs2",
    "cs 2": "cs2",
    "counter-strike 2": "cs2",
    "counter strike 2": "cs2",
    "counter-strike: 2": "cs2",
    "counter strike: 2": "cs2",
}
_P1_OUTCOMES = {"p1", "1", "w1", "home", "team1", "п1"}
_P2_OUTCOMES = {"p2", "2", "w2", "away", "team2", "п2"}
_MATCH_WINNER_MARKETS = {
    "match winner",
    "winner",
    "match_winner",
    "1x2",
    "исход",
    "исход матча",
    "победитель",
    "победа в матче",
}
_START_KEYS = ("startTime", "start_time", "startAt", "start_at", "startDate", "start_date")


class BetBoomAdapter:
    """Fetch and parse BetBoom's public pre-match siteapi line."""

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self.bookmaker = "betboom"
        self.settings = settings
        self._client = client
        self._health = AdapterHealth(
            bookmaker=self.bookmaker,
            status="degraded",
            last_success=None,
            error_count=0,
            adapter_version=ADAPTER_VERSION,
        )

    def parse_fixture(self, payload: dict) -> list[RawOdd]:
        """Convert a public siteapi line payload into match-winner quotes."""

        observed_at = datetime.now(timezone.utc)
        rows: list[RawOdd] = []
        seen: set[tuple[str, str, float]] = set()

        for event in _iter_events(payload):
            source_event_id = _identifier(
                _first(event, ("id", "eventId", "event_id", "ID", "Id"))
            )
            if not source_event_id:
                continue

            sport = _canon_sport(_event_sport(event))
            if sport not in _SUPPORTED_SPORTS:
                continue

            participants = _participants(event)
            league = _event_league(event)
            start_at = _first_start(event)
            is_live = _is_live(event)

            for outcome, odds in _extract_match_winner(event):
                key = (source_event_id, outcome, odds)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    RawOdd(
                        bookmaker=self.bookmaker,
                        source_event_id=source_event_id,
                        sport=sport,
                        league=league,
                        participants=participants,
                        market="match_winner",
                        outcome=outcome,
                        odds_decimal=odds,
                        observed_at=observed_at,
                        is_live=is_live,
                        start_at=start_at,
                    )
                )
        return rows

    def fetch_snapshot(self) -> list[RawOdd]:
        """Fetch the public line endpoint and parse it, degrading on failure."""

        client = self._client or httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True)
        owns_client = self._client is None
        try:
            host = self.settings.bookmaker_hosts.get(self.bookmaker, "https://betboom.ru").rstrip("/")
            path = self.settings.betboom_line_path
            if not path.startswith("/"):
                path = "/" + path
            response = client.get(f"{host}{path}")
            if response.status_code == 401:
                self._mark_failure()
                return []
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("unexpected payload type")
            if _is_error_envelope(payload):
                self._mark_failure()
                return []
            rows = self.parse_fixture(payload)
            if not rows and not _has_line_markers(payload):
                self._mark_failure()
                return []
            self._mark_success()
            return rows
        except Exception:
            self._mark_failure()
            return []
        finally:
            if owns_client:
                client.close()

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


def _iter_events(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []

    # Common shapes: {"events":[...]}, {"data":{"events":[...]}}, {"line":{"events":[...]}}
    for key in ("events", "matches", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    for container_key in ("data", "line", "result", "payload"):
        container = payload.get(container_key)
        if isinstance(container, dict):
            nested = _iter_events(container)
            if nested:
                return nested
        if isinstance(container, list):
            dicts = [item for item in container if isinstance(item, dict)]
            if dicts:
                return dicts

    # Fallback: walk top-level lists of dicts that look like events.
    found: list[dict] = []
    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            if any(_looks_like_event(v) for v in value[:3]):
                found.extend(value)  # type: ignore[arg-type]
    return found


def _looks_like_event(value: dict) -> bool:
    return any(k in value for k in ("markets", "outcomes", "sportName", "sportId", "team1", "team2"))


def _is_error_envelope(payload: dict) -> bool:
    """Detect 200-with-error payloads (e.g. 404 NOT_FOUND envelope)."""
    if not isinstance(payload, dict):
        return False
    code = payload.get("code")
    status = str(payload.get("status", "")).upper()
    err = payload.get("error")
    has_error_code = isinstance(code, int) and code >= 400
    has_error_status = status in {"NOT_FOUND", "ERROR", "FORBIDDEN", "UNAUTHORIZED"} or "NOT_FOUND" in status
    has_error_obj = isinstance(err, dict) and bool(err.get("message") or err.get("code"))
    # Positive line containers mean it's a real line payload even if odd fields exist.
    if _has_line_markers(payload):
        return False
    return has_error_code or has_error_status or has_error_obj


def _has_line_markers(payload: dict) -> bool:
    """True when payload has recognizable line/event containers."""
    if not isinstance(payload, dict):
        return False
    for key in ("events", "matches", "items"):
        v = payload.get(key)
        if isinstance(v, list):
            return True
    for key in ("data", "line", "result", "payload"):
        v = payload.get(key)
        if isinstance(v, (dict, list)):
            return True
    return False


def _event_sport(event: dict) -> object:
    for key in ("sportName", "sport_name", "sport", "sportType", "discipline"):
        if key in event and event[key] is not None:
            value = event[key]
            if isinstance(value, dict):
                return _first(value, ("name", "title", "slug", "code"))
            return value
    sport_id = event.get("sportId")
    if sport_id is not None:
        return sport_id
    return ""


def _event_league(event: dict) -> str:
    for key in ("tournamentName", "tournament_name", "league", "leagueName", "competition", "categoryName"):
        value = event.get(key)
        if isinstance(value, dict):
            text = _text(_first(value, ("name", "title", "caption")))
        else:
            text = _text(value)
        if text:
            return text
    return ""


def _extract_match_winner(event: dict) -> list[tuple[str, float]]:
    markets = event.get("markets")
    if not isinstance(markets, list):
        # Some shapes put outcomes directly on event.
        outcomes = event.get("outcomes")
        if isinstance(outcomes, list):
            return _extract_outcomes(outcomes)
        return []

    for market in markets:
        if not isinstance(market, dict):
            continue
        name = _text(_first(market, ("name", "title", "marketName", "type"))).casefold()
        if name and name not in _MATCH_WINNER_MARKETS and not _looks_like_match_winner(name):
            continue
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            outcomes = market.get("selections")
        if not isinstance(outcomes, list):
            continue
        extracted = _extract_outcomes(outcomes)
        if extracted:
            return extracted
    return []


def _looks_like_match_winner(name: str) -> bool:
    if "handicap" in name or "total" in name or "fora" in name or "тотал" in name or "фора" in name:
        return False
    return any(token in name for token in ("winner", "match", "исход", "побед"))


def _extract_outcomes(outcomes: list[object]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        label = _text(
            _first(outcome, ("short", "shortName", "code", "name", "title", "outcome", "type"))
        ).casefold()
        label = label.replace(" ", "")
        if label in _P1_OUTCOMES:
            canon = "p1"
        elif label in _P2_OUTCOMES:
            canon = "p2"
        else:
            continue
        odds = _parse_odds(_first(outcome, ("odds", "price", "value", "odd", "coef", "rate")))
        if odds is None:
            continue
        result.append((canon, odds))
    # Keep only p1/p2 pair if present.
    by_outcome = {o: v for o, v in result}
    if "p1" in by_outcome and "p2" in by_outcome:
        return [("p1", by_outcome["p1"]), ("p2", by_outcome["p2"])]
    return []


def _participants(event: dict) -> tuple[str, ...]:
    team1 = _text(_first(event, ("team1", "homeTeam", "home", "competitor1")))
    team2 = _text(_first(event, ("team2", "awayTeam", "away", "competitor2")))
    if team1 or team2:
        return tuple(name for name in (team1, team2) if name)

    competitors = event.get("competitors") or event.get("teams") or event.get("participants")
    if isinstance(competitors, list):
        names = []
        for item in competitors:
            if isinstance(item, dict):
                name = _text(_first(item, ("name", "title", "shortName")))
            else:
                name = _text(item)
            if name:
                names.append(name)
        if names:
            return tuple(names[:2])

    name = _text(_first(event, ("name", "title", "eventName")))
    for sep in (" - ", " — ", " vs ", " v "):
        if sep in name:
            parts = tuple(part.strip() for part in name.split(sep) if part.strip())
            if len(parts) >= 2:
                return parts[:2]
    return (name,) if name else ()


def _canon_sport(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Best-effort common IDs only where unambiguous in fixture/context; unknown ints ignored.
        return ""
    text = _text(value).casefold()
    if not text:
        return ""
    if text in _SPORT_ALIASES:
        return _SPORT_ALIASES[text]
    compact = text.replace("-", " ").replace("_", " ").strip()
    compact = " ".join(compact.split())
    return _SPORT_ALIASES.get(compact, compact.replace(" ", ""))


def _is_live(event: dict) -> bool:
    state = _text(_first(event, ("state", "status", "eventState"))).casefold()
    if state in {"live", "inplay", "in_play", "started"}:
        return True
    if state in {"prematch", "pre_match", "not_started", "upcoming"}:
        return False
    live = event.get("isLive")
    if isinstance(live, bool):
        return live
    return False


def _first_start(event: dict) -> datetime | None:
    for key in _START_KEYS:
        if key in event:
            parsed = _parse_start(event[key])
            if parsed is not None:
                return parsed
    return None


def _parse_start(value: object) -> datetime | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds <= 0:
            return None
        if seconds > 1e11:
            seconds = seconds / 1000
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


def _parse_odds(value: object) -> float | None:
    try:
        odds = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(odds) or odds <= 1:
        return None
    return odds


def _first(mapping: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _identifier(value: object) -> str:
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    return str(value).strip()
