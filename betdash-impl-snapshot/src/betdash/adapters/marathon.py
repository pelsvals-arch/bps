"""Marathon line adapter.

Prefers the public event-line JSON endpoint used by the redesigned site, and
falls back to legacy SSR attribute parsing when JSON is unavailable. Read-only:
no cookie-consent interaction, browser automation, or authenticated requests.
Failures are surfaced as degraded adapter health.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from lxml import html as lxml_html

from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.normalize.models import RawOdd

ADAPTER_VERSION = "marathon_api/1"
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
    "хоккей": "hockey",
    "ice hockey": "hockey",
    "футбол": "football",
    "football": "football",
}
_OUTCOME_BY_SUFFIX = {
    "1": "p1",
    "p1": "p1",
    "home": "p1",
    "3": "p2",
    "p2": "p2",
    "away": "p2",
}
_START_KEYS = ("startAt", "start_at", "startTime", "start_time", "start", "displayTime")
_JSON_ENDPOINT = "/eag/event-line/api/v1/main/sports/events"
_JSON_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-PAN-SOURCE": "REDESIGN_WEB",
    "X-PAN-TARGET": "BROWSER",
    "X-PAN-VERSION": "2.8.2",
    "Accept": "application/json",
    "Accept-Language": "ru-RU,ru;q=0.9",
}
_JSON_MARKET_MODELS = {
    "MTCH_R": "match_winner",
    "MTCH_DC": "match_winner",
    "MTCH_DNB": "match_winner",
    "MTCH_HB": "spread",
    "MTCH_HBP": "spread",
    "MTCH_TTLG": "total_points",
    "MTCH_TTLP": "total_points",
}


class MarathonAdapter:
    """Fetch and parse Marathon's public SSR pre-match page."""

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self.bookmaker = "marathon"
        self.settings = settings
        self._client = client
        self._health = AdapterHealth(
            bookmaker=self.bookmaker,
            status="degraded",
            last_success=None,
            error_count=0,
            adapter_version=ADAPTER_VERSION,
        )

    def parse_fixture(self, html_text: str) -> list[RawOdd]:
        """Extract match-winner quotes from Marathon SSR HTML."""

        observed_at = datetime.now(timezone.utc)
        try:
            doc = lxml_html.fromstring(html_text)
        except Exception:
            return []

        events_meta = _extract_embedded_events(doc)
        rows: list[RawOdd] = []
        seen: set[tuple[str, str, float]] = set()

        for node in doc.xpath("//*[@data-event-treeid or @data-event-treeId]"):
            source_event_id = _text(node.get("data-event-treeid") or node.get("data-event-treeId"))
            if not source_event_id:
                continue
            meta = events_meta.get(source_event_id, {})
            sport = _canon_sport(
                meta.get("sport") or meta.get("sportName") or node.get("data-sport")
            )
            if sport not in _SUPPORTED_SPORTS:
                continue

            participants = _participants(meta, node)
            league = _text(meta.get("league") or meta.get("tournament") or meta.get("category"))
            start_at = _first_start(meta)
            is_live = bool(meta.get("isLive") or meta.get("live"))

            for outcome, odds in _extract_prices(node, source_event_id):
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
        """Fetch the public line and parse it, degrading on failure."""

        client = self._client or httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True)
        owns_client = self._client is None
        try:
            host = self.settings.bookmaker_hosts.get(
                self.bookmaker, "https://new.marathonbet.ru"
            ).rstrip("/")
            rows = self._fetch_json(client, host)
            if rows is None:
                response = client.get(f"{host}/su/")
                response.raise_for_status()
                html_text = response.text
                rows = self.parse_fixture(html_text)
                if not rows and _is_unrecognized_or_blocked(html_text):
                    raise ValueError("marathon content blocked or unrecognized")
            self._mark_success()
            return rows
        except Exception:
            self._mark_failure()
            return []
        finally:
            if owns_client:
                client.close()

    def parse_json(self, payload: dict[str, Any]) -> list[RawOdd]:
        observed_at = datetime.now(timezone.utc)
        rows: list[RawOdd] = []
        seen: set[tuple[str, str, str, float | None, float]] = set()
        for sport in payload.get("items", []):
            if not isinstance(sport, dict):
                continue
            sport_name = _canon_sport(sport.get("name") or sport.get("sportCode"))
            for event in sport.get("events", []):
                if not isinstance(event, dict):
                    continue
                sport_code = _canon_sport(event.get("sportCode") or sport_name)
                if sport_code not in _SUPPORTED_SPORTS:
                    continue
                source_event_id = _text(event.get("eventId") or event.get("treeId"))
                if not source_event_id:
                    continue
                participants = _json_participants(event)
                league = _json_league(event)
                start_at = _parse_start(event.get("displayTime") or event.get("start"))
                is_live = bool(event.get("live"))
                for market in (event.get("markets") or {}).values():
                    if not isinstance(market, dict):
                        continue
                    canonical = _JSON_MARKET_MODELS.get(_text(market.get("model") or market.get("modelKey")).upper())
                    if not canonical:
                        continue
                    sel_rows=[]
                    for sel in (market.get("selections") or {}).values():
                        if not isinstance(sel, dict):
                            continue
                        outcome = _json_outcome(canonical, sel.get("name"), participants)
                        if not outcome:
                            continue
                        odds = _price_to_decimal((sel.get("coeff") or {}).get("price"))
                        if odds is None:
                            continue
                        line = _selection_line(canonical, sel.get("name"))
                        sel_rows.append((outcome,line,odds))
                    # for handicap/total keep complete pairs by line
                    if canonical in {"spread","set_handicap","total_points","total_maps","total_games"}:
                        by_line: dict[float|None,list[tuple[str,float|None,float]]] = {}
                        for outcome,line,odds in sel_rows:
                            by_line.setdefault(line,[]).append((outcome,line,odds))
                        sel_rows=[]
                        for line,grp in by_line.items():
                            outcomes={o for o,_,_ in grp}
                            need={"p1","p2"} if canonical in {"spread","set_handicap"} else {"over","under"}
                            if not need.issubset(outcomes):
                                continue
                            for row in grp:
                                if row[0] in need:
                                    sel_rows.append(row)
                    for outcome,line,odds in sel_rows:
                        key = (source_event_id, canonical, outcome, line, odds)
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append(
                            RawOdd(
                                bookmaker=self.bookmaker,
                                source_event_id=source_event_id,
                                sport=sport_code,
                                league=league,
                                participants=participants,
                                market=canonical,
                                outcome=outcome,
                                odds_decimal=odds,
                                observed_at=observed_at,
                                is_live=is_live,
                                start_at=start_at,
                                line=line,
                            )
                        )
        return rows

    def _fetch_json(self, client: httpx.Client, host: str) -> list[RawOdd] | None:
        try:
            response = client.get(f"{host}{_JSON_ENDPOINT}", headers=_JSON_HEADERS)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
        if not isinstance(payload, dict) or "items" not in payload:
            return None
        return self.parse_json(payload)

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


def _extract_embedded_events(doc: lxml_html.HtmlElement) -> dict[str, dict]:
    """Find event metadata from embedded SSR JSON blobs."""

    merged: dict[str, dict] = {}
    for script in doc.xpath("//script/text()"):
        if not isinstance(script, str) or not script.strip():
            continue
        for obj in _json_objects_from_script(script):
            _collect_events(obj, merged)
    return merged


def _json_objects_from_script(script: str) -> list[object]:
    objects: list[object] = []
    # Common patterns: window.__INIT_DATA__ = {...}; or var initData = {...};
    for match in re.finditer(r"(?s)(?:__INIT_DATA__|initData|reactData)\s*[:=]\s*(\{)", script):
        start = match.start(1)
        blob = _balanced_json(script, start)
        if not blob:
            continue
        try:
            objects.append(json.loads(blob))
        except Exception:
            continue
    # Fallback: any JSON object containing "events".
    if not objects and '"events"' in script:
        for match in re.finditer(r"(?s)\{", script):
            blob = _balanced_json(script, match.start(0))
            if not blob or '"events"' not in blob:
                continue
            try:
                obj = json.loads(blob)
            except Exception:
                continue
            if isinstance(obj, dict):
                objects.append(obj)
    return objects


def _balanced_json(text: str, start: int) -> str | None:
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _collect_events(value: object, out: dict[str, dict], *, in_events: bool = False) -> None:
    if isinstance(value, dict):
        # Direct event map: {"events": {"123": {...}}}
        for key in ("events", "eventsMap", "eventMap"):
            if key in value and isinstance(value[key], dict):
                for ev_id, ev in value[key].items():
                    if isinstance(ev, dict):
                        out.setdefault(str(ev_id).strip(), {}).update(ev)
        # A dict that looks like an event.
        ev_id = _text(value.get("id") or value.get("eventId") or value.get("treeId"))
        if ev_id and ("sport" in value or "name" in value or "league" in value):
            out.setdefault(ev_id, {}).update(value)
        for k, v in value.items():
            _collect_events(v, out, in_events=in_events or k in {"events", "eventsMap", "eventMap"})
        return
    if isinstance(value, list):
        for item in value:
            if in_events and isinstance(item, dict):
                ev_id = _text(item.get("id") or item.get("eventId") or item.get("treeId"))
                if ev_id:
                    out.setdefault(ev_id, {}).update(item)
            _collect_events(item, out, in_events=in_events)


def _extract_prices(node: lxml_html.HtmlElement, event_id: str) -> list[tuple[str, float]]:
    prices: list[tuple[str, float]] = []
    for sel in node.xpath(".//*[@data-selection-price]"):
        odds = _parse_odds(sel.get("data-selection-price"))
        if odds is None:
            continue
        key = _text(sel.get("data-selection-key") or sel.get("data-selection-id") or sel.get("data-key"))
        outcome = _outcome_from_key(key, event_id)
        if outcome is None:
            outcome = _outcome_from_text(sel)
        if outcome is not None:
            prices.append((outcome, odds))

    # Fallback: sometimes price is on the event node itself.
    if not prices:
        odds = _parse_odds(node.get("data-selection-price"))
        outcome = _outcome_from_key(_text(node.get("data-selection-key")), event_id)
        if odds is not None and outcome is not None:
            prices.append((outcome, odds))
    return prices


def _outcome_from_key(key: str, event_id: str) -> str | None:
    if not key:
        return None
    # Common: 12345678@Match_Result.1 or ...Result.3
    if event_id and event_id in key:
        # take tail after last dot or last separator
        pass
    tail = key.rsplit(".", 1)[-1].rsplit("@", 1)[-1].rsplit(":", 1)[-1].strip().casefold()
    if tail in _OUTCOME_BY_SUFFIX:
        return _OUTCOME_BY_SUFFIX[tail]
    # sometimes ..._1 / ..._3
    m = re.search(r"(?:^|[._:@-])(1|3|p1|p2|home|away)$", key.casefold())
    if m:
        return _OUTCOME_BY_SUFFIX.get(m.group(1))
    return None


def _outcome_from_text(node: lxml_html.HtmlElement) -> str | None:
    text = " ".join(node.itertext()).strip().casefold()
    if text in {"p1", "1", "home"}:
        return "p1"
    if text in {"p2", "2", "3", "away"}:
        return "p2"
    return None


def _canon_sport(value: object) -> str:
    text = _text(value).casefold()
    if not text:
        return ""
    return _SPORT_ALIASES.get(text, text.replace(" ", ""))


def _participants(meta: dict, node: lxml_html.HtmlElement) -> tuple[str, ...]:
    for key in ("participants", "competitors", "teams"):
        value = meta.get(key)
        if isinstance(value, list):
            names = tuple(_text(v.get("name") if isinstance(v, dict) else v) for v in value)
            names = tuple(n for n in names if n)
            if names:
                return names
    t1 = _text(meta.get("team1") or meta.get("home"))
    t2 = _text(meta.get("team2") or meta.get("away"))
    if t1 or t2:
        return tuple(n for n in (t1, t2) if n)
    name = _text(meta.get("name") or meta.get("eventName") or node.get("data-event-name"))
    for sep in (" vs ", " - ", " — "):
        if sep in name:
            parts = tuple(p.strip() for p in name.split(sep) if p.strip())
            if len(parts) >= 2:
                return parts[:2]
    return (name,) if name else ()


def _first_start(meta: dict) -> datetime | None:
    for key in _START_KEYS:
        if key in meta:
            parsed = _parse_start(meta[key])
            if parsed is not None:
                return parsed
    return None


def _parse_start(value: object) -> datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) > 1e11 else float(value)
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


def _looks_blocked(html_text: str) -> bool:
    text = html_text.casefold()
    markers = (
        "access denied",
        "captcha",
        "forbidden",
        "enable javascript",
        "checking your browser",
        "ddos-guard",
        "cloudflare",
        "robot check",
    )
    return any(marker in text for marker in markers)


def _looks_like_marathon_line(html_text: str) -> bool:
    text = html_text.casefold()
    line_markers = (
        "data-event-treeid",
        "data-selection-price",
        "__init_data__",
        "reactdata",
        "marathonbet",
        "eag/event-line/api",
    )
    return any(marker in text for marker in line_markers)


def _is_unrecognized_or_blocked(html_text: str) -> bool:
    if _looks_blocked(html_text):
        return True
    try:
        lxml_html.fromstring(html_text)
    except Exception:
        return True
    return not _looks_like_marathon_line(html_text)


def _json_participants(event: dict[str, Any]) -> tuple[str, ...]:
    members = event.get("members")
    if isinstance(members, dict):
        ordered: list[tuple[int, str]] = []
        for code, value in members.items():
            if not isinstance(value, dict):
                continue
            name = _text(value.get("name"))
            if not name:
                continue
            try:
                order = int(str(code))
            except ValueError:
                order = 999
            ordered.append((order, name))
        if ordered:
            return tuple(name for _, name in sorted(ordered))
    home = _text(((event.get("homeTeam") or {}).get("members") or [{}])[0].get("name") if isinstance(event.get("homeTeam"), dict) else "")
    away = _text(((event.get("awayTeam") or {}).get("members") or [{}])[0].get("name") if isinstance(event.get("awayTeam"), dict) else "")
    if home or away:
        return tuple(x for x in (home, away) if x)
    name = _text(event.get("name"))
    for sep in (" - ", " — ", " vs "):
        if sep in name:
            parts = tuple(p.strip() for p in name.split(sep) if p.strip())
            if len(parts) >= 2:
                return parts[:2]
    return (name,) if name else ()


def _json_league(event: dict[str, Any]) -> str:
    seo = event.get("seo") if isinstance(event.get("seo"), dict) else {}
    tournament = seo.get("tournament") if isinstance(seo.get("tournament"), dict) else {}
    name = _text(tournament.get("name"))
    if name:
        return name
    header = _text(event.get("header"))
    if "." in header:
        return header.split(".", 1)[1].strip()
    return header


def _price_to_decimal(price: Any) -> float | None:
    if not isinstance(price, dict):
        return _parse_odds(price)
    n = price.get("n")
    d = price.get("d")
    try:
        n_f = float(n)
        d_f = float(d)
    except (TypeError, ValueError):
        return None
    if d_f == 0:
        return None
    return _parse_odds(1.0 + n_f / d_f)


def _selection_line(market: str, name: object) -> float | None:
    if market not in {"spread", "total_points", "set_handicap", "total_maps", "total_games"}:
        return None
    text = _text(name)
    m = re.search(r"\(([+-]?\d+(?:[.,]\d+)?)\)", text)
    if not m:
        # totals sometimes "Больше (2.5)" or "Меньше 2.5"
        m = re.search(r"([+-]?\d+(?:[.,]\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _json_outcome(market: str, name: object, participants: tuple[str, ...] = ()) -> str | None:
    text = _text(name).casefold()
    if market == "match_winner":
        if "ничья" in text or text == "draw" or text == "x":
            return "draw"
        p1 = participants[0].casefold() if len(participants) > 0 else ""
        p2 = participants[1].casefold() if len(participants) > 1 else ""
        if p1 and p1 in text:
            return "p1"
        if p2 and p2 in text:
            return "p2"
        if "1" in text and "победа" in text:
            return "p1"
        if "2" in text and "победа" in text:
            return "p2"
        if text.startswith("1") or " п1" in f" {text}":
            return "p1"
        if text.startswith("2") or " п2" in f" {text}":
            return "p2"
        return None
    if market in {"spread", "set_handicap"}:
        m = re.match(r"^\s*([12])\b", text)
        if m:
            return "p1" if m.group(1) == "1" else "p2"
        # Marathon format: "<Participant> (+1.5)"
        if participants:
            p1 = participants[0].casefold()
            p2 = participants[1].casefold() if len(participants) > 1 else ""
            if p1 and p1 in text:
                return "p1"
            if p2 and p2 in text:
                return "p2"
        return None
    if market in {"total_points", "total_games", "total_maps"}:
        if "больше" in text or "over" in text or text.startswith("o "):
            return "over"
        if "меньше" in text or "under" in text or text.startswith("u "):
            return "under"
        return None
    return None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
