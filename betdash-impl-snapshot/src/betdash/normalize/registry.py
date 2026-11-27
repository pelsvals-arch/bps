"""Alias resolution, deterministic event ids and market-label mapping."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from betdash.normalize.models import CanonicalEvent, MarketMapping, RawOdd

MVP_SPORTS = {"tennis", "basketball", "dota2", "cs2"}

_PUNCTUATION = re.compile(r"[^\w\s]|_", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_SLUG_SEPARATORS = re.compile(r"[\W_]+", re.UNICODE)

_RESOLVED_CONFIDENCE = 1.0
_UNRESOLVED_CONFIDENCE = 0.0


def normalize_name(value: str) -> str:
    """Lowercase a name and replace punctuation with spaces.

    Both registry keys and lookup values pass through this function, so
    ``"Djokovic N."`` and ``"djokovic n"`` resolve to the same entity.
    """

    text = value.casefold()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


_SPORT_ALIASES: dict[str, str] = {
    "tennis": "tennis",
    "теннис": "tennis",
    "basketball": "basketball",
    "баскетбол": "basketball",
    "dota 2": "dota2",
    "dota2": "dota2",
    "дота 2": "dota2",
    "дота2": "dota2",
    "counter strike 2": "cs2",
    "counter-strike 2": "cs2",
    "cs2": "cs2",
    "cs 2": "cs2",
    "контр страйк 2": "cs2",
    "counter strike": "cs2",
}


def canon_sport(value: str) -> str:
    """Normalize sport labels to MVP canonical ids when possible."""

    key = normalize_name(value)
    if key in _SPORT_ALIASES:
        return _SPORT_ALIASES[key]
    compact = key.replace(" ", "")
    if compact in {"dota2", "dota"}:
        return "dota2"
    if compact in {"cs2", "counterstrike2"}:
        return "cs2"
    if "теннис" in key or "tennis" in key:
        return "tennis"
    if "баскет" in key or "basketball" in key:
        return "basketball"
    if "дота" in key or "dota" in key:
        return "dota2"
    if "counter strike" in key or "cs2" in key or "контр" in key:
        return "cs2"
    return key


# Labels shared by every supported bookmaker. Adapter tasks extend the
# book-specific table below when they learn a book's real wording.
_BASE_MARKET_LABELS: dict[str, str] = {
    # match winner
    "п1": "match_winner",
    "п2": "match_winner",
    "1": "match_winner",
    "2": "match_winner",
    "w1": "match_winner",
    "w2": "match_winner",
    "победа 1": "match_winner",
    "победа 2": "match_winner",
    "победа в матче": "match_winner",
    "исход": "match_winner",
    "исход матча": "match_winner",
    "match winner": "match_winner",
    "moneyline": "match_winner",
    # tennis
    "фора по сетам": "set_handicap",
    "фора 1 по сетам": "set_handicap",
    "фора 2 по сетам": "set_handicap",
    "гандикап по сетам": "set_handicap",
    "set handicap": "set_handicap",
    "тотал геймов": "total_games",
    "тб геймов": "total_games",
    "тм геймов": "total_games",
    "тотал по геймам": "total_games",
    "total games": "total_games",
    # Dota 2 / Counter-Strike 2
    "победа по картам": "map_winner",
    "победа на карте": "map_winner",
    "победа в карте": "map_winner",
    "победа по карте 1": "map_winner",
    "победа по карте 2": "map_winner",
    "map winner": "map_winner",
    "тотал карт": "total_maps",
    "тб карт": "total_maps",
    "тм карт": "total_maps",
    "тотал по картам": "total_maps",
    "total maps": "total_maps",
    # basketball
    "фора": "spread",
    "фора 1": "spread",
    "фора 2": "spread",
    "гандикап": "spread",
    "азиатская фора": "spread",
    "handicap": "spread",
    "spread": "spread",
    "тотал очков": "total_points",
    "тб очков": "total_points",
    "тм очков": "total_points",
    "тотал по очкам": "total_points",
    "total points": "total_points",
}

_BOOKMARKER_MARKET_LABELS: dict[str, dict[str, str]] = {
    "winline": {
        "победа первой": "match_winner",
        "победа второй": "match_winner",
    },
    "pari": {
        "победа участника 1": "match_winner",
        "победа участника 2": "match_winner",
    },
    "fonbet": {
        "исход 1": "match_winner",
        "исход 2": "match_winner",
    },
    "marathon": {},
    "betboom": {
        "исход 1": "match_winner",
        "исход 2": "match_winner",
    },
    "bettery": {},
}

_BASE_MARKETS = {
    normalize_name(label): market for label, market in _BASE_MARKET_LABELS.items()
}
_MARKETS_BY_BOOKMAKER = {
    bookmaker: {
        **_BASE_MARKETS,
        **{
            normalize_name(label): market
            for label, market in labels.items()
        },
    }
    for bookmaker, labels in _BOOKMARKER_MARKET_LABELS.items()
}


class EntityRegistry:
    """Map normalized participant aliases to canonical ids.

    A JSON fixture is a flat object of alias -> canonical id, where a value may
    also be a list when one alias is genuinely ambiguous.
    """

    def __init__(self, aliases: Mapping[str, str | Iterable[str]]):
        resolved: dict[str, set[str]] = {}
        for alias, targets in aliases.items():
            key = normalize_name(alias)
            if not key:
                raise ValueError(f"alias {alias!r} is empty after normalization")
            for target in _canonical_targets(alias, targets):
                resolved.setdefault(key, set()).add(target)
        self._aliases: dict[str, tuple[str, ...]] = {
            key: tuple(sorted(canonical_ids))
            for key, canonical_ids in resolved.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> EntityRegistry:
        """Load an alias registry from a UTF-8 JSON object."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("alias registry must be a JSON object")
        return cls(data)

    def resolve(self, name: str) -> tuple[str, ...]:
        """Return every canonical id the alias maps to, sorted."""

        return self._aliases.get(normalize_name(name), ())


def _canonical_targets(alias: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        targets = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        targets = tuple(value)
    else:
        raise ValueError(f"alias {alias!r} must map to a string or a list of strings")

    if not targets or not all(
        isinstance(target, str) and target.strip() for target in targets
    ):
        raise ValueError(f"alias {alias!r} must map to at least one non-empty string")
    return tuple(target.strip() for target in targets)


def normalize_event(raw: RawOdd, registry: EntityRegistry) -> CanonicalEvent:
    """Resolve a raw quote into a canonical event.

    Aliases that resolve to exactly one participant become canonical ids.
    Unknown and ambiguous aliases become provisional ``unresolved:`` or
    ``ambiguous:`` ids so the event still has a deterministic identity, but
    ``mapping_confidence`` is then zero and the collector drops it.
    """

    participant_ids: list[str] = []
    resolved_count = 0
    ambiguous = False
    for name in raw.participants:
        candidates = registry.resolve(name)
        if len(candidates) == 1:
            participant_ids.append(candidates[0])
            resolved_count += 1
        elif not candidates:
            participant_ids.append(f"name:{normalize_name(name)}")
        else:
            participant_ids.append(f"ambiguous:{normalize_name(name)}")
            ambiguous = True

    ids = tuple(participant_ids)
    fully_resolved = (
        bool(ids)
        and resolved_count == len(ids)
        and len(set(ids)) == len(ids)
    )
    if fully_resolved:
        confidence = _RESOLVED_CONFIDENCE
    elif len(ids) >= 2 and not ambiguous and len(set(ids)) == len(ids):
        # Deterministic name-based identity: usable for storage/display, but
        # lower confidence than registry-resolved entities.
        confidence = 0.6
    else:
        confidence = _UNRESOLVED_CONFIDENCE

    sport = canon_sport(raw.sport)
    league = raw.league.strip()
    start_at = raw.start_at or raw.observed_at
    if sport not in MVP_SPORTS:
        confidence = _UNRESOLVED_CONFIDENCE
    if len(ids) < 2 or _looks_like_prop_participants(ids):
        confidence = _UNRESOLVED_CONFIDENCE

    return CanonicalEvent(
        event_uid=_event_uid(sport, league, ids, start_at),
        sport=sport,
        league=league,
        participant_ids=ids,
        start_at=start_at,
        mapping_confidence=confidence,
    )


def _event_uid(
    sport: str,
    league: str,
    participant_ids: tuple[str, ...],
    start_at: datetime,
) -> str:
    payload = "|".join(
        (
            sport,
            normalize_name(league),
            "|".join(sorted(participant_ids)),
            _utc_date(start_at),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _utc_date(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.date().isoformat()


def canon_market_mapping(raw_market: str, bookmaker: str) -> MarketMapping:
    """Map a bookmaker's market label to a canonical market and confidence."""

    labels = _MARKETS_BY_BOOKMAKER.get(bookmaker.strip().casefold(), _BASE_MARKETS)
    market = labels.get(normalize_name(raw_market))
    if market is None:
        return MarketMapping(market=_unknown_market(raw_market), confidence=0.0)
    return MarketMapping(market=market, confidence=1.0)


def canon_market(raw_market: str, bookmaker: str) -> str:
    """Return only the canonical market name for a bookmaker label."""

    return canon_market_mapping(raw_market, bookmaker).market


_PROP_MARKERS = (
    "сет",
    "set",
    "эйс",
    "ace",
    "двойн",
    "double fault",
    "гейм",
    "game",
    "тай бр",
    "tiebreak",
    "tie break",
    "брейк",
    "break point",
    "тотал",
    "total",
    "фора",
    "handicap",
    "spread",
    "углов",
    "карточ",
    "удален",
    "очки",
    "points",
)


def _looks_like_prop_participants(participant_ids: tuple[str, ...]) -> bool:
    if len(participant_ids) != 1:
        return False
    name = str(participant_ids[0]).casefold()
    return any(marker in name for marker in _PROP_MARKERS)


def _unknown_market(raw_market: str) -> str:
    slug = _SLUG_SEPARATORS.sub("_", raw_market.casefold()).strip("_")
    return f"unknown_{slug}" if slug else "unknown"
