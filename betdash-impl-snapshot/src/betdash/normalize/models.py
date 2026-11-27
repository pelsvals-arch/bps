"""Immutable data contract shared by adapters, normalization and analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RawOdd:
    """One bookmaker quote exactly as an adapter observed it."""

    bookmaker: str
    source_event_id: str
    sport: str
    league: str
    participants: tuple[str, ...]
    market: str
    outcome: str
    odds_decimal: float
    observed_at: datetime
    is_live: bool
    start_at: datetime | None = None
    line: float | None = None


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """A raw quote's event after alias resolution.

    ``start_at`` is the quote's scheduled start when the adapter supplied one,
    otherwise it falls back to ``observed_at``.
    """

    event_uid: str
    sport: str
    league: str
    participant_ids: tuple[str, ...]
    start_at: datetime
    mapping_confidence: float


@dataclass(frozen=True, slots=True)
class MarketMapping:
    """Canonical market name plus the confidence of the label mapping."""

    market: str
    confidence: float
