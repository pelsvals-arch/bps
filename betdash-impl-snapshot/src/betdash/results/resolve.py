"""Resolve result rows to canonical events."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from betdash.normalize.models import CanonicalEvent
from betdash.normalize.registry import normalize_name

if TYPE_CHECKING:
    from betdash.results.ingest import ResultRow


def _event_date(value: datetime) -> date:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.date()


def _result_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return _event_date(value)
    if isinstance(value, date):
        return value
    text = str(value).strip()
    # tennis-data.co.uk commonly uses dd/mm/yyyy; also accept ISO.
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return date.fromisoformat(text)


def resolve_result(
    result: "ResultRow",
    events: list[CanonicalEvent],
    registry: object | None = None,
) -> str | None:
    """Return event_uid when exactly one event matches normalized names + date.

    Returns None when there is no match or when the match is ambiguous.
    An optional registry with a ``resolve(name) -> tuple[str,...]`` method can
    be provided to map result names through the same alias logic as Task 2.
    """

    raw_names = [result.participant_a, result.participant_b]
    norm_names = {normalize_name(n) for n in raw_names}
    if len(norm_names) != 2:
        return None
    try:
        target_date = _result_date(result.date)
    except Exception:
        return None

    resolved_ids: set[str] | None = None
    if registry is not None and hasattr(registry, "resolve"):
        ids: list[str] = []
        for n in raw_names:
            candidates = registry.resolve(n)
            if len(candidates) != 1:
                ids = []
                break
            ids.append(candidates[0])
        if len(ids) == 2:
            resolved_ids = set(ids)

    matches: list[str] = []
    for event in events:
        if _event_date(event.start_at) != target_date:
            continue
        event_names = {normalize_name(p) for p in event.participant_ids}
        if event_names == norm_names:
            matches.append(event.event_uid)
            continue
        if resolved_ids is not None and set(event.participant_ids) == resolved_ids:
            matches.append(event.event_uid)
            continue
        # Fallback: compare normalized result names to canonical-id tail tokens.
        event_tail_names = {normalize_name(p.split(":", 1)[-1].replace("_", " ")) for p in event.participant_ids}
        if event_tail_names == norm_names:
            matches.append(event.event_uid)
    if len(matches) == 1:
        return matches[0]
    return None
