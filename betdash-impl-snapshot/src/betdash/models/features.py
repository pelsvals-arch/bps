"""Feature extraction for baseline models.

All features must be computed with information observed before the event starts.
Callers should pass events whose snapshots/results are filtered by as_of/start.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from betdash.db.repository import Repository
from betdash.normalize.models import CanonicalEvent


def _to_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_result_date(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in (None, "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            dt = datetime.fromisoformat(raw) if fmt is None else datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _winner_from_score(score: str) -> str | None:
    from betdash.backtest.engine import BacktestEngine

    return BacktestEngine.winner_side_from_score(str(score or ""))


def build_features(
    event: CanonicalEvent,
    repo: Repository,
    as_of: datetime | None = None,
) -> dict[str, float]:
    """Build a numeric feature dict for an event.

    Uses only data observed before `event.start_at` and before `as_of` when
    provided, preventing future leakage.
    """

    start_at = event.start_at
    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=timezone.utc)
    else:
        start_at = start_at.astimezone(timezone.utc)
    if as_of is not None:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of = as_of.astimezone(timezone.utc)
        start_at = min(start_at, as_of)

    participants = tuple(str(p) for p in event.participant_ids)
    p1 = participants[0] if len(participants) > 0 else ""
    p2 = participants[1] if len(participants) > 1 else ""

    # Snapshot-derived pre-start market features.
    with repo._connect() as conn:
        snap_rows = conn.execute(
            """
            SELECT bookmaker, market, outcome, odds_decimal, observed_at
            FROM odds_snapshots
            WHERE event_uid = ? AND is_live = 0
            ORDER BY observed_at, id
            """,
            (event.event_uid,),
        ).fetchall()

    pre_start = []
    for row in snap_rows:
        observed_at = _to_dt(row["observed_at"])
        if observed_at is None or observed_at >= start_at:
            continue
        pre_start.append(row)

    latest_by_key: dict[tuple[str, str, str], Any] = {}
    for row in pre_start:
        key = (str(row["bookmaker"]), str(row["market"]), str(row["outcome"]))
        latest_by_key[key] = row

    p1_prices = [
        float(row["odds_decimal"])
        for row in latest_by_key.values()
        if str(row["outcome"]) == "p1" and str(row["market"]) in {"match_winner", "moneyline", "winner"}
    ]
    p2_prices = [
        float(row["odds_decimal"])
        for row in latest_by_key.values()
        if str(row["outcome"]) == "p2" and str(row["market"]) in {"match_winner", "moneyline", "winner"}
    ]
    p1_best = max(p1_prices) if p1_prices else 0.0
    p2_best = max(p2_prices) if p2_prices else 0.0
    p1_implied = (1.0 / p1_best) if p1_best > 0 else 0.0
    p2_implied = (1.0 / p2_best) if p2_best > 0 else 0.0
    implied_sum = p1_implied + p2_implied
    p1_consensus = (p1_implied / implied_sum) if implied_sum > 0 else 0.5

    # Result-derived historical features, using only results before event start.
    results = repo.results()
    event_day = start_at.date()
    history: list[dict[str, Any]] = []
    for row in results:
        row_dt = _parse_result_date(row.get("event_date"))
        if row_dt is None or row_dt.date() >= event_day:
            continue
        history.append(row)

    def _form(pid: str) -> float:
        wins = 0
        total = 0
        for row in history:
            pa = str(row.get("participant_a") or "")
            pb = str(row.get("participant_b") or "")
            side = _winner_from_score(str(row.get("score") or ""))
            if side is None:
                continue
            if pid and pid == pa:
                total += 1
                if side == "left":
                    wins += 1
            elif pid and pid == pb:
                total += 1
                if side == "right":
                    wins += 1
        return (wins / total) if total else 0.5

    def _last_play_day(pid: str) -> datetime | None:
        days: list[datetime] = []
        for row in history:
            pa = str(row.get("participant_a") or "")
            pb = str(row.get("participant_b") or "")
            if pid and pid in {pa, pb}:
                dt = _parse_result_date(row.get("event_date"))
                if dt is not None:
                    days.append(dt)
        return max(days) if days else None

    def _rest_days(pid: str) -> float:
        last_dt = _last_play_day(pid)
        if last_dt is None:
            return 30.0
        delta = start_at.date().toordinal() - last_dt.date().toordinal()
        return float(max(delta, 0))

    h2h_p1_wins = 0
    h2h_total = 0
    for row in history:
        pa = str(row.get("participant_a") or "")
        pb = str(row.get("participant_b") or "")
        if {pa, pb} != {p1, p2}:
            continue
        side = _winner_from_score(str(row.get("score") or ""))
        if side is None:
            continue
        h2h_total += 1
        winner = pa if side == "left" else pb
        if winner == p1:
            h2h_p1_wins += 1

    # Placeholders for features not present in current schema; kept as stable
    # numeric inputs for logistic regression and future data enrichment.
    surface = 0.0
    patch = 0.0
    map_feature = 0.0

    return {
        "p1_form": float(_form(p1)),
        "p2_form": float(_form(p2)),
        "p1_rest_days": float(_rest_days(p1)),
        "p2_rest_days": float(_rest_days(p2)),
        "surface": float(surface),
        "patch": float(patch),
        "map": float(map_feature),
        "h2h_p1": float(h2h_p1_wins / h2h_total) if h2h_total else 0.5,
        "market_p1_consensus": float(p1_consensus),
        "coverage_count": float(len({str(r["bookmaker"]) for r in pre_start})),
    }
