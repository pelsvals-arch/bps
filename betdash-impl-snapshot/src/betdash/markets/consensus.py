"""Consensus benchmark probabilities across bookmakers.

De-vig and consensus are benchmarks, not truth. Signals with fewer than two
qualifying books have zero benchmark coverage.
"""

from __future__ import annotations

from betdash.db.repository import Snapshot
from betdash.markets.de_vig import implied_probs


def consensus_prob(snapshots: list[Snapshot], outcome: str) -> tuple[float, int]:
    """Return (consensus probability, coverage) for an outcome.

    Each bookmaker is de-vigged separately. Only books with at least two
    outcomes present are included. Coverage is the number of qualifying books.
    When coverage < 2, returns ``(0.0, coverage)``.
    """
    by_book: dict[str, list[Snapshot]] = {}
    for snap in snapshots:
        by_book.setdefault(snap.bookmaker, []).append(snap)

    probs: list[float] = []
    coverage = 0
    for book_snaps in by_book.values():
        # Latest snapshot per outcome for this bookmaker.
        latest_by_outcome: dict[str, Snapshot] = {}
        for snap in book_snaps:
            prev = latest_by_outcome.get(snap.outcome)
            if prev is None or snap.observed_at >= prev.observed_at:
                latest_by_outcome[snap.outcome] = snap

        if len(latest_by_outcome) < 2:
            continue
        if outcome not in latest_by_outcome:
            continue

        outcomes = list(latest_by_outcome.keys())
        odds = [latest_by_outcome[o].odds_decimal for o in outcomes]
        devigged = implied_probs(odds)
        outcome_prob = devigged[outcomes.index(outcome)]
        probs.append(outcome_prob)
        coverage += 1

    if coverage < 2:
        return 0.0, coverage
    return sum(probs) / len(probs), coverage


def best_price(snapshots: list[Snapshot], outcome: str) -> float | None:
    """Return the maximum available decimal odds for an outcome, or None."""
    prices = [snap.odds_decimal for snap in snapshots if snap.outcome == outcome]
    if not prices:
        return None
    return max(prices)


def clv_included(has_close_snapshot: bool) -> bool:
    """CLV uses consensus at market_close_at.

    No closing snapshot means the signal is excluded from CLV but kept in
    other metrics. This helper encodes that rule for downstream consumers.
    """

    return bool(has_close_snapshot)
