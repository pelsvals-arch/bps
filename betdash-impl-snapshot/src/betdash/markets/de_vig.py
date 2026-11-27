"""De-vig utilities for normalizing bookmaker margins."""

from __future__ import annotations


def implied_probs(odds: list[float]) -> list[float]:
    """Normalize inverse decimal odds so probabilities sum to 1.

    This removes the bookmaker margin proportionally across outcomes.
    """
    if not odds:
        return []
    if any(o <= 0 for o in odds):
        raise ValueError("odds must be positive")

    inverses = [1.0 / o for o in odds]
    total = sum(inverses)
    if total <= 0:
        raise ValueError("invalid odds total")
    return [inv / total for inv in inverses]
