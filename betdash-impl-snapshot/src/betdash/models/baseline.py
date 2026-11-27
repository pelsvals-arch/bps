"""Baseline rating models.

Ratings are per-sport in the trainer; these classes hold rating state and
provide probability(a,b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt


@dataclass
class EloRating:
    k: float = 20.0
    base: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, player: str) -> float:
        return float(self.ratings.get(player, self.base))

    def probability(self, a: str, b: str) -> float:
        ra = self.get(a)
        rb = self.get(b)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, score_a: float) -> None:
        pa = self.probability(a, b)
        ra = self.get(a)
        rb = self.get(b)
        self.ratings[a] = ra + self.k * (score_a - pa)
        self.ratings[b] = rb + self.k * ((1.0 - score_a) - (1.0 - pa))


@dataclass
class GlickoState:
    rating: float = 1500.0
    rd: float = 350.0


@dataclass
class GlickoRating:
    """A lightweight Glicko-like rating sufficient for baseline probabilities.

    Uses Glicko-1 style g(RD) scaling with fixed volatility for MVP.
    """

    base: float = 1500.0
    q: float = field(default=0.00575646273, init=False)  # ln(10)/400
    ratings: dict[str, GlickoState] = field(default_factory=dict)

    def get(self, player: str) -> GlickoState:
        state = self.ratings.get(player)
        if state is None:
            state = GlickoState(rating=self.base, rd=350.0)
            self.ratings[player] = state
        return state

    @staticmethod
    def _g(rd: float, q: float) -> float:
        return 1.0 / sqrt(1.0 + (3.0 * q * q * rd * rd) / (pi * pi))

    def probability(self, a: str, b: str) -> float:
        ra = self.get(a)
        rb = self.get(b)
        g = self._g(sqrt(ra.rd * ra.rd + rb.rd * rb.rd), self.q)
        return 1.0 / (1.0 + 10 ** (-g * (ra.rating - rb.rating) / 400.0))

    def update(self, a: str, b: str, score_a: float) -> None:
        # Minimal update: adjust rating by expected outcome error, shrink RD slightly.
        pa = self.probability(a, b)
        ra = self.get(a)
        rb = self.get(b)
        k = 24.0
        ra.rating = ra.rating + k * (score_a - pa)
        rb.rating = rb.rating + k * ((1.0 - score_a) - (1.0 - pa))
        ra.rd = max(30.0, ra.rd * 0.98)
        rb.rd = max(30.0, rb.rd * 0.98)
