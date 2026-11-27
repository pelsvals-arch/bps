from __future__ import annotations

from datetime import datetime, timezone

from betdash.db.repository import Snapshot
from betdash.markets.consensus import best_price, consensus_prob
from betdash.markets.de_vig import implied_probs


NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def make_snapshot(bookmaker: str, outcome: str, odds_decimal: float, *, market: str = "match_winner", event_uid: str = "e1") -> Snapshot:
    return Snapshot(
        event_uid=event_uid,
        bookmaker=bookmaker,
        market=market,
        outcome=outcome,
        odds_decimal=odds_decimal,
        observed_at=NOW,
        source_event_id=f"{bookmaker}-{outcome}",
        adapter_version="1",
    )


def test_de_vig_removes_margin():
    p = implied_probs([1.90, 2.10])
    assert abs(sum(p) - 1.0) < 1e-9


def test_single_book_consensus_returns_zero_coverage():
    snaps = [make_snapshot("pari", "p1", 2.0)]
    prob, coverage = consensus_prob(snaps, "p1")
    assert coverage == 0 and prob == 0.0


def test_consensus_de_vigs_each_book_then_averages():
    # Book A: 2.0/2.0 -> fair 0.5/0.5. Book B: 1.8/2.2 -> inverses 0.555..,0.4545..
    # normalized p1 = 0.55, p2 = 0.45. Consensus for p1 = (0.5 + 0.55) / 2.
    snaps = [
        make_snapshot("a", "p1", 2.0),
        make_snapshot("a", "p2", 2.0),
        make_snapshot("b", "p1", 1.8),
        make_snapshot("b", "p2", 2.2),
    ]
    prob, coverage = consensus_prob(snaps, "p1")
    assert coverage == 2
    assert abs(prob - 0.525) < 1e-9


def test_consensus_ignores_books_with_fewer_than_two_outcomes():
    snaps = [
        make_snapshot("a", "p1", 2.0),
        make_snapshot("a", "p2", 2.0),
        make_snapshot("b", "p1", 1.5),  # only one outcome present
        make_snapshot("c", "p1", 1.9),
        make_snapshot("c", "p2", 2.1),
    ]
    prob, coverage = consensus_prob(snaps, "p1")
    assert coverage == 2
    assert 0.0 < prob < 1.0


def test_consensus_returns_zero_when_coverage_below_two():
    snaps = [
        make_snapshot("a", "p1", 2.0),
        make_snapshot("a", "p2", 2.0),
        make_snapshot("b", "p1", 1.8),  # missing p2
    ]
    prob, coverage = consensus_prob(snaps, "p1")
    assert prob == 0.0
    assert coverage == 1


def test_consensus_excludes_book_missing_target_outcome():
    snaps = [
        make_snapshot("a", "p1", 2.0),
        make_snapshot("a", "p2", 2.0),
        make_snapshot("b", "p2", 1.8),
        make_snapshot("b", "p3", 2.1),  # has two outcomes, but no p1
        make_snapshot("c", "p1", 1.9),
        make_snapshot("c", "p2", 2.1),
    ]
    prob, coverage = consensus_prob(snaps, "p1")
    assert coverage == 2
    assert 0.0 < prob < 1.0


def test_best_price_returns_maximum_or_none():
    snaps = [
        make_snapshot("a", "p1", 2.0),
        make_snapshot("b", "p1", 2.2),
        make_snapshot("c", "p2", 3.0),
    ]
    assert best_price(snaps, "p1") == 2.2
    assert best_price(snaps, "p2") == 3.0
    assert best_price(snaps, "missing") is None


def test_clv_excluded_without_closing_snapshot():
    from betdash.markets.consensus import clv_included

    assert clv_included(True) is True
    assert clv_included(False) is False
