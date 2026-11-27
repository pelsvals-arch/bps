"""Honest backtest metrics.

Signals with no closing snapshot are excluded from CLV but kept in other metrics.
Edge reporting should be guarded by minimum sample size.
"""

from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Any

from betdash.backtest.engine import BacktestReport, PredictionRecord


MIN_EDGE_SAMPLES = 30


def _safe_prob(p: float) -> float:
    return min(max(float(p), 1e-6), 1 - 1e-6)


def brier(predictions: list[PredictionRecord]) -> float:
    rows = [p for p in predictions if p.actual is not None]
    if not rows:
        return 0.0
    return sum((p.model_prob - p.actual) ** 2 for p in rows) / len(rows)


def log_loss(predictions: list[PredictionRecord]) -> float:
    rows = [p for p in predictions if p.actual is not None]
    if not rows:
        return 0.0
    total = 0.0
    for p in rows:
        prob = _safe_prob(p.model_prob)
        y = int(p.actual)
        total += -(y * log(prob) + (1 - y) * log(1 - prob))
    return total / len(rows)


def roi(predictions: list[PredictionRecord]) -> float:
    rows = [p for p in predictions if p.actual is not None]
    if not rows:
        return 0.0
    profit = 0.0
    for p in rows:
        profit += (p.odds_decimal - 1.0) if p.actual == 1 else -1.0
    return profit / len(rows)


def hit_rate(predictions: list[PredictionRecord]) -> float:
    rows = [p for p in predictions if p.actual is not None]
    if not rows:
        return 0.0
    return sum(1 for p in rows if p.actual == 1) / len(rows)


def clv(predictions: list[PredictionRecord]) -> float:
    rows = [p for p in predictions if p.clv is not None]
    if not rows:
        return 0.0
    return sum(float(p.clv) for p in rows) / len(rows)


def max_drawdown(predictions: list[PredictionRecord]) -> float:
    rows = [p for p in predictions if p.actual is not None]
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for p in rows:
        equity += (p.odds_decimal - 1.0) if p.actual == 1 else -1.0
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return abs(worst)


def calibration_bins(predictions: list[PredictionRecord], bins: int = 10) -> list[dict[str, float | int]]:
    rows = [p for p in predictions if p.actual is not None]
    out: list[dict[str, float | int]] = []
    if not rows:
        return out
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [p for p in rows if (lo <= p.model_prob < hi) or (i == bins - 1 and p.model_prob == 1.0)]
        if not bucket:
            continue
        out.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": len(bucket),
                "avg_pred": sum(p.model_prob for p in bucket) / len(bucket),
                "avg_actual": sum(int(p.actual) for p in bucket) / len(bucket),
            }
        )
    return out


def profit_by_market(predictions: list[PredictionRecord]) -> dict[str, float]:
    groups: dict[str, float] = defaultdict(float)
    for p in predictions:
        if p.actual is None:
            continue
        groups[p.market] += (p.odds_decimal - 1.0) if p.actual == 1 else -1.0
    return dict(groups)


def profit_by_league(predictions: list[PredictionRecord]) -> dict[str, float]:
    groups: dict[str, float] = defaultdict(float)
    for p in predictions:
        if p.actual is None:
            continue
        key = p.league or "unknown"
        groups[key] += (p.odds_decimal - 1.0) if p.actual == 1 else -1.0
    return dict(groups)


def all(report: BacktestReport) -> dict[str, Any]:
    preds = report.predictions
    resolved = [p for p in preds if p.actual is not None]
    edge_ok = len(resolved) >= MIN_EDGE_SAMPLES
    summary = {
        "brier": brier(preds),
        "log_loss": log_loss(preds),
        "clv": clv(preds),
        "roi": roi(preds),
        "hit_rate": hit_rate(preds),
        "max_drawdown": max_drawdown(preds),
        "sample_count": len(resolved),
        "clv_sample_count": report.clv_sample_count,
        "calibration_bins": calibration_bins(preds),
        "profit_by_market": profit_by_market(preds),
        "profit_by_league": profit_by_league(preds),
        "edge_report_allowed": edge_ok,
        "min_edge_samples": MIN_EDGE_SAMPLES,
    }
    if not edge_ok:
        summary["edge_warning"] = f"minimum sample guard: need >= {MIN_EDGE_SAMPLES} resolved signals"
    return summary
