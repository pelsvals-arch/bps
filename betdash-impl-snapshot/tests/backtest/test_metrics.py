from __future__ import annotations

from datetime import datetime, timezone

from betdash.backtest.engine import BacktestReport, PredictionRecord, Signal
from betdash.backtest import metrics


NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
START = datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc)


def test_metrics_all_returns_required_keys():
    pred = PredictionRecord(
        event_uid="e1",
        market="match_winner",
        outcome="p1",
        odds_decimal=2.0,
        model_prob=0.6,
        observed_at=NOW,
        event_start_at=START,
        actual=1,
        clv=0.05,
    )
    report = BacktestReport(predictions=[pred], clv_sample_count=1, signals=[])
    result = metrics.all(report)
    for key in ["brier", "log_loss", "clv", "roi", "hit_rate", "max_drawdown"]:
        assert key in result
