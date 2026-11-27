from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from betdash.backtest.engine import BacktestReport, PredictionRecord, Signal
from betdash.config import Settings
from betdash.db.repository import AdapterHealth
from betdash.risk.engine import ExposureState, RiskDecision, RiskEngine, RiskLimits


T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
START = datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc)


@dataclass
class _Signal:
    event_uid: str = "e1"
    market: str = "match_winner"
    outcome: str = "p1"
    odds_decimal: float = 2.0
    model_prob: float = 0.6
    observed_at: datetime = T0
    event_start_at: datetime = START
    stake: float = 20.0
    league: str = "ATP"
    sport: str = "tennis"
    correlation_key: str = "e1:match_winner"


def healthy_adapter_health():
    return [AdapterHealth("b1", "ok", None, 0, "v1"), AdapterHealth("b2", "ok", None, 0, "v1")]


@dataclass
class _Report:
    predictions: list = field(default_factory=list)
    clv_sample_count: int = 0
    signals: list = field(default_factory=list)
    loss: float = 0.0
    adapter_health: list = field(default_factory=healthy_adapter_health)
    event_exposure: dict = field(default_factory=dict)
    league_exposure: dict = field(default_factory=dict)
    sport_exposure: dict = field(default_factory=dict)
    correlated_exposure: dict = field(default_factory=dict)


def make_signal(stake=20, **kw):
    return _Signal(stake=stake, **kw)


def make_report(loss=120, **kw):
    return _Report(loss=loss, **kw)


def test_daily_loss_limit_skips_signal():
    engine = RiskEngine(Settings(max_daily_loss=100.0))
    decision = engine.evaluate(make_signal(stake=20), make_report(loss=120))
    assert decision.status == "skipped"


def test_max_stake_per_signal_skipped():
    engine = RiskEngine(Settings(), limits=RiskLimits(max_stake_per_signal=10.0))
    decision = engine.evaluate(make_signal(stake=20), make_report(loss=0))
    assert decision.status == "skipped"
    assert "max_stake_per_signal" in decision.reasons


def test_max_exposure_per_event_skipped():
    engine = RiskEngine(Settings(), limits=RiskLimits(max_exposure_per_event=50.0))
    state = ExposureState(event_exposure={"e1": 40.0})
    decision = engine.evaluate(make_signal(stake=20, event_uid="e1"), make_report(loss=0), state=state)
    assert decision.status == "skipped"
    assert "max_exposure_per_event" in decision.reasons


def test_max_exposure_per_league_skipped():
    engine = RiskEngine(Settings(), limits=RiskLimits(max_exposure_per_league=50.0))
    state = ExposureState(league_exposure={"ATP": 40.0})
    decision = engine.evaluate(make_signal(stake=20, league="ATP"), make_report(loss=0), state=state)
    assert decision.status == "skipped"
    assert "max_exposure_per_league" in decision.reasons


def test_max_exposure_per_sport_skipped():
    engine = RiskEngine(Settings(), limits=RiskLimits(max_exposure_per_sport=50.0))
    state = ExposureState(sport_exposure={"tennis": 45.0})
    decision = engine.evaluate(make_signal(stake=10, sport="tennis"), make_report(loss=0), state=state)
    assert decision.status == "skipped"
    assert "max_exposure_per_sport" in decision.reasons


def test_correlation_limit_skips_highly_related_outcomes():
    engine = RiskEngine(Settings(), limits=RiskLimits(max_correlated_exposure=30.0))
    state = ExposureState(correlated_exposure={"e1:match_winner": 25.0})
    decision = engine.evaluate(
        make_signal(stake=10, correlation_key="e1:match_winner"),
        make_report(loss=0),
        state=state,
    )
    assert decision.status == "skipped"
    assert "correlation_limit" in decision.reasons


def test_kill_switch_on_adapter_health_below_threshold():
    engine = RiskEngine(Settings(), limits=RiskLimits(min_adapter_health=0.5))
    health = [
        AdapterHealth("b1", "ok", None, 0, "v1"),
        AdapterHealth("b2", "degraded", None, 3, "v1"),
        AdapterHealth("b3", "down", None, 5, "v1"),
    ]
    decision = engine.evaluate(make_signal(), make_report(loss=0), adapter_health=health)
    assert decision.status == "kill_switch"
    assert "kill_switch_adapter_health" in decision.reasons


def test_kill_switch_on_anomalous_odds():
    engine = RiskEngine(Settings(), limits=RiskLimits(anomalous_odds_low=1.01, anomalous_odds_high=100.0))
    decision = engine.evaluate(make_signal(odds_decimal=500.0), make_report(loss=0))
    assert decision.status == "kill_switch"
    assert "kill_switch_anomalous_odds" in decision.reasons


def test_allowed_signal_returns_decision_with_reason_codes():
    engine = RiskEngine(Settings(max_daily_loss=1000.0), limits=RiskLimits(max_stake_per_signal=100.0))
    decision = engine.evaluate(make_signal(stake=10), make_report(loss=0))
    assert decision.status == "allowed"
    assert decision.reasons == ("allowed",)
    assert isinstance(decision, RiskDecision)


def test_evaluate_accepts_backtest_report_and_signal_types():
    # Interface contract: evaluate(signal: Signal, report: BacktestReport)
    engine = RiskEngine(Settings(max_daily_loss=1000.0))
    sig = Signal(
        event_uid="e1",
        market="match_winner",
        outcome="p1",
        odds_decimal=2.0,
        model_prob=0.6,
        observed_at=T0,
        event_start_at=START,
    )
    report = BacktestReport(predictions=[], clv_sample_count=0, signals=[sig])
    decision = engine.evaluate(sig, report, adapter_health=healthy_adapter_health())
    assert decision.status in {"allowed", "skipped", "kill_switch"}
    assert isinstance(decision.reasons, tuple)


def test_kill_switch_takes_precedence_over_skip_reasons():
    engine = RiskEngine(Settings(max_daily_loss=10.0), limits=RiskLimits(min_adapter_health=0.8))
    health = [AdapterHealth("b1", "down", None, 1, "v1")]
    decision = engine.evaluate(make_signal(odds_decimal=999.0), make_report(loss=999.0), adapter_health=health)
    assert decision.status == "kill_switch"
    assert "kill_switch_adapter_health" in decision.reasons


def test_daily_loss_uses_explicit_state_over_zero_report_loss():
    engine = RiskEngine(Settings(max_daily_loss=100.0))
    state = ExposureState(daily_loss=120.0)
    decision = engine.evaluate(
        make_signal(stake=20),
        make_report(loss=0),
        state=state,
        adapter_health=healthy_adapter_health(),
    )
    assert decision.status == "skipped"
    assert "max_daily_loss" in decision.reasons


def test_daily_loss_uses_max_of_state_and_report_when_state_passed():
    """Stale/zero ExposureState must not bypass a higher report loss."""
    engine = RiskEngine(Settings(max_daily_loss=100.0))
    state = ExposureState(daily_loss=0.0)
    decision = engine.evaluate(
        make_signal(stake=20),
        make_report(loss=120),
        state=state,
        adapter_health=healthy_adapter_health(),
    )
    assert decision.status == "skipped"
    assert "max_daily_loss" in decision.reasons


def test_missing_adapter_health_fails_closed():
    engine = RiskEngine(Settings(max_daily_loss=1000.0))

    @dataclass
    class ReportNoHealth:
        predictions: list = field(default_factory=list)
        clv_sample_count: int = 0
        signals: list = field(default_factory=list)
        loss: float = 0.0

    decision = engine.evaluate(make_signal(), ReportNoHealth())
    assert decision.status == "kill_switch"
    assert "kill_switch_adapter_health_unknown" in decision.reasons


def test_empty_adapter_health_fails_closed():
    engine = RiskEngine(Settings(max_daily_loss=1000.0))
    decision = engine.evaluate(make_signal(), make_report(loss=0), adapter_health=[])
    assert decision.status == "kill_switch"
    assert "kill_switch_adapter_health_unknown" in decision.reasons
