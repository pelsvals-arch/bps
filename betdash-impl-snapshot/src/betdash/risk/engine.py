"""Paper risk engine: decision with reason codes only, no execution layer.

Consumes: Settings, Signal/BacktestReport (and Snapshot-derived exposure can be
aggregated by callers into ExposureState before evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from betdash.backtest.engine import BacktestReport, Signal
from betdash.config import Settings
from betdash.db.repository import AdapterHealth

DecisionStatus = Literal["allowed", "skipped", "kill_switch"]


@dataclass(frozen=True, slots=True)
class RiskDecision:
    status: DecisionStatus
    reasons: tuple[str, ...] = ()
    stake: float | None = None


@dataclass(frozen=True, slots=True)
class ExposureState:
    """Current paper exposure snapshot used for limit checks."""

    event_exposure: dict[str, float] = field(default_factory=dict)
    league_exposure: dict[str, float] = field(default_factory=dict)
    sport_exposure: dict[str, float] = field(default_factory=dict)
    daily_loss: float = 0.0
    correlated_exposure: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_stake_per_signal: float = float("inf")
    max_exposure_per_event: float = float("inf")
    max_exposure_per_league: float = float("inf")
    max_exposure_per_sport: float = float("inf")
    max_daily_loss: float = float("inf")
    max_correlated_exposure: float = float("inf")
    min_adapter_health: float = 0.5
    anomalous_odds_low: float = 1.01
    anomalous_odds_high: float = 100.0


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stake(signal: Any) -> float:
    value = _get(signal, "stake", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _health_ratio(health: Iterable[AdapterHealth] | None) -> tuple[float | None, bool]:
    """Return (healthy_ratio, health_known).

    Missing or empty adapter health is treated as unknown/fail-closed by callers.
    """

    if health is None:
        return None, False
    rows = list(health)
    if not rows:
        return None, False
    ok = 0
    for row in rows:
        status = str(getattr(row, "status", "")).lower()
        if status in {"ok", "healthy", "up"}:
            ok += 1
    return ok / len(rows), True


class RiskEngine:
    """Evaluate paper signals against configured risk limits.

    No execution layer. Returns decisions with reason codes only.
    """

    def __init__(self, settings: Settings | None = None, limits: RiskLimits | None = None):
        self.settings = settings or Settings()
        base = limits or RiskLimits()
        self.limits = RiskLimits(
            max_stake_per_signal=_get(self.settings, "max_stake_per_signal", base.max_stake_per_signal),
            max_exposure_per_event=_get(self.settings, "max_exposure_per_event", base.max_exposure_per_event),
            max_exposure_per_league=_get(self.settings, "max_exposure_per_league", base.max_exposure_per_league),
            max_exposure_per_sport=_get(self.settings, "max_exposure_per_sport", base.max_exposure_per_sport),
            max_daily_loss=_get(self.settings, "max_daily_loss", base.max_daily_loss),
            max_correlated_exposure=_get(self.settings, "max_correlated_exposure", base.max_correlated_exposure),
            min_adapter_health=_get(self.settings, "min_adapter_health", base.min_adapter_health),
            anomalous_odds_low=_get(self.settings, "anomalous_odds_low", base.anomalous_odds_low),
            anomalous_odds_high=_get(self.settings, "anomalous_odds_high", base.anomalous_odds_high),
        )

    def evaluate(
        self,
        signal: Signal | Any,
        report: BacktestReport | Any = None,
        *,
        state: ExposureState | None = None,
        adapter_health: Iterable[AdapterHealth] | None = None,
    ) -> RiskDecision:
        reasons: list[str] = []
        stake = _stake(signal)
        exposure = state or self._state_from_report(report)
        health_rows = adapter_health if adapter_health is not None else _get(report, "adapter_health", None)

        # Kill switch checks. Missing/empty adapter health is fail-closed:
        # we cannot prove the data plane is healthy, so do not allow signals.
        ratio, health_known = _health_ratio(health_rows)
        if not health_known:
            reasons.append("kill_switch_adapter_health_unknown")
        elif ratio is not None and ratio < float(self.limits.min_adapter_health):
            reasons.append("kill_switch_adapter_health")
        odds = _get(signal, "odds_decimal", None)
        try:
            odds_f = float(odds) if odds is not None else None
        except (TypeError, ValueError):
            odds_f = None
        if odds_f is not None and (
            odds_f < float(self.limits.anomalous_odds_low) or odds_f > float(self.limits.anomalous_odds_high)
        ):
            reasons.append("kill_switch_anomalous_odds")
        if reasons:
            return RiskDecision(status="kill_switch", reasons=tuple(reasons), stake=stake)

        # Skip/limit checks.
        if stake > float(self.limits.max_stake_per_signal):
            reasons.append("max_stake_per_signal")

        event_uid = str(_get(signal, "event_uid", ""))
        league = str(_get(signal, "league", _get(signal, "competition", "")) or "")
        sport = str(_get(signal, "sport", "") or "")
        correlation_key = str(_get(signal, "correlation_key", f"{event_uid}:{_get(signal, 'market', '')}") or "")

        if event_uid:
            projected = exposure.event_exposure.get(event_uid, 0.0) + stake
            if projected > float(self.limits.max_exposure_per_event):
                reasons.append("max_exposure_per_event")
        if league:
            projected = exposure.league_exposure.get(league, 0.0) + stake
            if projected > float(self.limits.max_exposure_per_league):
                reasons.append("max_exposure_per_league")
        if sport:
            projected = exposure.sport_exposure.get(sport, 0.0) + stake
            if projected > float(self.limits.max_exposure_per_sport):
                reasons.append("max_exposure_per_sport")

        # Conservative daily-loss policy: use the worst known daily loss.
        # Explicit ExposureState is often the paper-position source of truth, but
        # a stale/zero state must not bypass a higher report loss. When state is
        # passed, take max(state.daily_loss, report.loss); otherwise use report loss.
        report_loss = float(_get(report, "loss", 0.0) or 0.0)
        daily_loss = max(float(exposure.daily_loss), report_loss) if state is not None else report_loss
        if daily_loss > float(self.limits.max_daily_loss):
            reasons.append("max_daily_loss")

        if correlation_key:
            projected = exposure.correlated_exposure.get(correlation_key, 0.0) + stake
            if projected > float(self.limits.max_correlated_exposure):
                reasons.append("correlation_limit")

        if reasons:
            # preserve order, dedupe
            deduped = tuple(dict.fromkeys(reasons))
            return RiskDecision(status="skipped", reasons=deduped, stake=stake)

        return RiskDecision(status="allowed", reasons=("allowed",), stake=stake)

    @staticmethod
    def _state_from_report(report: Any) -> ExposureState:
        if report is None:
            return ExposureState()
        daily_loss = float(_get(report, "loss", 0.0) or 0.0)
        # Best-effort exposure extraction from report if present.
        event_exposure = dict(_get(report, "event_exposure", {}) or {})
        league_exposure = dict(_get(report, "league_exposure", {}) or {})
        sport_exposure = dict(_get(report, "sport_exposure", {}) or {})
        correlated_exposure = dict(_get(report, "correlated_exposure", {}) or {})
        return ExposureState(
            event_exposure=event_exposure,
            league_exposure=league_exposure,
            sport_exposure=sport_exposure,
            daily_loss=daily_loss,
            correlated_exposure=correlated_exposure,
        )
