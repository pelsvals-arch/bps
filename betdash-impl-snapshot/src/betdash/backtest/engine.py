"""Walk-forward backtest engine.

No random split. Signals use only information observed before decision time.
CLV uses consensus at market_close_at; missing close means CLV excluded but the
signal stays in other metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from betdash.db.repository import Repository, Snapshot
from betdash.markets.consensus import best_price, consensus_prob


@dataclass(frozen=True, slots=True)
class Signal:
    event_uid: str
    market: str
    outcome: str
    odds_decimal: float
    model_prob: float
    observed_at: datetime
    event_start_at: datetime


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    event_uid: str
    market: str
    outcome: str
    odds_decimal: float
    model_prob: float
    observed_at: datetime
    event_start_at: datetime
    actual: int | None = None
    clv: float | None = None
    league: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestReport:
    predictions: list[PredictionRecord]
    clv_sample_count: int
    signals: list[Signal]


class Model(Protocol):
    def fit(self, rows: list[dict[str, Any]]) -> Any: ...

    def predict_prob(self, row: dict[str, Any]) -> float: ...


def _to_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class BacktestEngine:
    def __init__(self, repo: Repository, model: Any, *, train_window: timedelta = timedelta(days=30), test_window: timedelta = timedelta(days=7), embargo: timedelta = timedelta(minutes=5), min_train_samples: int = 1):
        self.repo = repo
        self.model = model
        self.train_window = train_window
        self.test_window = test_window
        self.embargo = embargo
        self.min_train_samples = min_train_samples

    def run(self, start: datetime, end: datetime, sport: str) -> BacktestReport:
        return self._run(start, end, sport, duplicated=False)

    def run_with_duplicates(self, start: datetime, end: datetime, sport: str, duplicated: bool) -> BacktestReport:
        # duplicated flag is accepted for interface compatibility; storage dedupe
        # and CLV sampling are always duplicate-safe.
        return self._run(start, end, sport, duplicated=duplicated)

    def _run(self, start: datetime, end: datetime, sport: str, *, duplicated: bool) -> BacktestReport:
        start = _normalize_dt(start)
        end = _normalize_dt(end)
        events = self._events(sport)
        snapshots = self._snapshots()
        results = self._results()

        predictions: list[PredictionRecord] = []
        signals: list[Signal] = []

        # Walk-forward windows anchored by observed_at/signal time. No random split.
        cursor = start
        while cursor < end:
            train_end = cursor
            train_start = cursor - self.train_window
            test_start = cursor
            test_end = min(end, cursor + self.test_window)

            train_rows = self._candidate_rows(events, snapshots, results, train_start, train_end, include_actual=True)
            train_rows = [
                r for r in train_rows
                if r["actual"] is not None and r["event_start_at"] <= train_end - self.embargo
            ]
            if len(train_rows) >= self.min_train_samples and hasattr(self.model, "fit"):
                self.model.fit(train_rows)

            test_rows = self._candidate_rows(events, snapshots, results, test_start, test_end, include_actual=True)
            for row in test_rows:
                model_prob = float(self.model.predict_prob(row))
                model_prob = min(max(model_prob, 1e-6), 1 - 1e-6)
                sig = Signal(
                    event_uid=row["event_uid"],
                    market=row["market"],
                    outcome=row["outcome"],
                    odds_decimal=row["odds_decimal"],
                    model_prob=model_prob,
                    observed_at=row["observed_at"],
                    event_start_at=row["event_start_at"],
                )
                signals.append(sig)

                clv = self._clv(row, snapshots)
                predictions.append(
                    PredictionRecord(
                        event_uid=row["event_uid"],
                        market=row["market"],
                        outcome=row["outcome"],
                        odds_decimal=row["odds_decimal"],
                        model_prob=model_prob,
                        observed_at=row["observed_at"],
                        event_start_at=row["event_start_at"],
                        actual=row["actual"],
                        clv=clv,
                        league=row.get("league"),
                    )
                )

            cursor = test_end
            if self.test_window <= timedelta(0):
                break

        clv_sample_count = len({(p.event_uid, p.market) for p in predictions if p.clv is not None})
        # Leakage guard required by brief: no prediction may use info at/after event start.
        for p in predictions:
            assert p.observed_at < p.event_start_at
        return BacktestReport(predictions=predictions, clv_sample_count=clv_sample_count, signals=signals)

    def _events(self, sport: str) -> dict[str, dict[str, Any]]:
        events = {}
        for e in self.repo.events_for_resolution():
            if e.sport != sport:
                continue
            market_close_at = self._market_close_at(e.event_uid)
            events[e.event_uid] = {
                "event_uid": e.event_uid,
                "sport": e.sport,
                "league": e.league,
                "participant_ids": e.participant_ids,
                "start_at": _normalize_dt(e.start_at),
                "market_close_at": market_close_at,
            }
        return events

    def _market_close_at(self, event_uid: str) -> datetime | None:
        close = self.repo.market_close_at(event_uid)
        return _normalize_dt(close) if close is not None else None

    def _snapshots(self) -> list[Snapshot]:
        with self.repo._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_uid, bookmaker, market, outcome, odds_decimal, observed_at,
                       source_event_id, source_url, adapter_version, is_live
                FROM odds_snapshots
                ORDER BY observed_at, id
                """
            ).fetchall()
        snaps: list[Snapshot] = []
        for row in rows:
            snaps.append(
                Snapshot(
                    event_uid=row["event_uid"],
                    bookmaker=row["bookmaker"],
                    market=row["market"],
                    outcome=row["outcome"],
                    odds_decimal=row["odds_decimal"],
                    observed_at=_to_dt(row["observed_at"]),
                    source_event_id=row["source_event_id"] or "",
                    adapter_version=row["adapter_version"],
                    is_live=bool(row["is_live"]),
                    source_url=row["source_url"],
                )
            )
        return snaps

    def _results(self) -> dict[str, dict[str, Any]]:
        rows = self.repo.results()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            out[row["event_uid"]] = row
        return out

    def _candidate_rows(self, events: dict[str, dict[str, Any]], snapshots: list[Snapshot], results: dict[str, dict[str, Any]], window_start: datetime, window_end: datetime, *, include_actual: bool) -> list[dict[str, Any]]:
        # Group snapshots by event/market/outcome and decision time candidates.
        by_event: dict[str, list[Snapshot]] = {}
        for s in snapshots:
            if s.is_live:
                continue
            by_event.setdefault(s.event_uid, []).append(s)

        rows: list[dict[str, Any]] = []
        for event_uid, event in events.items():
            event_snaps = by_event.get(event_uid, [])
            if not event_snaps:
                continue
            start_at = event["start_at"]
            # decision times are observed_at values before start and before embargo
            decision_times = sorted({s.observed_at for s in event_snaps if s.observed_at < start_at - self.embargo})
            for observed_at in decision_times:
                if not (window_start <= observed_at < window_end):
                    continue
                available = [s for s in event_snaps if s.observed_at <= observed_at]
                # Consider each outcome present at decision time.
                outcomes = sorted({s.outcome for s in available if s.observed_at == observed_at})
                for outcome in outcomes:
                    market = next(s.market for s in available if s.outcome == outcome and s.observed_at == observed_at)
                    market_snaps = [s for s in available if s.market == market]
                    consensus, coverage = consensus_prob(market_snaps, outcome)
                    if coverage < 2:
                        continue
                    price = best_price([s for s in market_snaps if s.observed_at == observed_at], outcome)
                    if price is None:
                        continue
                    row = {
                        "event_uid": event_uid,
                        "market": market,
                        "outcome": outcome,
                        "odds_decimal": float(price),
                        "observed_at": observed_at,
                        "event_start_at": start_at,
                        "coverage_count": coverage,
                        "consensus_prob": consensus,
                        "actual": None,
                        "league": event.get("league"),
                        "market_close_at": event.get("market_close_at"),
                        "participant_ids": event.get("participant_ids") or (),
                    }
                    actual = self._actual_for_row(row, results.get(event_uid)) if include_actual else None
                    if include_actual and actual is None:
                        continue
                    row["actual"] = actual
                    rows.append(row)
        # deterministic order
        rows.sort(key=lambda r: (r["observed_at"], r["event_uid"], r["market"], r["outcome"]))
        return rows

    def _actual_for_row(self, row: dict[str, Any], result: dict[str, Any] | None) -> int | None:
        if result is None:
            return None
        market = str(row.get("market") or "")
        outcome = str(row.get("outcome") or "")
        parts = tuple(str(p) for p in (row.get("participant_ids") or ()))
        if market not in {"match_winner", "moneyline", "winner"}:
            return None
        if outcome not in {"p1", "p2"} or len(parts) != 2:
            return None

        p1, p2 = parts[0], parts[1]
        pa = str(result.get("participant_a") or "")
        pb = str(result.get("participant_b") or "")
        score = str(result.get("score") or "")

        def _norm_pid(v: str) -> str:
            from betdash.normalize.registry import normalize_name
            text = str(v or "")
            for prefix in ("name:", "unresolved:", "ambiguous:", "resolved:"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            return normalize_name(text)

        n1, n2, na, nb = _norm_pid(p1), _norm_pid(p2), _norm_pid(pa), _norm_pid(pb)
        if not all([n1, n2, na, nb]):
            return None

        winner_side = self._winner_side_from_score(score)
        # tennis-data/retired rows may have empty/tied score; if Winner/Loser map
        # unambiguously to participants, treat participant_a as winner.
        if winner_side is None:
            score_norm = str(score or "").strip()
            nums = re.findall(r"\d+", score_norm)
            if score_norm and not (len(nums) == 2 and nums[0] == nums[1]):
                return None
            winner_side = "left"

        ordered = None
        if na == n1 and nb == n2:
            ordered = ("p1", "p2")
        elif na == n2 and nb == n1:
            ordered = ("p2", "p1")
        else:
            return None

        winner_outcome = ordered[0] if winner_side == "left" else ordered[1]
        return 1 if outcome == winner_outcome else 0

    @staticmethod
    def winner_side_from_score(score: str) -> str | None:
        """Return 'left' or 'right' winner side from a numeric score string."""
        # Parse full numeric score tokens; supports "101-110", "2:0", "2-1 (6-4,4-6,6-3)".
        import re

        nums = [int(n) for n in re.findall(r"\d+", score)]
        if len(nums) < 2 or len(nums) % 2 != 0:
            return None
        left_total = 0
        right_total = 0
        left_wins = 0
        right_wins = 0
        for i in range(0, len(nums), 2):
            l, r = nums[i], nums[i + 1]
            left_total += l
            right_total += r
            if l > r:
                left_wins += 1
            elif r > l:
                right_wins += 1
        if left_wins == right_wins:
            if left_total == right_total:
                return None
            return "left" if left_total > right_total else "right"
        return "left" if left_wins > right_wins else "right"

    @staticmethod
    def _winner_side_from_score(score: str) -> str | None:
        # Backward-compatible alias for internal callers/tests.
        return BacktestEngine.winner_side_from_score(score)

    def _clv(self, row: dict[str, Any], snapshots: list[Snapshot]) -> float | None:
        close_at = row.get("market_close_at")
        if close_at is None:
            return None
        close_at = _normalize_dt(close_at)
        event_uid = row["event_uid"]
        market = row["market"]
        outcome = row["outcome"]
        # Use snapshots observed at or before market_close_at, preferring latest at close.
        close_snaps = [s for s in snapshots if s.event_uid == event_uid and s.market == market and s.observed_at <= close_at]
        if not close_snaps:
            return None
        # Require an actual close snapshot at market_close_at for this event/market.
        has_close = any(s.observed_at == close_at for s in close_snaps)
        if not has_close:
            return None
        consensus_close, coverage = consensus_prob(close_snaps, outcome)
        if coverage < 2 or consensus_close <= 0:
            return None
        # CLV as decimal odds improvement over closing consensus fair odds.
        close_fair_odds = 1.0 / consensus_close
        return (float(row["odds_decimal"]) / close_fair_odds) - 1.0
