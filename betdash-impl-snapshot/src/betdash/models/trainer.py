"""Resumable incremental trainer for baseline models."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from apscheduler.schedulers.background import BackgroundScheduler

from betdash.backtest.engine import BacktestEngine
from betdash.config import Settings
from betdash.db.repository import Repository
from betdash.models.baseline import EloRating, GlickoRating
from betdash.models.features import build_features

FEATURES_VERSION = "baseline-v1"


@dataclass(frozen=True, slots=True)
class ModelRun:
    run_id: str
    features_version: str
    created_at: str


def _to_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class Trainer:
    """Incremental walk-forward trainer with atomic checkpoint resume."""

    def __init__(self, repo: Repository, settings: Settings):
        self.repo = repo
        self.settings = settings
        self.checkpoint_dir = Path(settings.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "trainer_checkpoint.json"
        self._lock = threading.Lock()
        self._scheduler: BackgroundScheduler | None = None
        self._elo: dict[str, EloRating] = {}
        self._glicko: dict[str, GlickoRating] = {}
        self._model: Any | None = None
        self._state: dict[str, Any] = {
            "last_run_id": None,
            "features_version": FEATURES_VERSION,
            "last_snapshot_id": 0,
            "last_result_id": 0,
            "runs": 0,
            "processed_event_uids": [],
            "processed_count": 0,
            "processed_rows": [],
        }
        self._load_checkpoint()

    @property
    def last_run_id(self) -> str | None:
        value = self._state.get("last_run_id")
        return str(value) if value else None

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict):
            self._state.update(data)
        self._state.setdefault("processed_event_uids", [])
        self._state.setdefault("processed_count", 0)
        self._state.setdefault("processed_rows", [])
        self._restore_ratings()

    def _restore_ratings(self) -> None:
        elo_state = self._state.get("elo") or {}
        glicko_state = self._state.get("glicko") or {}
        self._elo = {}
        self._glicko = {}
        if isinstance(elo_state, dict):
            for sport, ratings in elo_state.items():
                if isinstance(ratings, dict):
                    er = EloRating()
                    er.ratings = {str(k): float(v) for k, v in ratings.items()}
                    self._elo[str(sport)] = er
        if isinstance(glicko_state, dict):
            for sport, ratings in glicko_state.items():
                gr = GlickoRating()
                if isinstance(ratings, dict):
                    for player, st in ratings.items():
                        if isinstance(st, dict):
                            gs = gr.get(str(player))
                            gs.rating = float(st.get("rating", gs.rating))
                            gs.rd = float(st.get("rd", gs.rd))
                self._glicko[str(sport)] = gr

    def _serialize_ratings(self, elo: dict[str, EloRating] | None = None, glicko: dict[str, GlickoRating] | None = None) -> None:
        elo = self._elo if elo is None else elo
        glicko = self._glicko if glicko is None else glicko
        self._state["elo"] = {
            sport: {k: float(v) for k, v in r.ratings.items()}
            for sport, r in elo.items()
        }
        self._state["glicko"] = {
            sport: {
                k: {"rating": float(s.rating), "rd": float(s.rd)}
                for k, s in r.ratings.items()
            }
            for sport, r in glicko.items()
        }

    def _atomic_write_checkpoint(self) -> None:
        self._serialize_ratings()
        payload = json.dumps(self._state, ensure_ascii=False, sort_keys=True)
        tmp_path = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.checkpoint_path)

    def _watermarks(self) -> tuple[int, int]:
        with self.repo._connect() as conn:
            snap = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM odds_snapshots").fetchone()
            res = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM results").fetchone()
        return int(snap["m"] if snap else 0), int(res["m"] if res else 0)

    @staticmethod
    def _norm_pid(value: str) -> str:
        from betdash.normalize.registry import normalize_name
        text = str(value or "")
        for prefix in ("name:", "unresolved:", "ambiguous:", "resolved:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return normalize_name(text)

    @staticmethod
    def _actual_for_event(event: Any, result: dict[str, Any]) -> int | None:
        parts = tuple(str(p) for p in (getattr(event, "participant_ids", None) or ()))
        if len(parts) != 2:
            return None
        p1, p2 = parts
        pa = str(result.get("participant_a") or "")
        pb = str(result.get("participant_b") or "")
        score = str(result.get("score") or "")
        side = BacktestEngine.winner_side_from_score(score)
        n1, n2, na, nb = Trainer._norm_pid(p1), Trainer._norm_pid(p2), Trainer._norm_pid(pa), Trainer._norm_pid(pb)
        if not all([n1, n2, na, nb]):
            return None
        # tennis-data rows are Winner/Loser; retired/tied-score rows may have
        # empty or non-decisive score but still have a valid winner.
        if side is None:
            import re
            score_norm = score.strip()
            nums = re.findall(r"\d+", score_norm)
            if score_norm and not (len(nums) == 2 and nums[0] == nums[1]):
                return None
            side = "left"
        if na == n1 and nb == n2:
            ordered = ("p1", "p2")
        elif na == n2 and nb == n1:
            ordered = ("p2", "p1")
        else:
            return None
        winner = ordered[0] if side == "left" else ordered[1]
        return 1 if winner == "p1" else 0

    def _event_decision_time(self, event_uid: str, start_at: datetime) -> datetime | None:
        with self.repo._connect() as conn:
            rows = conn.execute(
                """
                SELECT observed_at, is_live
                FROM odds_snapshots
                WHERE event_uid = ?
                ORDER BY observed_at, id
                """,
                (event_uid,),
            ).fetchall()
        times: list[datetime] = []
        for row in rows:
            if bool(row["is_live"]):
                continue
            dt = _to_dt(row["observed_at"])
            if dt is None or dt >= start_at:
                continue
            times.append(dt)
        if not times:
            return None
        return max(times)

    def _collect_new_rows(
        self,
        elo: dict[str, EloRating],
        glicko: dict[str, GlickoRating],
        processed_uids: set[str],
    ) -> list[dict[str, Any]]:
        events = self.repo.events_for_resolution()
        results = {str(row["event_uid"]): row for row in self.repo.results()}

        candidates: list[tuple[datetime, str, Any, dict[str, Any], datetime]] = []
        for event in events:
            if event.event_uid in processed_uids:
                continue
            result = results.get(event.event_uid)
            if result is None:
                continue
            actual = self._actual_for_event(event, result)
            if actual is None:
                continue
            start_at = event.start_at
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
            else:
                start_at = start_at.astimezone(timezone.utc)
            observed_at = self._event_decision_time(event.event_uid, start_at)
            if observed_at is None:
                # Historical result-only rows (e.g. imported tennis-data) have no odds snapshots.
                # Use pre-start decision time; features must remain as-of before event.
                from datetime import timedelta
                observed_at = start_at - timedelta(minutes=1)
            candidates.append((start_at, event.event_uid, event, result, observed_at))

        candidates.sort(key=lambda x: (x[0], x[1]))

        new_rows: list[dict[str, Any]] = []
        for start_at, _uid, event, _result, observed_at in candidates:
            sport = str(event.sport)
            p1 = event.participant_ids[0] if len(event.participant_ids) > 0 else ""
            p2 = event.participant_ids[1] if len(event.participant_ids) > 1 else ""

            # Rating features as-of before this event outcome is applied.
            er = elo.setdefault(sport, EloRating())
            gr = glicko.setdefault(sport, GlickoRating())
            features = build_features(event, self.repo, as_of=observed_at)
            features["elo_p1"] = float(er.probability(p1, p2)) if p1 and p2 else 0.5
            features["glicko_p1"] = float(gr.probability(p1, p2)) if p1 and p2 else 0.5

            actual = self._actual_for_event(event, results[event.event_uid])
            if actual is None:
                continue
            new_rows.append(
                {
                    "event_uid": event.event_uid,
                    "sport": sport,
                    "observed_at": observed_at,
                    "event_start_at": start_at,
                    "features": features,
                    "actual": int(actual),
                    "p1": p1,
                    "p2": p2,
                }
            )

            # Update rating copies after recording features/label; committed on success only.
            if p1 and p2:
                er.update(p1, p2, float(actual))
                gr.update(p1, p2, float(actual))

        return new_rows

    def _fit_model(self, rows: list[dict[str, Any]]) -> Any:
        if not rows:
            return None
        feature_names = sorted({k for r in rows for k in r["features"].keys()})
        X = [[float(r["features"].get(k, 0.0)) for k in feature_names] for r in rows]
        y = [int(r["actual"]) for r in rows]

        if self.settings.enable_catboost:
            try:
                from catboost import CatBoostClassifier  # type: ignore

                if len(set(y)) >= 2:
                    model = CatBoostClassifier(
                        iterations=50,
                        depth=4,
                        learning_rate=0.1,
                        verbose=False,
                        allow_writing_files=False,
                    )
                    model.fit(X, y)
                    return {"type": "catboost", "model": model, "features": feature_names}
            except Exception:
                pass

        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            return None

        if len(set(y)) < 2:
            return {"type": "constant", "p": sum(y) / len(y), "features": feature_names}
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        return {"type": "logreg", "model": model, "features": feature_names}

    @staticmethod
    def _predict_with_model(model: Any, features: dict[str, float]) -> float:
        if not model:
            return 0.5
        if isinstance(model, dict) and model.get("type") == "constant":
            return float(model.get("p", 0.5))
        if isinstance(model, dict) and model.get("type") in {"logreg", "catboost"}:
            names = model["features"]
            clf = model["model"]
            X = [[float(features.get(k, 0.0)) for k in names]]
            try:
                prob = float(clf.predict_proba(X)[0][1])
            except Exception:
                prob = 0.5
            return min(max(prob, 1e-6), 1 - 1e-6)
        return 0.5

    @staticmethod
    def _implied_probs_from_outcomes(outcomes: dict[str, tuple[float, str | None]]) -> dict[str, float]:
        inv: dict[str, float] = {}
        for k, v in outcomes.items():
            try:
                odds = float(v[0])
            except Exception:
                continue
            if odds > 1.01 and math.isfinite(odds):
                inv[k] = 1.0 / odds
        total = sum(inv.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in inv.items()}

    def _market_probs(
        self,
        market: str,
        line: float | None,
        outcomes: dict[str, tuple[float, str | None]],
        p1_prob: float,
        model: Any,
    ) -> dict[str, float]:
        implied = self._implied_probs_from_outcomes(outcomes)
        market = str(market)
        # match winner: blend model/Elo baseline with devig market prior when available
        if market == "match_winner":
            base = min(max(float(p1_prob), 1e-6), 1 - 1e-6)
            if model is None and implied:
                p1 = float(implied.get("p1", base))
                p2 = float(implied.get("p2", 1.0 - p1))
                s = p1 + p2
                return {"p1": p1 / s, "p2": p2 / s} if s > 0 else {"p1": base, "p2": 1 - base}
            if implied:
                p1_m = float(implied.get("p1", base))
                p2_m = float(implied.get("p2", 1.0 - p1_m))
                sm = p1_m + p2_m
                if sm > 0:
                    p1_m, p2_m = p1_m / sm, p2_m / sm
                    # research blend: model 70%, market devig 30% when model exists
                    p1 = 0.7 * base + 0.3 * p1_m
                    p1 = min(max(p1, 1e-6), 1 - 1e-6)
                    return {"p1": p1, "p2": 1 - p1}
            return {"p1": base, "p2": 1 - base}

        # totals/spreads/handicaps: conservative baseline from devig odds + slight p1 edge when available
        if market in {"total_points", "total_games", "total_maps", "spread", "set_handicap", "map_handicap"}:
            if market.startswith("total") or market in {"total_points", "total_games", "total_maps"}:
                keys = ["over", "under"]
            else:
                keys = ["p1", "p2"]
            out: dict[str, float] = {}
            if implied:
                for k in keys:
                    if k in implied:
                        out[k] = float(implied[k])
            for k in keys:
                out.setdefault(k, 0.5)
            # small directional adjustment from p1 edge for two-sided non-total markets
            if keys == ["p1", "p2"]:
                edge = (float(p1_prob) - 0.5) * 0.15
                out["p1"] = min(max(out["p1"] + edge, 1e-6), 1 - 1e-6)
                out["p2"] = 1.0 - out["p1"]
            s = sum(out.values())
            return {k: v / s for k, v in out.items()} if s > 0 else {}

        # generic two-outcome fallback
        if implied:
            s = sum(implied.values())
            return {k: v / s for k, v in implied.items()} if s > 0 else {}
        if len(outcomes) == 2:
            ks = list(outcomes.keys())
            return {ks[0]: 0.5, ks[1]: 0.5}
        return {}

    def _walk_forward(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Explicit expanding-window walk-forward evaluation.

        For each fold, model is trained only on rows before the test window.
        """
        if not rows:
            return {"folds": [], "predictions": 0, "correct": 0, "accuracy": None}

        min_train = 1
        test_window = 1
        folds: list[dict[str, Any]] = []
        predictions = 0
        correct = 0
        i = min_train
        fold_no = 1
        while i < len(rows):
            train_rows = rows[:i]
            test_rows = rows[i : i + test_window]
            if not test_rows:
                break
            model = self._fit_model(train_rows)
            fold_preds = 0
            fold_correct = 0
            for row in test_rows:
                prob = self._predict_with_model(model, row["features"])
                pred = 1 if prob >= 0.5 else 0
                fold_preds += 1
                predictions += 1
                if pred == int(row["actual"]):
                    fold_correct += 1
                    correct += 1
            folds.append(
                {
                    "fold": fold_no,
                    "train_size": len(train_rows),
                    "test_size": len(test_rows),
                    "train_end_event_uid": train_rows[-1]["event_uid"],
                    "test_event_uids": [r["event_uid"] for r in test_rows],
                    "accuracy": (fold_correct / fold_preds) if fold_preds else None,
                }
            )
            i += test_window
            fold_no += 1

        return {
            "folds": folds,
            "predictions": predictions,
            "correct": correct,
            "accuracy": (correct / predictions) if predictions else None,
            "min_train_samples": min_train,
            "test_window": test_window,
        }

    def _write_model_run(self, run: ModelRun, metrics: dict[str, Any] | None = None) -> None:
        with self.repo._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_runs(run_id, features_version, created_at, metrics_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.features_version,
                    run.created_at,
                    json.dumps(metrics or {}, ensure_ascii=False),
                ),
            )

    def _build_metrics(self, processed_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], wf: dict[str, Any]) -> dict[str, Any]:
        from math import log

        rows = [r for r in processed_rows if r.get("actual") in (0, 1)]
        brier = logloss = roi = hit_rate = clv_avg = max_dd = None
        calib_bins: list[dict[str, Any]] = []
        bad_actual = self._count_bad_actual()

        if rows:
            # Use walk-forward-like evaluation: fit on all previous rows in order is expensive;
            # for diagnostic honesty use simple in-sample probability with explicit note in UI,
            # but prefer WF accuracy/preds when present.
            model = self._fit_model(processed_rows)
            probs: list[float] = []
            actuals: list[int] = []
            for r in rows:
                p = self._predict_with_model(model, r.get("features") or {})
                p = min(max(float(p), 1e-6), 1 - 1e-6)
                probs.append(p)
                actuals.append(int(r["actual"]))

            brier = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(rows)
            logloss = -sum(a * log(p) + (1 - a) * log(1 - p) for p, a in zip(probs, actuals)) / len(rows)
            hit_rate = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == bool(a)) / len(rows)

            for i in range(10):
                lo, hi = i / 10, (i + 1) / 10
                bucket = [(p, a) for p, a in zip(probs, actuals) if (lo <= p < hi) or (i == 9 and p == 1.0)]
                if bucket:
                    calib_bins.append({
                        "bin_lo": lo, "bin_hi": hi, "count": len(bucket),
                        "avg_pred": sum(p for p, _ in bucket) / len(bucket),
                        "avg_actual": sum(a for _, a in bucket) / len(bucket),
                    })

            # ROI/CLV/max_drawdown require executable odds/closing line at decision time.
            # Historical result-only imports (tennis-data) usually lack decision-time odds snapshots.
            roi = None
            clv_avg = None
            max_dd = None

        edge_allowed = len(rows) >= 30
        metrics = {
            "samples": len(processed_rows),
            "new_samples": len(new_rows),
            "sports": sorted({str(r.get("sport")) for r in processed_rows}),
            "catboost_enabled": bool(self.settings.enable_catboost),
            "walk_forward": wf,
            "processed_event_uids": [str(r.get("event_uid")) for r in processed_rows],
            "brier": brier,
            "log_loss": logloss,
            "roi": roi,
            "hit_rate": hit_rate,
            "max_drawdown": max_dd,
            "clv": clv_avg,
            "calibration_bins": calib_bins,
            "profit_by_market": {},
            "profit_by_league": {},
            "edge_report_allowed": edge_allowed,
            "min_edge_samples": 30,
            "bad_actual": bad_actual,
            "metrics_note": "Brier/logloss/hit-rate/calibration — модельные метрики на исторических результатах; ROI/CLV/max drawdown требуют odds/closing на момент решения и могут быть n/a. Это симуляция, не рекомендация ставки.",
        }
        if not edge_allowed:
            metrics["edge_warning"] = "minimum sample guard: need >= 30 resolved signals"
        return metrics

    def _count_bad_actual(self) -> int:
        """Count results that exist in DB but cannot be mapped to unambiguous actual."""
        try:
            events = {e.event_uid: e for e in self.repo.events_for_resolution()}
            bad = 0
            for row in self.repo.results():
                ev = events.get(str(row.get("event_uid")))
                if ev is None:
                    continue
                if self._actual_for_event(ev, row) is None:
                    bad += 1
            return bad
        except Exception:
            return 0

    def _generate_predictions(self, run_id: str, model: Any, elo: dict[str, EloRating], glicko: dict[str, GlickoRating]) -> list[dict[str, Any]]:
        events = self.repo.events_for_resolution()
        results = {str(row["event_uid"]) for row in self.repo.results()}
        now = datetime.now(timezone.utc)
        preds: list[dict[str, Any]] = []
        for event in events:
            if event.event_uid in results:
                continue
            if len(event.participant_ids) < 2:
                continue
            start_at = event.start_at
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
            else:
                start_at = start_at.astimezone(timezone.utc)
            seconds_to_start = (start_at - now).total_seconds()
            sport = str(event.sport)
            horizon_hours = 168
            if sport == "tennis":
                horizon_hours = max(int(getattr(self.settings, "tennis_prediction_horizon_hours", 72)), 1)
            # Keep recent 2 days for visibility; future events within horizon.
            if seconds_to_start < -2 * 86400:
                continue
            if seconds_to_start > horizon_hours * 3600:
                continue
            p1, p2 = event.participant_ids[0], event.participant_ids[1]
            er = elo.setdefault(sport, EloRating())
            gr = glicko.setdefault(sport, GlickoRating())
            as_of = now
            try:
                features = build_features(event, self.repo, as_of=as_of)
            except Exception:
                features = {}
            features["elo_p1"] = float(er.probability(p1, p2))
            features["glicko_p1"] = float(gr.probability(p1, p2))
            p1_prob = self._predict_with_model(model, features)
            if model is None:
                # fallback: blend elo/glicko when no trained model
                p1_prob = (features["elo_p1"] + features["glicko_p1"]) / 2.0
            p1_prob = min(max(float(p1_prob), 1e-6), 1 - 1e-6)

            snaps = self.repo.last_snapshots(event.event_uid)
            by_market: dict[tuple[str, float | None], dict[str, tuple[float, str | None]]] = {}
            for snap in snaps:
                line = getattr(snap, "line", None)
                key = (snap.market, float(line) if line is not None else None)
                slot = by_market.setdefault(key, {})
                cur = slot.get(snap.outcome)
                if cur is None or snap.odds_decimal > cur[0]:
                    slot[snap.outcome] = (snap.odds_decimal, snap.bookmaker)

            urgency = 1.0 / (1.0 + max(seconds_to_start, 0) / 3600)
            tennis_bonus = 0.2 if sport == "tennis" else 0.0
            by_market.setdefault(("match_winner", None), by_market.get(("match_winner", None), {}))

            for (market, line), outcomes in by_market.items():
                market_probs = self._market_probs(market, line, outcomes, p1_prob, model)
                if not market_probs:
                    continue
                conf = max(abs(p - 0.5) * 2 for p in market_probs.values())
                priority = urgency + tennis_bonus + conf * 0.1
                for outcome, prob in market_probs.items():
                    odds, bk = outcomes.get(outcome, (None, None))
                    preds.append({
                        "model_run_id": run_id,
                        "event_uid": event.event_uid,
                        "market": market,
                        "line": line,
                        "outcome": outcome,
                        "model_prob": min(max(float(prob), 1e-6), 1 - 1e-6),
                        "odds_decimal": odds,
                        "bookmaker": bk,
                        "is_research_only": 1,
                        "_priority_score": priority,
                        "_seconds_to_start": seconds_to_start,
                        "_sport": sport,
                    })
        # Keep most actionable events: soonest/most urgent, then higher confidence.
        preds.sort(key=lambda r: (-float(r.get("_priority_score", 0.0)), float(r.get("_seconds_to_start", 10**12)), -float(r.get("model_prob", 0.0))))
        max_predictions = max(int(getattr(self.settings, "max_predictions", 2000)), 100)
        # Trim by event to preserve both outcomes.
        kept: list[dict[str, Any]] = []
        seen_events: set[str] = set()
        for r in preds:
            uid = str(r["event_uid"])
            if uid not in seen_events and len(seen_events) >= max_predictions // 2:
                continue
            seen_events.add(uid)
            kept.append(r)
        for r in kept:
            r.pop("_priority_score", None)
            r.pop("_seconds_to_start", None)
            r.pop("_sport", None)
        return kept

    def step(self) -> ModelRun:
        with self._lock:
            snap_id, result_id = self._watermarks()

            processed_rows: list[dict[str, Any]] = copy.deepcopy(list(self._state.get("processed_rows") or []))
            processed_uids = {str(r.get("event_uid")) for r in processed_rows}

            # Work on copies so any exception before commit leaves in-memory state untouched.
            elo_next = copy.deepcopy(self._elo)
            glicko_next = copy.deepcopy(self._glicko)
            new_rows = self._collect_new_rows(elo_next, glicko_next, processed_uids)

            # No new processed events: keep state stable; avoid no-op model_run churn.
            if not new_rows:
                self._model = self._fit_model(processed_rows)
                self._state["last_snapshot_id"] = snap_id
                self._state["last_result_id"] = result_id
                if self.last_run_id is None:
                    run = ModelRun(
                        run_id=str(uuid4()),
                        features_version=FEATURES_VERSION,
                        created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    )
                    metrics = {
                        "samples": len(processed_rows),
                        "new_samples": 0,
                        "sports": sorted({str(r.get("sport")) for r in processed_rows}),
                        "catboost_enabled": bool(self.settings.enable_catboost),
                        "walk_forward": self._walk_forward(processed_rows),
                        "processed_event_uids": [str(r.get("event_uid")) for r in processed_rows],
                        "no_op": True,
                    }
                    self._write_model_run(run, metrics=metrics)
                    preds = self._generate_predictions(run.run_id, self._model, elo_next, glicko_next)
                    try:
                        self.repo.save_predictions(preds)
                    except Exception:
                        pass
                    self._state["last_run_id"] = run.run_id
                    self._state["features_version"] = run.features_version
                    self._state["runs"] = int(self._state.get("runs", 0)) + 1
                    self._state["predictions_count"] = len(preds)
                    self._atomic_write_checkpoint()
                    return run
                # refresh predictions on no-op so dashboard shows current upcoming events
                try:
                    preds = self._generate_predictions(self.last_run_id, self._model, elo_next, glicko_next)
                    self.repo.save_predictions(preds)
                    self._state["predictions_count"] = len(preds)
                except Exception:
                    pass
                self._atomic_write_checkpoint()
                return ModelRun(
                    run_id=self.last_run_id,
                    features_version=str(self._state.get("features_version") or FEATURES_VERSION),
                    created_at=str(self._state.get("last_run_created_at") or datetime.now(timezone.utc).isoformat(timespec="microseconds")),
                )

            for row in new_rows:
                processed_rows.append(
                    {
                        "event_uid": row["event_uid"],
                        "sport": row["sport"],
                        "observed_at": _to_dt(row["observed_at"]).isoformat() if _to_dt(row["observed_at"]) else None,
                        "event_start_at": _to_dt(row["event_start_at"]).isoformat() if _to_dt(row["event_start_at"]) else None,
                        "features": row["features"],
                        "actual": int(row["actual"]),
                        "p1": row["p1"],
                        "p2": row["p2"],
                    }
                )

            # Deduplicate by event_uid preserving order.
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for row in processed_rows:
                uid = str(row.get("event_uid"))
                if uid in seen:
                    continue
                seen.add(uid)
                deduped.append(row)
            processed_rows = deduped

            wf = self._walk_forward(processed_rows)
            model_next = self._fit_model(processed_rows)

            run = ModelRun(
                run_id=str(uuid4()),
                features_version=FEATURES_VERSION,
                created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            )
            metrics = self._build_metrics(processed_rows, new_rows, wf)
            self._write_model_run(run, metrics=metrics)
            preds = self._generate_predictions(run.run_id, model_next, elo_next, glicko_next)
            try:
                self.repo.save_predictions(preds)
            except Exception:
                pass

            # Commit in-memory state only after all operations that can fail succeeded.
            self._elo = elo_next
            self._glicko = glicko_next
            self._model = model_next
            self._state["last_run_id"] = run.run_id
            self._state["last_run_created_at"] = run.created_at
            self._state["features_version"] = run.features_version
            self._state["last_snapshot_id"] = snap_id
            self._state["last_result_id"] = result_id
            self._state["runs"] = int(self._state.get("runs", 0)) + 1
            self._state["processed_event_uids"] = [str(r.get("event_uid")) for r in processed_rows]
            self._state["processed_rows"] = processed_rows
            self._state["processed_count"] = len(processed_rows)
            self._state["predictions_count"] = len(preds)
            self._atomic_write_checkpoint()
            return run

    def start(self) -> None:
        if self._scheduler is not None:
            return
        scheduler = BackgroundScheduler()
        interval = max(int(getattr(self.settings, "poll_seconds", 90)), 1)
        scheduler.add_job(self.step, "interval", seconds=interval, max_instances=1)
        scheduler.start()
        self._scheduler = scheduler

    def stop(self) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        scheduler.shutdown(wait=True)
        self._scheduler = None
        with self._lock:
            pass
