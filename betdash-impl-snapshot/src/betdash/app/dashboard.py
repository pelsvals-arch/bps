"""Streamlit dashboard runtime.

The dashboard starts background collector and trainer workers once per process
and keeps them running while the dashboard is open. Workers are created through
``st.cache_resource`` so Streamlit reruns do not create duplicates.
"""

from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from betdash.backtest.engine import BacktestEngine
from betdash.config import Settings
from betdash.db.repository import Repository
from betdash.ingest.collector import Collector
from betdash.models.trainer import Trainer
from betdash.normalize import EntityRegistry
from betdash.risk.engine import ExposureState, RiskEngine


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_import_streamlit():
    import streamlit as st  # type: ignore

    return st


def _safe_import_plotly_express():
    import plotly.express as px  # type: ignore

    return px


def default_adapters(settings: Settings) -> list[Any]:
    """Build default public adapters.

    Imports are local so smoke checks/tests can patch runtime construction without
    importing every adapter dependency chain at module import time.
    """

    adapters: list[Any] = []
    try:
        from betdash.adapters.betboom import BetBoomAdapter

        adapters.append(BetBoomAdapter(settings=settings))
    except Exception:
        pass
    try:
        from betdash.adapters.json_family import JsonFamilyAdapter

        for bookmaker in ("pari", "fonbet", "bettery"):
            try:
                adapters.append(JsonFamilyAdapter(bookmaker=bookmaker, settings=settings))
            except Exception:
                continue
    except Exception:
        pass
    try:
        from betdash.adapters.marathon import MarathonAdapter

        adapters.append(MarathonAdapter(settings=settings))
    except Exception:
        pass
    try:
        from betdash.adapters.winline import WinlineAdapter

        adapters.append(WinlineAdapter(settings=settings))
    except Exception:
        pass
    return adapters


@dataclass
class StreamlitRuntime:
    """Process-local dashboard runtime owning workers and read models."""

    settings: Settings = field(default_factory=Settings)
    repository: Repository | None = None
    collector: Collector | None = None
    trainer: Trainer | None = None
    adapters: list[Any] = field(default_factory=list)
    registry: EntityRegistry = field(default_factory=lambda: EntityRegistry({}))
    risk_engine: RiskEngine | None = None
    started: bool = False
    _start_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _atexit_registered: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.repository is None:
            self.repository = Repository(self.settings.db_path)
        if self.risk_engine is None:
            self.risk_engine = RiskEngine(self.settings)
        if not self.adapters:
            self.adapters = default_adapters(self.settings)
        if self.collector is None:
            self.collector = Collector(self.settings, self.repository, self.adapters, self.registry)
        if self.trainer is None:
            self.trainer = Trainer(self.repository, self.settings)

    def start_workers(self) -> None:
        """Start collector and trainer once. Safe to call repeatedly."""

        with self._start_lock:
            if self.started:
                return
            if self.collector is not None:
                self.collector.start()
            if self.trainer is not None:
                self.trainer.start()
            self.started = True
            if not self._atexit_registered:
                atexit.register(self.stop_workers)
                self._atexit_registered = True

    def stop_workers(self) -> None:
        with self._start_lock:
            if not self.started:
                return
            try:
                if self.collector is not None:
                    self.collector.stop()
            finally:
                if self.trainer is not None:
                    self.trainer.stop()
                self.started = False

    # --- read models for screens / checks ---

    def adapter_status(self) -> list[dict[str, Any]]:
        repo = self.repository
        if repo is None:
            rows = []
        elif hasattr(repo, "list_adapter_health"):
            rows = repo.list_adapter_health()
        else:
            rows = repo.adapter_health()
        return [
            {
                "bookmaker": r.bookmaker,
                "status": r.status,
                "last_success": r.last_success.isoformat() if r.last_success else None,
                "error_count": r.error_count,
                "adapter_version": r.adapter_version,
            }
            for r in rows
        ]

    def training_status(self) -> dict[str, Any]:
        trainer = self.trainer
        return {
            "started": self.started,
            "last_run_id": trainer.last_run_id if trainer is not None else None,
            "features_version": getattr(trainer, "_state", {}).get("features_version") if trainer is not None else None,
            "processed_count": getattr(trainer, "_state", {}).get("processed_count", 0) if trainer is not None else 0,
        }

    def check(self) -> dict[str, Any]:
        """Lightweight smoke check payload; no browser required."""
        return {
            "ok": True,
            "workers_started": self.started,
            "adapters": self.adapter_status(),
            "training": self.training_status(),
            "db_path": self.settings.db_path,
            "poll_seconds": self.settings.poll_seconds,
        }

    @staticmethod
    def _event_label(event: Any) -> str:
        parts = getattr(event, "participant_ids", None) or ()
        names: list[str] = []
        for p in parts:
            text = str(p)
            for prefix in ("name:", "unresolved:", "ambiguous:", "resolved:"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            text = text.replace("_", " ").strip()
            if text:
                names.append(text.title() if text.islower() else text)
        if names:
            label = " — ".join(names)
        else:
            label = str(getattr(event, "event_uid", "unknown"))
        start = getattr(event, "start_at", None)
        if start is not None:
            try:
                label += f" ({start:%Y-%m-%d %H:%M UTC})"
            except Exception:
                pass
        return label

    def _event_labels(self, events: list[Any]) -> dict[str, str]:
        return {e.event_uid: self._event_label(e) for e in events}

    def opportunity_feed(self) -> list[dict[str, Any]]:
        repo = self.repository
        if repo is None:
            return []
        try:
            from betdash.markets.consensus import consensus_prob
        except Exception:
            consensus_prob = None  # type: ignore[assignment]

        events = repo.events_for_resolution()
        labels = self._event_labels(events)
        feed: list[dict[str, Any]] = []

        for event in events[:80]:
            snaps = [s for s in repo.last_snapshots(event.event_uid) if getattr(s, "market", None)]
            by_market: dict[tuple[str, float | None], list[Any]] = {}
            for snap in snaps:
                line = getattr(snap, "line", None)
                key = (str(snap.market), float(line) if line is not None else None)
                by_market.setdefault(key, []).append(snap)

            for (market, line), msnaps in by_market.items():
                by_outcome: dict[str, list[Any]] = {}
                for snap in msnaps:
                    by_outcome.setdefault(str(snap.outcome), []).append(snap)

                outcomes_data: list[dict[str, Any]] = []
                for outcome, osnaps in by_outcome.items():
                    best = max((float(r.odds_decimal) for r in osnaps), default=None)
                    best_bk = None
                    if best is not None:
                        for r in osnaps:
                            if float(r.odds_decimal) == best:
                                best_bk = str(r.bookmaker)
                                break
                    c_prob, coverage = (0.0, 0)
                    if consensus_prob is not None:
                        try:
                            c_prob, coverage = consensus_prob(msnaps, outcome)
                        except Exception:
                            c_prob, coverage = (0.0, 0)
                    edge = (best * c_prob - 1.0) if (best is not None and c_prob > 0) else None
                    outcomes_data.append(
                        {
                            "outcome": outcome,
                            "consensus_prob": float(c_prob),
                            "coverage_books": int(coverage),
                            "best_odds": best,
                            "best_bookmaker": best_bk,
                            "edge": edge,
                        }
                    )

                if not outcomes_data:
                    continue

                # weighted pick: highest consensus prob when coverage is adequate; else highest best odds
                with_cov = [o for o in outcomes_data if o["coverage_books"] >= 2 and o["consensus_prob"] > 0]
                if with_cov:
                    pick = max(with_cov, key=lambda o: (o["consensus_prob"], o["edge"] or -999))
                    pick_basis = "consensus"
                else:
                    pick = max(outcomes_data, key=lambda o: (o["best_odds"] or 0.0))
                    pick_basis = "best_odds_fallback"

                total_cov = max((o["coverage_books"] for o in outcomes_data), default=0)
                best_edge = pick["edge"]
                signal = "no_edge"
                if best_edge is not None and pick["coverage_books"] >= 2:
                    if best_edge >= 0.05:
                        signal = "value"
                    elif best_edge > 0:
                        signal = "small_edge"
                elif pick["coverage_books"] < 2:
                    signal = "low_coverage"

                probs_summary = " | ".join(
                    f"{o['outcome']}: {o['consensus_prob']*100:.1f}%"
                    for o in sorted(outcomes_data, key=lambda x: x["consensus_prob"], reverse=True)
                    if o["consensus_prob"] > 0
                )
                odds_summary = " | ".join(
                    f"{o['outcome']}: {o['best_odds']:.3g}" + (f" ({o['best_bookmaker']})" if o['best_bookmaker'] else "")
                    for o in sorted(outcomes_data, key=lambda x: (x["best_odds"] or 0), reverse=True)
                    if o["best_odds"] is not None
                )

                observed_at = max((s.observed_at for s in msnaps if getattr(s, "observed_at", None)), default=None)
                feed.append(
                    {
                        "event": labels.get(event.event_uid, event.event_uid),
                        "sport": event.sport,
                        "league": event.league,
                        "start_at": event.start_at.isoformat() if event.start_at else None,
                        "market": market,
                        "line": line,
                        "pick_outcome": pick["outcome"],
                        "pick_prob_%": round(pick["consensus_prob"] * 100, 1) if pick["consensus_prob"] > 0 else None,
                        "pick_best_odds": pick["best_odds"],
                        "pick_bookmaker": pick["best_bookmaker"],
                        "edge_%": round(pick["edge"] * 100, 2) if pick["edge"] is not None else None,
                        "signal": signal,
                        "coverage_books": pick["coverage_books"],
                        "pick_basis": pick_basis,
                        "probabilities": probs_summary or None,
                        "best_odds_by_outcome": odds_summary or None,
                        "observed_at": observed_at.isoformat() if observed_at else None,
                        "event_uid": event.event_uid,
                    }
                )

        sig_rank = {"value": 0, "small_edge": 1, "low_coverage": 2, "no_edge": 3}
        feed.sort(
            key=lambda r: (
                sig_rank.get(str(r.get("signal", "no_edge")), 9),
                -(float(r.get("edge_%") or -999)),
                -(float(r.get("pick_prob_%") or 0)),
            )
        )
        return feed

    def market_depth(self, event_uid: str | None = None) -> list[dict[str, Any]]:
        repo = self.repository
        if repo is None:
            return []
        events = repo.events_for_resolution()
        labels = self._event_labels(events)
        if event_uid:
            events = [e for e in events if e.event_uid == event_uid]
        rows: list[dict[str, Any]] = []
        for event in events[:20]:
            for snap in repo.last_snapshots(event.event_uid):
                rows.append(
                    {
                        "event": labels.get(event.event_uid, event.event_uid),
                        "sport": event.sport,
                        "league": event.league,
                        "start_at": event.start_at.isoformat() if event.start_at else None,
                        "bookmaker": snap.bookmaker,
                        "market": snap.market,
                        "outcome": snap.outcome,
                        "odds_decimal": snap.odds_decimal,
                        "observed_at": snap.observed_at.isoformat() if snap.observed_at else None,
                        "event_uid": event.event_uid,
                    }
                )
        return rows

    def model_predictions(self, limit: int = 200) -> list[dict[str, Any]]:
        repo = self.repository
        if repo is None or not hasattr(repo, "predictions"):
            return []
        try:
            rows = repo.predictions(limit=limit)
        except Exception:
            return []
        events = []
        try:
            events = repo.events_for_resolution()
        except Exception:
            events = []
        labels = self._event_labels(events)
        by_uid = {e.event_uid: e for e in events}
        out = []
        for r in rows:
            uid = str(r.get("event_uid"))
            ev = by_uid.get(uid)
            out.append({
                "event": labels.get(uid, uid),
                "sport": getattr(ev, "sport", None),
                "league": getattr(ev, "league", None),
                "start_at": getattr(ev.start_at, "isoformat", lambda: None)() if ev and ev.start_at else r.get("event_start_at"),
                "market": r.get("market"),
                "outcome": r.get("outcome"),
                "model_prob_%": round(float(r.get("model_prob", 0)) * 100, 2),
                "odds_decimal": r.get("odds_decimal"),
                "model_run_id": r.get("model_run_id"),
                "observed_at": r.get("observed_at"),
                "event_uid": uid,
            })
        return out

    def model_diagnostics(self) -> dict[str, Any]:
        repo = self.repository
        if repo is None:
            return {"runs": [], "training": self.training_status(), "predictions": []}
        rows = []
        try:
            with repo._connect() as conn:
                rows = conn.execute(
                    "SELECT run_id, features_version, created_at, metrics_json FROM model_runs ORDER BY created_at DESC LIMIT 20"
                ).fetchall()
        except Exception:
            rows = []
        runs = []
        for row in rows:
            metrics = {}
            try:
                import json

                metrics = json.loads(row["metrics_json"] or "{}")
            except Exception:
                metrics = {}
            runs.append(
                {
                    "run_id": row["run_id"],
                    "features_version": row["features_version"],
                    "created_at": row["created_at"],
                    "metrics": metrics,
                }
            )
        return {"runs": runs, "training": self.training_status(), "predictions": self.model_predictions(limit=100)}

    def backtest_lab(self) -> dict[str, Any]:
        repo = self.repository
        if repo is None:
            return {"report": None}
        events = repo.events_for_resolution()
        results = repo.results()
        # Minimal placeholder backtest using a dummy constant model over available data.
        class _Const:
            def fit(self, rows):
                return self

            def predict_prob(self, row):
                return 0.5

        report = None
        if events and results:
            try:
                engine = BacktestEngine(repo, _Const())
                starts = [e.start_at for e in events if e.start_at is not None]
                if starts:
                    start = min(starts)
                    end = max(starts)
                    if end <= start:
                        from datetime import timedelta

                        end = start + timedelta(days=1)
                    sport = events[0].sport
                    report = engine.run(start, end, sport)
            except Exception:
                report = None
        return {
            "report": report,
            "n_events": len(events),
            "n_results": len(results),
            "paper_only": True,
            "placeholder": True,
        }

    def risk_panel(self) -> dict[str, Any]:
        if self.repository is None:
            health = []
        elif hasattr(self.repository, "list_adapter_health"):
            health = self.repository.list_adapter_health()
        else:
            health = self.repository.adapter_health()
        state = ExposureState(daily_loss=0.0)
        # Paper exposure only; no execution.
        # NOTE: RiskEngine.evaluate() uses max(ExposureState.daily_loss, report.loss)
        # when state is passed to avoid stale-state bypass; keep state current anyway.
        return {
            "paper_only": True,
            "max_daily_loss": self.settings.max_daily_loss,
            "risk_policy": "max(state.daily_loss, report.loss) when state passed; no execution",
            "adapter_health": [
                {"bookmaker": h.bookmaker, "status": h.status, "error_count": h.error_count} for h in health
            ],
            "exposure_state": {
                "daily_loss": state.daily_loss,
                "event_exposure": dict(state.event_exposure),
                "league_exposure": dict(state.league_exposure),
                "sport_exposure": dict(state.sport_exposure),
            },
        }

    # --- Streamlit rendering ---

    def render(self) -> None:
        st = _safe_import_streamlit()
        px = _safe_import_plotly_express()
        st.set_page_config(page_title="BetDash", layout="wide")
        st.title("BetDash — local betting research dashboard")
        st.warning(
            "Симуляция/исследование только. Не рекомендация ставки и не автоматическое исполнение. "
            "EV/ROI/edge на малой выборке нестабильны; проверяйте калибровку, CLV и покрытие рынка.",
            icon="⚠️",
        )
        st.caption("Paper/simulation only. No betting execution. Public pre-match data.")

        pages = {
            "Opportunity feed": self._render_opportunity_feed,
            "Market depth": self._render_market_depth,
            "Model diagnostics": self._render_model_diagnostics,
            "Backtest lab": self._render_backtest_lab,
            "Data health": self._render_data_health,
            "Risk panel": self._render_risk_panel,
        }
        page = st.sidebar.radio("Screen", list(pages.keys()))
        st.sidebar.write("Training:", self.training_status())
        pages[page](st, px)

    def _render_opportunity_feed(self, st: Any, px: Any) -> None:
        st.header("Opportunity feed")
        st.caption(
            "Одна строка = один рынок одного матча. "
            "Pick — исход с максимальной консенсусной вероятностью (devig, >=2 книг) для этого рынка. "
            "Edge = best_odds * consensus_prob - 1. "
            "Это исследовательский индикатор, не рекомендация ставки; реальные результаты зависят от доступа к цене, лимитов и калибровки модели."
        )
        rows = self.opportunity_feed()
        if not rows:
            st.info("No snapshots yet. Collector will populate this feed.")
            return
        import pandas as pd

        df = pd.DataFrame(rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Рынки", len(df))
        c2.metric("Value (edge>=5%)", int((df.get("signal") == "value").sum()) if "signal" in df else 0)
        c3.metric("Small edge", int((df.get("signal") == "small_edge").sum()) if "signal" in df else 0)
        c4.metric("Low coverage", int((df.get("signal") == "low_coverage").sum()) if "signal" in df else 0)

        f1, f2, f3, f4, f5 = st.columns([1.2, 1, 1, 1, 1])
        with f1:
            sports = ["all"] + sorted(df["sport"].dropna().astype(str).unique().tolist()) if "sport" in df else ["all"]
            sport = st.selectbox("Спорт", sports)
        with f2:
            min_edge = st.number_input("Мин. edge %", value=0.0, step=0.5)
        with f3:
            min_cov = st.number_input("Мин. книг", value=2, min_value=0, step=1)
        with f4:
            min_odds = st.number_input("Мин. odds", value=1.01, min_value=1.01, step=0.05)
        with f5:
            max_odds = st.number_input("Макс. odds", value=50.0, min_value=1.01, step=0.5)

        show_all = st.checkbox("Показать все рынки (включая no_edge/low coverage)", value=False)

        view = df.copy()
        if sport != "all" and "sport" in view.columns:
            view = view[view["sport"].astype(str) == sport]
        if "edge_%" in view.columns:
            view = view[view["edge_%"].fillna(-999) >= float(min_edge)]
        if "coverage_books" in view.columns:
            view = view[view["coverage_books"].fillna(0) >= int(min_cov)]
        if "pick_best_odds" in view.columns:
            view = view[view["pick_best_odds"].fillna(0).between(float(min_odds), float(max_odds))]
        if not show_all and "signal" in view.columns:
            view = view[view["signal"].isin(["value", "small_edge", "low_coverage"])]
        if view.empty and not df.empty:
            st.info("Нет рынков после фильтров. Ослабьте пороги или включите 'Показать все рынки'.")
            view = df.head(60)

        show_cols = [c for c in [
            "event", "league", "start_at", "market", "line", "pick_outcome", "pick_prob_%", "pick_best_odds",
            "pick_bookmaker", "edge_%", "signal", "coverage_books", "probabilities", "best_odds_by_outcome"
        ] if c in view.columns]
        st.dataframe(view[show_cols] if show_cols else view, width="stretch")

        if not view.empty and {"edge_%", "event"}.issubset(view.columns):
            chart_df = view.dropna(subset=["edge_%"]).copy()
            if not chart_df.empty:
                chart_df["label"] = chart_df["event"].astype(str) + " / " + chart_df["market"].astype(str) + " " + chart_df.get("line", pd.Series([None]*len(chart_df))).fillna("").astype(str) + " / " + chart_df["pick_outcome"].astype(str)
                fig = px.bar(
                    chart_df.head(25),
                    x="label",
                    y="edge_%",
                    color="signal",
                    hover_data=["pick_prob_%", "pick_best_odds", "pick_bookmaker", "coverage_books", "probabilities"],
                )
                fig.update_layout(xaxis_tickangle=-35, height=450, yaxis_title="Edge %")
                st.plotly_chart(fig, width="stretch")

    def _render_market_depth(self, st: Any, px: Any) -> None:
        st.header("Market depth")
        rows = self.market_depth()
        if not rows:
            st.info("No market depth available yet.")
            return
        import pandas as pd

        df = pd.DataFrame(rows)
        if "event" in df.columns:
            options = sorted(df["event"].dropna().unique().tolist())
            selected = st.selectbox("Event", options)
            df = df[df["event"] == selected]
        if "market" in df.columns:
            mopts = ["all"] + sorted(df["market"].dropna().astype(str).unique().tolist())
            msel = st.selectbox("Market", mopts)
            if msel != "all":
                df = df[df["market"].astype(str) == msel]
        if "line" in df.columns:
            lopts = ["all"] + [str(x) for x in sorted(df["line"].dropna().unique().tolist())]
            lsel = st.selectbox("Line", lopts)
            if lsel != "all":
                try:
                    df = df[df["line"].astype(float) == float(lsel)]
                except Exception:
                    df = df[df["line"].astype(str) == lsel]
        st.dataframe(df, width="stretch")
        if not df.empty and {"outcome", "odds_decimal", "bookmaker"}.issubset(df.columns):
            fig = px.bar(df, x="outcome", y="odds_decimal", color="bookmaker", barmode="group")
            st.plotly_chart(fig, width="stretch")

    def _render_model_diagnostics(self, st: Any, px: Any) -> None:
        st.header("Model diagnostics")
        diag = self.model_diagnostics()
        st.json(diag.get("training", {}))

        preds = diag.get("predictions", [])
        st.subheader("Активные прогнозы модели")
        if preds:
            import pandas as pd
            from datetime import datetime, timezone

            pdf = pd.DataFrame(preds)
            if {"event_uid","market","model_prob_%"}.issubset(pdf.columns):
                pdf["event_start_at_dt"] = pd.to_datetime(pdf.get("event_start_at"), utc=True, errors="coerce")
                now = pd.Timestamp(datetime.now(timezone.utc))
                pdf["minutes_to_start"] = ((pdf["event_start_at_dt"] - now).dt.total_seconds() / 60).round(0)

                f1, f2, f3 = st.columns(3)
                sports = ["all"] + sorted([s for s in pdf.get("sport", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if s])
                sport = f1.selectbox("Sport", sports, index=0, key="md_sport")
                horizon_h = f2.slider("Горизонт, часов", 1, 168, 72, key="md_horizon")
                min_prob = f3.slider("Мин. вероятность выбора, %", 50.0, 95.0, 55.0, 0.5, key="md_min_prob")
                tennis_first = st.checkbox("Показывать теннис и ближайшие матчи первыми", value=True, key="md_tennis_first")

                view = pdf.copy()
                if sport != "all" and "sport" in view.columns:
                    view = view[view["sport"].astype(str) == sport]
                view = view[view["minutes_to_start"].isna() | (view["minutes_to_start"] <= horizon_h * 60)]

                ranked = view.sort_values(["event_uid","market","model_prob_%","outcome"], ascending=[True,True,False,True]).copy()
                top = ranked.groupby(["event_uid","market"], as_index=False).head(1).copy()
                ties = view.groupby(["event_uid","market"])["model_prob_%"].nunique().reset_index(name="prob_levels")
                top = top.merge(ties, on=["event_uid","market"], how="left")
                top["pick"] = top["outcome"].where(top["prob_levels"].fillna(1) > 1, "no_clear_pick")
                top = top[pd.to_numeric(top["model_prob_%"], errors="coerce").fillna(0) >= float(min_prob)]

                def _status(m: Any) -> str:
                    if pd.isna(m): return "unknown"
                    m = float(m)
                    if m <= 0: return "live/recent"
                    if m <= 60: return "soon_1h"
                    if m <= 360: return "soon_6h"
                    if m <= 1440: return "today_24h"
                    return "later"
                top["status"] = top["minutes_to_start"].map(_status)
                if tennis_first:
                    top["_pri"] = top.apply(lambda r: (0 if str(r.get("sport")) == "tennis" else 1, float(r["minutes_to_start"]) if pd.notna(r["minutes_to_start"]) else 10**9, -float(r.get("model_prob_%") or 0)), axis=1)
                    top = top.sort_values(["_pri"])
                else:
                    top = top.sort_values(["minutes_to_start","model_prob_%"], ascending=[True,False], na_position="last")

                st.caption("Ближайшие матчи и теннис приоритетны. Для тенниса расписание часто появляется поздно — прогнозы обновляются на каждом шаге trainer.")
                cols = [c for c in ["status","event","sport","league","start_at","minutes_to_start","market","pick","model_prob_%","odds_decimal","observed_at"] if c in top.columns]
                st.dataframe(top[cols].head(120), width="stretch")
                with st.expander("Все исходы (p1/p2 и др.)"):
                    raw_cols = [c for c in ["event","sport","league","start_at","minutes_to_start","market","outcome","model_prob_%","odds_decimal","observed_at"] if c in view.columns]
                    st.dataframe(view.sort_values(["minutes_to_start","model_prob_%"], ascending=[True,False], na_position="last")[raw_cols].head(400), width="stretch")
            else:
                st.dataframe(pdf, width="stretch")
        else:
            st.info("Прогнозы модели пока пусты. Collector собрал снапшоты, но для обучения нужны результаты матчей; после шага trainer появятся fallback/baseline прогнозы.")

        runs = diag.get("runs", [])
        st.subheader("Model runs")
        if not runs:
            st.info("No model runs yet.")
            return
        import pandas as pd

        flat = []
        for r in runs:
            m = r.get("metrics") or {}
            wf = m.get("walk_forward") or {}
            flat.append(
                {
                    "run_id": r["run_id"],
                    "created_at": r["created_at"],
                    "samples": m.get("samples"),
                    "new_samples": m.get("new_samples"),
                    "wf_accuracy": wf.get("accuracy"),
                    "wf_predictions": wf.get("predictions"),
                    "brier": m.get("brier"),
                    "log_loss": m.get("log_loss"),
                    "roi": m.get("roi"),
                    "hit_rate": m.get("hit_rate"),
                    "max_drawdown": m.get("max_drawdown"),
                    "clv": m.get("clv"),
                    "bad_actual": m.get("bad_actual"),
                    "edge_allowed": m.get("edge_report_allowed"),
                }
            )
        df = pd.DataFrame(flat)
        st.dataframe(df, width="stretch")

        latest = runs[0]
        m = latest.get("metrics") or {}
        st.subheader("Последний run: честные метрики")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Samples", m.get("samples", 0))
        c2.metric("Brier ↓", f"{m.get('brier'):.4f}" if m.get("brier") is not None else "n/a")
        c3.metric("Logloss ↓", f"{m.get('log_loss'):.4f}" if m.get("log_loss") is not None else "n/a")
        c4.metric("Hit rate", f"{m.get('hit_rate')*100:.2f}%" if m.get("hit_rate") is not None else "n/a")
        c5.metric("ROI (sim)", f"{m.get('roi')*100:.2f}%" if m.get("roi") is not None else "n/a")
        c6.metric("CLV", f"{m.get('clv')*100:.2f}%" if m.get("clv") is not None else "n/a")
        st.caption(
            "ROI/CLV — симуляция на исторических данных. CLV исключается там, где нет closing snapshot; "
            "остальные метрики считаются отдельно. При малой выборке edge нестабилен."
        )

        if m.get("edge_warning"):
            st.warning(str(m.get("edge_warning")))
        if m.get("bad_actual"):
            st.warning(f"bad_actual={m.get('bad_actual')} — часть результатов не сопоставилась однозначно, их нельзя считать в метриках.")

        calib = m.get("calibration_bins") or []
        if calib:
            cdf = pd.DataFrame(calib)
            if {"avg_pred", "avg_actual", "count"}.issubset(cdf.columns):
                st.subheader("Калибровка (предсказанная vs фактическая вероятность)")
                fig = px.line(cdf, x="avg_pred", y="avg_actual", markers=True, hover_data=["count", "bin_lo", "bin_hi"])
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
                fig.update_layout(xaxis_title="Avg predicted", yaxis_title="Avg actual", height=380)
                st.plotly_chart(fig, width="stretch")

        by_market = m.get("profit_by_market") or {}
        if by_market:
            mdf = pd.DataFrame([{"market": k, "profit_units": v} for k, v in by_market.items()]).sort_values("profit_units", ascending=False)
            st.subheader("Profit by market (sim, units)")
            st.plotly_chart(px.bar(mdf, x="market", y="profit_units"), width="stretch")

        if "wf_accuracy" in df.columns and df["wf_accuracy"].notna().any():
            fig = px.line(df.sort_values("created_at"), x="created_at", y="wf_accuracy", markers=True)
            fig.update_layout(yaxis_title="Walk-forward accuracy")
            st.plotly_chart(fig, width="stretch")

    def _render_backtest_lab(self, st: Any, px: Any) -> None:
        st.header("Backtest lab")
        st.caption("Paper-only placeholder/demo. No betting execution.")
        data = self.backtest_lab()
        st.write({"n_events": data["n_events"], "n_results": data["n_results"], "paper_only": data.get("paper_only", True)})
        report = data["report"]
        if report is None:
            st.info("No backtest report available yet (need events and results).")
            return
        preds = report.predictions
        st.write({"predictions": len(preds), "clv_sample_count": report.clv_sample_count})
        if preds:
            import pandas as pd

            labels = {}
            try:
                repo = self.repository
                if repo is not None:
                    labels = self._event_labels(repo.events_for_resolution())
            except Exception:
                labels = {}
            df = pd.DataFrame(
                [
                    {
                        "event": labels.get(p.event_uid, p.event_uid),
                        "market": p.market,
                        "outcome": p.outcome,
                        "odds_decimal": p.odds_decimal,
                        "model_prob": p.model_prob,
                        "actual": p.actual,
                        "clv": p.clv,
                        "event_uid": p.event_uid,
                    }
                    for p in preds
                ]
            )
            st.dataframe(df, width="stretch")
            if df["clv"].notna().any():
                fig = px.histogram(df[df["clv"].notna()], x="clv", nbins=20)
                st.plotly_chart(fig, width="stretch")

    def _render_data_health(self, st: Any, px: Any) -> None:
        st.header("Data health")
        repo = self.repository
        rows = self.adapter_status()
        import pandas as pd
        from pathlib import Path

        c1, c2, c3 = st.columns(3)
        n_events = len(repo.events_for_resolution()) if repo is not None else 0
        n_results = len(repo.results()) if repo is not None else 0
        c1.metric("Canonical events", n_events)
        c2.metric("Stored results", n_results)
        c3.metric("Adapters", len(rows))

        st.subheader("Adapter health")
        if not rows:
            st.warning("No adapter health rows recorded yet.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")
            if "error_count" in df.columns:
                fig = px.bar(df, x="bookmaker", y="error_count", color="status")
                st.plotly_chart(fig, width="stretch")

        st.subheader("Import results (tennis-data.co.uk CSV)")
        st.caption(
            "Загрузите CSV формата tennis-data.co.uk (Date,Winner,Loser,Score,Tournament). "
            "Результаты сопоставляются с canonical events по нормализованным именам и дате. "
            "Sackmann полезен для признаков, но без odds не даёт честный CLV/backtest."
        )
        create_missing = st.checkbox(
            "Создавать missing canonical events для исторических результатов (нужно для обучения по tennis-data)",
            value=True,
            key="create_missing_events_ingest",
        )
        up = st.file_uploader("CSV/XLSX file", type=["csv", "xlsx"], accept_multiple_files=False, key="results_csv_upload")
        if up is not None and st.button("Импортировать файл", type="primary", key="import_results_btn"):
            if repo is None:
                st.error("Repository unavailable")
            else:
                from tempfile import NamedTemporaryFile
                from betdash.results.ingest import ingest_tennis_results_report

                suffix = ".xlsx" if str(up.name).lower().endswith(".xlsx") else ".csv"
                with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(up.getvalue())
                    tmp_path = tmp.name
                try:
                    report = ingest_tennis_results_report(tmp_path, repo, create_missing_events=bool(create_missing))
                    st.session_state["last_ingest_report"] = report.as_dict()
                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                st.success("Импорт завершён. См. отчёт ниже.")
                try:
                    runtime = st.session_state.get("betdash_runtime")
                    trainer = getattr(runtime, "trainer", None) if runtime is not None else None
                    if trainer is not None and hasattr(trainer, "step"):
                        trainer.step()
                        st.info("Trainer step executed after import.")
                except Exception as e:
                    st.warning(f"Trainer step after import failed: {e}")

        rep = st.session_state.get("last_ingest_report")
        if rep:
            st.write("Последний отчёт импорта", rep)

    def _render_risk_panel(self, st: Any, px: Any) -> None:
        st.header("Risk panel (paper exposure only)")
        panel = self.risk_panel()
        st.warning("Paper/simulation only. No betting execution.")
        st.json(panel)


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: StreamlitRuntime | None = None


def _build_runtime() -> StreamlitRuntime:
    runtime = StreamlitRuntime()
    runtime.start_workers()
    return runtime


def get_runtime() -> StreamlitRuntime:
    """Return process-local singleton runtime, starting workers once."""

    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = _build_runtime()
        return _RUNTIME


def _reset_runtime_for_tests() -> None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = None


def _get_or_create_runtime() -> StreamlitRuntime:
    st = _safe_import_streamlit()

    @st.cache_resource(show_spinner=False)
    def _factory() -> StreamlitRuntime:
        return get_runtime()

    return _factory()


def run_dashboard() -> None:
    """Entry point used by Streamlit page and smoke tests.

    Smoke tests patch ``StreamlitRuntime`` in this module and expect a single
    ``start_workers()`` call. Repeated calls reuse the process-local singleton
    to avoid duplicate workers.
    """

    runtime = get_runtime()
    # Render only when running inside Streamlit; allows smoke tests to patch
    # StreamlitRuntime and call run_dashboard() without a Streamlit context.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            runtime.render()
    except Exception:
        # Not in a Streamlit script context; start-only mode.
        return


def run_dashboard_cached() -> None:
    """Streamlit-friendly entrypoint that keeps one runtime per process."""

    runtime = _get_or_create_runtime()
    try:
        runtime.render()
    except Exception:
        # Render failures should not crash background workers.
        raise
