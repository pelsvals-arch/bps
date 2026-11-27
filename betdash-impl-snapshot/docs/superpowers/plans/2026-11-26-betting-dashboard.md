# Bet Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local research dashboard that continuously collects public pre-match lines from six bookmakers, trains calibrated probability models while open, and shows honest walk-forward backtests of model-vs-market disagreements.

**Architecture:** Isolated bookmaker adapters emit a common `RawOdd`; normalization maps them into canonical events; an immutable SQLite `odds_snapshots` store feeds de-vig/consensus, baseline ratings, a backtest engine and an incremental trainer. One Streamlit process owns APScheduler threads for collection and training; a tray app controls that process.

**Tech Stack:** Python 3.14, pytest, httpx, lxml, websockets, APScheduler, SQLAlchemy + Alembic, SQLite WAL, DuckDB, pandas/polars, scikit-learn, CatBoost (after baseline), Streamlit + Plotly, pystray + Pillow.

**Spec:** `docs/superpowers/specs/2026-11-26-betting-dashboard-design.md`

## Global Constraints

- No paid APIs or subscriptions of any kind.
- Bookmakers: Winline, Pari, Fonbet, Marathon, BetBoom, Bettery only. Liga Stavok excluded.
- Sports in MVP: tennis, basketball, Dota 2, Counter-Strike 2. American football is post-MVP.
- No account automation, captcha solving, WAF bypass, residential proxies, or placing bets. Only public data.
- `tennis-sackmann-archive` and `tennis-data.co.uk` are non-commercial with attribution; no redistribution.
- Local only; no public server, no daemon/autostart beyond the tray app.
- Python 3.14; SQLite in WAL mode as source of truth; DuckDB for analytics; Streamlit UI; APScheduler inside the process.
- No neural networks, live models, props/combo markets, auto-staking, or universal cross-sport model in MVP.
- Validation is walk-forward chronological only; no random train/test split; no future information in features.
- Collector and trainer start when the dashboard opens and keep running while it is open; checkpoints survive restart.
- Tray icon starts/stops the dashboard and shows adapter/training status; it is not a second analytics UI.

## Review Focus

- Bookmaker response changes shape or returns an empty line → adapter records `AdapterHealth(status="degraded")`, keeps the last good snapshots, never raises into the UI. Test in Task 7 with a fake adapter returning `[]`.
- Duplicate/reordered snapshots for the same event, market and timestamp → saved once and counted once in CLV. Test in Tasks 1 and 10.
- Missing closing snapshot or fewer than two books with a price → signal excluded from CLV/benchmark, still visible in other metrics with `coverage_count`. Test in Task 9.
- Ambiguous entity mapping (same player/team/date in two events) → `mapping_confidence < threshold`, event excluded from backtest. Test in Task 2 and Task 8.
- Training loop restart or Streamlit rerun mid-checkpoint → resumes from last checkpoint without duplicate state or frozen UI. Test in Task 11.

---

### Task 1: Project scaffold, config, schema and snapshot repository

**Files:**
- Create: `pyproject.toml`, `.env.example`, `src/betdash/__init__.py`, `src/betdash/config.py`
- Create: `src/betdash/db/schema.sql`, `src/betdash/db/repository.py`
- Create: `tests/db/test_repository.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` from `pydantic-settings`; `Snapshot` and `AdapterHealth(bookmaker, status, last_success, error_count, adapter_version)` dataclasses; `Repository.save_snapshots(rows: list[Snapshot]) -> int`; `Repository.last_snapshots(event_uid: str) -> list[Snapshot]`; `Repository.adapter_health() -> list[AdapterHealth]`; `init_db(path: str) -> None`.

- [ ] **Step 1: Write failing tests for initialization, immutability and dedupe**

```python
def test_init_db_creates_tables(tmp_path):
    init_db(str(tmp_path / "bet.sqlite"))
    assert (tmp_path / "bet.sqlite").exists()

def test_save_snapshots_dedupes_same_key(tmp_path):
    repo = Repository(str(tmp_path / "bet.sqlite"))
    s = Snapshot(event_uid="e1", bookmaker="pari", market="match_winner",
                 outcome="p1", odds_decimal=2.0, observed_at=NOW,
                 source_event_id="p1", adapter_version="1", is_live=False)
    assert repo.save_snapshots([s, s]) == 1
    assert len(repo.last_snapshots("e1")) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/db/test_repository.py -v`
Expected: FAIL with missing module/symbol.

- [ ] **Step 3: Implement `init_db` and `Repository`**

Create `pyproject.toml` with pytest, pydantic-settings, SQLAlchemy, Alembic, httpx, lxml, websockets, APScheduler, pandas, scikit-learn, duckdb, streamlit, plotly, pystray, pillow. `Settings` fields: `db_path`, `poll_seconds=90`, `bookmaker_hosts`, `betboom_line_path`, `checkpoint_dir`, `enable_catboost=False`. `schema.sql` creates `events`, `participants`, `event_aliases`, `market_types`, `odds_snapshots` with UNIQUE(event_uid, bookmaker, market, outcome, observed_at, adapter_version), `results`, `feature_values`, `predictions`, `model_runs`, `adapter_health`; enable `PRAGMA journal_mode=WAL`. `save_snapshots` uses `INSERT ... ON CONFLICT DO NOTHING` and never updates existing rows. Add `README.md` naming the data sources and their non-commercial/attribution licenses.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/db/test_repository.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git init && git add pyproject.toml .env.example src tests && git commit -m "feat: scaffold project with immutable odds snapshot store"
```

---

### Task 2: Data contract and entity normalization

**Files:**
- Create: `src/betdash/normalize/models.py`, `src/betdash/normalize/registry.py`
- Create: `tests/normalize/test_registry.py`, `tests/normalize/fixtures/aliases.json`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `RawOdd`, `CanonicalEvent`, `EntityRegistry`, `normalize_event(raw: RawOdd, registry: EntityRegistry) -> CanonicalEvent`, `canon_market(raw_market: str, bookmaker: str) -> str`.

- [ ] **Step 1: Write failing tests for canonical ids and ambiguity**

```python
def test_normalize_event_resolves_aliases():
    reg = EntityRegistry.from_json("tests/normalize/fixtures/aliases.json")
    raw = RawOdd(bookmaker="pari", source_event_id="123", sport="tennis",
                 league="ATP", participants=("Djokovic N.", "Alcaraz C."),
                 market="П1", outcome="p1", odds_decimal=1.8, observed_at=NOW,
                 is_live=False)
    ev = normalize_event(raw, reg)
    assert ev.participant_ids == ("atp:djokovic", "atp:alcaraz")
    assert ev.mapping_confidence >= 0.8

def test_ambiguous_mapping_is_low_confidence():
    reg = EntityRegistry.from_json("tests/normalize/fixtures/aliases.json")
    raw = RawOdd(..., participants=("Ivanov", "Petrov"), ...)
    assert normalize_event(raw, reg).mapping_confidence < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/normalize -v`
Expected: FAIL with missing symbols.

- [ ] **Step 3: Implement models and registry**

`RawOdd` fields exactly as in the test. `CanonicalEvent(event_uid, sport, league, participant_ids, start_at, mapping_confidence)`. `EntityRegistry.from_json` loads alias→canonical map; `normalize_event` lowercases and strips punctuation, resolves aliases, uses deterministic `event_uid = sha1(sport|league|sorted(participant_ids)|date)`. `canon_market` maps per-book labels to `match_winner`, `set_handicap`, `total_games`, `map_winner`, `total_maps`, `spread`, `total_points`; unknown labels return `unknown_*` with confidence 0.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/normalize -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/normalize tests/normalize && git commit -m "feat: add raw event contract and entity normalization"
```

---

### Task 3: JSON-family adapters (Pari, Fonbet, Bettery)

**Files:**
- Create: `src/betdash/adapters/base.py`, `src/betdash/adapters/json_family.py`
- Create: `tests/adapters/test_json_family.py`, `tests/adapters/fixtures/json_family_sample.json`

**Interfaces:**
- Consumes: `RawOdd`, `Settings`.
- Produces: `Adapter` protocol with `fetch_snapshot() -> list[RawOdd]` and `health() -> AdapterHealth`; `JsonFamilyAdapter(bookmaker: str, settings: Settings)` with `parse_fixture(payload: dict) -> list[RawOdd]`.

- [ ] **Step 1: Write failing parser test**

```python
def test_parses_factor_921_923_as_p1_p2():
    raw = JsonFamilyAdapter("pari", Settings()).parse_fixture(FIXTURE)
    assert [(r.outcome, r.odds_decimal) for r in raw[:2]] == [("p1", 2.1), ("p2", 1.75)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_json_family.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement discovery and parsing**

`fetch_snapshot` calls `/urls.json?hash=` from the configured host, picks the first responding `clientsapi*/line*` host, then calls `/events/list?lang=ru&version=0&scopeMarket=<per-book>` with httpx and 10s timeout. Parse `sports → events → customFactors`, where factor id 921→`p1`, 923→`p2`; emit `RawOdd` with `observed_at=utcnow()`. Three bookmakers differ only in `scopeMarket` (Pari 2300, Fonbet 1600, Bettery 501) and host family. No cookies, login or captcha.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_json_family.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/adapters tests/adapters && git commit -m "feat: add Pari/Fonbet/Bettery line adapters"
```

---

### Task 4: Marathon SSR adapter

**Files:**
- Create: `src/betdash/adapters/marathon.py`
- Create: `tests/adapters/test_marathon.py`, `tests/adapters/fixtures/marathon_ssr.html`

**Interfaces:**
- Consumes: `RawOdd`, `Settings`, `Adapter` protocol.
- Produces: `MarathonAdapter(settings: Settings)` with `parse_fixture(html: str) -> list[RawOdd]`.

- [ ] **Step 1: Write failing test**

```python
def test_extracts_odds_from_data_attributes():
    rows = MarathonAdapter(Settings()).parse_fixture(FIXTURE_HTML)
    assert any(r.outcome == "p1" and r.odds_decimal == 2.875 for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_marathon.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Fetch `https://new.marathonbet.ru/su/` with httpx; parse SSR HTML via lxml, reading `data-event-treeId`, `data-selection-price` and embedded `initData/reactData`. Emit `RawOdd` for tennis/basketball/Dota 2/CS2 only. Do not accept cookie-consent interaction in code: the page is read-only; if content is blocked, mark health degraded.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_marathon.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/adapters/marathon.py tests/adapters/test_marathon.py tests/adapters/fixtures/marathon_ssr.html && git commit -m "feat: add Marathon SSR line adapter"
```

---

### Task 5: BetBoom public siteapi adapter

**Files:**
- Create: `src/betdash/adapters/betboom.py`
- Create: `tests/adapters/test_betboom.py`, `tests/adapters/fixtures/betboom_line.json`

**Interfaces:**
- Consumes: `RawOdd`, `Settings`, `Adapter`.
- Produces: `BetBoomAdapter(settings: Settings)` with `parse_fixture(payload: dict) -> list[RawOdd]`.

- [ ] **Step 1: Write failing test**

```python
def test_maps_cyber_and_tennis_counts():
    rows = BetBoomAdapter(Settings()).parse_fixture(FIXTURE)
    sports = {r.sport for r in rows}
    assert {"tennis", "basketball", "dota2", "cs2"} <= sports
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_betboom.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Read `https://sportbook.sporthub.bet/widgets/sportbook/v1/modern/runtime-env.js` for `API_URL` and use only the public `siteapi` path; store the concrete route in `Settings.betboom_line_path`, captured once from a public browser network log during implementation. No Qrator interaction, no account session. Parse the JSON fixture into `RawOdd`. If the site returns 401, mark degraded and return `[]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_betboom.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/adapters/betboom.py tests/adapters/test_betboom.py tests/adapters/fixtures/betboom_line.json && git commit -m "feat: add BetBoom public line adapter"
```

---

### Task 6: Winline binary WebSocket adapter

**Files:**
- Create: `src/betdash/adapters/winline.py`
- Create: `tests/adapters/test_winline.py`, `tests/adapters/fixtures/winline_frame.bin`

**Interfaces:**
- Consumes: `RawOdd`, `Settings`, `Adapter`.
- Produces: `WinlineAdapter(settings: Settings)` with `parse_frame(data: bytes) -> list[RawOdd]`.

- [ ] **Step 1: Write failing test**

```python
def test_decodes_market_0x0556_odds():
    rows = WinlineAdapter(Settings()).parse_frame(FIXTURE_BYTES)
    assert any(r.market == "match_winner" and r.odds_decimal == 2.5 for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_winline.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Connect to `wss://wss.winline.ru/data_ng?client=newsite&nb=true` without token; send init strings `lang`, `ru`, `data`, `WINLINE`, `getdate`. Decompress gzip frames; decode market id `0x0556` as `match_winner` and odds as `u16 / 100`. If frame layout differs from the fixture, keep the parser isolated in `parse_frame` and update the fixture rather than changing other adapters.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_winline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/adapters/winline.py tests/adapters/test_winline.py tests/adapters/fixtures/winline_frame.bin && git commit -m "feat: add Winline binary WebSocket adapter"
```

---

### Task 7: Collector, retry policy and adapter health

**Files:**
- Create: `src/betdash/ingest/collector.py`
- Create: `tests/ingest/test_collector.py`

**Interfaces:**
- Consumes: all `Adapter` implementations, `Repository`, `Settings`, `normalize_event`.
- Produces: `Collector(settings, repository, adapters, registry)`, `collector.poll_once() -> int`, `collector.start() -> None`, `collector.stop() -> None`.

- [ ] **Step 1: Write failing test for empty response**

```python
def test_empty_adapter_marks_degraded_and_keeps_last_snapshot():
    repo = Repository(str(tmp_path / "b.sqlite"))
    seed_snapshot(repo)
    collector = Collector(Settings(), repo, [FakeEmptyAdapter()], registry)
    assert collector.poll_once() == 0
    assert collector.health()[0].status == "degraded"
    assert len(repo.last_snapshots("e1")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_collector.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement collector**

`poll_once` iterates adapters with try/except, calls `fetch_snapshot`, normalizes with `normalize_event`, skips `mapping_confidence < 0.5`, saves through `Repository.save_snapshots`, updates `adapter_health` with `ok/degraded`, last_success, error count and adapter version. `start` registers an APScheduler interval job using `Settings.poll_seconds` (default 90). `stop` shuts the scheduler down cleanly. No exceptions escape to the caller.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_collector.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/ingest tests/ingest && git commit -m "feat: add resilient snapshot collector"
```

---

### Task 8: Result ingestion and cross-source entity resolution

**Files:**
- Create: `src/betdash/results/ingest.py`, `src/betdash/results/resolve.py`
- Create: `tests/results/test_resolve.py`, `tests/results/fixtures/tennis_results.csv`

**Interfaces:**
- Consumes: `Repository`, `CanonicalEvent`.
- Produces: `ResultRow`, `ingest_tennis_results(path: str, repo: Repository) -> int`, `resolve_result(result: ResultRow, events: list[CanonicalEvent]) -> str | None`.

- [ ] **Step 1: Write failing ambiguity test**

```python
def test_ambiguous_result_returns_none():
    events = [make_event("e1", "Djokovic", "Alcaraz", "2026-03-01"),
              make_event("e2", "Djokovic", "Alcaraz", "2026-03-01")]
    assert resolve_result(make_result("Djokovic", "Alcaraz", "2026-03-01"), events) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/results -v`
Expected: FAIL.

- [ ] **Step 3: Implement ingestion and resolver**

Parse `tennis-data.co.uk` CSV columns into `ResultRow(participant_a, participant_b, date, score, league)`; use the same alias registry logic as Task 2. `resolve_result` requires exactly one event match on normalized names + date; otherwise returns `None`. Store accepted rows in `results` with `event_uid`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/results -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/results tests/results && git commit -m "feat: add result ingestion with entity resolution"
```

---

### Task 9: De-vig, consensus and benchmark semantics

**Files:**
- Create: `src/betdash/markets/de_vig.py`, `src/betdash/markets/consensus.py`
- Create: `tests/markets/test_consensus.py`

**Interfaces:**
- Consumes: `Snapshot`, `Repository`.
- Produces: `implied_probs(odds: list[float]) -> list[float]`, `consensus_prob(snapshots: list[Snapshot], outcome: str) -> tuple[float, int]`, `best_price(snapshots: list[Snapshot], outcome: str) -> float | None`.

- [ ] **Step 1: Write failing tests for margin and missing books**

```python
def test_de_vig_removes_margin():
    p = implied_probs([1.90, 2.10])
    assert abs(sum(p) - 1.0) < 1e-9

def test_single_book_consensus_returns_zero_coverage():
    snaps = [make_snapshot("pari", "p1", 2.0)]
    prob, coverage = consensus_prob(snaps, "p1")
    assert coverage == 0 and prob == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/markets -v`
Expected: FAIL.

- [ ] **Step 3: Implement market math**

`implied_probs` normalizes inverse odds to sum 1. `consensus_prob` de-vigs each book separately, then averages only books with at least two outcomes present; returns `(0.0, coverage)` when coverage < 2. `best_price` returns the maximum available decimal odds. CLV uses consensus at `market_close_at`; no closing snapshot means the signal is excluded from CLV but kept in other metrics.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/markets -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/markets tests/markets && git commit -m "feat: add de-vig consensus and benchmark rules"
```

---

### Task 10: Walk-forward backtest engine and metrics

**Files:**
- Create: `src/betdash/backtest/engine.py`, `src/betdash/backtest/metrics.py`
- Create: `tests/backtest/test_engine.py`, `tests/backtest/test_metrics.py`

**Interfaces:**
- Consumes: `Repository`, `consensus_prob`, `ResultRow`, `Snapshot`.
- Produces: `Signal(event_uid, market, outcome, odds_decimal, model_prob, observed_at)`, `BacktestReport(predictions, clv_sample_count, signals)`, `BacktestEngine(repo, model)` with `run(start, end, sport) -> BacktestReport` and `run_with_duplicates(start, end, sport, duplicated: bool) -> BacktestReport`; `metrics.all(report) -> dict[str, float]` with `brier`, `log_loss`, `clv`, `roi`, `hit_rate`, `max_drawdown`.

- [ ] **Step 1: Write failing leakage and duplicate tests**

```python
def test_signal_uses_only_snapshots_before_decision_time():
    report = engine.run(START, END, "tennis")
    assert all(p.observed_at < p.event_start_at for p in report.predictions)

def test_duplicate_snapshot_counted_once_in_clv():
    report = engine.run_with_duplicates(START, END, "tennis", duplicated=True)
    assert report.clv_sample_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest -v`
Expected: FAIL.

- [ ] **Step 3: Implement engine and metrics**

Walk-forward splits by `observed_at` with configurable train/test windows and an embargo gap; no random split. For every signal require `coverage_count >= 2` and an available price at `observed_at`. Compute CLV against closing consensus; if absent, mark `clv=None`. Metrics include Brier, log-loss, calibration bins, simulated flat-stake ROI, hit rate, profit by market/league and max drawdown; apply a minimum-sample guard before reporting edge.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backtest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/backtest tests/backtest && git commit -m "feat: add walk-forward backtest and honest metrics"
```

---

### Task 11: Baseline ratings, features and continuous trainer

**Files:**
- Create: `src/betdash/models/features.py`, `src/betdash/models/baseline.py`, `src/betdash/models/trainer.py`
- Create: `tests/models/test_trainer.py`, `tests/models/test_baseline.py`

**Interfaces:**
- Consumes: `Repository`, `BacktestEngine`, `Settings`.
- Produces: `build_features(event: CanonicalEvent, repo: Repository) -> dict[str, float]`, `EloRating` and `GlickoRating` with `probability(a, b) -> float`, `ModelRun(run_id, features_version, created_at)`, `Trainer(repo, settings)` with `last_run_id`, `trainer.step() -> ModelRun`, `trainer.start() -> None`, `trainer.stop() -> None`.

- [ ] **Step 1: Write failing checkpoint test**

```python
def test_trainer_resumes_after_restart(tmp_path):
    t1 = Trainer(repo, Settings(checkpoint_dir=str(tmp_path)))
    t1.step()
    t2 = Trainer(repo, Settings(checkpoint_dir=str(tmp_path)))
    assert t2.last_run_id == t1.last_run_id
    assert t2.step().features_version == t1.step().features_version
```

- [ ] **Step 2: Run tests to verify it fails**

Run: `pytest tests/models -v`
Expected: FAIL.

- [ ] **Step 3: Implement models and trainer**

Separate Elo/Glicko state per sport; logistic regression on form/rest/surface/patch/map/H2H features; CatBoost only behind `Settings.enable_catboost` (default false until baseline passes). `Trainer.step` reads new snapshots and results, recomputes features with `as_of` timestamps, trains walk-forward, writes `model_runs` and checkpoint JSON atomically. `start` runs steps on APScheduler; `stop` waits for the current step. No neural networks.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/models tests/models && git commit -m "feat: add baseline ratings and resumable incremental trainer"
```

---

### Task 12: Paper risk engine

**Files:**
- Create: `src/betdash/risk/engine.py`
- Create: `tests/risk/test_engine.py`

**Interfaces:**
- Consumes: `Settings`, `Snapshot`, `BacktestReport`.
- Produces: `RiskEngine(settings: Settings)`, `evaluate(signal: Signal, report: BacktestReport) -> RiskDecision`, with `DecisionStatus` values `allowed`, `skipped`, `kill_switch`.

- [ ] **Step 1: Write failing limits test**

```python
def test_daily_loss_limit_skips_signal():
    engine = RiskEngine(Settings(max_daily_loss=100.0))
    decision = engine.evaluate(make_signal(stake=20), make_report(loss=120))
    assert decision.status == "skipped"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/risk -v`
Expected: FAIL.

- [ ] **Step 3: Implement paper risk rules**

Check max stake per signal, max exposure per event/league/sport, max daily loss, correlation limit on highly related outcomes, and a kill switch on adapter health below threshold or anomalous odds. No execution layer; returns a decision with reason codes only.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/risk -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/risk tests/risk && git commit -m "feat: add paper risk limits and kill switch"
```

---

### Task 13: Streamlit dashboard and tray controller

**Files:**
- Create: `src/betdash/app/dashboard.py`, `src/betdash/app/tray.py`, `src/betdash/app/__main__.py`
- Create: `tests/app/test_dashboard_smoke.py`

**Interfaces:**
- Consumes: `Collector`, `Trainer`, `Repository`, `BacktestEngine`, `Settings`.
- Produces: `StreamlitRuntime` with `start_workers()`, `run_dashboard() -> None`, `TrayController.start() -> None`, `TrayController.stop() -> None`.

- [ ] **Step 1: Write failing smoke test**

```python
def test_dashboard_starts_background_workers_once():
    with patch("betdash.app.dashboard.StreamlitRuntime") as rt:
        run_dashboard()
        assert rt.return_value.start_workers.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/app -v`
Expected: FAIL.

- [ ] **Step 3: Implement UI and tray**

Use `st.cache_resource` singleton to create one `Collector` and one `Trainer`; `start()` on first run, no duplicate workers on rerun. Screens: Opportunity feed, Market depth, Model diagnostics, Backtest lab, Data health, Risk panel (paper exposure only). Plotly charts. `TrayController` uses pystray: Start/Stop dashboard, Open in browser, status of adapters and trainer. Default collector/trainer stop on process exit via `atexit`.

- [ ] **Step 4: Run test and start dashboard once**

Run: `pytest tests/app -v && python -m betdash.app --check`
Expected: PASS; `--check` prints worker/adapter status and exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/betdash/app tests/app && git commit -m "feat: add dashboard screens and tray controller"
```
