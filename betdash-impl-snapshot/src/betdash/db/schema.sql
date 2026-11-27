-- SQLite schema for the local betting-research dashboard.
-- odds_snapshots is append-only: rows are never updated or deleted.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_uid TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    league TEXT NOT NULL,
    participant_ids TEXT,
    start_at TEXT,
    market_close_at TEXT,
    mapping_confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS canonical_events (
    event_uid TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    league TEXT NOT NULL,
    participant_ids TEXT NOT NULL, -- JSON array of strings
    start_at TEXT NOT NULL,
    mapping_confidence REAL NOT NULL,
    market_close_at TEXT
);

CREATE TABLE IF NOT EXISTS participants (
    participant_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    UNIQUE (bookmaker, source_event_id)
);

CREATE TABLE IF NOT EXISTS market_types (
    market TEXT PRIMARY KEY,
    sport TEXT,
    description TEXT,
    outcomes TEXT
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    outcome TEXT NOT NULL,
    line REAL,
    odds_decimal REAL NOT NULL,
    observed_at TEXT NOT NULL,
    source_event_id TEXT,
    source_url TEXT,
    adapter_version TEXT NOT NULL,
    is_live INTEGER NOT NULL DEFAULT 0 CHECK (is_live IN (0, 1)),
    line_key REAL NOT NULL DEFAULT -1e308,
    UNIQUE (event_uid, bookmaker, market, outcome, line_key, observed_at, adapter_version)
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_event_observed
    ON odds_snapshots (event_uid, observed_at);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid TEXT NOT NULL,
    participant_a TEXT,
    participant_b TEXT,
    score TEXT,
    league TEXT,
    event_date TEXT,
    source TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (event_uid, source)
);

CREATE TABLE IF NOT EXISTS feature_values (
    event_uid TEXT NOT NULL,
    feature TEXT NOT NULL,
    value REAL NOT NULL,
    source TEXT NOT NULL,
    as_of TEXT NOT NULL,
    transform_version TEXT NOT NULL,
    PRIMARY KEY (event_uid, feature, as_of, transform_version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id TEXT,
    event_uid TEXT NOT NULL,
    market TEXT NOT NULL,
    outcome TEXT NOT NULL,
    line REAL,
    model_prob REAL NOT NULL,
    odds_decimal REAL,
    bookmaker TEXT,
    observed_at TEXT NOT NULL,
    event_start_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    features_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS adapter_health (
    bookmaker TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    last_success TEXT,
    error_count INTEGER NOT NULL DEFAULT 0,
    adapter_version TEXT NOT NULL
);
