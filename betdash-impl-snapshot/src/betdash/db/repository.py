"""SQLite schema initialization and the append-only snapshot repository."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from betdash.normalize.models import CanonicalEvent

if TYPE_CHECKING:
    from betdash.results.ingest import ResultRow

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_INSERT_SNAPSHOT = """
INSERT INTO odds_snapshots (
    event_uid,
    bookmaker,
    market,
    outcome,
    line,
    line_key,
    odds_decimal,
    observed_at,
    source_event_id,
    source_url,
    adapter_version,
    is_live
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (event_uid, bookmaker, market, outcome, line_key, observed_at, adapter_version)
DO NOTHING
"""

_LATEST_SNAPSHOTS = """
SELECT
    event_uid,
    bookmaker,
    market,
    outcome,
    line,
    odds_decimal,
    observed_at,
    source_event_id,
    source_url,
    adapter_version,
    is_live
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY bookmaker, market, outcome, line_key
            ORDER BY observed_at DESC, id DESC
        ) AS row_number
    FROM odds_snapshots
    WHERE event_uid = ?
)
WHERE row_number = 1
ORDER BY bookmaker, market, outcome, line
"""

_UPSERT_ADAPTER_HEALTH = """
INSERT INTO adapter_health (
    bookmaker,
    status,
    last_success,
    error_count,
    adapter_version
) VALUES (?, ?, ?, ?, ?)
ON CONFLICT (bookmaker) DO UPDATE SET
    status = excluded.status,
    last_success = excluded.last_success,
    error_count = excluded.error_count,
    adapter_version = excluded.adapter_version
"""

_UPSERT_CANONICAL_EVENT = """
INSERT INTO canonical_events (
    event_uid,
    sport,
    league,
    participant_ids,
    start_at,
    mapping_confidence,
    market_close_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (event_uid) DO UPDATE SET
    sport = excluded.sport,
    league = excluded.league,
    participant_ids = excluded.participant_ids,
    start_at = excluded.start_at,
    mapping_confidence = excluded.mapping_confidence,
    market_close_at = COALESCE(excluded.market_close_at, canonical_events.market_close_at)
"""

_INSERT_RESULT = """
INSERT INTO results (
    event_uid,
    participant_a,
    participant_b,
    score,
    league,
    event_date,
    source
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (event_uid, source) DO NOTHING
"""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One observed price. Stored rows are immutable."""

    event_uid: str
    bookmaker: str
    market: str
    outcome: str
    odds_decimal: float
    observed_at: datetime
    source_event_id: str
    adapter_version: str
    is_live: bool = False
    source_url: str | None = None
    line: float | None = None


@dataclass(frozen=True, slots=True)
class AdapterHealth:
    """Last known state of a bookmaker adapter."""

    bookmaker: str
    status: str
    last_success: datetime | None
    error_count: int
    adapter_version: str


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="microseconds")


def _from_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def init_db(path: str) -> None:
    """Create the database file and all tables, idempotently."""

    db_path = Path(path)
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(schema)
        connection.commit()
    finally:
        connection.close()


def _snapshot_from_row(row: sqlite3.Row) -> Snapshot:
    keys = set(row.keys())
    return Snapshot(
        event_uid=row["event_uid"],
        bookmaker=row["bookmaker"],
        market=row["market"],
        outcome=row["outcome"],
        odds_decimal=row["odds_decimal"],
        observed_at=_from_iso(row["observed_at"]),
        source_event_id=row["source_event_id"],
        adapter_version=row["adapter_version"],
        is_live=bool(row["is_live"]),
        source_url=row["source_url"],
        line=row["line"] if "line" in keys else None,
    )


class Repository:
    """Append-only access to the local SQLite database."""

    def __init__(self, path: str):
        self.path = path
        init_db(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            self._ensure_odds_snapshot_columns(connection)
            self._ensure_prediction_columns(connection)
            self._ensure_canonical_close_column(connection)
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _ensure_odds_snapshot_columns(connection: sqlite3.Connection) -> None:
        cols = {row["name"] for row in connection.execute("PRAGMA table_info(odds_snapshots)").fetchall()}
        if "line" not in cols:
            connection.execute("ALTER TABLE odds_snapshots ADD COLUMN line REAL")
        if "line_key" not in cols:
            connection.execute("ALTER TABLE odds_snapshots ADD COLUMN line_key REAL NOT NULL DEFAULT -1e308")
            connection.execute("UPDATE odds_snapshots SET line_key = COALESCE(line, -1e308) WHERE line_key IS NULL OR line_key = -1e308")

    def save_snapshots(self, rows: list[Snapshot]) -> int:
        """Insert snapshots, skipping keys that already exist.

        Returns the number of rows actually inserted. Existing rows are never
        modified, so a later snapshot with the same key is discarded.
        """

        if not rows:
            return 0

        parameters = [
            (
                row.event_uid,
                row.bookmaker,
                row.market,
                row.outcome,
                row.line,
                float(row.line) if row.line is not None else -1e308,
                row.odds_decimal,
                _to_iso(row.observed_at),
                row.source_event_id,
                row.source_url,
                row.adapter_version,
                int(row.is_live),
            )
            for row in rows
        ]

        with self._connect() as connection:
            before = connection.total_changes
            connection.executemany(_INSERT_SNAPSHOT, parameters)
            return connection.total_changes - before

    def last_snapshots(self, event_uid: str) -> list[Snapshot]:
        """Return the latest stored price per bookmaker, market, outcome and line."""

        with self._connect() as connection:
            rows = connection.execute(_LATEST_SNAPSHOTS, (event_uid,)).fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def adapter_health(
        self,
        bookmaker: str | None = None,
        status: str | None = None,
        last_success: datetime | None = None,
        error_count: int | None = None,
        adapter_version: str | None = None,
    ) -> list[AdapterHealth] | None:
        """Return all adapter health rows, or upsert one when args are provided.

        Prefer `list_adapter_health()` for reads and `upsert_adapter_health()`
        for writes. This dual form is kept for backward compatibility.
        """

        if bookmaker is None and status is None and last_success is None and error_count is None and adapter_version is None:
            return self.list_adapter_health()

        if bookmaker is None or status is None or error_count is None or adapter_version is None:
            raise TypeError("adapter_health upsert requires bookmaker, status, error_count, adapter_version")
        self.upsert_adapter_health(
            AdapterHealth(
                bookmaker=bookmaker,
                status=status,
                last_success=last_success,
                error_count=error_count,
                adapter_version=adapter_version,
            )
        )
        return None

    def list_adapter_health(self) -> list[AdapterHealth]:
        """Return all adapter health rows."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT bookmaker, status, last_success, error_count, adapter_version
                FROM adapter_health
                ORDER BY bookmaker
                """
            ).fetchall()
        return [
            AdapterHealth(
                bookmaker=row["bookmaker"],
                status=row["status"],
                last_success=_from_iso(row["last_success"]),
                error_count=row["error_count"],
                adapter_version=row["adapter_version"],
            )
            for row in rows
        ]

    def health(self) -> list[AdapterHealth]:
        """Alias used by collector-facing interface."""
        return self.list_adapter_health()

    def upsert_adapter_health(self, health: AdapterHealth) -> None:
        """Record the latest adapter state, replacing the previous one."""

        with self._connect() as connection:
            connection.execute(
                _UPSERT_ADAPTER_HEALTH,
                (
                    health.bookmaker,
                    health.status,
                    _to_iso(health.last_success),
                    health.error_count,
                    health.adapter_version,
                ),
            )

    def upsert_canonical_event(self, event: CanonicalEvent, market_close_at: datetime | None = None) -> None:
        """Insert or replace a canonical event used for result resolution."""

        with self._connect() as connection:
            self._ensure_canonical_close_column(connection)
            connection.execute(
                _UPSERT_CANONICAL_EVENT,
                (
                    event.event_uid,
                    event.sport,
                    event.league,
                    json.dumps(list(event.participant_ids), ensure_ascii=False),
                    _to_iso(event.start_at),
                    event.mapping_confidence,
                    _to_iso(market_close_at),
                ),
            )

    def set_market_close_at(self, event_uid: str, market_close_at: datetime) -> None:
        """Persist authoritative market close time for a canonical event."""

        with self._connect() as connection:
            self._ensure_canonical_close_column(connection)
            connection.execute(
                """
                UPDATE canonical_events
                SET market_close_at = ?
                WHERE event_uid = ?
                """,
                (_to_iso(market_close_at), event_uid),
            )

    def market_close_at(self, event_uid: str) -> datetime | None:
        """Return authoritative market close time from canonical event storage."""

        with self._connect() as connection:
            self._ensure_canonical_close_column(connection)
            row = connection.execute(
                "SELECT market_close_at FROM canonical_events WHERE event_uid = ?",
                (event_uid,),
            ).fetchone()
            if row is None or row["market_close_at"] is None:
                # Backward-compatible fallback for legacy rows/tests that only seeded events table.
                row = connection.execute(
                    "SELECT market_close_at FROM events WHERE event_uid = ?",
                    (event_uid,),
                ).fetchone()
                if row is None:
                    return None
            return _from_iso(row["market_close_at"])

    @staticmethod
    def _ensure_canonical_close_column(connection: sqlite3.Connection) -> None:
        cols = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(canonical_events)").fetchall()
        }
        if "market_close_at" not in cols:
            connection.execute("ALTER TABLE canonical_events ADD COLUMN market_close_at TEXT")

    def events_for_resolution(self) -> list[CanonicalEvent]:
        """Return all canonical events as deterministic CanonicalEvent objects."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_uid, sport, league, participant_ids, start_at, mapping_confidence
                FROM canonical_events
                ORDER BY event_uid
                """
            ).fetchall()
        events: list[CanonicalEvent] = []
        for row in rows:
            raw_ids = json.loads(row["participant_ids"])
            participant_ids = tuple(str(value) for value in raw_ids)
            events.append(
                CanonicalEvent(
                    event_uid=row["event_uid"],
                    sport=row["sport"],
                    league=row["league"],
                    participant_ids=participant_ids,
                    start_at=_from_iso(row["start_at"]),
                    mapping_confidence=float(row["mapping_confidence"]),
                )
            )
        return events

    def save_result(self, event_uid: str, result: "ResultRow") -> bool:
        """Store an accepted result row; return True only when inserted."""

        with self._connect() as connection:
            before = connection.total_changes
            connection.execute(
                _INSERT_RESULT,
                (
                    event_uid,
                    result.participant_a,
                    result.participant_b,
                    result.score,
                    result.league,
                    result.date,
                    result.source,
                ),
            )
            return connection.total_changes - before == 1

    def results(self, event_uid: str | None = None) -> list[dict]:
        """Return stored result rows, optionally filtered by event_uid."""

        query = """
            SELECT event_uid, participant_a, participant_b, score, league, event_date, source, created_at
            FROM results
        """
        params: tuple[object, ...] = ()
        if event_uid is not None:
            query += " WHERE event_uid = ?"
            params = (event_uid,)
        query += " ORDER BY event_date, id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "event_uid": row["event_uid"],
                "participant_a": row["participant_a"],
                "participant_b": row["participant_b"],
                "score": row["score"],
                "league": row["league"],
                "event_date": row["event_date"],
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _ensure_prediction_columns(connection: sqlite3.Connection) -> None:
        cols = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()}
        if "line" not in cols:
            connection.execute("ALTER TABLE predictions ADD COLUMN line REAL")
        if "bookmaker" not in cols:
            connection.execute("ALTER TABLE predictions ADD COLUMN bookmaker TEXT")

    def save_predictions(self, rows: list[dict[str, Any]]) -> int:
        """Replace predictions for a model run in one transaction."""
        if not rows:
            return 0
        run_id = rows[0].get("model_run_id")
        with self._connect() as conn:
            self._ensure_prediction_columns(conn)
            if run_id:
                conn.execute("DELETE FROM predictions WHERE model_run_id = ?", (run_id,))
                conn.execute("DELETE FROM predictions WHERE model_run_id <> ?", (run_id,))
            before = conn.total_changes
            conn.executemany(
                """
                INSERT INTO predictions (
                    model_run_id, event_uid, market, outcome, line, model_prob,
                    odds_decimal, bookmaker, observed_at, event_start_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.get("model_run_id"),
                        r["event_uid"],
                        r.get("market", "match_winner"),
                        r.get("outcome", "p1"),
                        float(r["line"]) if r.get("line") is not None else None,
                        float(r.get("model_prob", 0.5)),
                        float(r["odds_decimal"]) if r.get("odds_decimal") is not None else None,
                        r.get("bookmaker"),
                        _to_iso(r.get("observed_at")) or _to_iso(datetime.now(timezone.utc)),
                        _to_iso(r.get("event_start_at")),
                    )
                    for r in rows
                ],
            )
            return conn.total_changes - before

    def predictions(self, model_run_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._ensure_prediction_columns(conn)
            query = """
                SELECT model_run_id, event_uid, market, outcome, line, model_prob, odds_decimal, bookmaker, observed_at, event_start_at, created_at
                FROM predictions
            """
            params: list[Any] = []
            if model_run_id is not None:
                query += " WHERE model_run_id = ?"
                params.append(model_run_id)
            query += " ORDER BY created_at DESC, id DESC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
