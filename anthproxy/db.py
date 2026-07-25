"""SQLite persistence layer for the anthproxy web UI.

Schema version: 7
Thread safety: threading.Lock() for all writes; WAL mode for concurrent reads.
Migrations: PRAGMA user_version tracks applied schema version (no Alembic).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import date as _date
from datetime import datetime as _datetime

from .model_tier import classify_model_tier
from .stats import MODEL_PRICING, _classify_model

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 7


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def _apply_migration_0(conn: sqlite3.Connection) -> None:
    """Create initial schema (version 0 → 1)."""
    statements = [
        # Main requests table
        """CREATE TABLE IF NOT EXISTS requests (
            id                      INTEGER PRIMARY KEY,
            session_id              TEXT    NOT NULL,
            conversation_anchor     TEXT,
            request_ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            requested_model         TEXT    NOT NULL,
            routed_model            TEXT,
            classification          TEXT    CHECK(classification IN ('trivial','standard','deep') OR classification IS NULL),
            reason_code             TEXT,
            estimated_input_tokens  INTEGER,
            input_tokens            INTEGER,
            output_tokens           INTEGER,
            cache_creation_tokens   INTEGER,
            cache_read_tokens       INTEGER,
            duration_ms             INTEGER,
            backend                 TEXT    NOT NULL,
            status                  TEXT    NOT NULL CHECK(status IN ('success','error','rate_limited')),
            error                   TEXT,
            applied                 INTEGER,
            cost_estimate           REAL,
            model_tier              TEXT    CHECK(model_tier IN ('haiku','sonnet','opus') OR model_tier IS NULL),
            attempt                 INTEGER NOT NULL DEFAULT 1
        )""",
        "CREATE INDEX IF NOT EXISTS ix_req_session ON requests(session_id)",
        "CREATE INDEX IF NOT EXISTS ix_req_ts ON requests(request_ts DESC)",
        "CREATE INDEX IF NOT EXISTS ix_req_session_ts ON requests(session_id, request_ts DESC)",
        """CREATE INDEX IF NOT EXISTS ix_req_anchor ON requests(session_id, conversation_anchor, request_ts DESC)
            WHERE conversation_anchor IS NOT NULL""",
        "CREATE INDEX IF NOT EXISTS ix_req_model_tier_ts ON requests(model_tier, request_ts DESC)",
        "CREATE INDEX IF NOT EXISTS ix_req_backend_ts ON requests(backend, request_ts DESC)",
        # Sessions table
        """CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            last_seen_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            display_name    TEXT,
            pinned_backend  TEXT,
            pinned_tier     TEXT CHECK(pinned_tier IN ('haiku','sonnet','opus') OR pinned_tier IS NULL)
        )""",
        # Config changes audit log
        """CREATE TABLE IF NOT EXISTS config_changes (
            id          INTEGER PRIMARY KEY,
            ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            event_type  TEXT    NOT NULL,
            actor       TEXT    NOT NULL,
            actor_id    TEXT,
            prev_value  TEXT,
            new_value   TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS ix_cc_ts ON config_changes(ts DESC)",
    ]
    for stmt in statements:
        conn.execute(stmt)


def _apply_migration_1(conn: sqlite3.Connection) -> None:
    """Add prompt_store table and 10 new columns to requests (version 1 → 2)."""
    statements = [
        # New prompt_store table
        """CREATE TABLE IF NOT EXISTS prompt_store (
            content_hash  TEXT    PRIMARY KEY,
            content_type  TEXT    NOT NULL CHECK(content_type IN ('system','tools')),
            content       TEXT    NOT NULL,
            char_count    INTEGER NOT NULL,
            first_seen_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )""",
        # New columns on requests (one per ALTER TABLE — SQLite limitation)
        "ALTER TABLE requests ADD COLUMN user_prompt_text TEXT",
        "ALTER TABLE requests ADD COLUMN system_prompt_sha256 TEXT",
        "ALTER TABLE requests ADD COLUMN tools_sha256 TEXT",
        "ALTER TABLE requests ADD COLUMN routing_recovered_via_walkback INTEGER",
        "ALTER TABLE requests ADD COLUMN classifier_model TEXT",
        "ALTER TABLE requests ADD COLUMN classifier_summary_json TEXT",
        "ALTER TABLE requests ADD COLUMN classifier_raw_response TEXT",
        "ALTER TABLE requests ADD COLUMN classifier_confidence REAL",
        "ALTER TABLE requests ADD COLUMN classifier_format TEXT",
        "ALTER TABLE requests ADD COLUMN cache_savings_usd REAL",
        # Partial indexes on the new SHA-256 columns
        """CREATE INDEX IF NOT EXISTS ix_req_sys_sha ON requests(system_prompt_sha256)
            WHERE system_prompt_sha256 IS NOT NULL""",
        """CREATE INDEX IF NOT EXISTS ix_req_tools_sha ON requests(tools_sha256)
            WHERE tools_sha256 IS NOT NULL""",
    ]
    for stmt in statements:
        conn.execute(stmt)


def _apply_migration_2(conn: sqlite3.Connection) -> None:
    """Add session summaries table (version 2 → 3)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id  TEXT PRIMARY KEY REFERENCES sessions(session_id),
            summary     TEXT NOT NULL,
            updated_at  TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )


def _apply_migration_3(conn: sqlite3.Connection) -> None:
    """Add per-conversation (sub-session) summaries table (version 3 → 4)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            session_id           TEXT NOT NULL,
            conversation_anchor  TEXT NOT NULL,
            summary              TEXT NOT NULL,
            updated_at           TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            PRIMARY KEY (session_id, conversation_anchor)
        )
        """
    )


def _apply_migration_4(conn: sqlite3.Connection) -> None:
    """Add parent_conversation_anchor and response_text to requests (v4 → v5)."""
    conn.execute("ALTER TABLE requests ADD COLUMN parent_conversation_anchor TEXT")
    conn.execute("ALTER TABLE requests ADD COLUMN response_text TEXT")


def _apply_migration_5(conn: sqlite3.Connection) -> None:
    """Add casefolded search columns and composite index (v5 → v6)."""
    conn.execute("ALTER TABLE requests ADD COLUMN user_prompt_search TEXT")
    conn.execute("ALTER TABLE requests ADD COLUMN response_search TEXT")
    # Batched backfill: casefold in Python (SQLite has no native casefold)
    min_row = conn.execute("SELECT MIN(id) FROM requests").fetchone()[0]
    max_row = conn.execute("SELECT MAX(id) FROM requests").fetchone()[0]
    if min_row is not None and max_row is not None:
        batch_size = 1000
        lo = min_row
        while lo <= max_row:
            hi = lo + batch_size - 1
            rows = conn.execute(
                "SELECT id, user_prompt_text, response_text FROM requests WHERE id BETWEEN ? AND ?",
                (lo, hi),
            ).fetchall()
            for row in rows:
                ups = row[1].casefold() if row[1] else None
                rs = row[2].casefold() if row[2] else None
                conn.execute(
                    "UPDATE requests SET user_prompt_search=?, response_search=? WHERE id=?",
                    (ups, rs, row[0]),
                )
            lo += batch_size
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_req_session_ts_id "
        "ON requests(session_id, request_ts DESC, id DESC)"
    )


def _apply_migration_6(conn: sqlite3.Connection) -> None:
    """Widen model_tier and pinned_tier CHECK constraints to include 'fable' (v6 → v7)."""
    with conn:
        # Rebuild requests table
        conn.executescript("""
DROP TABLE IF EXISTS requests_v7;
CREATE TABLE requests_v7 (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    conversation_anchor TEXT,
    request_ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    requested_model TEXT NOT NULL,
    routed_model TEXT,
    classification TEXT CHECK(classification IN ('trivial','standard','deep') OR classification IS NULL),
    reason_code TEXT,
    estimated_input_tokens INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_creation_tokens INTEGER,
    cache_read_tokens INTEGER,
    duration_ms INTEGER,
    backend TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('success','error','rate_limited')),
    error TEXT,
    applied INTEGER,
    cost_estimate REAL,
    model_tier TEXT CHECK(model_tier IN ('haiku','sonnet','opus','fable') OR model_tier IS NULL),
    attempt INTEGER NOT NULL DEFAULT 1,
    user_prompt_text TEXT,
    system_prompt_sha256 TEXT,
    tools_sha256 TEXT,
    routing_recovered_via_walkback INTEGER,
    classifier_model TEXT,
    classifier_summary_json TEXT,
    classifier_raw_response TEXT,
    classifier_confidence REAL,
    classifier_format TEXT,
    cache_savings_usd REAL,
    parent_conversation_anchor TEXT,
    response_text TEXT,
    user_prompt_search TEXT,
    response_search TEXT
);

INSERT INTO requests_v7 SELECT
    id, session_id, conversation_anchor, request_ts, requested_model, routed_model,
    classification, reason_code, estimated_input_tokens, input_tokens, output_tokens,
    cache_creation_tokens, cache_read_tokens, duration_ms, backend, status, error,
    applied, cost_estimate, model_tier, attempt, user_prompt_text, system_prompt_sha256,
    tools_sha256, routing_recovered_via_walkback, classifier_model,
    classifier_summary_json, classifier_raw_response, classifier_confidence,
    classifier_format, cache_savings_usd, parent_conversation_anchor, response_text,
    user_prompt_search, response_search
FROM requests;

DROP TABLE requests;
ALTER TABLE requests_v7 RENAME TO requests;

CREATE INDEX ix_req_session ON requests(session_id);
CREATE INDEX ix_req_ts ON requests(request_ts DESC);
CREATE INDEX ix_req_session_ts ON requests(session_id, request_ts DESC);
CREATE INDEX ix_req_anchor ON requests(session_id, conversation_anchor, request_ts DESC) WHERE conversation_anchor IS NOT NULL;
CREATE INDEX ix_req_model_tier_ts ON requests(model_tier, request_ts DESC);
CREATE INDEX ix_req_backend_ts ON requests(backend, request_ts DESC);
CREATE INDEX ix_req_sys_sha ON requests(system_prompt_sha256) WHERE system_prompt_sha256 IS NOT NULL;
CREATE INDEX ix_req_tools_sha ON requests(tools_sha256) WHERE tools_sha256 IS NOT NULL;
CREATE INDEX ix_req_session_ts_id ON requests(session_id, request_ts DESC, id DESC);

UPDATE requests
SET model_tier = 'fable'
WHERE model_tier IS NULL AND lower(routed_model) LIKE '%fable%';
""")

        # Rebuild sessions table
        conn.executescript("""
DROP TABLE IF EXISTS sessions_v7;
CREATE TABLE sessions_v7 (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    display_name TEXT,
    pinned_backend TEXT,
    pinned_tier TEXT CHECK(pinned_tier IN ('haiku','sonnet','opus','fable') OR pinned_tier IS NULL)
);

INSERT INTO sessions_v7 SELECT
    session_id, created_at, last_seen_at, display_name, pinned_backend, pinned_tier
FROM sessions;

DROP TABLE sessions;
ALTER TABLE sessions_v7 RENAME TO sessions;
""")

        # Post-migration integrity check
        orphans = conn.execute('PRAGMA foreign_key_check').fetchall()
        if orphans:
            raise RuntimeError(f'FK check failed after migration 6: {orphans}')


_MIGRATIONS: dict[int, object] = {
    0: _apply_migration_0,
    1: _apply_migration_1,
    2: _apply_migration_2,
    3: _apply_migration_3,
    4: _apply_migration_4,
    5: _apply_migration_5,
    6: _apply_migration_6,
}


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Apply pending schema migrations using PRAGMA user_version."""
    current = conn.execute("PRAGMA user_version;").fetchone()[0]
    for v in range(current, _SCHEMA_VERSION):
        with conn:
            _MIGRATIONS[v](conn)  # type: ignore[operator]
            conn.execute(f"PRAGMA user_version = {v + 1};")


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _tier_from_model(m):
    t = classify_model_tier(m or '')
    return None if t == 'other' else t


def compute_cost(model: str, stats: dict) -> float:
    """Return the USD cost for one request.

    Uses :func:`_classify_model` to map *model* to a pricing tier, then
    looks up the (input, output, cache_read, cache_write) tuple from
    ``MODEL_PRICING``.  Returns 0.0 on any error (unknown tier, missing
    pricing, bad stats values).
    """
    try:
        tier = _classify_model(model or '')
        price = MODEL_PRICING.get(tier)
        if price is None:
            return 0.0
        in_p, out_p, cr_p, cw_p = price
        return (
            int(stats.get('input_tokens') or 0) * in_p
            + int(stats.get('output_tokens') or 0) * out_p
            + int(stats.get('cache_read_tokens') or 0) * cr_p
            + int(stats.get('cache_creation_tokens') or 0) * cw_p
        ) / 1_000_000
    except Exception:
        return 0.0


def _compute_cache_savings(
    routed_model: str | None,
    cache_read_tokens: int | None,
) -> float | None:
    """Return USD saved from cache reads vs. full input price, or None.

    Computes ``cache_read_tokens * (input_price - cache_read_price) / 1_000_000``
    for the model's tier.  Returns None when the model is unknown, the tier has
    no pricing entry, or ``cache_read_tokens`` is zero/None.
    """
    if not cache_read_tokens:
        return None
    try:
        tier = _classify_model(routed_model or '')
        pricing = MODEL_PRICING.get(tier)
        if pricing is None:
            return None
        # pricing: (input, output, cache_read, cache_write) per MTok
        saved = cache_read_tokens * (pricing[0] - pricing[2]) / 1_000_000
        return saved if saved > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SessionDB
# ---------------------------------------------------------------------------

class SessionDB:
    """Thread-safe SQLite persistence layer for anthproxy request data.

    All write methods acquire ``self._lock`` before touching the connection so
    concurrent HTTP handler threads do not corrupt the database.  Read methods
    use per-thread read connections (``self._tls``) — WAL mode allows
    concurrent reads alongside a single writer.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL mode + relaxed fsync for throughput; busy_timeout prevents
        # "database is locked" errors under concurrent writers.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._lock = threading.Lock()
        self._db_path = db_path  # needed by per-thread read connections
        self._tls = threading.local()  # per-thread read connections
        self._retention_stop = threading.Event()
        self._retention_thread: threading.Thread | None = None
        self.ensure_schema()

    def ensure_schema(self) -> None:
        """Apply any pending schema migrations."""
        ensure_schema(self._conn)

    def _read_conn(self) -> sqlite3.Connection:
        """Return (or create) the per-thread read-only connection."""
        conn = getattr(self._tls, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._tls.conn = conn
        return conn

    # -----------------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------------

    def record_request(
        self,
        session_id: str,
        conversation_anchor: str | None,
        routing_decision,       # ModelRoutingDecision
        stats_dict: dict,       # {input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens}
        duration_ms: int,
        backend: str,
        status: str,            # 'success' | 'error' | 'rate_limited'
        error: str | None = None,
        attempt: int = 1,
        # New in schema v2:
        user_prompt_text: str | None = None,
        system_prompt_sha256: str | None = None,
        tools_sha256: str | None = None,
        routing_recovered_via_walkback: bool | None = None,
        classifier_model: str | None = None,
        classifier_summary_json: str | None = None,
        classifier_raw_response: str | None = None,
        classifier_confidence: float | None = None,
        classifier_format: str | None = None,
        prompt_store_entries: dict[str, tuple[str, str]] | None = None,
        response_text: str | None = None,
    ) -> int:
        """Insert one request row and upsert the owning session row.

        Returns the integer ``rowid`` of the newly inserted request row.
        Both operations are wrapped in a single transaction under ``self._lock``.
        Returns -1 if ``routing_decision`` is None.

        ``cache_savings_usd`` is computed internally from ``routed_model`` and
        ``cache_read_tokens``; it is NOT accepted as a parameter.

        ``prompt_store_entries`` is a dict mapping sha256 hex → (content_type, content).
        Each entry is upserted into ``prompt_store`` (INSERT OR IGNORE) within the
        same transaction.
        """
        if routing_decision is None:
            return -1

        routed_model = routing_decision.routed_model
        model_tier = _tier_from_model(routed_model)
        cost_estimate = compute_cost(routed_model, stats_dict)
        cache_read_tokens = stats_dict.get('cache_read_tokens')
        cache_savings_usd = _compute_cache_savings(routed_model, cache_read_tokens)
        user_prompt_search = user_prompt_text.casefold() if user_prompt_text else None
        response_search = response_text.casefold() if response_text else None

        # SQLite stores booleans as integers (0/1) or NULL
        walkback_int: int | None = None
        if routing_recovered_via_walkback is not None:
            walkback_int = int(routing_recovered_via_walkback)

        with self._lock:
            # Compute parent_anchor inside the lock to ensure atomicity with the INSERT.
            parent_anchor: str | None = None
            if conversation_anchor is not None:
                row = self._read_conn().execute(
                    """SELECT conversation_anchor FROM requests
                       WHERE session_id = ? AND conversation_anchor IS NOT NULL
                       ORDER BY request_ts ASC LIMIT 1""",
                    (session_id,),
                ).fetchone()
                if row:
                    earliest = row['conversation_anchor']
                    parent_anchor = earliest if earliest != conversation_anchor else None
            with self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO requests (
                        session_id, conversation_anchor, requested_model,
                        routed_model, classification, reason_code,
                        estimated_input_tokens, input_tokens, output_tokens,
                        cache_creation_tokens, cache_read_tokens, duration_ms,
                        backend, status, error, applied, cost_estimate,
                        model_tier, attempt,
                        user_prompt_text, system_prompt_sha256, tools_sha256,
                        routing_recovered_via_walkback, classifier_model,
                        classifier_summary_json, classifier_raw_response,
                        classifier_confidence, classifier_format, cache_savings_usd,
                        parent_conversation_anchor, response_text,
                        user_prompt_search, response_search
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        session_id,
                        conversation_anchor,
                        routing_decision.requested_model,
                        routed_model,
                        routing_decision.classification,
                        routing_decision.reason_code,
                        routing_decision.estimated_input_tokens,
                        stats_dict.get('input_tokens'),
                        stats_dict.get('output_tokens'),
                        stats_dict.get('cache_creation_tokens'),
                        cache_read_tokens,
                        duration_ms,
                        backend,
                        status,
                        error,
                        1 if routing_decision.applied else 0,
                        cost_estimate,
                        model_tier,
                        attempt,
                        user_prompt_text,
                        system_prompt_sha256,
                        tools_sha256,
                        walkback_int,
                        classifier_model,
                        classifier_summary_json,
                        classifier_raw_response,
                        classifier_confidence,
                        classifier_format,
                        cache_savings_usd,
                        parent_anchor,
                        response_text,
                        user_prompt_search,
                        response_search,
                    ),
                )
                rowid: int = cur.lastrowid  # type: ignore[assignment]
                # Upsert session: create on first request, bump last_seen_at on
                # subsequent requests.
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, last_seen_at)
                    VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    """,
                    (session_id,),
                )
                # Upsert prompt_store entries (INSERT OR IGNORE preserves first_seen_at)
                if prompt_store_entries:
                    for sha, (content_type, content) in prompt_store_entries.items():
                        self._conn.execute(
                            """
                            INSERT OR IGNORE INTO prompt_store
                                (content_hash, content_type, content, char_count)
                            VALUES (?, ?, ?, ?)
                            """,
                            (sha, content_type, content, len(content)),
                        )
        return rowid

    def update_request_on_retry(
        self,
        request_id: int,
        new_backend: str,
        attempt: int,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        cost_estimate: float,
        status: str,
        error: str | None = None,
        response_text: str | None = None,
    ) -> None:
        """Update an existing request row after a transparent retry completes.

        Recomputes ``cache_savings_usd`` from the stored ``routed_model`` and the
        new ``cache_read_tokens``.  All other new schema-v2 fields (classifier
        fields, routing_recovered_via_walkback, SHA-256 columns) are NOT updated.
        ``parent_conversation_anchor`` is immutable after INSERT and is never updated.
        """
        response_search = response_text.casefold() if response_text else None
        with self._lock:
            with self._conn:
                # Look up the routed_model stored at insert time so we can
                # recompute cache_savings_usd with the updated token count.
                row = self._conn.execute(
                    "SELECT routed_model FROM requests WHERE id=?", (request_id,)
                ).fetchone()
                routed_model_db: str | None = dict(row)['routed_model'] if row else None
                cache_savings_usd = _compute_cache_savings(routed_model_db, cache_read_tokens)

                self._conn.execute(
                    """
                    UPDATE requests
                    SET backend=?, attempt=?, status=?, error=?,
                        input_tokens=?, output_tokens=?,
                        cache_creation_tokens=?, cache_read_tokens=?,
                        cost_estimate=?, cache_savings_usd=?,
                        response_text=?, response_search=?
                    WHERE id=?
                    """,
                    (
                        new_backend, attempt, status, error,
                        input_tokens, output_tokens,
                        cache_creation_tokens, cache_read_tokens,
                        cost_estimate, cache_savings_usd,
                        response_text, response_search, request_id,
                    ),
                )

    def set_session_backend(self, session_id: str, backend: str | None) -> None:
        """Pin or unpin the backend for a session (upserts the session row)."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, pinned_backend)
                    VALUES (?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        pinned_backend = excluded.pinned_backend
                    """,
                    (session_id, backend),
                )

    def set_session_tier(self, session_id: str, tier: str | None) -> None:
        """Pin or unpin the model tier for a session (upserts the session row)."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, pinned_tier)
                    VALUES (?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        pinned_tier = excluded.pinned_tier
                    """,
                    (session_id, tier),
                )

    def record_config_change(
        self,
        event_type: str,
        actor: str,
        actor_id: str | None,
        prev_value: str | None,
        new_value: str | None,
    ) -> None:
        """Append one row to the config_changes audit log."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO config_changes
                        (event_type, actor, actor_id, prev_value, new_value)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (event_type, actor, actor_id, prev_value, new_value),
                )

    # -----------------------------------------------------------------------
    # Reads (per-thread read connections — WAL allows concurrent readers)
    # -----------------------------------------------------------------------

    def get_sessions(self, limit: int = 50, offset: int = 0, q: str | None = None) -> list[dict]:
        """Return paginated sessions with aggregate stats and summaries.

        Each dict contains: session_id, created_at, last_seen_at,
        display_name, pinned_backend, pinned_tier, summary, summary_updated_at,
        request_count, total_input_tokens, total_output_tokens,
        total_cache_creation, total_cache_read, estimated_cost_usd.

        If q is provided, filter to sessions where any recent request matches
        the search term in user_prompt_search or response_search (casefolded INSTR).
        """
        if not q:
            # Unfiltered path
            rows = self._read_conn().execute(
                """
                SELECT
                    s.session_id,
                    s.created_at,
                    s.last_seen_at,
                    s.display_name,
                    s.pinned_backend,
                    s.pinned_tier,
                    ss.summary,
                    ss.updated_at AS summary_updated_at,
                    COUNT(r.id)                               AS request_count,
                    COALESCE(SUM(r.input_tokens), 0)          AS total_input_tokens,
                    COALESCE(SUM(r.output_tokens), 0)         AS total_output_tokens,
                    COALESCE(SUM(r.cache_creation_tokens), 0) AS total_cache_creation,
                    COALESCE(SUM(r.cache_read_tokens), 0)     AS total_cache_read,
                    COALESCE(SUM(r.cost_estimate), 0.0)       AS estimated_cost_usd
                FROM sessions s
                LEFT JOIN session_summaries ss ON ss.session_id = s.session_id
                LEFT JOIN requests r ON s.session_id = r.session_id
                GROUP BY s.session_id
                ORDER BY s.last_seen_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        else:
            # Filtered path: search within latest 100 requests per session
            q_cf = q.casefold()
            rows = self._read_conn().execute(
                """
                WITH latest AS (
                    SELECT session_id, user_prompt_search, response_search,
                           ROW_NUMBER() OVER (
                               PARTITION BY session_id
                               ORDER BY request_ts DESC, id DESC) AS rn
                    FROM requests
                ),
                matched AS (
                    SELECT DISTINCT session_id FROM latest
                    WHERE rn <= 100
                      AND (INSTR(COALESCE(user_prompt_search, ''), ?) > 0
                        OR INSTR(COALESCE(response_search,  ''), ?) > 0)
                )
                SELECT
                    s.session_id,
                    s.created_at,
                    s.last_seen_at,
                    s.display_name,
                    s.pinned_backend,
                    s.pinned_tier,
                    ss.summary,
                    ss.updated_at AS summary_updated_at,
                    COUNT(r.id)                               AS request_count,
                    COALESCE(SUM(r.input_tokens), 0)          AS total_input_tokens,
                    COALESCE(SUM(r.output_tokens), 0)         AS total_output_tokens,
                    COALESCE(SUM(r.cache_creation_tokens), 0) AS total_cache_creation,
                    COALESCE(SUM(r.cache_read_tokens), 0)     AS total_cache_read,
                    COALESCE(SUM(r.cost_estimate), 0.0)       AS estimated_cost_usd
                FROM sessions s
                JOIN matched m ON m.session_id = s.session_id
                LEFT JOIN session_summaries ss ON ss.session_id = s.session_id
                LEFT JOIN requests r ON s.session_id = r.session_id
                GROUP BY s.session_id
                ORDER BY s.last_seen_at DESC
                LIMIT ? OFFSET ?
                """,
                (q_cf, q_cf, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_sessions_count(self, q: str | None = None) -> int:
        """Return the total number of sessions (for pagination).

        If q is provided, count only sessions with matching recent requests.
        """
        if not q:
            row = self._read_conn().execute("SELECT COUNT(*) FROM sessions").fetchone()
        else:
            q_cf = q.casefold()
            row = self._read_conn().execute(
                """
                WITH latest AS (
                    SELECT session_id, user_prompt_search, response_search,
                           ROW_NUMBER() OVER (
                               PARTITION BY session_id
                               ORDER BY request_ts DESC, id DESC) AS rn
                    FROM requests
                ),
                matched AS (
                    SELECT DISTINCT session_id FROM latest
                    WHERE rn <= 100
                      AND (INSTR(COALESCE(user_prompt_search, ''), ?) > 0
                        OR INSTR(COALESCE(response_search,  ''), ?) > 0)
                )
                SELECT COUNT(*) FROM matched
                """,
                (q_cf, q_cf),
            ).fetchone()
        return row[0] if row else 0

    def get_session(self, session_id: str) -> dict | None:
        """Return full session detail or None if the session is not found.

        The returned dict contains:
        - All columns from the sessions table plus summary and summary_updated_at.
        - Aggregate columns: request_count, total_input_tokens,
          total_output_tokens, total_cache_creation, total_cache_read,
          estimated_cost_usd.
        - model_breakdown: list of per-routed-model aggregate dicts.
        - conversations: list of per-conversation-anchor aggregate dicts
          (anchor, request_count, started_at, last_request_ts, cost_usd).
        """
        row = self._read_conn().execute(
            """
            SELECT
                s.*,
                ss.summary,
                ss.updated_at AS summary_updated_at
            FROM sessions s
            LEFT JOIN session_summaries ss ON ss.session_id = s.session_id
            WHERE s.session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None

        result = dict(row)

        # Aggregates
        agg = self._read_conn().execute(
            """
            SELECT
                COUNT(id)                               AS request_count,
                COALESCE(SUM(input_tokens), 0)          AS total_input_tokens,
                COALESCE(SUM(output_tokens), 0)         AS total_output_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS total_cache_creation,
                COALESCE(SUM(cache_read_tokens), 0)     AS total_cache_read,
                COALESCE(SUM(cost_estimate), 0.0)       AS estimated_cost_usd
            FROM requests
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if agg:
            result.update(dict(agg))

        # Per-model breakdown
        model_rows = self._read_conn().execute(
            """
            SELECT
                routed_model,
                COUNT(id)                               AS request_count,
                COALESCE(SUM(input_tokens), 0)          AS input_tokens,
                COALESCE(SUM(output_tokens), 0)         AS output_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation,
                COALESCE(SUM(cache_read_tokens), 0)     AS cache_read,
                COALESCE(SUM(cost_estimate), 0.0)       AS cost_usd
            FROM requests
            WHERE session_id = ?
            GROUP BY routed_model
            ORDER BY request_count DESC
            """,
            (session_id,),
        ).fetchall()
        result['model_breakdown'] = [dict(r) for r in model_rows]

        # Conversation list — one row per unique conversation_anchor, joined
        # with any generated per-conversation summary.
        conv_rows = self._read_conn().execute(
            """
            SELECT
                r.conversation_anchor                       AS conversation_anchor,
                COUNT(r.id)                                 AS request_count,
                MIN(r.request_ts)                           AS started_at,
                MAX(r.request_ts)                           AS last_request_ts,
                COALESCE(SUM(r.cost_estimate), 0.0)         AS cost_usd,
                cs.summary                                  AS summary,
                MIN(r.parent_conversation_anchor)           AS parent_conversation_anchor
            FROM requests r
            LEFT JOIN conversation_summaries cs
                ON cs.session_id = r.session_id
               AND cs.conversation_anchor = r.conversation_anchor
            WHERE r.session_id = ?
            GROUP BY r.conversation_anchor
            ORDER BY last_request_ts DESC
            """,
            (session_id,),
        ).fetchall()
        result['conversations'] = [dict(r) for r in conv_rows]

        return result

    def get_session_metadata(self, session_id: str) -> dict | None:
        """Return {pinned_backend, pinned_tier} or None if the session is not found.

        Lightweight read used on every request to check per-session pins.
        """
        row = self._read_conn().execute(
            "SELECT pinned_backend, pinned_tier FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_trace(
        self,
        session_id: str,
        anchor: str | None = None,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
    ) -> list[dict]:
        """Return request rows for a session.

        When *anchor* is given, rows are filtered to that conversation and
        returned in ascending timestamp order (conversation flow).  Without
        an anchor, all requests for the session are returned in descending
        order (most recent first).

        If q is provided, filter to rows where user_prompt_search or
        response_search (casefolded INSTR) match the query term.
        """
        q_cf = q.casefold() if q else None
        if anchor is not None:
            sql = """
                SELECT * FROM requests
                WHERE session_id = ? AND conversation_anchor = ?
            """
            params = [session_id, anchor]
            if q_cf:
                sql += """
                  AND (INSTR(COALESCE(user_prompt_search, ''), ?) > 0
                    OR INSTR(COALESCE(response_search,  ''), ?) > 0)
                """
                params.extend([q_cf, q_cf])
            sql += " ORDER BY request_ts ASC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = self._read_conn().execute(sql, params).fetchall()
        else:
            sql = "SELECT * FROM requests WHERE session_id = ?"
            params = [session_id]
            if q_cf:
                sql += """
                  AND (INSTR(COALESCE(user_prompt_search, ''), ?) > 0
                    OR INSTR(COALESCE(response_search,  ''), ?) > 0)
                """
                params.extend([q_cf, q_cf])
            sql += " ORDER BY request_ts DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = self._read_conn().execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_trace_count(
        self,
        session_id: str,
        anchor: str | None = None,
        q: str | None = None,
    ) -> int:
        """Return the total number of request rows matching a get_trace query.

        Mirrors the WHERE clause of :meth:`get_trace` without LIMIT/OFFSET so
        callers can compute pagination bounds.

        If q is provided, count only rows matching the search term.
        """
        q_cf = q.casefold() if q else None
        if anchor is not None:
            sql = "SELECT COUNT(*) FROM requests WHERE session_id = ? AND conversation_anchor = ?"
            params = [session_id, anchor]
            if q_cf:
                sql += """
                  AND (INSTR(COALESCE(user_prompt_search, ''), ?) > 0
                    OR INSTR(COALESCE(response_search,  ''), ?) > 0)
                """
                params.extend([q_cf, q_cf])
            row = self._read_conn().execute(sql, params).fetchone()
        else:
            sql = "SELECT COUNT(*) FROM requests WHERE session_id = ?"
            params = [session_id]
            if q_cf:
                sql += """
                  AND (INSTR(COALESCE(user_prompt_search, ''), ?) > 0
                    OR INSTR(COALESCE(response_search,  ''), ?) > 0)
                """
                params.extend([q_cf, q_cf])
            row = self._read_conn().execute(sql, params).fetchone()
        return row[0] if row else 0

    def get_cost(
        self,
        group_by: str = 'model',
        since: str = '-7 days',
        session_id: str | None = None,
    ) -> list[dict]:
        """Return cost aggregates grouped by model, tier, or backend.

        *group_by* must be ``'model'`` (default), ``'tier'``, or
        ``'backend'``.  *since* is a SQLite datetime modifier, e.g.
        ``'-7 days'``, ``'-1 days'``, ``'-30 days'``.

        Each returned dict has: key, requests, input_tokens, output_tokens,
        cache_creation, cache_read, cost_usd, cache_savings_usd.
        """
        if group_by == 'tier':
            key_col = 'model_tier'
        elif group_by == 'backend':
            key_col = 'backend'
        else:
            key_col = 'routed_model'

        where_clause = "WHERE request_ts >= datetime('now', ?)"
        params: list = [since]
        if session_id is not None:
            where_clause += ' AND session_id = ?'
            params.append(session_id)

        rows = self._read_conn().execute(
            f"""
            SELECT
                {key_col}                               AS key,
                COUNT(id)                               AS requests,
                COALESCE(SUM(input_tokens), 0)          AS input_tokens,
                COALESCE(SUM(output_tokens), 0)         AS output_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation,
                COALESCE(SUM(cache_read_tokens), 0)     AS cache_read,
                COALESCE(SUM(cost_estimate), 0.0)        AS cost_usd,
                COALESCE(SUM(cache_savings_usd), 0.0)    AS cache_savings_usd
            FROM requests
            {where_clause}
            GROUP BY {key_col}
            ORDER BY cost_usd DESC
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def get_routing(
        self,
        since: str = '-7 days',
        session_id: str | None = None,
    ) -> dict:
        """Return routing reason_code distribution and tier transition counts.

        Returns::

            {
                'reason_code_distribution': [
                    {'reason_code': str, 'applied': int, 'classification': str|None, 'cnt': int},
                    ...
                ],
                'tier_transitions': [
                    {'requested_model': str, 'routed_model': str, 'cnt': int},
                    ...
                ],
            }
        """
        where_clause = "WHERE request_ts >= datetime('now', ?)"
        params: list = [since]
        if session_id is not None:
            where_clause += ' AND session_id = ?'
            params.append(session_id)

        reason_rows = self._read_conn().execute(
            f"""
            SELECT reason_code, applied, classification, COUNT(id) AS cnt
            FROM requests
            {where_clause}
            GROUP BY reason_code, applied, classification
            ORDER BY cnt DESC
            """,
            params,
        ).fetchall()

        transition_rows = self._read_conn().execute(
            f"""
            SELECT requested_model, routed_model, COUNT(id) AS cnt
            FROM requests
            {where_clause}
            GROUP BY requested_model, routed_model
            ORDER BY cnt DESC
            """,
            params,
        ).fetchall()

        return {
            'reason_code_distribution': [dict(r) for r in reason_rows],
            'tier_transitions': [dict(r) for r in transition_rows],
        }

    def get_config_changes(self, limit: int = 100) -> list[dict]:
        """Return recent config_changes rows in descending timestamp order."""
        rows = self._read_conn().execute(
            "SELECT * FROM config_changes ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_session_overrides(self) -> list[dict]:
        """Return sessions that have a pinned backend or pinned tier.

        Each dict contains: session_id, display_name, pinned_backend,
        pinned_tier.  Ordered by last_seen_at DESC.
        """
        rows = self._read_conn().execute(
            """
            SELECT session_id, display_name, pinned_backend, pinned_tier
            FROM sessions
            WHERE pinned_backend IS NOT NULL OR pinned_tier IS NOT NULL
            ORDER BY last_seen_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_session_summary(self, session_id: str) -> dict | None:
        """Return a persisted session summary, or None when not generated yet."""
        row = self._read_conn().execute(
            """
            SELECT session_id, summary, updated_at
            FROM session_summaries
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def upsert_session_summary(self, session_id: str, summary: str) -> None:
        """Persist the latest generated summary for a session."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO session_summaries (session_id, summary, updated_at)
                    VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    ON CONFLICT(session_id) DO UPDATE SET
                        summary = excluded.summary,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    """,
                    (session_id, summary),
                )

    def get_sessions_for_summary(
        self,
        limit: int = 20,
    ) -> list[dict]:
        """Return recent sessions with missing or stale summaries.

        Each item contains:
        - session_id
        - recent_prompts: up to three newest non-empty user prompts
        - recent_system_prompt: newest stored system prompt, used only when
          no readable user prompts are available.
        """
        session_rows = self._read_conn().execute(
            """
            SELECT s.session_id
            FROM sessions AS s
            LEFT JOIN session_summaries AS ss
                ON ss.session_id = s.session_id
            WHERE s.last_seen_at >= strftime('%Y-%m-%dT%H:%M:%fZ','now','-24 hours')
              AND (
                  ss.updated_at IS NULL
                  OR ss.updated_at < s.last_seen_at
              )
            ORDER BY s.last_seen_at DESC, s.session_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        result: list[dict] = []
        conn = self._read_conn()
        for row in session_rows:
            session_id = row['session_id']

            prompt_rows = conn.execute(
                """
                SELECT user_prompt_text
                FROM requests
                WHERE session_id = ?
                  AND user_prompt_text IS NOT NULL
                  AND trim(user_prompt_text) != ''
                ORDER BY request_ts DESC
                LIMIT 3
                """,
                (session_id,),
            ).fetchall()

            system_row = conn.execute(
                """
                SELECT prompt_store.content
                FROM requests
                JOIN prompt_store
                  ON prompt_store.content_hash = requests.system_prompt_sha256
                WHERE requests.session_id = ?
                  AND requests.system_prompt_sha256 IS NOT NULL
                ORDER BY requests.request_ts DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            result.append(
                {
                    'session_id': session_id,
                    'recent_prompts': [
                        prompt['user_prompt_text'] for prompt in prompt_rows
                    ],
                    'recent_system_prompt': (
                        system_row['content'] if system_row is not None else None
                    ),
                }
            )

        return result

    def upsert_conversation_summary(
        self,
        session_id: str,
        conversation_anchor: str,
        summary: str,
    ) -> None:
        """Persist the latest generated summary for one conversation (sub-session)."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO conversation_summaries
                        (session_id, conversation_anchor, summary, updated_at)
                    VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    ON CONFLICT(session_id, conversation_anchor) DO UPDATE SET
                        summary = excluded.summary,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    """,
                    (session_id, conversation_anchor, summary),
                )

    def get_conversations_for_summary(
        self,
        limit: int = 20,
    ) -> list[dict]:
        """Return recent conversations with missing or stale summaries.

        Each item contains:
        - session_id
        - conversation_anchor
        - recent_prompts: up to three newest non-empty user prompts for that
          conversation.

        Only conversations with a non-NULL anchor and at least one request in
        the last 24 hours are considered; a conversation is returned when it has
        no summary yet, or its newest request post-dates the stored summary.
        """
        conn = self._read_conn()
        conv_rows = conn.execute(
            """
            SELECT
                r.session_id           AS session_id,
                r.conversation_anchor  AS conversation_anchor,
                MAX(r.request_ts)      AS last_request_ts
            FROM requests r
            LEFT JOIN conversation_summaries cs
                ON cs.session_id = r.session_id
               AND cs.conversation_anchor = r.conversation_anchor
            WHERE r.conversation_anchor IS NOT NULL
              AND r.request_ts >= strftime('%Y-%m-%dT%H:%M:%fZ','now','-24 hours')
            GROUP BY r.session_id, r.conversation_anchor
            HAVING cs.updated_at IS NULL OR cs.updated_at < MAX(r.request_ts)
            ORDER BY last_request_ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        result: list[dict] = []
        for row in conv_rows:
            session_id = row['session_id']
            anchor = row['conversation_anchor']

            prompt_rows = conn.execute(
                """
                SELECT user_prompt_text
                FROM requests
                WHERE session_id = ?
                  AND conversation_anchor = ?
                  AND user_prompt_text IS NOT NULL
                  AND trim(user_prompt_text) != ''
                ORDER BY request_ts DESC
                LIMIT 3
                """,
                (session_id, anchor),
            ).fetchall()

            result.append(
                {
                    'session_id': session_id,
                    'conversation_anchor': anchor,
                    'recent_prompts': [
                        prompt['user_prompt_text'] for prompt in prompt_rows
                    ],
                }
            )

        return result

    def get_request(self, request_id: int) -> dict | None:
        """Return a single request row joined with prompt content, or None.

        The returned dict contains all ``requests`` columns plus:
        - ``system_prompt_content`` — system prompt text (or None)
        - ``system_prompt_char_count`` — char count of system prompt (or None)
        - ``tools_content`` — tools JSON text (or None)
        - ``tools_char_count`` — char count of tools content (or None)
        """
        row = self._read_conn().execute(
            """
            SELECT
                r.*,
                sp.content   AS system_prompt_content,
                sp.char_count AS system_prompt_char_count,
                tp.content   AS tools_content,
                tp.char_count AS tools_char_count
            FROM requests AS r
            LEFT JOIN prompt_store AS sp ON r.system_prompt_sha256 = sp.content_hash
            LEFT JOIN prompt_store AS tp ON r.tools_sha256 = tp.content_hash
            WHERE r.id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def busy_secs_window(self, backend: str, since: str) -> float | None:
        """Return wall-clock busy seconds for *backend* requests since *since*.

        Computes the union of request time intervals [request_ts, request_ts +
        duration_ms], merging overlaps, so concurrent requests are not
        double-counted.  This yields actual wall-clock time the backend was
        active — bounded by the window length — rather than a naive sum of
        per-request durations (which inflates ~Nx under heavy concurrency).

        Returns None when no completed requests exist for that window.
        """
        row = self._read_conn().execute(
            """
            WITH iv AS (
                SELECT (julianday(request_ts) - 2440587.5) * 86400000.0 AS s,
                       (julianday(request_ts) - 2440587.5) * 86400000.0 + duration_ms AS e
                FROM requests
                WHERE backend = ?
                  AND request_ts >= ?
                  AND duration_ms IS NOT NULL
            ),
            ordered AS (
                SELECT s, e,
                       MAX(e) OVER (ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS run_max
                FROM iv
            ),
            islands AS (
                SELECT s, e,
                       SUM(CASE WHEN run_max IS NULL OR s > run_max THEN 1 ELSE 0 END)
                           OVER (ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp
                FROM ordered
            )
            SELECT SUM(span) FROM (
                SELECT MAX(e) - MIN(s) AS span FROM islands GROUP BY grp
            )
            """,
            (backend, since),
        ).fetchone()
        val = row[0] if row else None
        return val / 1000.0 if val is not None else None

    def get_stats(self, period: str = 'week', backend: str | None = None) -> dict:
        """Return time-bucketed aggregates for the given period.

        *period* must be ``'day'``, ``'week'``, ``'month'``, or
        ``'quarter'``.  *backend* filters by backend name; the special value
        ``'subscription'`` includes ``anthropic``, ``codex``, and
        ``openrouter``.

        Return structure::

            {
              'period': str,
              'backend_filter': str | None,
              'buckets': [
                {
                  'label': str,
                  'rows': [{'backend': str, 'model_tier': str, ...}],
                  'subtotal': {'requests': int, ...}
                }
              ],
              'total': {'requests': int, ...}
            }

        Empty periods are omitted.  Raises ``ValueError`` for unknown periods.
        """
        _NUMERIC_FIELDS = (
            'requests', 'input_tokens', 'output_tokens',
            'cache_read_tokens', 'cache_creation_tokens',
            'cost_usd', 'cache_savings_usd', 'active_time_secs',
        )

        period_map: dict[str, tuple[str, str]] = {
            'day':     ('-1 day',   "strftime('%H:00', request_ts)"),
            'week':    ('-7 days',  "date(request_ts)"),
            'month':   ('-30 days', "strftime('%Y-W%W', request_ts)"),
            'quarter': ('-90 days', "strftime('%Y-%m', request_ts)"),
        }
        if period not in period_map:
            raise ValueError(
                f"Invalid period: {period!r}. Must be 'day', 'week', 'month', or 'quarter'."
            )
        window, bucket_expr = period_map[period]

        params: list = [window]
        if backend == 'subscription':
            backend_filter = "AND backend IN ('anthropic','codex','openrouter')"
        elif backend is not None:
            backend_filter = "AND backend = ?"
            params.append(backend)
        else:
            backend_filter = ""

        rows = self._read_conn().execute(
            f"""
            SELECT
                ({bucket_expr}) AS bucket_key,
                backend,
                model_tier,
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
                COALESCE(SUM(cost_estimate), 0) AS cost_usd,
                COALESCE(SUM(cache_savings_usd), 0) AS cache_savings_usd,
                CAST(
                    (julianday(MAX(request_ts)) - julianday(MIN(request_ts))) * 86400
                    AS INTEGER
                ) AS active_time_secs
            FROM requests
            WHERE request_ts >= datetime('now', ?)
            {backend_filter}
            GROUP BY bucket_key, backend, model_tier
            ORDER BY bucket_key, backend, model_tier
            """,
            params,
        ).fetchall()

        # Group raw rows by bucket_key
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_bucket[row['bucket_key']].append(dict(row))

        def _label(key: str) -> str:
            if period == 'day':
                return key  # already "HH:00"
            elif period == 'week':
                try:
                    d = _date.fromisoformat(key)
                    return f"{d.strftime('%a')} {key}"
                except ValueError:
                    return key
            elif period == 'month':
                return key  # already "YYYY-Www"
            else:  # quarter
                try:
                    dt = _datetime.strptime(key, '%Y-%m')
                    return dt.strftime('%b %Y')
                except ValueError:
                    return key

        def _zero_total() -> dict:
            return {f: 0 for f in _NUMERIC_FIELDS}

        def _subtotal(bucket_rows: list[dict]) -> dict:
            st = _zero_total()
            for r in bucket_rows:
                for f in _NUMERIC_FIELDS:
                    st[f] += r.get(f) or 0
            return st

        buckets = []
        grand_total = _zero_total()
        for bucket_key in sorted(by_bucket.keys()):
            bucket_rows = by_bucket[bucket_key]
            # Strip internal bucket_key from each row
            rows_out = [
                {k: v for k, v in r.items() if k != 'bucket_key'}
                for r in bucket_rows
            ]
            st = _subtotal(bucket_rows)
            for f in _NUMERIC_FIELDS:
                grand_total[f] += st[f]
            buckets.append({
                'label': _label(bucket_key),
                'rows': rows_out,
                'subtotal': st,
            })

        return {
            'period': period,
            'backend_filter': backend,
            'buckets': buckets,
            'total': grand_total,
        }

    def get_prompt(self, sha256: str) -> dict | None:
        """Return a prompt_store row by content hash, or None if not found.

        The returned dict contains: content_hash, content_type, content,
        char_count, first_seen_at.
        """
        row = self._read_conn().execute(
            """
            SELECT content_hash, content_type, content, char_count, first_seen_at
            FROM prompt_store
            WHERE content_hash = ?
            """,
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # -----------------------------------------------------------------------
    # Retention
    # -----------------------------------------------------------------------

    def _run_retention(self, days: int) -> None:
        """Delete request rows older than *days* days in chunks of 1000.

        Does NOT delete session rows — they are lightweight metadata and
        serve as an index for request history even after rows are purged.

        After deleting stale request rows, removes orphaned ``prompt_store``
        entries (rows whose content_hash is no longer referenced by any
        request's ``system_prompt_sha256`` or ``tools_sha256``).
        """
        cutoff = f'-{days} days'
        while True:
            with self._lock:
                with self._conn:
                    cur = self._conn.execute(
                        """
                        DELETE FROM requests
                        WHERE id IN (
                            SELECT id FROM requests
                            WHERE request_ts < datetime('now', ?)
                            LIMIT 1000
                        )
                        """,
                        (cutoff,),
                    )
                deleted = cur.rowcount
            if deleted < 1000:
                break

        # Prune orphaned prompt_store entries after all stale request rows
        # have been removed.
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    DELETE FROM prompt_store
                    WHERE NOT EXISTS (
                        SELECT 1 FROM requests
                        WHERE requests.system_prompt_sha256 = prompt_store.content_hash
                           OR requests.tools_sha256 = prompt_store.content_hash
                    )
                    """
                )

    def start_retention_daemon(self, days: int = 7) -> None:
        """Start a daemon thread that runs a retention pass every 24 hours.

        The first pass runs immediately on startup so the database is trimmed
        even if the server was offline for a long time.  Each subsequent pass
        sleeps 24 hours before running again.  The thread is a ``daemon`` so it
        does not block process exit.

        Use stop_retention_daemon() to cleanly shut down the thread.
        """
        if self._retention_thread is not None:
            return

        def _worker() -> None:
            while not self._retention_stop.is_set():
                try:
                    self._run_retention(days)
                except Exception as exc:
                    logger.warning('SessionDB: retention pass error: %s', exc)
                self._retention_stop.wait(24 * 3600)

        self._retention_stop.clear()
        self._retention_thread = threading.Thread(
            target=_worker, daemon=True, name='anthproxy-db-retention'
        )
        self._retention_thread.start()

    def stop_retention_daemon(self) -> None:
        """Signal the retention daemon to stop and wait for it to exit."""
        self._retention_stop.set()
        if self._retention_thread is not None:
            self._retention_thread.join(timeout=5)
            self._retention_thread = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
