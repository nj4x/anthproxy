"""Tests for anthproxy/db.py — the SQLite persistence layer.

Each test opens its own isolated database via tempfile.mkstemp() so tests are
fully independent and leave no files behind.

The ModelRoutingDecision dataclass is faked with types.SimpleNamespace; the db
module only reads attributes, never checks the type.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import threading
from types import SimpleNamespace

import pytest

from anthproxy.db import (
    SessionDB,
    _MIGRATIONS,
    _SCHEMA_VERSION,
    _apply_migration_0,
    _apply_migration_1,  # noqa: F401
    _compute_cache_savings,
    compute_cost,
    _tier_from_model,
    ensure_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> tuple[SessionDB, str]:
    """Return (db, path) using a temp file.  Caller must close() and unlink."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return SessionDB(path), path


def _make_decision(
    requested: str = 'claude-sonnet-4-6',
    routed: str = 'sonnet',
    classification: str | None = 'standard',
    reason_code: str = 'classifier_standard',
    estimated_input_tokens: int = 100,
    applied: bool = True,
) -> SimpleNamespace:
    """Return a minimal routing-decision stand-in."""
    return SimpleNamespace(
        requested_model=requested,
        routed_model=routed,
        classification=classification,
        reason_code=reason_code,
        estimated_input_tokens=estimated_input_tokens,
        applied=applied,
    )


_DEFAULT_STATS = {
    'input_tokens': 500,
    'output_tokens': 200,
    'cache_creation_tokens': 50,
    'cache_read_tokens': 30,
}


def _insert(
    db: SessionDB,
    session_id: str = 'sess-1',
    anchor: str | None = 'anc-a',
    decision: SimpleNamespace | None = None,
    stats: dict | None = None,
    duration_ms: int = 250,
    backend: str = 'anthropic',
    status: str = 'success',
    error: str | None = None,
    attempt: int = 1,
    conversation_anchor: str | None = None,
    response_text: str | None = None,
) -> int:
    if decision is None:
        decision = _make_decision()
    if stats is None:
        stats = dict(_DEFAULT_STATS)
    # conversation_anchor kwarg takes precedence over anchor positional
    if conversation_anchor is None:
        conversation_anchor = anchor
    return db.record_request(
        session_id=session_id,
        conversation_anchor=conversation_anchor,
        routing_decision=decision,
        stats_dict=stats,
        duration_ms=duration_ms,
        backend=backend,
        status=status,
        error=error,
        attempt=attempt,
        response_text=response_text,
    )


# ---------------------------------------------------------------------------
# 1. Schema creation and migration
# ---------------------------------------------------------------------------

class TestSchema:
    def test_fresh_db_schema_version(self):
        db, path = _make_db()
        try:
            version = db._conn.execute("PRAGMA user_version;").fetchone()[0]
            assert version == _SCHEMA_VERSION
        finally:
            db.close()
            os.unlink(path)

    def test_tables_exist(self):
        db, path = _make_db()
        try:
            tables = {
                row[0]
                for row in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert 'requests' in tables
            assert 'sessions' in tables
            assert 'config_changes' in tables
            assert 'prompt_store' in tables
        finally:
            db.close()
            os.unlink(path)

    def test_indexes_exist(self):
        db, path = _make_db()
        try:
            indexes = {
                row[0]
                for row in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            for expected in (
                'ix_req_session',
                'ix_req_ts',
                'ix_req_session_ts',
                'ix_req_anchor',
                'ix_req_model_tier_ts',
                'ix_req_backend_ts',
                'ix_cc_ts',
                # Added in migration 1
                'ix_req_sys_sha',
                'ix_req_tools_sha',
            ):
                assert expected in indexes, f"Missing index: {expected}"
        finally:
            db.close()
            os.unlink(path)

    def test_migration_idempotent(self):
        """Running ensure_schema on an already-migrated DB is a no-op."""
        db, path = _make_db()
        try:
            # Run it a second time — should not raise or bump the version further
            ensure_schema(db._conn)
            version = db._conn.execute("PRAGMA user_version;").fetchone()[0]
            assert version == _SCHEMA_VERSION
        finally:
            db.close()
            os.unlink(path)

    def test_wal_mode(self):
        db, path = _make_db()
        try:
            mode = db._conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode == 'wal'
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 2. record_request
# ---------------------------------------------------------------------------

class TestRecordRequest:
    def test_returns_rowid(self):
        db, path = _make_db()
        try:
            rowid = _insert(db)
            assert isinstance(rowid, int)
            assert rowid >= 1
        finally:
            db.close()
            os.unlink(path)

    def test_sequential_rowids(self):
        db, path = _make_db()
        try:
            id1 = _insert(db, session_id='s1')
            id2 = _insert(db, session_id='s1')
            assert id2 > id1
        finally:
            db.close()
            os.unlink(path)

    def test_correct_fields_stored(self):
        db, path = _make_db()
        try:
            decision = _make_decision(
                requested='claude-opus-4-8',
                routed='opus',
                classification='deep',
                reason_code='classifier_deep',
                estimated_input_tokens=9000,
                applied=True,
            )
            stats = {
                'input_tokens': 1000,
                'output_tokens': 400,
                'cache_creation_tokens': 100,
                'cache_read_tokens': 60,
            }
            rowid = db.record_request(
                session_id='sess-x',
                conversation_anchor='conv-1',
                routing_decision=decision,
                stats_dict=stats,
                duration_ms=750,
                backend='bedrock',
                status='success',
                attempt=2,
            )
            row = dict(
                db._conn.execute(
                    "SELECT * FROM requests WHERE id = ?", (rowid,)
                ).fetchone()
            )
            assert row['session_id'] == 'sess-x'
            assert row['conversation_anchor'] == 'conv-1'
            assert row['requested_model'] == 'claude-opus-4-8'
            assert row['routed_model'] == 'opus'
            assert row['classification'] == 'deep'
            assert row['reason_code'] == 'classifier_deep'
            assert row['estimated_input_tokens'] == 9000
            assert row['input_tokens'] == 1000
            assert row['output_tokens'] == 400
            assert row['cache_creation_tokens'] == 100
            assert row['cache_read_tokens'] == 60
            assert row['duration_ms'] == 750
            assert row['backend'] == 'bedrock'
            assert row['status'] == 'success'
            assert row['applied'] == 1
            assert row['model_tier'] == 'opus'
            assert row['attempt'] == 2
            assert row['error'] is None
            assert row['cost_estimate'] > 0
        finally:
            db.close()
            os.unlink(path)

    def test_null_classification_stored(self):
        db, path = _make_db()
        try:
            decision = _make_decision(classification=None, reason_code='disabled')
            rowid = _insert(db, decision=decision)
            row = dict(
                db._conn.execute("SELECT classification FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['classification'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_error_request_stored(self):
        db, path = _make_db()
        try:
            rowid = _insert(
                db,
                status='error',
                error='upstream_failure',
                stats={'input_tokens': 0, 'output_tokens': 0,
                       'cache_creation_tokens': 0, 'cache_read_tokens': 0},
            )
            row = dict(
                db._conn.execute("SELECT status, error FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['status'] == 'error'
            assert row['error'] == 'upstream_failure'
        finally:
            db.close()
            os.unlink(path)

    def test_rate_limited_status(self):
        db, path = _make_db()
        try:
            rowid = _insert(db, status='rate_limited')
            row = dict(
                db._conn.execute("SELECT status FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['status'] == 'rate_limited'
        finally:
            db.close()
            os.unlink(path)

    def test_session_row_created_on_first_request(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='new-sess')
            row = db._conn.execute(
                "SELECT session_id FROM sessions WHERE session_id='new-sess'"
            ).fetchone()
            assert row is not None
        finally:
            db.close()
            os.unlink(path)

    def test_session_last_seen_updated(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            first = dict(
                db._conn.execute("SELECT last_seen_at FROM sessions WHERE session_id='s'").fetchone()
            )['last_seen_at']
            import time
            # A very small sleep to ensure the timestamp can differ
            time.sleep(0.01)
            _insert(db, session_id='s')
            second = dict(
                db._conn.execute("SELECT last_seen_at FROM sessions WHERE session_id='s'").fetchone()
            )['last_seen_at']
            # last_seen_at should be >= first (may be equal if same second)
            assert second >= first
        finally:
            db.close()
            os.unlink(path)

    def test_null_anchor_stored(self):
        db, path = _make_db()
        try:
            rowid = _insert(db, anchor=None)
            row = dict(
                db._conn.execute("SELECT conversation_anchor FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['conversation_anchor'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_model_tier_haiku(self):
        db, path = _make_db()
        try:
            decision = _make_decision(routed='claude-haiku-4-5-20251001')
            rowid = _insert(db, decision=decision)
            row = dict(
                db._conn.execute("SELECT model_tier FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['model_tier'] == 'haiku'
        finally:
            db.close()
            os.unlink(path)

    def test_model_tier_none_for_unknown(self):
        db, path = _make_db()
        try:
            decision = _make_decision(routed='gpt-5.4')
            rowid = _insert(db, decision=decision)
            row = dict(
                db._conn.execute("SELECT model_tier FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['model_tier'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_none_routing_decision_returns_minus_one(self):
        """m1: record_request must return -1 without raising when routing_decision is None."""
        db, path = _make_db()
        try:
            result = db.record_request(
                session_id='test',
                conversation_anchor=None,
                routing_decision=None,
                stats_dict={},
                duration_ms=0,
                backend='anthropic',
                status='success',
            )
            assert result == -1
        finally:
            db.close()
            os.unlink(path)

    def test_parent_anchor_first_request_null(self):
        """First request in a session has parent_anchor = NULL."""
        db, path = _make_db()
        try:
            rowid = _insert(db, session_id='s1', conversation_anchor='conv-1')
            row = dict(
                db._conn.execute(
                    "SELECT parent_conversation_anchor FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            assert row['parent_conversation_anchor'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_parent_anchor_second_anchor_points_to_first(self):
        """Second request with different anchor: parent_anchor = earliest anchor."""
        db, path = _make_db()
        try:
            rowid1 = _insert(db, session_id='s1', conversation_anchor='conv-1')
            rowid2 = _insert(db, session_id='s1', conversation_anchor='conv-2')
            row1 = dict(
                db._conn.execute(
                    "SELECT parent_conversation_anchor FROM requests WHERE id=?", (rowid1,)
                ).fetchone()
            )
            row2 = dict(
                db._conn.execute(
                    "SELECT parent_conversation_anchor FROM requests WHERE id=?", (rowid2,)
                ).fetchone()
            )
            assert row1['parent_conversation_anchor'] is None
            assert row2['parent_conversation_anchor'] == 'conv-1'
        finally:
            db.close()
            os.unlink(path)

    def test_parent_anchor_all_same_is_null(self):
        """When all requests use the same anchor: parent_anchor = NULL for all."""
        db, path = _make_db()
        try:
            rowid1 = _insert(db, session_id='s1', conversation_anchor='conv-1')
            rowid2 = _insert(db, session_id='s1', conversation_anchor='conv-1')
            row1 = dict(
                db._conn.execute(
                    "SELECT parent_conversation_anchor FROM requests WHERE id=?", (rowid1,)
                ).fetchone()
            )
            row2 = dict(
                db._conn.execute(
                    "SELECT parent_conversation_anchor FROM requests WHERE id=?", (rowid2,)
                ).fetchone()
            )
            assert row1['parent_conversation_anchor'] is None
            assert row2['parent_conversation_anchor'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_response_text_stored_and_retrieved(self):
        """response_text is stored and retrieved correctly."""
        db, path = _make_db()
        try:
            response_txt = 'This is the LLM response text.'
            rowid = _insert(db, session_id='s1', response_text=response_txt)
            row = dict(
                db._conn.execute(
                    "SELECT response_text FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            assert row['response_text'] == response_txt
        finally:
            db.close()
            os.unlink(path)

    def test_response_text_null_when_not_provided(self):
        """response_text is NULL when not provided."""
        db, path = _make_db()
        try:
            rowid = _insert(db, session_id='s1')  # no response_text arg
            row = dict(
                db._conn.execute(
                    "SELECT response_text FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            assert row['response_text'] is None
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 3. update_request_on_retry
# ---------------------------------------------------------------------------

class TestUpdateRequestOnRetry:
    def test_updates_correct_row(self):
        db, path = _make_db()
        try:
            rowid = _insert(db, backend='anthropic', status='rate_limited')
            db.update_request_on_retry(
                request_id=rowid,
                new_backend='bedrock',
                attempt=2,
                input_tokens=800,
                output_tokens=300,
                cache_creation_tokens=20,
                cache_read_tokens=10,
                cost_estimate=0.005,
                status='success',
            )
            row = dict(
                db._conn.execute("SELECT * FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['backend'] == 'bedrock'
            assert row['attempt'] == 2
            assert row['input_tokens'] == 800
            assert row['output_tokens'] == 300
            assert row['cache_creation_tokens'] == 20
            assert row['cache_read_tokens'] == 10
            assert row['cost_estimate'] == pytest.approx(0.005)
            assert row['status'] == 'success'
            assert row['error'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_update_sets_error_field(self):
        db, path = _make_db()
        try:
            rowid = _insert(db, status='success')
            db.update_request_on_retry(
                request_id=rowid,
                new_backend='bedrock',
                attempt=2,
                input_tokens=0,
                output_tokens=0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_estimate=0.0,
                status='error',
                error='connection_reset',
            )
            row = dict(
                db._conn.execute("SELECT status, error FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['status'] == 'error'
            assert row['error'] == 'connection_reset'
        finally:
            db.close()
            os.unlink(path)

    def test_update_does_not_touch_other_rows(self):
        db, path = _make_db()
        try:
            id1 = _insert(db, session_id='s1')
            id2 = _insert(db, session_id='s2')
            db.update_request_on_retry(
                request_id=id1,
                new_backend='bedrock',
                attempt=2,
                input_tokens=1,
                output_tokens=1,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_estimate=0.0,
                status='success',
            )
            row2 = dict(
                db._conn.execute("SELECT backend FROM requests WHERE id=?", (id2,)).fetchone()
            )
            assert row2['backend'] == 'anthropic'
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4. get_sessions — pagination and aggregates
# ---------------------------------------------------------------------------

class TestGetSessions:
    def _populate(self, db: SessionDB):
        for i in range(5):
            _insert(
                db,
                session_id=f'sess-{i}',
                stats={
                    'input_tokens': (i + 1) * 100,
                    'output_tokens': 50,
                    'cache_creation_tokens': 10,
                    'cache_read_tokens': 5,
                },
            )

    def test_returns_all_sessions(self):
        db, path = _make_db()
        try:
            self._populate(db)
            sessions = db.get_sessions(limit=10)
            assert len(sessions) == 5
        finally:
            db.close()
            os.unlink(path)

    def test_pagination_limit(self):
        db, path = _make_db()
        try:
            self._populate(db)
            page1 = db.get_sessions(limit=2, offset=0)
            assert len(page1) == 2
        finally:
            db.close()
            os.unlink(path)

    def test_pagination_offset(self):
        db, path = _make_db()
        try:
            self._populate(db)
            page1 = db.get_sessions(limit=2, offset=0)
            page2 = db.get_sessions(limit=2, offset=2)
            ids1 = {r['session_id'] for r in page1}
            ids2 = {r['session_id'] for r in page2}
            assert ids1.isdisjoint(ids2)
        finally:
            db.close()
            os.unlink(path)

    def test_aggregate_fields_present(self):
        db, path = _make_db()
        try:
            self._populate(db)
            row = db.get_sessions(limit=10)[0]
            assert 'request_count' in row
            assert 'total_input_tokens' in row
            assert 'total_output_tokens' in row
            assert 'total_cache_creation' in row
            assert 'total_cache_read' in row
            assert 'estimated_cost_usd' in row
        finally:
            db.close()
            os.unlink(path)

    def test_aggregate_values_correct(self):
        db, path = _make_db()
        try:
            # Insert two requests for one session
            _insert(
                db, session_id='s',
                stats={'input_tokens': 100, 'output_tokens': 50,
                       'cache_creation_tokens': 0, 'cache_read_tokens': 0},
            )
            _insert(
                db, session_id='s',
                stats={'input_tokens': 200, 'output_tokens': 80,
                       'cache_creation_tokens': 0, 'cache_read_tokens': 0},
            )
            rows = db.get_sessions(limit=10)
            assert len(rows) == 1
            row = rows[0]
            assert row['request_count'] == 2
            assert row['total_input_tokens'] == 300
            assert row['total_output_tokens'] == 130
        finally:
            db.close()
            os.unlink(path)

    def test_empty_db_returns_empty_list(self):
        db, path = _make_db()
        try:
            assert db.get_sessions() == []
        finally:
            db.close()
            os.unlink(path)

    def test_ordered_by_last_seen_desc(self):
        db, path = _make_db()
        try:
            # Insert in order; sessions table orders by last_seen_at DESC
            import time
            _insert(db, session_id='old')
            time.sleep(0.01)
            _insert(db, session_id='new')
            sessions = db.get_sessions()
            assert sessions[0]['session_id'] == 'new'
            assert sessions[1]['session_id'] == 'old'
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4b. get_sessions_count
# ---------------------------------------------------------------------------

class TestGetSessionsCount:
    def test_empty_db_returns_zero(self):
        """M2: get_sessions_count() returns 0 on an empty database."""
        db, path = _make_db()
        try:
            assert db.get_sessions_count() == 0
        finally:
            db.close()
            os.unlink(path)

    def test_count_matches_inserted_sessions(self):
        """M2: After inserting 3 distinct sessions, count == 3."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            _insert(db, session_id='s2')
            _insert(db, session_id='s3')
            assert db.get_sessions_count() == 3
        finally:
            db.close()
            os.unlink(path)

    def test_multiple_requests_same_session_counted_once(self):
        """Two requests for the same session still count as 1 session."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            _insert(db, session_id='s1')
            assert db.get_sessions_count() == 1
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5. get_session
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_returns_none_for_missing_session(self):
        db, path = _make_db()
        try:
            assert db.get_session('no-such-session') is None
        finally:
            db.close()
            os.unlink(path)

    def test_returns_header_fields(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            result = db.get_session('s1')
            assert result is not None
            assert result['session_id'] == 's1'
            assert 'created_at' in result
            assert 'last_seen_at' in result
        finally:
            db.close()
            os.unlink(path)

    def test_returns_aggregate_fields(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1',
                    stats={'input_tokens': 300, 'output_tokens': 100,
                           'cache_creation_tokens': 20, 'cache_read_tokens': 10})
            result = db.get_session('s1')
            assert result['request_count'] == 1
            assert result['total_input_tokens'] == 300
            assert result['total_output_tokens'] == 100
            assert result['total_cache_creation'] == 20
            assert result['total_cache_read'] == 10
            assert result['estimated_cost_usd'] > 0
        finally:
            db.close()
            os.unlink(path)

    def test_model_breakdown_present(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1',
                    decision=_make_decision(routed='sonnet'))
            _insert(db, session_id='s1',
                    decision=_make_decision(routed='haiku'))
            result = db.get_session('s1')
            assert 'model_breakdown' in result
            assert len(result['model_breakdown']) == 2
            models = {r['routed_model'] for r in result['model_breakdown']}
            assert models == {'sonnet', 'haiku'}
        finally:
            db.close()
            os.unlink(path)

    def test_conversations_list_present(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', anchor='conv-a')
            _insert(db, session_id='s1', anchor='conv-b')
            result = db.get_session('s1')
            assert 'conversations' in result
            anchors = {c['conversation_anchor'] for c in result['conversations']}
            assert anchors == {'conv-a', 'conv-b'}
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5b. Conversation (sub-session) summaries
# ---------------------------------------------------------------------------

class TestConversationSummaries:
    def test_upsert_surfaces_in_get_session(self):
        """A per-conversation summary appears on the matching conversations row."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', anchor='conv-a')
            _insert(db, session_id='s1', anchor='conv-b')
            db.upsert_conversation_summary('s1', 'conv-a', 'Working on auth flow')
            result = db.get_session('s1')
            by_anchor = {c['conversation_anchor']: c for c in result['conversations']}
            assert by_anchor['conv-a']['summary'] == 'Working on auth flow'
            assert by_anchor['conv-b']['summary'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_upsert_replaces_existing(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', anchor='conv-a')
            db.upsert_conversation_summary('s1', 'conv-a', 'first')
            db.upsert_conversation_summary('s1', 'conv-a', 'second')
            result = db.get_session('s1')
            conv = next(
                c for c in result['conversations']
                if c['conversation_anchor'] == 'conv-a'
            )
            assert conv['summary'] == 'second'
        finally:
            db.close()
            os.unlink(path)

    def test_for_summary_returns_conversations_needing_work(self):
        """Conversations with prompts and no summary are returned with recent prompts."""
        db, path = _make_db()
        try:
            db.record_request(
                session_id='s1',
                conversation_anchor='conv-a',
                routing_decision=_make_decision(),
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                user_prompt_text='Implement JWT middleware',
            )
            items = db.get_conversations_for_summary()
            assert len(items) == 1
            assert items[0]['session_id'] == 's1'
            assert items[0]['conversation_anchor'] == 'conv-a'
            assert items[0]['recent_prompts'] == ['Implement JWT middleware']
        finally:
            db.close()
            os.unlink(path)

    def test_for_summary_excludes_null_anchor(self):
        db, path = _make_db()
        try:
            db.record_request(
                session_id='s1',
                conversation_anchor=None,
                routing_decision=_make_decision(),
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                user_prompt_text='no anchor here',
            )
            assert db.get_conversations_for_summary() == []
        finally:
            db.close()
            os.unlink(path)

    def test_for_summary_excludes_up_to_date(self):
        """A conversation whose summary post-dates its newest request is skipped."""
        db, path = _make_db()
        try:
            db.record_request(
                session_id='s1',
                conversation_anchor='conv-a',
                routing_decision=_make_decision(),
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                user_prompt_text='first prompt',
            )
            # Summary generated after the request → up to date.
            db.upsert_conversation_summary('s1', 'conv-a', 'done')
            assert db.get_conversations_for_summary() == []
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. get_session_metadata
# ---------------------------------------------------------------------------

class TestGetSessionMetadata:
    def test_returns_none_for_missing(self):
        db, path = _make_db()
        try:
            assert db.get_session_metadata('ghost') is None
        finally:
            db.close()
            os.unlink(path)

    def test_returns_none_pins_by_default(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            meta = db.get_session_metadata('s1')
            assert meta is not None
            assert meta['pinned_backend'] is None
            assert meta['pinned_tier'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_returns_set_pins(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            db.set_session_backend('s1', 'bedrock')
            db.set_session_tier('s1', 'opus')
            meta = db.get_session_metadata('s1')
            assert meta['pinned_backend'] == 'bedrock'
            assert meta['pinned_tier'] == 'opus'
        finally:
            db.close()
            os.unlink(path)

    def test_only_two_fields_returned(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            meta = db.get_session_metadata('s1')
            assert set(meta.keys()) == {'pinned_backend', 'pinned_tier'}
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. get_trace
# ---------------------------------------------------------------------------

class TestGetTrace:
    def test_without_anchor_returns_all_for_session(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s', anchor='a')
            _insert(db, session_id='s', anchor='b')
            _insert(db, session_id='other', anchor='x')
            rows = db.get_trace('s')
            assert len(rows) == 2
            for r in rows:
                assert r['session_id'] == 's'
        finally:
            db.close()
            os.unlink(path)

    def test_without_anchor_ordered_desc(self):
        db, path = _make_db()
        try:
            import time
            id1 = _insert(db, session_id='s', anchor='a')
            time.sleep(0.01)
            id2 = _insert(db, session_id='s', anchor='b')
            rows = db.get_trace('s')
            # Most recent first
            assert rows[0]['id'] == id2
            assert rows[1]['id'] == id1
        finally:
            db.close()
            os.unlink(path)

    def test_with_anchor_filters_rows(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s', anchor='conv-1')
            _insert(db, session_id='s', anchor='conv-1')
            _insert(db, session_id='s', anchor='conv-2')
            rows = db.get_trace('s', anchor='conv-1')
            assert len(rows) == 2
            for r in rows:
                assert r['conversation_anchor'] == 'conv-1'
        finally:
            db.close()
            os.unlink(path)

    def test_with_anchor_ordered_asc(self):
        db, path = _make_db()
        try:
            import time
            id1 = _insert(db, session_id='s', anchor='c')
            time.sleep(0.01)
            id2 = _insert(db, session_id='s', anchor='c')
            rows = db.get_trace('s', anchor='c')
            # Conversation flow: oldest first
            assert rows[0]['id'] == id1
            assert rows[1]['id'] == id2
        finally:
            db.close()
            os.unlink(path)

    def test_limit_and_offset_respected(self):
        db, path = _make_db()
        try:
            for _ in range(5):
                _insert(db, session_id='s')
            page1 = db.get_trace('s', limit=3, offset=0)
            page2 = db.get_trace('s', limit=3, offset=3)
            assert len(page1) == 3
            assert len(page2) == 2
            ids1 = {r['id'] for r in page1}
            ids2 = {r['id'] for r in page2}
            assert ids1.isdisjoint(ids2)
        finally:
            db.close()
            os.unlink(path)

    def test_unknown_session_returns_empty_list(self):
        db, path = _make_db()
        try:
            assert db.get_trace('no-such') == []
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# Filter Tests: get_sessions and get_trace with q parameter
# ---------------------------------------------------------------------------

class TestGetSessionsFiltering:
    def test_get_sessions_q_no_match_returns_empty(self):
        """Regression test: non-matching search returns zero sessions, not all."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            _insert(db, session_id='s2')
            rows = db.get_sessions(q='xyz123xyz')  # Non-existent
            assert len(rows) == 0
            assert db.get_sessions_count(q='xyz123xyz') == 0
        finally:
            db.close()
            os.unlink(path)

    def test_get_sessions_q_matches_prompt(self):
        """Search finds session by user_prompt_search."""
        db, path = _make_db()
        try:
            # Insert a request with specific prompt text
            _insert(db, session_id='s1', response_text='hello world')
            _insert(db, session_id='s2', response_text='foo bar')
            # Query matches s1's response
            rows = db.get_sessions(q='hello')
            assert len(rows) == 1
            assert rows[0]['session_id'] == 's1'
        finally:
            db.close()
            os.unlink(path)

    def test_get_sessions_q_case_insensitive(self):
        """Search is case-insensitive via casefold."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', response_text='Hello World')
            rows_lower = db.get_sessions(q='hello')
            rows_upper = db.get_sessions(q='HELLO')
            assert len(rows_lower) == 1
            assert len(rows_upper) == 1
            assert rows_lower[0]['session_id'] == 's1'
            assert rows_upper[0]['session_id'] == 's1'
        finally:
            db.close()
            os.unlink(path)

    def test_get_sessions_q_none_returns_all(self):
        """q=None returns all sessions (back-compat)."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            _insert(db, session_id='s2')
            rows = db.get_sessions(q=None)
            assert len(rows) == 2
        finally:
            db.close()
            os.unlink(path)

    def test_get_sessions_q_empty_string_returns_all(self):
        """q='' (empty) returns all sessions (falsy branch)."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1')
            _insert(db, session_id='s2')
            rows = db.get_sessions(q='')
            assert len(rows) == 2
        finally:
            db.close()
            os.unlink(path)


class TestGetTraceFiltering:
    def test_get_trace_q_no_match_returns_empty(self):
        """Search for non-existent term returns zero requests."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', response_text='hello')
            _insert(db, session_id='s1', response_text='world')
            rows = db.get_trace('s1', q='xyz123xyz')
            assert len(rows) == 0
            assert db.get_trace_count('s1', q='xyz123xyz') == 0
        finally:
            db.close()
            os.unlink(path)

    def test_get_trace_q_matches_response_text(self):
        """Search finds request by response_search."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', response_text='apple')
            _insert(db, session_id='s1', response_text='banana')
            _insert(db, session_id='s1', response_text='cherry')
            rows = db.get_trace('s1', q='banana')
            assert len(rows) == 1
            assert rows[0]['response_text'] == 'banana'
        finally:
            db.close()
            os.unlink(path)

    def test_get_trace_q_with_anchor(self):
        """Search works with anchor filter together."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', anchor='conv-1', response_text='hello')
            _insert(db, session_id='s1', anchor='conv-1', response_text='world')
            _insert(db, session_id='s1', anchor='conv-2', response_text='hello')
            rows = db.get_trace('s1', anchor='conv-1', q='hello')
            # Only the 'hello' request in conv-1
            assert len(rows) == 1
            assert rows[0]['response_text'] == 'hello'
            assert rows[0]['conversation_anchor'] == 'conv-1'
        finally:
            db.close()
            os.unlink(path)

    def test_get_trace_q_none_returns_all(self):
        """q=None returns all requests (back-compat)."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', response_text='a')
            _insert(db, session_id='s1', response_text='b')
            rows = db.get_trace('s1', q=None)
            assert len(rows) == 2
        finally:
            db.close()
            os.unlink(path)

    def test_get_trace_q_case_insensitive(self):
        """Search is case-insensitive."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s1', response_text='Hello WORLD')
            rows_lower = db.get_trace('s1', q='hello')
            rows_upper = db.get_trace('s1', q='HELLO')
            assert len(rows_lower) == 1
            assert len(rows_upper) == 1
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 8. get_cost
# ---------------------------------------------------------------------------

class TestGetCost:
    def _setup(self, db: SessionDB):
        """Insert a haiku request and two sonnet requests."""
        _insert(
            db,
            session_id='s',
            decision=_make_decision(routed='haiku'),
            backend='anthropic',
            stats={'input_tokens': 100, 'output_tokens': 50,
                   'cache_creation_tokens': 0, 'cache_read_tokens': 0},
        )
        for _ in range(2):
            _insert(
                db,
                session_id='s',
                decision=_make_decision(routed='sonnet'),
                backend='bedrock',
                stats={'input_tokens': 200, 'output_tokens': 80,
                       'cache_creation_tokens': 0, 'cache_read_tokens': 0},
            )

    def test_group_by_model(self):
        db, path = _make_db()
        try:
            self._setup(db)
            rows = db.get_cost(group_by='model', since='-365 days')
            keys = {r['key'] for r in rows}
            assert 'haiku' in keys
            assert 'sonnet' in keys
            sonnet_row = next(r for r in rows if r['key'] == 'sonnet')
            assert sonnet_row['requests'] == 2
        finally:
            db.close()
            os.unlink(path)

    def test_group_by_tier(self):
        db, path = _make_db()
        try:
            self._setup(db)
            rows = db.get_cost(group_by='tier', since='-365 days')
            keys = {r['key'] for r in rows}
            assert 'haiku' in keys
            assert 'sonnet' in keys
        finally:
            db.close()
            os.unlink(path)

    def test_group_by_backend(self):
        db, path = _make_db()
        try:
            self._setup(db)
            rows = db.get_cost(group_by='backend', since='-365 days')
            keys = {r['key'] for r in rows}
            assert 'anthropic' in keys
            assert 'bedrock' in keys
            bedrock_row = next(r for r in rows if r['key'] == 'bedrock')
            assert bedrock_row['requests'] == 2
        finally:
            db.close()
            os.unlink(path)

    def test_session_filter(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1',
                    decision=_make_decision(routed='sonnet'), backend='anthropic',
                    stats={'input_tokens': 100, 'output_tokens': 50,
                           'cache_creation_tokens': 0, 'cache_read_tokens': 0})
            _insert(db, session_id='s2',
                    decision=_make_decision(routed='opus'), backend='anthropic',
                    stats={'input_tokens': 200, 'output_tokens': 80,
                           'cache_creation_tokens': 0, 'cache_read_tokens': 0})
            rows = db.get_cost(group_by='model', since='-365 days', session_id='s1')
            keys = {r['key'] for r in rows}
            assert 'sonnet' in keys
            assert 'opus' not in keys
        finally:
            db.close()
            os.unlink(path)

    def test_result_fields(self):
        db, path = _make_db()
        try:
            _insert(db, stats={'input_tokens': 100, 'output_tokens': 50,
                               'cache_creation_tokens': 0, 'cache_read_tokens': 0})
            rows = db.get_cost(group_by='model', since='-365 days')
            assert len(rows) >= 1
            row = rows[0]
            for field in ('key', 'requests', 'input_tokens', 'output_tokens',
                          'cache_creation', 'cache_read', 'cost_usd'):
                assert field in row, f"Missing field: {field}"
        finally:
            db.close()
            os.unlink(path)

    def test_since_window_excludes_old_rows(self):
        """
        Inserting a row then querying with '0 seconds' should yield zero rows
        (the row was inserted in the past, even if just microseconds ago).
        """
        db, path = _make_db()
        try:
            _insert(db)
            # '1 second' window forward (rows must be in the future), so no match
            rows = db.get_cost(since='+1 day')
            assert len(rows) == 0
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 9. get_routing
# ---------------------------------------------------------------------------

class TestGetRouting:
    def test_reason_code_distribution(self):
        db, path = _make_db()
        try:
            _insert(db, decision=_make_decision(reason_code='classifier_trivial',
                                                classification='trivial', applied=False))
            _insert(db, decision=_make_decision(reason_code='classifier_deep',
                                                classification='deep', applied=True))
            _insert(db, decision=_make_decision(reason_code='classifier_deep',
                                                classification='deep', applied=True))
            result = db.get_routing(since='-365 days')
            dist = result['reason_code_distribution']
            # Two deep requests should appear as a group with cnt=2
            deep = next(r for r in dist if r['reason_code'] == 'classifier_deep')
            assert deep['cnt'] == 2
            trivial = next(r for r in dist if r['reason_code'] == 'classifier_trivial')
            assert trivial['cnt'] == 1
        finally:
            db.close()
            os.unlink(path)

    def test_tier_transitions(self):
        db, path = _make_db()
        try:
            _insert(db, decision=_make_decision(requested='claude-sonnet-4-6', routed='haiku'))
            _insert(db, decision=_make_decision(requested='claude-sonnet-4-6', routed='haiku'))
            result = db.get_routing(since='-365 days')
            transitions = result['tier_transitions']
            pair = next(
                r for r in transitions
                if r['requested_model'] == 'claude-sonnet-4-6' and r['routed_model'] == 'haiku'
            )
            assert pair['cnt'] == 2
        finally:
            db.close()
            os.unlink(path)

    def test_result_structure(self):
        db, path = _make_db()
        try:
            result = db.get_routing(since='-365 days')
            assert 'reason_code_distribution' in result
            assert 'tier_transitions' in result
            assert isinstance(result['reason_code_distribution'], list)
            assert isinstance(result['tier_transitions'], list)
        finally:
            db.close()
            os.unlink(path)

    def test_session_filter(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s1',
                    decision=_make_decision(reason_code='classifier_deep',
                                           classification='deep', applied=True))
            _insert(db, session_id='s2',
                    decision=_make_decision(reason_code='classifier_trivial',
                                           classification='trivial', applied=False))
            result = db.get_routing(since='-365 days', session_id='s1')
            codes = {r['reason_code'] for r in result['reason_code_distribution']}
            assert 'classifier_deep' in codes
            assert 'classifier_trivial' not in codes
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 10. set_session_backend / set_session_tier
# ---------------------------------------------------------------------------

class TestSessionPins:
    def test_set_backend_on_existing_session(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_backend('s', 'bedrock')
            meta = db.get_session_metadata('s')
            assert meta['pinned_backend'] == 'bedrock'
        finally:
            db.close()
            os.unlink(path)

    def test_set_backend_upserts_new_session(self):
        """set_session_backend should create the session row if absent."""
        db, path = _make_db()
        try:
            db.set_session_backend('brand-new', 'plugin')
            meta = db.get_session_metadata('brand-new')
            assert meta is not None
            assert meta['pinned_backend'] == 'plugin'
        finally:
            db.close()
            os.unlink(path)

    def test_clear_backend(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_backend('s', 'bedrock')
            db.set_session_backend('s', None)
            meta = db.get_session_metadata('s')
            assert meta['pinned_backend'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_set_tier_on_existing_session(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_tier('s', 'opus')
            meta = db.get_session_metadata('s')
            assert meta['pinned_tier'] == 'opus'
        finally:
            db.close()
            os.unlink(path)

    def test_set_tier_upserts_new_session(self):
        db, path = _make_db()
        try:
            db.set_session_tier('brand-new', 'haiku')
            meta = db.get_session_metadata('brand-new')
            assert meta is not None
            assert meta['pinned_tier'] == 'haiku'
        finally:
            db.close()
            os.unlink(path)

    def test_clear_tier(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_tier('s', 'sonnet')
            db.set_session_tier('s', None)
            meta = db.get_session_metadata('s')
            assert meta['pinned_tier'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_backend_and_tier_independent(self):
        """Setting one pin must not disturb the other."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_backend('s', 'bedrock')
            db.set_session_tier('s', 'opus')
            meta = db.get_session_metadata('s')
            assert meta['pinned_backend'] == 'bedrock'
            assert meta['pinned_tier'] == 'opus'
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 11. record_config_change / get_config_changes
# ---------------------------------------------------------------------------

class TestConfigChanges:
    def test_record_and_retrieve(self):
        db, path = _make_db()
        try:
            db.record_config_change(
                event_type='backend_switch',
                actor='user',
                actor_id='sess-1',
                prev_value='anthropic',
                new_value='bedrock',
            )
            rows = db.get_config_changes()
            assert len(rows) == 1
            row = rows[0]
            assert row['event_type'] == 'backend_switch'
            assert row['actor'] == 'user'
            assert row['actor_id'] == 'sess-1'
            assert row['prev_value'] == 'anthropic'
            assert row['new_value'] == 'bedrock'
            assert 'ts' in row
        finally:
            db.close()
            os.unlink(path)

    def test_null_actor_id_and_values(self):
        db, path = _make_db()
        try:
            db.record_config_change('mode_change', 'system', None, None, 'rules')
            rows = db.get_config_changes()
            assert rows[0]['actor_id'] is None
            assert rows[0]['prev_value'] is None
            assert rows[0]['new_value'] == 'rules'
        finally:
            db.close()
            os.unlink(path)

    def test_ordered_desc_by_ts(self):
        db, path = _make_db()
        try:
            import time
            db.record_config_change('e1', 'a', None, None, 'v1')
            time.sleep(0.01)
            db.record_config_change('e2', 'a', None, None, 'v2')
            rows = db.get_config_changes()
            assert rows[0]['event_type'] == 'e2'
            assert rows[1]['event_type'] == 'e1'
        finally:
            db.close()
            os.unlink(path)

    def test_limit_respected(self):
        db, path = _make_db()
        try:
            for i in range(10):
                db.record_config_change(f'e{i}', 'a', None, None, str(i))
            rows = db.get_config_changes(limit=3)
            assert len(rows) == 3
        finally:
            db.close()
            os.unlink(path)

    def test_empty_returns_empty_list(self):
        db, path = _make_db()
        try:
            assert db.get_config_changes() == []
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 12. Concurrent record_request from multiple threads
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_writes_no_corruption(self):
        """100 concurrent threads each insert one request; all must succeed."""
        db, path = _make_db()
        try:
            errors: list[Exception] = []
            rowids: list[int] = []
            lock = threading.Lock()

            def _worker(i: int) -> None:
                try:
                    rowid = _insert(db, session_id=f'sess-{i}')
                    with lock:
                        rowids.append(rowid)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(100)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert errors == [], f"Thread errors: {errors}"
            count = db._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            assert count == 100
            # All rowids must be unique
            assert len(set(rowids)) == 100
        finally:
            db.close()
            os.unlink(path)

    def test_concurrent_mixed_writes(self):
        """Mix of record_request, set_session_backend, record_config_change."""
        db, path = _make_db()
        try:
            errors: list[Exception] = []
            lock = threading.Lock()

            def _req(i: int) -> None:
                try:
                    _insert(db, session_id=f's{i % 10}')
                    db.set_session_backend(f's{i % 10}', 'bedrock' if i % 2 else None)
                    db.record_config_change('ev', 'thr', str(i), None, str(i))
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            threads = [threading.Thread(target=_req, args=(i,)) for i in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert errors == [], f"Thread errors: {errors}"
        finally:
            db.close()
            os.unlink(path)

    def test_concurrent_reads_with_writes_no_exception(self):
        """M7: 10 reader threads concurrently calling get_sessions and
        get_session_metadata while a writer thread inserts records; no
        exceptions should occur."""
        db, path = _make_db()
        try:
            # Pre-create a session for metadata reads
            _insert(db, session_id='pre-session')

            errors: list[Exception] = []
            stop_event = threading.Event()
            lock = threading.Lock()

            def _write_worker() -> None:
                """Keep inserting rows until stop_event is set."""
                i = 0
                while not stop_event.is_set():
                    try:
                        _insert(db, session_id=f'write-sess-{i % 5}')
                    except Exception as exc:
                        with lock:
                            errors.append(exc)
                    i += 1

            def _read_worker() -> None:
                """Repeatedly call read methods to stress per-thread connections."""
                for _ in range(20):
                    try:
                        db.get_sessions()
                        db.get_session_metadata('pre-session')
                    except Exception as exc:
                        with lock:
                            errors.append(exc)

            writer = threading.Thread(target=_write_worker)
            readers = [threading.Thread(target=_read_worker) for _ in range(10)]

            writer.start()
            for r in readers:
                r.start()
            for r in readers:
                r.join(timeout=30)
            stop_event.set()
            writer.join(timeout=5)

            assert errors == [], f"Thread errors: {errors}"
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# 13. Cost computation
# ---------------------------------------------------------------------------

class TestCostComputation:
    def test_sonnet_known_cost(self):
        """sonnet: (input=3.0, output=15.0, cr=0.30, cw=3.75) per million."""
        stats = {
            'input_tokens': 1_000_000,
            'output_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
        }
        cost = compute_cost('sonnet', stats)
        assert cost == pytest.approx(3.0, rel=1e-4)

    def test_opus_output_cost(self):
        """opus output_price = 25.0 per million."""
        stats = {
            'input_tokens': 0,
            'output_tokens': 1_000_000,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
        }
        cost = compute_cost('opus', stats)
        assert cost == pytest.approx(25.0, rel=1e-4)

    def test_haiku_combined_cost(self):
        """haiku: (1.0, 5.0, 0.10, 1.25) per million.  100k tokens each."""
        stats = {
            'input_tokens': 100_000,
            'output_tokens': 100_000,
            'cache_creation_tokens': 100_000,
            'cache_read_tokens': 100_000,
        }
        cost = compute_cost('haiku', stats)
        expected = (100_000 * 1.0 + 100_000 * 5.0 + 100_000 * 0.10 + 100_000 * 1.25) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_unknown_model_returns_zero(self):
        cost = compute_cost('gpt-4o', {'input_tokens': 1000})
        assert cost == 0.0

    def test_empty_model_returns_zero(self):
        cost = compute_cost('', _DEFAULT_STATS)
        assert cost == 0.0

    def test_none_values_in_stats_treated_as_zero(self):
        stats = {
            'input_tokens': None,
            'output_tokens': None,
            'cache_creation_tokens': None,
            'cache_read_tokens': None,
        }
        cost = compute_cost('sonnet', stats)
        assert cost == 0.0

    def test_cost_estimate_stored_on_record(self):
        """Verify cost is persisted in the requests table."""
        db, path = _make_db()
        try:
            decision = _make_decision(routed='sonnet')
            stats = {
                'input_tokens': 1_000_000,
                'output_tokens': 0,
                'cache_creation_tokens': 0,
                'cache_read_tokens': 0,
            }
            rowid = db.record_request(
                session_id='s', conversation_anchor=None,
                routing_decision=decision, stats_dict=stats,
                duration_ms=100, backend='anthropic', status='success',
            )
            row = dict(db._conn.execute("SELECT cost_estimate FROM requests WHERE id=?", (rowid,)).fetchone())
            assert row['cost_estimate'] == pytest.approx(3.0, rel=1e-4)
        finally:
            db.close()
            os.unlink(path)

    def test_fable_cost(self):
        """fable: (10.0, 30.0, 1.0, 12.50) per million."""
        stats = {
            'input_tokens': 1_000_000,
            'output_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
        }
        cost = compute_cost('fable', stats)
        assert cost == pytest.approx(10.0, rel=1e-4)

    def test_model_with_context_suffix(self):
        """opus[1m] should still resolve to opus pricing."""
        stats = {
            'input_tokens': 1_000_000,
            'output_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
        }
        cost = compute_cost('opus[1m]', stats)
        assert cost == pytest.approx(5.0, rel=1e-4)

    def test_compute_cost_is_public_import(self):
        """Rename fix: compute_cost must be importable as a public symbol."""
        from anthproxy.db import compute_cost as _cc  # noqa: F401
        assert callable(_cc)


# ---------------------------------------------------------------------------
# _tier_from_model unit tests
# ---------------------------------------------------------------------------

class TestTierFromModel:
    @pytest.mark.parametrize("model,expected", [
        ('haiku', 'haiku'),
        ('sonnet', 'sonnet'),
        ('opus', 'opus'),
        ('claude-haiku-4-5-20251001', 'haiku'),
        ('claude-sonnet-4-6', 'sonnet'),
        ('claude-opus-4-8', 'opus'),
        ('opus[1m]', 'opus'),
        ('opus:1m', 'opus'),
        ('CLAUDE-OPUS-4-8', 'opus'),   # uppercase
        ('gpt-5.4', None),
        ('plugin-model-1', None),
        ('', None),
        ('fable', 'fable'),            # fable is now rank 3
    ])
    def test_tier_from_model(self, model, expected):
        assert _tier_from_model(model) == expected


# ---------------------------------------------------------------------------
# Retention logic (no daemon — call _run_retention directly)
# ---------------------------------------------------------------------------

class TestRetention:
    def test_run_retention_deletes_old_rows(self):
        """Insert rows with a past timestamp and verify retention removes them."""
        db, path = _make_db()
        try:
            # Insert 3 rows with old timestamps directly
            with db._conn:
                for _ in range(3):
                    db._conn.execute(
                        """
                        INSERT INTO requests (
                            session_id, request_ts, requested_model, backend, status, attempt
                        ) VALUES ('s', datetime('now', '-100 days'), 'sonnet', 'anthropic', 'success', 1)
                        """
                    )
            # Insert 1 recent row
            _insert(db)
            count_before = db._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            assert count_before == 4
            # Retention with 90 days should remove the 3 old rows
            db._run_retention(90)
            count_after = db._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            assert count_after == 1
        finally:
            db.close()
            os.unlink(path)

    def test_run_retention_does_not_delete_sessions(self):
        """Retention must never delete session rows."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db._run_retention(0)  # 0 days: delete everything
            sess_count = db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            assert sess_count == 1
        finally:
            db.close()
            os.unlink(path)

    def test_run_retention_keeps_recent_rows(self):
        """Rows newer than the cutoff must survive."""
        db, path = _make_db()
        try:
            _insert(db)  # inserted now, should survive 90-day cutoff
            db._run_retention(90)
            count = db._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            assert count == 1
        finally:
            db.close()
            os.unlink(path)

    def test_run_retention_removes_orphaned_prompt_store_rows(self):
        """After deleting old requests, orphaned prompt_store rows must be pruned."""
        db, path = _make_db()
        try:
            sha = 'abc123'
            _insert_v2(
                db,
                session_id='s',
                system_prompt_sha256=sha,
                prompt_store_entries={sha: ('system', 'hello world')},
            )
            # Backdate so the 90-day retention window will delete the row.
            with db._conn:
                db._conn.execute(
                    "UPDATE requests SET request_ts = datetime('now', '-100 days')"
                )
            assert db._conn.execute(
                "SELECT COUNT(*) FROM prompt_store WHERE content_hash=?", (sha,)
            ).fetchone()[0] == 1

            db._run_retention(90)

            assert db._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
            assert db._conn.execute(
                "SELECT COUNT(*) FROM prompt_store WHERE content_hash=?", (sha,)
            ).fetchone()[0] == 0
        finally:
            db.close()
            os.unlink(path)

    def test_run_retention_keeps_referenced_prompt_store_rows(self):
        """Prompt_store rows still referenced by a surviving request must remain."""
        db, path = _make_db()
        try:
            sha = 'keepme'
            _insert_v2(
                db,
                session_id='s',
                system_prompt_sha256=sha,
                prompt_store_entries={sha: ('system', 'keep this')},
            )
            with db._conn:
                db._conn.execute(
                    """
                    INSERT INTO requests (
                        session_id, request_ts, requested_model, backend, status, attempt
                    ) VALUES ('s', datetime('now', '-100 days'), 'sonnet', 'anthropic', 'success', 1)
                    """
                )
            db._run_retention(90)
            assert db._conn.execute(
                "SELECT COUNT(*) FROM prompt_store WHERE content_hash=?", (sha,)
            ).fetchone()[0] == 1
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# Schema migration 1 (v1 → v2)
# ---------------------------------------------------------------------------

class TestMigration1:
    def _make_v1_db(self) -> tuple[str, sqlite3.Connection]:
        """Return (path, conn) for a raw v1 database (migration_0 applied, user_version=1)."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        with conn:
            _apply_migration_0(conn)
            conn.execute("PRAGMA user_version = 1;")
        return path, conn

    def test_fresh_db_has_prompt_store_table(self):
        """A fresh SessionDB (v2) must include the prompt_store table."""
        db, path = _make_db()
        try:
            tables = {
                r[0] for r in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert 'prompt_store' in tables
        finally:
            db.close()
            os.unlink(path)

    def test_fresh_db_has_new_requests_columns(self):
        """A fresh SessionDB must have all 10 new columns on requests."""
        db, path = _make_db()
        try:
            cols = {
                r[1] for r in db._conn.execute('PRAGMA table_info(requests)').fetchall()
            }
            for col in (
                'user_prompt_text', 'system_prompt_sha256', 'tools_sha256',
                'routing_recovered_via_walkback', 'classifier_model',
                'classifier_summary_json', 'classifier_raw_response',
                'classifier_confidence', 'classifier_format', 'cache_savings_usd',
            ):
                assert col in cols, f"Missing column: {col}"
        finally:
            db.close()
            os.unlink(path)

    def test_migration_v1_to_v2(self):
        """Opening a v1 DB via SessionDB must upgrade it fully, including the v2 artifacts."""
        path, raw_conn = self._make_v1_db()
        raw_conn.close()
        try:
            db = SessionDB(path)
            try:
                version = db._conn.execute("PRAGMA user_version;").fetchone()[0]
                assert version == _SCHEMA_VERSION
                tables = {
                    r[0] for r in db._conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                assert 'prompt_store' in tables
                cols = {
                    r[1] for r in db._conn.execute('PRAGMA table_info(requests)').fetchall()
                }
                assert 'cache_savings_usd' in cols
                assert 'system_prompt_sha256' in cols
            finally:
                db.close()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# record_request — new v2 parameters
# ---------------------------------------------------------------------------

def _insert_v2(
    db: SessionDB,
    session_id: str = 'sess-1',
    anchor: str | None = 'anc-a',
    decision: SimpleNamespace | None = None,
    stats: dict | None = None,
    duration_ms: int = 250,
    backend: str = 'anthropic',
    status: str = 'success',
    system_prompt_sha256: str | None = None,
    tools_sha256: str | None = None,
    routing_recovered_via_walkback: bool | None = None,
    classifier_model: str | None = None,
    classifier_summary_json: str | None = None,
    classifier_raw_response: str | None = None,
    classifier_confidence: float | None = None,
    classifier_format: str | None = None,
    user_prompt_text: str | None = None,
    prompt_store_entries: dict | None = None,
) -> int:
    if decision is None:
        decision = _make_decision()
    if stats is None:
        stats = dict(_DEFAULT_STATS)
    return db.record_request(
        session_id=session_id,
        conversation_anchor=anchor,
        routing_decision=decision,
        stats_dict=stats,
        duration_ms=duration_ms,
        backend=backend,
        status=status,
        user_prompt_text=user_prompt_text,
        system_prompt_sha256=system_prompt_sha256,
        tools_sha256=tools_sha256,
        routing_recovered_via_walkback=routing_recovered_via_walkback,
        classifier_model=classifier_model,
        classifier_summary_json=classifier_summary_json,
        classifier_raw_response=classifier_raw_response,
        classifier_confidence=classifier_confidence,
        classifier_format=classifier_format,
        prompt_store_entries=prompt_store_entries,
    )


class TestRecordRequestV2:
    def test_new_fields_stored_when_provided(self):
        """All new v2 columns must be stored in the requests table."""
        db, path = _make_db()
        try:
            sha = 'deadbeef'
            rowid = _insert_v2(
                db,
                system_prompt_sha256=sha,
                tools_sha256='cafebabe',
                routing_recovered_via_walkback=True,
                classifier_model='claude-haiku-4-5',
                classifier_summary_json='{"label":"standard"}',
                classifier_raw_response='standard',
                classifier_confidence=0.95,
                classifier_format='plain',
                user_prompt_text='Hello world',
            )
            row = dict(
                db._conn.execute("SELECT * FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            assert row['user_prompt_text'] == 'Hello world'
            assert row['system_prompt_sha256'] == sha
            assert row['tools_sha256'] == 'cafebabe'
            assert row['routing_recovered_via_walkback'] == 1
            assert row['classifier_model'] == 'claude-haiku-4-5'
            assert row['classifier_summary_json'] == '{"label":"standard"}'
            assert row['classifier_raw_response'] == 'standard'
            assert row['classifier_confidence'] == pytest.approx(0.95)
            assert row['classifier_format'] == 'plain'
        finally:
            db.close()
            os.unlink(path)

    def test_new_fields_null_when_not_provided(self):
        """New v2 columns default to NULL when not supplied."""
        db, path = _make_db()
        try:
            rowid = _insert(db)
            row = dict(
                db._conn.execute("SELECT * FROM requests WHERE id=?", (rowid,)).fetchone()
            )
            for col in (
                'user_prompt_text', 'system_prompt_sha256', 'tools_sha256',
                'routing_recovered_via_walkback', 'classifier_model',
                'classifier_summary_json', 'classifier_raw_response',
                'classifier_confidence', 'classifier_format',
            ):
                assert row[col] is None, f"Expected {col!r} to be NULL"
        finally:
            db.close()
            os.unlink(path)

    def test_walkback_false_stored_as_zero(self):
        """routing_recovered_via_walkback=False → 0 in the DB."""
        db, path = _make_db()
        try:
            rowid = _insert_v2(db, routing_recovered_via_walkback=False)
            row = dict(
                db._conn.execute(
                    "SELECT routing_recovered_via_walkback FROM requests WHERE id=?",
                    (rowid,),
                ).fetchone()
            )
            assert row['routing_recovered_via_walkback'] == 0
        finally:
            db.close()
            os.unlink(path)

    def test_walkback_none_stored_as_null(self):
        """routing_recovered_via_walkback=None → NULL in the DB."""
        db, path = _make_db()
        try:
            rowid = _insert_v2(db, routing_recovered_via_walkback=None)
            row = dict(
                db._conn.execute(
                    "SELECT routing_recovered_via_walkback FROM requests WHERE id=?",
                    (rowid,),
                ).fetchone()
            )
            assert row['routing_recovered_via_walkback'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_prompt_store_entry_inserted(self):
        """Providing prompt_store_entries must upsert rows into prompt_store."""
        db, path = _make_db()
        try:
            sha = 'aabbcc'
            content = 'You are a helpful assistant.'
            _insert_v2(
                db,
                system_prompt_sha256=sha,
                prompt_store_entries={sha: ('system', content)},
            )
            row = dict(
                db._conn.execute(
                    "SELECT * FROM prompt_store WHERE content_hash=?", (sha,)
                ).fetchone()
            )
            assert row['content_hash'] == sha
            assert row['content_type'] == 'system'
            assert row['content'] == content
            assert row['char_count'] == len(content)
            assert row['first_seen_at'] is not None
        finally:
            db.close()
            os.unlink(path)

    def test_prompt_store_deduplication(self):
        """Inserting the same sha256 twice must result in exactly one prompt_store row."""
        db, path = _make_db()
        try:
            sha = 'dedup01'
            content = 'System prompt text.'
            entries = {sha: ('system', content)}
            _insert_v2(db, system_prompt_sha256=sha, prompt_store_entries=entries)
            _insert_v2(db, system_prompt_sha256=sha, prompt_store_entries=entries)
            count = db._conn.execute(
                "SELECT COUNT(*) FROM prompt_store WHERE content_hash=?", (sha,)
            ).fetchone()[0]
            assert count == 1
        finally:
            db.close()
            os.unlink(path)

    def test_multiple_prompt_store_entries(self):
        """A single record_request call can upsert multiple prompt_store entries."""
        db, path = _make_db()
        try:
            sys_sha = 'sys001'
            tools_sha = 'tools001'
            _insert_v2(
                db,
                system_prompt_sha256=sys_sha,
                tools_sha256=tools_sha,
                prompt_store_entries={
                    sys_sha: ('system', 'System text'),
                    tools_sha: ('tools', '[{"name":"bash"}]'),
                },
            )
            count = db._conn.execute("SELECT COUNT(*) FROM prompt_store").fetchone()[0]
            assert count == 2
        finally:
            db.close()
            os.unlink(path)

    def test_cache_savings_usd_computed_for_haiku(self):
        """cache_savings_usd is computed from haiku pricing: (1.0 - 0.10) per MTok."""
        db, path = _make_db()
        try:
            cache_read = 1_000_000
            decision = _make_decision(routed='haiku')
            stats = {
                'input_tokens': 100,
                'output_tokens': 50,
                'cache_creation_tokens': 0,
                'cache_read_tokens': cache_read,
            }
            rowid = db.record_request(
                session_id='s',
                conversation_anchor=None,
                routing_decision=decision,
                stats_dict=stats,
                duration_ms=100,
                backend='anthropic',
                status='success',
            )
            row = dict(
                db._conn.execute(
                    "SELECT cache_savings_usd FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            # haiku: input=1.0, cache_read=0.10 → savings = (1.0-0.10)*1M/1M = 0.90
            assert row['cache_savings_usd'] == pytest.approx(0.90, rel=1e-4)
        finally:
            db.close()
            os.unlink(path)

    def test_cache_savings_usd_none_when_no_cache_read(self):
        """cache_savings_usd is NULL when cache_read_tokens is zero."""
        db, path = _make_db()
        try:
            stats = {
                'input_tokens': 100,
                'output_tokens': 50,
                'cache_creation_tokens': 0,
                'cache_read_tokens': 0,
            }
            rowid = db.record_request(
                session_id='s',
                conversation_anchor=None,
                routing_decision=_make_decision(routed='sonnet'),
                stats_dict=stats,
                duration_ms=100,
                backend='anthropic',
                status='success',
            )
            row = dict(
                db._conn.execute(
                    "SELECT cache_savings_usd FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            assert row['cache_savings_usd'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_cache_savings_usd_none_for_unknown_model(self):
        """cache_savings_usd is NULL when the routed model is not a known tier."""
        db, path = _make_db()
        try:
            stats = {
                'input_tokens': 100,
                'output_tokens': 50,
                'cache_creation_tokens': 0,
                'cache_read_tokens': 500_000,
            }
            rowid = db.record_request(
                session_id='s',
                conversation_anchor=None,
                routing_decision=_make_decision(routed='gpt-4o'),
                stats_dict=stats,
                duration_ms=100,
                backend='openai',
                status='success',
            )
            row = dict(
                db._conn.execute(
                    "SELECT cache_savings_usd FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            assert row['cache_savings_usd'] is None
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# update_request_on_retry — cache_savings_usd recomputation
# ---------------------------------------------------------------------------

class TestUpdateRequestOnRetryV2:
    def test_updates_cache_savings_usd(self):
        """After retry, cache_savings_usd reflects the new cache_read_tokens."""
        db, path = _make_db()
        try:
            decision = _make_decision(routed='sonnet')
            rowid = _insert(db, decision=decision, status='rate_limited')
            # sonnet: input=3.0, cache_read=0.30
            db.update_request_on_retry(
                request_id=rowid,
                new_backend='bedrock',
                attempt=2,
                input_tokens=100,
                output_tokens=50,
                cache_creation_tokens=0,
                cache_read_tokens=1_000_000,
                cost_estimate=0.01,
                status='success',
            )
            row = dict(
                db._conn.execute(
                    "SELECT cache_savings_usd FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            # sonnet savings = (3.0 - 0.30) * 1_000_000 / 1_000_000 = 2.70
            assert row['cache_savings_usd'] == pytest.approx(2.70, rel=1e-4)
        finally:
            db.close()
            os.unlink(path)

    def test_updates_cache_savings_usd_to_none_when_no_cache_read(self):
        """Retry with cache_read_tokens=0 must set cache_savings_usd to NULL."""
        db, path = _make_db()
        try:
            rowid = _insert(db, status='rate_limited')
            db.update_request_on_retry(
                request_id=rowid,
                new_backend='bedrock',
                attempt=2,
                input_tokens=100,
                output_tokens=50,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_estimate=0.001,
                status='success',
            )
            row = dict(
                db._conn.execute(
                    "SELECT cache_savings_usd FROM requests WHERE id=?", (rowid,)
                ).fetchone()
            )
            assert row['cache_savings_usd'] is None
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_session_overrides
# ---------------------------------------------------------------------------

class TestGetSessionOverrides:
    def test_returns_only_pinned_sessions(self):
        """Sessions without pins must not appear in the result."""
        db, path = _make_db()
        try:
            _insert(db, session_id='pinned')
            _insert(db, session_id='unpinned')
            db.set_session_backend('pinned', 'bedrock')
            result = db.get_session_overrides()
            ids = {r['session_id'] for r in result}
            assert 'pinned' in ids
            assert 'unpinned' not in ids
        finally:
            db.close()
            os.unlink(path)

    def test_returns_expected_fields(self):
        """Each row must include session_id, display_name, pinned_backend, pinned_tier."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_backend('s', 'plugin')
            result = db.get_session_overrides()
            assert len(result) == 1
            row = result[0]
            assert row['session_id'] == 's'
            assert row['pinned_backend'] == 'plugin'
            assert row['pinned_tier'] is None
            assert 'display_name' in row
        finally:
            db.close()
            os.unlink(path)

    def test_empty_when_no_pins(self):
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            assert db.get_session_overrides() == []
        finally:
            db.close()
            os.unlink(path)

    def test_tier_pin_also_included(self):
        """A session with only a tier pin must appear."""
        db, path = _make_db()
        try:
            _insert(db, session_id='s')
            db.set_session_tier('s', 'opus')
            result = db.get_session_overrides()
            assert len(result) == 1
            assert result[0]['pinned_tier'] == 'opus'
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_request
# ---------------------------------------------------------------------------

class TestGetRequest:
    def test_returns_none_for_missing_id(self):
        db, path = _make_db()
        try:
            assert db.get_request(9999) is None
        finally:
            db.close()
            os.unlink(path)

    def test_returns_basic_row_fields(self):
        db, path = _make_db()
        try:
            rowid = _insert(db, session_id='s', backend='anthropic')
            result = db.get_request(rowid)
            assert result is not None
            assert result['id'] == rowid
            assert result['session_id'] == 's'
            assert result['backend'] == 'anthropic'
        finally:
            db.close()
            os.unlink(path)

    def test_joins_system_prompt_content(self):
        """get_request must join prompt_store for system_prompt_sha256."""
        db, path = _make_db()
        try:
            sha = 'sys_join_test'
            content = 'You are a coding assistant.'
            rowid = _insert_v2(
                db,
                system_prompt_sha256=sha,
                prompt_store_entries={sha: ('system', content)},
            )
            result = db.get_request(rowid)
            assert result is not None
            assert result['system_prompt_content'] == content
            assert result['system_prompt_char_count'] == len(content)
        finally:
            db.close()
            os.unlink(path)

    def test_joins_tools_content(self):
        """get_request must join prompt_store for tools_sha256."""
        db, path = _make_db()
        try:
            sha = 'tools_join_test'
            tools = '[{"name":"bash","description":"Run bash"}]'
            rowid = _insert_v2(
                db,
                tools_sha256=sha,
                prompt_store_entries={sha: ('tools', tools)},
            )
            result = db.get_request(rowid)
            assert result is not None
            assert result['tools_content'] == tools
            assert result['tools_char_count'] == len(tools)
        finally:
            db.close()
            os.unlink(path)

    def test_null_join_when_no_prompt_store_entry(self):
        """When no prompt_store entries exist, joined columns must be NULL."""
        db, path = _make_db()
        try:
            rowid = _insert(db)
            result = db.get_request(rowid)
            assert result is not None
            assert result['system_prompt_content'] is None
            assert result['tools_content'] is None
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def _populate(self, db: SessionDB, backend: str = 'anthropic') -> None:
        """Insert 3 requests with known token counts."""
        for _ in range(3):
            _insert(
                db,
                decision=_make_decision(routed='sonnet'),
                backend=backend,
                stats={
                    'input_tokens': 100,
                    'output_tokens': 50,
                    'cache_creation_tokens': 10,
                    'cache_read_tokens': 0,
                },
            )

    def test_raises_on_invalid_period(self):
        db, path = _make_db()
        try:
            with pytest.raises(ValueError, match="Invalid period"):
                db.get_stats(period='invalid')
        finally:
            db.close()
            os.unlink(path)

    def test_returns_required_keys(self):
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('week')
            assert 'period' in result
            assert 'backend_filter' in result
            assert 'buckets' in result
            assert 'total' in result
            assert result['period'] == 'week'
            assert result['backend_filter'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_bucket_structure(self):
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('week')
            assert len(result['buckets']) >= 1
            bucket = result['buckets'][0]
            assert 'label' in bucket
            assert 'rows' in bucket
            assert 'subtotal' in bucket
        finally:
            db.close()
            os.unlink(path)

    def test_total_aggregates_correctly(self):
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('week')
            assert result['total']['requests'] == 3
            assert result['total']['input_tokens'] == 300
            assert result['total']['output_tokens'] == 150
        finally:
            db.close()
            os.unlink(path)

    def test_backend_filter_subscription(self):
        """'subscription' backend filter must include anthropic rows."""
        db, path = _make_db()
        try:
            self._populate(db, backend='anthropic')
            _insert(db, backend='bedrock', decision=_make_decision(routed='haiku'),
                    stats={'input_tokens': 999, 'output_tokens': 1,
                           'cache_creation_tokens': 0, 'cache_read_tokens': 0})
            result = db.get_stats('week', backend='subscription')
            assert result['total']['requests'] == 3
        finally:
            db.close()
            os.unlink(path)

    def test_backend_filter_named(self):
        """Named backend filter returns only that backend's rows."""
        db, path = _make_db()
        try:
            self._populate(db, backend='anthropic')
            _insert(db, backend='bedrock', decision=_make_decision(routed='haiku'),
                    stats={'input_tokens': 999, 'output_tokens': 1,
                           'cache_creation_tokens': 0, 'cache_read_tokens': 0})
            result = db.get_stats('week', backend='bedrock')
            assert result['total']['requests'] == 1
        finally:
            db.close()
            os.unlink(path)

    def test_day_period_label_format(self):
        """Day buckets must be labelled 'HH:00'."""
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('day')
            for bucket in result['buckets']:
                assert len(bucket['label']) == 5
                assert bucket['label'][2] == ':'
        finally:
            db.close()
            os.unlink(path)

    def test_week_period_label_format(self):
        """Week buckets must be labelled 'Ddd YYYY-MM-DD'."""
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('week')
            for bucket in result['buckets']:
                parts = bucket['label'].split(' ')
                assert len(parts) == 2
                assert len(parts[0]) == 3   # "Mon", "Tue", ...
                assert len(parts[1]) == 10  # "YYYY-MM-DD"
        finally:
            db.close()
            os.unlink(path)

    def test_month_period(self):
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('month')
            assert 'buckets' in result
        finally:
            db.close()
            os.unlink(path)

    def test_quarter_period_label_format(self):
        """Quarter buckets must be labelled 'MMM YYYY'."""
        db, path = _make_db()
        try:
            self._populate(db)
            result = db.get_stats('quarter')
            for bucket in result['buckets']:
                parts = bucket['label'].split(' ')
                assert len(parts) == 2
                assert len(parts[0]) == 3   # "Jan", "Feb", ...
                assert len(parts[1]) == 4   # "2026"
        finally:
            db.close()
            os.unlink(path)

    def test_empty_db_returns_empty_buckets(self):
        db, path = _make_db()
        try:
            result = db.get_stats('week')
            assert result['buckets'] == []
            assert result['total']['requests'] == 0
        finally:
            db.close()
            os.unlink(path)

    def test_cache_savings_usd_in_total(self):
        """cache_savings_usd must be summed in the total."""
        db, path = _make_db()
        try:
            for _ in range(2):
                db.record_request(
                    session_id='s',
                    conversation_anchor=None,
                    routing_decision=_make_decision(routed='haiku'),
                    stats_dict={
                        'input_tokens': 100,
                        'output_tokens': 50,
                        'cache_creation_tokens': 0,
                        'cache_read_tokens': 1_000_000,
                    },
                    duration_ms=100,
                    backend='anthropic',
                    status='success',
                )
            result = db.get_stats('week')
            # Each request saves (1.0-0.10)*1M/1M = 0.90; 2 requests → 1.80
            assert result['total']['cache_savings_usd'] == pytest.approx(1.80, rel=1e-4)
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_prompt
# ---------------------------------------------------------------------------

class TestGetPrompt:
    def test_returns_none_on_miss(self):
        db, path = _make_db()
        try:
            assert db.get_prompt('nonexistent_sha') is None
        finally:
            db.close()
            os.unlink(path)

    def test_returns_row_on_hit(self):
        db, path = _make_db()
        try:
            sha = 'prompt01'
            content = 'System: you are a helpful bot.'
            _insert_v2(
                db,
                system_prompt_sha256=sha,
                prompt_store_entries={sha: ('system', content)},
            )
            result = db.get_prompt(sha)
            assert result is not None
            assert result['content_hash'] == sha
            assert result['content_type'] == 'system'
            assert result['content'] == content
            assert result['char_count'] == len(content)
            assert result['first_seen_at'] is not None
        finally:
            db.close()
            os.unlink(path)

    def test_returns_tools_type(self):
        db, path = _make_db()
        try:
            sha = 'tools_sha_001'
            tools_content = '[{"name":"bash"}]'
            _insert_v2(
                db,
                tools_sha256=sha,
                prompt_store_entries={sha: ('tools', tools_content)},
            )
            result = db.get_prompt(sha)
            assert result is not None
            assert result['content_type'] == 'tools'
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# _compute_cache_savings unit tests
# ---------------------------------------------------------------------------

class TestComputeCacheSavings:
    def test_haiku_savings(self):
        """haiku: (1.0 - 0.10) * tokens / 1M."""
        savings = _compute_cache_savings('haiku', 1_000_000)
        assert savings == pytest.approx(0.90, rel=1e-4)

    def test_sonnet_savings(self):
        """sonnet: (3.0 - 0.30) * tokens / 1M."""
        savings = _compute_cache_savings('sonnet', 1_000_000)
        assert savings == pytest.approx(2.70, rel=1e-4)

    def test_opus_savings(self):
        """opus: (5.0 - 0.50) * tokens / 1M."""
        savings = _compute_cache_savings('opus', 1_000_000)
        assert savings == pytest.approx(4.50, rel=1e-4)

    def test_zero_tokens_returns_none(self):
        assert _compute_cache_savings('haiku', 0) is None

    def test_none_tokens_returns_none(self):
        assert _compute_cache_savings('haiku', None) is None

    def test_unknown_model_returns_none(self):
        assert _compute_cache_savings('gpt-4o', 1_000) is None

    def test_empty_model_returns_none(self):
        assert _compute_cache_savings('', 1_000) is None

    def test_none_model_returns_none(self):
        assert _compute_cache_savings(None, 1_000) is None

    def test_fractional_tokens(self):
        """Small token counts still produce a float savings."""
        savings = _compute_cache_savings('haiku', 100)
        assert savings == pytest.approx(100 * 0.90 / 1_000_000, rel=1e-4)


class TestBusySecsWindow:
    """Wall-clock busy-time (interval union) over a usage window."""

    @staticmethod
    def _insert_at(db, request_ts: str, duration_ms: int, backend: str = 'anthropic'):
        rowid = _insert(db, duration_ms=duration_ms, backend=backend)
        db._conn.execute(
            "UPDATE requests SET request_ts = ? WHERE id = ?",
            (request_ts, rowid),
        )
        db._conn.commit()

    def test_no_requests_returns_none(self):
        db, path = _make_db()
        try:
            assert db.busy_secs_window('anthropic', '2026-01-01T00:00:00.000Z') is None
        finally:
            db.close()
            os.unlink(path)

    def test_single_request_returns_duration(self):
        db, path = _make_db()
        try:
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 10_000)
            busy = db.busy_secs_window('anthropic', '2026-01-01T00:00:00.000Z')
            assert busy == pytest.approx(10.0, abs=0.01)
        finally:
            db.close()
            os.unlink(path)

    def test_overlapping_requests_deduplicated(self):
        """Two overlapping 10s requests (5s apart) cover 15s wall-clock, not 20s."""
        db, path = _make_db()
        try:
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 10_000)
            self._insert_at(db, '2026-01-01T00:00:05.000Z', 10_000)
            busy = db.busy_secs_window('anthropic', '2026-01-01T00:00:00.000Z')
            assert busy == pytest.approx(15.0, abs=0.02)
        finally:
            db.close()
            os.unlink(path)

    def test_disjoint_requests_summed(self):
        """Non-overlapping intervals add up; overlaps within a cluster do not."""
        db, path = _make_db()
        try:
            # Cluster: [0,+10s] and [+5s,+10s] -> 15s union
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 10_000)
            self._insert_at(db, '2026-01-01T00:00:05.000Z', 10_000)
            # Disjoint: [+60s,+5s] -> +5s
            self._insert_at(db, '2026-01-01T00:01:00.000Z', 5_000)
            busy = db.busy_secs_window('anthropic', '2026-01-01T00:00:00.000Z')
            assert busy == pytest.approx(20.0, abs=0.02)
        finally:
            db.close()
            os.unlink(path)

    def test_fully_nested_request(self):
        """An interval fully contained in another does not extend the union."""
        db, path = _make_db()
        try:
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 30_000)   # [0, 30]
            self._insert_at(db, '2026-01-01T00:00:10.000Z', 5_000)    # [10, 15] nested
            busy = db.busy_secs_window('anthropic', '2026-01-01T00:00:00.000Z')
            assert busy == pytest.approx(30.0, abs=0.02)
        finally:
            db.close()
            os.unlink(path)

    def test_since_boundary_excludes_earlier(self):
        db, path = _make_db()
        try:
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 10_000)   # before window
            self._insert_at(db, '2026-01-01T01:00:00.000Z', 10_000)   # in window
            busy = db.busy_secs_window('anthropic', '2026-01-01T00:30:00.000Z')
            assert busy == pytest.approx(10.0, abs=0.02)
        finally:
            db.close()
            os.unlink(path)

    def test_backend_filter(self):
        db, path = _make_db()
        try:
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 10_000, backend='anthropic')
            self._insert_at(db, '2026-01-01T00:00:00.000Z', 99_000, backend='codex')
            busy = db.busy_secs_window('anthropic', '2026-01-01T00:00:00.000Z')
            assert busy == pytest.approx(10.0, abs=0.02)
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# ADR 0011: Schema migration and record_request for weighted-blend columns
# ---------------------------------------------------------------------------

class TestWeightedBlendMigration:
    """_apply_migration_7 adds 5 new columns; record_request stores them."""

    def test_migration_7_present(self):
        assert 7 in _MIGRATIONS

    def test_migration_7_adds_columns(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            # Apply migrations 0–6 so we land at v7, then apply migration 7.
            from anthproxy.db import _apply_migration_7
            for i in range(7):
                from anthproxy.db import _MIGRATIONS
                with conn:
                    _MIGRATIONS[i](conn)
                    conn.execute(f"PRAGMA user_version = {i + 1};")
            with conn:
                _apply_migration_7(conn)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
            assert 'system_prompt_tier' in cols
            assert 'system_prompt_score' in cols
            assert 'user_prompt_score' in cols
            assert 'routing_weighted_score' in cols
            assert 'system_prompt_classification_failed' in cols
            conn.close()
        finally:
            os.unlink(path)

    def test_record_request_stores_blend_fields(self):
        db, path = _make_db()
        try:
            decision = _make_decision()
            row_id = db.record_request(
                session_id='sess-blend',
                conversation_anchor=None,
                routing_decision=decision,
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                system_prompt_tier='trivial',
                system_prompt_score=0.0,
                user_prompt_score=2.0,
                routing_weighted_score=1.40,
                system_prompt_classification_failed=False,
            )
            row = db.get_request(row_id)
            assert row is not None
            assert row['system_prompt_tier'] == 'trivial'
            assert row['system_prompt_score'] == pytest.approx(0.0)
            assert row['user_prompt_score'] == pytest.approx(2.0)
            assert row['routing_weighted_score'] == pytest.approx(1.40)
            assert row['system_prompt_classification_failed'] == 0
        finally:
            db.close()
            os.unlink(path)

    def test_record_request_blend_fields_default_null(self):
        db, path = _make_db()
        try:
            row_id = _insert(db)
            row = db.get_request(row_id)
            assert row is not None
            assert row['system_prompt_tier'] is None
            assert row['system_prompt_score'] is None
            assert row['user_prompt_score'] is None
            assert row['routing_weighted_score'] is None
            assert row['system_prompt_classification_failed'] == 0
        finally:
            db.close()
            os.unlink(path)

    def test_record_request_classification_failed_stored_as_int(self):
        db, path = _make_db()
        try:
            decision = _make_decision()
            row_id = db.record_request(
                session_id='sess-blend-fail',
                conversation_anchor=None,
                routing_decision=decision,
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                system_prompt_tier='standard',
                system_prompt_score=1.0,
                user_prompt_score=1.0,
                routing_weighted_score=1.0,
                system_prompt_classification_failed=True,
            )
            row = db.get_request(row_id)
            assert row is not None
            assert row['system_prompt_classification_failed'] == 1
        finally:
            db.close()
            os.unlink(path)

class TestNumericScoreMigration:
    """Migration 8: add user_prompt_tier column and rescale old 0-2 fractional scores."""

    def _apply_through_migration_7(self):
        """Return an open in-memory connection with migrations 0-7 applied."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        from anthproxy.db import _MIGRATIONS
        for i in range(8):
            with conn:
                _MIGRATIONS[i](conn)
                conn.execute(f"PRAGMA user_version = {i + 1};")
        return conn

    def test_migration_8_adds_user_prompt_tier_column(self):
        conn = self._apply_through_migration_7()
        from anthproxy.db import _apply_migration_8
        with conn:
            _apply_migration_8(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        assert 'user_prompt_tier' in cols
        conn.close()

    def test_migration_8_rescales_fractional_old_scale_rows(self):
        conn = self._apply_through_migration_7()
        # Insert a row with old-scale fractional scores.
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, request_ts, requested_model, routed_model,
                    backend, status, applied, system_prompt_score,
                    user_prompt_score, routing_weighted_score
                ) VALUES (
                    'sess1', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    'sonnet', 'haiku', 'anthropic', 'success', 1,
                    0.75, 1.5, 1.05
                )"""
            )
        from anthproxy.db import _apply_migration_8
        with conn:
            _apply_migration_8(conn)
        row = conn.execute("SELECT system_prompt_score, user_prompt_score, routing_weighted_score FROM requests WHERE session_id='sess1'").fetchone()
        assert row['system_prompt_score'] == pytest.approx(38.0)   # round(0.75*50)
        assert row['user_prompt_score'] == pytest.approx(75.0)     # round(1.5*50)
        assert row['routing_weighted_score'] == pytest.approx(53.0) # round(1.05*50)
        conn.close()

    def test_migration_8_does_not_rescale_integer_rows(self):
        """Integer-valued rows (new 0-100 scale) must not be double-rescaled."""
        conn = self._apply_through_migration_7()
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, request_ts, requested_model, routed_model,
                    backend, status, applied, system_prompt_score,
                    user_prompt_score, routing_weighted_score
                ) VALUES (
                    'sess2', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    'sonnet', 'haiku', 'anthropic', 'success', 1,
                    50.0, 75.0, 60.0
                )"""
            )
        from anthproxy.db import _apply_migration_8
        with conn:
            _apply_migration_8(conn)
        row = conn.execute("SELECT system_prompt_score, user_prompt_score, routing_weighted_score FROM requests WHERE session_id='sess2'").fetchone()
        assert row['system_prompt_score'] == pytest.approx(50.0)
        assert row['user_prompt_score'] == pytest.approx(75.0)
        assert row['routing_weighted_score'] == pytest.approx(60.0)
        conn.close()

    def test_migration_8_rescales_null_system_score_with_fractional_user_score(self):
        """Rows with NULL system_prompt_score but fractional user_prompt_score must be rescaled."""
        conn = self._apply_through_migration_7()
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, request_ts, requested_model, routed_model,
                    backend, status, applied, system_prompt_score,
                    user_prompt_score, routing_weighted_score
                ) VALUES (
                    'sess_null_sys', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    'sonnet', 'haiku', 'anthropic', 'success', 1,
                    NULL, 1.5, 1.05
                )"""
            )
        from anthproxy.db import _apply_migration_8
        with conn:
            _apply_migration_8(conn)
        row = conn.execute("SELECT system_prompt_score, user_prompt_score, routing_weighted_score FROM requests WHERE session_id='sess_null_sys'").fetchone()
        assert row['system_prompt_score'] is None  # Still NULL
        assert row['user_prompt_score'] == pytest.approx(75.0)  # round(1.5*50)
        assert row['routing_weighted_score'] == pytest.approx(53.0)  # round(1.05*50)
        conn.close()

    def test_migration_8_preserves_exact_integer_boundary_values(self):
        """Rows with exactly 0.0, 1.0, or 2.0 (old-scale boundary) are not rescaled."""
        conn = self._apply_through_migration_7()
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, request_ts, requested_model, routed_model,
                    backend, status, applied, system_prompt_score,
                    user_prompt_score, routing_weighted_score
                ) VALUES (
                    'sess_trivial', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    'sonnet', 'haiku', 'anthropic', 'success', 1,
                    0.0, 0.0, 0.0
                )"""
            )
            conn.execute(
                """INSERT INTO requests (
                    session_id, request_ts, requested_model, routed_model,
                    backend, status, applied, system_prompt_score,
                    user_prompt_score, routing_weighted_score
                ) VALUES (
                    'sess_standard', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    'sonnet', 'haiku', 'anthropic', 'success', 1,
                    1.0, 1.0, 1.0
                )"""
            )
            conn.execute(
                """INSERT INTO requests (
                    session_id, request_ts, requested_model, routed_model,
                    backend, status, applied, system_prompt_score,
                    user_prompt_score, routing_weighted_score
                ) VALUES (
                    'sess_deep', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    'sonnet', 'haiku', 'anthropic', 'success', 1,
                    2.0, 2.0, 2.0
                )"""
            )
        from anthproxy.db import _apply_migration_8
        with conn:
            _apply_migration_8(conn)
        trivial_row = conn.execute("SELECT system_prompt_score FROM requests WHERE session_id='sess_trivial'").fetchone()
        standard_row = conn.execute("SELECT system_prompt_score FROM requests WHERE session_id='sess_standard'").fetchone()
        deep_row = conn.execute("SELECT system_prompt_score FROM requests WHERE session_id='sess_deep'").fetchone()
        assert trivial_row['system_prompt_score'] == pytest.approx(0.0)
        assert standard_row['system_prompt_score'] == pytest.approx(1.0)
        assert deep_row['system_prompt_score'] == pytest.approx(2.0)
        conn.close()

    def test_record_request_stores_user_prompt_tier(self):
        db, path = _make_db()
        try:
            decision = _make_decision()
            row_id = db.record_request(
                session_id='sess-upt',
                conversation_anchor=None,
                routing_decision=decision,
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                system_prompt_tier='standard',
                system_prompt_score=50,
                user_prompt_score=75,
                routing_weighted_score=65,
                user_prompt_tier='deep',
            )
            row = db.get_request(row_id)
            assert row is not None
            assert row['user_prompt_tier'] == 'deep'
        finally:
            db.close()
            os.unlink(path)

    def test_record_request_user_prompt_tier_defaults_null(self):
        db, path = _make_db()
        try:
            row_id = _insert(db)
            row = db.get_request(row_id)
            assert row is not None
            assert row['user_prompt_tier'] is None
        finally:
            db.close()
            os.unlink(path)


class TestRoutingEconomicsMigration:
    """Migration 9: add net_savings_usd and classifier_overhead_usd columns."""

    def _apply_through_migration_8(self):
        """Return an open in-memory connection with migrations 0-8 applied (v9)."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        from anthproxy.db import _MIGRATIONS
        for i in range(9):
            with conn:
                _MIGRATIONS[i](conn)
                conn.execute(f"PRAGMA user_version = {i + 1};")
        return conn

    def test_migration_9_adds_economics_columns(self):
        conn = self._apply_through_migration_8()
        from anthproxy.db import _apply_migration_9
        with conn:
            _apply_migration_9(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        assert 'net_savings_usd' in cols
        assert 'classifier_overhead_usd' in cols
        conn.close()

    def test_migration_9_columns_default_null(self):
        conn = self._apply_through_migration_8()
        from anthproxy.db import _apply_migration_9
        with conn:
            _apply_migration_9(conn)
            conn.execute(
                "INSERT INTO requests(session_id, requested_model, backend, status) "
                "VALUES('s', 'm', 'b', 'success')"
            )
        row = conn.execute(
            "SELECT net_savings_usd, classifier_overhead_usd FROM requests"
        ).fetchone()
        assert row['net_savings_usd'] is None
        assert row['classifier_overhead_usd'] is None
        conn.close()


class TestRecordRequestEconomics:
    """record_request persists economics only for applied=True requests."""

    def _fetch(self, db, rowid):
        return db._conn.execute(
            "SELECT net_savings_usd, classifier_overhead_usd FROM requests WHERE id = ?",
            (rowid,),
        ).fetchone()

    def test_applied_persists_economics(self):
        db, path = _make_db()
        try:
            rowid = db.record_request(
                session_id='sess-econ',
                conversation_anchor=None,
                routing_decision=_make_decision(routed='haiku', applied=True),
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                net_savings_usd=4.0,
                classifier_overhead_usd=0.001,
            )
            row = self._fetch(db, rowid)
            assert row['net_savings_usd'] == pytest.approx(4.0)
            assert row['classifier_overhead_usd'] == pytest.approx(0.001)
        finally:
            db.close()
            os.unlink(path)

    def test_not_applied_forces_null(self):
        db, path = _make_db()
        try:
            rowid = db.record_request(
                session_id='sess-econ',
                conversation_anchor=None,
                routing_decision=_make_decision(routed='sonnet', applied=False),
                stats_dict=dict(_DEFAULT_STATS),
                duration_ms=100,
                backend='anthropic',
                status='success',
                net_savings_usd=4.0,
                classifier_overhead_usd=0.001,
            )
            row = self._fetch(db, rowid)
            assert row['net_savings_usd'] is None
            assert row['classifier_overhead_usd'] is None
        finally:
            db.close()
            os.unlink(path)

    def test_defaults_null_when_not_passed(self):
        db, path = _make_db()
        try:
            rowid = _insert(db, decision=_make_decision(applied=True))
            row = self._fetch(db, rowid)
            assert row['net_savings_usd'] is None
            assert row['classifier_overhead_usd'] is None
        finally:
            db.close()
            os.unlink(path)


# ---------------------------------------------------------------------------
# ADR 0018: Regroup sessions collapsed by the removed 128-char key truncation
# ---------------------------------------------------------------------------

class TestSessionUngroupingMigration:
    """Migration 10 splits truncated session keys back apart by conversation anchor."""

    # A truncated key: exactly 128 chars, cut mid-JSON so it never closes its brace.
    TRUNCATED = ('{"device_id":"' + 'a' * 64 + '","session_id":"' + 'b' * 34)[:128]

    def _apply_through_migration_9(self):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        from anthproxy.db import _MIGRATIONS
        for i in range(10):
            with conn:
                _MIGRATIONS[i](conn)
                conn.execute(f"PRAGMA user_version = {i + 1};")
        return conn

    def _insert_request(self, conn, session_id, anchor, ts='2026-08-01T00:00:00.000Z'):
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, conversation_anchor, request_ts, requested_model,
                    backend, status
                ) VALUES (?, ?, ?, 'sonnet', 'anthropic', 'success')""",
                (session_id, anchor, ts),
            )
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, last_seen_at) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO NOTHING",
                (session_id, ts, ts),
            )

    def _run(self, conn):
        from anthproxy.db import _apply_migration_10
        with conn:
            _apply_migration_10(conn)

    def test_schema_version_is_12(self):
        assert _SCHEMA_VERSION == 12

    def test_splits_truncated_key_by_conversation_anchor(self):
        conn = self._apply_through_migration_9()
        for i in range(5):
            for _ in range(3):
                self._insert_request(conn, self.TRUNCATED, f'anchor-{i}')
        self._run(conn)
        keys = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM requests")}
        assert len(keys) == 5
        assert self.TRUNCATED not in keys
        assert all(k.startswith(self.TRUNCATED + '_') for k in keys)
        counts = conn.execute(
            "SELECT session_id, COUNT(*) c FROM requests GROUP BY session_id"
        ).fetchall()
        assert all(r['c'] == 3 for r in counts)
        conn.close()

    def test_synthetic_keys_are_deterministic(self):
        expected = self.TRUNCATED + '_' + hashlib.sha256(b'anchor-x').hexdigest()[:16]
        conn = self._apply_through_migration_9()
        self._insert_request(conn, self.TRUNCATED, 'anchor-x')
        self._run(conn)
        row = conn.execute("SELECT session_id FROM requests").fetchone()
        assert row['session_id'] == expected
        conn.close()

    def test_leaves_non_truncated_sessions_untouched(self):
        full = '{"device_id":"abc","session_id":"def","account_uuid":"ghi"}'
        exactly_128_valid_json = '{"device_id":"' + 'c' * 112 + '"}'
        assert len(exactly_128_valid_json) == 128
        conn = self._apply_through_migration_9()
        self._insert_request(conn, full, 'anchor-1')
        self._insert_request(conn, full, 'anchor-2')
        self._insert_request(conn, exactly_128_valid_json, 'anchor-3')
        self._run(conn)
        keys = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM requests")}
        assert keys == {full, exactly_128_valid_json}
        conn.close()

    def test_rows_without_anchor_keep_original_key(self):
        conn = self._apply_through_migration_9()
        self._insert_request(conn, self.TRUNCATED, None)
        self._insert_request(conn, self.TRUNCATED, 'anchor-1')
        self._run(conn)
        keys = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM requests")}
        assert self.TRUNCATED in keys
        assert len(keys) == 2
        # The old session row survives because it still owns the anchorless request.
        surviving = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (self.TRUNCATED,)
        ).fetchone()[0]
        assert surviving == 1
        conn.close()

    def test_sessions_table_consistent_with_requests(self):
        conn = self._apply_through_migration_9()
        for i in range(4):
            self._insert_request(conn, self.TRUNCATED, f'anchor-{i}', ts=f'2026-08-0{i + 1}T00:00:00.000Z')
        self._run(conn)
        missing = conn.execute(
            "SELECT COUNT(*) FROM requests r "
            "WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = r.session_id)"
        ).fetchone()[0]
        assert missing == 0
        orphans = conn.execute(
            "SELECT COUNT(*) FROM sessions s "
            "WHERE NOT EXISTS (SELECT 1 FROM requests r WHERE r.session_id = s.session_id)"
        ).fetchone()[0]
        assert orphans == 0
        conn.close()

    def test_session_timestamps_derive_from_requests(self):
        conn = self._apply_through_migration_9()
        self._insert_request(conn, self.TRUNCATED, 'anchor-1', ts='2026-08-01T00:00:00.000Z')
        self._insert_request(conn, self.TRUNCATED, 'anchor-1', ts='2026-08-09T00:00:00.000Z')
        self._run(conn)
        row = conn.execute("SELECT created_at, last_seen_at FROM sessions").fetchone()
        assert row['created_at'] == '2026-08-01T00:00:00.000Z'
        assert row['last_seen_at'] == '2026-08-09T00:00:00.000Z'
        conn.close()

    def test_conversation_summaries_follow_their_session(self):
        conn = self._apply_through_migration_9()
        self._insert_request(conn, self.TRUNCATED, 'anchor-1')
        with conn:
            conn.execute(
                "INSERT INTO conversation_summaries (session_id, conversation_anchor, summary, updated_at) "
                "VALUES (?, 'anchor-1', 'a summary', '2026-08-01T00:00:00.000Z')",
                (self.TRUNCATED,),
            )
        self._run(conn)
        row = conn.execute("SELECT session_id, summary FROM conversation_summaries").fetchone()
        new_key = conn.execute("SELECT session_id FROM requests").fetchone()['session_id']
        assert row['session_id'] == new_key
        assert row['summary'] == 'a summary'
        conn.close()

    def test_stale_session_summary_removed_with_collapsed_session(self):
        conn = self._apply_through_migration_9()
        self._insert_request(conn, self.TRUNCATED, 'anchor-1')
        with conn:
            conn.execute(
                "INSERT INTO session_summaries (session_id, summary, updated_at) "
                "VALUES (?, 'collapsed summary', '2026-08-01T00:00:00.000Z')",
                (self.TRUNCATED,),
            )
        self._run(conn)
        assert conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0] == 0
        conn.close()

    def test_orphaned_conversation_summaries_cleaned(self):
        """Conversation summaries with no backing requests are deleted."""
        conn = self._apply_through_migration_9()
        self._insert_request(conn, self.TRUNCATED, 'anchor-1')
        # Insert an orphaned summary (no request row for this anchor).
        with conn:
            conn.execute(
                "INSERT INTO conversation_summaries (session_id, conversation_anchor, summary, updated_at) "
                "VALUES (?, 'orphan-anchor', 'orphaned', '2026-08-01T00:00:00.000Z')",
                (self.TRUNCATED,),
            )
        self._run(conn)
        orphaned = conn.execute(
            "SELECT COUNT(*) FROM conversation_summaries WHERE conversation_anchor = 'orphan-anchor'"
        ).fetchone()[0]
        assert orphaned == 0
        conn.close()

    def test_migration_is_idempotent(self):
        conn = self._apply_through_migration_9()
        for i in range(3):
            self._insert_request(conn, self.TRUNCATED, f'anchor-{i}')
        self._run(conn)
        first = sorted(r[0] for r in conn.execute("SELECT session_id FROM requests"))
        self._run(conn)
        assert sorted(r[0] for r in conn.execute("SELECT session_id FROM requests")) == first
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
        conn.close()

    def test_regroups_at_production_scale(self):
        """The reported corruption: 11,267 requests collapsed into 708 conversations."""
        conn = self._apply_through_migration_9()
        with conn:
            conn.executemany(
                """INSERT INTO requests (
                    session_id, conversation_anchor, request_ts, requested_model,
                    backend, status
                ) VALUES (?, ?, '2026-08-01T00:00:00.000Z', 'sonnet', 'anthropic', 'success')""",
                [(self.TRUNCATED, f'anchor-{i % 708}') for i in range(11267)],
            )
            conn.execute(
                "INSERT INTO sessions (session_id) VALUES (?)", (self.TRUNCATED,)
            )
        self._run(conn)
        assert conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM requests"
        ).fetchone()[0] == 708
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 11267
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 708
        conn.close()

    def test_no_op_when_no_truncated_keys(self):
        conn = self._apply_through_migration_9()
        self._insert_request(conn, 'plain-session', 'anchor-1')
        self._run(conn)
        assert conn.execute("SELECT session_id FROM requests").fetchone()[0] == 'plain-session'
        conn.close()

    def test_runs_on_startup_via_ensure_schema(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            from anthproxy.db import _MIGRATIONS
            for i in range(10):
                with conn:
                    _MIGRATIONS[i](conn)
                    conn.execute(f"PRAGMA user_version = {i + 1};")
            self._insert_request(conn, self.TRUNCATED, 'anchor-1')
            self._insert_request(conn, self.TRUNCATED, 'anchor-2')
            conn.close()

            db = SessionDB(path)
            try:
                keys = {r[0] for r in db._conn.execute("SELECT DISTINCT session_id FROM requests")}
                assert len(keys) == 2
                assert self.TRUNCATED not in keys
                assert db._conn.execute("PRAGMA user_version;").fetchone()[0] == _SCHEMA_VERSION
            finally:
                db.close()
        finally:
            os.unlink(path)


class TestUntrackedSessionMigration:
    """Migration 11 renames the empty-string session key to UNTRACKED_SESSION_ID."""

    def _apply_through_migration_10(self):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        from anthproxy.db import _MIGRATIONS
        for i in range(11):
            with conn:
                _MIGRATIONS[i](conn)
                conn.execute(f"PRAGMA user_version = {i + 1};")
        return conn

    def _run(self, conn):
        from anthproxy.db import _apply_migration_11
        with conn:
            _apply_migration_11(conn)

    def test_renames_empty_session_id_on_requests(self):
        from anthproxy.constants import UNTRACKED_SESSION_ID
        conn = self._apply_through_migration_10()
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, conversation_anchor, request_ts, requested_model,
                    backend, status
                ) VALUES ('', 'anchor-1', '2026-08-01T00:00:00.000Z', 'sonnet',
                    'anthropic', 'success')"""
            )
        self._run(conn)
        keys = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM requests")}
        assert keys == {UNTRACKED_SESSION_ID}
        conn.close()

    def test_merges_existing_untracked_row_on_sessions(self):
        from anthproxy.constants import UNTRACKED_SESSION_ID
        conn = self._apply_through_migration_10()
        with conn:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, last_seen_at) "
                "VALUES ('', '2026-08-01T00:00:00.000Z', '2026-08-01T00:00:00.000Z')"
            )
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, last_seen_at) "
                "VALUES (?, '2026-08-02T00:00:00.000Z', '2026-08-02T00:00:00.000Z')",
                (UNTRACKED_SESSION_ID,),
            )
        self._run(conn)
        rows = conn.execute("SELECT session_id FROM sessions").fetchall()
        assert [r[0] for r in rows] == [UNTRACKED_SESSION_ID]
        conn.close()

    def test_no_op_when_no_empty_session_ids(self):
        conn = self._apply_through_migration_10()
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    session_id, conversation_anchor, request_ts, requested_model,
                    backend, status
                ) VALUES ('plain-session', 'anchor-1', '2026-08-01T00:00:00.000Z',
                    'sonnet', 'anthropic', 'success')"""
            )
        self._run(conn)
        assert conn.execute("SELECT session_id FROM requests").fetchone()[0] == 'plain-session'
        conn.close()

    def test_runs_on_startup_via_ensure_schema(self):
        from anthproxy.constants import UNTRACKED_SESSION_ID
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            from anthproxy.db import _MIGRATIONS
            for i in range(11):
                with conn:
                    _MIGRATIONS[i](conn)
                    conn.execute(f"PRAGMA user_version = {i + 1};")
            with conn:
                conn.execute(
                    """INSERT INTO requests (
                        session_id, conversation_anchor, request_ts, requested_model,
                        backend, status
                    ) VALUES ('', 'anchor-1', '2026-08-01T00:00:00.000Z', 'sonnet',
                        'anthropic', 'success')"""
                )
            conn.close()

            db = SessionDB(path)
            try:
                keys = {r[0] for r in db._conn.execute("SELECT DISTINCT session_id FROM requests")}
                assert keys == {UNTRACKED_SESSION_ID}
                assert db._conn.execute("PRAGMA user_version;").fetchone()[0] == _SCHEMA_VERSION
            finally:
                db.close()
        finally:
            os.unlink(path)
