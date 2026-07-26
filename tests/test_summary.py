"""Tests for the background summary daemon (session + conversation summaries)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from anthproxy.local.backend import LocalBackend
from anthproxy.summary import SummaryDaemon


def _make_daemon(db) -> SummaryDaemon:
    registry = MagicMock()
    return SummaryDaemon(db, registry, interval=0.01, startup_grace=0.0)


class TestRefreshConversations:
    def test_upserts_generated_summary(self):
        db = MagicMock()
        db.get_conversations_for_summary.return_value = [
            {
                'session_id': 's1',
                'conversation_anchor': 'conv-a',
                'recent_prompts': ['Add rate limiting to the API'],
            }
        ]
        daemon = _make_daemon(db)
        daemon._generate_summary = MagicMock(return_value='Adding API rate limiting')

        daemon._refresh_conversations()

        daemon._generate_summary.assert_called_once_with(
            ['Add rate limiting to the API'], None
        )
        db.upsert_conversation_summary.assert_called_once_with(
            's1', 'conv-a', 'Adding API rate limiting'
        )

    def test_skips_conversation_without_prompts(self):
        db = MagicMock()
        db.get_conversations_for_summary.return_value = [
            {'session_id': 's1', 'conversation_anchor': 'conv-a', 'recent_prompts': []}
        ]
        daemon = _make_daemon(db)
        daemon._generate_summary = MagicMock(return_value='unused')

        daemon._refresh_conversations()

        daemon._generate_summary.assert_not_called()
        db.upsert_conversation_summary.assert_not_called()

    def test_empty_summary_not_persisted(self):
        db = MagicMock()
        db.get_conversations_for_summary.return_value = [
            {'session_id': 's1', 'conversation_anchor': 'conv-a', 'recent_prompts': ['x']}
        ]
        daemon = _make_daemon(db)
        daemon._generate_summary = MagicMock(return_value='')

        daemon._refresh_conversations()

        db.upsert_conversation_summary.assert_not_called()

    def test_generation_error_is_isolated(self):
        db = MagicMock()
        db.get_conversations_for_summary.return_value = [
            {'session_id': 's1', 'conversation_anchor': 'conv-a', 'recent_prompts': ['x']},
            {'session_id': 's1', 'conversation_anchor': 'conv-b', 'recent_prompts': ['y']},
        ]
        daemon = _make_daemon(db)
        daemon._generate_summary = MagicMock(
            side_effect=[RuntimeError('boom'), 'ok summary']
        )

        daemon._refresh_conversations()

        # First conversation failed but the second still got persisted.
        db.upsert_conversation_summary.assert_called_once_with('s1', 'conv-b', 'ok summary')


class TestRunLoop:
    def test_run_invokes_both_refreshers(self):
        db = MagicMock()
        daemon = _make_daemon(db)  # startup_grace=0.0
        daemon._refresh_all = MagicMock()

        # Stop the daemon during the first pass so the loop exits after one cycle.
        def _refresh_and_stop():
            daemon._stop.set()

        daemon._refresh_conversations = MagicMock(side_effect=_refresh_and_stop)

        daemon._run()

        daemon._refresh_all.assert_called_once()
        daemon._refresh_conversations.assert_called_once()


class TestGetCredentials:
    def test_bedrock_skipped(self):
        db = MagicMock()
        daemon = _make_daemon(db)
        snapshot = SimpleNamespace(name='bedrock', backend=MagicMock(), config=MagicMock())
        assert daemon._get_credentials(snapshot) is None

    def test_local_credential_free(self):
        db = MagicMock()
        daemon = _make_daemon(db)
        snapshot = SimpleNamespace(name='local', backend=LocalBackend(), config=MagicMock())
        assert daemon._get_credentials(snapshot) == {}
