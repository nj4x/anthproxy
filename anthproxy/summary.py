"""Background generation of concise technical session summaries."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_INTERVAL_SECS = 60.0
_STARTUP_GRACE_SECS = 10.0
_MAX_USER_PROMPT_CHARS = 800
_MAX_SYSTEM_PROMPT_CHARS = 1200

_USER_PROMPT_SYSTEM = (
    "You summarize recent activity for a technical AI proxy dashboard. "
    "Using the supplied recent user messages, write one or two concise, specific "
    "sentences about the work in progress. Name languages, frameworks, files, "
    "features, or concrete errors when evident. Do not speculate. "
    "Example: User was implementing JWT auth middleware in Go and debugging a "
    "403 response. Return only the summary."
)

_SYSTEM_PROMPT_SYSTEM = (
    "You summarize technical AI sessions for a proxy dashboard. The supplied "
    "content is a session system prompt because no readable user prompt has been "
    "recorded yet. Write one or two concise, specific sentences describing the "
    "project context or technical task implied by it. Name technologies, roles, "
    "or domains only when stated. Do not speculate. Return only the summary."
)


class SummaryDaemon:
    """Generate stored session summaries in a daemon thread."""

    def __init__(
        self,
        db,
        registry,
        interval: float = _INTERVAL_SECS,
        startup_grace: float = _STARTUP_GRACE_SECS,
    ) -> None:
        self._db = db
        self._registry = registry
        self._interval = interval
        self._startup_grace = startup_grace
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name='anthproxy-session-summaries',
            daemon=True,
        )
        self._thread.start()
        logger.info(
            'Session summary daemon started (interval=%.0fs)',
            self._interval,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        if self._stop.wait(self._startup_grace):
            return

        while not self._stop.is_set():
            try:
                self._refresh_all()
            except Exception:
                logger.exception('Session summary daemon refresh failed')
            if self._stop.is_set():
                return
            try:
                self._refresh_conversations()
            except Exception:
                logger.exception('Conversation summary daemon refresh failed')
            self._stop.wait(self._interval)

    def _refresh_all(self) -> None:
        for item in self._db.get_sessions_for_summary():
            if self._stop.is_set():
                return

            session_id = item['session_id']
            prompts: list[str] = item['recent_prompts']
            system_prompt: str | None = item['recent_system_prompt']

            if not prompts and not system_prompt:
                logger.debug(
                    'Session summary skipped: no prompt content (session=%s…)',
                    session_id[:16],
                )
                continue

            try:
                summary = self._generate_summary(prompts, system_prompt)
                if summary:
                    self._db.upsert_session_summary(session_id, summary)
            except Exception as exc:
                logger.warning(
                    'Session summary failed (session=%s…): %s',
                    session_id[:16],
                    exc,
                )

    def _refresh_conversations(self) -> None:
        for item in self._db.get_conversations_for_summary():
            if self._stop.is_set():
                return

            session_id = item['session_id']
            anchor = item['conversation_anchor']
            prompts: list[str] = item['recent_prompts']

            if not prompts:
                logger.debug(
                    'Conversation summary skipped: no prompt content '
                    '(session=%s… anchor=%s)',
                    session_id[:16],
                    anchor,
                )
                continue

            try:
                summary = self._generate_summary(prompts, None)
                if summary:
                    self._db.upsert_conversation_summary(session_id, anchor, summary)
            except Exception as exc:
                logger.warning(
                    'Conversation summary failed (session=%s… anchor=%s): %s',
                    session_id[:16],
                    anchor,
                    exc,
                )

    def _generate_summary(
        self,
        prompts: list[str],
        system_prompt: str | None,
    ) -> str:
        """Call the active backend directly and return extracted text.

        snapshot() acquires and releases registry state internally before
        returning. Credential parsing and the backend network call occur only
        after that return, so no registry or selector lock is held across
        provider I/O.
        """
        snapshot = self._registry.snapshot()

        # Determine credentials and system prompt.
        credentials = self._get_credentials(snapshot)
        if credentials is None:
            logger.debug(
                'Skipping summary: backend %s has no eligible server credentials',
                snapshot.name,
            )
            return ''

        if prompts:
            recent_text = '\n\n---\n\n'.join(
                prompt[:_MAX_USER_PROMPT_CHARS]
                for prompt in reversed(prompts)
            )
            system = _USER_PROMPT_SYSTEM
            user_content = (
                'Recent user messages from one session:\n\n'
                f'{recent_text}\n\n'
                'Summarize the current technical work.'
            )
        else:
            system = _SYSTEM_PROMPT_SYSTEM
            user_content = (
                'System prompt from a session with no readable user prompt:\n\n'
                f'{(system_prompt or "")[:_MAX_SYSTEM_PROMPT_CHARS]}\n\n'
                'Summarize the implied technical context.'
            )

        payload = {
            'model': snapshot.config.auto_model_routing_classifier_model,
            'max_tokens': 120,
            'temperature': 0,
            'system': system,
            'messages': [
                {
                    'role': 'user',
                    'content': user_content,
                }
            ],
            '_anthproxy_internal_classifier': True,
        }

        send_fn = getattr(
            snapshot.backend,
            'send_classifier_message',
            snapshot.backend.send_message,
        )
        response = send_fn(payload, credentials, snapshot.config)

        if not isinstance(response, dict):
            return ''

        content = response.get('content')
        if not isinstance(content, list):
            return ''

        text_parts: list[str] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get('type') == 'text'
                and isinstance(block.get('text'), str)
            ):
                text_parts.append(block['text'])

        return ' '.join(text_parts).strip()

    def _get_credentials(self, snapshot) -> dict | None:
        """Return credentials for the active backend, or None if ineligible."""
        backend_name = snapshot.name

        # Bedrock requires request-scoped AWS credentials; skip it.
        if backend_name == 'bedrock':
            return None

        # Anthropic, Codex: parse_credentials('') returns {} (proxy-owned credentials).
        if backend_name in ('anthropic', 'codex'):
            return snapshot.backend.parse_credentials('')

        # OpenRouter: check configured API key.
        if backend_name == 'openrouter':
            if snapshot.config.openrouter_api_key:
                return snapshot.backend.parse_credentials('')
            return None

        # Gauss: check server-owned UMS_TOKEN.
        if backend_name == 'gauss':
            if snapshot.config.gauss_ums_token:
                return snapshot.backend.parse_credentials('')
            return None

        # Local: credential-free.
        if backend_name == 'local':
            return {}

        # Unknown backend; skip safely.
        logger.debug('Skipping summary: unknown backend %s', backend_name)
        return None
