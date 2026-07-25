"""Persistence and analytics interfaces for handlers and admin."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class HandlerStore(Protocol):
    """What the request path (handlers.py) needs from SessionDB."""

    def record_request(
        self,
        session_id: str,
        conversation_anchor: str | None,
        routing_decision,
        stats_dict: dict,
        duration_ms: int,
        backend: str,
        status: str,
        error: str | None,
        attempt: int,
        user_prompt_text: str | None = None,
        system_prompt_sha256: str | None = None,
        tools_sha256: str | None = None,
        routing_recovered_via_walkback: int | None = None,
        classifier_model: str | None = None,
        classifier_summary_json: str | None = None,
        classifier_raw_response: str | None = None,
        classifier_confidence: float | None = None,
        classifier_format: str | None = None,
        prompt_store_entries: dict | None = None,
        response_text: str | None = None,
    ) -> int:
        ...

    def update_request_on_retry(
        self,
        request_id: int,
        new_backend: str,
        attempt: int,
        cost_estimate: float,
        status: str,
        error: str | None,
        **stats_dict,
    ) -> None:
        ...

    def get_session_metadata(self, session_id: str) -> dict | None:
        ...


@runtime_checkable
class RequestReader(Protocol):
    """Read-only access for admin GET endpoints."""

    def get_sessions(self, limit=50, offset=0, q=None) -> list[dict]:
        ...

    def get_sessions_count(self, q=None) -> int:
        ...

    def get_session(self, session_id: str) -> dict | None:
        ...

    def get_trace(self, session_id, anchor=None, limit=100, offset=0, q=None) -> list[dict]:
        ...

    def get_trace_count(self, session_id, anchor=None, q=None) -> int:
        ...

    def get_session_summary(self, session_id: str) -> dict | None:
        ...

    def get_cost(self, group_by='model', since='-7 days', session_id=None) -> dict:
        ...

    def get_routing(self, since='-7 days', session_id=None) -> dict:
        ...

    def get_config_changes(self, limit=100) -> list[dict]:
        ...

    def busy_secs_window(self, backend: str, since: str) -> float:
        ...

    def get_session_overrides(self) -> dict:
        ...

    def get_stats(self, period='week', backend=None) -> dict:
        ...

    def get_request(self, request_id) -> dict | None:
        ...

    def get_prompt(self, sha256: str) -> dict | None:
        ...


@runtime_checkable
class AdminStore(RequestReader, Protocol):
    """Write access for admin POST endpoints (extends RequestReader)."""

    def set_session_backend(self, session_id: str, backend: str | None) -> None:
        ...

    def set_session_tier(self, session_id: str, tier: str | None) -> None:
        ...

    def record_config_change(
        self,
        event_type: str,
        actor: str,
        actor_id: str,
        prev_value: str | None,
        new_value: str | None,
    ) -> None:
        ...
