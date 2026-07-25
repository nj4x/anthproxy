from abc import ABC, abstractmethod
import dataclasses
import http.client
import logging
import ssl
import threading
import time

from ..config import Config

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class UsageRateLimitError(RuntimeError):
    retry_after: float | None = None


@dataclasses.dataclass(frozen=True)
class FiveHourStatus:
    """5-hour usage window status for a subscription backend.

    ``available`` is True when the window has remaining quota, False when
    it is confirmed exhausted, or None when the usage endpoint could not be
    reached (transient error — caller should be conservative).
    ``resets_at`` is a POSIX timestamp (float) for when the window resets,
    or None when unavailable.
    ``utilization`` is the 5-hour usage percentage (0–100) when known, else None.
    ``weekly_utilization`` is the weekly usage percentage (0–100) when known, else None.
    """
    available: bool | None
    resets_at: float | None
    utilization: float | None = None
    weekly_utilization: float | None = None


class SubscriptionBackend:
    """Cache-guarded usage scaffolding for subscription backends (Codex, Anthropic).

    Mixin that provides ``get_usage`` (TTL-bounded cache + locking) and
    ``get_usage_markdown`` (try/except wrapper + provider-name logging).
    Concrete backends must set ``_PROVIDER_NAME`` and implement the three
    ``_…_impl`` hooks; everything else is inherited.

    Usage cache attributes:
        _usage_lock           threading.Lock — guards cache and cooldown state
        _usage_cache          dict | None — last successful fetch result
        _usage_cached_at      float — time.monotonic() stamp of last successful fetch
        _usage_backoff_until  float — time.monotonic() until which usage 429s are cached
        _usage_backoff_error  UsageRateLimitError | None — last usage 429 metadata
    """

    _PROVIDER_NAME: str = ''      # override in subclass (e.g. 'Codex', 'Anthropic')
    _USAGE_CACHE_TTL: float = 300.0
    _USAGE_RATE_LIMIT_TTL: float = 300.0

    def __init__(self):
        self._usage_lock = threading.Lock()
        self._usage_cache: dict | None = None
        self._usage_cached_at: float = 0.0
        self._usage_backoff_until: float = 0.0
        self._usage_backoff_error: UsageRateLimitError | None = None

    def _fetch_usage_data(self, config) -> dict:
        """Fetch the raw usage dict from the upstream endpoint. Override in subclass."""
        raise NotImplementedError

    def _format_usage_markdown_impl(self, usage: dict) -> str:
        """Format a usage dict as Markdown. Override in subclass."""
        raise NotImplementedError

    def _usage_failure_markdown_impl(self, message: str) -> str:
        """Return a Markdown error message when usage fetch fails. Override in subclass."""
        raise NotImplementedError

    def get_usage(self, config) -> dict:
        """Return the cached (or freshly fetched) usage dict.

        Raises ``AnthropicRequestError`` or ``RuntimeError`` on failure.
        Thread-safe; at most one successful upstream request per
        ``_USAGE_CACHE_TTL`` seconds, and usage-endpoint 429s are cached until
        their retry window elapses.
        """
        with self._usage_lock:
            now = time.monotonic()
            if (
                self._usage_cache is not None
                and now - self._usage_cached_at < self._USAGE_CACHE_TTL
            ):
                return self._usage_cache
            if self._usage_backoff_error is not None and now < self._usage_backoff_until:
                raise self._usage_backoff_error
            try:
                usage = self._fetch_usage_data(config)
            except UsageRateLimitError as exc:
                retry_after = exc.retry_after
                if retry_after is None:
                    retry_after = self._USAGE_RATE_LIMIT_TTL
                retry_after = max(0.0, retry_after)
                self._usage_backoff_until = time.monotonic() + retry_after
                self._usage_backoff_error = UsageRateLimitError(retry_after=retry_after)
                raise self._usage_backoff_error
            self._usage_cache = usage
            self._usage_cached_at = time.monotonic()
            self._usage_backoff_until = 0.0
            self._usage_backoff_error = None
            return usage

    def invalidate_usage_cache(self) -> None:
        with self._usage_lock:
            self._usage_cached_at = 0.0
            self._usage_cache = None

    def get_usage_markdown(self, config) -> str:
        """Return subscription usage as Markdown (cached ``_USAGE_CACHE_TTL`` s)."""
        from ..mapper import AnthropicRequestError
        try:
            usage = self.get_usage(config)
        except AnthropicRequestError as exc:
            logger.warning('%s usage lookup authentication failure: %s',
                           self._PROVIDER_NAME, exc.message)
            return self._usage_failure_markdown_impl(exc.message)
        except (OSError, http.client.HTTPException, ssl.SSLError, ValueError, RuntimeError) as exc:
            logger.warning('%s usage lookup failed: %s', self._PROVIDER_NAME, exc)
            return self._usage_failure_markdown_impl(str(exc))
        return self._format_usage_markdown_impl(usage)


class Backend(ABC):
    @abstractmethod
    def parse_credentials(self, api_key: str) -> dict:
        """Parse the x-api-key header value into backend-specific credentials."""
        ...

    @abstractmethod
    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        ...

    @abstractmethod
    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        """Return a generator yielding SSE event strings."""
        ...

    @abstractmethod
    def count_tokens(self, payload: dict, credentials: dict, config: Config) -> dict:
        ...

    def send_classifier_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        """Send an internal complexity-classifier request and return the response dict.

        The default implementation routes through the normal non-streaming
        ``send_message`` path.  Backends with session-specific side effects
        (e.g. Gauss, which maintains a request history and kill switch) should
        override this method to use a fresh isolated context so classifier calls
        are invisible to the user session.

        The ``payload`` always carries ``_anthproxy_internal_classifier = True``.
        Mappers are responsible for stripping that key before sending to the
        upstream provider.
        """
        return self.send_message(payload, credentials, config)

    def store_cached_credential(self, key: str, value: str) -> None:
        pass
