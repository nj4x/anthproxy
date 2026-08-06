import calendar
import dataclasses
import datetime as dt
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from ._shared import UsageRateLimitError


logger = logging.getLogger(__name__)

_USAGE_TTL_SECONDS = 300.0
_DEFAULT_COOLDOWN_SECONDS = 300.0
_MIN_PROBE_COOLDOWN_SECONDS = 60.0
_MAX_TOKENS = 64


@dataclasses.dataclass(frozen=True)
class OAuthRequestCredentials:
    generation: int
    access_token: str = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class OAuthTokenSnapshot:
    generation: int = 0
    fingerprint: str | None = None
    usage: dict | None = None
    usage_age_seconds: float | None = None
    health_ok: bool | None = None
    cooldown_remaining_seconds: float = 0.0
    monthly_blocked: bool = False
    burn: float | None = None
    eligible: bool = False
    usage_stale: bool = False
    month_elapsed_pct: float = 0.0


@dataclasses.dataclass
class _TokenState:
    credential: OAuthRequestCredentials
    fingerprint: str
    usage: dict | None = None
    usage_at: float = 0.0
    usage_month: tuple[int, int] | None = None
    health_ok: bool | None = None
    cooldown_until: float = 0.0
    monthly_blocked_until: dt.datetime | None = None
    probing: bool = False


class OAuthTokenRegistry:
    def __init__(
        self,
        usage_probe: Callable[[str], dict] | None = None,
        health_probe: Callable[[str], None] | None = None,
        wake: Callable[[], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], dt.datetime] | None = None,
    ):
        self._usage_probe = usage_probe
        self._health_probe = health_probe
        self._wake = wake
        self._monotonic = monotonic
        self._utcnow = utcnow or (lambda: dt.datetime.now(dt.timezone.utc))
        self._lock = threading.Lock()
        self._generation = 0
        self._states: OrderedDict[int, _TokenState] = OrderedDict()
        self._fingerprints: dict[str, int] = {}
        self._latest_generation = 0

    def set_wake(self, wake: Callable[[], None]) -> None:
        with self._lock:
            self._wake = wake

    def observe(self, access_token: str) -> OAuthRequestCredentials:
        fingerprint = hashlib.sha256(access_token.encode()).hexdigest()[:16]
        with self._lock:
            generation = self._fingerprints.get(fingerprint)
            state = self._states.get(generation) if generation is not None else None
            if state is not None and state.credential.access_token == access_token:
                self._states.move_to_end(state.credential.generation)
                self._latest_generation = state.credential.generation
                return state.credential
            self._generation += 1
            credential = OAuthRequestCredentials(self._generation, access_token)
            state = _TokenState(credential=credential, fingerprint=fingerprint)
            # Carry over fresh usage from the previous latest token so eligibility
            # is maintained across token rotation (one enterprise account, new
            # access_token on expiry).  The probe for the new token will refresh it;
            # snapshot() recomputes age and eligible from usage_at so TTL semantics
            # are preserved automatically.  Do NOT carry over health_ok — the new
            # token must prove its own validity via probe, else a revoked/wrongly-
            # scoped token would be used anyway.
            prev_state = self._states.get(self._latest_generation)
            if prev_state is not None and prev_state.usage is not None:
                state.usage = prev_state.usage
                state.usage_at = prev_state.usage_at
                state.usage_month = prev_state.usage_month
            self._states[credential.generation] = state
            self._fingerprints[fingerprint] = credential.generation
            self._latest_generation = credential.generation
            while len(self._states) > _MAX_TOKENS:
                old_generation, old_state = self._states.popitem(last=False)
                if self._fingerprints.get(old_state.fingerprint) == old_generation:
                    del self._fingerprints[old_state.fingerprint]
            wake = self._wake
        if wake is not None:
            wake()
        return credential

    def credentials(self, generation: int) -> OAuthRequestCredentials | None:
        with self._lock:
            state = self._states.get(generation)
            return state.credential if state is not None else None

    def snapshot(self, generation: int | None = None) -> OAuthTokenSnapshot:
        now = self._monotonic()
        wall = self._utcnow()
        with self._lock:
            selected = generation if generation is not None else self._latest_generation
            state = self._states.get(selected)
            if state is None:
                return OAuthTokenSnapshot()
            usage = state.usage
            usage_month = state.usage_month
            age = now - state.usage_at if state.usage_at else None
            health_ok = state.health_ok
            cooldown = max(0.0, state.cooldown_until - now)
            monthly_blocked = (
                state.monthly_blocked_until is not None
                and wall < state.monthly_blocked_until
            )
            fingerprint = state.fingerprint
        burn, usage_valid, cap_reached = _usage_burn(usage)
        monthly_blocked = monthly_blocked or (
            cap_reached and usage_month == (wall.year, wall.month)
        )
        usage_stale = age is not None and age > _USAGE_TTL_SECONDS
        month_elapsed_pct = _month_elapsed_pct(wall)
        eligible = (
            age is not None
            and age <= _USAGE_TTL_SECONDS
            and usage_month == (wall.year, wall.month)
            and health_ok is True
            and cooldown <= 0
            and not monthly_blocked
            and usage_valid
        )
        return OAuthTokenSnapshot(
            generation=selected,
            fingerprint=fingerprint,
            usage=usage,
            usage_age_seconds=age,
            health_ok=health_ok,
            cooldown_remaining_seconds=cooldown,
            monthly_blocked=monthly_blocked,
            burn=burn,
            eligible=eligible,
            usage_stale=usage_stale,
            month_elapsed_pct=month_elapsed_pct,
        )

    def record_probe_success(self, generation: int, usage: dict, health_ok: bool) -> bool:
        now = self._monotonic()
        wall = self._utcnow()
        _burn, valid, cap_reached = _usage_burn(usage)
        with self._lock:
            state = self._states.get(generation)
            if state is None:
                return False
            state.usage = usage
            state.usage_at = now
            state.usage_month = (wall.year, wall.month)
            state.health_ok = health_ok
            state.probing = False
            # Usage is authoritative for cap state only when the probe produced a
            # valid reading (enabled, parseable, non-negative).  Set the monthly
            # block when capped; clear any provisional 429-driven block only when
            # a fresh probe *confirms* under-cap (valid and not capped).  Leave
            # monthly_blocked_until unchanged on indeterminate/unparseable usage
            # so a probe hiccup cannot silently lift a genuine month-long park.
            if cap_reached:
                state.monthly_blocked_until = _next_month(wall)
            elif valid:
                state.monthly_blocked_until = None
            return True

    def record_probe_failure(self, generation: int, stage: str) -> bool:
        with self._lock:
            state = self._states.get(generation)
            if state is None:
                return False
            if stage == 'usage':
                state.usage = None
                state.usage_at = 0.0
                state.usage_month = None
            state.health_ok = False
            state.probing = False
            return True

    def mark_cooldown(self, generation: int, retry_after: float | None = None) -> bool:
        duration = retry_after if retry_after is not None else _DEFAULT_COOLDOWN_SECONDS
        with self._lock:
            state = self._states.get(generation)
            if state is None:
                return False
            state.cooldown_until = max(
                state.cooldown_until,
                self._monotonic() + max(0.0, duration),
            )
            return True

    def mark_cap_exhausted(self, generation: int) -> bool:
        """Park a token until the next UTC month after a spend-cap 429.

        Used when an OAuth 429 carries no Retry-After guidance: the token was
        eligible (last probe under-cap), so the 429 itself is the exhaustion
        signal.  The block is provisional — the next under-cap probe clears it
        (see :meth:`record_probe_success`), bounding a false-positive park.
        """
        wall = self._utcnow()
        with self._lock:
            state = self._states.get(generation)
            if state is None:
                return False
            state.monthly_blocked_until = _next_month(wall)
            return True

    def tick(self, force: bool = False, background: bool = False) -> OAuthTokenSnapshot:
        now = self._monotonic()
        with self._lock:
            pending = []
            for state in self._states.values():
                due = state.usage_at == 0.0 or now - state.usage_at >= _USAGE_TTL_SECONDS
                if (
                    not state.probing
                    and (force or due)
                    and now >= state.cooldown_until
                    and self._usage_probe is not None
                ):
                    state.probing = True
                    pending.append((state.credential, state.health_ok is not True))
        for credential, health_due in pending:
            if background:
                threading.Thread(
                    target=self._probe,
                    args=(credential, health_due),
                    name='oauth-usage-probe',
                    daemon=True,
                ).start()
            else:
                self._probe(credential, health_due)
        return self.snapshot()

    def _probe(self, credential: OAuthRequestCredentials, health_due: bool) -> None:
        fp = hashlib.sha256(credential.access_token.encode()).hexdigest()[:16]
        try:
            usage = self._usage_probe(credential.access_token)
        except UsageRateLimitError as exc:
            cooldown = exc.retry_after if exc.retry_after is not None else _DEFAULT_COOLDOWN_SECONDS
            cooldown = max(cooldown, _MIN_PROBE_COOLDOWN_SECONDS)
            logger.info(
                'OAuth enterprise usage probe throttled (stage=usage, token=%s): %s; backing off %.0fs',
                fp, exc, cooldown,
            )
            self.mark_cooldown(credential.generation, cooldown)
            self.record_probe_failure(credential.generation, 'usage')
            return
        except Exception as exc:
            logger.warning(
                'OAuth enterprise usage probe failed (stage=usage, token=%s): %s: %s',
                fp, type(exc).__name__, exc,
            )
            self.record_probe_failure(credential.generation, 'usage')
            return
        health_ok = False
        if health_due and self._health_probe is not None:
            try:
                self._health_probe(credential.access_token)
                health_ok = True
            except Exception as exc:
                logger.warning(
                    'OAuth enterprise health probe failed (stage=health, token=%s): %s: %s',
                    fp, type(exc).__name__, exc,
                )
                self.record_probe_failure(credential.generation, 'health')
                return
        else:
            # No health probe configured or not due; successful usage probe
            # implies token validity, so mark as healthy.
            health_ok = True
        burn, valid, _cap = _usage_burn(usage)
        logger.debug(
            'OAuth enterprise usage probe ok (token=%s): valid=%s burn=%s health_ok=%s',
            fp, valid, burn, health_ok,
        )
        self.record_probe_success(credential.generation, usage, health_ok=health_ok)


def _month_elapsed_pct(now: dt.datetime) -> float:
    """Fraction of the current UTC month already elapsed by calendar day, 0–100.

    Matches the real spend-cap reset boundary (UTC month rollover).  ``day`` is
    1-based, so day 1 yields 0.0 and the last day yields ``(n-1)/n × 100``; the
    result is in ``[0, ~96.8]`` by construction, needing no clamp.
    """
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    return (now.day - 1) / days_in_month * 100.0


def _next_month(now: dt.datetime) -> dt.datetime:
    if now.month == 12:
        return dt.datetime(now.year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return dt.datetime(now.year, now.month + 1, 1, tzinfo=dt.timezone.utc)


def _usage_burn(usage: dict | None) -> tuple[float | None, bool, bool]:
    # ``burn`` is the raw percent of the monthly quota consumed (0-100), the
    # same "how full is this quota window" measure the selector already uses
    # for personal weekly utilization.  Keeping both sides on the raw-percent
    # scale is what makes the oauth-vs-personal comparison in
    # ``snapshot_for_request`` meaningful; a pace-projected rate would not be
    # comparable to the personal side's raw snapshot.
    if not isinstance(usage, dict):
        return None, False, False
    extra = usage.get('extra_usage')
    if not isinstance(extra, dict):
        return None, False, False
    try:
        monthly_limit = float(extra['monthly_limit'])
        used_credits = float(extra['used_credits'])
        utilization = float(extra['utilization'])
    except (KeyError, TypeError, ValueError):
        return None, False, False
    enabled = extra.get('is_enabled') is True
    cap_reached = extra.get('spend_limit_reached') is True or utilization >= 100.0
    valid = enabled and monthly_limit > 0 and used_credits >= 0 and utilization >= 0
    burn = utilization if valid else None
    return burn, valid, cap_reached
