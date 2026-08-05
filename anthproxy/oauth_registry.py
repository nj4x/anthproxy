import dataclasses
import datetime as dt
import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable


_USAGE_TTL_SECONDS = 300.0
_DEFAULT_COOLDOWN_SECONDS = 300.0
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
        burn, usage_valid, cap_reached = _usage_burn(usage, wall)
        monthly_blocked = monthly_blocked or (
            cap_reached and usage_month == (wall.year, wall.month)
        )
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
        )

    def record_probe_success(self, generation: int, usage: dict, health_ok: bool) -> bool:
        now = self._monotonic()
        wall = self._utcnow()
        _burn, _valid, cap_reached = _usage_burn(usage, wall)
        with self._lock:
            state = self._states.get(generation)
            if state is None:
                return False
            state.usage = usage
            state.usage_at = now
            state.usage_month = (wall.year, wall.month)
            state.health_ok = health_ok
            state.probing = False
            if cap_reached:
                state.monthly_blocked_until = _next_month(wall)
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

    def tick(self, force: bool = False, background: bool = False) -> OAuthTokenSnapshot:
        now = self._monotonic()
        with self._lock:
            pending = []
            for state in self._states.values():
                due = state.usage_at == 0.0 or now - state.usage_at >= _USAGE_TTL_SECONDS
                if (
                    not state.probing
                    and (force or due)
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
        try:
            usage = self._usage_probe(credential.access_token)
        except Exception:
            self.record_probe_failure(credential.generation, 'usage')
            return
        if health_due and self._health_probe is not None:
            try:
                self._health_probe(credential.access_token)
            except Exception:
                self.record_probe_failure(credential.generation, 'health')
                return
        self.record_probe_success(credential.generation, usage, health_ok=True)


def _next_month(now: dt.datetime) -> dt.datetime:
    if now.month == 12:
        return dt.datetime(now.year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return dt.datetime(now.year, now.month + 1, 1, tzinfo=dt.timezone.utc)


def _usage_burn(usage: dict | None, now: dt.datetime) -> tuple[float | None, bool, bool]:
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
    month_start = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
    month_end = _next_month(now)
    elapsed = (now - month_start).total_seconds()
    duration = (month_end - month_start).total_seconds()
    elapsed_fraction = max(elapsed / duration, 1 / duration)
    burn = utilization / elapsed_fraction if valid else None
    return burn, valid, cap_reached
