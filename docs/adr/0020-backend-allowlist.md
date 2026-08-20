---
artifact-type: adr
lineage-rules: exempt
---

# ADR-0020: Restrict available backends with a `--backends` allowlist

**Status:** Accepted (with known limitations — see below)

**Date:** 2026-08-19

## Problem

Every backend that ships in the package is unconditionally live. `discover_backends()` imports each directory under `anthproxy/` that carries both `__init__.py` and `backend.py`, each package self-registers, and from that moment on the backend is selectable everywhere: `--backend` choices, `/backend` local commands, the `prefer:` override header, admin API switching, and `AutoSelector` rotation.

An operator who uses two backends still carries all five. Concretely:

- In auto mode, `__main__.py` calls `anthropic_auth.ensure_credentials()` and `codex_auth.ensure_credentials()` unconditionally — "for both subscription backends regardless of `--backend`". An operator who never uses Codex is still prompted for Codex credentials on startup.
- `AutoSelector` ranks candidates from hardcoded constants (`_PRIORITY = SUBSCRIPTION_BACKENDS` from selector.py:41, `_FALLBACK = 'bedrock'` from selector.py:42) and never consults the registry, so rotation can land on a backend the operator has no intention of using.
- The local-command help, the admin UI backend list, and `--backend` choices all advertise backends that will fail the moment they are selected.

There is no way to say "this deployment uses exactly these backends."

## Decision

Add `--backends` / `ANTHPROXY_BACKENDS`: a comma-separated allowlist of backend names. Absent, behavior is unchanged (all discovered backends).

1. **Keep `_BACKENDS` complete; apply the allowlist inside public accessors.** Discovery runs exactly as today — every backend package is imported, self-registers, and passes the existing integrity checks (`_assert_registered`, `_assert_plugin_hooks`, name/directory agreement, built-in completeness). The allowlist is not applied by removing entries from `_BACKENDS`; it is applied as a filter inside every public accessor: `backend_names()`, `list_backends()`, and `get_backend()`. `build_backend()` is not filtered directly — it resolves classes through `get_backend()` and inherits the filter from it. `_INTERNAL_BACKENDS` (currently `{'oauth'}`) is permanently exempt from the filter — internal backends remain resolvable via all public accessors and cannot be allowlisted or excluded. This exemption is load-bearing: `backend_names()` already omits internal backends, but `get_backend()` and `list_backends()` read `_BACKENDS` directly today, so an unqualified filter would make `oauth` unresolvable under any allowlist. The full internal set survives so re-discovery stays idempotent — a second call to `discover_backends()` sees the same complete registry and produces the same filtered public view. The allowlist is stored as module-level `_enabled_backends` in `backends_registry.py`, written by a new public function `set_enabled_backends(allowed)`. `parse_args()` calls `set_enabled_backends()` unconditionally — with the full discovered set when `--backends` is absent — ensuring repeated parses are idempotent and the "absent → unchanged behavior" guarantee holds in-process. Any test that installs a fake backend under an allowlist restriction via `temporary_registry` must rely on that same test's allowlist removal or override; the overlay does not clear the allowlist.

2. **Excluded means absent from the public view, not from `_BACKENDS`.** After the filter is active, `backend_names()`, `list_backends()`, and `get_backend()` all agree: an excluded backend is unknown to callers. There is no second "enabled" view to keep in sync; the filter is the only view.

3. **Allowlist only.** No negation syntax and no companion `--disable-backends`. One way to express the set.

4. **Validation is fail-fast, at parse time; parsing follows `_parse_classification_str` precedent.** `--backends` is parsed as: split on commas, strip whitespace from each token, drop empty tokens, de-duplicate (preserving first occurrence). The result is validated against the full discovered `backend_names()`; unknown tokens and an empty resulting set are `p.error()`. Internal backends like `oauth` are absent from `backend_names()` and are rejected as unknown tokens with the standard error message. The CLI flag takes precedence over the `ANTHPROXY_BACKENDS` environment variable; the env var is only consulted when the flag is absent. `--backend` loses its argparse `choices=` (which cannot express a dependency on another flag). The ordering is: `--backends` tokens are validated against the unfiltered `backend_names()` (discovery has already run, the filter has not yet been installed); `set_enabled_backends()` then installs the filter; `--backend` is validated last, against the now-filtered `backend_names()`. Validation of the two flags therefore straddles the install point — `--backends` validates before it, `--backend` validates after it. No caller downstream of `set_enabled_backends()` can obtain the full discovered set.

5. **No implicit repair of an explicit `--backend`.** When `--backend` is passed explicitly — via the CLI flag or the `ANTHPROXY_BACKEND` environment variable — its value must be a member of the allowed set; if it is not, that is a hard error naming both flags. An operator who names a backend explicitly gets an error, never a substitution.

6. **The `--backend` default is repaired when excluded.** When `--backend` is not explicitly passed (CLI flag absent, env var absent, using the config default) and the resolved default value is not in the enabled set, the default is repaired to the first enabled backend name. This is logged as `Backend default repaired: bedrock → anthropic`. If no enabled backends exist, validation in §4 has already failed and this rule does not apply. The distinction from §5 is intent: a default the operator never chose is an artifact of packaging, not a request, so silently narrowing it to the set they did choose is the faithful reading; an explicit value is a request and is honored or refused, never rewritten.

7. **Credential preparation and token refresh are gated by the allowlist.** In auto mode, `ensure_credentials` is called only for subscription backends in the enabled set. The OAuth-backed subset of the intersected pool — specifically, the hardcoded `("anthropic", "codex")` tuple at selector.py:474 — must be filtered against the enabled set before token refresh; OpenRouter is deliberately absent from this loop (it uses a static API key, not OAuth), so its membership in `_PRIORITY` is a separate, pre-existing invariant.

8. **`AutoSelector` intersects its candidates; subscription commands degrade gracefully.** `_PRIORITY` and `_FALLBACK` are intersected with the enabled set at construction. All module-level reads of `_PRIORITY` inside the class — selector.py:398, :430, :431, :543, :611 — and of `_FALLBACK` — selector.py:612, :694 — must each consult the intersected pool; the intersection must be stored on the `AutoSelector` instance, and every such read must be converted to an instance attribute, not a module-level rebinding — a partial conversion leaves `IndexError` live at the unconverted sites. The shared `SUBSCRIPTION_BACKENDS` tuple in `constants.py` is never rebound; the intersection is per-instance selector state, reconciling this decision with ADR-0007's reference to the `_PRIORITY = SUBSCRIPTION_BACKENDS` import-time alias (cited there as `selector.py:40`; the line is now selector.py:41) and §9's deferred stats.py. When `bedrock` is excluded in auto mode, the intersected `_FALLBACK` is empty. **Known limitation, accepted for v1**: this ADR does not fully specify the resulting selector behavior. `_compute_best_unlocked` may return `None` in this state rather than a real subscription-only mode (`self._mode == "subscription"` is a distinct, unrelated code path), and `registry.switch(None, ...)` would then fail the `name not in backend_names()` guard at server.py:582. This is deferred to implementation: the rotation code must be extended to treat empty-`_FALLBACK` as "keep the current subscription backend, do not attempt a fallback switch" rather than propagating `None`. Tracked as follow-up hardening, not blocking initial shipment, because it only triggers when an operator both excludes `bedrock` and exhausts every remaining subscription backend simultaneously — a narrow, observable-in-testing rotation state.

   **Auto mode with no subscription backends.** When the enabled set contains no subscription backend, `auto_backend` is implicitly set to `False` at startup with a log line `Auto-backend disabled: no subscription backends in enabled set`. A bedrock-only deployment requires no second flag: `--backends bedrock` alone is sufficient and boots without error.

   **Degradation of subscription commands.** selector.py:611 `return _PRIORITY[0], ...` raises `IndexError` when the intersected pool is empty; this is caught and converted to a typed failure — "no subscription backends are enabled". `/backend subscription` (the command branch at handlers.py:2025), `restrict_subscription()`, and the session-subscription path surface that typed failure rather than propagating the exception. Separately, server.py:968 already returns `SwitchResult(kind='failed')` when no candidate prepares, so it needs only its iteration source intersected, not new error handling.

9. **`stats.py` is deferred, not fixed.** It uses the hardcoded `SUBSCRIPTION_BACKENDS` literal (stats.py:48, :516, :520) rather than registry accessors, and interprets recorded history, not live routing. Its constants are not updated by this ADR — see the consumer enumeration in Consequences for the rationale.

10. **The admin API needs no change.** `GET /admin/backends` already reports `registry.list_backends()`; the accessor filter makes it correct for free. No `disabled` field is added.

## Consequences

**One mechanism, many call sites fixed for free.** Roughly thirteen places consult `backend_names()` or `list_backends()` — `--backend` choices, the `prefer:` override gate, `switch`/`instance`/`set_session_backend` membership guards, local-command parsing and help text, two admin endpoints. Filtering inside the public accessors corrects all of them without touching any of them. The alternative — threading an enabled set to each consumer — would have needed every site updated and would silently regress the first time a new one was added.

**The registry keeps two tiers.** Internally `_BACKENDS` holds the full discovered set; externally the public accessors expose the filtered view, minus the permanent `_INTERNAL_BACKENDS` exemption. Re-discovery stays idempotent: a second `discover_backends()` call sees the same complete internal state and produces the same filtered output. Callers always use public accessors and need not know the distinction.

**Subscription commands now have explicit degradation.** When no subscription backends are in the enabled set, commands and code paths that previously assumed at least one subscription backend existed return a typed error instead of raising `IndexError`. This is a visible contract change for those paths.

**Startup gets quieter for restricted deployments.** No credential prompt for a backend the operator excluded. This is the visible payoff; the import-cost saving is not, and was not the goal.

**Import cost is knowingly retained.** Excluding `bedrock` still imports `botocore`. Accepted: the flag reduces operator-facing surface, not startup latency. If import time later matters, lazy imports are a separable change that does not disturb this decision.

**Selector correctness now depends on an intersection that its own constants do not express.** `_PRIORITY` and `_FALLBACK` remain hardcoded; the allowlist is applied on top at construction. A future contributor editing those constants must remember the intersection exists.

**Most misconfiguration is caught at startup.** Every validation in this ADR is a startup error, and a proxy that boots and then fails every switch attempt is worse than one that refuses to boot with a message naming the flag. One site is not static, however: server.py:470 resolves the session-subscription fallback dynamically and could otherwise name an excluded backend at request time. It is repaired here (see below) to use an enabled fallback, or to degrade to the typed failure in §8 when no enabled subscription backend exists.

**Support loses a "why is this missing" affordance.** From inside the UI, an excluded backend is indistinguishable from one that was never built. Diagnosing it means asking for the `--backends` value. Accepted for now; a `disabled` field is a small, separable follow-up if it becomes a real pain.

### `SUBSCRIPTION_BACKENDS` consumers and their disposition

`SUBSCRIPTION_BACKENDS` is a hardcoded tuple in `constants.py` that bypasses the registry entirely. Every consumer is enumerated below; each is either corrected by this ADR or explicitly deferred. There is no third category.

**(a) Fixed by intersection in this ADR.** These sites select or refresh a backend on a live path, so an excluded name here is a real routing fault:

- **selector.py:41–42** — `_PRIORITY` / `_FALLBACK` definitions, and every in-class read of them (selector.py:398, :430, :431, :543, :611 for `_PRIORITY`; :612, :694 for `_FALLBACK`). Intersected per §8.
- **selector.py:474** — the `("anthropic", "codex")` OAuth refresh tuple. Filtered per §7.
- **server.py:470** — `name = self._active if self._active in SUBSCRIPTION_BACKENDS else 'anthropic'`. This is live routing code, not a plumbing nit: it can hand `snapshot()` a backend the operator excluded. Corrected to `... else (first enabled backend in SUBSCRIPTION_BACKENDS)`; if the intersection is empty, the session-subscription command degrades to the typed failure specified in §8.
- **server.py:968** — the `for sub in SUBSCRIPTION_BACKENDS` preparation loop in `set_session_subscription()`. Its iteration source is intersected; its existing `SwitchResult(kind='failed')` return already covers the empty case.

**(b) Deferred, with rationale.** These sites either read history rather than route, or name backends only in prose:

- **stats.py:48, :516, :520** — the import, a docstring, and the `record_backend in SUBSCRIPTION_BACKENDS` membership test. Deferred because stats interpret *recorded* history: a backend excluded today may legitimately appear in last week's records, and filtering the membership test would hide real data rather than correct it. Narrowing this tuple would be a data-visibility regression, not a fix.
- **handlers.py:1809, :1851** — help-text rows that interpolate the static tuple into the subscription-backends description. Deferred because they are cosmetic: they may over-advertise, but they do not route.
- **handlers.py:2035, :2121** — the global and session subscription confirmation strings, which interpolate `"/".join(SUBSCRIPTION_BACKENDS)`. Deferred for the same reason; the underlying command behavior is corrected by §8 even though the confirmation text may name an excluded backend.
- **server.py:467, :493** — membership *tests* (`resolved in SUBSCRIPTION_BACKENDS`, `prefer_backend not in SUBSCRIPTION_BACKENDS`). Deferred because a test against the wider tuple is permissive, never generative: it can admit a name for further checking but cannot itself produce an excluded backend, and the `prefer_backend in backend_names()` guard at server.py:489 already applies the filter on that path.
- **server.py:891, :914** — iteration over the tuple to read cached usage and cached instances. Deferred because both skip names absent from `self._instances`, and an excluded backend is never instantiated, so it can never appear in the result.

**Other hardcoded sites — future work.**

- **handlers.py:2107 and the surrounding session-subscription branch** were previously cited here as interpolating `SUBSCRIPTION_BACKENDS`. That citation was wrong: handlers.py:2107 is the `if arg == 'subscription':` branch test and contains no interpolation. The actual interpolation sites are handlers.py:1809, :1851, :2035, and :2121, enumerated above.
- **`anthproxy/refresher.py:70-85`** — `TokenRefresher._refresh_all` (refresher.py:70-73) hardcodes the same `anthropic`/`codex` pair as two sequential `_refresh_one` calls rather than a tuple, and `_refresh_one` (refresher.py:75-85) branches on those two literal names. It is unwired (not instantiated anywhere); wiring it up in the future must apply the same allowlist gate as `_refresh_tokens`.

**Help text is partly fixed.** The available-backends portion of help (which consults `backend_names()`) is fixed by the accessor filter; the subscription-backends portion (hardcoded `SUBSCRIPTION_BACKENDS` at handlers.py:1809 and :1851) is deferred to future work and may still name an excluded backend.

**Anything else hardcoding a backend name outside the registry can break.** The enumeration above is exhaustive as of this ADR. Code added later that bypasses the registry to name a backend directly is making an assumption the allowlist can invalidate.

## Alternatives considered

- **Filter at discovery time — never import excluded packages.** Rejected. It would have delivered the import saving, but the built-in completeness check exists precisely to catch a broken install, and relaxing it to accommodate a routine config flag trades a real integrity guarantee for a benefit that is not the point of the feature.
- **Keep the registry complete; add `enabled_backend_names()`.** Rejected as a separate accessor surface. The chosen design keeps `_BACKENDS` complete internally but applies the filter inside the existing public accessors — one view, not two. Threading a separate `enabled_*` function to each of the thirteen call sites would silently regress the first time a new site was added.
- **Denylist syntax (`--backends -bedrock`) or a separate `--disable-backends`.** Rejected. Two ways to express one set means combinations that conflict and a precedence rule nobody recalls under pressure.
- **Error when the `--backend` default is excluded, instead of repairing it.** Rejected in favor of §6. The concern was that a default changing based on an unrelated flag is a silent behavior change — but the default was never a choice the operator expressed, and forcing `--backends anthropic --backend anthropic` on every restricted deployment is ceremony that teaches operators to pass a flag they do not need. The repair is logged, and the case that actually warranted an error — an *explicit* `--backend` outside the set — is still a hard error under §5.
- **Require a second flag to disable auto mode in a bedrock-only deployment.** Rejected in favor of the implicit disable in §8. "Auto-select among zero candidates" has exactly one sensible meaning, and demanding the operator state it produces a startup failure whose only remedy is typing something the config already implies.
- **Warn and continue on unknown tokens.** Rejected. `--backends anthropic,codx` would degrade to a different, working backend set — the kind of failure noticed only in production.
- **Environment variable only, no CLI flag.** Rejected. It would have sidestepped the ordering question (discovery runs before `parse_args()`), but that question dissolves anyway once the filter is applied after registration inside the accessors.

## Known limitations (accepted for v1)

These gaps were identified during design review and are explicitly accepted as deferred risk rather than blocking issues, so implementation can proceed against a coherent-but-incomplete design:

- **Empty-`_FALLBACK` selector behavior** — see §8. Only reachable when bedrock is excluded and every remaining subscription backend is simultaneously exhausted.
- **Accessor-rejection failure shape is unspecified.** `get_backend()` returning nothing for an excluded name is not given a uniform contract; call sites that already wrap backend construction in broad exception handling (e.g. handlers.py:2206's credential-cache endpoint, handlers.py:1955's usage-status loop) will surface whatever their existing generic handling produces — an opaque 500 or a silently skipped entry — rather than a message naming the allowlist. Tighten this once the feature is in use and the actual failure shapes are observed.
- **The hardcoded-backend-name enumeration in Consequences is not exhaustive.** It covers `SUBSCRIPTION_BACKENDS` consumers. Other single-name literals outside that constant (e.g. handlers.py:1950, handlers.py:2206, server.py:178/:202/:594/:597, __main__.py:63-86's explicit `anthropic`/`codex` credential calls) were not individually audited for allowlist interaction. `__main__.py:63-86` in particular is the site §7 actually modifies to gate credential prep and should be treated as in-scope for implementation even though it is not part of the `SUBSCRIPTION_BACKENDS` enumeration.
- **Persisted session-backend pins referencing an excluded backend** (`db.py` session rows written before a restart with a narrower `--backends`) are not given an explicit policy — whether the stale row is cleared, silently ignored, or surfaced. Implementation should not crash on this case; exact UX is deferred.
- **"First enabled backend"** (§6, §7, §8) is not pinned to a specific ordering. Implementation should use `_DECLARED_ORDER` (the same order `backend_names()` already returns) for consistency, but this ADR does not mandate it as a hard requirement.
- **`_enabled_backends` write-timing invariant is implied, not enforced.** §1 states `set_enabled_backends()` is called from `parse_args()`; it does not forbid a hypothetical later call from live server code, which would race unsynchronized accessor reads. In practice nothing in this codebase calls it outside startup, so this is a documentation gap, not a live bug.

## Scope

This ADR covers restricting which backends are available at runtime. It does not change discovery, the two-file convention, the plugin hook contract, or how a backend is selected among those enabled.
