# Routing and Classification

The domain that governs how anthproxy selects a model tier for each incoming request. The classifier observes signals from the request and assigns a complexity label; routing maps that label to a concrete model.

## Language

**Tier**:
One of three named complexity bands — `trivial`, `standard`, `deep` — each mapped to a model alias (`haiku`, `sonnet`, `opus`). The router selects a tier; the tier resolves to a model.
_Avoid_: level, grade, class

**Tier Score**:
The numerical equivalent of a tier used in weighted combination: `trivial=0`, `standard=1`, `deep=2`. Never persisted directly; always derivable from the tier label.
_Avoid_: complexity score, numeric tier

**Weighted Tier Score**:
The blended float produced by combining the system-prompt tier score and the user-prompt tier score using their configured weights. Compared against two configurable thresholds to select the final tier.
_Avoid_: combined score, blended score

**User-Prompt Classification**:
The tier label produced by the LLM classifier (or rules engine) for the bounded `final_user_text` portion of the request. The dominant routing signal.
_Avoid_: request classification, prompt tier

**System-Prompt Classification**:
The tier label produced by a specialized LLM classifier call against a bounded preview of the system prompt. An auxiliary signal that reflects the complexity implied by the agent role or instructions.
_Avoid_: agent classification, system tier

**System-Prompt Tier Cache**:
An ephemeral in-memory dict keyed by `system_prompt_sha256`, mapping to `(tier_label, tier_score)`. Populated lazily on the first classifier call for a given system prompt SHA; evicted on server restart.
_Avoid_: system prompt cache, prompt classification cache

**Short Affirmation**:
A user message whose stripped text is a bare acknowledgement or continuation signal ("yes", "go ahead", "proceed"). Detected by `is_short_affirmation()`. Triggers the prior-response context path rather than classifying the affirmation text alone.
_Avoid_: continuation turn, acknowledgement

**Prior Response Summary**:
A bounded excerpt (tail-capped to a configurable char limit) of the last assistant message in `payload['messages']`, injected into the classifier input when a short affirmation is detected. Gives the classifier context about what the user is agreeing to.
_Avoid_: last response, assistant context

**Routing Summary**:
The bounded, provider-agnostic struct (`RoutingSummary`) extracted from a request payload and passed to the classifier. Contains `final_user_text`, message counts, tool counts, and — when applicable — `prior_response_summary`. Never contains system prompts, tool schemas, credentials, or full history text.
_Avoid_: classifier input, classification payload

**Context Key**:
The composite routing cache key `session_id + "\x00" + first_user_message_hash`. Used for the tier cache and the session-context size floor to isolate sub-agents from their parent sessions.
_Avoid_: session key, routing key

## Backend Selection

**Pace Delta**:
A backend's quota consumption relative to the linear passage of its own quota window: `burn% − elapsed%`. Negative means behind schedule (headroom to spare); positive means ahead of schedule. Comparable across windows of differing length.
_Avoid_: burn rate, utilization delta

**Enabled Backend Set**:
The backends an operator has made available for this deployment. Defaults to every backend the installation provides; an operator may narrow it. A backend outside the set does not exist as far as selection, switching, and credential preparation are concerned — it is not merely hidden from view.
_Avoid_: allowed backends, active backends, backend allowlist

**Self-Pace Gate**:
The absolute test `oauth_delta < 0` — is this backend behind its *own* schedule — evaluated before any cross-backend comparison. Distinct from the pace-delta *comparison*, which asks only which backend is further behind.
_Avoid_: pace check, budget gate

## Chaining

**Peer**:
Another anthproxy instance reached over its public Messages interface and treated as a backend. A peer is defined by what it exposes, not by where it runs — an instance on the same host is still a peer.
_Avoid_: remote, upstream proxy, parent proxy

**Peer Backend**:
The named backend that dispatches to a Peer. Ordinary in every respect that selection, pinning, enablement, and override care about; distinguished from a provider backend only in that its capacity belongs to another instance.
_Avoid_: proxy backend, relay backend

**Chain**:
Two or more instances linked by peer dispatch. Describes the topology, never a single instance's configuration.
_Avoid_: pipeline, cascade

**Hop**:
One instance's handling of a request within a chain, from inbound receipt to outbound dispatch. The unit of accounting, control, and configuration — each hop routes, records, and is configured on its own terms.
_Avoid_: leg, stage, link

**Classification Authority**:
The property of being the hop that performs model-tier classification for a request. Held by exactly one hop per request: the innermost one, i.e. the hop that dispatches to a provider rather than to a peer.
_Avoid_: routing owner, primary router

**Neutral Status**:
The capacity report a Peer Backend gives the selector — available, with utilization unknown. Distinct from *unavailable* (a veto) and from *known-idle* (an invitation); it makes the peer eligible but ranks it behind every backend whose real utilization is known.
_Avoid_: unknown status, default status
