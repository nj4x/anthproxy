# Credentials and error handling

Credential isolation/atomicity and client-facing failure shaping.

- Keep subscription credentials separate from `~/.claude` to avoid refresh-token races and CLI logout.
- Credential writes must be atomic, with cross-process refresh serialization where supported.
- Distinguish interactive startup credential setup from non-interactive runtime readiness checks.
- All client-facing failures use Anthropic error envelopes.
- SSE response headers are committed before upstream priming. A pre-stream upstream failure is therefore emitted as an in-band SSE error, not an HTTP retryable status; preserve keepalive behavior while priming. That error frame is written out of band, so the stream itself carries no failure marker — the recording path must capture the raised upstream error, or a request that returned nothing is recorded as a zero-cost success.
- Every failed dispatch leaves a request row once a backend was involved, for every backend (ADR-0025). Usage that was never learned is recorded as absent, not zero: pass an empty `stats_dict` so token columns and `cost_estimate` are NULL. A pre-dispatch client error (malformed JSON, bad content type) has no backend to attribute and still records nothing.
