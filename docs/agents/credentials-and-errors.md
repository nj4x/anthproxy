# Credentials and error handling

Credential isolation/atomicity and client-facing failure shaping.

- Keep subscription credentials separate from `~/.claude` to avoid refresh-token races and CLI logout.
- Credential writes must be atomic, with cross-process refresh serialization where supported.
- Distinguish interactive startup credential setup from non-interactive runtime readiness checks.
- All client-facing failures use Anthropic error envelopes.
- SSE response headers are committed before upstream priming. A pre-stream upstream failure is therefore emitted as an in-band SSE error, not an HTTP retryable status; preserve keepalive behavior while priming.
