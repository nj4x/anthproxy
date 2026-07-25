# Concurrency and selector

Lock-scope discipline and selector behavior invariants.

- `BackendRegistry._state_lock` and selector locks guard only short state transitions. Never hold them across inference, streaming, network I/O, OAuth refresh, usage lookup, backend construction, or candidate preparation. `_prepare_lock` serializes slow candidate preparation.
- Do not describe selector behavior as a fixed priority or drift-prone algorithm summary, and do not claim that token refresh runs independently.
- Preserve incumbent-hold and exhaustion-parking ordering. Unknown utilization remains neutral, and transient usage-fetch gaps do not remove incumbency. Subscription-only operation skips parking.
