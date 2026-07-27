# Fix classifier transparency field recording order in handlers.py

The classifier metadata fields (`classifier_model`, `classifier_summary_json`, `classifier_raw_response`, `classifier_format`, `classifier_confidence`) are never persisted to the database despite being populated in `ModelRoutingDecision`. Root cause: `_extract_prompt_capture()` reads `self._routing` to populate these fields, but it is called from `prepare_routing()` at line 823 (before `self._routing` is assigned at line 970). So `self._routing` contains the previous request's value (or `None`), and all classifier transparency fields are lost.

**Fix:** Pass the `routing` local variable from `prepare_routing()` into `_extract_prompt_capture()` as a parameter, rather than having it read `self._routing`. This requires:
1. Add `routing` parameter to `_extract_prompt_capture(self, payload, routing)`
2. Update the four assignments in `_extract_prompt_capture()` to use `getattr(routing, field, None)` where `field` is one of the four classifier fields; this preserves None-safety for potential future call sites even though the current call at line 823 guarantees a non-None `ModelRoutingDecision`.
3. Update the call at line 823 in `prepare_routing()` to pass `routing` as an argument

**Order of operations:** routing is computed (lines 693–780), then prompt capture is extracted (line 823) with routing as a parameter, then both are returned in `PreparedRequest`. When `prepare_routing()` returns, the handler assigns `self._routing = prepared.routing` (line 970) for downstream use; by then, the prompt capture fields have already been extracted with the correct routing data.

## Consequences

- Classifier metadata is now persisted to the database on every request where classification fires.
- The fix is purely internal to `handlers.py`; no config changes, no schema changes.
- Backward-compatible: pre-existing requests with NULL classifier fields are unaffected; the fix prevents future NULL writes.
- **Audit before implementing:** verify that `_extract_prompt_capture()` is called only from `prepare_routing()` at line 823. Search for all call sites (grep `_extract_prompt_capture` in `handlers.py`); if any other callers exist, update their signatures to pass the `routing` parameter as well or refactor to pass `routing` as a closure/context variable.
