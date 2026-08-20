# ADR-0018: Remove session-key truncation and repair historical data

**Status:** Proposed

**Date:** 2026-08-17

## Problem

`anthproxy/handlers.py:320` truncates `metadata.user_id` to 128 chars via `return user_id[:128]`. Claude Code sends `{"device_id":"<64hex>","session_id":"<36uuid>","account_uuid":"<36uuid>"}` (~191 chars), so truncation cuts mid-field into `account_uuid`, destroying the JSON and severing the nested `session_id` — the only field distinguishing sessions.

**Immediate impact:**
- All requests from a single machine+account collapse into one session key
- 10,621 requests over 7 days merged into one session (conversation #309434 cluster)
- Session backend overrides leak across concurrent Claude Code windows on the same machine

**Historical impact:**
- 11,267 requests (708 distinct conversations) bundled into 7 collapsed session keys
- User session tracking and admin UI reporting corrupted

**Documented violation:**
`docs/agents/model-routing.md` states: "The session key is the full `metadata.user_id` JSON blob, not a bare UUID." The truncation violates this invariant without comment.

## Decision

1. **Remove truncation immediately.** Change `handlers.py:320` from `return user_id[:128]` to `return user_id`.
2. **Repair historical data.** Regroup the 11,267 truncated rows by `conversation_anchor` (708 distinct values) into separate sessions, assigning synthetic session keys composed of the truncated prefix + conversation anchor hash. Rebuild the sessions table to reflect the split.
3. **Update tests.** Replace `test_user_id_truncated_to_128` (which asserts the bug) with tests verifying full-blob preservation and that distinct metadata produces distinct keys. Add a regression test that per-session backend/routing overrides do not leak to concurrent sessions.

## Rationale

- **Compliance:** Restores adherence to the documented invariant.
- **Correctness:** Each Claude Code session becomes a distinct key; session overrides are isolated.
- **Data integrity:** 708 lost conversation boundaries are recoverable via `conversation_anchor` clustering without data loss.
- **Simplicity:** No hashing or parsing; the full blob is the key. TEXT columns handle ~191 chars without practical overhead in SQLite.

## Alternatives considered

- **Option B (hash the blob):** Hashes full metadata to 64-char sha256. Pros: fixed size. Cons: loses human-readability; historical data remains corrupt and diverges from the hash.
- **Option C (parse nested `session_id`):** Extract the UUID from the nested `session_id` field (36 chars). Pros: documented as a label field. Cons: contradicts the documented rule ("full JSON blob"); historical data remains corrupt.
- **Option D (increase limit):** Raise to 256 or higher. Pros: simpler. Cons: still truncates, violates the invariant, doesn't solve why truncation is needed.

Option A (this ADR) is chosen because it restores compliance with the documented invariant, fixes going forward, and enables full historical repair.

## Migration strategy

Two sequential tickets:

1. **Remove truncation (immediate):** Drop `[:128]`, update tests, deploy. Stops new collisions; new requests land in correct sessions and overrides isolate.
2. **Backfill historical sessions (follow-up):** Versioned migration in `db.py` regrouping the 11,267 rows. Requires no code changes to the handler; purely data reorganization. Demoable: admin UI reports 708 sessions instead of one 10k-row blob.

Both are deployed independently; slice 1 unblocks session correctness immediately.
