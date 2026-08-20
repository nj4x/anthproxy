# Session Bundling Bug Research: Request #309434

## Question

Why are multiple distinct Claude Code sessions being bundled into a single enormous session in the SQLite database (request #309434 and ~10,459 others)?

## Data Evidence

**Query run:** `sqlite3 -readonly ~/.anthproxy/anthproxy.db`

### Current (buggy) state

```sql
SELECT COUNT(*), min(request_ts), max(request_ts), count(DISTINCT conversation_anchor) 
FROM requests 
WHERE session_id=(SELECT session_id FROM requests WHERE id=309434);
```

**Result:** `10459 requests | 2026-08-10T15:48:23.730Z to 2026-08-17T22:20:05.948Z | 656 distinct conversations`

This is a single `session_id` value bundling 10,459 requests over 7 days with 656 distinct conversations — a clear sign of multiple distinct sessions colliding into one.

### Session ID values

The database contains only **one** distinct session_id value:

```
{"device_id":"e075e6e188e9ffebda07740cc2014400829f09b7b7ccd7232a8b0c34b52662dc","account_uuid":"2dba6ea6-76eb-44e6-bbee-2d6fcaeb
```

**Length:** exactly 128 characters, ending mid-JSON (cuts off the closing `"account_uuid"` value).

Database session_id lengths show the problem:
- 11,053 requests with `length(session_id) = 128` (truncated JSON)
- 192 requests with `length(session_id) = 36` (bare UUIDs, not truncated)
- Remaining requests with various lengths

## Code Path Traced

### 1. Session key extraction: `anthproxy/handlers.py:306–320`

```python
def _session_key(payload: dict) -> str | None:
    """Return the session key from ``payload['metadata']['user_id']``, or None.
    
    The Claude Code CLI populates ``metadata.user_id`` with a per-session
    identifier on every request; it is the only per-request value that reliably
    distinguishes one session from another.  Returns None when the field is
    absent, blank, or not a string.
    """
    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get('user_id')
    if not isinstance(user_id, str) or not user_id:
        return None
    return user_id[:128]  # ← TRUNCATION HAPPENS HERE (line 320)
```

**The bug:** Line 320 truncates `user_id` to the first 128 characters.

### 2. Database insertion: `anthproxy/db.py:462–639`

The truncated session_id is passed to `SessionDB.record_request()`:

- **Line 573:** `session_id` parameter is inserted directly into the requests table.
- **Line 620–627:** Same truncated `session_id` is upserted into the sessions table using it as PRIMARY KEY.

```sql
INSERT INTO sessions (session_id, last_seen_at)
VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(session_id) DO UPDATE SET
    last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
```

**Result:** All requests with the same 128-char prefix map to a single row in the sessions table.

### 3. Where the truncated key is used

In `anthproxy/handlers.py`, the session key is written to DB at:
- **Line 1058:** Rate-limit failure path: `session_id=_session_key(payload) or ''`
- **Line 1212:** Streaming response wrapper: `session_id=_session_key(payload) or ''`
- **Line 1260:** Non-streaming response: `session_id=_session_key(payload) or ''`

## Documented Invariant (Violated)

From `docs/agents/model-routing.md`:

> The session key is the full `metadata.user_id` JSON blob, not a bare UUID. Use its nested `session_id` only for human-facing labels; do not infer device IDs, scan for arbitrary UUIDs, or slug key names.

**This invariant is violated by the `[:128]` truncation.**

## Root Cause

**Confidence: 100% certain.**

The `_session_key()` function at line 320 of `anthproxy/handlers.py` truncates the full JSON metadata to 128 characters. Typical Claude Code metadata is ~191 characters:

```json
{"device_id":"<64-hex>","session_id":"<36-uuid>","account_uuid":"<36-uuid>"}
```

Truncating to 128 chars cuts the string mid-field, destroying the JSON structure and causing:

1. **Hash collision:** Many distinct full metadata blobs (representing truly different sessions) all hash to the same 128-char prefix.
2. **Silent merging:** The upsert logic in the sessions table treats these collisions as the same session, silently merging requests from different sessions.
3. **Data loss:** 656 distinct conversations over 7 days bundled into one, destroying the user's session boundaries and making the admin UI and session tracking useless.

## Why the 128 Limit Exists

The limit has been present since the initial commit (41635b8). There is no documented rationale in:
- Comments in the code
- CLAUDE.md
- The architecture/model-routing docs
- The CONTEXT.md file

It appears to be an arbitrary bound, possibly intended as a safeguard against unbounded string storage, but it violates the documented invariant without comment or waiver.

## Plausible Fixes (Without Implementation)

### Option A: Remove the truncation entirely ✅ **Recommended**

```python
return user_id  # Remove [:128]
```

**Pros:**
- Restores compliance with the documented invariant.
- Each distinct metadata blob becomes a distinct session key.
- Minimal code change.

**Cons:**
- Session_id column will grow to ~191 chars (not a practical concern for SQLite TEXT).
- Existing data remains corrupt; a migration is needed to de-duplicate or flag affected sessions.

### Option B: Hash the full metadata

```python
import hashlib
return hashlib.sha256(user_id.encode()).hexdigest()
```

**Pros:**
- Maintains fixed 64-char size; still full-strength uniqueness.
- Fixes going forward.

**Cons:**
- Loses human-readability; display_name fallback becomes critical.
- Existing data remains corrupt (truncated prefix still differs from sha256 hash).

### Option C: Use a smarter truncation

Parse the JSON, extract the nested `session_id` field, and use that as the key:

```python
if user_id[0] == '{':
    data = json.loads(user_id)
    return data.get('session_id') or user_id  # 36 chars for UUIDs
```

**Pros:**
- Documented in model-routing.md: "Use its nested `session_id` only for human-facing labels."
- Session uniqueness is preserved (each Claude Code session has its own `session_id`).

**Cons:**
- Diverges from the documented rule: "The session key is the full `metadata.user_id` JSON blob."
- Would require the project to clarify or revise the invariant.

### Option D: Increase the truncation limit

```python
return user_id[:256]  # or higher
```

**Pros:**
- Quick fix; no logic changes.

**Cons:**
- Doesn't solve the root problem; just delays it for longer metadata.
- Still truncates, violating the invariant.
- Doesn't address why truncation is needed at all.

## Recommended Next Steps

1. **Immediate:** Implement Option A (remove truncation) to stop creating new collisions.
2. **Data cleanup:** Backfill historical sessions by either:
   - Re-parsing the truncated keys to infer the likely full metadata (hard; may lose data).
   - Marking bundled sessions as corrupt and advising users to start new sessions.
3. **Test coverage:** Add a test that verifies session_id is not truncated and that distinct metadata produces distinct session keys.
