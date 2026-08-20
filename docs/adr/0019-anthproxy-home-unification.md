# ADR-0019: Unify configuration and state under ANTHPROXY_HOME

**Status:** Proposed

**Date:** 2026-08-19

## Problem

anthproxy state is scattered across four independent home-relative directories, each with its own hardcoded default:

| Path | Contents | Owner |
| --- | --- | --- |
| `~/.anthproxy/` | `config.json`, `anthproxy.db`, `stats/` | anthproxy |
| `~/.anthropic/` | `auth.json` (subscription OAuth credentials) | anthproxy |
| `~/.bedrock/` | `credentials.json`, `token-estimator.json` | anthproxy |
| `~/.codex/` | Codex CLI credentials | **shared with the real Codex CLI** |

Three of the four are anthproxy-owned but live in unrelated locations. Consequences:

- Relocating anthproxy state (containers, multi-instance setups, non-`$HOME` deployments, backup/restore) requires overriding four separate paths rather than one.
- `~/.anthropic` and `~/.bedrock` are top-level dotdirs whose names imply vendor ownership, not anthproxy ownership.
- `StatsCollector` resolves its own stats directory internally, so `Config` is not the single source of truth for path resolution.

`~/.codex` is genuinely shared with an external tool. Redirecting it would either break the Codex CLI or desynchronize its credential refresh.

## Decision

1. **Introduce a single root.** Add `ANTHPROXY_HOME` (env var) and `--anthproxy-home` (CLI flag), defaulting to `~/.anthproxy`. All anthproxy-owned paths resolve relative to it.

2. **Relocate anthproxy-owned directories as subdirectories:**
   - `~/.anthropic/auth.json` → `$ANTHPROXY_HOME/anthropic/auth.json`
   - `~/.bedrock/*` → `$ANTHPROXY_HOME/bedrock/*`
   - `config.json`, `anthproxy.db`, `stats/` remain in place (now expressed relative to `ANTHPROXY_HOME`).

3. **Leave `~/.codex` untouched.** It is co-owned by an external tool and is out of scope for unification.

4. **Add `stats_dir` to `Config`** and thread it through to `StatsCollector`, so path resolution lives in one place instead of being rediscovered by the consumer.

5. **Migrate manually, not automatically.** Add `anthproxy migrate --dry-run` (report the planned moves) and `anthproxy migrate` (perform them). Startup never mutates the filesystem on the user's behalf.

6. **Keep `ANTHPROXY_CONFIG` as a file-level override.** It points at a specific model-configuration file rather than a directory; leaving it unchanged preserves backward compatibility for existing users without adding a second directory-resolution mechanism.

## Consequences

**Relocation becomes a single knob.** Setting `ANTHPROXY_HOME` moves everything anthproxy owns. Containerized and multi-instance deployments no longer need to coordinate four environment variables.

**Old paths remain readable, never writable.** If a new path is absent and the corresponding legacy path exists, anthproxy reads the legacy path. Writes always target the new layout. This follows the precedent already set by `stats.jsonl` handling, and means an un-migrated install keeps working while a migrated one never writes back into `~/.anthropic` or `~/.bedrock`.

**The credentials-separation invariant is preserved.** `docs/agents/credentials-and-errors.md` mandates: *"Keep subscription credentials separate from `~/.claude` to avoid refresh-token races and CLI logout."* Retaining distinct `anthropic/` and `bedrock/` subdirectories — rather than flattening credentials into `$ANTHPROXY_HOME` root — keeps anthproxy's OAuth store structurally distinct from the Claude CLI's, so the two never contend over the same refresh token and anthproxy activity cannot log the CLI out. The unification changes *where* the credential tree is rooted, not *whether* it is separate.

**Manual migration keeps the user in control.** No surprise filesystem mutations on upgrade; `--dry-run` makes the change auditable before it happens. The cost is that users must run one command to get the unified layout — acceptable given the backward-compatible read path.

**Tests pass explicit paths.** Tests that previously relied on default path resolution now pass explicit directories. This removes hidden `$HOME` coupling from the suite and prevents tests from reading or writing real user state.

**`~/.codex` remains an exception.** The layout is "unified except for genuinely shared external state" rather than fully unified. This is a deliberate, documented carve-out, not an oversight.

## Alternatives considered

- **Auto-migrate on startup.** Rejected: silently moving credential files during a routine upgrade risks data loss and violates the principle that startup does not mutate user state. Failure mid-move would leave credentials in an indeterminate location.
- **Flatten credentials into `$ANTHPROXY_HOME` root.** Rejected: blurs the credential-isolation boundary the credentials-and-errors constraint depends on, and creates filename collision risk between providers.
- **XDG-style split (`$XDG_CONFIG_HOME` for config, `$XDG_STATE_HOME` for db/stats).** Deferred, not rejected. It is the more standards-conformant layout, but it doubles the number of roots users must reason about and complicates relocation — the exact problem this ADR solves. A single directory is the simpler first step; an XDG split can layer on later, since `ANTHPROXY_HOME` would remain the fallback for both roots.
- **Redirect `~/.codex` too.** Rejected: breaks or desynchronizes the external Codex CLI.

## Scope

This ADR covers the first part of configuration consolidation: establishing a single root and relocating anthproxy-owned paths beneath it. Future work may introduce an XDG-style config/state split if the single-directory model proves limiting.
