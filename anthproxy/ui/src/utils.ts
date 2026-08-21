export interface SessionLabel {
  label: string;
  tooltip: string;
}

/** Mirrors anthproxy/constants.py:UNTRACKED_SESSION_ID. */
export const UNTRACKED_SESSION_ID = '__untracked__';

const BACKEND_LABELS: Record<string, string> = {
  oauth: 'Anthropic-OAuth',
};

/**
 * Human-facing label for a stored backend name.  The Anthropic OAuth token is
 * persisted as the bare backend name "oauth"; operators see it as "Anthropic-OAuth".
 * Unknown names pass through unchanged.
 */
export function backendLabel(name: string | null | undefined): string {
  if (name == null) return '';
  return BACKEND_LABELS[name] ?? name;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SESSION_PREFIX_LENGTH = 8;
const RAW_LABEL_MAX_LENGTH = 20;

function jsonSessionId(value: string): string | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      'session_id' in parsed &&
      typeof parsed.session_id === 'string' &&
      parsed.session_id.trim()
    ) {
      return parsed.session_id.trim();
    }
  } catch {
    // Non-JSON session identifiers are handled below.
  }
  return null;
}

function compactRawLabel(value: string): string {
  return value.length > RAW_LABEL_MAX_LENGTH
    ? `${value.slice(0, RAW_LABEL_MAX_LENGTH)}…`
    : value;
}

/**
 * Produces a readable, compact label without losing the persisted identifier.
 * The tooltip always retains the exact raw value for inspection and copying.
 * Special-cases the __untracked__ sentinel for requests with no metadata.user_id.
 */
export function parseSessionId(value: string): SessionLabel {
  if (value === UNTRACKED_SESSION_ID) {
    return { label: '— Untracked —', tooltip: 'Requests with no session ID (e.g., system utility calls)' };
  }

  const nestedSessionId = jsonSessionId(value);
  if (nestedSessionId) {
    return { label: nestedSessionId.slice(0, SESSION_PREFIX_LENGTH), tooltip: value };
  }

  if (UUID_RE.test(value)) {
    return { label: value.slice(0, SESSION_PREFIX_LENGTH), tooltip: value };
  }

  return { label: compactRawLabel(value), tooltip: value };
}

/**
 * Labels serialized context keys as "session · anchor" while leaving the
 * legacy standalone hash anchors unchanged.
 */
export function parseConversationAnchor(value: string): SessionLabel {
  const separator = value.lastIndexOf('\0');
  if (separator > 0 && separator < value.length - 1) {
    const sessionId = value.slice(0, separator);
    const anchor = value.slice(separator + 1);
    if (jsonSessionId(sessionId) || UUID_RE.test(sessionId)) {
      return {
        label: `${parseSessionId(sessionId).label} · ${anchor}`,
        tooltip: value,
      };
    }
  }

  return { label: value, tooltip: value };
}
