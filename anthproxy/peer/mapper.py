"""Request mapping for the peer backend.

A peer is another anthproxy instance speaking the native Anthropic Messages
wire format, so the body is copied as-is.  Two internal keys are handled, and
they are not the same kind of thing:

* ``_anthproxy_internal_classifier`` is an anthproxy sentinel — dropped.
* ``_anthropic_beta`` is the client's own ``anthropic-beta`` header lifted into
  the payload by the handler.  It is not a valid Messages field, so it is
  stripped from the body, but its value is re-emitted as an outbound
  ``anthropic-beta`` header: the peer speaks native Anthropic and can honour it.

Unlike ``local``, the requested model is transmitted exactly as received — no
alias lookup, no ``"default"`` substitution.  Resolving the model is the peer's
job (ADR-0021 §5).
"""

import json

_INTERNAL_KEYS = frozenset({'_anthropic_beta', '_anthproxy_internal_classifier'})


def _build_body(payload: dict) -> bytes:
    """Build the outbound request body for the peer, model verbatim."""
    body = {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}
    return json.dumps(body).encode('utf-8')


def _beta_header(payload: dict) -> str:
    """Return the comma-joined ``anthropic-beta`` value to forward, or ``''``."""
    raw = payload.get('_anthropic_beta') or []
    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, (list, tuple)):
        # A client may put this key in the request body itself, in which case it
        # reaches the mapper without passing through the handler's header lift.
        return ''
    seen: set[str] = set()
    betas: list[str] = []
    for beta in raw:
        if not isinstance(beta, str):
            continue
        item = beta.strip()
        if item and item not in seen:
            seen.add(item)
            betas.append(item)
    return ','.join(betas)
