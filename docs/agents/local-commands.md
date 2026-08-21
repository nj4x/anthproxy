# Routing and local commands

Local-command matching order and non-interactive backend switching.

See `README.md` and `config.py` for local-command syntax, runtime controls, and request-override behavior.

- Match local commands only after request-text wrappers are stripped and before credential parsing or inference dispatch.
- Backend switching is non-interactive: prepare candidates before commit, and do not commit a failed preparation.
- Installing or clearing session overrides must not alter global routing state. Pinned-session rate-limit failures return directly; override storage remains bounded.
- **Directives stop at the hop that receives them** (ADR-0024). A `proxy-*` command never crosses a peer dispatch — interception short-circuits before dispatch, which is intended rather than incidental — and no part of `X-Anthproxy-Override` is forwarded either, not even the subset that would parse at the peer. Consequence: a client cannot address the inner instance of a chain at all; configuring or querying it means talking to it directly. `prefer:peer` is the intended escape hatch — a directive naming the peer is meaningful at the outer hop and is honoured there.
