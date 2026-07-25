# Routing and local commands

Local-command matching order and non-interactive backend switching.

See `README.md` and `config.py` for local-command syntax, runtime controls, and request-override behavior.

- Match local commands only after request-text wrappers are stripped and before credential parsing or inference dispatch.
- Backend switching is non-interactive: prepare candidates before commit, and do not commit a failed preparation.
- Installing or clearing session overrides must not alter global routing state. Pinned-session rate-limit failures return directly; override storage remains bounded.
