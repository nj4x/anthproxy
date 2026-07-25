"""Canonical tier vocabulary and ranking for model identifiers.

Tier order: haiku=0 < sonnet=1 < opus=2 < fable=3.
Unknown / non-tier identifiers rank as None (fail-open per CLAUDE.md).
"""

TIER_RANK: dict[str, int] = {"haiku": 0, "sonnet": 1, "opus": 2, "fable": 3}


def classify_model_tier(model: str) -> str:
    """Case-insensitive substring match, highest tier first.

    Returns one of 'fable'|'opus'|'sonnet'|'haiku', else 'other'.
    Order matters: fable → opus → sonnet → haiku so a suffixed or
    full provider ID resolves to its coarse tier. Non-string inputs return 'other'.
    """
    if not isinstance(model, str):
        return "other"
    m = (model or "").lower()
    for tier in ("fable", "opus", "sonnet", "haiku"):
        if tier in m:
            return tier
    return "other"


def model_tier_rank(model: str) -> int | None:
    """Rank of a model's coarse tier, or None for unknown/non-tier IDs.

    None (not -1) preserves router fail-open: callers that must force an
    integer adapt it themselves (see admin adapter).
    """
    t = classify_model_tier(model)
    return TIER_RANK.get(t)  # 'other' → None
