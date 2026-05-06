"""
Cross-run memory backed by the Actor's Key-Value Store.

For each (category) we persist the most recent briefing. On the next run for
the same category, the orchestrator loads it and the synthesis LLM diffs
against it — opportunities that recur get tagged `momentum_watch` instead of
being re-surfaced as if they were new.

Keys are slugified from category to keep them KV-safe (no spaces, slashes,
or quotes).
"""

from __future__ import annotations

import re
from typing import Any

from apify import Actor


def _key(category: str) -> str:
    """Sanitize an arbitrary category string into a KV-store-safe key."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", category.strip().lower()).strip("_")
    return f"briefing_{slug or 'default'}"


async def load_prior_briefing(category: str) -> dict[str, Any]:
    """Return the most recent briefing for this category, or {} if none."""
    value = await Actor.get_value(_key(category))
    return value if isinstance(value, dict) else {}


async def save_briefing(category: str, briefing: dict[str, Any]) -> None:
    """Persist today's briefing so the next run can diff against it."""
    await Actor.set_value(_key(category), briefing)
