"""
Disk cache for Actor outputs.

For the live demo we don't want a 60-second TikTok Shop scrape blocking the
audience. The first time we run the agent for a given (actor, input) pair we
hit Apify for real. The second time, we serve from disk in milliseconds.

Enabled by setting env var TRENDSTRIKE_USE_CACHE=1. With cache off, the
wrapper is a passthrough — production behaviour is unchanged.

Demo plan:
    1. Pre-run the agent for "fashion and apparel" / "US" once (warms cache)
    2. During the demo, cache hits = instant
    3. Tell judges honestly: "scrapes are cached; in production it runs daily"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
USE_CACHE_ENV = "TRENDSTRIKE_USE_CACHE"


def _enabled() -> bool:
    return os.environ.get(USE_CACHE_ENV) == "1"


def _key(actor_id: str, run_input: dict[str, Any]) -> str:
    """Hash actor_id + input to a stable filename."""
    safe_actor = actor_id.replace("/", "__")
    blob = json.dumps(run_input, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return f"{safe_actor}__{digest}.json"


async def cached_call_actor(
    real_call: Callable[[str, dict[str, Any]], Awaitable[list[dict[str, Any]]]],
    actor_id: str,
    run_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Wrap a `mcp.call_actor` invocation with read-through disk caching.

    real_call: the underlying coroutine (typically `mcp.call_actor`)
    actor_id, run_input: passed through unchanged

    Behaviour:
        cache disabled  → passthrough
        cache enabled, cold → call real, save result, return
        cache enabled, warm → load from disk, skip real call
    """
    if not _enabled():
        return await real_call(actor_id, run_input)

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / _key(actor_id, run_input)

    if path.exists():
        log.info("[cache HIT]  %s", path.name)
        return json.loads(path.read_text())

    log.info("[cache MISS] %s — calling Actor live", path.name)
    items = await real_call(actor_id, run_input)
    path.write_text(json.dumps(items, default=str))
    log.info("[cache SAVE] %s (%d items)", path.name, len(items))
    return items
