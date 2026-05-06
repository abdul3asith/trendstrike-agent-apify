"""
TrendStrike Actor entry point.

Runs the autonomous TikTok Shop opportunity agent:
    1. Read input (category, country, max_opps, OpenAI key)
    2. Load prior briefing from KV store (cross-run memory)
    3. Run the orchestrator (3 sub-agents → synthesis)
    4. Save today's briefing to KV (for tomorrow)
    5. Render briefing as Markdown, store under KV key 'BRIEFING'
    6. Push one structured row to the Dataset for downstream consumers
       (Lovable webapp polls the Dataset to render in the UI)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from apify import Actor
from openai import AsyncOpenAI

from src.briefing import render_briefing
from src.memory import load_prior_briefing, save_briefing
from src.orchestrator import run_orchestrator


async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}

        category: str = actor_input.get("category") or "fashion and apparel"
        country: str = actor_input.get("country") or "US"
        max_opps: int = int(actor_input.get("max_opportunities") or 5)
        openai_key: str = (
            actor_input.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or ""
        )
        apify_token: str = os.environ.get("APIFY_TOKEN") or ""

        if not openai_key:
            raise ValueError(
                "OpenAI API key required. Pass via input.openai_api_key "
                "or env OPENAI_API_KEY."
            )
        if not apify_token:
            raise ValueError(
                "APIFY_TOKEN env var missing — set automatically inside Apify "
                "runtime; required for local runs."
            )

        Actor.log.info(
            "TrendStrike starting: category=%r country=%s max_opps=%d",
            category,
            country,
            max_opps,
        )

        # PHASE A — load yesterday's briefing for the diff
        prior = await load_prior_briefing(category)
        prior_opportunities = prior.get("opportunities", [])
        Actor.log.info("Memory: %d prior opportunities", len(prior_opportunities))

        # PHASE B — run the agent
        client = AsyncOpenAI(api_key=openai_key)
        result = await run_orchestrator(
            client=client,
            apify_token=apify_token,
            category=category,
            country=country,
            max_opps=max_opps,
            prior_opportunities=prior_opportunities,
            log=Actor.log,
        )
        opportunities = result["opportunities"]

        # PHASE C — render briefing
        briefing_md = render_briefing(category, country, opportunities)

        # PHASE D — persist for tomorrow's diff
        now_iso = datetime.now(timezone.utc).isoformat()
        await save_briefing(
            category=category,
            briefing={
                "date": now_iso,
                "category": category,
                "country": country,
                "opportunities": opportunities,
            },
        )

        # PHASE E — push to Dataset (one row per run, Lovable consumes this)
        await Actor.push_data(
            {
                "date": now_iso,
                "category": category,
                "country": country,
                "opportunities_count": len(opportunities),
                "top_pick": opportunities[0]["title"] if opportunities else None,
                "briefing_markdown": briefing_md,
                "opportunities": opportunities,
            }
        )

        # PHASE F — store rendered briefing under a stable KV key
        # so external callers (e.g. webhook → Lovable) can fetch directly
        await Actor.set_value("BRIEFING", briefing_md, content_type="text/markdown")

        Actor.log.info(
            "Done. %d opportunities surfaced. Briefing in KV under 'BRIEFING'.",
            len(opportunities),
        )


if __name__ == "__main__":
    asyncio.run(main())
