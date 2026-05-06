"""
Orchestrator — the parent agent.

Runs the three sub-agents serially (memory-bounded by Apify free tier),
then performs one final synthesis LLM call that reads all three reports
plus yesterday's briefing and writes today's ranked opportunities.

Why serial fan-out instead of parallel?
    Apify free tier caps total concurrent Actor memory at 8 GB. Three
    Actors at 1 GB each plus one already running quickly trips that cap.
    Serial = predictable, demo-safe. Cache makes warm runs near-instant
    regardless of order.

Why GPT-5 only at synthesis?
    Sub-agents do mechanical work (pick an Actor, build an input, summarize
    raw items) — gpt-4.1 handles those at 2-5x the speed. The synthesis
    step reasons across three independent reports plus memory; that's
    where the reasoning model earns its weight.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from src.mcp_bridge import ApifyMcpBridge
from src.subagents import (
    run_creator_analyst,
    run_product_hunter,
    run_trend_scout,
)

# The orchestrator's synthesis call uses GPT-5; sub-agents already use
# gpt-4.1 internally (see src/subagents.py).
SYNTHESIS_MODEL = "gpt-5-mini"


SYNTHESIS_PROMPT = """You are TrendStrike, an autonomous research agent helping a TikTok Shop seller find emerging product opportunities.

Three sub-agents have already gathered intelligence:

▸ TrendScout — what's trending on TikTok in the seller's market
▸ ProductHunter — what's already on TikTok Shop in the seller's category, with prices, sales, ratings, saturation reads
▸ CreatorAnalyst — which creators are active in this space and rising

You also have the seller's PRIOR BRIEFING from a previous run. Do NOT re-surface opportunities that were already in the prior briefing UNLESS momentum has materially changed (e.g. mentions doubled, a major creator picked it up). When you reuse a prior opportunity, mark it `momentum_watch: true` and explain in `why_now` what changed.

Your job: pick the top {max_opps} *commercial* opportunities — concrete products this seller could plausibly add to their catalog this week. Filter aggressively. A trending dance is not an opportunity. A trending product format with rising creator coverage and existing-but-not-saturated TikTok Shop listings IS.

Respond with strict JSON:
{{"opportunities": [
  {{
    "title": "short product name",
    "category": "sub-category in the seller's space",
    "signal_strength": 1-10,
    "why_now": "one sentence on what's driving this trend right now",
    "evidence": ["2-4 short bullets citing specific data — creator names, view counts, prices, sales numbers from the reports above"],
    "price_band": "e.g. $28-$45 USD",
    "top_competitors": ["1-3 existing TikTok Shop sellers in this niche"],
    "recommended_action": "one concrete step the seller can take this week",
    "source_videos": ["TikTok URLs from the sub-agent data, max 3"],
    "momentum_watch": false
  }}
]}}

No prose, no markdown fences, only valid JSON."""


async def run_orchestrator(
    client: AsyncOpenAI,
    apify_token: str,
    category: str,
    country: str,
    max_opps: int,
    prior_opportunities: list[dict[str, Any]],
    log: Any,
) -> dict[str, Any]:
    """Run the full agent: 3 sub-agents serial, then synthesis."""

    log.info("=" * 60)
    log.info("PHASE 1: Sub-agent fan-out (serial)")
    log.info("=" * 60)

    # Each sub-agent gets its own MCP bridge — clean session lifecycle,
    # no risk of one cancelling another's context.
    async with ApifyMcpBridge(apify_token=apify_token) as mcp:
        scout = await run_trend_scout(client, mcp, category, country, log)
    async with ApifyMcpBridge(apify_token=apify_token) as mcp:
        hunter = await run_product_hunter(client, mcp, category, country, log)
    async with ApifyMcpBridge(apify_token=apify_token) as mcp:
        analyst = await run_creator_analyst(client, mcp, category, country, log)

    log.info("")
    log.info("Sub-agent results:")
    log.info("  TrendScout      → %d trends", len(scout.get("trends", [])))
    log.info("  ProductHunter   → %d sub-categories", len(hunter.get("products", [])))
    log.info("  CreatorAnalyst  → %d creators", len(analyst.get("creators", [])))

    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2: Load prior briefing (cross-run memory)")
    log.info("=" * 60)
    log.info("  Prior briefing: %d opportunities", len(prior_opportunities))

    log.info("")
    log.info("=" * 60)
    log.info("PHASE 3: Synthesis with %s", SYNTHESIS_MODEL)
    log.info("=" * 60)

    synthesis_input = {
        "seller_category": category,
        "country": country,
        "trend_scout_report": scout,
        "product_hunter_report": hunter,
        "creator_analyst_report": analyst,
        "prior_briefing_opportunities": prior_opportunities,
    }

    response = await client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIS_PROMPT.format(max_opps=max_opps),
            },
            {
                "role": "user",
                "content": json.dumps(synthesis_input, default=str)[:60000],
            },
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Synthesis returned invalid JSON: %s\n%s", e, raw[:500])
        return {
            "opportunities": [],
            "sub_agent_reports": {
                "trends": scout,
                "products": hunter,
                "creators": analyst,
            },
        }

    opportunities = parsed.get("opportunities", [])[:max_opps]

    # Tag momentum_watch by checking against prior list — belt-and-suspenders
    # for the LLM's own self-tagging
    prior_titles = {(o.get("title") or "").lower().strip() for o in prior_opportunities}
    for opp in opportunities:
        if (opp.get("title") or "").lower().strip() in prior_titles:
            opp["momentum_watch"] = True

    log.info("  ✓ Synthesized %d opportunities", len(opportunities))
    log.info(
        "  ✓ %d are momentum_watch (recurring from prior briefing)",
        sum(1 for o in opportunities if o.get("momentum_watch")),
    )

    return {
        "opportunities": opportunities,
        "sub_agent_reports": {
            "trends": scout,
            "products": hunter,
            "creators": analyst,
        },
    }
