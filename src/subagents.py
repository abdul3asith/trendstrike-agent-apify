"""
Sub-agents.

Each sub-agent is a small LLM-driven loop with one focused task and one Actor
it uses. They run in parallel from the orchestrator, each with its own
context — this is the "delegate to sub-agents" pattern Apify documents in
their MCP and OpenClaw integration guides.

Sub-agent flow (same pattern for all three):
    1. Search MCP for a relevant Actor (returns Markdown)
    2. LLM reads the Markdown and picks the best candidate
    3. Fetch Actor details (Markdown — includes input schema)
    4. LLM constructs a valid input object for the Actor
    5. Call the Actor via MCP, get raw dataset items
    6. LLM summarizes the raw items into a structured report

Currently implemented:
    - run_trend_scout — finds trending hashtags, sounds, videos in the seller's market
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from src.mcp_bridge import ApifyMcpBridge

# We tag the LLM model in one place so the orchestrator can override it.
DEFAULT_MODEL = "gpt-5"

# How many raw items we feed back to the LLM for summarization.
# Larger = better summaries, but more tokens. 50 is a sweet spot.
MAX_RAW_ITEMS_FOR_SUMMARY = 50


# ============================================================================
# Prompts. Kept at module level so they're easy to read and tune.
# ============================================================================

ACTOR_PICKER_PROMPT = """You are an autonomous research agent. You just searched the Apify Store for an Actor that can complete a sub-task.

Sub-task: {task}
Seller context: category="{category}", country="{country}"

Below is the Markdown response from the Apify Store with candidate Actors:

---
{search_md}
---

Pick the SINGLE Actor best suited for the sub-task. Prefer:
- Actors whose name and description directly match the task
- Higher-rated, more-used Actors (when stats are visible)
- Actors that mention exactly the data shape you need

You MUST pick from the candidates above. Do not invent actor IDs.

Respond with strict JSON only:
{{"actor_id": "username/name", "rationale": "one sentence"}}"""


INPUT_BUILDER_PROMPT = """You are configuring an Apify Actor for a sub-task.

Sub-task: {task}
Seller context: category="{category}", country="{country}"

Below is the Actor's full information (Markdown — README, input schema, pricing):

---
{actor_md}
---

Build a JSON object that satisfies the Actor's input schema and will return useful results for the sub-task. Be concrete: pick keywords, hashtags, or country codes that fit the seller's category. Keep result counts conservative (10-30 items) to control cost. If the schema requires a field you don't have a value for, use a sensible default.

Respond with strict JSON only — the input object itself, no wrapper."""


SUMMARIZER_PROMPT = """You are TrendScout. You ran a TikTok-related Actor and got back raw data items.

Seller category: {category}
Country: {country}

Filter the raw data down to trends that are PLAUSIBLY commercial — i.e. they could plausibly map to a product the seller might add to their TikTok Shop catalog. Aggressively ignore pure dance, music, comedy, or meme trends with no product angle.

For each surviving trend, capture:
- name (the hashtag or sound or video theme)
- type ("hashtag" | "sound" | "video_theme")
- evidence (one short sentence — what makes this commercial?)
- view_count (integer, or null if unknown)
- url (string, or null)
- commercial_angle (what kind of product this could become)

Respond with strict JSON only:
{{"trends": [{{...}}, ...]}}

Cap at 20 trends. If the raw data has nothing commercial, return {{"trends": []}}."""


# ============================================================================
# Helpers
# ============================================================================


async def _llm_json(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_chars: int = 60_000,
) -> dict[str, Any]:
    """Call the LLM expecting a JSON object back."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt[:max_chars]}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")


# ============================================================================
# TrendScout
# ============================================================================


async def run_trend_scout(
    client: AsyncOpenAI,
    mcp: ApifyMcpBridge,
    category: str,
    country: str,
    log: Any,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """TrendScout: find commercially relevant trends for the seller's category."""
    task = f"Find trending hashtags or sounds in {country} on TikTok that relate to {category} products."

    # Step 1: search MCP for candidate Actors
    log.info("[TrendScout] searching for trends Actor")
    search_md = await mcp.search_actors_markdown(
        keywords="tiktok trends hashtag", limit=8
    )
    valid_ids = ApifyMcpBridge.extract_actor_ids(search_md)
    if not valid_ids:
        raise RuntimeError("[TrendScout] no candidate Actors found in search")
    log.info("[TrendScout] candidates: %s", valid_ids)

    # Step 2: LLM picks the best one
    pick = await _llm_json(
        client,
        model,
        ACTOR_PICKER_PROMPT.format(
            task=task, category=category, country=country, search_md=search_md
        ),
    )
    chosen = pick.get("actor_id", "")
    if chosen not in valid_ids:
        # LLM hallucinated. Fall back to first valid candidate.
        log.warning("[TrendScout] LLM picked invalid id %r, falling back", chosen)
        chosen = valid_ids[0]
    log.info("[TrendScout] chose %s — %s", chosen, pick.get("rationale", ""))

    # Step 3: fetch the chosen Actor's details (schema, README)
    actor_md = await mcp.fetch_actor_details(chosen)

    # Step 4: LLM builds the run input
    run_input = await _llm_json(
        client,
        model,
        INPUT_BUILDER_PROMPT.format(
            task=task, category=category, country=country, actor_md=actor_md
        ),
    )
    log.info("[TrendScout] run input keys: %s", list(run_input.keys()))

    # Step 5: actually run the Actor
    log.info("[TrendScout] calling %s", chosen)
    raw_items = await mcp.call_actor(chosen, run_input)
    log.info("[TrendScout] got %d raw items", len(raw_items))
    if not raw_items:
        return {"actor_used": chosen, "trends": []}

    # Step 6: LLM summarizes raw → structured trends
    summary_input = json.dumps(raw_items[:MAX_RAW_ITEMS_FOR_SUMMARY], default=str)
    summary = await _llm_json(
        client,
        model,
        SUMMARIZER_PROMPT.format(category=category, country=country)
        + "\n\nRAW DATA:\n"
        + summary_input,
        max_chars=50_000,
    )
    trends = summary.get("trends", [])
    log.info("[TrendScout] surfaced %d commercial trends", len(trends))

    return {"actor_used": chosen, "trends": trends}
