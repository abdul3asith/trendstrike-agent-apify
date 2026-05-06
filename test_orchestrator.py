"""
Test the full orchestrator: 3 sub-agents → synthesis → briefing.

Run cold (warms cache):
    TRENDSTRIKE_USE_CACHE=1 python test_orchestrator.py

Run warm (fast):
    TRENDSTRIKE_USE_CACHE=1 python test_orchestrator.py
"""

import asyncio
import json
import logging
import os
import sys
import time

from openai import AsyncOpenAI

from src.orchestrator import run_orchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


async def main() -> None:
    apify_token = os.environ.get("APIFY_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not apify_token or not openai_key:
        print("✗ Missing APIFY_TOKEN or OPENAI_API_KEY")
        sys.exit(1)

    cache_state = "ON" if os.environ.get("TRENDSTRIKE_USE_CACHE") == "1" else "OFF"
    print(f"\nCache: {cache_state}\n")

    client = AsyncOpenAI(api_key=openai_key)
    log = logging.getLogger("test")

    category = "fashion and apparel"
    country = "US"
    max_opps = 5

    # Fake "prior briefing" so we can see momentum_watch tagging in action.
    # Real production reads this from Apify KV store.
    prior = [
        {"title": "Cargo skirts in earth tones"},
        {"title": "Slouchy cropped denim jackets"},
    ]

    t0 = time.perf_counter()
    result = await run_orchestrator(
        client=client,
        apify_token=apify_token,
        category=category,
        country=country,
        max_opps=max_opps,
        prior_opportunities=prior,
        log=log,
    )
    elapsed = time.perf_counter() - t0

    opps = result["opportunities"]

    print("\n" + "=" * 70)
    print(f"BRIEFING  ({elapsed:.1f}s total)")
    print("=" * 70)
    print(f"Category: {category}  •  Region: {country}")
    print(f"{len(opps)} opportunities surfaced.\n")

    for i, opp in enumerate(opps, 1):
        watch = " 🔁 momentum watch" if opp.get("momentum_watch") else ""
        print(f"{i}. {opp.get('title')}{watch}")
        print(
            f"   Signal: {opp.get('signal_strength')}/10  •  Price: {opp.get('price_band', '?')}"
        )
        print(f"   Why now: {opp.get('why_now', '?')}")
        for ev in (opp.get("evidence") or [])[:3]:
            print(f"     - {ev}")
        if opp.get("recommended_action"):
            print(f"   ▶ {opp['recommended_action']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
