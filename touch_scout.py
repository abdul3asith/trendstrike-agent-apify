"""
Test the demo path: TrendScout + ProductHunter + CreatorAnalyst, serial, cached.

Run cold (warms cache):
    TRENDSTRIKE_USE_CACHE=1 python test_subagents.py

Run warm (instant):
    TRENDSTRIKE_USE_CACHE=1 python test_subagents.py
"""

import asyncio
import logging
import os
import sys
import time

from openai import AsyncOpenAI

from src.mcp_bridge import ApifyMcpBridge
from src.subagents import (
    run_creator_analyst,
    run_product_hunter,
    run_trend_scout,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


async def main() -> None:
    apify_token = os.environ.get("APIFY_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not apify_token or not openai_key:
        print("✗ Missing APIFY_TOKEN or OPENAI_API_KEY")
        sys.exit(1)

    cache_state = "ON" if os.environ.get("TRENDSTRIKE_USE_CACHE") == "1" else "OFF"
    print(f"\nCache: {cache_state}")

    client = AsyncOpenAI(api_key=openai_key)
    log = logging.getLogger("test")
    category = "fashion and apparel"
    country = "US"

    print(f"Running 3 sub-agents serially  category={category!r}, country={country}\n")

    t0 = time.perf_counter()
    async with ApifyMcpBridge(apify_token=apify_token) as mcp:
        scout = await run_trend_scout(client, mcp, category, country, log)
    async with ApifyMcpBridge(apify_token=apify_token) as mcp:
        hunter = await run_product_hunter(client, mcp, category, country, log)
    async with ApifyMcpBridge(apify_token=apify_token) as mcp:
        analyst = await run_creator_analyst(client, mcp, category, country, log)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"RESULT  ({elapsed:.1f}s total)")
    print("=" * 60)

    print(f"\nTrendScout      → {scout['actor_used']}")
    print(f"  trends:       {len(scout['trends'])}")
    for t in scout["trends"][:3]:
        print(f"    • {t.get('name')} — {t.get('commercial_angle', '')[:60]}")

    print(f"\nProductHunter   → {hunter['actor_used']}")
    print(f"  sub-categories: {len(hunter['products'])}")
    for p in hunter["products"][:3]:
        print(
            f"    • {p.get('sub_category')} ({p.get('price_band')}) — {p.get('saturation')} saturation"
        )

    print(f"\nCreatorAnalyst  → {analyst['actor_used']}")
    print(f"  creators:     {len(analyst['creators'])}")
    for c in analyst["creators"][:3]:
        rising = " 📈" if c.get("rising") else ""
        print(f"    • {c.get('handle')} ({c.get('followers')} followers){rising}")


if __name__ == "__main__":
    asyncio.run(main())
