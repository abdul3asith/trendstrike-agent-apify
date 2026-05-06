"""
Verification test for the rewritten MCP bridge.

Run: python test_bridge.py
"""

import asyncio
import logging
import os
import sys

from src.mcp_bridge import ApifyMcpBridge

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


async def main() -> None:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        print("✗ APIFY_TOKEN not set")
        sys.exit(1)

    print("\n=== TEST 1: Connect ===")
    async with ApifyMcpBridge(apify_token=token) as mcp:
        print("✓ Connected")

        print("\n=== TEST 2: list_tools ===")
        tools = await mcp.list_tools()
        print(f"✓ Server exposes {len(tools)} tools: {tools}")
        # Sanity-check the three we'll rely on
        for needed in ("search-actors", "fetch-actor-details", "call-actor"):
            assert needed in tools, f"Missing required tool: {needed}"
        print("✓ All required tools present")

        print("\n=== TEST 3: search_actors_markdown ===")
        md = await mcp.search_actors_markdown("tiktok shop", limit=5)
        print(f"✓ Got {len(md)} chars of Markdown")
        print("First 400 chars:")
        print("  " + md[:400].replace("\n", "\n  "))

        print("\n=== TEST 4: extract_actor_ids ===")
        ids = ApifyMcpBridge.extract_actor_ids(md)
        print(f"✓ Extracted {len(ids)} actor IDs:")
        for aid in ids:
            print(f"  • {aid}")
        if not ids:
            print("⚠ No IDs extracted — regex may need tuning")
            return

        print("\n=== TEST 5: fetch_actor_details ===")
        chosen = ids[0]
        details_md = await mcp.fetch_actor_details(chosen)
        print(f"✓ Got {len(details_md)} chars of details for {chosen}")
        print("First 400 chars:")
        print("  " + details_md[:400].replace("\n", "\n  "))

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
