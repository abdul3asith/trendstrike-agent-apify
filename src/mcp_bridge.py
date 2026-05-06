"""
Apify MCP bridge.

Connects to Apify's hosted MCP server at https://mcp.apify.com using the
official MCP Python SDK over Streamable HTTP. This is the only file in
the project that talks to the outside world directly.

Important things we learned from probing the live server (protocol 2025-11-25):
    - The search-actors tool takes `keywords` (not `search`) as its arg.
    - It returns ONE TextContent block formatted as Markdown, not JSON.
      We pass that Markdown to the LLM verbatim — it's designed to be read.
    - call-actor's response shape varies; we handle several.

Exposes:
    list_tools() — sanity check for what's available
    search_actors_markdown(keywords) — raw Markdown for the LLM to reason over
    extract_actor_ids(markdown) — pull "username/name" tokens out as a fallback
    fetch_actor_details(actor_id) — Markdown + (when present) input schema
    call_actor(actor_id, run_input) — run an Actor, return dataset items
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

APIFY_MCP_URL = "https://mcp.apify.com"

log = logging.getLogger(__name__)

# Regex for "username/name" actor IDs in Markdown like `lemur/tiktok-shop-creators`
_ACTOR_ID_PATTERN = re.compile(r"`([a-zA-Z0-9][\w.-]*\/[a-zA-Z0-9][\w.-]*)`")


class ApifyMcpBridge:
    """Async context-managed MCP client for the Apify hosted server."""

    def __init__(self, apify_token: str) -> None:
        self._token = apify_token
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "ApifyMcpBridge":
        self._stack = AsyncExitStack()
        headers = {"Authorization": f"Bearer {self._token}"}
        transport = await self._stack.enter_async_context(
            streamablehttp_client(APIFY_MCP_URL, headers=headers)
        )
        read_stream, write_stream, _ = transport
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        log.info("Connected to mcp.apify.com")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.aclose()

    async def _call_tool_raw(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool and return joined text from all content blocks.

        We don't try to JSON-parse — Apify's tools return Markdown for human
        and LLM consumption. Callers can JSON-parse if they know the tool
        returns JSON (call-actor outputs do).
        """
        assert self._session is not None, "Bridge not entered as context manager"
        result = await self._session.call_tool(name, arguments=arguments)
        if result.isError:
            raise RuntimeError(f"MCP tool {name} returned error: {result.content!r}")
        text_parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()

    async def list_tools(self) -> list[str]:
        """List the names of every tool the server currently exposes."""
        assert self._session is not None
        tools = await self._session.list_tools()
        return [t.name for t in tools.tools]

    async def search_actors_markdown(self, keywords: str, limit: int = 5) -> str:
        """Search Apify Store. Returns Markdown — feed it to the LLM.

        Server expects `keywords`, NOT `search`. Verified against
        protocol version 2025-11-25.
        """
        return await self._call_tool_raw(
            "search-actors", {"keywords": keywords, "limit": limit}
        )

    @staticmethod
    def extract_actor_ids(markdown: str) -> list[str]:
        """Pull every "username/name" token out of a Markdown response.

        Used as a sanity-check fallback: if the LLM picks an actor that
        isn't in this list, we know it hallucinated.
        """
        return list(dict.fromkeys(_ACTOR_ID_PATTERN.findall(markdown)))

    async def fetch_actor_details(self, actor_id: str) -> str:
        """Get an Actor's README/schema/pricing as Markdown."""
        return await self._call_tool_raw("fetch-actor-details", {"actor": actor_id})

    async def call_actor(
        self, actor_id: str, run_input: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Run an Actor and return its dataset items.

        call-actor returns dataset items as JSON inside a TextContent block.
        We try several known response shapes and warn on the unknown.
        """
        raw = await self._call_tool_raw(
            "call-actor", {"actor": actor_id, "input": run_input}
        )
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("call_actor returned non-JSON for %s: %s", actor_id, raw[:200])
            return []

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if isinstance(parsed.get("items"), list):
                return parsed["items"]
            # Some versions return run metadata only — fetch via dataset
            dataset_id = parsed.get("defaultDatasetId") or parsed.get("datasetId")
            if dataset_id:
                items_raw = await self._call_tool_raw(
                    "get-actor-output", {"datasetId": dataset_id, "limit": 100}
                )
                try:
                    items = json.loads(items_raw)
                except json.JSONDecodeError:
                    return []
                if isinstance(items, list):
                    return items
                if isinstance(items, dict) and isinstance(items.get("items"), list):
                    return items["items"]
        log.warning("call_actor returned unexpected shape for %s", actor_id)
        return []
