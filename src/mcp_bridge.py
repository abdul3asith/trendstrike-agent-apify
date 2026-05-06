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


class ActorInputValidationError(RuntimeError):
    """Raised when call-actor rejects our input as schema-invalid.

    Carries both the human-readable error and the schema text Apify returned,
    so a caller can feed both back to the LLM for self-correction.
    """

    def __init__(self, errors: str, schema_text: str) -> None:
        super().__init__(errors)
        self.errors = errors
        self.schema_text = schema_text


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

    async def _call_tool_collect_blocks(
        self, name: str, arguments: dict[str, Any]
    ) -> list[str]:
        """Same as _call_tool_raw, but returns each text block separately.

        Some tools (notably call-actor with large outputs) split their
        response across multiple TextContent blocks. Joining them with \\n
        produces invalid JSON; parsing per-block and concatenating the
        results works correctly.

        On error responses for call-actor, Apify returns up to three blocks:
        a human message, the input schema, and a validation-error summary.
        We detect that pattern and raise ActorInputValidationError so the
        caller can self-correct.
        """
        assert self._session is not None, "Bridge not entered as context manager"
        result = await self._session.call_tool(name, arguments=arguments)

        if result.isError:
            block_texts = [
                getattr(b, "text", "") for b in result.content if hasattr(b, "text")
            ]
            joined = " | ".join(block_texts)
            # Detect the input-validation failure pattern so the caller can retry
            if name == "call-actor" and "Input validation failed" in joined:
                schema_text = ""
                errors_text = joined
                for t in block_texts:
                    if "Input schema" in t:
                        schema_text = t
                    elif "Validation errors" in t:
                        errors_text = t
                raise ActorInputValidationError(errors_text, schema_text)
            raise RuntimeError(f"MCP tool {name} returned error: {result.content!r}")

        return [block.text for block in result.content if hasattr(block, "text")]

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

        Apify's call-actor splits large outputs across multiple TextContent
        blocks. We parse each block independently and concatenate the
        resulting items. Each block may be:
            - a JSON list of items
            - a JSON dict like {"items": [...]} or {"defaultDatasetId": "..."}
            - free-form text (header/footer the LLM would normally read)
        """
        blocks = await self._call_tool_collect_blocks(
            "call-actor", {"actor": actor_id, "input": run_input}
        )
        if not blocks:
            return []

        items: list[dict[str, Any]] = []
        dataset_id_fallback: str | None = None

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                # Likely a header/footer Markdown line — skip silently
                continue

            if isinstance(parsed, list):
                items.extend(p for p in parsed if isinstance(p, dict))
            elif isinstance(parsed, dict):
                if isinstance(parsed.get("items"), list):
                    items.extend(p for p in parsed["items"] if isinstance(p, dict))
                else:
                    # Single item, or run-metadata block
                    if "defaultDatasetId" in parsed or "datasetId" in parsed:
                        dataset_id_fallback = parsed.get(
                            "defaultDatasetId"
                        ) or parsed.get("datasetId")
                    elif parsed:  # treat as a single item
                        items.append(parsed)

        # If no inline items but we got a dataset id, fetch from there
        if not items and dataset_id_fallback:
            extra = await self._call_tool_collect_blocks(
                "get-actor-output",
                {"datasetId": dataset_id_fallback, "limit": 100},
            )
            for block in extra:
                try:
                    parsed = json.loads(block.strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    items.extend(p for p in parsed if isinstance(p, dict))
                elif isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                    items.extend(p for p in parsed["items"] if isinstance(p, dict))

        log.info("call_actor: %s returned %d items", actor_id, len(items))
        return items
