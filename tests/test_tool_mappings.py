from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


def install_stub_modules() -> None:
    """Install minimal stubs so the module can be imported without dependencies."""
    if "mcp.server.fastmcp" not in sys.modules:
        mcp_module = types.ModuleType("mcp")
        mcp_server_module = types.ModuleType("mcp.server")
        fastmcp_module = types.ModuleType("mcp.server.fastmcp")

        class DummyFastMCP:
            def __init__(self, name: str):
                self.name = name
                self.registered_tools: dict[str, object] = {}

            def tool(self, name: str | None = None):
                def decorator(func):
                    self.registered_tools[name or func.__name__] = func
                    return func

                return decorator

            def run(self, transport: str = "stdio") -> None:
                return None

        fastmcp_module.FastMCP = DummyFastMCP
        sys.modules["mcp"] = mcp_module
        sys.modules["mcp.server"] = mcp_server_module
        sys.modules["mcp.server.fastmcp"] = fastmcp_module

    if "httpx" not in sys.modules:
        httpx_module = types.ModuleType("httpx")

        class DummyTimeout:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class DummyTimeoutException(Exception):
            pass

        class DummyConnectError(Exception):
            pass

        class DummyHTTPStatusError(Exception):
            def __init__(self, message: str, response):
                super().__init__(message)
                self.response = response

        class DummyAsyncClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                raise NotImplementedError

            async def get(self, *args, **kwargs):
                raise NotImplementedError

        httpx_module.Timeout = DummyTimeout
        httpx_module.TimeoutException = DummyTimeoutException
        httpx_module.ConnectError = DummyConnectError
        httpx_module.HTTPStatusError = DummyHTTPStatusError
        httpx_module.AsyncClient = DummyAsyncClient
        sys.modules["httpx"] = httpx_module


install_stub_modules()
exa_pool_mcp = importlib.import_module("exa_pool_mcp")


class ToolMappingTests(unittest.IsolatedAsyncioTestCase):
    def test_registers_official_and_legacy_tools(self) -> None:
        registered_tools = exa_pool_mcp.mcp.registered_tools
        self.assertIn("web_search_exa", registered_tools)
        self.assertIn("web_search_advanced_exa", registered_tools)
        self.assertIn("get_code_context_exa", registered_tools)
        self.assertIn("deep_researcher_start", registered_tools)
        self.assertIn("exa_search", registered_tools)
        self.assertIn("exa_get_research", registered_tools)

    def test_build_web_search_advanced_payload_maps_filters(self) -> None:
        payload = exa_pool_mcp.build_web_search_advanced_payload(
            query="ai agents",
            num_results=12,
            search_type="neural",
            category="news",
            include_domains=["example.com"],
            exclude_domains=["ignored.com"],
            start_published_date="2026-01-01",
            end_published_date="2026-01-31",
            start_crawl_date="2026-01-01",
            end_crawl_date="2026-01-31",
            include_text=["OpenAI"],
            exclude_text=["rumor"],
            user_location="US",
            moderation=True,
            additional_queries=["ai news"],
            text_max_characters=2500,
            context_max_characters=8000,
            enable_summary=True,
            summary_query="latest launches",
            enable_highlights=True,
            highlights_num_sentences=3,
            highlights_per_url=2,
            highlights_query="funding",
            livecrawl="preferred",
            livecrawl_timeout=15000,
            subpages=2,
            subpage_target=["pricing"],
        )

        self.assertEqual(payload["type"], "neural")
        self.assertEqual(payload["category"], "news")
        self.assertEqual(payload["includeDomains"], ["example.com"])
        self.assertEqual(payload["excludeDomains"], ["ignored.com"])
        self.assertEqual(payload["includeText"], ["OpenAI"])
        self.assertEqual(payload["excludeText"], ["rumor"])
        self.assertEqual(payload["contents"]["text"], {"maxCharacters": 2500})
        self.assertEqual(payload["contents"]["context"], {"maxCharacters": 8000})
        self.assertEqual(payload["contents"]["summary"], {"query": "latest launches"})
        self.assertEqual(
            payload["contents"]["highlights"],
            {
                "numSentences": 3,
                "highlightsPerUrl": 2,
                "query": "funding",
            },
        )
        self.assertEqual(payload["contents"]["subpages"], 2)
        self.assertEqual(payload["contents"]["subpageTarget"], ["pricing"])

    def test_build_deep_search_payload_maps_schema_and_queries(self) -> None:
        payload = exa_pool_mcp.build_deep_search_payload(
            objective="Find latest agent benchmarks",
            search_type="deep-reasoning",
            num_results=6,
            highlight_max_characters=5000,
            search_queries=["agent benchmark", "llm eval"],
            output_schema={"type": "object", "properties": {"winner": {"type": "string"}}},
            system_prompt="Return the strongest source first.",
            structured_output=False,
        )

        self.assertEqual(payload["type"], "deep-reasoning")
        self.assertEqual(payload["additionalQueries"], ["agent benchmark", "llm eval"])
        self.assertEqual(payload["outputSchema"]["type"], "object")
        self.assertEqual(payload["systemPrompt"], "Return the strongest source first.")
        self.assertEqual(
            payload["contents"]["highlights"]["maxCharacters"],
            5000,
        )

    def test_build_answer_payload_omits_legacy_text_flag(self) -> None:
        payload = exa_pool_mcp.build_answer_payload(
            query="What changed in GPT-5.4?",
            include_text=True,
            output_schema={"type": "object"},
        )

        self.assertEqual(payload["query"], "What changed in GPT-5.4?")
        self.assertEqual(payload["outputSchema"], {"type": "object"})
        self.assertNotIn("text", payload)

    async def test_web_search_exa_returns_context_text(self) -> None:
        with patch.object(
            exa_pool_mcp,
            "perform_exa_request",
            new=AsyncMock(return_value={"context": "condensed context", "results": []}),
        ):
            result = await exa_pool_mcp.web_search_exa("latest AI agent news")

        self.assertEqual(result, "condensed context")

    async def test_deep_researcher_check_formats_running_status(self) -> None:
        with patch.object(
            exa_pool_mcp,
            "perform_exa_request",
            new=AsyncMock(return_value={"status": "running", "researchId": "research_1"}),
        ):
            result = await exa_pool_mcp.deep_researcher_check("research_1")

        self.assertIn('"status": "running"', result)
        self.assertIn("Continue polling", result)

    async def test_exa_answer_omits_legacy_text_flag_in_request(self) -> None:
        with patch.object(
            exa_pool_mcp,
            "request_and_format",
            new=AsyncMock(return_value="ok"),
        ) as request_and_format:
            result = await exa_pool_mcp.exa_answer(
                "What changed in GPT-5.4?",
                include_text=True,
                output_schema={"type": "object"},
            )

        self.assertEqual(result, "ok")
        request_and_format.assert_awaited_once_with(
            exa_pool_mcp.ANSWER_ENDPOINT,
            data={
                "query": "What changed in GPT-5.4?",
                "outputSchema": {"type": "object"},
            },
        )

    async def test_context_endpoint_404_is_reported_explicitly(self) -> None:
        class FakeResponse:
            status_code = 404
            reason_phrase = "Not Found"

            def json(self):
                return {"message": "missing"}

            def raise_for_status(self):
                raise exa_pool_mcp.httpx.HTTPStatusError("Not Found", self)

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return FakeResponse()

            async def get(self, *args, **kwargs):
                return FakeResponse()

        with patch.object(exa_pool_mcp.httpx, "AsyncClient", FakeAsyncClient):
            result = await exa_pool_mcp.perform_exa_request(
                exa_pool_mcp.CONTEXT_ENDPOINT,
                data={"query": "react useState"},
            )

        self.assertEqual(
            result,
            "Error 404: Endpoint not found: /context. The Exa Pool backend does not expose the code context API.",
        )

    async def test_context_endpoint_405_is_reported_explicitly(self) -> None:
        class FakeResponse:
            status_code = 405
            reason_phrase = "Method Not Allowed"

            def json(self):
                return {"message": "method not allowed"}

            def raise_for_status(self):
                raise exa_pool_mcp.httpx.HTTPStatusError("Method Not Allowed", self)

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return FakeResponse()

            async def get(self, *args, **kwargs):
                return FakeResponse()

        with patch.object(exa_pool_mcp.httpx, "AsyncClient", FakeAsyncClient):
            result = await exa_pool_mcp.perform_exa_request(
                exa_pool_mcp.CONTEXT_ENDPOINT,
                data={"query": "react useState"},
            )

        self.assertEqual(
            result,
            "Error 405: Method not allowed for /context. The Exa Pool backend does not expose the code context API.",
        )

    async def test_deep_search_rejects_long_query_variants(self) -> None:
        result = await exa_pool_mcp.deep_search_exa(
            objective="Find AI benchmarks",
            search_queries=["this variant has six separate words total"],
        )

        self.assertEqual(result, "Error: each search_queries entry must be 5 words or fewer")


if __name__ == "__main__":
    unittest.main()
