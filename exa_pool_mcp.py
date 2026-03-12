# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]>=1.0.0",
#     "httpx[socks]>=0.27.0",
# ]
# ///
"""
Exa Pool MCP Server
Wraps the Exa Pool API as an MCP server for Claude Code.

This server exposes an Exa-compatible MCP tool surface on top of an
Exa Pool / ExaFree deployment.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any, Optional, Sequence

import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging (critical for stdio transport - never use print!)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("exa-pool")

# API Configuration
EXA_POOL_BASE_URL = os.getenv("EXA_POOL_BASE_URL", "")
EXA_POOL_API_KEY = os.getenv("EXA_POOL_API_KEY", "")

# HTTP client timeout settings
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

SEARCH_ENDPOINT = "/search"
CONTENTS_ENDPOINT = "/contents"
CONTEXT_ENDPOINT = "/context"
ANSWER_ENDPOINT = "/answer"
FIND_SIMILAR_ENDPOINT = "/findSimilar"
RESEARCH_ENDPOINT = "/research/v1"

SEARCH_CATEGORIES = {
    "company",
    "financial report",
    "github",
    "news",
    "pdf",
    "people",
    "personal site",
    "research paper",
    "tweet",
}
LEGACY_SEARCH_TYPES = {"auto", "deep", "deep-reasoning", "fast", "neural"}
WEB_SEARCH_TYPES = {"auto", "fast"}
ADVANCED_SEARCH_TYPES = {"auto", "fast", "neural"}
DEEP_SEARCH_TYPES = {"deep", "deep-reasoning"}
BASIC_LIVECRAWL_OPTIONS = {"fallback", "preferred"}
ADVANCED_LIVECRAWL_OPTIONS = {"always", "fallback", "never", "preferred"}
RESEARCH_MODELS = {"exa-research-fast", "exa-research", "exa-research-pro"}


# ============================================================================
# Helper Functions
# ============================================================================


def format_error(status_code: int, message: str) -> str:
    """Format error messages consistently."""
    return f"Error {status_code}: {message}"


def format_json_response(data: Any) -> str:
    """Format JSON response for readability."""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error(f"Failed to format JSON: {exc}")
        return str(data)


def parse_iso_date(value: str) -> None:
    """Validate an ISO-8601 date or datetime string."""
    if "T" in value or value.endswith("Z"):
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return

    date.fromisoformat(value)


def clean_string(value: str, field_name: str) -> tuple[Optional[str], Optional[str]]:
    """Return a stripped string or a validation error."""
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        return None, f"Error: {field_name} parameter is required and cannot be empty"
    return cleaned, None


def validate_choice(
    value: str,
    field_name: str,
    allowed_values: Sequence[str],
) -> Optional[str]:
    """Validate a string enum value."""
    if value not in allowed_values:
        return (
            f"Error: {field_name} must be one of: "
            + ", ".join(sorted(allowed_values))
        )
    return None


def validate_int_range(
    value: int,
    field_name: str,
    minimum: int,
    maximum: int,
) -> Optional[str]:
    """Validate an integer range."""
    if not minimum <= value <= maximum:
        return f"Error: {field_name} must be between {minimum} and {maximum}"
    return None


def validate_optional_iso_date(value: Optional[str], field_name: str) -> Optional[str]:
    """Validate optional ISO date fields used by advanced search."""
    if value is None:
        return None

    cleaned, error = clean_string(value, field_name)
    if error:
        return error

    try:
        parse_iso_date(cleaned or "")
    except ValueError:
        return f"Error: {field_name} must be a valid ISO 8601 date or datetime"

    return None


def validate_optional_string_list(
    values: Optional[Sequence[str]],
    field_name: str,
    *,
    max_items: Optional[int] = None,
) -> tuple[Optional[list[str]], Optional[str]]:
    """Validate optional string arrays used by advanced tools."""
    if values is None:
        return None, None

    cleaned_values: list[str] = []
    for value in values:
        cleaned, error = clean_string(value, field_name)
        if error:
            return None, error
        cleaned_values.append(cleaned or "")

    if max_items is not None and len(cleaned_values) > max_items:
        return None, f"Error: {field_name} can contain at most {max_items} items"

    return cleaned_values, None


def validate_url(url: str, field_name: str = "url") -> tuple[Optional[str], Optional[str]]:
    """Validate URL inputs."""
    cleaned, error = clean_string(url, field_name)
    if error:
        return None, error

    if not cleaned.startswith(("http://", "https://")):
        return (
            None,
            f"Error: Invalid URL format for {field_name}. URLs must start with http:// or https://",
        )

    return cleaned, None


def validate_urls(urls: Sequence[str], field_name: str = "urls") -> tuple[Optional[list[str]], Optional[str]]:
    """Validate non-empty URL lists."""
    if not urls:
        return None, f"Error: {field_name} parameter is required and cannot be empty"

    if len(urls) > 100:
        return None, "Error: Maximum 100 URLs allowed per request"

    cleaned_urls: list[str] = []
    for url in urls:
        cleaned, error = validate_url(url, field_name="url")
        if error:
            return None, error
        cleaned_urls.append(cleaned or "")

    return cleaned_urls, None


def build_legacy_search_payload(
    query: str,
    num_results: int,
    search_type: str,
    include_text: bool,
) -> dict[str, Any]:
    """Build the legacy exa_search payload."""
    payload: dict[str, Any] = {
        "query": query,
        "numResults": num_results,
        "type": search_type,
    }
    if include_text:
        payload["contents"] = {"text": True}
    return payload


def build_legacy_contents_payload(
    urls: Sequence[str],
    include_text: bool,
    include_html: bool,
) -> dict[str, Any]:
    """Build the legacy contents payload."""
    payload: dict[str, Any] = {"urls": list(urls), "text": include_text}
    if include_html:
        payload["htmlContent"] = True
    return payload


def build_find_similar_payload(
    url: str,
    num_results: int,
    include_text: bool,
) -> dict[str, Any]:
    """Build a findSimilar request payload."""
    payload: dict[str, Any] = {"url": url, "numResults": num_results}
    if include_text:
        payload["contents"] = {"text": True}
    return payload


def build_answer_payload(
    query: str,
    include_text: bool,
    output_schema: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build an answer request payload.

    The legacy include_text flag is retained in the MCP schema for compatibility,
    but Exa Pool backends may return degraded answers when the field is sent to
    /answer. Intentionally omit it from the upstream payload.
    """
    _ = include_text
    payload: dict[str, Any] = {"query": query}
    if output_schema is not None:
        payload["outputSchema"] = output_schema
    return payload


def build_web_search_payload(
    query: str,
    num_results: int,
    search_type: str,
    livecrawl: str,
    category: Optional[str],
    context_max_characters: int,
) -> dict[str, Any]:
    """Build the official web_search_exa payload."""
    payload: dict[str, Any] = {
        "query": query,
        "type": search_type,
        "numResults": num_results,
        "contents": {
            "text": True,
            "context": {"maxCharacters": context_max_characters},
            "livecrawl": livecrawl,
        },
    }
    if category is not None:
        payload["category"] = category
    return payload


def build_web_search_advanced_payload(
    query: str,
    num_results: int,
    search_type: str,
    category: Optional[str],
    include_domains: Optional[Sequence[str]],
    exclude_domains: Optional[Sequence[str]],
    start_published_date: Optional[str],
    end_published_date: Optional[str],
    start_crawl_date: Optional[str],
    end_crawl_date: Optional[str],
    include_text: Optional[Sequence[str]],
    exclude_text: Optional[Sequence[str]],
    user_location: Optional[str],
    moderation: Optional[bool],
    additional_queries: Optional[Sequence[str]],
    text_max_characters: Optional[int],
    context_max_characters: Optional[int],
    enable_summary: bool,
    summary_query: Optional[str],
    enable_highlights: bool,
    highlights_num_sentences: Optional[int],
    highlights_per_url: Optional[int],
    highlights_query: Optional[str],
    livecrawl: str,
    livecrawl_timeout: Optional[int],
    subpages: Optional[int],
    subpage_target: Optional[Sequence[str]],
) -> dict[str, Any]:
    """Build the official web_search_advanced_exa payload."""
    contents: dict[str, Any] = {
        "text": {"maxCharacters": text_max_characters} if text_max_characters else True,
        "livecrawl": livecrawl,
    }

    if context_max_characters is not None:
        contents["context"] = {"maxCharacters": context_max_characters}

    if livecrawl_timeout is not None:
        contents["livecrawlTimeout"] = livecrawl_timeout

    if enable_summary:
        contents["summary"] = {"query": summary_query} if summary_query else True

    if enable_highlights:
        contents["highlights"] = {
            key: value
            for key, value in {
                "numSentences": highlights_num_sentences,
                "highlightsPerUrl": highlights_per_url,
                "query": highlights_query,
            }.items()
            if value is not None
        }

    if subpages is not None:
        contents["subpages"] = subpages

    if subpage_target is not None:
        contents["subpageTarget"] = list(subpage_target)

    payload: dict[str, Any] = {
        "query": query,
        "type": search_type,
        "numResults": num_results,
        "contents": contents,
    }

    optional_fields = {
        "category": category,
        "includeDomains": list(include_domains) if include_domains is not None else None,
        "excludeDomains": list(exclude_domains) if exclude_domains is not None else None,
        "startPublishedDate": start_published_date,
        "endPublishedDate": end_published_date,
        "startCrawlDate": start_crawl_date,
        "endCrawlDate": end_crawl_date,
        "includeText": list(include_text) if include_text is not None else None,
        "excludeText": list(exclude_text) if exclude_text is not None else None,
        "userLocation": user_location,
        "moderation": moderation,
        "additionalQueries": list(additional_queries)
        if additional_queries is not None
        else None,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value

    return payload


def build_company_research_payload(company_name: str, num_results: int) -> dict[str, Any]:
    """Build the company research search payload."""
    return {
        "query": f"{company_name} company",
        "type": "auto",
        "numResults": num_results,
        "category": "company",
        "contents": {
            "text": {"maxCharacters": 7000},
        },
    }


def build_people_search_payload(query: str, num_results: int) -> dict[str, Any]:
    """Build the people search payload."""
    return {
        "query": f"{query} profile",
        "type": "auto",
        "numResults": num_results,
        "category": "people",
        "contents": {
            "text": {"maxCharacters": 2000},
        },
    }


def build_crawling_payload(url: str, max_characters: int) -> dict[str, Any]:
    """Build the crawling payload."""
    return {
        "ids": [url],
        "contents": {
            "text": {"maxCharacters": max_characters},
            "livecrawl": "preferred",
        },
    }


def build_code_context_payload(query: str, tokens_num: int) -> dict[str, Any]:
    """Build the code context payload."""
    return {"query": query, "tokensNum": tokens_num}


def build_deep_search_payload(
    objective: str,
    search_type: str,
    num_results: int,
    highlight_max_characters: int,
    search_queries: Optional[Sequence[str]],
    output_schema: Optional[dict[str, Any]],
    system_prompt: Optional[str],
    structured_output: bool,
) -> dict[str, Any]:
    """Build the deep search payload."""
    payload: dict[str, Any] = {
        "query": objective,
        "type": search_type,
        "numResults": num_results,
        "contents": {
            "highlights": {
                "maxCharacters": highlight_max_characters,
            }
        },
    }

    if search_queries is not None:
        payload["additionalQueries"] = list(search_queries)

    if output_schema is not None:
        payload["outputSchema"] = output_schema
    elif structured_output:
        payload["outputSchema"] = {"type": "object"}

    if system_prompt is not None:
        payload["systemPrompt"] = system_prompt

    return payload


def build_research_payload(
    instructions: str,
    model: str,
    output_schema: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the deep research payload."""
    payload: dict[str, Any] = {"instructions": instructions, "model": model}
    if output_schema is not None:
        payload["outputSchema"] = output_schema
    return payload


def format_search_result_or_default(data: Any, empty_message: str) -> str:
    """Return formatted search JSON or a user-friendly empty result message."""
    if not isinstance(data, dict):
        return format_json_response(data)

    results = data.get("results")
    if isinstance(results, list) and not results:
        return empty_message

    return format_json_response(data)


def format_deep_search_response(
    data: Any,
    *,
    structured_output: bool,
) -> str:
    """Format deep search responses similar to the official Exa MCP server."""
    if not isinstance(data, dict):
        return format_json_response(data)

    if structured_output:
        return format_json_response(
            {
                "output": data.get("output"),
                "results": data.get("results"),
                "searchTime": data.get("searchTime"),
                "costDollars": data.get("costDollars"),
            }
        )

    parts: list[str] = []
    output = data.get("output")
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(f"## Answer\n\n{content}")
        elif content is not None:
            parts.append(f"## Answer\n\n{format_json_response(content)}")

        grounding = output.get("grounding")
        citations: list[str] = []
        if isinstance(grounding, list):
            for item in grounding:
                if not isinstance(item, dict):
                    continue
                entry_citations = item.get("citations")
                if not isinstance(entry_citations, list):
                    continue
                for citation in entry_citations:
                    if not isinstance(citation, dict):
                        continue
                    title = citation.get("title") or "Untitled"
                    url = citation.get("url") or ""
                    citations.append(f"- [{title}]({url})")
        if citations:
            parts.append("## Citations\n\n" + "\n".join(citations))

    results = data.get("results")
    if isinstance(results, list) and results:
        result_sections: list[str] = []
        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            lines = [f"### {index}. {result.get('title') or 'Untitled'}"]
            lines.append(f"**URL:** {result.get('url') or ''}")
            if result.get("publishedDate"):
                lines.append(f"**Published:** {result['publishedDate']}")
            if result.get("image"):
                lines.append(f"**Image:** {result['image']}")
            highlights = result.get("highlights")
            if isinstance(highlights, list) and highlights:
                lines.append("")
                lines.append("\n\n".join(str(item) for item in highlights))
            result_sections.append("\n".join(lines))
        if result_sections:
            parts.append("## Results\n\n" + "\n\n---\n\n".join(result_sections))

    if not parts:
        return "No results found. Please try a different query."

    return "\n\n---\n\n".join(parts)


def format_research_status_response(data: Any) -> str:
    """Format research polling results in a polling-friendly JSON envelope."""
    if not isinstance(data, dict):
        return format_json_response(data)

    research_id = data.get("researchId")
    status = data.get("status")

    if status == "completed":
        return format_json_response(
            {
                "success": True,
                "status": status,
                "researchId": research_id,
                "report": (data.get("output") or {}).get("content")
                if isinstance(data.get("output"), dict)
                else None,
                "parsedOutput": (data.get("output") or {}).get("parsed")
                if isinstance(data.get("output"), dict)
                else None,
                "citations": data.get("citations"),
                "model": data.get("model"),
                "costDollars": data.get("costDollars"),
                "message": "Deep research completed. Here is the research report.",
            }
        )

    if status in {"pending", "running"}:
        return format_json_response(
            {
                "success": True,
                "status": status,
                "researchId": research_id,
                "message": "Research in progress. Continue polling with the same research ID.",
                "nextAction": "Call deep_researcher_check again with the same research ID.",
            }
        )

    if status == "failed":
        return format_json_response(
            {
                "success": False,
                "status": status,
                "researchId": research_id,
                "message": "Deep research task failed. Start a new task with updated instructions.",
            }
        )

    if status == "canceled":
        return format_json_response(
            {
                "success": False,
                "status": status,
                "researchId": research_id,
                "message": "Deep research task was canceled.",
            }
        )

    return format_json_response(data)


async def perform_exa_request(
    endpoint: str,
    *,
    method: str = "POST",
    data: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> Any | str:
    """
    Make a request to the Exa Pool API with proper error handling.

    Returns the decoded JSON body on success, or an error string on failure.
    """
    url = f"{EXA_POOL_BASE_URL.rstrip('/')}{endpoint}"
    headers = {
        "Authorization": f"Bearer {EXA_POOL_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "exa-pool-mcp-server/1.0",
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if method == "POST":
                response = await client.post(url, json=data, headers=headers)
            elif method == "GET":
                response = await client.get(url, params=params, headers=headers)
            else:
                return f"Error: Unsupported HTTP method: {method}"

            if response.status_code == 401:
                logger.error("Authentication failed - API key may be invalid")
                return format_error(
                    401,
                    "Authentication failed. API key may be invalid.",
                )

            if response.status_code == 403:
                logger.error("Access forbidden")
                return format_error(403, "Access denied.")

            if response.status_code == 404:
                logger.error(f"Endpoint not found: {endpoint}")
                if endpoint == CONTEXT_ENDPOINT:
                    return format_error(
                        404,
                        "Endpoint not found: /context. The Exa Pool backend does not expose the code context API.",
                    )
                return format_error(404, f"Endpoint not found: {endpoint}")

            if response.status_code == 405:
                logger.error(f"Method not allowed for endpoint: {endpoint}")
                if endpoint == CONTEXT_ENDPOINT:
                    return format_error(
                        405,
                        "Method not allowed for /context. The Exa Pool backend does not expose the code context API.",
                    )
                return format_error(
                    405,
                    f"Method not allowed for endpoint: {endpoint}",
                )

            if response.status_code == 429:
                logger.warning("Rate limited")
                return format_error(429, "Rate limited. Please try again later.")

            if response.status_code >= 500:
                logger.error(f"Server error: {response.status_code}")
                return format_error(
                    response.status_code,
                    "Exa Pool server error. The service may be temporarily unavailable.",
                )

            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        logger.error(f"Request timeout for {endpoint}")
        return (
            "Error: Request timed out after 30 seconds. "
            "The Exa Pool API may be slow or unavailable."
        )

    except httpx.ConnectError as exc:
        logger.error(f"Connection error: {exc}")
        return (
            f"Error: Unable to connect to Exa Pool API at {EXA_POOL_BASE_URL}. "
            "Please check the service status."
        )

    except httpx.HTTPStatusError as exc:
        logger.error(f"HTTP error {exc.response.status_code}: {exc}")
        return format_error(
            exc.response.status_code,
            f"HTTP request failed: {exc.response.reason_phrase}",
        )

    except ValueError as exc:
        logger.error(f"Invalid JSON response: {exc}")
        return "Error: Received invalid JSON response from Exa Pool API."

    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return f"Error: {type(exc).__name__}: {exc}"


async def request_and_format(
    endpoint: str,
    *,
    method: str = "POST",
    data: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> str:
    """Execute an Exa request and JSON-format successful responses."""
    result = await perform_exa_request(
        endpoint,
        method=method,
        data=data,
        params=params,
    )
    if isinstance(result, str):
        return result
    return format_json_response(result)


# ============================================================================
# MCP Tools - Official Exa MCP Compatibility Layer
# ============================================================================


@mcp.tool(name="web_search_exa")
async def web_search_exa(
    query: str,
    numResults: int = 8,
    livecrawl: str = "fallback",
    type: str = "auto",
    category: Optional[str] = None,
    contextMaxCharacters: int = 10000,
) -> str:
    """
    Search the web and return a context string optimized for LLM use.

    This mirrors the official Exa MCP web_search_exa tool.
    """
    cleaned_query, error = clean_string(query, "query")
    if error:
        return error

    if category is not None:
        category, error = clean_string(category, "category")
        if error:
            return error
        error = validate_choice(category, "category", SEARCH_CATEGORIES)
        if error:
            return error

    for field_name, value, allowed in (
        ("type", type, WEB_SEARCH_TYPES),
        ("livecrawl", livecrawl, BASIC_LIVECRAWL_OPTIONS),
    ):
        error = validate_choice(value, field_name, allowed)
        if error:
            return error

    for field_name, value, minimum, maximum in (
        ("numResults", numResults, 1, 100),
        ("contextMaxCharacters", contextMaxCharacters, 1, 1000000),
    ):
        error = validate_int_range(value, field_name, minimum, maximum)
        if error:
            return error

    payload = build_web_search_payload(
        cleaned_query or "",
        numResults,
        type,
        livecrawl,
        category,
        contextMaxCharacters,
    )
    logger.info(
        "Running web_search_exa: query='%s', numResults=%s, type=%s",
        cleaned_query,
        numResults,
        type,
    )

    result = await perform_exa_request(SEARCH_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        context = result.get("context")
        if isinstance(context, str) and context.strip():
            return context

    return format_search_result_or_default(
        result,
        "No search results found. Please try a different query.",
    )


@mcp.tool(name="web_search_advanced_exa")
async def web_search_advanced_exa(
    query: str,
    numResults: int = 10,
    type: str = "auto",
    category: Optional[str] = None,
    includeDomains: Optional[list[str]] = None,
    excludeDomains: Optional[list[str]] = None,
    startPublishedDate: Optional[str] = None,
    endPublishedDate: Optional[str] = None,
    startCrawlDate: Optional[str] = None,
    endCrawlDate: Optional[str] = None,
    includeText: Optional[list[str]] = None,
    excludeText: Optional[list[str]] = None,
    userLocation: Optional[str] = None,
    moderation: Optional[bool] = None,
    additionalQueries: Optional[list[str]] = None,
    textMaxCharacters: Optional[int] = None,
    contextMaxCharacters: Optional[int] = None,
    enableSummary: bool = False,
    summaryQuery: Optional[str] = None,
    enableHighlights: bool = False,
    highlightsNumSentences: Optional[int] = None,
    highlightsPerUrl: Optional[int] = None,
    highlightsQuery: Optional[str] = None,
    livecrawl: str = "fallback",
    livecrawlTimeout: Optional[int] = None,
    subpages: Optional[int] = None,
    subpageTarget: Optional[list[str]] = None,
) -> str:
    """
    Advanced Exa search with full access to filters and content extraction options.
    """
    cleaned_query, error = clean_string(query, "query")
    if error:
        return error

    error = validate_choice(type, "type", ADVANCED_SEARCH_TYPES)
    if error:
        return error

    error = validate_choice(livecrawl, "livecrawl", ADVANCED_LIVECRAWL_OPTIONS)
    if error:
        return error

    if category is not None:
        category, error = clean_string(category, "category")
        if error:
            return error
        error = validate_choice(category, "category", SEARCH_CATEGORIES)
        if error:
            return error

    if userLocation is not None:
        userLocation, error = clean_string(userLocation, "userLocation")
        if error:
            return error

    if summaryQuery is not None:
        summaryQuery, error = clean_string(summaryQuery, "summaryQuery")
        if error:
            return error

    if highlightsQuery is not None:
        highlightsQuery, error = clean_string(highlightsQuery, "highlightsQuery")
        if error:
            return error

    for field_name, value, minimum, maximum in (
        ("numResults", numResults, 1, 100),
        ("textMaxCharacters", textMaxCharacters or 1, 1, 1000000)
        if textMaxCharacters is not None
        else ("_skip", 1, 1, 1),
        ("contextMaxCharacters", contextMaxCharacters or 1, 1, 1000000)
        if contextMaxCharacters is not None
        else ("_skip", 1, 1, 1),
        ("highlightsNumSentences", highlightsNumSentences or 1, 1, 20)
        if highlightsNumSentences is not None
        else ("_skip", 1, 1, 1),
        ("highlightsPerUrl", highlightsPerUrl or 1, 1, 20)
        if highlightsPerUrl is not None
        else ("_skip", 1, 1, 1),
        ("livecrawlTimeout", livecrawlTimeout or 1, 1, 120000)
        if livecrawlTimeout is not None
        else ("_skip", 1, 1, 1),
        ("subpages", subpages or 1, 1, 10) if subpages is not None else ("_skip", 1, 1, 1),
    ):
        if field_name == "_skip":
            continue
        error = validate_int_range(value, field_name, minimum, maximum)
        if error:
            return error

    for field_name, value in (
        ("startPublishedDate", startPublishedDate),
        ("endPublishedDate", endPublishedDate),
        ("startCrawlDate", startCrawlDate),
        ("endCrawlDate", endCrawlDate),
    ):
        error = validate_optional_iso_date(value, field_name)
        if error:
            return error

    includeDomains, error = validate_optional_string_list(
        includeDomains,
        "includeDomains",
    )
    if error:
        return error

    excludeDomains, error = validate_optional_string_list(
        excludeDomains,
        "excludeDomains",
    )
    if error:
        return error

    includeText, error = validate_optional_string_list(includeText, "includeText")
    if error:
        return error

    excludeText, error = validate_optional_string_list(excludeText, "excludeText")
    if error:
        return error

    additionalQueries, error = validate_optional_string_list(
        additionalQueries,
        "additionalQueries",
    )
    if error:
        return error

    subpageTarget, error = validate_optional_string_list(
        subpageTarget,
        "subpageTarget",
    )
    if error:
        return error

    payload = build_web_search_advanced_payload(
        cleaned_query or "",
        numResults,
        type,
        category,
        includeDomains,
        excludeDomains,
        startPublishedDate,
        endPublishedDate,
        startCrawlDate,
        endCrawlDate,
        includeText,
        excludeText,
        userLocation,
        moderation,
        additionalQueries,
        textMaxCharacters,
        contextMaxCharacters,
        enableSummary,
        summaryQuery,
        enableHighlights,
        highlightsNumSentences,
        highlightsPerUrl,
        highlightsQuery,
        livecrawl,
        livecrawlTimeout,
        subpages,
        subpageTarget,
    )
    logger.info(
        "Running web_search_advanced_exa: query='%s', numResults=%s, type=%s",
        cleaned_query,
        numResults,
        type,
    )
    return await request_and_format(SEARCH_ENDPOINT, data=payload)


@mcp.tool(name="company_research_exa")
async def company_research_exa(companyName: str, numResults: int = 3) -> str:
    """Research a company using Exa search category presets."""
    cleaned_company_name, error = clean_string(companyName, "companyName")
    if error:
        return error

    error = validate_int_range(numResults, "numResults", 1, 100)
    if error:
        return error

    payload = build_company_research_payload(cleaned_company_name or "", numResults)
    logger.info("Running company_research_exa for '%s'", cleaned_company_name)

    result = await perform_exa_request(SEARCH_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    return format_search_result_or_default(
        result,
        "No company information found. Please try a different company name.",
    )


@mcp.tool(name="people_search_exa")
async def people_search_exa(query: str, numResults: int = 8) -> str:
    """Find public profiles and professional information for people."""
    cleaned_query, error = clean_string(query, "query")
    if error:
        return error

    error = validate_int_range(numResults, "numResults", 1, 100)
    if error:
        return error

    payload = build_people_search_payload(cleaned_query or "", numResults)
    logger.info("Running people_search_exa for '%s'", cleaned_query)

    result = await perform_exa_request(SEARCH_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    return format_search_result_or_default(
        result,
        "No people profiles found. Please try a different query.",
    )


@mcp.tool(name="crawling_exa")
async def crawling_exa(url: str, maxCharacters: int = 3000) -> str:
    """Fetch content for a single known URL."""
    cleaned_url, error = validate_url(url)
    if error:
        return error

    error = validate_int_range(maxCharacters, "maxCharacters", 1, 1000000)
    if error:
        return error

    payload = build_crawling_payload(cleaned_url or "", maxCharacters)
    logger.info("Running crawling_exa for '%s'", cleaned_url)

    result = await perform_exa_request(CONTENTS_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    return format_search_result_or_default(
        result,
        "No content found for the provided URL.",
    )


@mcp.tool(name="get_code_context_exa")
async def get_code_context_exa(query: str, tokensNum: int = 5000) -> str:
    """Fetch code examples and documentation context from the Exa /context API."""
    cleaned_query, error = clean_string(query, "query")
    if error:
        return error

    error = validate_int_range(tokensNum, "tokensNum", 1000, 50000)
    if error:
        return error

    payload = build_code_context_payload(cleaned_query or "", tokensNum)
    logger.info("Running get_code_context_exa for '%s'", cleaned_query)

    result = await perform_exa_request(CONTEXT_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    if not isinstance(result, dict):
        return format_json_response(result)

    code_context = result.get("response")
    if isinstance(code_context, str) and code_context.strip():
        return code_context

    return (
        "No code snippets or documentation found. Try a more specific library, "
        "framework, or API query."
    )


@mcp.tool(name="deep_search_exa")
async def deep_search_exa(
    objective: str,
    search_queries: Optional[list[str]] = None,
    type: str = "deep",
    numResults: int = 8,
    highlightMaxCharacters: int = 4000,
    outputSchema: Optional[dict[str, Any]] = None,
    systemPrompt: Optional[str] = None,
    structuredOutput: bool = False,
) -> str:
    """Run deep search using Exa's deep and deep-reasoning search types."""
    cleaned_objective, error = clean_string(objective, "objective")
    if error:
        return error

    error = validate_choice(type, "type", DEEP_SEARCH_TYPES)
    if error:
        return error

    for field_name, value, minimum, maximum in (
        ("numResults", numResults, 1, 100),
        ("highlightMaxCharacters", highlightMaxCharacters, 1, 1000000),
    ):
        error = validate_int_range(value, field_name, minimum, maximum)
        if error:
            return error

    if outputSchema is not None and not isinstance(outputSchema, dict):
        return "Error: outputSchema must be a JSON object"

    if systemPrompt is not None:
        systemPrompt, error = clean_string(systemPrompt, "systemPrompt")
        if error:
            return error
        if len(systemPrompt) > 32000:
            return "Error: systemPrompt must be 32000 characters or less"

    search_queries, error = validate_optional_string_list(
        search_queries,
        "search_queries",
        max_items=5,
    )
    if error:
        return error

    if search_queries is not None:
        for search_query in search_queries:
            if len(search_query) > 200:
                return (
                    "Error: each search_queries entry must be 200 characters or less"
                )
            if len(search_query.split()) > 5:
                return "Error: each search_queries entry must be 5 words or fewer"

    payload = build_deep_search_payload(
        cleaned_objective or "",
        type,
        numResults,
        highlightMaxCharacters,
        search_queries,
        outputSchema,
        systemPrompt,
        structuredOutput,
    )
    logger.info("Running deep_search_exa for '%s'", cleaned_objective)

    result = await perform_exa_request(SEARCH_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    return format_deep_search_response(
        result,
        structured_output=bool(outputSchema) or structuredOutput,
    )


@mcp.tool(name="deep_researcher_start")
async def deep_researcher_start(
    instructions: str,
    model: str = "exa-research-fast",
    outputSchema: Optional[dict[str, Any]] = None,
) -> str:
    """Start an asynchronous deep research task."""
    cleaned_instructions, error = clean_string(instructions, "instructions")
    if error:
        return error

    if len(cleaned_instructions or "") > 4096:
        return "Error: instructions must be 4096 characters or less"

    error = validate_choice(model, "model", RESEARCH_MODELS)
    if error:
        return error

    if outputSchema is not None and not isinstance(outputSchema, dict):
        return "Error: outputSchema must be a JSON object"

    payload = build_research_payload(cleaned_instructions or "", model, outputSchema)
    logger.info("Starting deep research task with model '%s'", model)

    result = await perform_exa_request(RESEARCH_ENDPOINT, data=payload)
    if isinstance(result, str):
        return result

    if not isinstance(result, dict) or not result.get("researchId"):
        return "Failed to start research task. Please try again."

    return format_json_response(
        {
            "success": True,
            "researchId": result.get("researchId"),
            "model": model,
            "instructions": cleaned_instructions,
            "message": (
                "Deep research task started. "
                "Use deep_researcher_check with the returned researchId."
            ),
            "nextStep": {
                "tool": "deep_researcher_check",
                "researchId": result.get("researchId"),
            },
        }
    )


@mcp.tool(name="deep_researcher_check")
async def deep_researcher_check(researchId: str) -> str:
    """Check the status of a deep research task."""
    cleaned_research_id, error = clean_string(researchId, "researchId")
    if error:
        return error

    logger.info("Checking deep research task '%s'", cleaned_research_id)
    result = await perform_exa_request(
        f"{RESEARCH_ENDPOINT}/{cleaned_research_id}",
        method="GET",
    )
    if isinstance(result, str):
        return result

    return format_research_status_response(result)


# ============================================================================
# MCP Tools - Legacy Compatibility Layer
# ============================================================================


@mcp.tool(name="exa_search")
async def exa_search(
    query: str,
    num_results: int = 10,
    search_type: str = "auto",
    include_text: bool = False,
) -> str:
    """Legacy search tool kept for backwards compatibility."""
    cleaned_query, error = clean_string(query, "query")
    if error:
        return error

    error = validate_int_range(num_results, "num_results", 1, 100)
    if error:
        return error

    error = validate_choice(search_type, "search_type", LEGACY_SEARCH_TYPES)
    if error:
        return error

    payload = build_legacy_search_payload(
        cleaned_query or "",
        num_results,
        search_type,
        include_text,
    )
    logger.info(
        "Running exa_search: query='%s', num_results=%s, search_type=%s",
        cleaned_query,
        num_results,
        search_type,
    )
    return await request_and_format(SEARCH_ENDPOINT, data=payload)


@mcp.tool(name="exa_get_contents")
async def exa_get_contents(
    urls: list[str],
    include_text: bool = True,
    include_html: bool = False,
) -> str:
    """Legacy batch contents tool kept for backwards compatibility."""
    cleaned_urls, error = validate_urls(urls)
    if error:
        return error

    payload = build_legacy_contents_payload(
        cleaned_urls or [],
        include_text,
        include_html,
    )
    logger.info("Running exa_get_contents for %s URLs", len(cleaned_urls or []))
    return await request_and_format(CONTENTS_ENDPOINT, data=payload)


@mcp.tool(name="exa_find_similar")
async def exa_find_similar(
    url: str,
    num_results: int = 10,
    include_text: bool = False,
) -> str:
    """Legacy find-similar tool kept for backwards compatibility."""
    cleaned_url, error = validate_url(url)
    if error:
        return error

    error = validate_int_range(num_results, "num_results", 1, 100)
    if error:
        return error

    payload = build_find_similar_payload(cleaned_url or "", num_results, include_text)
    logger.info("Running exa_find_similar for '%s'", cleaned_url)
    return await request_and_format(FIND_SIMILAR_ENDPOINT, data=payload)


@mcp.tool(name="exa_answer")
async def exa_answer(
    query: str,
    include_text: bool = False,
    output_schema: Optional[dict[str, Any]] = None,
) -> str:
    """Legacy answer tool kept for backwards compatibility.

    The include_text parameter is preserved for compatibility but not forwarded
    to the Exa Pool /answer endpoint.
    """
    cleaned_query, error = clean_string(query, "query")
    if error:
        return error

    if output_schema is not None and not isinstance(output_schema, dict):
        return "Error: output_schema must be a JSON object"

    payload = build_answer_payload(cleaned_query or "", include_text, output_schema)
    logger.info("Running exa_answer for '%s'", cleaned_query)
    return await request_and_format(ANSWER_ENDPOINT, data=payload)


@mcp.tool(name="exa_create_research")
async def exa_create_research(
    instructions: str,
    model: str = "exa-research",
    output_schema: Optional[dict[str, Any]] = None,
) -> str:
    """Legacy research creation tool kept for backwards compatibility."""
    cleaned_instructions, error = clean_string(instructions, "instructions")
    if error:
        return error

    if len(cleaned_instructions or "") > 4096:
        return "Error: instructions must be 4096 characters or less"

    error = validate_choice(model, "model", RESEARCH_MODELS)
    if error:
        return error

    if output_schema is not None and not isinstance(output_schema, dict):
        return "Error: output_schema must be a JSON object"

    payload = build_research_payload(
        cleaned_instructions or "",
        model,
        output_schema,
    )
    logger.info("Running exa_create_research with model '%s'", model)
    return await request_and_format(RESEARCH_ENDPOINT, data=payload)


@mcp.tool(name="exa_get_research")
async def exa_get_research(research_id: str) -> str:
    """Legacy research polling tool kept for backwards compatibility."""
    cleaned_research_id, error = clean_string(research_id, "research_id")
    if error:
        return error

    logger.info("Running exa_get_research for '%s'", cleaned_research_id)
    return await request_and_format(
        f"{RESEARCH_ENDPOINT}/{cleaned_research_id}",
        method="GET",
    )


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Run the MCP server using stdio transport."""
    logger.info("Starting Exa Pool MCP Server")
    if EXA_POOL_BASE_URL:
        logger.info(f"API configured: {EXA_POOL_BASE_URL[:20]}...")
    else:
        logger.warning("EXA_POOL_BASE_URL not set - server may not function correctly")
    logger.info("Transport: stdio")

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as exc:  # pragma: no cover - top-level logging
        logger.error(f"Server error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
