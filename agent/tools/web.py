import time
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

_scrape_cache: dict[str, tuple[float, str, str]] = {}
CACHE_TTL = 900  # 15 minutes


def make_web_tools(firecrawl_api_key: str | None = None) -> list[BaseTool]:
    @tool
    def WebFetch(url: str) -> str:
        """
- Fetches content from a URL via Firecrawl scrape and returns markdown.
- Use this tool when you need live page content.

Usage notes:
  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions. All MCP-provided tools start with "mcp__".
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large
  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL."""
        if not firecrawl_api_key:
            return "Error: FIRECRAWL_API_KEY is not configured. Set it in your .env file."

        if url.startswith("http://"):
            url = "https://" + url[7:]

        now = time.time()
        cached = _scrape_cache.get(url)
        if cached and (now - cached[0]) < CACHE_TTL:
            source_url = cached[1]
            markdown_content = cached[2]
        else:
            try:
                from firecrawl import Firecrawl
                fc = Firecrawl(api_key=firecrawl_api_key)
                result = fc.scrape(url, formats=["markdown"])
            except Exception as exc:
                return f"Error fetching URL: {exc}"

            result_data = _to_dict(result)
            markdown_content = _extract_markdown(result, result_data)

            if not markdown_content:
                return "Error: No content could be extracted from the URL."

            source_url = _extract_source_url(url, result, result_data)
            _scrape_cache[url] = (now, source_url, markdown_content)
            _clean_cache(now)

        if len(markdown_content) > 50000:
            markdown_content = markdown_content[:50000] + "\n\n... (content truncated)"

        return f"# Content from {source_url}\n\n{markdown_content}"

    @tool
    def WebSearch(
        query: str,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
    ) -> str:
        """
- Allows Claude to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks
- Use this tool for accessing information beyond Claude's knowledge cutoff
- Searches are performed automatically within a single API call

Usage notes:
  - Domain filtering is supported to include or block specific websites
  - Web search is only available in the US
  - Account for "Today's date" in <env>. For example, if <env> says "Today's date: 2025-07-01", and the user wants the latest docs, do not use 2024 in the search query. Use 2025."""
        if not firecrawl_api_key:
            return "Error: FIRECRAWL_API_KEY is not configured. Set it in your .env file."

        search_query = query
        if allowed_domains:
            domain_filters = " ".join(f"site:{d}" for d in allowed_domains)
            search_query = f"{query} {domain_filters}"
        if blocked_domains:
            domain_filters = " ".join(f"-site:{d}" for d in blocked_domains)
            search_query = f"{search_query} {domain_filters}"

        try:
            from firecrawl import Firecrawl
            fc = Firecrawl(api_key=firecrawl_api_key)
            results = fc.search(search_query, limit=5, sources=["web"])
        except Exception as exc:
            return f"Error performing search: {exc}"

        web_results = _extract_web_results(results)

        if not web_results:
            return "No search results found."

        lines: list[str] = []
        for i, item in enumerate(web_results, 1):
            item_data = _to_dict(item) if not isinstance(item, dict) else item
            if not item_data:
                continue

            title = str(item_data.get("title") or "No title")
            item_url = str(item_data.get("url") or "")
            description = str(item_data.get("description") or "")

            lines.append(f"{i}. [{title}]({item_url})")
            if description:
                lines.append(f"   {description}")
            lines.append("")

        return "\n".join(lines) if lines else "No search results found."

    return [WebFetch, WebSearch]


def _to_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped

    return {}


def _extract_markdown(result: object, result_data: dict[str, object]) -> str:
    markdown = result_data.get("markdown")
    if isinstance(markdown, str) and markdown:
        return markdown

    markdown_attr = getattr(result, "markdown", None)
    if isinstance(markdown_attr, str):
        return markdown_attr

    return ""


def _extract_source_url(
    fallback_url: str,
    result: object,
    result_data: dict[str, object],
) -> str:
    metadata = result_data.get("metadata")
    metadata_data = _to_dict(metadata) if not isinstance(metadata, dict) else metadata

    source_url = metadata_data.get("source_url")
    if isinstance(source_url, str) and source_url:
        return source_url

    url_value = metadata_data.get("url")
    if isinstance(url_value, str) and url_value:
        return url_value

    source_url_attr = getattr(getattr(result, "metadata", None), "source_url", None)
    if isinstance(source_url_attr, str) and source_url_attr:
        return source_url_attr

    return fallback_url


def _extract_web_results(results: object) -> list[dict[str, object]]:
    results_data = _to_dict(results)
    payload: object = results_data

    if isinstance(results_data.get("data"), dict):
        payload = results_data["data"]

    web_results: object = None
    if isinstance(payload, dict):
        web_results = payload.get("web")
    if not isinstance(web_results, list):
        web_results = getattr(results, "web", None)

    if not isinstance(web_results, list):
        return []

    normalized: list[dict[str, object]] = []
    for item in web_results:
        item_data = _to_dict(item) if not isinstance(item, dict) else item
        if item_data:
            normalized.append(item_data)

    return normalized


def _clean_cache(now: float) -> None:
    expired = [k for k, (ts, _, _) in _scrape_cache.items() if (now - ts) >= CACHE_TTL]
    for k in expired:
        del _scrape_cache[k]
