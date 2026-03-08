import time
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

_scrape_cache: dict[str, tuple[float, str]] = {}
CACHE_TTL = 900  # 15 minutes


def make_web_tools(firecrawl_api_key: str | None = None) -> list[BaseTool]:
    @tool
    def WebFetch(url: str, prompt: str) -> str:
        """
- Fetches content from a specified URL and processes it using an AI model
- Takes a URL and a prompt as input
- Fetches the URL content, converts HTML to markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions. All MCP-provided tools start with "mcp__".
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - The prompt should describe what information you want to extract from the page
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large
  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL
  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content."""
        if not firecrawl_api_key:
            return "Error: FIRECRAWL_API_KEY is not configured. Set it in your .env file."

        if url.startswith("http://"):
            url = "https://" + url[7:]

        now = time.time()
        cached = _scrape_cache.get(url)
        if cached and (now - cached[0]) < CACHE_TTL:
            markdown_content = cached[1]
        else:
            try:
                from firecrawl import Firecrawl
                fc = Firecrawl(api_key=firecrawl_api_key)
                result = fc.scrape(url, formats=["markdown"])
            except Exception as exc:
                return f"Error fetching URL: {exc}"

            markdown_content = ""
            if isinstance(result, dict):
                markdown_content = result.get("markdown", "")
            elif hasattr(result, "markdown"):
                markdown_content = result.markdown or ""

            if not markdown_content:
                return "Error: No content could be extracted from the URL."

            _scrape_cache[url] = (now, markdown_content)
            _clean_cache(now)

        if len(markdown_content) > 50000:
            markdown_content = markdown_content[:50000] + "\n\n... (content truncated)"

        return f"# Content from {url}\n\n{markdown_content}\n\n---\nPrompt: {prompt}"

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
            results = fc.search(search_query, limit=5)
        except Exception as exc:
            return f"Error performing search: {exc}"

        web_results = []
        if isinstance(results, dict):
            data = results.get("data", results)
            if isinstance(data, dict):
                web_results = data.get("web", [])
            elif isinstance(data, list):
                web_results = data
        elif isinstance(results, list):
            web_results = results
        elif hasattr(results, "data"):
            web_results = results.data if isinstance(results.data, list) else []

        if not web_results:
            return "No search results found."

        lines: list[str] = []
        for i, item in enumerate(web_results, 1):
            if isinstance(item, dict):
                title = item.get("title", "No title")
                item_url = item.get("url", "")
                description = item.get("description", "")
            elif hasattr(item, "title"):
                title = getattr(item, "title", "No title")
                item_url = getattr(item, "url", "")
                description = getattr(item, "description", "")
            else:
                continue

            lines.append(f"{i}. [{title}]({item_url})")
            if description:
                lines.append(f"   {description}")
            lines.append("")

        return "\n".join(lines) if lines else "No search results found."

    return [WebFetch, WebSearch]


def _clean_cache(now: float) -> None:
    expired = [k for k, (ts, _) in _scrape_cache.items() if (now - ts) >= CACHE_TTL]
    for k in expired:
        del _scrape_cache[k]
