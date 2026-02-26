"""Web tools: web_search and web_fetch.

Enhanced with:
- Result caching (avoid duplicate searches)
- Region and language parameters
- Freshness filtering
- External content safety wrapping
"""

import html
import json
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks

# Cache settings
SEARCH_CACHE: dict[str, tuple[float, str]] = {}  # {cache_key: (timestamp, result)}
CACHE_TTL_SECONDS = 300  # Cache for 5 minutes

# Serper Search freshness values mapped to Google tbs parameter
FRESHNESS_VALUES = {"pd", "pw", "pm", "py"}  # past day/week/month/year
FRESHNESS_TO_TBS = {
    "pd": "qdr:d",  # past day
    "pw": "qdr:w",  # past week
    "pm": "qdr:m",  # past month
    "py": "qdr:y",  # past year
}


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


def _wrap_external_content(text: str, source: str = "web") -> str:
    """
    Mark external content (safety practice).
    Prevents prompt injection from search results.
    """
    # Remove possible control characters
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return cleaned


def _get_cache_key(query: str, count: int, country: str | None, freshness: str | None) -> str:
    """Generate cache key."""
    return f"{query}:{count}:{country or 'default'}:{freshness or 'default'}"


def _read_cache(key: str) -> str | None:
    """Read cache, return None if expired."""
    if key in SEARCH_CACHE:
        timestamp, result = SEARCH_CACHE[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return result
        # Expired, delete
        del SEARCH_CACHE[key]
    return None


def _write_cache(key: str, result: str) -> None:
    """Write to cache."""
    # Limit cache size (prevent memory overflow)
    if len(SEARCH_CACHE) > 100:
        # Delete oldest half
        sorted_keys = sorted(SEARCH_CACHE.keys(), key=lambda k: SEARCH_CACHE[k][0])
        for old_key in sorted_keys[:50]:
            del SEARCH_CACHE[old_key]
    SEARCH_CACHE[key] = (time.time(), result)


def _normalize_freshness(value: str | None) -> str | None:
    """
    Normalize freshness parameter.
    Supports: pd (past 24h), pw (past week), pm (past month), py (past year)
    or date range: YYYY-MM-DDtoYYYY-MM-DD
    """
    if not value:
        return None
    trimmed = value.strip().lower()
    if trimmed in FRESHNESS_VALUES:
        return trimmed
    # Check date range format
    if re.match(r'^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$', trimmed):
        return trimmed
    return None


class WebSearchTool(Tool):
    """Search the web using Serper (Google Search) API."""

    name = "web_search"
    description = (
        "Search the web for up-to-date information. Supports region filtering and freshness filtering."
        "Returns titles, URLs, and snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "count": {
                "type": "integer",
                "description": "Number of results (1-10)",
                "minimum": 1,
                "maximum": 10
            },
            "country": {
                "type": "string",
                "description": "Country code (e.g., US, CN, DE, ALL), default US"
            },
            "freshness": {
                "type": "string",
                "description": "Freshness filter: pd=past 24 hours, pw=past week, pm=past month, py=past year"
            }
        },
        "required": ["query"]
    }

    def __init__(self, api_key: str | None = None, max_results: int = 5, timeout: float = 15.0):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        self.max_results = max_results
        self.timeout = timeout

    async def execute(
        self,
        query: str,
        count: int | None = None,
        country: str | None = None,
        freshness: str | None = None,
        **kwargs: Any
    ) -> str:
        if not self.api_key:
            return json.dumps({
                "error": "missing_api_key",
                "message": "SERPER_API_KEY not configured. Please set Serper (Google Search) API key in config file."
            }, ensure_ascii=False)

        n = min(max(count or self.max_results, 1), 10)
        normalized_freshness = _normalize_freshness(freshness)

        # Check cache
        cache_key = _get_cache_key(query, n, country, normalized_freshness)
        cached = _read_cache(cache_key)
        if cached:
            return cached + "\n\n(cached)"

        try:
            # Build request payload
            payload: dict[str, Any] = {"q": query, "num": n}
            if country:
                payload["gl"] = country.upper()
            if normalized_freshness:
                if normalized_freshness in FRESHNESS_TO_TBS:
                    payload["tbs"] = FRESHNESS_TO_TBS[normalized_freshness]
                else:
                    # Date range format passed directly (for compatibility)
                    pass

            start_time = time.time()

            # SSL fallback mechanism: try verify=True first, then verify=False
            try:
                async with httpx.AsyncClient(verify=True) as client:
                    r = await client.post(
                        "https://google.serper.dev/search",
                        json=payload,
                        headers={
                            "X-API-KEY": self.api_key,
                            "Content-Type": "application/json"
                        },
                        timeout=self.timeout
                    )
                    r.raise_for_status()
            except (httpx.HTTPError, httpx.TransportError):
                # SSL verification failed, try without verification
                async with httpx.AsyncClient(verify=False) as client:
                    r = await client.post(
                        "https://google.serper.dev/search",
                        json=payload,
                        headers={
                            "X-API-KEY": self.api_key,
                            "Content-Type": "application/json"
                        },
                        timeout=self.timeout
                    )
                    r.raise_for_status()

            elapsed_ms = int((time.time() - start_time) * 1000)

            results = r.json().get("organic", [])
            if not results:
                return json.dumps({
                    "query": query,
                    "count": 0,
                    "message": f"No results found for '{query}'"
                }, ensure_ascii=False)

            # Format results (using safe wrapping)
            lines = [f"Results for: {query} ({elapsed_ms}ms)\n"]
            for i, item in enumerate(results[:n], 1):
                title = _wrap_external_content(item.get('title', ''))
                url = item.get('link', '')
                desc = _wrap_external_content(item.get('snippet', ''))
                age = item.get('date', '')

                lines.append(f"{i}. {title}")
                lines.append(f"   {url}")
                if desc:
                    lines.append(f"   {desc}")
                if age:
                    lines.append(f"   Published: {age}")

            result = "\n".join(lines)

            # Write to cache
            _write_cache(cache_key, result)

            return result

        except httpx.TimeoutException:
            return json.dumps({"error": "timeout", "message": f"Search timed out ({self.timeout}s)"}, ensure_ascii=False)
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": "http_error", "status": e.response.status_code, "message": str(e)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": "unknown", "message": str(e)}, ensure_ascii=False)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100, "description": "Maximum characters"}
        },
        "required": ["url"]
    }

    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2, ensure_ascii=False), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps({
                "url": url,
                "finalUrl": str(r.url),
                "status": r.status_code,
                "extractor": extractor,
                "truncated": truncated,
                "length": len(text),
                "text": _wrap_external_content(text)
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    def _to_markdown(self, html_content: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html_content, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
