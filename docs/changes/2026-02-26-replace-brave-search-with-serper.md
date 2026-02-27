# Replace Brave Search with Serper

**Date:** 2026-02-26

## Overview

Replaced the web search tool from Brave Search to Serper (Google Search API), with enhanced features.

## Files Modified

### 1. `nanobot/agent/tools/web.py`
- **Changed**: `WebSearchTool` now uses Serper API instead of Brave Search
- **API Endpoint**: `https://google.serper.dev/search` (POST request)
- **Authentication**: `X-API-KEY` header instead of `X-Subscription-Token`
- **Environment Variable**: `SERPER_API_KEY` instead of `BRAVE_API_KEY`
- **Response Parsing**: Uses `organic` array with `link` field instead of `web.results` with `url`

### Enhancements Added:
- **Result Caching**: 5-minute TTL cache to avoid duplicate searches
- **Country Parameter**: Region filtering (e.g., US, CN, DE)
- **Freshness Parameter**: Time-based filtering (pd=past 24h, pw=past week, pm=past month, py=past year)
- **Content Safety**: `_wrap_external_content` to prevent prompt injection
- **SSL Fallback**: Tries `verify=True` first, falls back to `verify=False` on failure
- **Bug Fix**: Fixed the undefined `api_key` variable bug (now uses `self.api_key`)

### 2. `nanobot/config/schema.py`
- Updated comment in `WebSearchConfig` from "Brave Search API key" to "Serper API key"

### 3. `nanobot/agent/loop.py`
- Renamed parameter `brave_api_key` → `serper_api_key` throughout

### 4. `nanobot/agent/subagent.py`
- Renamed parameter `brave_api_key` → `serper_api_key` throughout

### 5. `nanobot/cli/commands.py`
- Renamed parameter `brave_api_key` → `serper_api_key` throughout

### 6. `README.md`
- Updated documentation: "Serper" instead of "Brave Search"
- Updated link: https://serper.dev/

### 7. `CLAUDE.md`
- Updated documentation: "web_search (Serper)" instead of "web_search (Brave)"

## Configuration

Users should set their Serper API key in:
- Config file: `~/.nanobot/config.json` under `tools.web.search.apiKey`
- Or environment variable: `SERPER_API_KEY`

Get API key at: https://serper.dev/

## Verification

- All imports work successfully
- `nanobot status` runs without errors
