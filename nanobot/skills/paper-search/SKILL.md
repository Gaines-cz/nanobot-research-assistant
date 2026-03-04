---
name: paper-search
description: Search and download papers to supplement local knowledge base. Checks for paper-search MCP tools, expands keywords with explicit user confirmation, downloads to ~/.nanobot/workspace/docs/${domain}/ and runs nanobot rag refresh to index.
---

# Paper Search & Download

Use this skill when the user explicitly asks to search and download papers.

## Trigger Phrases

- "Help me search and download papers to the local knowledge base"
- "Search for papers and download to the knowledge base"
- "Find some relevant papers and save them locally"
- "Download papers to the docs directory"

## Step 1: Check MCP Tools

First check if paper-search related tools are available:
- Search tools: `mcp_paper_search_server_search_*` (e.g., `search_arxiv`, `search_semantic_scholar`)
- Download tools: `mcp_paper_search_server_download_*`

**If no tools found:**
- Tell user to install https://github.com/openags/paper-search-mcp
- Explain that they need to add this MCP server to their config
- Stop the workflow

## Step 2: Keyword Expansion & Explicit Confirmation

**LLM performs keyword expansion**:
1. Extract research topic from user request
2. LLM generates 3-5 keywords (synonyms, related terms)
3. Present to user for **explicit confirmation**

**Example:**
- User: "large language models"
- Expanded: ["large language models", "LLM", "transformer", "pretrained models"]

**Confirmation rules (explicit confirmation required for file downloads):**
- User explicitly says "yes", "ok", "go ahead", "sure" → proceed
- User has modifications → adjust and ask for confirmation again
- User doesn't reply or gives vague response → ask again, NO silent confirmation

## Step 3: Determine Domain Directory

Determine save directory based on topic:
- Large language models → `llm/`
- Computer vision → `cv/`
- Multimodal → `multimodal/`
- Reinforcement learning → `rl/`
- Other topics → concise English keywords (e.g., `nlp/`, `robotics/`, `ai-safety/`)

Full path: `~/.nanobot/workspace/docs/${domain}/`

## Step 4: Search for Papers

**Search strategy:**
- Use multiple keywords with OR logic
- Return 5-10 papers per source is sufficient
- Prioritize authoritative sources like arXiv, Semantic Scholar
- Deduplication: check duplicates by title + author

**Result display format:**
```
Found the following papers:

1. [2024] Attention Is All You Need
   Authors: Vaswani et al.
   Abstract: ...
   Source: arXiv

2. ...
```

Ask user to select papers to download (specify numbers like "1, 3, 5", or "download all").

## Step 5: Download Papers

1. Use `pathlib.Path.home()` to get user's home directory, create target directory if it doesn't exist
2. Use corresponding download tool to download selected papers
3. Save to `~/.nanobot/workspace/docs/${domain}/`
4. Use paper title as filename, replace special characters with underscores

**Error handling:**
- Tool call failed → retry once, skip and continue to next paper if still fails
- Download link invalid → skip the paper, log a note
- Network timeout → wait 2 seconds then retry once

## Step 6: Update Knowledge Base Index

After download completes, use `exec` tool to run:
```bash
nanobot rag refresh
```

This will scan the `docs/` directory and index new papers into the RAG knowledge base, so user can search these new papers with the `search_knowledge` tool.
