# Add RAG (Retrieval-Augmented Generation) Functionality

**Date:** 2026-02-26

## Overview

Added RAG (Retrieval-Augmented Generation) functionality to nanobot, enabling the agent to search and reference local documents (PDF, Markdown, Word) using hybrid search (vector + full-text).

## Features

- **Hybrid Search**: Combines vector similarity search with SQLite FTS5 full-text search
- **Local Embeddings**: Uses sentence-transformers for local, API-free embeddings
- **Auto-scan**: Automatically scans documents on startup, detects changes via file mtime
- **Multi-format Support**: PDF, Markdown, Word (.docx), and plain text files
- **Directory Structure**: Supports nested directories under `workspace/docs/`
- **SQLite-based Storage**: Uses sqlite-vec for vectors, FTS5 for full-text

## New Files

### 1. `nanobot/rag/__init__.py`
- RAG module initialization
- Exports: `DocumentParser`, `DocumentStore`, `EmbeddingProvider`, `SentenceTransformerEmbeddingProvider`, `SearchResult`

### 2. `nanobot/rag/embeddings.py`
- `EmbeddingProvider`: Abstract base class for embedding providers
- `SentenceTransformerEmbeddingProvider`: Local embedding using sentence-transformers
  - Lazy model loading with caching
  - Thread-safe model initialization
  - Runs in thread pool to avoid blocking async loop

### 3. `nanobot/rag/parser.py`
- `DocumentParser`: Parse various document formats
  - `parse_pdf()`: Uses pypdf
  - `parse_word()`: Uses python-docx
  - `parse_markdown()`: Plain text read
  - `parse_text()`: Fallback for other text formats
- `chunk_text()`: Splits text at sentence boundaries with configurable size/overlap

### 4. `nanobot/rag/store.py`
- `DocumentStore`: Unified document storage and retrieval
- `SearchResult`: Dataclass for search results

**SQLite Schema**:
- `documents`: Document metadata (path, filename, mtime, etc.)
- `chunks`: Text chunks with position info
- `chunk_embeddings`: sqlite-vec virtual table for vector search
- `chunks_fts`: FTS5 virtual table for full-text search

**Key Methods**:
- `scan_and_index()`: Scans docs, detects adds/updates/deletes via mtime
- `search()`: Hybrid search with reranking
- `_vector_search()`: Cosine similarity search
- `_fulltext_search()`: BM25-based full-text search

### 5. `nanobot/agent/tools/rag.py`
- `RetrieveTool`: The `retrieve` tool for the agent
- Parameters: `query` (required), `top_k` (optional, default: 5)
- Auto-initializes document store on first use

## Modified Files

### 1. `pyproject.toml`
Added `rag` optional dependency group:
```toml
rag = [
    "sqlite-vec>=0.1.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "sentence-transformers>=3.0.0",
]
```

### 2. `nanobot/config/schema.py`
Added `RAGConfig` class:
```python
class RAGConfig(Base):
    enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    auto_scan_on_startup: bool = True
```

Added to `ToolsConfig`:
```python
class ToolsConfig(Base):
    # ...
    rag: RAGConfig = Field(default_factory=RAGConfig)
```

### 3. `nanobot/agent/loop.py`
- Added import for `RetrieveTool`
- Added `rag_config` parameter to `AgentLoop.__init__`
- Added `_rag_initialized` flag and `_retrieve_tool` reference
- `_register_default_tools()`: Registers `RetrieveTool` if enabled
- `_init_rag()`: New method to scan documents on first use
- `run()` and `process_direct()`: Call `_init_rag()`

### 4. `nanobot/cli/commands.py`
- Added `rag_config=config.tools.rag` to all 3 `AgentLoop` initializations:
  - Gateway command
  - Agent command
  - Cron run command

## How It Works

### Document Uniqueness
- Identified by **absolute file path** (UNIQUE constraint in SQLite)
- Same filename in different directories are treated as different documents

### Change Detection
- Uses file `mtime` (last modified time)
- On each scan:
  - **Added**: File exists but not in DB
  - **Updated**: File `mtime > stored_at`
  - **Deleted**: In DB but file doesn't exist

### Hybrid Search & Reranking (RRF)
Uses **RRF (Reciprocal Rank Fusion)** for robust results:

1. Run vector search and full-text search in parallel, each returning top_k * 2 results
2. Apply RRF formula: `score = 1 / (k + rank)`, where `k = 60`
3. Accumulate scores for chunks that appear in both result lists
4. Sort by combined RRF score, return top_k

**Why RRF?**
- No manual weight tuning needed
- Robust to different score scales between vector and full-text
- Well-proven in information retrieval

## Usage

### Installation
```bash
pip install -e ".[rag]"
```

### Adding Documents
Place documents in:
```
~/.nanobot/workspace/docs/
├── ai/
│   ├── paper1.pdf
│   └── notes.md
└── projects/
    └── proposal.docx
```

### Configuration
```json
{
  "tools": {
    "rag": {
      "enabled": true,
      "chunkSize": 1000,
      "chunkOverlap": 200,
      "topK": 5,
      "embeddingModel": "all-MiniLM-L6-v2",
      "autoScanOnStartup": true
    }
  }
}
```

### Agent Usage
The agent automatically uses the `retrieve` tool when it needs to reference your documents.

## Verification
- All files pass Python syntax check
- Imports resolve correctly
- Configuration schema loads with defaults
