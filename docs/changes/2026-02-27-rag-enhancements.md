# RAG Enhancements

**Date:** 2026-02-27

## Overview

Subsequent enhancements to the RAG functionality added after initial implementation.

## Changes

### 1. Tool Rename: `retrieve` → `search_knowledge`

**Files Modified:**
- `nanobot/agent/tools/rag.py`
- `nanobot/agent/loop.py`

**Changes:**
- Renamed tool from `retrieve` to `search_knowledge`
- Updated description to be more specific about semantic search

### 2. CLI Commands: `rag refresh`, `rag status`, `rag rebuild`

**File Modified:**
- `nanobot/cli/commands.py`

**New Commands:**

| Command | Description |
|---------|-------------|
| `nanobot rag refresh` | Scan for new/changed/deleted documents |
| `nanobot rag status` | Show index statistics |
| `nanobot rag rebuild` | Delete existing index and rebuild from scratch |

### 3. FTS5 Query Sanitization

**File Modified:**
- `nanobot/rag/store.py`

**Problem:**
FTS5 special characters (`"()`, `)*#^`-:{}`) in search queries caused syntax errors.

**Solution:**
Added `_sanitize_fts_query()` method to filter special characters before executing FTS5 queries.

### 4. Graceful Vector Search Fallback

**File Modified:**
- `nanobot/rag/store.py`

**Changes:**
- `embedding_provider` is now optional in `DocumentStore.__init__()`
- Vector search automatically falls back to full-text only if:
  - sqlite-vec extension not available
  - macOS system Python (limited extension support)
  - Vector table creation fails
- Added `_vector_enabled` flag to track availability
- All vector operations guarded with `_vector_enabled` checks

### 5. Optimized `rag status` Command

**File Modified:**
- `nanobot/cli/commands.py`

**Improvement:**
- Removed unnecessary embedding model loading for status command
- Now uses `DocumentStore(db_path)` without embedding provider for faster execution

## Configuration

No configuration changes required. Existing configuration continues to work.

## Verification

```bash
# Check rag commands
nanobot rag --help

# Refresh index
nanobot rag refresh

# Check status
nanobot rag status

# Rebuild index (with confirmation)
nanobot rag rebuild
```
