# RAG CLI Search Command & Threshold Fix

**Date:** 2026-02-28

## Changes

### 1. New Command: `nanobot rag search`

Added direct search command to CLI for quick document lookup.

**Usage:**
```bash
# Basic search
nanobot rag search "transformer attention"

# Specify top-k
nanobot rag search "query" --top-k 10
nanobot rag search "query" -k 10
```

**Output:**
```
Results for: neural networks

[1] 2502.00897v1 (score: 1.50)
    Section: References
    a multiscale fourier feature physics-informed
neural network with adaptive activation functions...
```

### 2. Fixed: Search Threshold Too High

**Problem:**
Search returned 0 results even when matching content existed in database.

**Root Cause:**
- BM25 scores are negative (more negative = more relevant)
- Normalized BM25 scores were ~0.1-0.2
- Default threshold was 0.6, which filtered out all results

**Solution:**
Updated default thresholds in `nanobot/config/schema.py`:

| Threshold | Before | After |
|-----------|--------|-------|
| bm25_threshold | 0.6 | 0.05 |
| vector_threshold | 0.75 | 0.3 |
| rerank_threshold | 0.8 | 0.5 |
| dedup_threshold | 0.9 | 0.7 |

### 3. Improved: rag status Command

Enhanced status command to show vector search capability status.

**Output:**
```
Search Capabilities
  Vector search: enabled
```

## Files Modified

| File | Change |
|------|--------|
| `nanobot/cli/commands.py` | Added `rag_search()` command |
| `nanobot/config/schema.py` | Lowered default thresholds |

## Verification

```bash
# Test search
nanobot rag search "neural networks"

# Check status
nanobot rag status
```