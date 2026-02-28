# RAG Optimization Complete (Phases 1-5)

**Date:** 2026-02-28

## Overview

Complete RAG (Retrieval-Augmented Generation) optimization to solve the fragmentary search results problem. Implements all 5 phases from the optimization plan, including semantic chunking, 3-step search pipeline, Cross-Encoder reranking, and tool integration.

## Problem

Previous RAG implementation had issues:
- Documents were chopped into tiny fragments
- Search results lacked context
- No awareness of document structure/sections
- No reranking for result quality

## Solution Implemented

### Phase 1: Data Layer & Context Expansion

**File:** `nanobot/rag/store.py` (already implemented before this phase)

- Extended `documents` table with `title`, `doc_type`, `abstract`
- Extended `chunks` table with `chunk_type`, `section_title`
- Implemented `_step1_core_chunk_recall()` - BM25 + vector with dual thresholds
- Implemented `_step2_context_expansion()` - prev 1 + core + next 1
- Implemented `_step3_document_level()` - Top 3 documents by average score
- Implemented `_merge_context_and_document_results()` - combine context and document results
- Added `avg_chunk_size`, `last_scan_at` to stats

### Phase 2: Semantic Chunking (Phase 2b)

**File:** `nanobot/rag/parser.py`

**New Dataclass:**
- `SemanticChunk`: `content`, `chunk_type`, `section_title`, `start_pos`, `end_pos`

**New Methods:**
- `extract_metadata()`: Extracts document title, doc_type (paper/lab_note/concept/other), and abstract
- `detect_section_headings()`: Detects section headings in multiple formats:
  - English: Abstract, 1. Introduction, 2. Related Work, 3. Method, 4. Experiments, 5. Results, 6. Conclusion, References
  - Chinese: 摘要, 1. 引言, 2. 相关工作, 3. 方法, 4. 实验, 5. 结果, 6. 结论, 参考文献
  - Markdown: #, ##, ### headings
- `chunk_by_paragraph_sentence()`: Phase 2a - chunks by paragraph/sentence boundaries with overlap
- `chunk_with_section_awareness()`: Phase 2b main method - prefers section boundaries, falls back to paragraph/sentence
- Backward compatibility maintained for `chunk_text()`

**Configuration:**
- `chunk_strategy`: "phase2b"
- `min_chunk_size`: 500
- `max_chunk_size`: 800
- `chunk_overlap_ratio`: 0.12 (12% overlap)

### Phase 3: 3-Step Search Pipeline

**File:** `nanobot/rag/store.py`

**Step 1: Core Chunk Recall (`_step1_core_chunk_recall()`)**
- Parallel BM25 full-text search + vector similarity search
- Dual thresholds: BM25 ≥ 0.6, vector ≥ 0.75
- Returns Top 5 core chunks using RRF (Reciprocal Rank Fusion)

**Step 2: Context Expansion (`_step2_context_expansion()`)**
- For each core chunk: retrieves prev 1 + core + next 1
- Combines into continuous context
- Configurable via `context_prev_chunks`, `context_next_chunks`

**Step 3: Document-Level Prioritization (`_step3_document_level()`)**
- Groups chunks by doc_id
- Calculates average score per document
- Returns Top 3 highest-scoring documents
- Configurable via `top_documents`

**Merge (`_merge_context_and_document_results()`)**
- Combines context-expanded chunks and document-level results
- Adds document bonus score

### Phase 4: Rerank & Deduplication (M4 Optimized)

**New File:** `nanobot/rag/rerank.py`

**Components:**
- `Reranker` (abstract base class): Interface for rerankers
- `CrossEncoderReranker`:
  - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, fast on M4)
  - Lazy model loading with thread safety
  - Runs in thread pool to avoid blocking async loop
  - Only reranks top-20 candidates (M4 performance optimization)
- `SemanticDeduplicator`:
  - Removes chunks with cosine similarity ≥ 0.9
  - Uses embedding-based comparison
- `RerankService`:
  - Orchestrates rerank + dedup pipeline
  - Applies rerank threshold (≥ 0.8)
  - Falls back gracefully if rerank unavailable

**Integration in `store.py`:**
- `_apply_rerank()` method applies rerank and dedup
- `search_advanced()` integrates all 4 phases
- `_rerank_service` initialized in `__init__`

**Configuration:**
- `enable_rerank`: true
- `rerank_model`: "cross-encoder/ms-marco-MiniLM-L-6-v2"
- `rerank_threshold`: 0.8
- `dedup_threshold`: 0.9
- `rerank_top_k`: 20

### Phase 5: Tool Optimization & Integration

**Files:**
- `nanobot/agent/tools/rag.py`: Already uses `search_advanced()` with context formatting
- `nanobot/cli/commands.py`: Fixed RAG CLI commands

**CLI Fixes:**
- `rag_refresh`: Now passes `rag_config` to DocumentStore
- `rag_rebuild`: Now passes `rag_config` to DocumentStore
- `rag_status`: DocumentStore now supports initialization without `embedding_provider` for read-only stats

**DocumentStore Improvements:**
- `embedding_provider` now optional (for read-only stats)
- `_vector_enabled` only set if `embedding_provider` provided
- `_init_schema` skips vector table creation if no embedding provider

## New Files

1. `nanobot/rag/rerank.py` - Reranker and deduplicator (Phase 4)
2. `tests/test_rag_parser.py` - 8 tests for Phase 2
3. `tests/test_rag_search.py` - 6 tests for Phase 3
4. `tests/test_rag_rerank.py` - 7 tests for Phase 4
5. `tests/test_rag_integration.py` - End-to-end integration test
6. `docs/changes/2026-02-28-rag-optimization-phase2-phase4.md` - Previous phase report
7. `docs/changes/2026-02-28-rag-optimization-complete.md` - This report

## Modified Files

1. `nanobot/rag/parser.py` - Added Phase 2b semantic chunking
2. `nanobot/rag/store.py` - Integrated Phase 3-4, fixed FTS issue, optional embedding_provider
3. `nanobot/rag/__init__.py` - Exported new classes (SemanticChunk, CrossEncoderReranker, etc.)
4. `nanobot/cli/commands.py` - Fixed RAG CLI commands to pass rag_config

## Complete Search Flow

```
User Query
    ↓
[1] Core Chunk Recall (Top 5, dual thresholds)
    ├─ BM25 full-text search (≥ 0.6)
    └─ Vector similarity search (≥ 0.75)
    ↓
[2] Context Expansion
    └─ For each core chunk: prev 1 + core + next 1
    ↓
[3] Document-Level Prioritization
    └─ Top 3 docs by average chunk score
    ↓
[4] Rerank & Dedup (M4 optimized)
    ├─ Cross-Encoder rerank (only top-20)
    ├─ Keep only ≥ 0.8
    └─ Semantic dedup (≥ 0.9)
    ↓
Final Results (with context, sections, doc info)
```

## Semantic Chunking Flow

```
Document Text
    ↓
[1] Detect Section Headings
    ├─ Abstract, Introduction, Method, etc.
    └─ Markdown #, ##, ###
    ↓
[2] Split at Section Boundaries
    └─ One chunk per section (if size ok)
    ↓
[3] Split Large Sections
    └─ By paragraph/sentence boundaries
    ↓
[4] Add Overlap (10-15%)
    ↓
Semantic Chunks (with section_title, chunk_type)
```

## Configuration

```json
{
  "tools": {
    "rag": {
      "enabled": true,
      "chunkStrategy": "phase2b",
      "minChunkSize": 500,
      "maxChunkSize": 800,
      "chunkOverlapRatio": 0.12,
      "enableContextExpansion": true,
      "contextPrevChunks": 1,
      "contextNextChunks": 1,
      "enableDocumentLevel": true,
      "topDocuments": 3,
      "bm25Threshold": 0.6,
      "vectorThreshold": 0.75,
      "enableRerank": true,
      "rerankThreshold": 0.8,
      "dedupThreshold": 0.9,
      "rerankTopK": 20,
      "rerankModel": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
  }
}
```

## Backward Compatibility

- All existing code continues to work unchanged
- `chunk_text()` maintains same signature and behavior
- DocumentStore constructor accepts optional `embedding_provider` for read-only use
- Config defaults maintain backward compatibility
- Old databases are automatically migrated (adds new columns)

## Verification

- All unit tests pass: `test_rag_parser.py` (8), `test_rag_search.py` (6), `test_rag_rerank.py` (7)
- Integration test passes: `test_rag_integration.py`
- CLI commands work: `rag refresh`, `rag rebuild`, `rag status`
- Agent tool integration works: `SearchKnowledgeTool` uses `search_advanced()`

## Files Summary

| File | Operation |
|------|-----------|
| `nanobot/rag/parser.py` | Modified (Phase 2b) |
| `nanobot/rag/store.py` | Modified (Phase 3-4, fixes) |
| `nanobot/rag/rerank.py` | New (Phase 4) |
| `nanobot/rag/__init__.py` | Modified (exports) |
| `nanobot/cli/commands.py` | Modified (CLI fixes) |
| `nanobot/agent/tools/rag.py` | Already integrated |
| `tests/test_rag_parser.py` | New |
| `tests/test_rag_search.py` | New |
| `tests/test_rag_rerank.py` | New |
| `tests/test_rag_integration.py` | New |
