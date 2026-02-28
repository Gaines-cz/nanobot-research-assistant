# RAG Optimization: Phase 2-4 Implementation

**Date:** 2026-02-28

## Overview

Implemented Phases 2-4 of the RAG optimization plan to solve the fragmentary search results problem. This adds semantic chunking with section awareness, the complete 3-step search pipeline, and Cross-Encoder reranking with deduplication.

## Features Added

### Phase 2: Semantic Chunking (Phase 2b)

**File:** `nanobot/rag/parser.py`

- **`SemanticChunk` dataclass**: New dataclass with `chunk_type`, `section_title`, `start_pos`, `end_pos`
- **`extract_metadata()`**: Extracts document title, doc_type (paper/lab_note/concept/other), and abstract
- **`detect_section_headings()`**: Detects section headings in multiple formats:
  - English: Abstract, 1. Introduction, 2. Related Work, 3. Method, 4. Experiments, 5. Results, 6. Conclusion, References
  - Chinese: 摘要, 1. 引言, 2. 相关工作, 3. 方法, 4. 实验, 5. 结果, 6. 结论, 参考文献
  - Markdown: #, ##, ### headings
- **`chunk_by_paragraph_sentence()`**: Phase 2a - chunks by paragraph/sentence boundaries with overlap
- **`chunk_with_section_awareness()`**: Phase 2b main method - prefers section boundaries, falls back to paragraph/sentence
- **Backward compatibility**: `chunk_text()` still works as before

### Phase 3: 3-Step Search Pipeline

**File:** `nanobot/rag/store.py`

All three steps are already implemented and integrated:

- **Step 1: Core Chunk Recall** (`_step1_core_chunk_recall()`)
  - Parallel BM25 + vector search
  - Dual thresholds: BM25 ≥ 0.6, vector ≥ 0.75
  - Returns Top 5 core chunks using RRF fusion

- **Step 2: Context Expansion** (`_step2_context_expansion()`)
  - For each core chunk: retrieves prev 1 + core + next 1
  - Combines into continuous context
  - Configurable via `context_prev_chunks`, `context_next_chunks`

- **Step 3: Document-Level Prioritization** (`_step3_document_level()`)
  - Groups chunks by doc_id
  - Calculates average score per document
  - Returns Top 3 highest-scoring documents

- **Merge** (`_merge_context_and_document_results()`)
  - Combines context-expanded chunks and document-level results
  - Adds document bonus score

### Phase 4: Rerank & Deduplication (M4 Optimized)

**New File:** `nanobot/rag/rerank.py`

- **`Reranker` (abstract base class)**: Interface for rerankers
- **`CrossEncoderReranker`**:
  - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, fast on M4)
  - Lazy model loading with thread safety
  - Runs in thread pool to avoid blocking async loop
  - Only reranks top-20 candidates (M4 performance optimization)
- **`SemanticDeduplicator`**:
  - Removes chunks with cosine similarity ≥ 0.9
  - Uses embedding-based comparison
- **`RerankService`**:
  - Orchestrates rerank + dedup pipeline
  - Applies rerank threshold (≥ 0.8)
  - Falls back gracefully if rerank unavailable

**Integration in `store.py`:**
- `_apply_rerank()` method applies rerank and dedup
- `search_advanced()` integrates all 4 phases

### Configuration Schema

**File:** `nanobot/config/schema.py`

All config fields already in place (from Phase 1):

```python
class RAGConfig(Base):
    # Chunking (Phase 2b)
    chunk_strategy: str = "phase2b"
    min_chunk_size: int = 500
    max_chunk_size: int = 800
    chunk_overlap_ratio: float = 0.12  # 12% overlap

    # Context expansion (Phase 3)
    enable_context_expansion: bool = True
    context_prev_chunks: int = 1
    context_next_chunks: int = 1

    # Document-level (Phase 3)
    enable_document_level: bool = True
    top_documents: int = 3

    # Thresholds
    bm25_threshold: float = 0.6
    vector_threshold: float = 0.75
    rerank_threshold: float = 0.8
    dedup_threshold: float = 0.9

    # Reranker (Phase 4, M4 optimized)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_rerank: bool = True
    rerank_top_k: int = 20  # Only rerank top-20
```

### Tool Integration

**File:** `nanobot/agent/tools/rag.py`

- Already uses `search_advanced()` for improved results
- `_format_advanced_results()` shows section titles and combined content

### Bug Fixes

- **Fixed FTS5 query issue**: Added proper fallback for empty sanitized queries
- **Added multiple fallbacks**: FTS with BM25 → FTS without BM25 → LIKE query

## New Files

1. `nanobot/rag/rerank.py` - Reranker and deduplicator
2. `tests/test_rag_parser.py` - Tests for Phase 2
3. `tests/test_rag_search.py` - Tests for Phase 3
4. `tests/test_rag_rerank.py` - Tests for Phase 4
5. `tests/test_rag_integration.py` - End-to-end integration tests

## Modified Files

1. `nanobot/rag/parser.py` - Added Phase 2b semantic chunking
2. `nanobot/rag/store.py` - Integrated Phase 3-4, fixed FTS issue
3. `nanobot/rag/__init__.py` - Exported new classes

## How It Works

### Complete Search Flow

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

### Semantic Chunking Flow

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

## Verification

- All unit tests pass: `test_rag_parser.py`, `test_rag_search.py`, `test_rag_rerank.py`
- Integration test passes: `test_rag_integration.py`
- Backward compatibility maintained for existing code

## Usage

The optimization is enabled by default with the config defaults. No changes required for existing users - they automatically get better search results!

To customize (optional):
```json
{
  "tools": {
    "rag": {
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
      "rerankTopK": 20
    }
  }
}
```
