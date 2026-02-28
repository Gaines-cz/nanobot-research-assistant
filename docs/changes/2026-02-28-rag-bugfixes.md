# RAG Bugfixes

**Date:** 2026-02-28

## Overview

Fixed critical and medium issues found in the RAG implementation.

## Issues Fixed

### 🔴 P0 Critical

#### Issue 1: `_step2_context_expansion` only fetched single chunk instead of looping

**Location:** `nanobot/rag/store.py` lines 740-756

**Problem:**
```python
# Only fetched chunk_idx - prev_count, not looping through prev_count
cursor = db.execute("""
    SELECT content FROM chunks
    WHERE doc_id = ? AND chunk_index = ?
""", (doc_id, chunk_idx - prev_count))
```

If `context_prev_chunks = 2`, code skipped the middle chunk and jumped directly to `chunk_idx - 2`.

**Fix:**
```python
# Get prev_count chunks before
prev_contents = []
for i in range(1, prev_count + 1):
    cursor = db.execute("""
        SELECT content FROM chunks
        WHERE doc_id = ? AND chunk_index = ?
    """, (doc_id, chunk_idx - i))
    prev_row = cursor.fetchone()
    if prev_row:
        prev_contents.insert(0, prev_row[0])  # Insert at beginning to maintain order
prev_content = "\n\n".join(prev_contents) if prev_contents else None

# Get next_count chunks after
next_contents = []
for i in range(1, next_count + 1):
    cursor = db.execute("""
        SELECT content FROM chunks
        WHERE doc_id = ? AND chunk_index = ?
    """, (doc_id, chunk_idx + i))
    next_row = cursor.fetchone()
    if next_row:
        next_contents.append(next_row[0])
next_content = "\n\n".join(next_contents) if next_contents else None
```

---

#### Issue 2: `rag.py` deleted database on every startup

**Location:** `nanobot/agent/tools/rag.py` lines 76-77

**Problem:**
```python
if db_path.exists():
    db_path.unlink()  # Deleted every time!
```

The database was deleted and all documents were reindexed every time the tool was invoked, leading to very poor user experience.

**Fix:**
- Removed the database deletion code entirely
- Database is now preserved between sessions

---

### 🟡 P1 Medium

#### Issue 3: `chunk_with_section_awareness` had incorrect overlap logic

**Location:** `nanobot/rag/parser.py` lines 418-427

**Problem:**
```python
for i in range(1, len(chunks)):
    prev_chunk = chunks[i - 1]
    curr_chunk = chunks[i]
    # Add overlap from previous chunk to current chunk
    overlap_text = prev_chunk.content[-overlap_size:]
    if overlap_text:
        # Prepend overlap, but adjust start_pos accordingly
        curr_chunk.content = overlap_text + curr_chunk.content  # Duplicate content!
        curr_chunk.start_pos -= len(overlap_text)
```

This caused duplicate content to be indexed, and overlap should be handled **during** chunking, not by prepending after chunking.

**Fix:**
- Removed the incorrect overlap logic entirely
- Overlap is now properly handled by `chunk_by_paragraph_sentence()` internally

---

#### Issue 4: `SECTION_PATTERNS` had hardcoded section numbers

**Location:** `nanobot/rag/parser.py` lines 29-69

**Problem:**
```python
(r"^\s*1\.\s+(Introduction)\s*$", "introduction"),  # Hardcoded 1.
(r"^\s*2\.\s+(Related\s+Work)\s*$", "introduction"),  # Hardcoded 2.
```

Would not match if paper section numbering didn't start at 1 or had different structure.

**Fix:**
```python
# English numbered sections - generic, no hardcoded numbers
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(Introduction)\s*$", "introduction"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(Related\s+Work|Background|Prior\s+Work)\s*$", "introduction"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(Method|Methods|Methodology|Approach)\s*$", "method"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(Experiment|Experiments|Experimental\s+Setup)\s*$", "experiment"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(Result|Results|Evaluation)\s*$", "result"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(Conclusion|Conclusions|Summary)\s*$", "conclusion"),
# Chinese numbered sections - generic, no hardcoded numbers
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(引言)\s*$", "introduction"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(相关工作|背景|文献综述)\s*$", "introduction"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(方法|算法|方法论)\s*$", "method"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(实验|实验设置)\s*$", "experiment"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(结果|评估)\s*$", "result"),
(r"^\s*(\d+(?:\.\d+)*)\.?\s+(结论|总结)\s*$", "conclusion"),
```

Also updated `detect_section_headings()` to correctly extract the title from patterns with number prefixes:
```python
# If there are capture groups, find the last non-numeric one as title
if match.groups():
    title = None
    for group in match.groups():
        if group and not re.match(r"^\d+(?:\.\d+)*$", group.strip()):
            title = group.strip()
            break
    if not title:
        title = line.strip()
else:
    title = line.strip()
```

---

### 🟢 P2 Minor

#### Issue 5: `rerank_and_dedup` only reranked top-20 in original order, not by score

**Status:** Deferred for now
- The current implementation is acceptable since the candidates already come sorted from the 3-step pipeline
- Requires additional data structure changes to properly handle

#### Issue 6: `_init_schema` had unnecessary ALTER TABLE statements

**Status:** Kept as-is
- The ALTER TABLE statements are needed for backward compatibility with existing databases
- `CREATE TABLE IF NOT EXISTS` doesn't add new columns to existing tables
- The try/except pattern safely skips ALTER TABLE if columns already exist

#### Issue 7: Document-level prioritization didn't extract "abstract + core chapters"

**Status:** Deferred for future enhancement
- Current implementation calculates document average scores correctly
- Full content extraction would require additional schema changes and design decisions

---

## Modified Files

1. `nanobot/agent/tools/rag.py` - Removed database deletion on startup
2. `nanobot/rag/store.py` - Fixed context expansion to loop through prev/next chunks
3. `nanobot/rag/parser.py` - Fixed section patterns (no hardcoded numbers), removed bad overlap logic

## Verification

- All 21 unit tests pass: `test_rag_parser.py` (8), `test_rag_search.py` (6), `test_rag_rerank.py` (7)

## Summary of Critical Fixes

1. ✅ **No more deleting database on every startup** - User experience drastically improved
2. ✅ **Context expansion works correctly** - Gets N chunks before/after, not just the Nth one
3. ✅ **No duplicate content from bad overlap** - Overlap handled properly during chunking
4. ✅ **Section detection more robust** - Works with any section numbering (1., 2., or 0., 1., or 1.1., etc.)
