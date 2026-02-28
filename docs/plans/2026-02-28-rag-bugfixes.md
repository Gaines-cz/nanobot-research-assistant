# RAG 实现 Bug 修复方案

**Date:** 2026-02-28

## 概述

本文档列出了当前 RAG 实现中发现的所有问题及其修复方案。

---

## 🔴 P0 严重问题

### Issue 1: `_step2_context_expansion` 只取单 chunk，没有循环获取

**位置**: `nanobot/rag/store.py` lines 740-756

**问题描述**:
```python
# 只取 chunk_idx - prev_count，不是循环取 prev_count 个
cursor = db.execute("""
    SELECT content FROM chunks
    WHERE doc_id = ? AND chunk_index = ?
""", (doc_id, chunk_idx - prev_count))
```

如果 `context_prev_chunks = 2`，代码会跳过中间的 chunk，直接跳到 `chunk_idx - 2`。

**修复方案**:

```python
def _step2_context_expansion(self, core_results: list[SearchResult]) -> list[ChunkInfo]:
    """
    Step 2: Expand context around core chunks.
    Returns chunks with previous and next chunks included.
    """
    if not self.config.enable_context_expansion:
        # ... 保持不变 ...

    db = self._get_db()
    expanded_chunks = []
    prev_count = self.config.context_prev_chunks
    next_count = self.config.context_next_chunks

    for result in core_results:
        cursor = db.execute("""
            SELECT c.id, c.doc_id, c.chunk_index, c.content, c.chunk_type, c.section_title
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE d.path = ? AND c.chunk_index = ?
        """, (result.path, result.chunk_index))
        row = cursor.fetchone()
        if not row:
            continue

        chunk_id, doc_id, chunk_idx, content, chunk_type, section_title = row

        # ========== 修复开始 ==========
        # 获取前 prev_count 个 chunk
        prev_contents = []
        for i in range(1, prev_count + 1):
            cursor = db.execute("""
                SELECT content FROM chunks
                WHERE doc_id = ? AND chunk_index = ?
            """, (doc_id, chunk_idx - i))
            prev_row = cursor.fetchone()
            if prev_row:
                prev_contents.insert(0, prev_row[0])  # 插入前面保持顺序
        prev_content = "\n\n".join(prev_contents) if prev_contents else None

        # 获取后 next_count 个 chunk
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
        # ========== 修复结束 ==========

        expanded_chunks.append(ChunkInfo(
            id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_idx,
            content=content,
            score=result.score,
            source=result.source,
            chunk_type=chunk_type,
            section_title=section_title,
            prev_content=prev_content,
            next_content=next_content,
        ))

    return expanded_chunks
```

---

### Issue 2: `rag.py` 每次启动删除数据库

**位置**: `nanobot/agent/tools/rag.py` lines 76-77

**问题描述**:
```python
if db_path.exists():
    db_path.unlink()  # 每次都删除！
```

每次调用工具都会重新索引所有文档，用户体验很差。

**修复方案**:

```python
def _ensure_initialized(self) -> None:
    """Initialize the document store if not already initialized."""
    if self._doc_store is not None:
        return

    self._rag_dir = self.workspace / "rag"
    self._rag_dir.mkdir(parents=True, exist_ok=True)
    self._docs_dir = self.workspace / "docs"
    self._docs_dir.mkdir(parents=True, exist_ok=True)

    db_path = self._rag_dir / "docs.db"

    # ========== 修复开始 ==========
    # 不要删除数据库！直接用现有数据库
    # ========== 修复结束 ==========

    embedding_provider = SentenceTransformerEmbeddingProvider(self.embedding_model)
    self._doc_store = DocumentStore(db_path, embedding_provider, self.rag_config)
```

---

## 🟡 P1 中等问题

### Issue 3: `chunk_with_section_awareness` overlap 逻辑错误

**位置**: `nanobot/rag/parser.py` lines 418-427

**问题描述**:
```python
for i in range(1, len(chunks)):
    prev_chunk = chunks[i - 1]
    curr_chunk = chunks[i]
    # Add overlap from previous chunk to current chunk
    overlap_text = prev_chunk.content[-overlap_size:]
    if overlap_text:
        # Prepend overlap, but adjust start_pos accordingly
        curr_chunk.content = overlap_text + curr_chunk.content  # 重复内容会被索引！
        curr_chunk.start_pos -= len(overlap_text)
```

这会导致内容重复，而且 overlap 应该在切分**过程中**处理，不是切分后拼接。

**修复方案**:

```python
@staticmethod
def chunk_with_section_awareness(
    text: str,
    min_chunk_size: int = 500,
    max_chunk_size: int = 800,
    overlap_ratio: float = 0.12,
) -> list[SemanticChunk]:
    """
    Phase 2b main method: Chunk with section awareness.
    """
    # First detect section headings
    headings = DocumentParser.detect_section_headings(text)

    # If no headings detected, fall back to paragraph/sentence chunking
    if not headings:
        return DocumentParser.chunk_by_paragraph_sentence(
            text, min_chunk_size, max_chunk_size, overlap_ratio
        )

    # With headings: Create sections based on headings
    overlap_size = int(max_chunk_size * overlap_ratio)
    chunks: list[SemanticChunk] = []

    # Create sections between headings
    sections: list[tuple[str, int, int, Optional[str], ChunkType]] = []

    # First section (before first heading)
    if headings[0][1] > 0:
        sections.append((text[0:headings[0][1]], 0, headings[0][1], None, "other"))

    # Sections from headings
    for i, (title, start, end, chunk_type) in enumerate(headings):
        section_start = start
        if i < len(headings) - 1:
            section_end = headings[i + 1][1]
        else:
            section_end = len(text)
        section_content = text[section_start:section_end]
        sections.append((section_content, section_start, section_end, title, chunk_type))

    # Process each section
    for section_content, section_start, section_end, section_title, chunk_type in sections:
        if not section_content.strip():
            continue

        # If section is small enough, keep as single chunk
        if len(section_content) <= max_chunk_size:
            chunks.append(SemanticChunk(
                content=section_content,
                chunk_type=chunk_type,
                section_title=section_title,
                start_pos=section_start,
                end_pos=section_end
            ))
        else:
            # Section is too big - split internally using paragraph/sentence
            # chunk_by_paragraph_sentence 已经处理了 overlap，不需要再手动添加
            sub_chunks = DocumentParser.chunk_by_paragraph_sentence(
                section_content, min_chunk_size, max_chunk_size, overlap_ratio
            )
            # Adjust positions and add section metadata
            for sub_chunk in sub_chunks:
                sub_chunk.start_pos += section_start
                sub_chunk.end_pos += section_start
                sub_chunk.chunk_type = chunk_type
                sub_chunk.section_title = section_title
                chunks.append(sub_chunk)

    # ========== 修复开始 ==========
    # 删除下面这段错误的 overlap 逻辑！
    # ========== 修复结束 ==========

    return chunks
```

---

### Issue 4: SECTION_PATTERNS 硬编码章节编号

**位置**: `nanobot/rag/parser.py` lines 29-69

**问题描述**:
```python
(r"^\s*1\.\s+(Introduction)\s*$", "introduction"),  # 硬编码 1.
(r"^\s*2\.\s+(Related\s+Work)\s*$", "introduction"),  # 硬编码 2.
```

如果论文章节编号不从 1 开始，或者结构不同，就匹配不到。

**修复方案**:

```python
# Section heading patterns for Phase 2b - ordered by specificity (more specific first)
SECTION_PATTERNS = [
    # ========== 修复开始 ==========
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
    # ========== 修复结束 ==========

    # English section headings without numbers
    (r"^\s*(Abstract)\s*$", "abstract"),
    (r"^\s*(Introduction)\s*$", "introduction"),
    (r"^\s*(Related\s+Work)\s*$", "introduction"),
    (r"^\s*(Method|Methods|Methodology)\s*$", "method"),
    (r"^\s*(Experiment|Experiments|Experimental\s+Setup)\s*$", "experiment"),
    (r"^\s*(Result|Results)\s*$", "result"),
    (r"^\s*(Conclusion|Conclusions)\s*$", "conclusion"),
    (r"^\s*(References|Bibliography)\s*$", "other"),
    # Chinese section headings without numbers
    (r"^\s*(摘要)\s*$", "abstract"),
    (r"^\s*(引言)\s*$", "introduction"),
    (r"^\s*(相关工作|背景)\s*$", "introduction"),
    (r"^\s*(方法)\s*$", "method"),
    (r"^\s*(实验)\s*$", "experiment"),
    (r"^\s*(结果)\s*$", "result"),
    (r"^\s*(结论)\s*$", "conclusion"),
    (r"^\s*(参考文献)\s*$", "other"),
    # Markdown headings
    (r"^#\s+(.*)$", "other"),
    (r"^##\s+(.*)$", "other"),
    (r"^###\s+(.*)$", "other"),
]
```

同时需要更新 `detect_section_headings()` 方法来提取正确的标题：

```python
@staticmethod
def detect_section_headings(text: str) -> list[tuple[str, int, int, ChunkType]]:
    """
    Phase 2b: Detect obvious section headings.
    """
    headings = []
    lines = text.splitlines()
    current_pos = 0

    for line in lines:
        line_start = current_pos
        line_end = current_pos + len(line)

        for pattern, chunk_type in DocumentParser.SECTION_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # ========== 修复开始 ==========
                # 如果有捕获组，用最后一个非数字捕获组作为标题
                if match.groups():
                    # 找到最后一个不是纯数字的捕获组
                    title = None
                    for group in match.groups():
                        if group and not re.match(r"^\d+(?:\.\d+)*$", group.strip()):
                            title = group.strip()
                            break
                    if not title:
                        title = line.strip()
                else:
                    title = line.strip()
                # ========== 修复结束 ==========

                headings.append((title, line_start, line_end, chunk_type))
                break

        current_pos = line_end + 1

    return headings
```

---

### Issue 5: `rerank_and_dedup` 只 rerank 原始顺序的前 20，不是得分最高的

**位置**: `nanobot/rag/rerank.py` line 238

**问题描述**:
```python
rerank_candidates = candidate_texts[:self.rerank_top_k]  # 原始顺序，不是得分顺序
```

**修复方案**:

需要配合 `store.py` 中的调用一起修改。

首先，在 `store.py` 中调用 rerank 之前先按 score 排序：

```python
# 在 _apply_rerank 方法中，在调用 rerank_and_dedup 之前
# 先按原始分数排序，取 top-20
if len(candidates) > self.config.rerank_top_k:
    # 假设 candidates 有分数信息，需要调整数据结构
    # 或者在传入之前先排序
    pass
```

更简单的方案：在 `rerank_service.py` 中，不做截断，由调用者传入正确的数量。

---

## 🟢 P2 轻微问题

### Issue 6: `_init_schema` 中 ALTER TABLE 没必要

**位置**: `nanobot/rag/store.py` lines 164-175

**问题描述**:
既然是新项目，直接在 `CREATE TABLE` 里包含所有列就行，不需要先建表再 ALTER TABLE。

**修复方案**:

```python
def _init_schema(self) -> None:
    """Initialize the database schema."""
    db = self._get_db()

    # ========== 修复开始 ==========
    # 直接创建包含所有列的表
    db.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER,
            mtime REAL NOT NULL,
            stored_at REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            doc_type TEXT,
            abstract TEXT
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_pos INTEGER,
            end_pos INTEGER,
            chunk_type TEXT,
            section_title TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(doc_id, chunk_index)
        )
    """)

    # 删除 ALTER TABLE 相关代码！
    # ========== 修复结束 ==========

    # sqlite-vec virtual table for embeddings (only if vector enabled)
    if self._vector_enabled:
        # ... 保持不变 ...

    # FTS5 virtual table for full-text search
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content=chunks,
            content_rowid=id,
            tokenize='porter unicode61'
        )
    """)

    # ... 其余保持不变 ...
```

---

## 📋 功能补充

### Issue 7: 文档级优先策略没有提取"摘要 + 核心章节"

**问题描述**:
原始设计要求"从 Top3 文档中提取摘要 + 核心章节的完整内容"，但当前实现只是计算了文档平均得分。

**修复方案**:

在 `store.py` 中添加辅助方法：

```python
def _get_document_core_content(self, doc_id: int) -> str:
    """
    获取文档的核心内容：摘要 + 核心章节（method/experiment/result）
    """
    db = self._get_db()

    # 获取文档摘要
    cursor = db.execute("SELECT abstract FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    abstract = row[0] if row and row[0] else ""

    # 获取核心章节
    cursor = db.execute("""
        SELECT section_title, content
        FROM chunks
        WHERE doc_id = ?
          AND chunk_type IN ('method', 'experiment', 'result', 'conclusion')
        ORDER BY chunk_index
    """, (doc_id,))

    core_parts = []
    if abstract:
        core_parts.append(f"## Abstract\n{abstract}")

    for section_title, content in cursor:
        if section_title:
            core_parts.append(f"## {section_title}\n{content}")
        else:
            core_parts.append(content)

    return "\n\n".join(core_parts)
```

然后在 `_merge_context_and_document_results` 中使用：

```python
# ... 在处理 doc_info_cache 之后 ...
for doc, score in top_docs:
    core_content = self._get_document_core_content(doc.id)
    if core_content:
        # 添加一个特殊的 document-level 结果
        # ...
```

---

## 📝 修复优先级

| 优先级 | Issue | 预计修复时间 |
|-------|-------|-------------|
| 🔴 P0 | Issue 2: 每次启动删数据库 | 5 分钟 |
| 🔴 P0 | Issue 1: 上下文扩展只取单 chunk | 30 分钟 |
| 🟡 P1 | Issue 3: overlap 逻辑错误 | 20 分钟 |
| 🟡 P1 | Issue 4: SECTION_PATTERNS 硬编码 | 30 分钟 |
| 🟡 P1 | Issue 5: rerank 只 rerank 原始顺序 | 15 分钟 |
| 🟢 P2 | Issue 6: ALTER TABLE 没必要 | 10 分钟 |
| 🟢 P2 | Issue 7: 文档级内容提取 | 45 分钟 |

**总计**: ~3 小时