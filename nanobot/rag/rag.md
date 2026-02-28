# RAG 系统说明文档

**Date:** 2026-02-28

---

## 一、概述

RAG (Retrieval-Augmented Generation) 模块是 nanobot 的核心知识检索组件，为 AI 助手提供本地文档知识库搜索能力。

**定位**：本地个人科研助手
**特点**：轻量级、无需外部 API、纯本地运行

---

## 二、架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG 检索流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Query ──▶ Query Expansion ──▶ Core Recall ──▶ Context Expansion        │
│                (缩写展开)          (Top5)            (±1 chunk)            │
│                                                                    │       │
│                                                                    ▼       │
│                                              Document-level          │       │
│                                              Prioritization          │       │
│                                              (Top3 docs)            │       │
│                                                                    │       │
│                                                                    ▼       │
│                                              Merge + Rerank ──▶ Final Results
│                                              (Top8)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心组件

### 3.1 模块结构

```
nanobot/rag/
├── __init__.py       # 模块导出
├── store.py          # 搜索引擎（核心）
├── parser.py         # 文档解析
├── embeddings.py     # 向量嵌入
├── rerank.py         # Cross-Encoder 重排
└── query.py          # Query 改写
```

### 3.2 数据流

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   文档文件    │───▶│   解析器      │───▶│   分块器     │
│ (PDF/MD/DOCX)│    │  DocumentParser │  │ chunk_text() │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                                ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  最终结果     │◀───│   重排器      │◀───│   搜索引擎    │
│ search_advanced()│    │  RerankService │   │ DocumentStore │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                    ┌──────────────┐            │
                    │  向量嵌入     │◀───────────┘
                    │ embeddings.py│
                    └──────────────┘
```

---

## 四、检索流程详解

### 4.1 入口：`search_advanced()`

```python
async def search_advanced(query: str) -> list[SearchResultWithContext]:
```

**完整流程**（4 个阶段）：

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: Query 改写 (Query Expansion)                           │
│   - 展开缩写：llm → large language model                      │
│   - 术语映射：大模型 → large language model, llm              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Core Recall (核心召回)                                 │
│   - FTS5 全文检索 → Top10                                     │
│   - 向量检索 → Top10                                           │
│   - RRF 融合 (k=60) → Top5                                     │
│   - 阈值过滤：BM25 ≥ 0.05, Vector ≥ 0.3                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Context Expansion (上下文扩展)                         │
│   - 扩展范围：prev chunk + core + next chunk                  │
│   - 配置：context_prev_chunks=1, context_next_chunks=1        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Document-level Prioritization (文档级优先)             │
│   - 计算每篇文档的平均得分                                     │
│   - 取 Top3 文档，给予 10% 加权 bonus                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Merge + Rerank (融合 + 重排)                          │
│   - 合并 core results + doc-level bonus                        │
│   - Cross-Encoder 重排 (Top20 候选)                            │
│   - 语义去重 (相似度 ≥ 0.9)                                    │
│   - 输出 Top8                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、各组件详解

### 5.1 文档解析 (`parser.py`)

**支持格式**：
- PDF (.pdf) - 使用 PyMuPDF (首选) 或 pypdf (备用)
- Markdown (.md, .markdown)
- Word (.docx, .doc)
- 纯文本 (.txt)

**分块策略**（Phase 2b - Section-aware）：

```python
chunk_with_section_aware()  # 默认
├── 1. 检测章节标题 (detect_section_headings)
│   ├── 英文：Introduction, Method, Experiment, Result, Conclusion
│   ├── 中文：引言, 方法, 实验, 结果, 结论
│   └── Markdown: ##, ###
├── 2. 按章节分块
│   └── 小章节直接成块
├── 3. 大章节内部再分
│   └── 按段落/句子分块，保留 12% overlap
```

**元数据提取**：
- `title`: 文档标题（文件名）
- `doc_type`: paper / lab_note / concept / other
- `abstract`: 摘要（从 Abstract 章节提取）

### 5.2 向量嵌入 (`embeddings.py`)

**默认模型**：`all-MiniLM-L6-v2` (384 维)

```python
# 可选模型
- all-MiniLM-L6-v2      # 默认，快速，384 维
- all-mpnet-base-v2     # 更高质量，768 维
- paraphrase-multilingual-MiniLM-L12-v2  # 多语言
```

**批量嵌入**：支持批量处理优化

### 5.3 搜索引擎 (`store.py`)

**核心类**：`DocumentStore`

**数据库表**：
```sql
documents     -- 文档元信息
chunks        -- 文本块
chunk_embeddings  -- 向量 (sqlite-vec)
chunks_fts    -- FTS5 全文索引
```

**关键方法**：

| 方法 | 功能 |
|------|------|
| `scan_and_index()` | 扫描文档并索引 |
| `search_advanced()` | 高级检索（推荐） |
| `search()` | 基础检索 |
| `_vector_search()` | 向量检索 |
| `_fulltext_search()` | FTS5 全文检索 |
| `_step1_core_chunk_recall()` | 核心召回 |
| `_step2_context_expansion()` | 上下文扩展 |
| `_step3_document_level()` | 文档级优先 |
| `_apply_rerank()` | Cross-Encoder 重排 |

### 5.4 Query 改写 (`query.py`)

**功能**：展开缩写和术语

```python
# 示例
"transformer vs cnn"
→ "transformer vs cnn convolutional neural network transformer ..."

"大模型 微调"
→ "大模型 微调 fine-tuning fine tuning large language model llm"
```

**术语词典**：内置 30+ 常见 ML/AI 术语

### 5.5 重排器 (`rerank.py`)

**组件**：
- `CrossEncoderReranker`: Cross-Encoder 重排
- `SemanticDeduplicator`: 语义去重
- `RerankService`: 组合服务

**模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2` (轻量级)

**流程**：
```
1. 取 Top20 候选
2. Cross-Encoder 打分
3. 阈值过滤 (≥ 0.8)
4. 语义去重 (相似度 ≥ 0.9)
5. 排序输出
```

---

## 六、配置项

### 6.1 RAGConfig

```python
class RAGConfig:
    # === 基础配置 ===
    enabled: bool = True
    chunk_strategy: str = "phase2b"      # 分块策略

    # === 分块参数 ===
    min_chunk_size: int = 500
    max_chunk_size: int = 800
    chunk_overlap_ratio: float = 0.12   # 12% overlap

    # === 上下文扩展 ===
    enable_context_expansion: bool = True
    context_prev_chunks: int = 1
    context_next_chunks: int = 1

    # === 文档级优先 ===
    enable_document_level: bool = True
    top_documents: int = 3

    # === 阈值 ===
    bm25_threshold: float = 0.05
    vector_threshold: float = 0.3
    rerank_threshold: float = 0.5
    dedup_threshold: float = 0.7

    # === 重排器 ===
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    enable_rerank: bool = True
    rerank_top_k: int = 20

    # === Query 改写 ===
    enable_query_expand: bool = True

    # === PDF 解析 ===
    pdf_parser: str = "pymupdf"  # "pypdf" | "pymupdf"
```

---

## 七、工具接口

### 7.1 SearchKnowledgeTool

```python
class SearchKnowledgeTool(Tool):
    name = "search_knowledge"

    parameters = {
        "query": "搜索内容",
        "top_k": 5,  # 默认返回数
    }

    async def execute(query: str, top_k: int = 5) -> str
```

---

## 八、使用示例

### 8.1 代码调用

```python
from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider
from nanobot.config.schema import RAGConfig
from pathlib import Path

# 初始化
embedding_provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
config = RAGConfig()
store = DocumentStore(Path("~/nanobot/rag/docs.db"), embedding_provider, config)

# 索引文档
await store.scan_and_index(Path("~/nanobot/docs"))

# 搜索
results = await store.search_advanced("Transformer 的注意力机制")
for r in results:
    print(f"[{r.rank}] {r.document.title}: {r.combined_content[:200]}...")
```

### 8.2 CLI 调用

```bash
# 扫描索引文档
nanobot rag scan

# 搜索
nanobot rag search "Transformer attention mechanism"
```

---

## 九、关键算法

### 9.1 RRF (Reciprocal Rank Fusion)

```python
# 融合多个排序结果
k = 60  # 经验参数
score = sum(1 / (k + rank) for rank in ranks)
```

### 9.2 BM25 评分

- 使用 SQLite FTS5 内置的 `bm25()` 函数
- 分数越低越相关

### 9.3 向量相似度

```python
# 余弦相似度转分数 (sqlite-vec 返回 distance)
similarity = max(0, min(1, 1 - distance / 2))
```

### 9.4 语义去重

```python
# 余弦相似度 ≥ 阈值则去除
if cosine_similarity(emb1, emb2) >= 0.9:
    # 去除重复
```

---

## 十、数据库结构

```sql
-- 文档表
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    filename TEXT,
    file_type TEXT,
    file_size INTEGER,
    mtime REAL,
    stored_at REAL,
    title TEXT,
    doc_type TEXT,
    abstract TEXT
);

-- 文本块表
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER,
    chunk_index INTEGER,
    content TEXT,
    start_pos INTEGER,
    end_pos INTEGER,
    chunk_type TEXT,
    section_title TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(id)
);

-- 向量表 (sqlite-vec)
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT32[384]
);

-- FTS5 全文索引
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);
```

---

## 十一、性能

| 操作 | 数据量 | 预期延迟 |
|------|--------|----------|
| 文档解析 | 1 页 PDF | ~50ms |
| 向量嵌入 | 1 chunk | ~10ms |
| FTS5 搜索 | 万条 | ~20ms |
| 向量搜索 | 万条 | ~50ms |
| RRF 融合 | Top10 | ~5ms |
| Context Expansion | Top5 | ~10ms |
| Cross-Encoder | Top20 | ~200ms |
| **总计** | - | **< 400ms** |

---

## 十二、扩展点

1. **新增文档格式**：在 `parser.py` 添加解析方法
2. **新增嵌入模型**：在 `embeddings.py` 添加 provider
3. **新增重排器**：在 `rerank.py` 实现 `Reranker` 接口
4. **图搜索增强**：待实现（见 docs/plans/）

---

## 十三、依赖

```toml
rag = [
    "sqlite-vec>=0.1.0",
    "pymupdf>=1.24.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "sentence-transformers>=3.0.0",
]
```
