# RAG 优化计划：解决片段零碎问题

## 概述

当前 RAG 模块存在问题：文档切分太零碎，召回的片段缺少上下文，噪声较多。按科研逻辑重构切分和检索策略。

## 核心设计

### 1. 文本切块重构（核心）

**采用分阶段妥协方案，做到 2b：**

- **Phase 2a**: 按句子/段落边界切分，优先在句子边界切分（而不是硬截断），相邻块保留 10-15% overlap
- **Phase 2b**: 检测明显的章节标题（Abstract, 1. Introduction, References 等），如果检测到优先在章节边界切分

**块大小**: 500-800 字，相邻 chunk 之间保留 10-15% overlap

**元数据**: chunk_type, section_title, chunk_order（即 chunk_index）直接合并到 chunks 表（不单独建 chunk_meta 表，但逻辑上保留 chunk_meta 的功能

### 2. 检索策略优化（已澄清：串行三步流程）

**核心逻辑：先精准召回核心片段 → 再拓展完整上下文 → 最后锚定高相关文档

---

#### 第一步：核心 chunk 精准召回（过滤低相关噪声，只留核心）

- 接收用户的科研查询（如 "YOLOv9 实验参数优化"）

- **并行执行两类检索**：
  - **SQLite 全文检索（FTS5）：匹配查询中的关键词/专业术语，仅保留 BM25 得分≥0.6 的 chunk（过滤低相关碎片）
  - **SQLite 向量检索（Bi-Encoder）：匹配查询与文本的语义关联，仅保留余弦相似度≥0.75 的 chunk（进一步剔除噪声）

- **融合结果**：去重后，选取得分最高的 **Top5 核心 chunk（这 5 个 chunk 是与查询最相关的基础片段）

---

#### 第二步：上下文拓展召回（把单一片段补成完整逻辑）

- 基于 Top5 核心 chunk，关联 chunks 表（用 chunk_index 作为 chunk_order），提取每个核心 chunk 的关键元信息：所属文档 ID（doc_id）、该 chunk 在文档中的顺序（chunk_order = chunk_index）

- 以 doc_id 为依据，召回同一文档中"核心 chunk 的前 1 个 chunk + 核心 chunk + 后 1 个 chunk（比如核心 chunk 是文档的第 5 块，就召回 4、5、6 块）

- 将这 3 个连续的 chunk 合并成一段完整文本，形成具备上下文逻辑的片段（比如"实验参数→操作步骤→实验结果"的完整内容），解决单一片段零碎的问题

---

#### 第三步：文档级优先策略（锚定高相关完整文档，提升结果价值）

- 针对第二步拓展出的所有 chunk，按 doc_id 分组，计算每个文档内所有 chunk 的平均得分（比如某文档有 10 个 chunk，取这 10 个 chunk 的全文/向量得分平均值）

- 按平均得分排序，选取 **Top3 平均得分最高的高相关文档**（比如 YOLOv9 核心论文、实验笔记、参数优化文档）

- 从这 3 个高相关文档中，直接提取"文档摘要 + 核心章节（方法/实验/结论）的完整内容（而非零散 chunk），替代原本的单一片段

---

#### 最终输出：合并"上下文拓展的片段"和"文档级核心内容"，去重后剔除残留的低相关噪声，输出完整、无零碎干扰的检索结果

---

### 3. 噪声过滤

- **基础过滤**: 保留符合得分阈值的 chunk（BM25≥0.6、向量相似度≥0.75）- *注：这些是初始值，未来考虑相对阈值*
- **进阶过滤**: Cross-Encoder Reranker，二次打分，模型用 `cross-encoder/ms-marco-MiniLM-L-6-v2（考虑 MacBook Pro M4 24GB 性能），保留≥0.8 的内容
- **语义去重**: 剔除 chunk 间相似度≥0.9 的重复内容

### 4. 元数据管理

- **chunks 表扩展**：直接添加 chunk_type, section_title 字段（chunk_order 用 chunk_index）
- **documents 表扩展**：存储 title, doc_type, abstract
- **索引监控**：在 stats 中增加 avg_chunk_size, last_scan_at 等

---

## 关于 FTS5 tokenizer 'porter unicode61'

**问题**: 这不是中英文都适合吗？

**回答**: `porter` 是英文词干提取器，对中文没有帮助；但 `unicode61` 支持 Unicode 分词。

**结论**: 保持 `tokenize='porter unicode61' 是合理的：
- 对英文：porter 词干提取 + unicode61
- 对中文：unicode61 可以按字符/词切分（取决于内容），虽然没有专门的中文分词，但够用了
- 如果未来要优化中文，再考虑 `tokenize='unicode61'` 或添加中文分词器

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      RetrieveTool                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │  search()                                      │   │
│  │  1. 核心 chunk 精准召回 (Top5, 双阈值)│   │
│  │  2. 上下文拓展召回 (前1+核心+后1)         │   │
│  │  3. 文档级优先 (Top3 文档)             │   │
│  │  4. Cross-Encoder Rerank (Top20)            │   │
│  │  5. 语义去重                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       DocumentStore (SQLite)                  │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  - documents (title, abstract, doc_type)       │   │
│  │  - chunks (chunk_type, section_title, chunk_index) │   │
│  │  - chunk_embeddings                        │   │
│  │  - chunks_fts (BM25, porter unicode61)      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       DocumentParser (Phase 2b)              │   │
│  │  - chunk_by_paragraph_sentence()          │   │
│  │  - detect_section_headings() (明显标题)       │   │
│  │  - merge_small_chunks() / overlap          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       Reranker (新增)                        │   │
│  │  - cross_encoder_rank(ms-marco-MiniLM-L-6-v2)  │   │
│  │  - semantic_dedup()                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 数据库 Schema（chunk_meta 功能合并到 chunks）

```sql
-- 文档表（扩展）
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER,
    mtime REAL NOT NULL,
    stored_at REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- 新增字段
    title TEXT,           -- 文档标题
    doc_type TEXT,        -- "paper", "lab_note", "concept", "other"
    abstract TEXT         -- 摘要
);

-- Chunk 表（已合并 chunk_meta 功能，用 chunk_index 作为 chunk_order
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,   -- 同时用作 chunk_order
    content TEXT NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    -- 新增字段（从 chunk_meta 合并过来
    chunk_type TEXT,           -- "abstract", "introduction", "method", "experiment",
                               -- "result", "conclusion", "lab_record", "concept_def", "other"
    section_title TEXT,        -- 章节标题（如果有）
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_order ON chunks(doc_id, chunk_index);

-- sqlite-vec 向量表
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT32[{{dimensions}}]
);

-- FTS5 全文搜索表（保持 porter unicode61）
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- FTS 触发器
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
```

## 核心数据结构

```python
from dataclasses import dataclass
from typing import Literal

ChunkType = Literal[
    "abstract", "introduction", "method", "experiment",
    "result", "conclusion", "lab_record", "concept_def", "other"
]

DocType = Literal["paper", "lab_note", "concept", "other"

@dataclass
class DocumentInfo:
    id: int
    path: str
    filename: str
    file_type: str
    title: str | None = None
    doc_type: DocType = "other"
    abstract: str | None = None

@dataclass
class ChunkInfo:
    id: int
    doc_id: int
    chunk_index: int  # 同时用作 chunk_order
    content: str
    chunk_type: ChunkType | None = None
    section_title: str | None = None

@dataclass
class SearchResult:
    doc: DocumentInfo
    chunk: ChunkInfo
    bm25_score: float | None = None
    vector_score: float | None = None
    combined_score: float = 0.0
    source: str = "hybrid"

@dataclass
class SearchResultWithContext:
    doc: DocumentInfo
    core_chunk: ChunkInfo
    prev_chunks: list[ChunkInfo]
    next_chunks: list[ChunkInfo]
    combined_content: str
    core_score: float
    final_score: float  # after rerank
```

## 配置

```python
class RAGConfig(Base):
    """RAG configuration."""

    enabled: bool = True

    # Chunking strategy (Phase 2b)
    chunk_strategy: str = "phase2b"  # "fixed" | "paragraph" | "phase2b"
    min_chunk_size: int = 500
    max_chunk_size: int = 800
    chunk_overlap_ratio: float = 0.12  # 12% overlap between chunks

    # Context expansion
    enable_context_expansion: bool = True
    context_prev_chunks: int = 1
    context_next_chunks: int = 1

    # Document-level search
    enable_document_level: bool = True
    top_documents: int = 3

    # Thresholds (初始值，未来考虑相对阈值)
    bm25_threshold: float = 0.6
    vector_threshold: float = 0.75
    rerank_threshold: float = 0.8
    dedup_threshold: float = 0.9

    # Reranker (MacBook Pro M4 24GB 优化)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_rerank: bool = True
    rerank_top_k: int = 20  # 只对 top-20 rerank

    # Legacy / fallback
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    auto_scan_on_startup: bool = True
```

## 模块设计

### 1. DocumentParser - Phase 2b 实现

位置：`nanobot/rag/parser.py`

```python
class SemanticChunk:
    content: str
    chunk_type: ChunkType | None
    section_title: str | None
    start_pos: int
    end_pos: int

class DocumentParser:
    """智能文档解析器（Phase 2b）"""

    @staticmethod
    def parse(path: Path) -> tuple[str, str]:
        """解析文档"""

    @staticmethod
    def extract_metadata(path: Path, content: str) -> dict:
        """提取文档元数据: {title, doc_type, abstract}"""

    @staticmethod
    def chunk_by_paragraph_sentence(
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 800,
        overlap_ratio: float = 0.12,
    ) -> list[SemanticChunk]:
        """
        Phase 2a: 按段落/句子边界切分
        - 优先在段落边界切分
        - 段落太长时在句子边界切分
        - 相邻块保留 overlap_ratio 的重叠
        """

    @staticmethod
    def detect_section_headings(text: str) -> list[tuple[str, int, int]]:
        """
        Phase 2b: 检测明显的章节标题
        支持格式:
        - "Abstract", "摘要"
        - "1. Introduction", "1.1 相关工作"
        - "References", "参考文献"
        - "Method", "Methods", "方法"
        - "Experiment", "Experiments", "实验"
        - "Results", "结果"
        - "Conclusion", "Conclusions", "结论"
        """

    @staticmethod
    def chunk_with_section_awareness(
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 800,
        overlap_ratio: float = 0.12,
    ) -> list[SemanticChunk]:
        """
        Phase 2b 主方法:
        1. 先检测章节标题
        2. 如果检测到，优先在章节边界切分
        3. 如果没检测到，回退到 paragraph/sentence 切分
        4. 所有策略都保留 overlap
        """
```

### 2. Reranker - MacBook Pro M4 24GB 优化

位置：`nanobot/rag/rerank.py`（新增）

```python
from abc import ABC, abstractmethod

class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[int, float]]:
        """返回 (index, score) 按得分降序"""

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        MacBook Pro M4 24GB 优化:
        - 模型: cross-encoder/ms-marco-MiniLM-L-6-v2 (轻量，速度快)
        - 只对 top-20 进行 rerank
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    async def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[int, float]]:
        """使用 Cross-Encoder 重排序"""
        # 返回 [(index, score), ...] 按 score 降序

class SemanticDeduplicator:
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold

    async def deduplicate(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> list[int]:
        """返回保留的 chunk 索引列表，剔除相似度≥0.9 的重复内容"""
```

### 3. DocumentStore - 文档存储与检索（串行三步流程）

位置：`nanobot/rag/store.py`

```python
class DocumentStore:
    """文档存储：核心召回 → 上下文拓展 → 文档级优先 + Rerank"""

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResultWithContext]:
        """
        主搜索入口，串行三步流程：

        第一步：核心 chunk 精准召回
        第二步：上下文拓展召回
        第三步：文档级优先策略
        第四步：Cross-Encoder Rerank（≥0.8，只对 top-20）
        第五步：语义去重（≥0.9）

        最后合并第二步和第三步结果
        """

    async def _step1_core_chunk_recall(
        self,
        query: str,
        top_k: int = 5,
        bm25_threshold: float = 0.6,
        vector_threshold: float = 0.75,
    ) -> list[SearchResult]:
        """
        第一步：核心 chunk 精准召回
        - 并行全文+向量检索
        - 双阈值过滤
        - 返回 Top5 核心 chunk
        """

    async def _step2_context_expansion(
        self,
        core_results: list[SearchResult],
        prev_count: int = 1,
        next_count: int = 1,
    ) -> list[SearchResultWithContext]:
        """
        第二步：上下文拓展召回
        - 对每个核心 chunk，召回前1+核心+后1
        - 合并成完整上下文
        - 返回带上下文的结果
        """

    async def _step3_document_level(
        self,
        expanded_results: list[SearchResultWithContext],
        top_docs: int = 3,
    ) -> list[SearchResultWithContext]:
        """
        第三步：文档级优先策略
        - 按 doc_id 分组，计算平均得分
        - 选取 Top3 高相关文档
        - 从 Top3 文档中提取摘要+核心章节
        - 返回文档级结果
        """

    async def _apply_rerank(
        self,
        query: str,
        results: list[SearchResultWithContext],
        rerank_threshold: float = 0.8,
        rerank_top_k: int = 20,
    ) -> list[SearchResultWithContext]:
        """
        Cross-Encoder 二次打分，保留≥0.8 的内容
        只对 top-20 进行 rerank（M4 优化）
        """

    async def _apply_dedup(
        self,
        results: list[SearchResultWithContext],
        dedup_threshold: float = 0.9,
    ) -> list[SearchResultWithContext]:
        """语义去重，剔除相似度≥0.9 的重复内容"""

    async def _merge_context_and_document_results(
        self,
        context_results: list[SearchResultWithContext],
        document_results: list[SearchResultWithContext],
    ) -> list[SearchResultWithContext]:
        """合并第二步和第三步的结果，去重"""

    def get_stats(self) -> dict:
        """
        索引监控新增字段：
        - avg_chunk_size: 平均 chunk 大小
        - last_scan_at: 上次扫描时间
        """
```

## 实施阶段（调整后）

| 阶段 | 内容 | 工作量 |
|-----|------|--------|
| **Phase 1** | 数据层重构 + 上下文拓展 | 2-3 天 |
| **Phase 2** | 语义切分（Phase 2b：明显章节检测） | 3-5 天 |
| **Phase 3** | 检索优化（串行三步流程） | 3-5 天 |
| **Phase 4** | Rerank & 去重（M4 优化） | 2-3 天 |
| **Phase 5** | 工具优化 + 集成测试 | 2-3 天 |

### Phase 1: 数据层重构 + 上下文拓展（高性价比，先做）
- [ ] 扩展 documents 表（title, abstract, doc_type）
- [ ] 扩展 chunks 表（chunk_type, section_title）- chunk_index 用作 chunk_order
- [ ] 更新 `_init_schema()`
- [ ] 实现 `_step1_core_chunk_recall()` - 核心 chunk 精准召回
- [ ] 实现 `_step2_context_expansion()` - 上下文拓展
- [ ] 索引监控：在 get_stats() 增加 avg_chunk_size, last_scan_at
- [ ] **效果**: 解决 60% 的上下文问题

### Phase 2: 语义切分（Phase 2b）
- [ ] `DocumentParser.chunk_by_paragraph_sentence()` - 句子/段落边界切分 + overlap
- [ ] `DocumentParser.detect_section_headings()` - 明显章节标题检测
- [ ] `DocumentParser.chunk_with_section_awareness()` - 主切分逻辑
- [ ] 块大小控制在 500-800 字，10-15% overlap
- [ ] 单元测试：`tests/test_rag_parser.py`

### Phase 3: 检索优化（串行三步流程）
- [ ] `_step1_core_chunk_recall()` - 并行全文+向量，双阈值，Top5
- [ ] `_step2_context_expansion()` - 前1+核心+后1
- [ ] `_step3_document_level()` - 文档级优先策略
- [ ] `_merge_context_and_document_results()` - 合并第二步和第三步结果
- [ ] 单元测试：`tests/test_rag_search.py`

### Phase 4: Rerank & 去重（M4 优化）
- [ ] `CrossEncoderReranker.rerank()` - Cross-Encoder 集成（`cross-encoder/ms-marco-MiniLM-L-6-v2"）
- [ ] 只对 top-20 进行 rerank（M4 24GB 优化）
- [ ] `SemanticDeduplicator.deduplicate()` - 语义去重（≥0.9）
- [ ] 单元测试：`tests/test_rag_rerank.py`

### Phase 5: 工具优化
- [ ] 返回格式优化，展示完整上下文
- [ ] 集成测试：`tests/test_rag_integration.py`

## 文件清单

| 文件 | 操作 |
|------|------|
| `nanobot/rag/parser.py` | 重写（Phase 2b） |
| `nanobot/rag/store.py` | 重写（串行三步流程） |
| `nanobot/rag/rerank.py` | 新增（M4 优化） |
| `nanobot/rag/__init__.py` | 修改 |
| `nanobot/config/schema.py` | 修改（overlap, 监控字段） |
| `nanobot/agent/tools/rag.py` | 修改 |
| `tests/test_rag_parser.py` | 新增 |
| `tests/test_rag_search.py` | 新增 |
| `tests/test_rag_rerank.py` | 新增 |
| `tests/test_rag_integration.py` | 新增 |

## 搜索结果返回格式示例

```
Results for "YOLOv9 实验参数优化":

[1] YOLOv9: Learning What You Want to Learn.pdf (score: 0.92, context-expanded)
Type: paper
Section: 3.2 实验参数
─────────────────────────────────────────────────────────────
[Prev] ...本节介绍实验设置和参数选择...
[Core] 我们使用的主要参数：
- 学习率: 0.01
- Batch size: 64
- Epochs: 300
- Optimizer: SGD with momentum...
[Next] ...实验结果表明这些参数的有效性...
─────────────────────────────────────────────────────────────

[2] YOLOv9 实验笔记.md (score: 0.88, document-level)
Type: lab_note
From: 文档级检索
...
```

## 硬件配置说明

**目标设备**: MacBook Pro M4 24GB

**Reranker 模型选择**: `cross-encoder/ms-marco-MiniLM-L-6-v2"`
- 模型大小: ~80MB
- 推理速度: 在 M4 上可以接受
- 只对 top-20 进行 rerank，避免性能问题

**Embedding 模型**: `all-MiniLM-L6-v2"`（保持不变）
- 模型大小: ~80MB
- 推理速度: 快

## 关于阈值的说明

当前方案保留了 BM25≥0.6、向量相似度≥0.75、Rerank≥0.8 作为**初始值**。

**未来优化方向**（当前不做）：
- 改用相对阈值（百分位或相对于最高分）
- 让这些阈值可以动态调整

**原因**: 绝对阈值在不同文档库和查询下表现差异很大，但作为起点可以接受。

## 风险和注意事项

1. **向后兼容性**: 新项目直接推倒重建，不需要迁移
2. **性能考虑**: Cross-Encoder Rerank 只对 top-20，在 M4 24GB 上没问题
3. **FTS5 tokenizer**: 保持 `porter unicode61"`，中英文都够用
4. **chunk overlap**: 10-15% 的 overlap 能有效防止信息丢失，同时不会造成太多冗余
5. **chunk_meta 功能**: 逻辑上保留，但字段合并到 chunks 表，用 chunk_index 作为 chunk_order
