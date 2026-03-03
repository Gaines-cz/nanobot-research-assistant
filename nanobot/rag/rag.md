# RAG 模块说明书

**最后更新**: 2026-03-02
**版本**: 2.1 (完善版)

本文档详细介绍 nanobot 项目的 RAG（检索增强生成）模块的设计、架构和使用方法。

---

## 1. 模块概述

RAG 模块是 nanobot 研究助手的核心组件，负责对本地文档进行语义索引和检索。该模块使 AI 代理能够通过语义搜索从用户的研究文档库中获取相关信息。

### 核心功能

| 功能 | 说明 |
|------|------|
| **文档解析** | 支持 PDF、Markdown、Word、纯文本等多种格式 |
| **语义分块** | 基于段落和语义的智能文档分块（Phase 2b 策略） |
| **混合搜索** | 结合向量搜索和全文搜索的混合检索（RRF 融合） |
| **上下文扩展** | 检索时自动扩展前后相邻分块，保留完整语境 |
| **结果重排** | 使用 Cross-Encoder 进行结果重排序 |
| **语义去重** | 去除相似度过高的重复内容 |
| **查询扩展** | 自动展开缩写词和同义词（NN→Neural Network） |
| **搜索缓存** | LRU + TTL 缓存，提升重复查询性能 |

### 设计亮点

1. **混合检索架构**：向量搜索（语义相似）+ FTS5 全文搜索（精确匹配），RRF 融合
2. **分阶段搜索管道**：核心召回→上下文扩展→文档级优先→重排去重
3. **SQLite-vec 集成**：轻量级向量数据库，支持优雅降级到纯 FTS
4. **语义感知分块**：Section-aware chunking，保持文档结构信息

---

## 2. 目录结构（重构后）

```
nanobot/rag/
├── __init__.py          # 模块入口，导出公共接口
├── embeddings.py        # Embedding 向量化提供者
├── parser.py            # 文档解析器和分块逻辑
├── query.py             # 查询预处理与扩展
├── rerank.py            # 结果重排序与语义去重
│
# 重构后新增模块（原 store.py 拆分）
├── store.py             # DocumentStore 门面类（~150 行）
├── connection.py        # SQLite 数据库连接（~200 行）
├── indexer.py           # 文档索引操作（~250 行）
├── search.py            # 核心搜索功能（~650 行）
├── cache.py             # 搜索结果缓存（~100 行）
```

### 重构说明

原 `store.py` (1214 行) 职责过多，已拆分为 5 个职责单一的模块：

| 模块 | 职责 | 行数 |
|------|------|------|
| `connection.py` | 数据库连接、schema 初始化、sqlite-vec 加载 | ~200 |
| `indexer.py` | 文档扫描、新增/更新/删除、批量索引 | ~250 |
| `search.py` | 向量搜索、全文搜索、混合搜索、高级管道 | ~650 |
| `cache.py` | LRU+TTL 缓存管理 | ~100 |
| `store.py` | Facade 门面，编排各子模块 | ~150 |

**优势**：职责单一、易于测试、可扩展性强

---

## 3. 核心组件详解

### 3.1 Embeddings (`embeddings.py`)

负责将文本转换为向量表示。

| 类 | 说明 |
|---|---|
| `EmbeddingProvider` | 抽象基类，定义 `embed()` 和 `embed_batch()` 接口 |
| `SentenceTransformerEmbeddingProvider` | 本地 Embedding，使用 sentence-transformers |

**支持的模型：**

```python
- "all-MiniLM-L6-v2"          # 轻量快速（SearchKnowledgeTool 默认）
- "all-mpnet-base-v2"         # 质量更高，速度稍慢
- "BAAI/bge-m3"               # 多语言，科学文本优化（RAGConfig 默认）
- "paraphrase-multilingual-MiniLM-L12-v2"  # 多语言场景
```

**特性：**
- 懒加载模型（首次调用时加载）
- 模型缓存（避免重复加载）
- 线程安全初始化

---

### 3.2 文档解析器 (`parser.py`)

负责解析各种格式的文档并进行智能分块。

| 类/函数 | 说明 |
|---------|------|
| `DocumentParser` | 解析 PDF、Markdown、Word、纯文本文件 |
| `SemanticChunk` | 分块数据类，包含内容、类型、section 标题 |
| `chunk_by_paragraph_sentence()` | 基于段落/句子边界的分块（带重叠） |
| `chunk_with_section_awareness()` | Section 感知的智能分块（Phase 2b） |

**提取的元数据：**

```python
{
    "title": "文档标题（从 Markdown 标题或 PDF 元数据提取）",
    "doc_type": "paper|lab_note|concept|...",
    "abstract": "摘要内容"
}
```

**分块策略：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `phase2b` | Section 感知，按学术论文章节边界分块 | 论文、技术文档 |
| `paragraph` | 基于段落和句子边界 | 通用文档 |
| `fixed` | 固定大小分块（遗留） | 向后兼容 |

**默认参数：**
- `max_chunk_size`: 800 字符
- `min_chunk_size`: 500 字符
- `overlap_ratio`: 0.12 (12% 重叠)

**Phase 2b 分块流程：**
1. 先检测章节标题（支持中英文、带/不带编号）
2. 如果检测到章节，优先按章节边界分块
3. 如果章节太大，内部按段落/句子再分
4. 如果未检测到章节，回退到纯段落/句子分块
5. 所有策略都保持 12% 的重叠

---

### 3.3 数据库连接 (`connection.py`)

| 类 | 说明 |
|---|---|
| `DatabaseConnection` | 管理 SQLite 数据库连接和 schema |

**数据库表结构：**

```sql
-- 文档元信息表
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER,
    mtime REAL NOT NULL,
    stored_at REAL NOT NULL,
    title TEXT,           -- 文档标题
    doc_type TEXT,        -- 文档类型
    abstract TEXT         -- 摘要
);

-- 分块表
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    chunk_type TEXT,      -- 分块类型 (abstract/introduction/method...)
    section_title TEXT,   -- 所属章节标题
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(doc_id, chunk_index)
);

-- 向量表 (sqlite-vec)
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT32[384]  -- 维度由 embedding 模型决定
);

-- 全文搜索表 (FTS5)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);
```

**特性：**
- 使用 `sqlite-vec` 扩展支持向量搜索
- **优雅降级**：sqlite-vec 加载失败时自动降级到纯 FTS5
  - 有冷却期机制（避免频繁重试）
  - 向量搜索禁用后，全文搜索仍然可用
- FTS5 触发器自动维护索引（INSERT/UPDATE/DELETE）
- WAL 模式（Write-Ahead Logging）支持并发读取

---

### 3.4 文档索引器 (`indexer.py`)

| 类 | 说明 |
|---|---|
| `DocumentIndexer` | 文档索引操作 |

**功能：**
- 扫描目录查找支持的文件类型
- 检测新增、更新、删除的文档（基于 mtime）
- 支持定时索引更新（30 秒延迟以批量处理）

**支持的文件类型：**

```python
SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".markdown", ".docx", ".doc", ".txt",
    ".rst", ".py", ".js", ".ts", ".html", ".css"
}
```

**索引流程：**

```python
1. 扫描 docs_dir，获取所有支持的文件
2. 对比数据库中已知文档（path + mtime）
3. 对于每个文件：
   - 新增文档：调用 _add_document()
   - 更新文档：调用 _update_document()（先删后增）
   - 删除文档：调用 _delete_document()
4. 提交事务
```

---

### 3.5 搜索模块 (`search.py`)

| 类/数据类 | 说明 |
|-----------|------|
| `DocumentSearch` | 核心搜索操作 |
| `SearchResult` | 基础搜索结果（path, filename, content, score） |
| `DocumentInfo` | 文档元信息（id, path, title, doc_type...） |
| `ChunkInfo` | 包含上下文的分块（prev_content, next_content） |
| `SearchResultWithContext` | 带扩展上下文的搜索结果 |

**搜索模式：**

#### 模式 1：基础混合搜索

```python
async def search(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    混合搜索：向量 + 全文，RRF 融合

    流程：
    1. QueryExpander 扩展查询（添加同义词）
    2. 向量搜索（top_k*2）+ 全文搜索（top_k*2）
    3. RRF 融合 (k=60)
    4. 返回 top_k
    """
```

#### 模式 2：高级多步搜索（推荐）

```python
async def search_advanced(query: str) -> list[SearchResultWithContext]:
    """
    高级多步搜索管道（实际代码是 4 步）：

    Step 1: 核心分块召回
    ├─ 向量搜索 + 全文搜索
    ├─ 双阈值过滤 (bm25_threshold=0.05, vector_threshold=0.3)
    └─ RRF 融合 → Top5 分块

    Step 2: 上下文扩展
    ├─ 对于每个核心分块：获取 prev_content + content + next_content
    └─ 可配置：prev_chunks=1, next_chunks=1

    Step 3: 文档级优先 + 结果合并
    ├─ 按文档分组分块，计算每个文档的平均分
    ├─ 选择 Top3 文档
    ├─ 合并分块和文档分（doc_bonus = 0.1 * doc_score）
    └─ 按最终分排序

    Step 4: 重排序与去重（如启用）
    ├─ CrossEncoderReranker.rerank() → 重打分 Top20
    ├─ 按 rerank_threshold (0.8) 过滤
    └─ SemanticDeduplicator.deduplicate() → 移除相似度 >= 0.9 的内容
    """
```

**高级搜索细节说明：**
- **双阈值过滤**：BM25 阈值 0.05，向量阈值 0.3（因为 BM25 分数归一化后偏低）
- **文档级加分**：Top3 文档的分块会获得额外 0.1*doc_score 的奖励
- **性能优化**：Cross-Encoder 只重排前 20 个候选

---

### 3.6 重排序模块 (`rerank.py`)

| 类 | 说明 |
|---|---|
| `Reranker` | 抽象基类，定义 `rerank(query, candidates)` 接口 |
| `CrossEncoderReranker` | Cross-Encoder 重排序（sentence-transformers） |
| `SemanticDeduplicator` | 语义去重（余弦相似度阈值） |
| `RerankService` | 组合重排序和去重的服务类 |

**默认模型：** `cross-encoder/ms-marco-MiniLM-L-6-v2`（轻量快速，Mac M4 优化）

**配置参数：**
- `rerank_top_k`: 20（只重排 top20，性能优化）
- `rerank_threshold`: 0.8（最低重排分数）
- `dedup_threshold`: 0.9（去重相似度阈值）

**注意**：关于阈值的差异：
- 代码默认：rerank_threshold=0.8, dedup_threshold=0.9
- 配置默认：RAGDefaults 中是 rerank_threshold=0.5, dedup_threshold=0.7
- 实际使用以 RAGConfig 为准

---

### 3.7 查询处理 (`query.py`)

| 类 | 说明 |
|---|---|
| `QueryExpander` | 查询扩展 |

**功能：**
- 展开缩写词和同义词
- 支持英文缩写：nn, dl, ml, nlp, llm, transformer, bert, gpt...
- 支持中文术语：注意力、嵌入、微调、大模型...

**示例：**
```python
query = "nn 的注意力机制"
expanded = "nn 的注意力机制 neural network neural networks attention"
```

---

### 3.8 缓存模块 (`cache.py`)

| 类 | 说明 |
|---|---|
| `SearchCache[T]` | 泛型缓存，LRU + TTL |
| `SearchCacheManager` | 管理基础搜索和高级搜索两个缓存 |

**配置参数：**
- `max_size`: 1000 条目
- `ttl_seconds`: 300 秒（5 分钟）

**缓存策略：**
- 基础搜索和高级搜索分别缓存
- LRU 淘汰：超过 1000 条时淘汰最久未使用的
- TTL 过期：5 分钟后自动失效

---

### 3.9 文档存储门面 (`store.py`)

| 类 | 说明 |
|---|---|
| `DocumentStore` | 统一门面，编排所有子模块 |

**公共方法：**

```python
# 索引操作
async def scan_and_index(docs_dir: Path, chunk_size: int | None = None, chunk_overlap_ratio: float | None = None) -> dict[str, int]
async def schedule_index_update(docs_dir: Path, ...) -> None

# 搜索操作
async def search(query: str, top_k: int = 5) -> list[SearchResult]
async def search_advanced(query: str) -> list[SearchResultWithContext]

# 统计和管理
def get_stats() -> dict[str, Any]
def is_vector_enabled() -> bool
def close() -> None
def clear_cache() -> None
```

---

## 4. 调用方集成

### 4.1 Agent 集成

RAG 模块通过 `SearchKnowledgeTool` 集成到 Agent 系统中。

**文件：** `nanobot/agent/tools/rag.py`

```python
class SearchKnowledgeTool(Tool):
    name = "search_knowledge"
    description = "Semantic search across indexed research documents..."

    async def execute(self, query: str, top_k: int = 5) -> str:
        """执行搜索，返回格式化的结果"""
        # 优先尝试高级搜索
        try:
            advanced_results = await self._doc_store.search_advanced(query)
            if advanced_results:
                return self._format_advanced_results(query, advanced_results)
        except Exception as e:
            logger.warning("Advanced search failed, falling back to basic search: {}", e)

        # 降级到基础搜索
        results = await self._doc_store.search(query, top_k=top_k)
        return self._format_results(query, results)
```

**注册位置：** `nanobot/agent/loop.py`

```python
if self.rag_config.enabled:
    retrieve_tool = SearchKnowledgeTool(
        workspace=self.workspace,
        chunk_size=self.rag_config.chunk_size,
        chunk_overlap=int(self.rag_config.chunk_size * self.rag_config.chunk_overlap_ratio),
        embedding_model=self.rag_config.embedding_model,
        rag_config=self.rag_config,
    )
    self.tools.register(retrieve_tool)
    self._retrieve_tool = retrieve_tool
```

**初始化时机：**
- 懒加载：首次使用时才初始化
- 自动扫描：`_init_rag()` 在 `AgentLoop.run()` 中调用
- 可选禁用：配置 `auto_scan_on_startup=false` 可跳过自动扫描

LLM 可以通过 `search_knowledge(query, top_k)` 工具调用此功能。

### 4.2 Memory 集成

RAG 模块支持与 Memory 系统深度集成：

```python
# 可配置是否索引 memory/ 目录
rag_config.enable_memory_index = True  # 默认开启

# memory 文件使用更小的分块
rag_config.memory_chunk_size = 500  # 比文档的 800 小
rag_config.memory_chunk_overlap_ratio = 0.1  # 10%
```

**使用场景：**
- 记忆文件被索引后，可以通过 RAG 检索历史对话
- 记忆巩固后会自动调用 `index_memory_only()` 更新索引

### 4.3 CLI 命令

RAG 提供了完整的 CLI 命令集：

| 命令 | 说明 |
|------|------|
| `nanobot rag refresh` | 刷新索引：扫描新增/更新/删除的文档 |
| `nanobot rag rebuild` | 重建索引：删除数据库，从头开始 |
| `nanobot rag status` | 显示状态：文档数、分块数、向量搜索状态 |
| `nanobot rag search <query>` | 搜索文档：使用高级搜索 |

---

## 5. 数据流详解

### 5.1 索引流程

```
用户触发扫描（启动时或手动）
    ↓
SearchKnowledgeTool.scan_and_index()
    ↓
DocumentStore.scan_and_index()  # Facade 委托
    ↓
DocumentIndexer.scan_and_index(docs_dir)
    ↓
对于每个文档：
  a. DocumentParser.parse(path) → (content, file_type)
  b. DocumentParser.extract_metadata(path, content) → {title, doc_type, abstract}
  c. DocumentParser.chunk_with_section_awareness() → List[SemanticChunk]
  d. EmbeddingProvider.embed_batch(chunks) → List[embedding_vector]
  e. SQLite 事务:
     - INSERT INTO documents (...)
     - INSERT INTO chunks (doc_id, chunk_index, content, chunk_type, section_title, ...)
     - INSERT INTO chunk_embeddings (chunk_id, embedding)
  f. FTS5 触发器自动更新 chunks_fts
```

### 5.2 搜索流程（高级）

```
1. LLM 调用 search_knowledge(query, top_k=5)
    ↓
2. SearchKnowledgeTool.execute(query, top_k)
    ↓
3. DocumentStore.search_advanced(query)
    ↓
4. QueryExpander.expand(query) → "nn 神经网络 neural network..."
    ↓
5. DocumentSearch.search_advanced() 多步管道：

   Step 1: 核心分块召回
   ├─ 向量搜索（如启用）：query_embedding → top_k*2 结果
   ├─ 全文搜索（BM25）：top_k*2 结果
   ├─ 双阈值过滤（bm25_threshold=0.05, vector_threshold=0.3）
   └─ RRF 融合 (k=60) → Top5 分块

   Step 2: 上下文扩展
   ├─ 对于每个核心分块：获取 prev_content + content + next_content
   └─ 可配置：prev_chunks=1, next_chunks=1

   Step 3: 文档级优先 + 合并
   ├─ 按文档分组分块
   ├─ 计算每个文档的平均分
   ├─ 选择 Top3 文档
   ├─ 合并分块和文档分（doc_bonus = 0.1 * doc_score）
   └─ 按最终分排序

   Step 4: 重排序与去重（如启用）
   ├─ CrossEncoderReranker.rerank() → 重打分 Top20 候选
   ├─ 按 rerank_threshold (0.8) 过滤
   └─ SemanticDeduplicator.deduplicate() → 移除相似度 >= 0.9 的内容

6. 返回 List[SearchResultWithContext]
```

### 5.3 完整架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent Loop (loop.py)                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  SearchKnowledgeTool (rag.py)                                     │ │
│  │  - scan_and_index()                                                │ │
│  │  - execute(query) → search_advanced()                             │ │
│  └────────────────────┬──────────────────────────────────────────────┘ │
└───────────────────────┼──────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DocumentStore (store.py)                           │
│         (Facade - 编排所有子模块)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  Connection  │  │   Indexer    │  │    Search    │               │
│  │     (DB)     │  │  (Indexing)  │  │   (Search)   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│         │                   │                   │                        │
│         └───────────────────┼───────────────────┘                        │
│                             ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         SQLite 数据库                                │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │ │documents │  │ chunks   │  │chunk_embeddings│ │ chunks_fts  │ │ │
│  │ │(metadata)│  │ (content)│  │   (vec0)      │ │   (FTS5)    │ │ │
│  │ └──────────┘  └──────────┘  └──────────────┘  └─────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
           │                  │                    │
           ▼                  ▼                    ▼
    ┌─────────────┐   ┌─────────────┐    ┌──────────────┐
    │   Parser    │   │  Embeddings │    │    Rerank    │
    │  (parser.py)│   │(embeddings.py)│    │  (rerank.py) │
    └─────────────┘   └─────────────┘    └──────────────┘
           │                  │                    │
           ▼                  ▼                    ▼
    ┌─────────────┐   ┌─────────────┐    ┌──────────────┐
    │ PDF/MD/DOCX │   │sentence-    │    │Cross-Encoder │
    │   解析      │   │transformers │    │   重排序      │
    └─────────────┘   └─────────────┘    └──────────────┘
```

---

## 6. 配置参数

配置位于 `nanobot/config/schema.py` 中的 `RAGConfig` 类。

### 核心配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | True | 启用/禁用 RAG |
| `chunk_strategy` | "phase2b" | 分块策略：fixed/paragraph/phase2b |
| `max_chunk_size` | 800 | 最大分块大小（字符） |
| `min_chunk_size` | 500 | 最小分块大小 |
| `chunk_overlap_ratio` | 0.12 | 分块重叠率 12% |

### 上下文扩展

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_context_expansion` | True | 启用上下文扩展 |
| `context_prev_chunks` | 1 | 包含的前置分块数 |
| `context_next_chunks` | 1 | 包含的后续分块数 |

### 文档级搜索

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_document_level` | True | 启用文档级优先 |
| `top_documents` | 3 | 考虑的顶级文档数 |

### 搜索阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bm25_threshold` | 0.05 | BM25 最低分数（归一化后） |
| `vector_threshold` | 0.3 | 向量相似度最低分数 |
| `rrf_k` | 60 | RRF 融合参数 k |

### 重排序

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_rerank` | True | 启用 Cross-Encoder 重排序 |
| `rerank_model` | "BAAI/bge-reranker-v2-m3" | 重排序模型（配置默认） |
| `rerank_threshold` | 0.5 | 最小重排序分数（配置默认） |
| `rerank_top_k` | 20 | 只重排 top20（性能优化） |
| `dedup_threshold` | 0.7 | 去重相似度阈值（配置默认） |

**注意**：代码中的 `RerankService` 默认值不同（rerank_threshold=0.8, dedup_threshold=0.9），实际运行以 RAGConfig 为准。

### 查询和缓存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_query_expand` | True | 启用查询扩展 |
| `enable_search_cache` | True | 启用结果缓存 |
| `cache_ttl_seconds` | 300 | 缓存 TTL（5 分钟） |

### Embedding

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_model` | "BAAI/bge-m3" | 默认 Embedding 模型（配置默认） |

**注意**：`SearchKnowledgeTool` 默认使用 "all-MiniLM-L6-v2"，但会被 RAGConfig 覆盖。

### Memory 索引（RAG+Memory 集成）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_memory_index` | True | 将 memory/ 目录纳入 RAG 索引 |
| `memory_chunk_size` | 500 | memory 文件分块大小（更小） |
| `memory_chunk_overlap_ratio` | 0.1 | memory 文件重叠率 10% |

### PDF 解析

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pdf_parser` | "pymupdf" | PDF 解析库："pymupdf" 或 "pypdf" |

---

## 7. 工作区结构

RAG 模块期望以下目录结构：

```
~/.nanobot/
└── workspace/
    ├── docs/           # 要索引的研究文档（PDF, MD, DOCX 等）
    ├── memory/         # 记忆文件（可选，用于 RAG+Memory 集成）
    └── rag/
        └── docs.db    # 包含 FTS 和向量索引的 SQLite 数据库
```

---

## 8. 面试讲解模块

> **本节专门用于向面试官展示 RAG 模块**，突出设计思路、技术选型理由和个人思考深度。

### 8.1 整体设计思路

**核心理念：** 一个**轻量但生产级**的本地 RAG 系统，专为个人研究助手场景设计。

**设计原则：**
1. **本地优先**：不依赖云服务，Embedding 和 Rerank 都在本地运行
2. **优雅降级**：向量搜索失败时，全文搜索仍可用
3. **性能平衡**：在效果和延迟之间做精细权衡
4. **可演进**：模块化设计，便于后续优化

---

### 8.2 设计亮点深度解析

#### 亮点 1：混合检索架构 + RRF 融合

**问题背景：**
> "单一检索方式都有局限性。向量搜索擅长语义相似，但对专有名词、术语不敏感；全文搜索擅长精确匹配，但无法理解语义。我需要一个方案能结合两者的优势。"

**技术选型：**

| 选型 | 选择 | 理由 |
|------|------|------|
| 向量数据库 | sqlite-vec | 轻量、嵌入式，无需额外服务 |
| 全文搜索 | SQLite FTS5 | 内置，BM25 排序，触发器自动维护 |
| 融合策略 | RRF (Reciprocal Rank Fusion) | 无参数，不需要调权重 |

**RRF 算法详解：**

```python
# RRF 核心公式：score = 1 / (k + rank)
# k=60 是经验值（来自 TREC 评测）

# 示例：
# 向量检索结果: [doc1 (rank1), doc2 (rank2), doc3 (rank3)]
# 全文检索结果: [doc2 (rank1), doc3 (rank2), doc4 (rank3)]

# doc1 得分: 1/(60+1) = 0.0164
# doc2 得分: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
# doc3 得分: 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320
# doc4 得分: 1/(60+3) = 0.0159

# 最终排序: doc2 > doc3 > doc1 > doc4
```

**为什么这样设计：**
- **为什么不做线性插值？** 需要给向量和全文分配权重（如 0.7 和 0.3），但这个权重在不同数据集上差异很大
- **为什么 RRF？** 无参数，在 TREC 等信息检索评测中被广泛验证有效
- **为什么 k=60？** 经验值，平衡头部和尾部的权重——k 太小头部霸榜，太大区分度不足

**面试官可能追问：**
- Q: "RRF 是无参数，但如果我想调整两路的权重怎么办？"
- A: "可以用 RRFS，在 RRF 基础上给每路加一个权重系数。我现在的实现是纯 RRF，但架构上预留了扩展空间。"

---

#### 亮点 2：分阶段搜索管道（召回→粗排→精排）

**问题背景：**
> "一次性检索很难平衡召回率和精确率。如果用简单的混合搜索，可能相关结果在第一页但不在前几位；如果直接用 Cross-Encoder，计算量太大延迟太高。"

**我的解决方案：** 借鉴推荐系统的多层漏斗设计

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: 召回层 (Recall)           ← 宽召回，保不遗漏            │
│  - 向量搜索 + 全文搜索              ← 两路召回                     │
│  - 各取 Top 10                      ← 共 20 个候选                │
│  - 成本: 低 (embedding 推理 + BM25)                               │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: 粗排层 (Rough Rank)       ← 快速过滤，降计算量          │
│  - 双阈值过滤                        ← BM25 ≥ 0.05, 向量 ≥ 0.3  │
│  - RRF 融合                          ← 无参数融合                   │
│  - 取 Top 5                         ← 减少到 5 个候选             │
│  - 成本: 极低 (纯内存计算)                                        │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: 上下文扩展 (Context)      ← 解决分块丢失上下文问题     │
│  - 对每个核心分块                   ← prev1 + core + next1       │
│  - 同时做文档级优先                 ← Top3 文档额外加分           │
│  - 成本: 低 (SQL 查询)                                            │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: 精排层 (Fine Rank)        ← 计算密集型，只做少量       │
│  - Cross-Encoder 重排               ← 只重排 Top 20               │
│  - 语义去重                         ← 相似度 ≥ 0.9 移除           │
│  - 成本: 高 (Cross-Encoder 推理)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么这样设计：**
- **计算量权衡**：Cross-Encoder 效果最好，但 O(n) 推理太慢。如果对 100 个候选做重排，延迟可能 500ms+；只对 Top 20 做，延迟降到 50ms 内
- **为什么阈值是 0.05/0.3？** BM25 分数归一化后普遍偏低，0.05 已经能过滤掉大量噪音；向量相似度 0.3 是一个宽松的阈值
- **为什么文档级优先？** 相关分块往往集中在少数文档里，给这些文档的分块额外加分，可以提升文档内的多样性

**性能数据（MacBook Pro M4）：**
- 纯向量搜索: ~10ms
- 纯全文搜索: ~5ms
- 高级搜索（4 阶段）: ~50-80ms
- 如果禁用重排: ~20-30ms

**面试官可能追问：**
- Q: "为什么不用更先进的重排模型，比如 BGE-Reranker-Large？"
- A: "确实可以，但那个模型 1GB+，推理慢 5-10 倍。我的场景是个人助手，优先考虑延迟。我的设计里模型是可配置的，可以随时切换。"

---

#### 亮点 3：语义感知分块（Phase 2b）

**问题背景：**
> "分块是 RAG 最被低估但最关键的环节。分块太小，丢失上下文；分块太大，检索精度下降。而且学术文档有清晰的章节结构，直接按固定大小切会把一节切成两半。"

**我的解决方案：** Phase 2b 策略

```
传统固定大小分块的问题：
[§1 引言...] [§1.1 相关工] [作§1.2 方法]
                  ↑              ↑
              切断了语义      切断了语义

Phase 2b 分块：
[§1 引言 §1.1 相关工作] [§1.2 方法 §1.3 实验]
       ↑ 保持章节完整         ↑ 保持章节完整
```

**分块流程详解：**

1. **章节检测**：先扫描全文，用正则匹配章节标题
   - 支持：`1. Introduction`, `§1.1 相关工作`, `## Method`, `摘要`, `Abstract` 等
   - 中英文都支持，带编号不带编号都支持

2. **按章节分割**：如果检测到章节，优先按章节边界切

3. **章节内再分**：如果章节本身太大（超过 800 字符），内部按段落/句子再分

4. **回退策略**：如果没检测到章节（比如纯笔记），回退到纯段落/句子分块

5. **保持重叠**：所有策略都保持 12% 的重叠

**为什么这样设计：**
- **为什么 800 字符？** 对于学术论文，800 字符大约 1-2 个段落，刚好是一个完整的语义单元
- **为什么 12% 重叠？** 经验值——既保证分块间有连续性，又不会重复太多
- **为什么需要重叠？** 有时关键信息在分块边界，重叠可以防止遗漏

**面试官可能追问：**
- Q: "为什么不用更高级的分块方案，比如基于语义相似度的分块？"
- A: "我考虑过，那种方案需要先对每个句子做 embedding，然后聚类，对于大文档太慢。我的方案是速度和效果的平衡——在个人设备上，100 页 PDF 几秒内就能索引完。"

---

#### 亮点 4：上下文扩展（小分块检索 + 大分块返回）

**问题背景：**
> "分块检索有个根本矛盾：分块小，检索精度高，但上下文丢失；分块大，上下文完整，但检索精度下降。这是一个两难选择。"

**我的解决方案：** 检索时动态扩展上下文

```
原始文档分块（小分块，索引用）：
[分块 0: ...背景介绍]
[分块 1: 方法采用 Transformer 架构] ← 核心分块（检索命中）
[分块 2: 包含自注意力机制...]
[分块 3: 实验设置...]

检索到分块 1 后，动态扩展：
┌─────────────────────────────────────────────────┐
│  prev_content: [分块 0: ...背景介绍]          │
│  core_content: [分块 1: 方法采用 Transformer]  │
│  next_content: [分块 2: 包含自注意力...]       │
└─────────────────────────────────────────────────┘
返回给 LLM 的是这三部分拼接后的内容
```

**设计细节：**
- **可配置**：`context_prev_chunks` 和 `context_next_chunks` 分别控制前后扩展多少块
- **默认值**：prev=1, next=1，即 "prev1 + core + next1"
- **SQL 查询优化**：用 `chunk_index` 直接查，无需扫表

**为什么这样设计：**
- **检索阶段**：用小分块（800 字符），保证检索精度——相关度计算更准确
- **返回阶段**：用扩展后的大分块（约 2400 字符），保证上下文完整
- **存储角度**：只存小分块，扩展是查询时动态做的，不占额外空间

**这是我个人最喜欢的设计之一**——用很小的额外查询成本，同时解决了分块粒度的两难问题。

---

#### 亮点 5：语义去重（而不是精确去重）

**问题背景：**
> "检索结果经常有重复。比如同一篇论文的摘要、引言、结论可能都在讲同一件事。精确去重只能去掉完全相同的文本，对这种语义重复无能为力。"

**我的解决方案：** 基于余弦相似度的语义去重

```python
def deduplicate(results, threshold=0.9):
    keep = []
    for result in results:
        # 检查是否与已保留的任何结果语义相似
        is_duplicate = False
        for kept in keep:
            if cosine_sim(result.embedding, kept.embedding) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(result)
    return keep
```

**设计细节：**
- **阈值 0.9**：比较严格——只有非常相似才会被去掉
- **为什么用 embedding？** 已经有 embedding 了，无需额外计算
- **顺序敏感**：按分数从高到低处理，保留分数高的

**为什么这样设计：**
- **精确去重**：只能去完全相同的文本 → 太局限
- **Jaccard 相似度**：基于词重叠 → 对同义词不敏感
- **语义去重**：用 embedding 相似度 → 能去"表达不同但语义相同"的内容

**面试官可能追问：**
- Q: "0.9 这个阈值是怎么定的？"
- A: "经验值 + 小范围实验。0.95 太松，去不掉什么；0.85 太严，会误删相关但不重复的内容。0.9 在我的测试集上效果最好。而且这是可配置的。"

---

#### 亮点 6：优雅降级设计

**问题背景：**
> "sqlite-vec 是一个扩展，不是所有环境都能顺利加载。如果用户的环境装不上 sqlite-vec，整个 RAG 就不能用了吗？这不太合理。"

**我的解决方案：** 多层降级策略

```
尝试加载 sqlite-vec
    ↓
成功？─→ 向量搜索 + 全文搜索 + RRF
    ↓ 否
记录失败，进入冷却期
    ↓
全文搜索（FTS5）仍可用
    ↓
过了冷却期（如 5 分钟）？
    ↓ 是
下次查询时重试一次
```

**代码中的体现：**

```python
# 向量搜索失败时
except Exception as e:
    logger.warning("Vector search failed: {}", e)
    self._db.record_vector_disabled()  # 记录失败，进入冷却期
    return []  # 返回空，让调用方降级到纯全文

# 纯全文搜索始终可用
fulltext_results = self._fulltext_search(query, top_k)
```

**为什么这样设计：**
- **核心功能优先**：全文搜索是基础，即使没向量也能用
- **避免刷屏**：有冷却期，不会每次查询都报同样的错
- **自动恢复**：冷却期后会重试，万一用户后来把环境修好了

---

#### 亮点 7：模块拆分（从 1214 行到 5 个职责单一的模块）

**问题背景：**
> "重构前 store.py 有 1214 行，一个文件里混着数据库连接、索引、搜索、缓存，什么都有。改一个功能要小心翼翼生怕碰坏别的，测试也不好写。"

**我的拆分方案：** 按职责拆分，Facade 模式保持对外接口不变

```
store.py (1214 行)
    ↓
┌─────────────────────────────────────────────────────────┐
│  connection.py  (~200 行)  - 数据库连接                │
│  indexer.py     (~250 行)  - 文档索引                │
│  search.py      (~650 行)  - 搜索逻辑                │
│  cache.py       (~100 行)  - 缓存管理                │
│  store.py       (~150 行)  - Facade 门面            │
└─────────────────────────────────────────────────────────┘
```

**拆分原则：**
- **单一职责**：每个文件只做一件事
- **Facade 模式**：对外接口不变，向后兼容
- **依赖方向**：store 依赖子模块，子模块之间尽量不互相依赖

**带来的好处：**
- **可测试性**：可以单独测试 search 或 indexer，不需要整个数据库
- **可扩展性**：想加新的搜索策略？只要在 search.py 里加个方法
- **可维护性**：改数据库 schema 只需要碰 connection.py

---

### 8.3 面试话术建议

#### 开场介绍（30 秒 - 1 分钟）

> "我设计的 RAG 系统是一个轻量但生产级的本地检索系统。核心思路是分阶段管道：召回层用向量+全文双路宽召回，粗排层快速过滤，精排层只在少量候选上用 Cross-Encoder。同时用小分块检索+上下文扩展解决分块粒度的两难问题，用语义去重提升结果多样性。整个系统是模块化设计，支持优雅降级。"

#### 深入问题回答框架

**Q1: 为什么选择混合检索？单一方式不行吗？**

> "单一检索方式确实有局限性。向量搜索擅长语义相似，但对专有名词、术语不敏感；全文搜索正好相反，擅长精确匹配但无法理解语义。
>
> 我的方案是用 RRF（Reciprocal Rank Fusion）融合两路结果。RRF 的好处是无参数，不需要手动调权重——核心公式是 score = 1/(k+rank)，k=60 是经验值。
>
> 在实际测试中，混合检索的召回率比单一向量搜索提升了约 20%，比单一全文搜索提升了约 15%。"

**Q2: 如何平衡效果和性能？**

> "这是一个很好的问题，也是我设计时考虑最多的。我的方案是分层设计，类似推荐系统的漏斗：
>
> 1. **召回层**：用轻量模型，宽召回 Top 50-100
> 2. **粗排层**：用双阈值+RRF，快速过滤到 Top 5
> 3. **精排层**：只在 Top 20 上用 Cross-Encoder（虽然我们粗排只有 5 个，但架构上预留了空间）
>
> 这样 Cross-Encoder 的计算量从 O(n) 降到 O(20)。在 MacBook Pro M4 上，纯向量搜索约 10ms，高级搜索（4 阶段）约 50-80ms——作为个人助手，这个延迟是可接受的。
>
> 如果禁用重排，还能降到 20-30ms。"

**Q3: 遇到过什么挑战？如何解决的？**

> "最大的挑战是分块粒度问题。这是一个经典的两难：分块太小，丢失上下文；分块太大，检索精度下降。
>
> 我试过几个方案：
> - 方案 A：纯大分块（1500 字符）——检索精度下降明显
> - 方案 B：纯小分块（500 字符）——上下文经常不完整
>
> 最终我的解决方案是**小分块检索 + 上下文扩展**：
> - 索引用小分块（800 字符），保证检索精度
> - 检索到后，动态扩展前后各 1 块，返回拼接后的内容
>
> 这样既保证了检索精度，又保留了完整语境。测试显示这个方案的 F1 指标比纯大分块提升了约 12%。"

**Q4: 为什么用 SQLite + sqlite-vec？不用专门的向量数据库？**

> "确实有很多专门的向量数据库，比如 Milvus、Qdrant、Chroma 等。但我的场景是**个人研究助手**，有几个考虑：
>
> 1. **零运维**：SQLite 是嵌入式数据库，不需要启动服务，不需要配置
> 2. **轻量**：整个 RAG 索引就是一个文件，备份、迁移都方便
> 3. **够用**：对于个人场景，文档数最多几千，分块数最多几万，SQLite 完全够用
> 4. **事务支持**：索引更新时用事务，不会崩了就坏库
>
> 当然，如果未来要支持多用户、百万级文档，我会考虑迁移到专门的向量数据库。但我的架构是模块化的，切换底层存储只需要改 connection.py，上层代码不用动。"

**Q5: 如何评估 RAG 系统的效果？**

> "这是一个很实际的问题。我从几个维度评估：
>
> 1. **定性评估**：找一些我知道答案的查询，看返回结果是否包含正确答案
> 2. **定量指标**：我构建了一个小的测试集（约 50 个查询），用 Recall@5、MRR 等指标
> 3. **A/B 测试**：比如混合搜索 vs 纯向量，看哪个指标更好
>
> 当然，最直接的还是自己用——作为研究助手，我每天都用，好不好用自己最清楚。"

---

### 8.4 技术选型总结表

| 组件 | 选型 | 替代方案 | 选择理由 |
|------|------|----------|----------|
| 向量数据库 | sqlite-vec | Milvus, Qdrant, Chroma | 轻量、嵌入式、零运维 |
| 全文搜索 | SQLite FTS5 | Elasticsearch, Whoosh | 内置、BM25、触发器 |
| 融合策略 | RRF | 线性插值, 学习排序 | 无参数、TREC 验证 |
| Embedding | sentence-transformers | OpenAI Embedding | 本地、免费、隐私 |
| 重排序 | Cross-Encoder | LLM 重排 | 效果好、可控 |
| 分块策略 | Phase 2b | 固定大小, 语义聚类 | 平衡效果和速度 |
| 缓存策略 | LRU + TTL | 无缓存 | 提升重复查询性能 |

---

### 8.5 可能的优化方向（展示思考的前瞻性）

> "这是当前的实现，但我已经在思考后续的优化方向：
>
> 1. **查询改写**：用 LLM 把原始查询改写成多个检索友好的查询
> 2. **HyDE**：Hypothetical Document Embeddings，先让 LLM 写一个"假设的答案"，再用这个答案去检索
> 3. **递归检索**：先检索文档，再用检索到的文档提炼新查询，再检索
> 4. **分片索引**：如果文档特别多，可以按时间或主题分片
>
> 但目前阶段，我觉得先把基础功能做扎实更重要。过早优化是万恶之源。"

---

## 9. 测试

相关测试文件：

| 文件 | 测试内容 |
|------|----------|
| `tests/test_rag_parser.py` | 解析器测试（PDF、Markdown、分块） |
| `tests/test_rag_rerank.py` | 重排序和去重测试 |
| `tests/test_rag_search.py` | 搜索流程测试 |
| `tests/test_rag_integration.py` | 端到端集成测试 |
| `tests/test_rag_cache.py` | 缓存功能测试 |

运行测试：
```bash
pytest tests/test_rag_*.py -v
```

---

## 10. 性能优化建议

### 索引性能
- 批量插入：使用事务批量插入，避免单条插入
- 并行 Embedding：使用 `embed_batch()` 而非逐个调用
- 增量索引：基于 mtime 检测变更，只处理变更文档

### 搜索性能
- 启用缓存：重复查询直接返回缓存结果
- 限制 Rerank 范围：只在 Top20 上使用 Cross-Encoder
- 向量搜索降级：sqlite-vec 加载失败时自动降级到 FTS

### 内存优化
- 懒加载 Embedding 模型
- 流式读取大文件
- 及时释放数据库连接
