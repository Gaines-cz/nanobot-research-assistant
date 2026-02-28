# RAG (Retrieval-Augmented Generation) 集成方案

## 概述

为 nanobot 添加 RAG 功能，使其能够作为个人科研助手，支持索引和检索科研论文、项目书和笔记等文档。

## 需求

### 功能需求

| 功能 | 说明 |
|------|------|
| 文档格式 | PDF、Markdown、Word (.docx) |
| 向量存储 | sqlite-vec（轻量级，零依赖外部服务） |
| 集成方式 | 作为 Tools（由 LLM 自主调用） |
| 文档位置 | `workspace/docs` 目录 |

### 非功能需求

- 保持 nanobot 的轻量级特性（~4000 行核心代码）
- 可选依赖，不增加基础包的安装体积
- 使用现有的 LLM provider 进行 embeddings（不强制依赖 sentence-transformers）

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  AgentLoop                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌──────────────────────────────────────────────┐  │
│  │   ToolRegistry  │────▶│          RAG Tools                           │  │
│  │                 │     │  - index_documents (索引文档)                │  │
│  │                 │     │  - search_documents (搜索文档)               │  │
│  │                 │     │  - list_documents (列出文档)                 │  │
│  │                 │     │  - delete_document (删除文档)                │  │
│  └─────────────────┘     └──────────────────────────────────────────────┘  │
│                                      │                                        │
│                                      ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        VectorStore (sqlite-vec)                       │  │
│  │  - add_document(path, chunks, embeddings)                            │  │
│  │  - search(query_embedding, top_k) -> [chunk, score]                 │  │
│  │  - list_documents() -> [path, metadata]                              │  │
│  │  - delete_document(path)                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                        │
│                      ┌───────────────┴───────────────┐                        │
│                      ▼                               ▼                        │
│  ┌──────────────────────────────┐ ┌──────────────────────────────┐        │
│  │      DocumentParser          │ │      EmbeddingProvider        │        │
│  │  - parse_pdf()               │ │  (LiteLLM / sentence-        │        │
│  │  - parse_word()              │ │   transformers)               │        │
│  │  - parse_markdown()          │ │  - embed(text) -> [float]    │        │
│  │  - chunk_text()              │ └──────────────────────────────┘        │
│  └──────────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
nanobot/
├── rag/                        # 新增: RAG 模块
│   ├── __init__.py
│   ├── store.py                # VectorStore 和 DocumentParser
│   └── embeddings.py           # Embedding provider 封装
├── agent/
│   └── tools/
│       └── rag.py              # 新增: RAG 工具
├── config/
│   └── schema.py               # 修改: 添加 RAGConfig
```

## 详细设计

### 1. VectorStore (nanobot/rag/store.py)

使用 sqlite-vec 作为向量数据库。

**数据库 schema:**

```sql
-- 文档元数据表
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,  -- 'pdf', 'markdown', 'word'
    file_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON
);

-- 文档块表
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index)
);

-- sqlite-vec 虚拟表存储向量
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT32[{{dimensions}}]
);
```

**VectorStore 类接口:**

```python
class VectorStore:
    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        pass

    async def add_document(
        self,
        path: Path,
        content: str,
        file_type: str,
        metadata: dict | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """添加文档到向量库，返回 document_id"""
        pass

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """搜索相关文档块"""
        pass

    def list_documents(self) -> list[DocumentInfo]:
        """列出所有已索引文档"""
        pass

    def delete_document(self, path: str | Path) -> bool:
        """删除文档及其向量"""
        pass

    def get_document(self, path: str | Path) -> DocumentInfo | None:
        """获取文档信息"""
        pass
```

**数据类:**

```python
@dataclass
class DocumentInfo:
    id: int
    path: str
    filename: str
    file_type: str
    file_size: int | None
    created_at: datetime
    updated_at: datetime
    metadata: dict | None
    chunk_count: int


@dataclass
class SearchResult:
    document_id: int
    path: str
    filename: str
    chunk_index: int
    content: str
    score: float
```

### 2. DocumentParser (nanobot/rag/store.py)

文档解析器，支持多种格式。

```python
class DocumentParser:
    @staticmethod
    def parse(path: Path) -> tuple[str, str]:
        """自动检测格式并解析，返回 (content, file_type)"""
        ext = path.suffix.lower()
        if ext == '.pdf':
            return DocumentParser.parse_pdf(path), 'pdf'
        elif ext in ('.md', '.markdown'):
            return DocumentParser.parse_markdown(path), 'markdown'
        elif ext in ('.docx', '.doc'):
            return DocumentParser.parse_word(path), 'word'
        elif ext in ('.txt', '.rst', '.py', '.json'):
            return DocumentParser.parse_text(path), 'text'
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def parse_pdf(path: Path) -> str:
        """解析 PDF，返回纯文本"""
        pass

    @staticmethod
    def parse_word(path: Path) -> str:
        """解析 Word (.docx)，返回纯文本"""
        pass

    @staticmethod
    def parse_markdown(path: Path) -> str:
        """解析 Markdown，直接返回文本"""
        pass

    @staticmethod
    def parse_text(path: Path) -> str:
        """解析纯文本文件"""
        pass

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[tuple[str, int, int]]:
        """
        将文本分块，返回 list of (content, start_pos, end_pos)
        按句子边界分割，避免截断句子
        """
        pass
```

### 3. EmbeddingProvider (nanobot/rag/embeddings.py)

Embedding 提供者，支持 LiteLLM 和 sentence-transformers。

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed single text"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts"""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensions"""
        pass


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """使用 LiteLLM 的 embeddings API"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        pass

    async def embed(self, text: str) -> list[float]:
        pass

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass

    @property
    def dimensions(self) -> int:
        # 根据模型返回维度
        pass


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """使用 sentence-transformers 本地 embedding（可选）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        pass

    async def embed(self, text: str) -> list[float]:
        pass

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass

    @property
    def dimensions(self) -> int:
        pass
```

### 4. RAG Tools (nanobot/agent/tools/rag.py)

四个工具供 LLM 使用：

#### IndexDocumentsTool

索引 `workspace/docs` 下的文档。

```python
class IndexDocumentsTool(Tool):
    name = "index_documents"
    description = "Index documents from the docs directory for semantic search."

    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific file/directory paths to index (optional: if not provided, indexes all docs)",
            },
            "reindex": {
                "type": "boolean",
                "description": "Whether to reindex already indexed files",
                "default": False,
            },
        },
    }

    async def execute(self, paths: list[str] | None = None, reindex: bool = False) -> str:
        """索引文档，返回结果摘要"""
        pass
```

#### SearchDocumentsTool

搜索相关文档。

```python
class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = "Search indexed documents for relevant information."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, top_k: int = 5) -> str:
        """搜索并返回格式化的结果"""
        # 返回格式示例:
        # Results for "transformer architecture":
        #
        # [1] paper-attention-is-all-you-need.pdf (score: 0.89)
        # Chunk 3: "The Transformer follows this overall architecture using..."
        #
        # [2] survey-transformers.pdf (score: 0.82)
        # Chunk 7: "Transformer models have revolutionized NLP by..."
        pass
```

#### ListDocumentsTool

列出已索引文档。

```python
class ListDocumentsTool(Tool):
    name = "list_documents"
    description = "List all indexed documents."

    parameters = {
        "type": "object",
        "properties": {},
    }

    async def execute(self) -> str:
        """列出所有已索引文档"""
        pass
```

#### DeleteDocumentTool

删除索引的文档。

```python
class DeleteDocumentTool(Tool):
    name = "delete_document"
    description = "Delete a document from the index."

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the document to delete",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str) -> str:
        pass
```

### 5. 配置 (nanobot/config/schema.py)

```python
class RAGConfig(Base):
    """RAG configuration."""

    enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"  # "auto", "litellm", "sentence-transformers"


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    rag: RAGConfig = Field(default_factory=RAGConfig)  # 新增
```

### 6. AgentLoop 集成 (nanobot/agent/loop.py)

在 `_register_default_tools()` 中注册 RAG 工具：

```python
def _register_default_tools(self) -> None:
    # ... 现有工具 ...

    # RAG tools (optional dependencies)
    try:
        from nanobot.agent.tools.rag import (
            IndexDocumentsTool,
            SearchDocumentsTool,
            ListDocumentsTool,
            DeleteDocumentTool,
        )
        from nanobot.rag.store import VectorStore, DocumentParser
        from nanobot.rag.embeddings import LiteLLMEmbeddingProvider

        # 初始化 VectorStore（懒加载或在需要时初始化）
        # ...

        self.tools.register(IndexDocumentsTool(...))
        self.tools.register(SearchDocumentsTool(...))
        self.tools.register(ListDocumentsTool(...))
        self.tools.register(DeleteDocumentTool(...))
    except ImportError:
        logger.debug("RAG dependencies not installed, skipping RAG tools")
```

### 7. 依赖 (pyproject.toml)

作为可选依赖组：

```toml
[project.optional-dependencies]
rag = [
    "sqlite-vec>=0.1.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
]
rag-local = [
    "sqlite-vec>=0.1.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "sentence-transformers>=3.0.0",
]
```

## 工作流示例

### 索引文档

```
用户: "帮我索引一下 docs 目录下的论文"
    ↓
LLM: [调用 index_documents]
    ↓
IndexDocumentsTool:
  - 扫描 workspace/docs
  - 检测文件变更
  - DocumentParser.parse() 提取文本
  - 分块
  - 生成 embeddings
  - 存入 sqlite-vec
    ↓
返回: "Indexed 5 documents: paper1.pdf, paper2.pdf, ..."
```

### 检索文档

```
用户: "关于 transformer 架构，这些论文里说了什么？"
    ↓
LLM: [调用 search_documents(query="transformer architecture")]
    ↓
SearchDocumentsTool:
  - 生成 query embedding
  - sqlite-vec 相似度搜索
  - 返回 top_k 结果
    ↓
返回格式化结果给 LLM
    ↓
LLM: 根据检索结果回答用户问题
```

## 配置示例

```json
{
  "tools": {
    "rag": {
      "enabled": true,
      "chunkSize": 1000,
      "chunkOverlap": 200,
      "topK": 5,
      "embeddingModel": "text-embedding-3-small",
      "embeddingProvider": "auto"
    }
  }
}
```

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `nanobot/rag/__init__.py` | 新增 | RAG 模块初始化 |
| `nanobot/rag/store.py` | 新增 | VectorStore + DocumentParser |
| `nanobot/rag/embeddings.py` | 新增 | EmbeddingProvider |
| `nanobot/agent/tools/rag.py` | 新增 | RAG 工具 |
| `nanobot/config/schema.py` | 修改 | 添加 RAGConfig |
| `nanobot/agent/loop.py` | 修改 | 注册 RAG 工具 |
| `pyproject.toml` | 修改 | 添加可选依赖 |

## 测试计划

- 单元测试：`tests/test_rag_store.py`
- 工具测试：`tests/test_rag_tools.py`
- 集成测试：`tests/test_rag_integration.py`

## 后续扩展

- 支持更多文档格式（EPUB、HTML 等）
- 元数据过滤（按日期、作者、标签搜索）
- 混合检索（BM25 + 向量）
- 重排序（Reranker）
- 批量索引/更新优化