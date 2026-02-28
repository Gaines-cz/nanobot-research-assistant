# RAG + 图搜索混合检索方案

## 概述

在现有 RAG（全文 + 向量检索）基础上，增加图搜索能力，实现更完整的科研文档检索。

## 背景

### 现有架构

- **全文检索**：SQLite FTS5（关键词匹配、实验ID、公式）
- **向量检索**：sqlite-vec（语义匹配）
- **检索结果**：混合搜索，取 Top15

### 目标场景（个人科研助手）

| 需求 | 说明 |
|------|------|
| 精准问答 | 某篇论文讲了什么 |
| 关系发现 | 这篇论文引用了哪些？某作者还写了什么？ |
| 主题探索 | 哪些论文讨论了这个主题？ |

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Unified Retrieve Tool                              │
│                    (用户无感知，内部协调各组件)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────────┐     ┌───────────────────┐           ┌───────────────────┐
│    基础召回层      │     │    图搜索增强层    │           │    重排序层        │
├───────────────────┤     ├───────────────────┤           ├───────────────────┤
│ • FTS5 全文检索   │     │ • 实体提取        │           │ • 加权融合        │
│ • 向量相似度检索  │ ──▶ │ • NetworkX 图检索 │ ─────────▶ │ • Cross-Encoder   │
│ • 去重 + 初步加权 │     │ • 关联补充        │           │ • 最终 Top8       │
└───────────────────┘     └───────────────────┘           └───────────────────┘
        │                             │
        ▼                             ▼
    Top15 结果              最多补充 5 条关联结果
```

## 分层设计

### 第一层：基础召回（精准优先）

**目标**：保证基础召回的准确性

| 搜索类型 | 技术 | 权重 |
|----------|------|------|
| 全文检索 | SQLite FTS5 (BM25) | 40% |
| 向量检索 | sqlite-vec (余弦相似度) | 30% |

**处理流程**：

1. 并行执行 FTS5 和向量搜索
2. 使用 RRF (Reciprocal Rank Fusion) 进行初步融合
3. 去重（基于 `path:chunk_index` 唯一键）
4. 取 Top15 结果作为基础召回

**RRF 公式**：

```
RRF_score(doc) = Σ 1 / (k + rank_i)
其中 k = 60（经验值），rank_i 是该文档在第 i 个排序结果中的排名
```

### 第二层：图搜索增强（关联提升）

**目标**：从 Top15 扩展到关联文档，提升完整性

**实体定义**（使用 spaCy NER）：

| 实体类型 | 来源 | 示例 | spaCy 标签 |
|----------|------|------|------------|
| 方法/模型 | 论文内容 | "Transformer", "BERT", "LSTM" | ENTITY TYPE 或自定义 |
| 论文名 | 标题/引用 | "Attention is All You Need" | WORK OF ART |
| 作者 | PDF 元数据 | "Vaswani", "Devlin" | PERSON |
| 数据集 | 内容 | "ImageNet", "COCO" | 自定义 |

### 第二层：图搜索增强（关联提升）

**目标**：从 Top15 扩展到关联文档，提升完整性

**spaCy NER 模型**：

```bash
# 安装
pip install spacy
python -m spacy download en_core_sci_sm  # 科学论文模型（推荐，~40MB）
# 或
python -m spacy download en_core_web_sm  # 通用模型（~40MB）
```

**spaCy NER 使用**：

```python
import spacy

nlp = spacy.load("en_core_sci_sm")

def extract_entities(text: str) -> list[dict]:
    """从文本中提取实体"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,  # ORG, PERSON, WORK_OF_ART 等
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities

# 示例输出
# [{'text': 'Transformer', 'label': 'ENTITY_TYPE'}, ...]
```

**图结构**：

```python
# NetworkX 有向图
# 节点
node: {
    "id": "keyword_Transformer",
    "type": "keyword" | "author" | "topic",
    "name": "Transformer",
    "doc_ids": [1, 2, 5],  # 关联的文档 ID
}

# 边
edge: {
    "source": "keyword_Transformer",
    "target": "keyword_Attention",
    "weight": 5,  # 共现次数
}
```

**图构建**（索引时）：

1. 用 spaCy NER 从文档内容提取实体（方法、论文名、作者等）
2. 从 PDF 元数据提取作者、标题
3. 构建节点（实体）和边（共现关系）
4. 持久化为 GraphML（启动时加载）

**图检索**：

```
输入：Top15 结果中的核心实体
   │
   ▼
1-2 跳图遍历
   │
   ▼
获取关联实体及其关联文档
   │
   ▼
补充最多 5 条关联文本
```

### 第三层：重排序（最终融合）

**目标**：整合所有来源，输出最终 Top8

**加权公式**：

```
final_score = 全文_score * 0.4 + 向量_score * 0.3 + 图谱_score * 0.2 + 时间_score * 0.1
```

**时间权重**：
- 使用文档的 `mtime`（修改时间）
- 最近修改的文档权重更高
- 归一化到 [0, 1] 区间

**Cross-Encoder 重排**：

```python
# 使用轻量级 Cross-Encoder 对 Top15 进行精排
# 模型：cross-encoder/ms-marco-MiniLM-L-6-v2
candidates = [doc.content for doc in top15]
scores = cross_encoder.predict([(query, doc) for doc in candidates])
sorted_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
return sorted_docs[:8]
```

## 数据结构

### 1. 文档表（已有）

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    filename TEXT,
    file_type TEXT,
    mtime REAL,
    stored_at REAL
);
```

### 2. Chunk 表（已有）

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER,
    chunk_index INTEGER,
    content TEXT,
    start_pos INTEGER,
    end_pos INTEGER
);
```

### 3. 实体表（新增）

```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'keyword', 'author', 'topic'
    doc_ids TEXT,        -- JSON array: [1,2,3]
    UNIQUE(name, type)
);
```

### 4. 实体关系表（新增）

```sql
CREATE TABLE entity_relations (
    id INTEGER PRIMARY KEY,
    source_id INTEGER,
    target_id INTEGER,
    weight INTEGER DEFAULT 1,
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);
```

### 5. GraphML 持久化

```xml
<!-- workspace/rag/graph.graphml -->
<graphml>
  <node id="keyword_Transformer" type="keyword" name="Transformer" doc_ids="[1,2,5]"/>
  <edge source="keyword_Transformer" target="keyword_Attention" weight="5"/>
</graphml>
```

## 检索流程（完整）

```
1. 查询输入：query = "Transformer 的注意力机制"
                 │
                 ▼
2. 基础召回（并行）
   ├── FTS5 搜索 → Top15
   └── 向量搜索 → Top15
                 │
                 ▼
3. RRF 融合 + 去重 → Top15
                 │
                 ▼
4. 提取核心实体
   - 从 Top15 内容提取关键词
   - 与已知实体匹配
                 │
                 ▼
5. 图搜索（1-2 跳）
   - 遍历 NetworkX 图
   - 获取关联实体
   - 补充关联文档（最多 5 条）
                 │
                 ▼
6. 加权融合
   - 全文 40% + 向量 30% + 图谱 20% + 时间 10%
   - 基础结果 + 补充结果
                 │
                 ▼
7. Cross-Encoder 重排
   - 对 Top15 做精排
   - 输出最终 Top8
```

## 工具接口

### Unified Retrieve Tool

```python
class RetrieveTool(Tool):
    name = "retrieve"
    description = "Search local documents with hybrid search (fulltext + vector + graph)."

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 8, "maximum": 20},
            "use_graph": {"type": "boolean", "default": True},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, top_k: int = 8, use_graph: bool = True) -> str:
        pass
```

### 单独图搜索 Tool（可选）

```python
class GraphSearchTool(Tool):
    name = "graph_search"
    description = "Find related entities and documents via knowledge graph."

    parameters = {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},  # 实体名称
            "hops": {"type": "integer", "default": 2, "maximum": 3},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["entity"],
    }
```

## 配置

```json
{
  "tools": {
    "rag": {
      "enabled": true,
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "top_k": 8,
      "embedding_model": "all-MiniLM-L6-v2",
      "use_graph": true,
      "graph_hops": 2,
      "graph_max补充": 5,
      "weights": {
        "fulltext": 0.4,
        "vector": 0.3,
        "graph": 0.2,
        "time": 0.1
      }
    }
  }
}
```

## 依赖

```toml
[project.optional-dependencies]
rag = [
    "sqlite-vec>=0.1.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "networkx>=3.0",
    "spacy>=3.7.0",
]
rag-local = [
    "sentence-transformers>=3.0.0",
]
reranker = [
    "sentence-transformers>=3.0.0",
]

# 模型下载（需单独执行）
# python -m spacy download en_core_sci_sm
```

## 文件结构

```
nanobot/
├── rag/
│   ├── __init__.py
│   ├── store.py          # 修改：增加图搜索功能
│   ├── embeddings.py     # 现有
│   ├── parser.py         # 现有
│   └── graph.py          # 新增：图搜索模块
├── agent/
│   └── tools/
│       └── rag.py        # 修改：统一 RetrieveTool
└── config/
    └── schema.py         # 修改：RAGConfig 增配
```

## 性能考虑

| 场景 | 数据量 | 预期延迟 |
|------|--------|----------|
| 基础召回 | 万条 chunk | < 100ms |
| 图检索 | 万条边 | < 50ms |
| Cross-Encoder | Top15 重排 | < 200ms |
| **总计** | - | **< 500ms** |

## 后续扩展

- [ ] 支持更多实体类型（DOI、项目名）
- [ ] 增量图更新（文档变更时只更新相关实体）
- [ ] 可视化知识图谱
- [ ] 多语言 embedding 支持

## 总结

| 层级 | 目标 | 关键技术 |
|------|------|----------|
| 基础召回 | 精准 | FTS5 + 向量 + RRF |
| 图搜索 | 关联 | spaCy NER + NetworkX + 实体关系 |
| 重排序 | 精排 | 加权融合 + Cross-Encoder |

该方案在保持轻量级的前提下，提供了完整的检索能力。
- spaCy `en_core_sci_sm` 模型（~40MB）用于科学论文实体识别
- NetworkX 处理万级数据完全可行
- 整体延迟 < 500ms