# RAG 图搜索增强方案（简化版）

**Date:** 2026-02-28
**Status:** Draft

---

## 一、目标

在现有 RAG 基础上增加**图搜索补充**，核心思路：
- 基础召回：保持现有（Top15）
- 图搜索：只是补充关联文档（最多5条）
- 最终融合：简单加权 → 重排 → Top8

---

## 二、整体流程

```
Query
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 第一步：基础召回                                            │
│   - FTS5 全文检索 → Top15                                 │
│   - 向量检索 → Top15                                      │
│   - RRF 融合 → Top15 (core_results)                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二步：图搜索增强（补充关联）                               │
│   - 从 core_results 提取关键词（简单匹配）                  │
│   - NetworkX 1-2 跳，获取关联关键词                        │
│   - 用关联关键词再检索 → 补充最多 5 条                     │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 最终融合                                                   │
│   - core_results + graph_results                           │
│   - 加权：全文40%+向量30%+图20%+时间10%                   │
│   - Cross-Encoder 重排 → Top8                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、关键词提取（最简方案）

不用 NER，就是从现有 chunks 里提取高频词：

```python
# 预定义关键词列表（可配置扩展）
KNOWN_KEYWORDS = {
    # 模型
    "transformer", "bert", "gpt", "llm", "cnn", "rnn", "lstm", "gan", "vae",
    "resnet", "vgg", "yolo", "alphafold", "chatgpt", "claude", "gemini",
    # 方法
    "attention", "self-attention", "cross-attention", "embedding",
    "fine-tuning", "pretraining", "transfer learning", "zero-shot", "few-shot",
    "backpropagation", "optimizer", "loss function",
    # 数据集
    "imagenet", "coco", "wikitext", "glue", "squad", "arxiv",
    # 评估
    "accuracy", "f1", "bleu", "rouge", "perplexity", "loss",
    # 实验
    "ablation", "benchmark", "baseline", "hyperparameter", "learning rate",
}

def extract_keywords_from_chunks(chunks: list[str]) -> list[str]:
    """从 chunks 内容中提取已知关键词"""
    import re

    found = set()
    text = " ".join(chunks).lower()

    for keyword in KNOWN_KEYWORDS:
        # 词边界匹配
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            found.add(keyword)

    return list(found)
```

---

## 四、图构建（索引时）

索引文档时，构建关键词共现图：

```python
import networkx as nx

class KeywordGraph:
    """关键词共现图"""

    def __init__(self):
        self.graph = nx.Graph()  # 无向图

    def add_chunk_keywords(self, keywords: list[str]):
        """添加 chunk 的关键词，建立共现关系"""
        if len(keywords) < 2:
            return

        # 添加节点
        for kw in keywords:
            if kw not in self.graph:
                self.graph.add_node(kw, weight=0)
            self.graph.nodes[kw]['weight'] += 1

        # 添加边（共现）
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                w1, w2 = keywords[i], keywords[j]
                if self.graph.has_edge(w1, w2):
                    self.graph[w1][w2]['cooccur'] += 1
                else:
                    self.graph.add_edge(w1, w2, cooccur=1)

    def find_related(self, seed_keywords: list[str], hops: int = 2, top_k: int = 5) -> list[str]:
        """从种子关键词出发，找到关联关键词"""
        if not seed_keywords:
            return []

        related = {}

        for seed in seed_keywords:
            if seed not in self.graph:
                continue

            # BFS 遍历
            visited = {seed: 0}
            queue = [(seed, 0)]

            while queue:
                node, dist = queue.pop(0)
                if dist >= hops:
                    continue

                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

                        # 权重 = 共现次数 * 节点出现次数
                        cooccur = self.graph[node][neighbor].get('cooccur', 1)
                        node_weight = self.graph.nodes[neighbor].get('weight', 1)
                        score = cooccur * node_weight

                        if neighbor not in related:
                            related[neighbor] = 0
                        related[neighbor] += score

        # 排序返回
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_related[:top_k]]
```

---

## 五、图持久化

索引时构建图，查询时加载：

```python
# 保存到 SQLite
def save_graph(self, db):
    import pickle
    data = pickle.dumps(self.graph)
    db.execute("INSERT OR REPLACE INTO graph_data (data) VALUES (?)", (data,))

# 加载图
def load_graph(self, db):
    import pickle
    row = db.execute("SELECT data FROM graph_data ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        self.graph = pickle.loads(row[0])
```

```sql
CREATE TABLE IF NOT EXISTS graph_data (
    id INTEGER PRIMARY KEY,
    data BLOB,
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);
```

---

## 六、检索流程（伪代码）

```python
async def search_with_graph(query: str, top_k: int = 8):
    # ========== 第一步：基础召回 ==========
    # 现有实现，取 Top15
    core_results = await self._base_recall(query, top_k=15)

    # ========== 第二步：图搜索补充 ==========
    graph_supplement = []
    if self.enable_graph and self._keyword_graph:
        # 2.1 从 core_results 提取关键词
        contents = [r.content for r in core_results]
        seed_keywords = extract_keywords_from_chunks(contents)

        # 2.2 图检索获取关联关键词
        related_keywords = self._keyword_graph.find_related(
            seed_keywords,
            hops=2,
            top_k=5
        )

        # 2.3 用关联关键词补充检索
        if related_keywords:
            # 组合查询
            extended_query = query + " " + " ".join(related_keywords)
            graph_supplement = await self._base_recall(extended_query, top_k=5)

            # 标记来源
            for r in graph_supplement:
                r.source = "graph"

    # ========== 最终融合 ==========
    # 合并 core_results + graph_supplement
    all_results = core_results + graph_supplement
    # 去重（按 path:chunk_index）

    # 加权融合
    for r in all_results:
        r.final_score = (
            r.fulltext_score * 0.4 +
            r.vector_score * 0.3 +
            (r.graph_score if r.source == "graph" else 0) * 0.2 +
            r.time_score * 0.1
        )

    # Cross-Encoder 重排
    final_results = await self.rerank(query, all_results)

    return final_results[:top_k]
```

---

## 七、配置

```python
class RAGConfig:
    # 基础召回（现有）
    top_k_base: int = 15

    # 图搜索增强
    enable_graph: bool = True
    graph_hops: int = 2
    graph_supplement_max: int = 5

    # 加权
    weight_fulltext: float = 0.4
    weight_vector: float = 0.3
    weight_graph: float = 0.2
    weight_time: float = 0.1
```

---

## 八、文件改动

| 文件 | 改动 |
|------|------|
| `nanobot/rag/graph.py` | **新增**：关键词图模块 |
| `nanobot/rag/store.py` | **修改**：增加图搜索阶段 |
| `nanobot/config/schema.py` | **修改**：添加图搜索配置 |
| `pyproject.toml` | **修改**：添加 networkx 依赖 |

---

## 九、实现顺序

1. **graph.py**：关键词图类（构建 + 检索）
2. **store.py**：集成图搜索到检索流程
3. **schema.py**：配置项
4. **测试**

---

## 十、简单对比

| | 原有方案 | 本方案 |
|---|---|---|
| 实体提取 | spaCy NER | 预定义关键词匹配 |
| 图构建 | 复杂 | 简单共现 |
| 改动量 | 大 | 小 |
| 依赖 | spaCy | networkx |

保持简单，就是：
1. 预定义一批关键词
2. 索引时建立共现关系
3. 查询时从结果扩展关联
