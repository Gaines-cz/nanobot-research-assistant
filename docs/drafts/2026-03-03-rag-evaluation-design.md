# RAG 评测模块设计方案

**日期**: 2026-03-03
**状态**: 设计定稿
**版本**: v1.2
**更新说明**: v1.2 - 补充代码确认后的微调细则

---

## 1. 设计目标

为 nanobot 研究助手构建一套**自动化、可量化、可扩展**的 RAG 评测系统，核心目标：

| 目标 | 说明 |
|------|------|
| 效果验证 | 验证搜索结果是否相关 |
| 回归测试 | 防止代码变更导致功能退化 |
| 参数优化 | 辅助调优阈值、模型等参数 |
| 问题定位 | 识别检索失败的具体原因 |

---

## 2. 核心方案：段落自测（增强版）

### 2.1 核心思路

利用 RAG 索引中已有的 chunk，自动构造测试数据，无需人工准备。

**基础版**（Fallback）：
```
原始 chunk: "Attention(Q,K,V) = softmax(QK^T/d_k)V 是 Transformer 的核心..."

自动拆分：
- query: chunk 的第一句/前 100 字符（作为问题）
- golden_context: chunk 的剩余部分（作为参考答案）
```

**增强版**（默认，LLM-powered）：
```
原始 chunk: "Attention(Q,K,V) = softmax(QK^T/d_k)V 是 Transformer 的核心..."

LLM 生成（使用 memory_model）：
- query: "Transformer 中的 Attention 机制公式是什么？"（疑问句形式）
- golden_context: 原始 chunk 内容
- source_chunk_id: 123（关键：用于精确匹配）
- difficulty: "medium"（LLM 评估难度）
- tags: ["transformer", "attention"]
```

### 2.2 LLM Query 生成 Prompt

```python
CHUNK_TO_QUERY_PROMPT = """
根据以下文本片段，生成 1-2 个高质量的测试问题。

要求：
1. 问题应该是用户真实会问的形式（疑问句）
2. 问题不能直接包含答案关键词
3. 问题应该能通过检索这段文本得到答案
4. 评估问题难度（easy/medium/hard）
5. 提取 2-3 个关键词标签

文本：
{text}

生成的问题（JSON 格式）：
{{
  "queries": [
    {{
      "text": "问题 1",
      "difficulty": "medium",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}
"""
```

### 2.3 优势对比

| 优势 | 基础版 | 增强版 |
|------|--------|--------|
| 完全自动化 | ✅ | ✅ |
| 语义相关 | ✅ | ✅ |
| 接近真实查询 | ❌ | ✅ |
| 难度分级 | ❌ | ✅ |
| 成本 | 零成本 | 约 $0.01/100 条 |

### 2.4 评判方式

**多级评判标准**（优先级从高到低，满足任一即通过）：

| 优先级 | 级别 | 条件 | 说明 |
|--------|------|------|------|
| 1 | **ID 匹配** | Top-K 结果中包含 source_chunk_id | 最可靠（推荐首选） |
| 2 | **强通过** | Top-1 结果与 golden_context 相似度 >= 0.7 | 语义精确匹配 |
| 3 | **弱通过** | Top-3 中有结果相似度 >= 0.6 | 放宽条件 |
| 4 | **关键词通过** | Top-1 包含 golden_context 50%+ 关键词 | Fallback |

> **注意**（v1.2 更新）：推荐默认使用 **hybrid 模式**（ID 优先 + 语义 Fallback），因为：
> 1. `search_advanced()` 会做语义去重和上下文合并，原始 chunk ID 可能不在结果中
> 2. ID 匹配是最直接、最可靠的评判方式，但需要 Fallback 机制
> 3. hybrid 模式兼顾可靠性和稳健性

**相似度计算**：使用 cosine similarity 与 RAG 搜索一致的 embedding 模型。

---

## 3. 评测指标

### 3.1 核心指标

| 指标 | 说明 | 计算方式 | 目标值 |
|------|------|----------|--------|
| **Recall@K** | Top-K 结果中命中的比例 | hits / total_queries | >= 0.8 |
| **MRR** | 首个相关结果的排名倒数平均值 | mean(1 / first_hit_rank) | >= 0.7 |
| **NDCG@K** | 考虑排名质量的归一化折扣累积增益 | DCG / IDCG | >= 0.75 |
| **Hit Rate@K** | 至少命中 1 个的比例 | queries_with_hits / total | >= 0.9 |

### 3.2 辅助指标

| 指标 | 说明 | 用途 |
|------|------|------|
| **Avg Latency** | 搜索平均耗时（ms） | 性能监控 |
| **Pass Rate** | 通过阈值 query 的比例 | 直观指标 |
| **Difficulty Breakdown** | 按难度分组的 Recall | 识别薄弱环节 |
| **Failure Reason Breakdown** | 按失败原因分组的统计 | 定位问题 |

### 3.3 通过标准

```
整体通过条件（满足全部）：
- Recall@5 >= 0.80
- MRR >= 0.70
- Hit Rate@5 >= 0.90

可选 --strict 模式：
- Recall@5 >= 0.85
- NDCG@5 >= 0.80
```

---

## 4. 数据结构

### 4.1 测试数据集格式（增强版）

```yaml
# data/eval/datasets/default.yaml
version: "1.0"
generated_at: "2026-03-03T10:00:00"
generated_by: "llm-enhanced"
model_used: "qwen3.5-plus"
embedding_model: "BAAI/bge-m3"  # 记录生成时的 embedding 模型
num_samples: 50

queries:
  - id: "q1"
    query: "Transformer 中的 Attention 公式是什么？"  # LLM 生成
    golden_context: "Attention(Q,K,V) = softmax(QK^T/d_k)V 是 Transformer 的核心..."
    source_chunk_id: 123  # 关键：用于 ID 精确匹配
    source_doc: "transformer paper.md"
    difficulty: "medium"  # 难度分级
    tags: ["transformer", "attention"]

  - id: "q2"
    query: "残差连接的作用"
    golden_context: "残差连接可以缓解深层网络梯度消失问题"
    source_chunk_id: 456
    source_doc: "resnet paper.md"
    difficulty: "easy"
    tags: ["resnet", "deep-learning"]
```

### 4.2 评测结果格式（增强版）

```yaml
# data/eval/results/2026-03-03.yaml
timestamp: "2026-03-03T10:00:00"
dataset: "default"
config:
  embedding_model: "BAAI/bge-m3"
  top_k: 5
  similarity_threshold: 0.7
  judge_mode: "id_match"  # 新增：评判模式 (id_match / semantic / hybrid)

metrics:
  recall@5: 0.86
  mrr: 0.79
  ndcg@5: 0.82
  hit_rate@5: 0.94
  avg_latency_ms: 145

# 按难度分组统计
difficulty_breakdown:
  easy:
    count: 15
    recall@5: 0.93
  medium:
    count: 25
    recall@5: 0.84
  hard:
    count: 10
    recall@5: 0.70

# 按失败原因分组统计（新增）
failure_breakdown:
  not_in_index: 2      # 原始 chunk 未被索引
  recall_failed: 3     # 召回阶段未找到
  rerank_filtered: 1   # 被重排序过滤
  dedup_removed: 0     # 被语义去重移除
  low_similarity: 2    # 相似度过低（仅 semantic 模式）

# 详细结果
details:
  - query_id: "q1"
    query: "Transformer 中的 Attention 公式是什么？"
    golden_context_hash: "sha256:abc123..."
    hit: true  # 是否通过
    hit_rank: 1  # 首次命中的排名
    hit_reason: "id_match"  # 通过原因（新增）
    similarity_scores: [0.85, 0.72, 0.68, 0.45, 0.32]  # Top-5 相似度
    found_chunk_ids: [123, 456, 789, 101, 202]  # Top-5 的 chunk ID（新增）
    latency_ms: 132
    difficulty: "medium"
```

---

## 5. 模块架构

### 5.1 目录结构

```
nanobot/eval/
├── __init__.py
├── dataset.py          # 测试数据集管理
├── generator.py        # 自动生成测试数据（LLM-enhanced）
├── evaluator.py        # 评测执行器
├── metrics.py          # 指标计算
├── judge.py            # 评判逻辑（ID匹配 + 相似度 + 关键词）
└── cli.py              # CLI 命令

~/.nanobot/workspace/eval/
├── datasets/
│   ├── default.yaml      # 默认测试集
│   └── auto.yaml         # 自动生成的
├── results/
│   ├── 2026-03-03.yaml   # 按日期存储
│   └── 2026-03-04.yaml
└── reports/
    └── summary.md        # 汇总报告
```

### 5.2 核心类设计

```python
# dataset.py
class EvalQuery:
    """单条测试查询"""
    id: str
    query: str
    golden_context: str
    source_chunk_id: int  # 关键：用于 ID 匹配
    source_doc: str
    difficulty: str  # easy/medium/hard
    tags: list[str]

class EvalDataset:
    """测试数据集管理"""

    def __init__(self):
        self.version: str = "1.0"
        self.generated_at: str
        self.generated_by: str  # "llm-enhanced" or "basic"
        self.embedding_model: str  # 记录生成时的 embedding 模型
        self.queries: list[EvalQuery]

    @classmethod
    def load(cls, path: Path) -> "EvalDataset"
    def save(self, path: Path) -> None
    def filter_by_difficulty(self, difficulty: str) -> "EvalDataset"
    def sample(self, n: int) -> "EvalDataset"
    def validate_compatibility(self, current_embedding_model: str) -> bool:
        """检查数据集是否与当前 embedding 模型兼容"""

# generator.py
class DatasetGenerator:
    """自动生成测试数据集（LLM-enhanced）"""

    def __init__(
        self,
        doc_store: DocumentStore,
        llm_provider: LLMProvider,
        memory_model: str = "qwen3.5-plus",
    ):
        self.doc_store = doc_store
        self.llm_provider = llm_provider
        self.memory_model = memory_model

    async def generate(
        self,
        num_samples: int = 50,
        min_chunk_length: int = 200,
        strategy: str = "llm-enhanced",  # or "basic"
        sampling: str = "stratified",  # "random" or "stratified"
        output_path: Path = None,
    ) -> EvalDataset:
        """
        从索引中采样生成测试数据

        流程：
        1. 从 chunks 表中采样（支持分层采样）
        2. LLM 生成问题（增强版）或提取第一句（基础版）
        3. LLM 评估难度、提取标签
        4. 返回 EvalDataset
        """
        # Fallback: 如果 LLM 失败，使用基础版

    async def _stratified_sample(
        self,
        num_samples: int,
    ) -> list[tuple[int, str, str]]:
        """
        分层采样，确保覆盖：
        1. 不同文档类型（paper/note/concept）
        2. 不同 chunk_type（abstract/method/conclusion）
        3. 不同长度范围（短/中/长）
        """
        # 按类别分层采样

# evaluator.py
class QueryResult:
    """单条查询评测结果"""
    query_id: str
    query: str
    hit: bool
    hit_rank: int | None
    hit_reason: str  # "id_match" / "strong_semantic" / "weak_semantic" / "keyword"
    failure_reason: str | None  # "not_in_index" / "recall_failed" / "rerank_filtered" / ...
    similarity_scores: list[float]
    found_chunk_ids: list[int]
    latency_ms: float
    difficulty: str

class RAGEvaluator:
    """RAG 评测执行器"""

    def __init__(
        self,
        doc_store: DocumentStore,
        dataset: EvalDataset,
        embedding_provider: EmbeddingProvider,
    ):
        self.doc_store = doc_store
        self.dataset = dataset
        self.embedding_provider = embedding_provider
        self.judge = ResultJudge(similarity_threshold=0.7)

    async def evaluate(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        judge_mode: str = "hybrid",  # "id_match" / "semantic" / "hybrid"
    ) -> EvalResult:
        """
        执行评测

        流程：
        1. 对每个 query 执行 search_advanced()
        2. 应用评判标准（优先 ID 匹配）
        3. 记录失败原因
        4. 统计各项指标
        """

    async def _evaluate_single(
        self,
        query: EvalQuery,
        top_k: int,
        judge_mode: str,
    ) -> QueryResult:
        """评测单个 query"""
        start_time = time.time()

        # 执行搜索
        results = await self.doc_store.search_advanced(query.query)

        latency_ms = (time.time() - start_time) * 1000

        # 提取结果信息
        found_chunk_ids = [r.chunk.id for r in results]
        similarity_scores = [...]  # 计算与 golden_context 的相似度

        # 应用评判标准
        hit, hit_reason, failure_reason = self.judge.judge(
            results=results,
            query=query,
            similarity_scores=similarity_scores,
            mode=judge_mode,
        )

        return QueryResult(
            query_id=query.id,
            query=query.query,
            hit=hit,
            hit_rank=self._find_hit_rank(results, query) if hit else None,
            hit_reason=hit_reason,
            failure_reason=failure_reason,
            similarity_scores=similarity_scores,
            found_chunk_ids=found_chunk_ids,
            latency_ms=latency_ms,
            difficulty=query.difficulty,
        )

# judge.py
class ResultJudge:
    """评判逻辑"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.strong_threshold = similarity_threshold  # 0.7
        self.weak_threshold = 0.6
        self.keyword_overlap_threshold = 0.5

    def judge(
        self,
        results: list[SearchResultWithContext],
        query: EvalQuery,
        similarity_scores: list[float],
        mode: str = "hybrid",
    ) -> tuple[bool, str, str | None]:
        """
        评判搜索结果是否通过

        Args:
            results: 搜索结果列表
            query: 测试查询
            similarity_scores: 与 golden_context 的相似度列表
            mode: 评判模式
                - "id_match": 仅 ID 匹配（推荐）
                - "semantic": 仅语义相似度
                - "hybrid": ID 优先 + 语义 Fallback

        Returns:
            (是否通过, 通过原因, 失败原因)
            通过原因: "id_match" / "strong_semantic" / "weak_semantic" / "keyword"
            失败原因: "not_in_index" / "recall_failed" / "rerank_filtered" / "low_similarity"
        """
        if mode == "id_match":
            return self._judge_by_id(results, query)
        elif mode == "semantic":
            return self._judge_by_semantic(results, query, similarity_scores)
        else:  # hybrid
            # 先尝试 ID 匹配
            hit, reason, failure = self._judge_by_id(results, query)
            if hit:
                return hit, reason, None
            # Fallback 到语义匹配
            return self._judge_by_semantic(results, query, similarity_scores)

    def _judge_by_id(
        self,
        results: list[SearchResultWithContext],
        query: EvalQuery,
    ) -> tuple[bool, str | None, str | None]:
        """ID 精确匹配评判"""
        target_id = query.source_chunk_id

        for i, result in enumerate(results):
            if result.chunk.id == target_id:
                return True, f"id_match_rank_{i+1}", None

        # 未找到，尝试分析原因
        # 检查是否在索引中（需要查询数据库）
        return False, None, "recall_failed"

    def _judge_by_semantic(
        self,
        results: list[SearchResultWithContext],
        query: EvalQuery,
        similarity_scores: list[float],
    ) -> tuple[bool, str | None, str | None]:
        """语义相似度评判"""
        # 强通过：Top-1 >= 0.7
        if similarity_scores and similarity_scores[0] >= self.strong_threshold:
            return True, "strong_semantic", None

        # 弱通过：Top-3 >= 0.6
        for i, score in enumerate(similarity_scores[:3]):
            if score >= self.weak_threshold:
                return True, "weak_semantic", None

        # 关键词 Fallback
        if results:
            overlap = self._keyword_overlap(
                results[0].combined_content,
                query.golden_context
            )
            if overlap >= self.keyword_overlap_threshold:
                return True, "keyword", None

        return False, None, "low_similarity"

    def _keyword_overlap(self, text1: str, text2: str) -> float:
        """计算关键词重叠比例"""
        # 提取关键词（简单分词）
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words2:
            return 0.0
        overlap = len(words1 & words2)
        return overlap / len(words2)

# metrics.py
class MetricsCalculator:
    """指标计算"""

    @staticmethod
    def recall_at_k(results: list[QueryResult], k: int) -> float:
        """计算 Recall@K"""

    @staticmethod
    def mrr(results: list[QueryResult]) -> float:
        """计算 Mean Reciprocal Rank"""

    @staticmethod
    def ndcg_at_k(results: list[QueryResult], k: int) -> float:
        """计算 NDCG@K"""

    @staticmethod
    def hit_rate_at_k(results: list[QueryResult], k: int) -> float:
        """计算 Hit Rate@K"""

    @staticmethod
    def avg_latency(results: list[QueryResult]) -> float:
        """计算平均延迟"""

    @staticmethod
    def failure_breakdown(results: list[QueryResult]) -> dict[str, int]:
        """按失败原因分组统计"""
```

### 5.3 评判流程

```
                    检索结果 (Top-5)
                         │
                         ▼
            ┌────────────────────────────────┐
            │  结果中包含 source_chunk_id？    │─── Yes ───> ✅ ID 匹配通过
            └────────────────────────────────┘
                         │ No
                         ▼
            ┌────────────────────────┐
            │  Top-1 相似度 >= 0.7？   │─── Yes ───> ✅ 强通过（语义）
            └────────────────────────┘
                         │ No
                         ▼
            ┌────────────────────────┐
            │ Top-3 相似度 >= 0.6？    │─── Yes ───> ✅ 弱通过（语义）
            └────────────────────────┘
                         │ No
                         ▼
            ┌────────────────────────┐
            │ 关键词重叠 >= 50%?       │─── Yes ───> ✅ 关键词通过
            └────────────────────────┘
                         │ No
                         ▼
                    ❌ 未通过
                    记录失败原因
```

---

## 6. CLI 接口

### 6.1 命令列表

| 命令 | 说明 |
|------|------|
| `nanobot eval generate` | 自动生成测试数据集 |
| `nanobot eval run` | 运行评测 |
| `nanobot eval compare` | 对比两次评测结果 |
| `nanobot eval report` | 生成评测报告 |
| `nanobot eval diff` | 分析两次评测的差异（失败案例对比） |

### 6.2 使用示例

```bash
# 生成测试数据集（LLM-enhanced 模式，分层采样）
nanobot eval generate --num-samples 50 --strategy llm-enhanced --sampling stratified

# 生成测试数据集（基础版，快速）
nanobot eval generate --num-samples 50 --strategy basic

# 运行评测（Hybrid 模式，推荐：ID 优先 + 语义 Fallback）
nanobot eval run --dataset data/eval/datasets/auto.yaml --judge-mode hybrid

# 运行评测（仅 ID 匹配模式）
nanobot eval run --dataset data/eval/datasets/auto.yaml --judge-mode id_match

# 严格模式评测
nanobot eval run --dataset data/eval/datasets/auto.yaml --strict

# 对比两次结果
nanobot eval compare \
    --before data/eval/results/before.yaml \
    --after data/eval/results/after.yaml

# 生成详细报告
nanobot eval report --result data/eval/results/2026-03-03.yaml --output reports/rag-eval.md
```

### 6.3 输出示例（增强版）

```
$ nanobot eval run --dataset data/eval/datasets/auto.yaml --judge-mode hybrid

================================================================================
                           RAG 评测报告
================================================================================
数据集：auto.yaml (50 条)
生成方式：LLM-enhanced (qwen3.5-plus)
评测时间：2026-03-03 10:30:00
配置：embedding_model=BAAI/bge-m3, top_k=5, judge_mode=hybrid

--------------------------------------------------------------------------------
📊 核心指标
--------------------------------------------------------------------------------
  Recall@5:     0.86  ████████████████████░░  (43/50)
  MRR:          0.79  ██████████████████░░░░  (平均排名 1.3)
  NDCG@5:       0.82  ███████████████████░░░
  Hit Rate@5:   0.94  █████████████████████░  (47/50 至少命中 1)
  Avg Latency:  145ms

--------------------------------------------------------------------------------
📈 按难度分析
--------------------------------------------------------------------------------
  Easy (15 条):   Recall@5 = 0.93  █████████████████████░
  Medium (25 条): Recall@5 = 0.84  ███████████████████░░░
  Hard (10 条):   Recall@5 = 0.70  ██████████████░░░░░░░░

--------------------------------------------------------------------------------
📉 失败原因分析
--------------------------------------------------------------------------------
  recall_failed:    5 条 (71%)  召回阶段未找到目标 chunk
  rerank_filtered:  1 条 (14%)  被重排序移除
  dedup_removed:    1 条 (14%)  被语义去重移除

--------------------------------------------------------------------------------
🔍 失败案例分析 (Top-5)
--------------------------------------------------------------------------------
  #12 [Hard] "LayerNorm 和 BatchNorm 的区别"
     期望：chunk_id=789 来自 "norm paper.md"
     实际：未在 Top-5 结果中
     失败原因：recall_failed
     可能改进：检查 chunk 是否被正确索引

  #27 [Medium] "GELU 激活函数的公式"
     期望：chunk_id=456 来自 "activation functions.md"
     实际：排名第 8（在 Top-5 外）
     失败原因：recall_failed
     可能改进：优化查询扩展或分块策略

--------------------------------------------------------------------------------
✅ 评测通过 (Recall@5 = 0.86 >= 0.80)
================================================================================
```

### 6.4 返回值约定

```bash
# 评测通过
echo $?  # 0

# 评测未通过（--fail-under 模式下）
nanobot eval run --dataset auto.yaml --fail-under 0.8
echo $?  # 1
```

---

## 7. 进阶功能（后续）

### 7.1 人工校验模式

自动生成的数据集可能不够精确，可以添加人工校验接口：

```bash
# 导出为人工校验格式
nanobot eval export --dataset data/eval/datasets/auto.yaml --format csv

# 导入校验后的版本
nanobot eval import --dataset data/eval/datasets/auto.annotated.yaml
```

### 7.2 多模型对比

```bash
# 对比不同 embedding 模型
nanobot eval compare-models \
    --model1 all-MiniLM-L6-v2 \
    --model2 BAAI/bge-m3 \
    --dataset data/eval/datasets/auto.yaml
```

### 7.3 CI/CD 集成

```yaml
# .github/workflows/eval.yml
name: RAG Evaluation
on: [push, pull_request]
jobs:
  eval:
    runs-on: macos
    steps:
      - uses: actions/checkout@v4
      - name: Run RAG Eval
        run: |
          nanobot eval generate --num-samples 50 --sampling stratified
          nanobot eval run --dataset data/eval/datasets/auto.yaml --fail-under 0.8
```

---

## 8. 存储结构

```
~/.nanobot/
└── workspace/
    └── eval/
        ├── datasets/
        │   ├── default.yaml      # 默认测试集
        │   └── auto.yaml         # 自动生成的
        ├── results/
        │   ├── 2026-03-03.yaml   # 按日期存储
        │   └── 2026-03-04.yaml
        └── reports/
            └── summary.md        # 汇总报告
```

---

## 9. 实现计划

| 阶段 | 任务 | 预估时间 |
|------|------|----------|
| Phase 1 | 核心模块（dataset, generator, evaluator, metrics, judge） | 3-4 小时 |
| Phase 2 | CLI 命令 | 1-2 小时 |
| Phase 3 | 测试和调优（使用现有 RAG 索引测试） | 1-2 小时 |
| Phase 4 | CI/CD 集成（可选） | 1 小时 |

**总计**: 约 6-9 小时

---

## 10. 代码确认后的微调细则（v1.2 新增）

基于实际代码审查，补充以下重要实施细节：

### 10.1 数据库字段确认

**确认结果**：
- ✅ `documents` 表存在 `doc_type` 字段（用于分层采样）
- ✅ `chunks` 表存在 `chunk_type` 字段（用于分层采样）
- ✅ `SearchResultWithContext.chunk.id` 可用（用于 ID 匹配）

**结论**：分层采样可以正常使用。

### 10.2 搜索接口对比

| 接口 | 去重 | 合并 | 适用评测 |
|------|------|------|------------|
| `search_advanced()` | ✅ | ✅ | ❌ ID 匹配可能因去重/合并失败 |
| `search()` | ❌ | ❌ | ⚠️ 无 chunk.id，只有 path + chunk_index |

**结论**：`search_advanced()` 是实际生产使用的接口，评测时使用该接口更真实，但必须使用 hybrid 模式。

### 10.3 默认评判模式调整

**原设计**：推荐默认使用 `id_match`

**微调后**：推荐默认使用 `hybrid` 模式（ID 优先 + 语义 Fallback）

**理由**：
1. `search_advanced()` 会做语义去重和上下文合并，原始 chunk ID 可能不在结果中
2. hybrid 模式更稳健，能应对实际搜索流程

### 10.4 Relevance 明确定义

在计算 NDCG/MRR 指标时，relevance 判定标准（满足任一即算 relevant）：

| 模式 | Relevant 判定 |
|------|-----------|
| `id_match` | Top-K 包含 source_chunk_id |
| `semantic` | Top-1 相似度 >= 0.7 或 Top-3 >= 0.6 |
| `hybrid`（推荐） | ID 匹配 或 语义匹配任一满足 |

### 10.5 实施注意事项

1. **chunk ID 获取方式**：
   - `SearchResultWithContext.chunk.id` 可直接获取原始 chunk ID
   - 注意：去重后该 ID 可能不在最终结果列表中

2. **ID 匹配失败分析**：
   - 记录 `found_chunk_ids` 包含所有返回结果的 ID
   - 如果 `source_chunk_id` 不在 `found_chunk_ids` 中，尝试语义匹配

3. **分层采样实现**：
   - 按 `doc_type`（documents.doc_type）
   - 按 `chunk_type`（chunks.chunk_type）
   - 按 chunk 长度范围（短/中/长）

---

## 11. 技术决策总结（v1.2 更新）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Query 生成 | LLM-enhanced（默认） | 更接近真实用户查询，支持难度分级 |
| Fallback | 基础版（第一句提取） | LLM 失败时的保底方案 |
| **评判标准** | **Hybrid 模式（默认）** | ID 优先 + 语义 Fallback，应对去重/合并 |
| **采样策略** | **分层采样（默认）** | 确保覆盖不同类型、长度、来源 |
| 难度分级 | LLM 评估 | 自动化，无需人工标注 |
| 结果存储 | YAML（详细结果） | 可读性好，便于分析 |
| 持久化 | 默认保存 | 支持历史对比 |
| **失败原因** | **分类记录** | 便于定位问题 |
| **搜索接口** | **search_advanced()** | 使用实际生产接口，评测结果更真实 |

---

## 12. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM 生成质量不稳定 | 测试集质量下降 | Fallback 到基础版，人工抽检 |
| Embedding 维度变化 | 历史结果不可比 | 记录 embedding_model，版本管理 |
| 测试集过大 | 评测时间长 | 支持 --num-samples 限制，采样 |
| 阈值设置不当 | 误判通过/失败 | 实测调整，提供 --strict 模式 |
| **ID 匹配失败** | **无法评判** | **Hybrid 模式 Fallback 到语义匹配** |

---

## 13. 下一步行动

1. **确认设计方案** - 本设计文档
2. **创建任务列表** - 按 Phase 拆分
3. **实现 Phase 1** - 核心模块
4. **使用现有 RAG 索引测试** - 验证流程
5. **调整参数** - 基于实测结果优化阈值