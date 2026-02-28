# RAG 轻量级实体增强方案

**Date:** 2026-02-28
**Status:** Design
**Type:** Feature

---

## 概述

在当前 RAG 架构基础上，增加**轻量级实体增强**能力，**不引入 spaCy、不引入图数据库**，仅通过关键词匹配和 SQLite 查询实现。

### 目标

- 从已有检索结果中提取关键词
- 通过关键词匹配召回更多相关 chunk
- 保持轻量级，不增加外部依赖
- 控制补充结果数量，避免噪声

---

## 方案设计

### 整体 Pipeline（在 [3] 和 [4] 之间插入）

```
用户 Query
    ↓
[1] Query 扩展 → 缩写/同义词展开
    ↓
[2] 基础召回 → BM25 + Vector 混合检索 + RRF 融合
    ↓
[3] 上下文扩展 → 补充前后 chunk + Document-level 加权
    ↓
[新增] 轻量级实体增强 ← 本方案新增
    ↓
[4] Cross-Encoder 重排 → Top25 重排 + 语义去重
    ↓
最终结果
```

---

## 模块设计

### 1. 关键词提取器 (`nanobot/rag/keyword.py`)

#### 设计思路

不做 NER，仅用以下方法提取关键词：
1. **从 section_title 提取**：已有语义 chunk 的 section_title 本身就是关键词
2. **从 chunk 内容提取**：用 TF-IDF 或简单统计提取高频词
3. **从查询本身提取**：直接保留用户查询中的重要词
4. **利用现有术语词典**：复用 `QUERY_ABBREVIATIONS` 中的科研术语

#### 接口设计

```python
from dataclasses import dataclass
from typing import Set, List

@dataclass
class KeywordSet:
    """关键词集合，带有来源标记。"""
    words: Set[str]
    source: str  # "section_title" | "content" | "query" | "abbreviation"


class KeywordExtractor:
    """轻量级关键词提取器（无 NER）。"""

    def __init__(self, min_word_len: int = 3, max_words: int = 20):
        self.min_word_len = min_word_len
        self.max_words = max_words
        # 停用词（中英文）
        self.stopwords = self._load_stopwords()

    def extract_from_results(
        self,
        results: List[SearchResultWithContext],
        query: str
    ) -> Set[str]:
        """
        从检索结果和查询中提取关键词。

        Args:
            results: 上下文扩展后的结果
            query: 原始用户查询

        Returns:
            关键词集合
        """
        keywords: Set[str] = set()

        # 1. 从查询提取
        query_keywords = self._extract_from_query(query)
        keywords.update(query_keywords)

        # 2. 从 section_title 提取
        for result in results:
            if result.chunk.section_title:
                section_keywords = self._extract_from_text(result.chunk.section_title)
                keywords.update(section_keywords)

        # 3. 从 chunk 内容提取（只取前 3 个高相关结果）
        for result in results[:3]:
            content_keywords = self._extract_from_text(result.chunk.content)
            keywords.update(content_keywords)

        # 4. 匹配现有术语词典
        abbr_keywords = self._match_abbreviations(keywords)
        keywords.update(abbr_keywords)

        # 过滤并返回 Top N
        return self._filter_and_rank(keywords)

    def _extract_from_query(self, query: str) -> Set[str]:
        """从查询中提取关键词（保留所有词，除了停用词）。"""
        words = self._tokenize(query)
        return {w for w in words if w not in self.stopwords and len(w) >= self.min_word_len}

    def _extract_from_text(self, text: str) -> Set[str]:
        """从文本中提取关键词（简单统计）。"""
        import re
        from collections import Counter

        # 提取符合科研术语模式的词
        # - 包含大小写混合 (YOLO, Transformer)
        # - 包含数字 (BERT4Rec, V2)
        # - 包含连字符 (state-of-the-art)
        pattern = r'\b(?:[A-Z][a-z]+)+(?:[A-Z][a-z]*)*\b|\b\w+(?:-\w+)+\b|\b\w*\d\w*\b'
        candidates = re.findall(pattern, text)

        # 统计频率，取 Top 10
        counter = Counter(candidates)
        return {w for w, _ in counter.most_common(10) if len(w) >= self.min_word_len}

    def _match_abbreviations(self, words: Set[str]) -> Set[str]:
        """匹配现有术语词典，扩展相关术语。"""
        from nanobot.rag.query import QUERY_ABBREVIATIONS

        expansions: Set[str] = set()
        for word in words:
            word_lower = word.lower()
            if word_lower in QUERY_ABBREVIATIONS:
                expansions.update(QUERY_ABBREVIATIONS[word_lower])
        return expansions

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（按空格和标点分割）。"""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _filter_and_rank(self, keywords: Set[str]) -> Set[str]:
        """过滤停用词并返回 Top N。"""
        # 移除停用词
        filtered = {w for w in keywords if w.lower() not in self.stopwords}
        # 只返回前 max_words 个（按长度降序，长词更可能是专业术语）
        sorted_words = sorted(filtered, key=lambda x: (-len(x), x))
        return set(sorted_words[:self.max_words])

    def _load_stopwords(self) -> Set[str]:
        """加载停用词（中英文）。"""
        return {
            # 英文停用词
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
            "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "whom", "whose", "where",
            "when", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now", "here",
            "there", "then", "once", "if", "about", "after", "again", "against",
            # 中文停用词
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
            "没有", "看", "好", "自己", "这",
        }
```

---

### 2. SQLite 实体匹配召回器

#### 设计思路

利用现有的 SQLite FTS5 索引，用关键词做 OR 查询，召回未在 [3] 阶段出现的 chunk。

#### 新增方法（在 `DocumentStore` 中）

```python
async def _boost_with_keywords(
    self,
    existing_results: List[SearchResultWithContext],
    keywords: Set[str],
    max_boost: int = 5,
    min_bm25: float = 0.1,
    min_vector: float = 0.2,
) -> List[SearchResultWithContext]:
    """
    用关键词召回更多相关 chunk。

    Args:
        existing_results: 已有结果（去重时用）
        keywords: 关键词集合
        max_boost: 最多补充多少条
        min_bm25: BM25 阈值
        min_vector: 向量相似度阈值

    Returns:
        补充的 chunk 列表
    """
    if not keywords or max_boost <= 0:
        return []

    # 构建 OR 查询
    keyword_list = list(keywords)
    fts_query = " OR ".join(f'"{k}"' for k in keyword_list[:10])  # 最多用 10 个关键词

    # 已有结果的 key 集合（用于去重）
    existing_keys = {f"{r.document.path}:{r.chunk.chunk_index}" for r in existing_results}

    # 用 FTS5 做关键词检索
    boosted_chunks = self._fulltext_search_keywords(fts_query, existing_keys, max_boost * 2)

    # 如果有 vector search，也可以做向量检索（可选）
    if self._vector_enabled and len(keyword_list) > 0:
        # 用关键词拼接成查询做向量检索
        vector_query = " ".join(keyword_list[:5])
        vector_chunks = await self._vector_search_keywords(vector_query, existing_keys, max_boost * 2)
        boosted_chunks.extend(vector_chunks)

    # 去重、排序、过滤
    boosted_chunks = self._deduplicate_and_filter(
        boosted_chunks,
        min_bm25,
        min_vector,
        max_boost
    )

    # 转换为 SearchResultWithContext
    return await self._expand_boosted_chunks(boosted_chunks)

def _fulltext_search_keywords(
    self,
    fts_query: str,
    existing_keys: Set[str],
    limit: int,
) -> List[Tuple[SearchResult, str]]:
    """关键词全文检索，排除已有结果。"""
    db = self._get_db()
    results: List[Tuple[SearchResult, str]] = []

    try:
        cursor = db.execute("""
            SELECT
                d.path,
                d.filename,
                c.chunk_index,
                c.content,
                bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            JOIN documents d ON c.doc_id = d.id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
        """, (fts_query, limit))

        for row in cursor:
            key = f"{row[0]}:{row[2]}"
            if key in existing_keys:
                continue

            bm25_score = row[4] if row[4] is not None else 1.0
            normalized = 1.0 / (1.0 + bm25_score)

            results.append((
                SearchResult(
                    path=row[0],
                    filename=row[1],
                    chunk_index=row[2],
                    content=row[3],
                    score=normalized,
                    source="keyword_boost",
                ),
                "fulltext"
            ))
    except Exception as e:
        logger.warning("Keyword FTS search failed: {}", e)

    return results

async def _vector_search_keywords(
    self,
    query: str,
    existing_keys: Set[str],
    limit: int,
) -> List[Tuple[SearchResult, str]]:
    """关键词向量检索，排除已有结果。"""
    if not self._vector_enabled:
        return []

    results = await self._vector_search(query, limit * 2)
    filtered: List[Tuple[SearchResult, str]] = []

    for r in results:
        key = f"{r.path}:{r.chunk_index}"
        if key not in existing_keys:
            filtered.append((r, "vector"))

    return filtered

def _deduplicate_and_filter(
    self,
    chunks: List[Tuple[SearchResult, str]],
    min_bm25: float,
    min_vector: float,
    max_boost: int,
) -> List[SearchResult]:
    """去重、过滤、排序。"""
    seen_paths: Set[str] = set()
    filtered: List[SearchResult] = []

    for chunk, source in chunks:
        # 检查阈值
        if source == "fulltext" and chunk.score < min_bm25:
            continue
        if source == "vector" and chunk.score < min_vector:
            continue

        # 去重（同一文档只留一个）
        if chunk.path in seen_paths:
            continue
        seen_paths.add(chunk.path)

        filtered.append(chunk)

    # 按得分排序
    filtered.sort(key=lambda x: x.score, reverse=True)
    return filtered[:max_boost]

async def _expand_boosted_chunks(
    self,
    chunks: List[SearchResult],
) -> List[SearchResultWithContext]:
    """将补充的 chunk 转换为 SearchResultWithContext（添加上下文）。"""
    # 复用现有的 _step2_context_expansion 逻辑
    return self._step2_context_expansion(chunks)
```

---

### 3. Pipeline 集成

#### 修改 `DocumentStore.search_advanced`

```python
async def search_advanced(self, query: str) -> List[SearchResultWithContext]:
    """
    Advanced multi-step search pipeline:
    1. Core chunk recall (BM25 + vector, dual thresholds, Top5)
    2. Context expansion (prev1 + core + next1)
    3. Document-level prioritization (Top3 docs)
    [NEW] 3.5. Lightweight keyword boost
    4. Cross-Encoder rerank (Top25, M4 optimized) + semantic dedup
    """
    # Step 1-3: Core recall -> Context expansion -> Document-level -> Merge
    core_results = await self._step1_core_chunk_recall(query)
    if not core_results:
        return []

    expanded_chunks = self._step2_context_expansion(core_results)
    top_docs = self._step3_document_level(core_results)
    merged_results = self._merge_context_and_document_results(expanded_chunks, top_docs)

    # [NEW] Step 3.5: Lightweight keyword boost
    if self.config.enable_keyword_boost:
        boosted_results = await self._apply_keyword_boost(query, merged_results)
        # 合并时，boosted 结果放在后面，得分稍低
        merged_results = self._merge_with_boosted(merged_results, boosted_results)

    # Step 4: Apply rerank and dedup
    if self.config.enable_rerank and self._rerank_service:
        final_results = await self._apply_rerank(query, merged_results)
        return final_results

    return merged_results

async def _apply_keyword_boost(
    self,
    query: str,
    base_results: List[SearchResultWithContext],
) -> List[SearchResultWithContext]:
    """应用关键词增强。"""
    from nanobot.rag.keyword import KeywordExtractor

    # 提取关键词
    extractor = KeywordExtractor()
    keywords = extractor.extract_from_results(base_results, query)

    if not keywords:
        return []

    logger.debug("Keyword boost using: {}", ", ".join(keywords))

    # 召回补充 chunk
    return await self._boost_with_keywords(
        base_results,
        keywords,
        max_boost=self.config.keyword_boost_max,
        min_bm25=self.config.keyword_boost_min_bm25,
        min_vector=self.config.keyword_boost_min_vector,
    )

def _merge_with_boosted(
    self,
    base_results: List[SearchResultWithContext],
    boosted_results: List[SearchResultWithContext],
) -> List[SearchResultWithContext]:
    """合并基础结果和补充结果。"""
    if not boosted_results:
        return base_results

    # 对补充结果做降权处理（确保不喧宾夺主）
    for r in boosted_results:
        r.final_score *= 0.8  # 降权 20%

    # 合并、去重、重新排序
    merged = base_results.copy()
    seen_keys = {f"{r.document.path}:{r.chunk.chunk_index}" for r in merged}

    for r in boosted_results:
        key = f"{r.document.path}:{r.chunk.chunk_index}"
        if key not in seen_keys:
            merged.append(r)

    # 重新排序
    merged.sort(key=lambda x: x.final_score, reverse=True)

    # 限制总数（给重排阶段）
    max_candidates = self.config.rerank_top_k
    return merged[:max_candidates]
```

---

### 4. 配置项（`RAGConfig`）

```python
class RAGConfig(Base):
    # ... 现有配置 ...

    # 轻量级关键词增强
    enable_keyword_boost: bool = True
    keyword_boost_max: int = 5          # 最多补充 5 条
    keyword_boost_min_bm25: float = 0.1  # BM25 阈值
    keyword_boost_min_vector: float = 0.2  # 向量阈值
```

---

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `nanobot/config/schema.py` | 添加 `enable_keyword_boost` 等配置 |
| `nanobot/rag/keyword.py` | **新增** - 关键词提取器 |
| `nanobot/rag/store.py` | 添加 `_boost_with_keywords`、`_apply_keyword_boost` 等方法，修改 `search_advanced` |
| `nanobot/rag/__init__.py` | 导出 `KeywordExtractor`（可选） |

---

## 边界控制

| 控制项 | 值 | 说明 |
|--------|-----|------|
| 最多补充 chunk 数 | 5 条 | 不稀释原有结果 |
| 关键词数量 | Top 20 | 避免查询过长 |
| 补充结果降权 | × 0.8 | 确保原有结果优先 |
| 候选集总数 | Top 25 | 给重排阶段 |
| 最小 BM25 阈值 | 0.1 | 过滤低质结果 |
| 最小向量阈值 | 0.2 | 过滤低质结果 |

---

## 验证方案

1. **功能验证**：
   ```bash
   # 先索引一些论文
   nanobot rag refresh

   # 搜索，应该能看到补充结果
   nanobot rag search "YOLOv9"
   ```

2. **日志验证**：
   - 查看 `Keyword boost using: ...` 日志确认关键词被提取
   - 确认补充结果数量 ≤ 5 条

3. **性能验证**：
   - 关键词提取 + 召回耗时 < 200ms
   - 整体 pipeline 延迟增加不明显

---

## 不做什么

- ❌ 不引入 spaCy 或任何 NER 模型
- ❌ 不引入图数据库（NetworkX 也不用）
- ❌ 不做复杂的图遍历
- ❌ 不改变现有 pipeline 的核心逻辑
- ❌ 不新增外部依赖

---

## 总结

这个方案**完全轻量**：
- ✅ 仅利用现有数据和索引
- ✅ 不新增外部依赖
- ✅ 边界控制严格（最多 5 条补充）
- ✅ 可以通过配置开关
