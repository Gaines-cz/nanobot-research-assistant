# 记忆去重功能实现

**Date:** 2026-02-28
**Type:** Feature
**Priority:** P1

---

## 概述

为 `MemoryStore` 添加了 `append_with_dedup()` 方法，在追加内容前自动检查重复，避免 LLM 在 memory consolidation 时重复追加相似内容到记忆文件。

## 问题背景

当前 `append()` 方法直接追加内容，没有任何去重检查。LLM 在 memory consolidation 时可能重复追加相似内容到 PAPERS.md 等文件，导致：
- 记忆文件膨胀
- 重复信息干扰检索
- 用户看到冗余内容

## 解决方案

### 分层去重

**Layer 1: 精确匹配去重（必选）**
- 检查 `content.strip()` 是否已存在于文件中
- 零依赖，快速，无额外配置
- 适用于完全相同的重复内容

**Layer 2: 语义相似去重（可选）**
- 使用 embedding 计算语义相似度
- 需要注入 `EmbeddingProvider`
- 可配置相似度阈值（默认 0.8）
- 适用于表述不同但含义相同的内容

---

## 改动详情

### 1. `nanobot/agent/memory.py`

**修改构造函数：**
```python
def __init__(self, workspace: Path, embedding_provider: Optional[EmbeddingProvider] = None):
    self.memory_dir = ensure_dir(workspace / "memory")
    self._embedding_provider = embedding_provider
```

**新增方法：**

```python
async def append_with_dedup(
    self,
    file: MemoryFile,
    content: str,
    *,
    similarity_threshold: float = 0.8,
) -> bool:
    """
    追加内容，自动跳过重复内容。

    Returns:
        True: 成功追加
        False: 跳过（重复内容）
    """
```

**辅助方法：**
- `_is_semantically_similar()` - 检查内容是否与已有内容语义相似
- `_cosine_similarity()` - 计算余弦相似度

### 2. `nanobot/agent/context.py`

**修改构造函数：**
```python
def __init__(self, workspace: Path, embedding_provider: Optional[Any] = None):
    self.workspace = workspace
    self.memory = MemoryStore(workspace, embedding_provider)
    self.skills = SkillsLoader(workspace)
```

### 3. `nanobot/agent/loop.py`

**初始化 embedding provider：**
```python
# Create embedding provider for memory dedup (if RAG dependencies available)
self._embedding_provider = None
if self.rag_config.enabled:
    try:
        from nanobot.rag import SentenceTransformerEmbeddingProvider
        self._embedding_provider = SentenceTransformerEmbeddingProvider(
            self.rag_config.embedding_model
        )
    except ImportError:
        logger.debug("RAG dependencies not installed, semantic dedup disabled")

self.context = ContextBuilder(workspace, self._embedding_provider)
```

**更新 `_consolidate_memory`：**
```python
return await MemoryStore(self.workspace, self._embedding_provider).consolidate(...)
```

### 4. `tests/test_memory_consolidation_types.py`

新增 `TestMemoryDeduplication` 测试类，包含 10 个测试用例：
- `test_append_with_dedup_exact_match` - 精确匹配去重
- `test_append_with_dedup_new_content` - 新内容正常追加
- `test_append_with_dedup_whitespace_normalized` - 空白字符标准化
- `test_append_with_dedup_empty_content` - 空内容处理
- `test_append_with_dedup_partial_match_not_blocked` - 部分匹配不阻止追加
- `test_append_with_dedup_semantic_similar` - 语义相似去重
- `test_append_with_dedup_semantic_different` - 语义不同正常追加
- `test_append_with_dedup_embedding_failure_graceful` - embedding 失败优雅降级
- `test_cosine_similarity` - 余弦相似度计算

---

## API 使用示例

```python
from nanobot.agent.memory import MemoryStore, MemoryFile

# 方式1：精确去重（无需 embedding provider）
store = MemoryStore(workspace)
result = await store.append_with_dedup(MemoryFile.PAPERS, content)
# result: True 表示已追加，False 表示跳过

# 方式2：语义去重（需要 embedding provider）
from nanobot.rag import SentenceTransformerEmbeddingProvider
embedding_provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
store = MemoryStore(workspace, embedding_provider)
result = await store.append_with_dedup(
    MemoryFile.PAPERS,
    content,
    similarity_threshold=0.8,
)
```

---

## 验收标准

| 标准 | 状态 |
|------|------|
| 精确去重：相同内容不会重复追加 | ✅ |
| 语义去重：相似内容（阈值以上）不会重复追加 | ✅ |
| 无依赖降级：没有 embedding provider 时，精确去重仍正常工作 | ✅ |
| 返回值正确：返回 `True` 表示已追加，`False` 表示跳过 | ✅ |
| 日志记录：跳过时记录 DEBUG 日志 | ✅ |
| 向后兼容：现有代码无需修改 | ✅ |

---

## 测试结果

```
tests/test_memory_consolidation_types.py::TestMemoryDeduplication ... 10 passed
tests/test_context_prompt_cache.py ... 2 passed
Total: 31 tests passed
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| embedding 调用耗时 | 中 | 语义去重设为可选，失败时不阻止追加 |
| 阈值设置不当 | 低 | 使用保守默认值 0.8，允许配置 |
| 误判阻止追加 | 低 | 失败时不阻止，记录警告日志 |

---

## 后续优化建议

1. **批量 embedding 优化**：当前对每个段落单独调用 `embed()`，可优化为 `embed_batch()` 减少网络请求
2. **缓存机制**：缓存已计算的 embedding，避免重复计算
3. **可配置开关**：添加配置项控制是否启用语义去重