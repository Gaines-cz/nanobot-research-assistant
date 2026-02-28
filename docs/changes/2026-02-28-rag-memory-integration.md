# RAG + Memory 打通与 Consolidation 优化

**Date:** 2026-02-28
**Status:** 已完成

---

## 一、概述

本改动实现了两大功能：

1. **RAG + Memory 打通**：RAG 索引同时包含 `docs/` 和 `memory/` 目录，支持统一搜索
2. **Consolidation 优化**：新增 `MemoryStoreOptimized` 类，实现"压缩 → RAG搜索 → LLM决策 → 写入"的优化流程

---

## 二、改动文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `nanobot/agent/tools/rag.py` | 修改 | 支持 memory/ 索引 |
| `nanobot/agent/memory.py` | 新增 | MemoryStoreOptimized 类 |
| `nanobot/agent/loop.py` | 修改 | 集成优化版 consolidation |
| `nanobot/config/schema.py` | 新增 | RAG 配置项 |

---

## 三、详细改动

### 3.1 nanobot/agent/tools/rag.py

**改动内容**：

```python
# 1. 新增属性
self._memory_dir: Path | None = None

# 2. _ensure_initialized() 新增
self._memory_dir = self.workspace / "memory"
self._memory_dir.mkdir(parents=True, exist_ok=True)

# 3. scan_and_index() 修改
async def scan_and_index(self, include_memory: bool = None) -> dict[str, int]:
    """
    扫描并索引 docs/ 和 memory/ 目录
    - 默认使用 RAGConfig.enable_memory_index 配置
    - memory/ 使用较小的 chunk size (500)
    """
    # Index docs/
    docs_stats = await self._doc_store.scan_and_index(...)

    # Index memory/
    if include_memory:
        memory_stats = await self._doc_store.scan_and_index(
            self._memory_dir,
            chunk_size=self.rag_config.memory_chunk_size,  # 500
            chunk_overlap=self.rag_config.memory_chunk_overlap,  # 50
        )

    return total_stats

# 4. 新增方法
async def index_memory_only(self) -> dict[str, int]:
    """仅索引 memory/ 目录（用于 consolidation 更新后）"""
```

### 3.2 nanobot/config/schema.py

**新增配置项**：

```python
# Memory index (for RAG + Memory integration)
enable_memory_index: bool = True      # 是否将 memory/ 纳入 RAG 索引
memory_chunk_size: int = 500         # memory 文件的分块大小
memory_chunk_overlap: int = 50       # memory 文件的重叠大小
```

### 3.3 nanobot/agent/memory.py

**新增内容**：

```python
# 1. 新增导入
if TYPE_CHECKING:
    from nanobot.rag.store import DocumentStore

# 2. MemoryStore 新增 rag_store 参数
def __init__(
    self,
    workspace: Path,
    embedding_provider: Optional[EmbeddingProvider] = None,
    rag_store: Optional["DocumentStore"] = None,
):
    self.memory_dir = ensure_dir(workspace / "memory")
    self._embedding_provider = embedding_provider
    self._rag_store = rag_store  # 新增

# 3. 新增 MemoryStoreOptimized 类
class MemoryStoreOptimized:
    """
    优化版 MemoryStore，支持 RAG 搜索的 consolidation

    流程：
    1. LLM 压缩消息为摘要
    2. RAG 搜索相关记忆
    3. LLM 决策 (create/merge/replace/skip)
    4. 写入记忆文件
    """

    def __init__(self, memory_store: MemoryStore, rag_store: "DocumentStore"):
        self._memory = memory_store
        self._rag_store = rag_store

    async def consolidate(self, session, provider, model, *, archive_all=False, memory_window=50) -> bool:
        """优化版 consolidation"""
        # Step 1: 获取旧消息
        # Step 2: LLM 压缩
        summary = await self._compress_messages(old_messages, provider, model)

        # Step 3: RAG 搜索记忆
        related_memory = await self._search_related_memory(summary, top_k=3)

        # Step 4: LLM 决策
        if related_memory:
            decision = await self._decide_with_context(summary, related_memory, provider, model)
        else:
            decision = {"action": "create", "memory_update": summary}

        # Step 5: 写入
        ...
```

**新增 Prompt**：

```python
COMPRESSION_PROMPT = """将以下对话压缩为关键信息点（5-10 条）。..."""

DECISION_PROMPT = """判断如何处理新信息。..."""
```

### 3.4 nanobot/agent/loop.py

**改动内容**：

```python
# 1. 新增导入
from nanobot.agent.memory import MemoryStore, MemoryStoreOptimized

# 2. _consolidate_memory() 修改
async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
    """优先使用优化版 consolidation"""
    if self._retrieve_tool is not None:
        try:
            self._retrieve_tool._ensure_initialized()
            rag_store = self._retrieve_tool._doc_store

            if rag_store is not None:
                memory_store = MemoryStore(self.workspace, self._embedding_provider)
                optimized = MemoryStoreOptimized(memory_store, rag_store)
                return await optimized.consolidate(...)
        except Exception as e:
            logger.warning("Optimized consolidation failed, falling back: {}", e)

    # 降级到原版
    return await MemoryStore(self.workspace, self._embedding_provider).consolidate(...)
```

---

## 四、配置说明

### 4.1 RAG 配置

```python
class RAGConfig:
    # ... 现有配置 ...

    # Memory 索引（新增）
    enable_memory_index: bool = True      # 默认开启
    memory_chunk_size: int = 500         # memory 文件分块
    memory_chunk_overlap: int = 50
```

### 4.2 使用方式

```bash
# 重新扫描索引（包括 memory/）
nanobot rag scan

# 或在代码中
await tool.scan_and_index(include_memory=True)
```

---

## 五、Consolidation 流程对比

### 5.1 原有流程

```
消息累积 → LLM 处理全部消息 → 写入 MEMORY.md + HISTORY.md

问题：
- Token 消耗高（每次 100 条原始消息）
- 无 RAG 搜索，无法判断是否重复
- LLM 决策缺乏参考依据
```

### 5.2 优化后流程

```
消息累积
    ↓
LLM 压缩 → 摘要 (5-10 条关键点)
    ↓
RAG 搜索 MEMORY.md → 相关记忆 (top_k=3)
    ↓
├─ 无相似内容 → create new
└─ 有相似内容 → LLM 决策
    ├─ skip: 忽略
    ├─ merge: 合并新旧
    └─ replace: 替换旧内容
    ↓
写入 HISTORY.md + 更新 MEMORY.md
```

### 5.3 LLM 调用次数

| 场景 | 原有 | 优化后 |
|------|------|--------|
| Consolidation | 1 次 | 1-2 次 |
| Token 消耗 | 高 | 中 |

---

## 六、验收测试

```python
# 验证 1: 配置
from nanobot.config.schema import RAGConfig
c = RAGConfig()
assert c.enable_memory_index == True
assert c.memory_chunk_size == 500

# 验证 2: RAG Tool
from nanobot.agent.tools.rag import SearchKnowledgeTool
tool = SearchKnowledgeTool(workspace=Path(tmpdir), rag_config=c)
await tool.scan_and_index()  # 同时索引 docs/ + memory/

# 验证 3: MemoryStoreOptimized
from nanobot.agent.memory import MemoryStoreOptimized
optimized = MemoryStoreOptimized(memory_store, rag_store)
await optimized.consolidate(session, provider, model)
```

---

## 七、注意事项

1. **首次使用**：需要运行 `nanobot rag scan` 重新索引（包含 memory/）
2. **降级处理**：如果 RAG 不可用，自动降级到原版 MemoryStore
3. **索引更新**：每次 consolidation 后，memory/ 内容有更新，需要重新索引

---

## 八、代码审查修复

### 已修复问题

| 问题 | 修复 |
|------|------|
| write_file 方法不存在 | 改为 replace() |
| 只写入 PROFILE.md | 添加 target_file 分类 (profile/projects/papers/decisions/todos) |
| 缺少记忆分类 Prompt | 在 DECISION_PROMPT 中添加目标文件说明 |
| RAG 搜索路径过滤不准确 | 使用跨平台路径比较 |
| re 导入位置冗余 | 移除重复导入 |

### 决策 JSON 格式

```json
{
    "action": "create" | "merge" | "replace" | "skip",
    "target_file": "profile" | "projects" | "papers" | "decisions" | "todos",
    "reason": "判断理由",
    "history_entry": "HISTORY.md 条目",
    "memory_update": "记忆内容"
}
```

---

## 九、代码审查修复 (2026-03-01)

### 问题 1: write_file 方法不存在 🔴

**位置**: memory.py:778, 780

**问题**: 使用了不存在的 `write_file()` 方法

**修复**:
```python
# 修复前
self._memory.write_file(MemoryFile.PROFILE, merged)

# 修复后
self._memory.replace(MemoryFile.PROFILE, merged)
```

---

### 问题 2: 只写入 PROFILE.md 🔴

**问题**: 优化版 consolidation 只能写入 PROFILE.md，忽略了分类存储

**修复**: 添加 `target_file` 字段，支持分类存储

```python
# 新增字段映射
file_map = {
    "profile": MemoryFile.PROFILE,
    "projects": MemoryFile.PROJECTS,
    "papers": MemoryFile.PAPERS,
    "decisions": MemoryFile.DECISIONS,
    "todos": MemoryFile.TODOS,
}
target_memory_file = file_map.get(target_file, MemoryFile.PROFILE)
```

---

### 问题 3: 缺少记忆分类 Prompt 🟡

**问题**: DECISION_PROMPT 没有告诉 LLM 应该写入哪个文件

**修复**: 在 Prompt 中添加目标文件说明

```python
## 目标文件
根据内容类型选择目标文件：
- profile: 用户偏好、个人习惯、身份信息
- projects: 项目知识、技术栈、开发进度
- papers: 论文笔记、研究记录、学习内容
- decisions: 决策记录、为什么选 A 不选 B
- todos: 当前任务、下一步计划、待办事项
```

**新增决策字段**:
```json
{
    "action": "create" | "merge" | "replace" | "skip",
    "target_file": "profile" | "projects" | "papers" | "decisions" | "todos",
    "reason": "判断理由",
    "history_entry": "HISTORY.md 条目",
    "memory_update": "记忆内容"
}
```

---

### 问题 4: RAG 搜索路径过滤不准确 🟡

**问题**: Windows 路径使用 `\`，简单字符串匹配可能失效

**修复**: 使用跨平台路径比较

```python
# 修复前
if "/memory/" in r.document.path or r.document.path.endswith("/memory"):

# 修复后
memory_dir_str = str(self._memory.memory_dir)
memory_dir_str = memory_dir_str.replace("\\", "/").rstrip("/")

for r in results:
    doc_path = r.document.path.replace("\\", "/").rstrip("/")
    if memory_dir_str in doc_path or doc_path.endswith("/memory"):
        memory_results.append(r.combined_content)
```

---

### 问题 5: re 导入位置不规范 🟢

**问题**: 在方法内部导入 re，位置不规范

**修复**: 移除重复导入（文件顶部已有）

---

## 十、后续可优化项

1. **增量索引**：只更新变化的文件，不全量重索引
2. **搜索优化**：使用更轻量的搜索方法
3. **去重优化**：HISTORY.md 写入前去重
4. **RAG 索引更新**：consolidation 后自动更新 memory 索引
