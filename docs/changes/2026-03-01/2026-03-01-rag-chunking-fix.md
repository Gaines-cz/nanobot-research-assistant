# RAG Chunking 参数修复

**创建日期**: 2026-03-01
**类型**: Bug 修复
**影响范围**: RAG 模块、Memory 模块

---

## 一、问题概述

在检查 config 模块时发现了以下问题：

### 1.1 `scan_and_index` 参数被忽略

**问题**: `DocumentStore.scan_and_index()` 方法接收 `chunk_size` 和 `chunk_overlap` 参数，但在 `_add_document` 内部**完全没有使用**这些参数，而是直接使用 `self.config` 中的值。

```python
# _add_document 方法 (修复前)
min_chunk_size = self.config.min_chunk_size  # 总是使用 config，忽略传入参数
max_chunk_size = self.config.max_chunk_size
overlap_ratio = self.config.chunk_overlap_ratio
```

这意味着从 memory consolidation 传入的特殊 memory chunk 参数会被完全忽略。

### 1.2 `memory_chunk_overlap` 语义不明确

**问题**:
- `RAGDefaults.MEMORY_CHUNK_OVERLAP = 50` - 绝对值（字符数）
- 但新的语义分块使用的是 `chunk_overlap_ratio` (比例，如 0.12)
- 这导致概念混淆

### 1.3 `enable_memory_index` 配置存在但未使用

**问题**: memory consolidation 流程中，RAG 索引更新是无条件执行的，没有检查 `enable_memory_index` 配置。

### 1.4 `_migrate_config` 迁移逻辑不完整

**问题**: 迁移函数没有处理旧的 `memoryChunkOverlap` 字段到新的 `memoryChunkOverlapRatio` 的转换。

---

## 二、解决方案

### 2.1 修改 `scan_and_index` 签名

```python
# 修复前
async def scan_and_index(
    self,
    docs_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict[str, int]:

# 修复后
async def scan_and_index(
    self,
    docs_dir: Path,
    chunk_size: int | None = None,
    chunk_overlap_ratio: float | None = None,
) -> dict[str, int]:
```

### 2.2 修改 `_add_document` 使用传入参数

```python
# 修复后
async def _add_document(
    self,
    db: sqlite3.Connection,
    path: Path,
    chunk_size: int | None = None,
    chunk_overlap_ratio: float | None = None,
) -> None:
    # Use provided parameters or fall back to config
    max_chunk_size = chunk_size if chunk_size is not None else self.config.max_chunk_size
    min_chunk_size = max_chunk_size // 2
    overlap_ratio = chunk_overlap_ratio if chunk_overlap_ratio is not None else self.config.chunk_overlap_ratio
```

### 2.3 修改 `memory_chunk_overlap` 为比例形式

```python
# RAGDefaults 修复前
MEMORY_CHUNK_OVERLAP: int = 50

# RAGDefaults 修复后
MEMORY_CHUNK_OVERLAP_RATIO: float = 0.1  # 10% overlap for memory files

# RAGConfig 修复前
memory_chunk_overlap: int = RAGDefaults.MEMORY_CHUNK_OVERLAP

# RAGConfig 修复后
memory_chunk_overlap_ratio: float = RAGDefaults.MEMORY_CHUNK_OVERLAP_RATIO
```

### 2.4 添加 `enable_memory_index` 检查

```python
# memory.py 修复后
if action in ("create", "merge", "replace"):
    if not self._rag_store.config.enable_memory_index:
        logger.debug("Memory index update skipped (enable_memory_index=False)")
    else:
        await self._rag_store.scan_and_index(
            self._memory.memory_dir,
            chunk_size=self._rag_store.config.memory_chunk_size,
            chunk_overlap_ratio=self._rag_store.config.memory_chunk_overlap_ratio,
        )
```

### 2.5 完善迁移逻辑

```python
def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # Migrate old memory_chunk_overlap (int) to memory_chunk_overlap_ratio (float)
    if "tools" in data and "rag" in data["tools"]:
        rag_cfg = data["tools"]["rag"]
        if "memoryChunkOverlap" in rag_cfg and "memoryChunkOverlapRatio" not in rag_cfg:
            old_value = rag_cfg.pop("memoryChunkOverlap")
            if isinstance(old_value, int) and old_value < 100:
                rag_cfg["memoryChunkOverlapRatio"] = old_value / 500  # Assume 500 chunk size
            else:
                rag_cfg["memoryChunkOverlapRatio"] = old_value

    return data
```

---

## 三、配置变更

### 3.1 RAGDefaults 变更

| 字段 | 修复前 | 修复后 |
|------|--------|--------|
| `MEMORY_CHUNK_OVERLAP` | `int = 50` | `MEMORY_CHUNK_OVERLAP_RATIO: float = 0.1` |

### 3.2 RAGConfig 变更

| 字段 | 修复前 | 修复后 |
|------|--------|--------|
| `memory_chunk_overlap` | `int` | `memory_chunk_overlap_ratio: float` |
| `chunk_overlap` | `int = 200` | `chunk_overlap: int | None = None` (deprecated) |

---

## 四、影响文件清单

| 文件 | 变更内容 |
|------|----------|
| `nanobot/config/schema.py` | 修改 `memory_chunk_overlap` 为 `memory_chunk_overlap_ratio`；标记 `chunk_overlap` 为 deprecated |
| `nanobot/config/loader.py` | 完善 `_migrate_config` 迁移逻辑 |
| `nanobot/rag/store.py` | 修改 `scan_and_index`、`_add_document`、`_update_document`、`schedule_index_update` 签名 |
| `nanobot/agent/memory.py` | 添加 `enable_memory_index` 检查；更新调用参数 |
| `nanobot/agent/tools/rag.py` | 更新调用参数 |
| `nanobot/agent/loop.py` | 更新调用参数 |
| `nanobot/cli/commands.py` | 更新调用参数和显示 |
| `tests/test_rag_integration.py` | 修复测试使用新参数 `chunk_overlap_ratio` |

---

## 五、验证结果

### 配置验证

```python
from nanobot.config.schema import RAGConfig

cfg = RAGConfig()
print(f'chunk_overlap_ratio: {cfg.chunk_overlap_ratio}')  # 0.12
print(f'chunk_overlap: {cfg.chunk_overlap}')  # None (deprecated)
print(f'memory_chunk_overlap_ratio: {cfg.memory_chunk_overlap_ratio}')  # 0.1
```

### 迁移验证

```python
from nanobot.config.loader import _migrate_config

# 旧格式 50 -> 0.1 ratio
test = {'tools': {'rag': {'memoryChunkOverlap': 50}}}
result = _migrate_config(test)
print(result['tools']['rag']['memoryChunkOverlapRatio'])  # 0.1
```

### 模块导入验证

```python
from nanobot.rag.store import DocumentStore
from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.rag import SearchKnowledgeTool
from nanobot.config.schema import Config
# 所有模块导入成功
```

---

## 六、向后兼容性

- 保留 `chunk_overlap` 字段作为 deprecated 选项，旧配置仍可工作
- 迁移逻辑自动将旧的 `memoryChunkOverlap` 转换为 `memoryChunkOverlapRatio`
