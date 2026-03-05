# RAG & Memory 模块改进计划

**计划日期**: 2026-03-03
**状态**: 待执行
**基于审计报告**: [2026-03-03-rag-memory-audit.md](../issues/2026-03-03-rag-memory-audit.md)

---

## 概述

本计划基于 RAG & Memory 模块审计报告，分三个阶段修复问题并优化架构。

- **Phase 1**: 修复 P0 问题（本周）
- **Phase 2**: 修复 P1 问题（下周）
- **Phase 3**: 架构优化（后续）

---

## Phase 1: 修复 P0 问题（本周）

**目标**: 解决所有严重问题，降低系统风险。

### Task 1.1: 统一默认模型

**问题**: P0-1
**文件**: `nanobot/agent/tools/rag.py`, `nanobot/config/schema.py`
**预估时间**: 10 分钟

**实现方案**:

```python
# SearchKnowledgeTool.__init__() 中
# 将默认值从 "all-MiniLM-L6-v2" 改为 "BAAI/bge-m3"
# 或者
# 完全移除默认值，强制使用 rag_config.embedding_model
```

**验收标准**:
- [ ] `SearchKnowledgeTool` 和 `RAGConfig` 默认模型一致
- [ ] 添加模型兼容性检查（如果数据库已有向量，提示重建索引）

---

### Task 1.2: 修复搜索缓存 Key

**问题**: P0-4
**文件**: `nanobot/rag/search.py`
**预估时间**: 15 分钟

**实现方案**:

```python
# 新增辅助函数
import hashlib

def _get_cache_key(query: str, top_k: int | None = None) -> str:
    """生成安全的缓存 key。"""
    key_bytes = query.encode("utf-8")
    if top_k is not None:
        key_bytes += f":{top_k}".encode("utf-8")
    return hashlib.sha256(key_bytes).hexdigest()

# 使用方式
cache_key = _get_cache_key(query, top_k)  # 基础搜索
cache_key = _get_cache_key(query)          # 高级搜索
```

**验收标准**:
- [ ] 缓存 key 使用 SHA-256 哈希
- [ ] 基础搜索包含 top_k，高级搜索不包含
- [ ] 测试缓存碰撞场景（如果有）

---

### Task 1.3: 提取余弦相似度到共享模块

**问题**: P0-8
**文件**: `nanobot/utils/helpers.py`, `nanobot/agent/memory.py`, `nanobot/rag/rerank.py`
**预估时间**: 15 分钟

**实现方案**:

```python
# nanobot/utils/helpers.py 新增
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    import math

    if len(a) != len(b):
        from loguru import logger
        logger.warning("Embedding length mismatch: {} vs {}", len(a), len(b))
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
```

**验收标准**:
- [ ] `helpers.py` 中有 `cosine_similarity()` 函数
- [ ] `memory.py` 导入使用
- [ ] `rerank.py` 导入使用
- [ ] 两个模块的测试通过

---

### Task 1.4: 修复 RAG 记忆搜索路径匹配

**问题**: P0-3
**文件**: `nanobot/agent/memory.py`
**预估时间**: 15 分钟

**实现方案**:

```python
from pathlib import Path

async def _search_related_memory(self, summary: str, top_k: int = 3) -> list[str]:
    """Search for related memories using RAG."""
    if not self._rag_store:
        return []

    try:
        results = await self._rag_store.search_advanced(summary)
        memory_path = self.memory_dir.resolve()

        memory_results = []
        for r in results:
            doc_path = Path(r.document.path).resolve()
            # 使用 Path.is_relative_to() 检查
            if doc_path == memory_path or memory_path in doc_path.parents:
                memory_results.append(r.combined_content)

        return memory_results[:top_k]
    except Exception as e:
        logger.warning("Memory search failed: {}", e)
        return []
```

**验收标准**:
- [ ] 使用 `Path` 对象而非字符串匹配
- [ ] 使用 `is_relative_to()` 或 `parents` 检查
- [ ] 跨平台测试通过（Windows/Mac/Linux）

---

### Task 1.5: 优化 Memory 索引更新

**问题**: P0-2
**文件**: `nanobot/rag/store.py`, `nanobot/rag/indexer.py`, `nanobot/agent/memory.py`
**预估时间**: 30 分钟

**实现方案**:

```python
# 方案：新增单文件索引接口

# indexer.py 新增
async def index_single_file(
    self,
    file_path: Path,
    chunk_size: Optional[int] = None,
    chunk_overlap_ratio: Optional[float] = None,
) -> bool:
    """索引单个文件（增量更新用）。"""
    # 实现只索引单个文件的逻辑
    # 检查 mtime，如果没变化就跳过
    pass

# store.py 新增
async def index_single_file(
    self,
    file_path: Path,
    chunk_size: Optional[int] = None,
    chunk_overlap_ratio: Optional[float] = None,
) -> bool:
    return await self._indexer.index_single_file(
        file_path, chunk_size, chunk_overlap_ratio
    )

# memory.py 中使用
# 不再调用 scan_and_index(memory_dir)
# 而是只索引变化的那个记忆文件
```

**验收标准**:
- [ ] `DocumentStore.index_single_file()` 接口可用
- [ ] 记忆巩固后只索引变化的文件
- [ ] 性能测试：记忆更新速度提升 10x+

---

### Task 1.6: 统一去重阈值

**问题**: P0-5
**文件**: `nanobot/config/schema.py`, `nanobot/rag/rerank.py`
**预估时间**: 5 分钟

**实现方案**:

明确优先级：`RAGConfig` > `RerankService` 默认值

**验收标准**:
- [ ] 文档中明确说明配置优先级
- [ ] 代码中注释清楚

---

### Task 1.7: 评估 RAG-based consolidation

**问题**: P0-6
**文件**: `nanobot/agent/memory.py`
**预估时间**: 30 分钟（调研 + 测试）

**实现方案**:

做 A/B 测试，比较：
- RAG-based consolidation (3 次 LLM)
- Direct consolidation (1 次 LLM)

**验收标准**:
- [ ] A/B 测试结果记录
- [ ] 决定是否保留 RAG-based consolidation
- [ ] 如果保留，简化流程

---

### Task 1.8: 改进原子操作回滚

**问题**: P0-7
**文件**: `nanobot/agent/memory.py`
**预估时间**: 20 分钟

**实现方案**:

```python
def safe_replace(self, file: MemoryFile, content: str) -> None:
    """安全替换文件内容（用临时文件 + rename）。"""
    path = self.memory_dir / file.value
    # 临时文件
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # 先写临时文件
    tmp_path.write_text(content.strip() + "\n", encoding="utf-8")
    # rename 是原子操作
    tmp_path.rename(path)

# 在 apply_operations_atomic() 中使用
```

**验收标准**:
- [ ] 所有文件写入都用临时文件 + rename
- [ ] 测试磁盘满场景（模拟）
- [ ] 回滚逻辑更健壮

---

## Phase 1 总结

| 任务 | 时间 | 优先级 |
|------|------|--------|
| Task 1.1: 统一默认模型 | 10m | P0 |
| Task 1.2: 修复缓存 Key | 15m | P0 |
| Task 1.3: 提取余弦相似度 | 15m | P0 |
| Task 1.4: 修复路径匹配 | 15m | P0 |
| Task 1.5: Memory 索引优化 | 30m | P0 |
| Task 1.6: 统一去重阈值 | 5m | P0 |
| Task 1.7: 评估 RAG-based | 30m | P0 |
| Task 1.8: 改进原子操作 | 20m | P0 |
| **总计** | **约 2.5 小时** | - |

---

## Phase 2: 修复 P1 问题（下周）

### Task 2.1: 添加索引元数据表

**问题**: P1-1
**文件**: `nanobot/rag/connection.py`
**预估时间**: 25 分钟

**实现方案**:

```sql
CREATE TABLE index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- 初始数据
INSERT INTO index_metadata VALUES
    ('schema_version', '1', strftime('%s', 'now')),
    ('embedding_model', 'BAAI/bge-m3', strftime('%s', 'now')),
    ('embedding_dimension', '1024', strftime('%s', 'now'));
```

**验收标准**:
- [ ] 元数据表存在
- [ ] 初始化时检查版本兼容性
- [ ] 模型变更时提示重建索引

---

### Task 2.2: 改进 get_memory_context()

**问题**: P1-3
**文件**: `nanobot/agent/memory.py`
**预估时间**: 20 分钟

**实现方案选项**:

1. **方案 A**: 改进关键词匹配（用更精确的模式）
2. **方案 B**: 用 embedding 做相似度检索（如果 RAG 可用）

**验收标准**:
- [ ] 关键词匹配更精确（如 "\bpapers?\b"）
- [ ] 或者用 embedding 相似度检索
- [ ] 测试误匹配场景

---

### Task 2.3: 消除 consolidation 代码重复

**问题**: P1-4
**文件**: `nanobot/agent/memory.py`
**预估时间**: 20 分钟

**实现方案**:

```python
def _get_old_messages(
    self,
    session: Session,
    archive_all: bool,
    memory_window: int,
) -> tuple[list[dict], int]:
    """获取要巩固的旧消息（两个策略共享）。"""
    # 提取共同逻辑
    pass

# 两个 consolidation 方法都调用这个
```

**验收标准**:
- [ ] 共同逻辑提取到私有方法
- [ ] 两个策略都能正常工作
- [ ] 测试通过

---

### Task 2.4: 修复 update_section() 正则

**问题**: P1-5
**文件**: `nanobot/agent/memory.py`
**预估时间**: 15 分钟

**验收标准**:
- [ ] 修复边界情况
- [ ] 添加单元测试覆盖边缘情况

---

### Task 2.5: 添加记忆访问统计

**问题**: P1-6
**文件**: `nanobot/agent/memory.py`
**预估时间**: 25 分钟

**实现方案**:

```python
@dataclass
class MemoryStats:
    """记忆访问统计。"""
    path: str
    access_count: int = 0
    last_access_at: Optional[float] = None
```

**验收标准**:
- [ ] 记录每次记忆访问
- [ ] 提供 get_stats() 接口
- [ ] 不影响性能

---

## Phase 3: 架构优化（后续）

### Task 3.1: 解耦 RAG + Memory（事件驱动）

**预估时间**: 1 小时

**实现方案**:

```python
# 引入事件总线（如果还没有）
from nanobot.bus.events import Event, EventBus

class MemoryFileUpdatedEvent(Event):
    """记忆文件更新事件。"""
    file_path: Path

# MemoryStore 发布事件
# RAG Indexer 订阅事件并更新索引
```

**验收标准**:
- [ ] MemoryStore 不再直接依赖 DocumentStore
- [ ] 通过事件通信
- [ ] 性能不下降

---

### Task 3.2: 添加搜索 explain 模式

**预估时间**: 30 分钟

**验收标准**:
- [ ] 可以返回每一步的中间结果
- [ ] 便于调试和优化

---

### Task 3.3: 添加记忆冷热数据分离

**预估时间**: 45 分钟

**验收标准**:
- [ ] 长期不用的记忆自动归档
- [ ] 检索时可以选择是否包含归档

---

## 测试计划

### 单元测试
- [ ] 每个修复都添加对应单元测试
- [ ] 覆盖边界情况

### 集成测试
- [ ] RAG + Memory 集成测试
- [ ] 端到端测试

### 性能测试
- [ ] 索引 100 个文档的时间
- [ ] 100 次搜索的平均延迟
- [ ] 记忆更新的时间（优化前后对比）

---

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 索引重建需要时间 | 中 | 低 | 提供迁移工具，后台重建 |
| 模型变更导致不兼容 | 高 | 中 | 添加版本检查，提示用户 |
| 事件驱动引入复杂度 | 中 | 低 | 保持接口简单，做好文档 |

---

## 附录：快速检查清单

执行前确认：
- [ ] 备份当前数据库
- [ ] 在 dev 分支测试
- [ ] 准备回滚方案

执行后确认：
- [ ] 所有测试通过
- [ ] 性能数据记录
- [ ] 文档更新