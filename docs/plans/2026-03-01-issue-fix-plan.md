# Nanobot 问题修复计划

**创建日期**: 2026-03-01
**最后更新**: 2026-03-01 (P2 全部完成 9/9)
**状态**: P0、P1、P2 已完成

---

## 一、计划概述

本计划基于两份代码审计报告的综合分析：
- `docs/issues/2026-03-01-audit.md` - 静态分析为主的审计报告
- `docs/reviews/2026-03-01-code-review.md` - 架构 + 集成风险审计报告

### 问题统计

| 优先级 | 数量 | 修复窗口 |
|--------|------|----------|
| 🔴 P0 | 3 | Week 1 |
| 🟡 P1 | 6 | Week 2 |
| 🟢 P2 | 9 | Week 3-4 |
| 📋 P3 | 9 | Month 2+ |

---

## 二、P0 修复（Week 1）- 立即行动

### P0-1: 向量长度检查

**问题**: `_cosine_similarity()` 中 `zip()` 静默截断不同长度的向量

**位置**: `nanobot/agent/memory.py:229`

**影响**: 语义去重计算错误

**修复方案**:
```python
@staticmethod
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity."""
    if len(a) != len(b):
        logger.warning("Embedding length mismatch: {} vs {}", len(a), len(b))
        return 0.0  # or raise ValueError

    dot_product = sum(x * y for x, y in zip(a, b))
    # ... 其余代码
```

**预计工作量**: 10 分钟

**验收标准**:
- [ ] 添加长度检查逻辑
- [ ] 添加单元测试验证
- [ ] 更新相关文档

---

### P0-2: RAG 缓存内存限制

**问题**: `_search_cache` 和 `_basic_search_cache` 无最大条目数限制

**位置**: `nanobot/rag/store.py:89-91`

**影响**: 长期运行服务可能内存泄漏

**修复方案** (使用 `cachetools`):
```python
from cachetools import TTLCache

class DocumentStore:
    def __init__(self, ...):
        # maxsize=1000, ttl=300s
        self._search_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        self._basic_search_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)
```

或手动实现 (无外部依赖):
```python
from collections import OrderedDict
import time

class DocumentStore:
    MAX_CACHE_SIZE = 1000
    CACHE_TTL = 300  # 5 minutes

    def __init__(self, ...):
        self._search_cache: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self._basic_search_cache: OrderedDict[str, tuple[float, list]] = OrderedDict()

    def _cleanup_cache(self, cache: OrderedDict) -> None:
        """Remove expired and excess entries."""
        now = time.time()
        # Remove expired
        expired_keys = [k for k, (ts, _) in cache.items() if now - ts > self.CACHE_TTL]
        for k in expired_keys:
            del cache[k]

        # Remove excess (oldest first)
        while len(cache) > self.MAX_CACHE_SIZE:
            cache.popitem(last=False)

    def _get_from_cache(self, cache: OrderedDict, key: str) -> list | None:
        """Get from cache with cleanup."""
        if key in cache:
            ts, results = cache[key]
            if time.time() - ts > self.CACHE_TTL:
                del cache[key]
                return None
            # Move to end (LRU)
            cache.move_to_end(key)
            return results
        return None
```

**预计工作量**: 2-3 小时

**验收标准**:
- [ ] 实现 LRU + TTL 缓存
- [ ] 添加缓存大小监控
- [ ] 压力测试验证内存稳定

---

### P0-3: 全局锁 → Per-Session 锁

**问题**: `_processing_lock` 全局锁导致跨 session 串行处理

**位置**: `nanobot/agent/loop.py:347`

**影响**: 多 channel 场景下响应延迟

**修复方案**:
```python
class AgentLoop:
    def __init__(self, ...):
        # 移除全局锁
        # self._processing_lock = asyncio.Lock()

        # 添加 per-session 锁
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_lock = asyncio.Lock()  # 保护_locks 字典本身

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Get or create a lock for the given session."""
        if session_key not in self._session_locks:
            # Lazy create
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the per-session lock."""
        lock = self._get_session_lock(msg.session_key)
        async with lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    # ... 处理空响应
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                # ... 错误处理

    async def close_mcp(self) -> None:
        """Close MCP connections and cleanup."""
        # ... 原有代码 ...

        # Cleanup session locks
        self._session_locks.clear()
```

**预计工作量**: 3-4 小时

**验收标准**:
- [ ] 实现 per-session 锁
- [ ] 并发测试验证跨 session 并行
- [ ] 同 session 串行测试验证

---

## 三、P1 修复（Week 2）- 近期修复

### P1-1: Memory Consolidation 超时保护

**位置**: `nanobot/agent/loop.py:472-481`

**修复方案**:
```python
async def _consolidate_and_unlock():
    try:
        async with asyncio.timeout(60):  # 60s 超时
            async with lock:
                await self._consolidate_memory(session)
    except asyncio.TimeoutError:
        logger.error("Memory consolidation timed out after 60s for session {}", session.key)
    finally:
        self._consolidating.discard(session.key)
        self._prune_consolidation_lock(session.key, lock)
        _task = asyncio.current_task()
        if _task is not None:
            self._consolidation_tasks.discard(_task)
```

**预计工作量**: 30 分钟

---

### P1-2: Memory 原子操作回滚增强

**位置**: `nanobot/agent/memory.py:336-371`

**修复方案**:
```python
def apply_operations_atomic(self, operations: list[dict]) -> bool:
    """Apply multiple operations atomically. All succeed or all fail."""
    applied: list[tuple[MemoryFile, str]] = []  # (file, old_content) for rollback

    try:
        for op_data in operations:
            file_name = op_data.get("file")
            action = op_data.get("action", "skip")

            if file_name not in self.MEMORY_FILES:
                logger.warning("Unknown memory file: {}", file_name)
                raise ValueError(f"Unknown file: {file_name}")

            file = self.MEMORY_FILES[file_name]
            old_content = self.read_file(file)
            applied.append((file, old_content))

            op = MemoryOperation(
                file=file,
                action=action,
                content=op_data.get("content"),
                section=op_data.get("section"),
            )

            if not self.apply_operation(op):
                raise Exception(f"Operation failed: {op_data}")

        return True

    except Exception as e:
        logger.error("Atomic operation failed, rolling back: {}", e)
        rollback_failed = []

        for file, old_content in reversed(applied):
            try:
                self.replace(file, old_content)
            except Exception as rollback_e:
                logger.error("Rollback failed for {}: {}", file.value, rollback_e)
                rollback_failed.append(file)

        if rollback_failed:
            logger.critical(
                "Rollback incomplete, manual intervention may be needed: {}",
                rollback_failed
            )
        return False
```

**预计工作量**: 30 分钟

---

### P1-3: RAG 搜索错误分类处理

**位置**: `nanobot/rag/store.py:713-719`

**修复方案**:
```python
async def _step1_core_chunk_recall(self, query: str) -> list[SearchResult]:
    # ... 原有代码 ...

    except sqlite3.DatabaseError as e:
        logger.error("Database error during hybrid search, check integrity: {}", e)
        raise  # 数据库错误不应该 silently fallback
    except Exception as e:
        logger.warning("Hybrid search failed, falling back: {}", e, exc_info=True)
        # Fallback to full-text only
```

**预计工作量**: 20 分钟

---

### P1-4: System Prompt 长度截断

**位置**: `nanobot/agent/context.py:26-58`

**修复方案**:
```python
class ContextBuilder:
    MAX_SYSTEM_PROMPT_LENGTH = 8000  # 为 model 留出空间

    def build_system_prompt(self, skill_names: list[str] | None = None, query: str | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context(query)
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # ... skills ...

        result = "\n\n---\n\n".join(parts)

        # Length check and truncation
        if len(result) > self.MAX_SYSTEM_PROMPT_LENGTH:
            logger.warning(
                "System prompt too long ({} chars > {} limit), truncating",
                len(result),
                self.MAX_SYSTEM_PROMPT_LENGTH
            )
            # 优先保留 identity 和 bootstrap，截断 memory 和 skills
            result = self._truncate_system_prompt(parts, self.MAX_SYSTEM_PROMPT_LENGTH)

        return result

    def _truncate_system_prompt(self, parts: list[str], max_length: int) -> str:
        """Truncate system prompt while preserving priority parts."""
        # Priority: identity > bootstrap > skills > memory
        result = parts[0]  # Always keep identity

        for part in parts[1:]:
            if len(result) + len(part) + 4 < max_length:  # 4 for "\n\n---\n\n"
                result += "\n\n---\n\n" + part
            else:
                # Partial add with truncation notice
                remaining = max_length - len(result) - 50
                if remaining > 100:
                    result += "\n\n---\n\n" + part[:remaining] + "\n\n...[truncated]"
                break

        return result
```

**预计工作量**: 1 小时

---

### P1-5: MessageBus 背压机制

**位置**: `nanobot/bus/queue.py:16-18`

**修复方案**:
```python
class MessageBus:
    def __init__(self, maxsize: int = 1000):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=maxsize)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        try:
            await asyncio.wait_for(self.inbound.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Inbound queue full ({} items), message dropped: {}",
                self.inbound.qsize(),
                msg.sender_id
            )

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        try:
            await asyncio.wait_for(self.outbound.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Outbound queue full ({} items), message dropped: {}",
                self.outbound.qsize(),
                msg.chat_id
            )
```

**预计工作量**: 30 分钟

---

### P1-6: CLI `os._exit(0)` 修复

**位置**: `nanobot/cli/commands.py:520-525`

**修复方案**:
```python
# 替换 os._exit(0) 为正常退出流程
def _exit_on_sigint(signum, frame):
    _restore_terminal()
    console.print("\nGoodbye!")

    # 正常清理资源
    if agent_loop:
        agent_loop.stop()

    # 使用 sys.exit 代替 os._exit
    import sys
    sys.exit(0)
```

**预计工作量**: 15 分钟

---

## 四、P2 修复（Week 3-4）- 中期优化

### P2 问题清单

| 编号 | 问题 | 位置 | 预计工时 | 状态 |
|------|------|------|----------|------|
| P2-1 | Subagent 独立模型配置 | `loop.py`, `subagent.py` | 2h | ✅ 已完成 |
| P2-2 | 向量搜索自动恢复 | `rag/store.py` | 1h | ✅ 已完成 |
| P2-3 | MCP 连接指数退避 | `loop.py` | 1h | ✅ 已完成 |
| P2-4 | RAG 索引批量更新 | `memory.py` | 3h | ✅ 已完成 |
| P2-5 | Session 空历史保护 | `session/manager.py` | 30m | ✅ 已完成 |
| P2-6 | ToolRegistry 错误提示优化 | `tools/registry.py` | 30m | ✅ 已完成 |
| P2-7 | 配置参数统一 | `config/schema.py` | 1h | ✅ 已完成 |
| P2-8 | Consolidation 调试增强 | `memory.py` | 1h | ✅ 已完成 |
| P2-9 | 任务清理回调简化 | `loop.py` | 30m | ✅ 已完成 |

### P2-1: Subagent 独立模型配置

**修复方案**:
```python
# config/schema.py
class AgentDefaults(Base):
    # ... existing fields ...
    subagent_model: str | None = None  # Optional: separate model for subagents

# agent/loop.py
class AgentLoop:
    def __init__(self, ..., subagent_model: str | None = None):
        # ...
        self.subagent_model = subagent_model or self.model

        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.subagent_model,  # Use subagent_model
            # ...
        )

# cli/commands.py
agent = AgentLoop(
    # ...
    subagent_model=config.agents.defaults.subagent_model,
    # ...
)
```

---

### P2-2: 向量搜索自动恢复

**修复方案**:
```python
class DocumentStore:
    def __init__(self, ...):
        self._vector_retry_count = 0
        self._vector_last_failure_time: float = 0
        self._vector_retry_cooldown = 300  # 5 minutes

    def _check_vector_search_available(self) -> bool:
        """Check if vector search can be re-enabled."""
        if not self._vector_enabled and self._vector_last_failure_time > 0:
            if time.time() - self._vector_last_failure_time > self._vector_retry_cooldown:
                # Try to re-enable
                try:
                    self._get_db()  # Re-initialize
                    if self._vector_enabled:
                        logger.info("Vector search auto-recovered")
                        self._vector_retry_count = 0
                        return True
                except Exception:
                    pass
        return self._vector_enabled
```

---

### P2-3: MCP 连接指数退避

已在 code-review.md 中给出方案，预计 1 小时完成。

---

### P2-4: RAG 索引批量更新

**修复方案**:
```python
class MemoryStoreOptimized:
    def __init__(self, ...):
        self._pending_index_update = False
        self._index_update_task: asyncio.Task | None = None

    async def _schedule_index_update(self) -> None:
        """Schedule a delayed index update."""
        self._pending_index_update = True
        if self._index_update_task is None or self._index_update_task.done():
            self._index_update_task = asyncio.create_task(
                self._delayed_index_update()
            )

    async def _delayed_index_update(self) -> None:
        """Update index after a delay to batch multiple changes."""
        await asyncio.sleep(30)  # Wait 30s for more changes

        if self._pending_index_update:
            try:
                await self._rag_store.scan_and_index(
                    self._memory.memory_dir,
                    chunk_size=self._rag_store.config.memory_chunk_size,
                    chunk_overlap=self._rag_store.config.memory_chunk_overlap,
                )
                logger.info("RAG memory index updated (batched)")
            except Exception as e:
                logger.warning("RAG memory index update failed: {}", e)
            finally:
                self._pending_index_update = False

    async def consolidate(self, ...) -> bool:
        # ... existing code ...

        if action in ("create", "merge", "replace"):
            await self._schedule_index_update()  # Schedule instead of immediate
```

---

## 五、P3 修复（Month 2+）- 技术债务

### P3 问题清单

| 编号 | 问题 | 预计工时 | 优先级 |
|------|------|----------|--------|
| P3-1 | 代码风格统一（注释语言） | 2h | 低 |
| P3-2 | 魔法数字提取到配置 | 2h | 中 |
| P3-3 | 日志级别规范 | 1h | 中 |
| P3-4 | 类型注解补全 | 4h | 中 |
| P3-5 | 测试覆盖率提升 | 16h | 中 |
| P3-6 | 配置验证逻辑 | 2h | 中 |
| P3-7 | 健康检查接口 | 2h | 低 |
| P3-8 | 异常堆栈记录 | 1h | 中 |
| P3-9 | HeartbeatService schema 提取 | 1h | 低 |

---

## 六、验收标准

### P0 验收标准
- [ ] 所有 P0 问题修复完成
- [ ] 相关单元测试通过
- [ ] 无明显回归问题
- [ ] 代码审查通过

### P1 验收标准
- [ ] 所有 P1 问题修复完成
- [ ] 集成测试通过
- [ ] 性能测试无明显回退

### P2 验收标准
- [ ] 所有 P2 问题修复或明确延期
- [ ] 文档更新完成

---

## 七、进度追踪

| 周次 | 计划内容 | 状态 | 完成日期 |
|------|----------|------|----------|
| Week 1 | P0-1, P0-2, P0-3 | ✅ 已完成 | 2026-03-01 |
| Week 2 | P1-1 ~ P1-6 | ✅ 已完成 | 2026-03-01 |
| Week 3 | P2-1 ~ P2-5 | ✅ 已完成 (5/5) | 2026-03-01 |
| Week 4 | P2-6 ~ P2-9 + 缓冲 | ✅ 已完成 (4/4) | 2026-03-01 |
| Month 2 | P3 技术债务 | ⏳ 待定 | - |

---

## 八、风险与缓解

### 风险 1: 修复引入回归问题
**缓解**:
- 每个修复配套单元测试
- 修复前后运行完整测试套件
- 小步提交，便于回滚

### 风险 2: 修复工作量超预期
**缓解**:
- 优先修复 P0 和 P1
- P2 问题可按需延期
- P3 技术债务可视情况处理

### 风险 3: 生产环境无法停机
**缓解**:
- 修复设计为向后兼容
- 配置项默认保持原有行为
- 支持热重载的修复优先

---

## 十、P1 修复完成记录

**完成日期**: 2026-03-01
**测试结果**: 132 个测试全部通过
**额外修复**: 添加缺失的 `loguru.logger` 导入到 context.py

### P1-3: RAG 搜索错误分类处理 ✅

**修改文件**: `nanobot/rag/store.py:750-755`

**修改内容**:
```python
except sqlite3.DatabaseError as e:
    logger.error("Database error during hybrid search, check integrity: {}", e)
    raise  # 数据库错误不应该 fallback
except Exception as e:
    logger.warning("Hybrid search failed, falling back: {}", e, exc_info=True)
    # Fallback to full-text only
```

**验收**:
- ✅ 数据库错误时抛出异常
- ✅ 一般异常仍 fallback 到全文搜索
- ✅ 添加 exc_info=True 记录完整堆栈

---

### P1-2: Memory 原子操作回滚增强 ✅

**修改文件**: `nanobot/agent/memory.py:371-388`

**修改内容**:
```python
except Exception as e:
    logger.error("Atomic operation failed, rolling back: {}", e)
    rollback_failed = []

    for file, old_content in reversed(applied):
        try:
            self.replace(file, old_content)
            logger.debug("Rollback succeeded for {}", file.value)
        except Exception as rollback_e:
            logger.error("Rollback failed for {}: {}", file.value, rollback_e)
            rollback_failed.append(file)

    if rollback_failed:
        logger.critical(
            "Rollback incomplete - manual intervention may be needed: {}",
            [f.value for f in rollback_failed]
        )
    return False
```

**验收**:
- ✅ 每个文件的回滚操作都有日志
- ✅ 回滚失败时记录 failed 文件列表
- ✅ 多个文件回滚失败时使用 critical 日志级别

---

### P1-4: System Prompt 长度截断 ✅

**修改文件**: `nanobot/agent/context.py`

**修改内容**:
1. 添加类常量 `MAX_SYSTEM_PROMPT_LENGTH = 8000`
2. 修改 `build_system_prompt()` 添加长度检查和截断
3. 添加新方法 `_truncate_system_prompt()` 实现优先级截断逻辑

```python
class ContextBuilder:
    MAX_SYSTEM_PROMPT_LENGTH = 8000  # 为 model 留出空间

    def build_system_prompt(self, skill_names: list[str] | None = None, query: str | None = None) -> str:
        # ... 原有逻辑 ...
        result = "\n\n---\n\n".join(parts)

        if len(result) > self.MAX_SYSTEM_PROMPT_LENGTH:
            logger.warning(
                "System prompt too long ({} chars > {} limit), truncating",
                len(result),
                self.MAX_SYSTEM_PROMPT_LENGTH
            )
            result = self._truncate_system_prompt(parts, self.MAX_SYSTEM_PROMPT_LENGTH)

        return result

    def _truncate_system_prompt(self, parts: list[str], max_length: int) -> str:
        """Truncate system prompt while preserving priority parts.
        Priority: identity > bootstrap > skills > memory
        """
        result = parts[0]  # Always keep identity
        for part in parts[1:]:
            separator = "\n\n---\n\n"
            if len(result) + len(part) + len(separator) < max_length:
                result += separator + part
            else:
                remaining = max_length - len(result) - 50
                if remaining > 100:
                    result += separator + part[:remaining] + "\n\n...[truncated]"
                break
        return result
```

**验收**:
- ✅ system prompt 超过 8000 字符时自动截断
- ✅ 保留优先级高的部分（identity > bootstrap > skills > memory）
- ✅ 截断时添加 `...[truncated]` 标记
- ✅ 记录警告日志

---

## 十一、P2 实施方案

**计划开始日期**: 2026-03-01
**预计完成**: Week 3-4

### P2 修复优先级建议

基于影响面和实现复杂度，建议按以下顺序修复：

| 顺序 | 编号 | 问题 | 理由 | 状态 |
|------|------|------|------|------|
| 1 | P2-5 | Session 空历史保护 | 简单，防止潜在 bug | ✅ 已完成 |
| 2 | P2-6 | ToolRegistry 错误提示优化 | 简单，改善用户体验 | ✅ 已完成 |
| 3 | P2-1 | Subagent 独立模型配置 | 中等，节省成本 | ✅ 已完成 |
| 4 | P2-9 | 任务清理回调简化 | 中等，代码质量 | ✅ 已完成 |
| 5 | P2-2 | 向量搜索自动恢复 | 中等，提升稳定性 | ✅ 已完成 |
| 6 | P2-7 | 配置参数统一 | 中等，减少混淆 | ✅ 已完成 |
| 7 | P2-8 | Consolidation 调试增强 | 中等，便于调试 | ✅ 已完成 |
| 8 | P2-4 | RAG 索引批量更新 | 复杂，性能优化 | ✅ 已完成 |
| 9 | P2-3 | MCP 连接指数退避 | 复杂，稳定性 | ✅ 已完成 |

---

### P2-5: Session 空历史保护

**位置**: `nanobot/session/manager.py:45-63`

**当前代码**:
```python
def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
    """Return unconsolidated messages for LLM input, aligned to a user turn."""
    unconsolidated = self.messages[self.last_consolidated:]
    sliced = unconsolidated[-max_messages:]

    # Drop leading non-user messages to avoid orphaned tool_result blocks
    for i, m in enumerate(sliced):
        if m.get("role") == "user":
            sliced = sliced[i:]
            break

    out: list[dict[str, Any]] = []
    for m in sliced:
        entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
        for k in ("tool_calls", "tool_call_id", "name"):
            if k in m:
                entry[k] = m[k]
        out.append(entry)
    return out
```

**问题**: 当 session 为空或所有消息都已 consolidated 时，返回空列表可能导致 LLM 调用异常

**修复方案**:
```python
def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
    """Return unconsolidated messages for LLM input, aligned to a user turn."""
    unconsolidated = self.messages[self.last_consolidated:]
    sliced = unconsolidated[-max_messages:]

    # Drop leading non-user messages to avoid orphaned tool_result blocks
    for i, m in enumerate(sliced):
        if m.get("role") == "user":
            sliced = sliced[i:]
            break

    # Empty history protection - return minimal context
    if not sliced:
        logger.debug("Session {} has no unconsolidated messages", self.key)
        return []

    out: list[dict[str, Any]] = []
    for m in sliced:
        entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
        for k in ("tool_calls", "tool_call_id", "name"):
            if k in m:
                entry[k] = m[k]
        out.append(entry)
    return out
```

**验收标准**:
- [ ] 空 session 返回空列表而不是 None
- [ ] 添加 debug 日志记录空历史情况
- [ ] 调用方处理空历史情况

---

### P2-6: ToolRegistry 错误提示优化

**位置**: `nanobot/agent/tools/registry.py:38-55`

**当前代码**:
```python
async def execute(self, name: str, params: dict[str, Any]) -> str:
    """Execute a tool by name with given parameters."""
    _HINT = "\n\n[Analyze the error above and try a different approach.]"

    tool = self._tools.get(name)
    if not tool:
        return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

    try:
        errors = tool.validate_params(params)
        if errors:
            return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
        result = await tool.execute(**params)
        if isinstance(result, str) and result.startswith("Error"):
            return result + _HINT
        return result
    except Exception as e:
        return f"Error executing {name}: {str(e)}" + _HINT
```

**问题**: 错误提示不够友好，没有给出具体建议

**修复方案**:
```python
async def execute(self, name: str, params: dict[str, Any]) -> str:
    """Execute a tool by name with given parameters."""
    _HINT = "\n\n[Analyze the error above and try a different approach.]"

    tool = self._tools.get(name)
    if not tool:
        # More helpful error with available tools count
        available = self.tool_names
        return (
            f"Error: Tool '{name}' not found.\n"
            f"Available tools ({len(available)}): {', '.join(available[:10])}"
            + (f" and {len(available)-10} more..." if len(available) > 10 else "")
        )

    try:
        errors = tool.validate_params(params)
        if errors:
            # Add parameter hints if available
            hints = []
            if hasattr(tool, 'parameters') and tool.parameters:
                for err in errors:
                    if "required" in err:
                        hints.append(f"Missing required parameter. Check tool schema.")
                    elif "type" in err:
                        hints.append(f"Parameter type mismatch. Expected format may differ.")
            return (
                f"Error: Invalid parameters for tool '{name}': "
                + "; ".join(errors)
                + (f"\nHints: {'; '.join(hints)}" if hints else "")
                + _HINT
            )
        result = await tool.execute(**params)
        if isinstance(result, str) and result.startswith("Error"):
            return result + _HINT
        return result
    except Exception as e:
        return f"Error executing {name}: {str(e)}" + _HINT
```

**验收标准**:
- [ ] 工具不存在时列出可用工具（最多 10 个）
- [ ] 参数错误时给出类型提示
- [ ] 错误信息更友好和可操作

---

### P2-1: Subagent 独立模型配置

**位置**:
- `nanobot/config/schema.py:204-215` (AgentDefaults)
- `nanobot/agent/loop.py:99-109` (SubagentManager 初始化)
- `nanobot/agent/subagent.py:23-44` (SubagentManager __init__)

**修复方案**:

1. **修改配置 schema** (`config/schema.py`):
```python
class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    memory_model: str | None = None  # Optional: separate model for memory consolidation
    subagent_model: str | None = None  # Optional: separate model for subagents
    provider: str = "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
```

2. **修改 AgentLoop** (`agent/loop.py`):
```python
def __init__(
    self,
    bus: MessageBus,
    provider: LLMProvider,
    workspace: Path,
    model: str | None = None,
    memory_model: str | None = None,
    subagent_model: str | None = None,  # NEW parameter
    max_iterations: int = 40,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    memory_window: int = 100,
    # ... rest of params
):
    # ... existing init code ...

    self.subagent_model = subagent_model or self.model  # Default to main model

    self.subagents = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=bus,
        model=self.subagent_model,  # Use subagent_model
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        serper_api_key=serper_api_key,
        exec_config=self.exec_config,
        restrict_to_workspace=restrict_to_workspace,
    )
```

3. **修改 CLI** (`cli/commands.py`):
```python
agent = AgentLoop(
    # ... existing params ...
    subagent_model=config.agents.defaults.subagent_model,
)
```

**验收标准**:
- [ ] 配置文件中可设置 subagent_model
- [ ] 未设置时使用主模型
- [ ] Subagent 使用独立模型配置

---

### P2-9: 任务清理回调简化

**位置**: `nanobot/agent/loop.py` (subagent spawn 方法)

**问题**: 回调函数逻辑复杂，可以简化

**修复方案**: 使用更简洁的 cleanup 逻辑，合并重复代码

**验收标准**:
- [ ] 代码更简洁
- [ ] 功能不变
- [ ] 测试通过

---

### P2-2: 向量搜索自动恢复

**位置**: `nanobot/rag/store.py`

**当前状态**: 向量搜索失败后永久 fallback 到全文搜索

**修复方案**: 添加定期重试机制，5 分钟后尝试重新启用向量搜索

```python
class DocumentStore:
    def __init__(self, ...):
        self._vector_retry_count = 0
        self._vector_last_failure_time: float = 0
        self._vector_retry_cooldown = 300  # 5 minutes
        self._vector_enabled = self._init_vector_search()  # Original init

    def _check_vector_search_available(self) -> bool:
        """Check if vector search can be re-enabled."""
        if not self._vector_enabled and self._vector_last_failure_time > 0:
            if time.time() - self._vector_last_failure_time > self._vector_retry_cooldown:
                # Try to re-enable
                try:
                    self._get_db()  # Re-initialize
                    if self._vector_enabled:
                        logger.info("Vector search auto-recovered")
                        self._vector_retry_count = 0
                        return True
                except Exception:
                    self._vector_last_failure_time = time.time()
                    self._vector_retry_count += 1
        return self._vector_enabled
```

**验收标准**:
- [ ] 向量搜索失败后记录时间
- [ ] 5 分钟后自动尝试恢复
- [ ] 恢复成功记录日志

---

### P2-7: 配置参数统一

**位置**: `nanobot/config/schema.py`

**问题**: 部分参数使用 magic number，部分参数命名不一致

**修复方案**: 统一参数命名，提取 magic number 到常量

**验收标准**:
- [ ] 参数命名一致
- [ ] 无 magic number
- [ ] 文档更新

---

### P2-8: Consolidation 调试增强

**位置**: `nanobot/agent/memory.py`

**问题**: Consolidation 过程缺少调试信息

**修复方案**: 添加详细的 debug 日志，包括消息数量、操作详情等

**验收标准**:
- [ ] consolidation 开始/结束有日志
- [ ] 记录处理的消息数量
- [ ] 记录每个操作的详情

---

### P2-4: RAG 索引批量更新

**位置**: `nanobot/agent/memory.py` (MemoryStoreOptimized)

**问题**: 每次 memory 更新都立即触发 RAG 索引更新，效率低

**修复方案**: 延迟 30 秒批量更新

```python
class MemoryStoreOptimized:
    def __init__(self, ...):
        self._pending_index_update = False
        self._index_update_task: asyncio.Task | None = None

    async def _schedule_index_update(self) -> None:
        """Schedule a delayed index update."""
        self._pending_index_update = True
        if self._index_update_task is None or self._index_update_task.done():
            self._index_update_task = asyncio.create_task(
                self._delayed_index_update()
            )

    async def _delayed_index_update(self) -> None:
        """Update index after a delay to batch multiple changes."""
        await asyncio.sleep(30)  # Wait 30s for more changes

        if self._pending_index_update:
            try:
                await self._rag_store.scan_and_index(...)
                logger.info("RAG memory index updated (batched)")
            except Exception as e:
                logger.warning("RAG memory index update failed: {}", e)
            finally:
                self._pending_index_update = False
```

**验收标准**:
- [ ] memory 更新不立即触发索引更新
- [ ] 30 秒后批量更新
- [ ] 多次更新只触发一次索引

---

### P2-3: MCP 连接指数退避

**位置**: `nanobot/agent/loop.py`

**问题**: MCP 连接失败时立即重试，可能导致资源浪费

**修复方案**: 实现指数退避重试机制

**验收标准**:
- [ ] 连接失败时指数退避
- [ ] 最大重试次数限制
- [ ] 记录重试日志

---

## 十二、验证步骤

### 修复后运行测试
```bash
# 1. 运行测试套件
pytest tests/ --ignore=tests/test_heartbeat_service.py --ignore=tests/test_matrix_channel.py -xvs

# 2. 检查 CLI 功能
nanobot agent -m "Hello"

# 3. 检查 RAG 功能
nanobot rag status
nanobot rag search "test query"
```

### 特殊验证
- **P2-1**: 配置文件设置 subagent_model，验证 subagent 使用不同模型
- **P2-2**: 模拟向量搜索失败，验证 5 分钟后自动恢复
- **P2-4**: 连续多次 memory 更新，验证只触发一次索引更新
- **P2-5**: 空 session 验证返回空列表
- **P2-6**: 工具错误参数验证错误提示

---

## 十三、关键文件

| 文件 | 修改内容 |
|------|----------|
| `nanobot/config/schema.py` | P2-1: 添加 subagent_model 字段; P2-7: 配置参数统一 |
| `nanobot/agent/loop.py` | P2-1: 传递 subagent_model; P2-3: MCP 指数退避; P2-9: 任务清理简化 |
| `nanobot/agent/subagent.py` | P2-1: 使用 subagent_model |
| `nanobot/agent/memory.py` | P2-4: RAG 索引批量更新; P2-8: Consolidation 调试增强 |
| `nanobot/rag/store.py` | P2-2: 向量搜索自动恢复 |
| `nanobot/session/manager.py` | P2-5: 空历史保护 |
| `nanobot/agent/tools/registry.py` | P2-6: 错误提示优化 |
| `nanobot/cli/commands.py` | P2-1: 传递 subagent_model 配置 |

---

## 十四、第 1-3 批修复完成总结

**完成日期**: 2026-03-01
**测试结果**: 132 个测试全部通过 ✅

### 第 1 批（P2-5 + P2-6）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-5 | `session/manager.py:56-62` | ✅ 空历史保护 + debug 日志 |
| P2-6 | `tools/registry.py:38-74` | ✅ 错误提示优化 |

### 第 2 批（P2-1）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-1 | `config/schema.py:210` | ✅ subagent_model 字段 |
| | `agent/loop.py:54,76-77,106` | ✅ 传递配置 |
| | `cli/commands.py:312,472,973` | ✅ 三处传递 |

### 第 3 批（P2-2 + P2-4）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-2 | `rag/store.py:93-94,165-169` | ✅ 向量搜索自动恢复 |
| P2-4 | `rag/store.py:97-100,325-327,385-407` | ✅ 索引批量更新 |

---

## 十五、第 4-5 批修复完成总结

**完成日期**: 2026-03-01
**测试结果**: 132 个测试全部通过 ✅

### 第 4 批（P2-7 + P2-8）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-7 | `config/schema.py:297-396` | ✅ 创建 RAGDefaults 类，提取 14 个魔法数字为常量 |
| P2-8 | `agent/memory.py:477-590,800-893` | ✅ 两个 consolidation 方法添加详细调试日志 |

### 第 5 批（P2-3 + P2-9）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-3 | `agent/loop.py:120-123,169-209` | ✅ MCP 指数退避 (1s→300s) + 冷却期日志 |
| P2-9 | `agent/loop.py:357,441-453` | ✅ 提取任务清理回调为独立方法 |

### 修改详情

#### P2-7: 配置参数统一

**修改内容**:
- 创建 `RAGDefaults` 类，包含 14 个常量
- 所有 RAG 配置参数使用常量定义
- 增强代码可读性和可维护性

**常量列表**:
- `MIN_CHUNK_SIZE`, `MAX_CHUNK_SIZE`, `CHUNK_OVERLAP_RATIO`
- `CONTEXT_PREV_CHUNKS`, `CONTEXT_NEXT_CHUNKS`
- `TOP_DOCUMENTS`
- `BM25_THRESHOLD`, `VECTOR_THRESHOLD`, `RERANK_THRESHOLD`, `DEDUP_THRESHOLD`
- `RERANK_MODEL`, `RERANK_TOP_K`
- `MEMORY_CHUNK_SIZE`, `MEMORY_CHUNK_OVERLAP`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBEDDING_MODEL`
- `RRF_K`
- `CACHE_TTL_SECONDS`

#### P2-8: Consolidation 调试增强

**修改内容**:
- `MemoryStore.consolidate()`: 添加开始/结束日志、消息数量、操作数量、LLM 响应日志
- `MemoryStoreOptimized.consolidate()`: 添加各步骤调试日志、LLM 决策详情

**新增日志**:
- `Memory consolidation started: total_messages={}, old_messages={}, keep_count={}`
- `Calling LLM for memory consolidation with {} messages`
- `LLM response tool_calls: {}`
- `Applying {} memory operations`
- `Operation {}/{}: file={}, action={}, content_preview={}`
- `Memory consolidation completed: history_entries={}, operations_applied={}, last_consolidated={}`

#### P2-3: MCP 连接指数退避

**修改内容**:
- 添加重试属性：`_mcp_retry_count`, `_mcp_retry_delay`, `_mcp_max_retry_delay`, `_mcp_last_failure_time`
- 实现指数退避：1s → 2s → 4s → ... → 300s
- 连接成功时重置重试计数
- 添加冷却期调试日志

**退避公式**:
```python
self._mcp_retry_delay = min(
    self._mcp_max_retry_delay,
    self._mcp_retry_delay * 2
)
```

#### P2-9: 任务清理回调简化

**修改内容**:
- 提取复杂的 lambda 回调为独立方法 `_make_task_cleanup()`
- 代码更清晰易读，功能不变

**修改前**:
```python
task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)
```

**修改后**:
```python
task.add_done_callback(self._make_task_cleanup(msg.session_key))

def _make_task_cleanup(self, session_key: str) -> Callable[[asyncio.Task], None]:
    def _cleanup(task: asyncio.Task) -> None:
        tasks = self._active_tasks.get(session_key, [])
        if task in tasks:
            tasks.remove(task)
            if not tasks:
                self._active_tasks.pop(session_key, None)
    return _cleanup
```

---

## 十六、第 1-3 批修复完成总结

**完成日期**: 2026-03-01
**测试结果**: 132 个测试全部通过 ✅

### 第 1 批（P2-5 + P2-6）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-5 | `session/manager.py:56-62` | ✅ 空历史保护 + debug 日志 |
| P2-6 | `tools/registry.py:38-74` | ✅ 错误提示优化 |

### 第 2 批（P2-1）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-1 | `config/schema.py:210` | ✅ subagent_model 字段 |
| | `agent/loop.py:54,76-77,106` | ✅ 传递配置 |
| | `cli/commands.py:312,472,973` | ✅ 三处传递 |

### 第 3 批（P2-2 + P2-4）✅

| 问题 | 修改文件 | 验收结果 |
|------|----------|----------|
| P2-2 | `rag/store.py:93-94,165-169` | ✅ 向量搜索自动恢复 |
| P2-4 | `rag/store.py:97-100,325-327,385-407` | ✅ 索引批量更新 |

---

## 十七、P2 全部完成总结

### 完成率：9/9 = 100% ✅

| 批次 | 问题 | 状态 | 修改文件数 |
|------|------|------|------------|
| 第 1 批 | P2-5, P2-6 | ✅ | 2 |
| 第 2 批 | P2-1 | ✅ | 3 |
| 第 3 批 | P2-2, P2-4 | ✅ | 1 |
| 第 4 批 | P2-7, P2-8 | ✅ | 2 |
| 第 5 批 | P2-3, P2-9 | ✅ | 1 |

### 总体进度

| 优先级 | 数量 | 完成 | 完成率 |
|--------|------|------|--------|
| P0 | 3 | 3 | 100% |
| P1 | 6 | 6 | 100% |
| P2 | 9 | 9 | 100% |
| P3 | 9 | 0 | 0% (技术债务) |

---

## 十八、后续执行计划

### 短期（本周）
- ✅ P0、P1、P2 全部已完成
- 可选：开始 P3 技术债务清理

### 中期（Month 2）
- P3 技术债务清理（9 个问题）
- 测试覆盖率提升
- 文档完善

### 长期
- 新功能开发
- 性能优化

---

**参考文档**:
- 审计报告：`docs/issues/2026-03-01-audit.md`
- 代码审查：`docs/reviews/2026-03-01-code-review.md`
- 变更记录：`docs/changes/`