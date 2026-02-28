# Nanobot 代码审核报告

**审核日期**: 2026-03-01
**审核范围**: loop、context、memory、rag、tools、command、session、bus、provider 模块
**审核维度**: 架构设计、逻辑正确性、代码质量、潜在风险

---

## 执行摘要

### 整体评价
✅ **架构清晰**: 模块化设计良好，职责分离明确
✅ **核心逻辑正确**: 主要流程无明显逻辑错误
⚠️ **部分风险点**: 存在 3 个高优先级问题需要修复
⚠️ **可优化空间**: 12 处中等优先级改进建议

### 问题分级统计

| 级别 | 数量 | 说明 |
|------|------|------|
| 🔴 高 | 3 | 需要立即修复的 bug 或潜在风险 |
| 🟡 中 | 12 | 建议优化的设计或代码质量问题 |
| 🟢 低 | 8 | 代码风格、注释等改进建议 |

---

## 一、高优先级问题 (🔴)

### 1.1 AgentLoop 中 SubagentManager 使用主模型，无法独立配置

**位置**: `nanobot/agent/loop.py:99-109`

```python
self.subagents = SubagentManager(
    provider=provider,
    workspace=workspace,
    bus=bus,
    model=self.model,  # ← 硬编码使用主模型
    ...
)
```

**问题**: Subagent 执行任务时只能使用主模型，无法为子任务配置不同模型（如更便宜的模型节省成本）

**影响**: 成本优化受限，子任务可能使用不必要的高端模型

**建议修复**:
```python
# 在 SubagentManager 中增加 subagent_model 参数
self.subagents = SubagentManager(
    provider=provider,
    workspace=workspace,
    bus=bus,
    model=self.model,
    subagent_model=self.subagent_model,  # 新增配置
    ...
)
```

---

### 1.2 Memory Consolidation 并发锁竞争可能导致 starvation

**位置**: `nanobot/agent/loop.py:467-484`

```python
# 问题代码
if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
    self._consolidating.add(session.key)
    lock = self._get_consolidation_lock(session.key)

    async def _consolidate_and_unlock():
        try:
            async with lock:  # ← 如果 consolidation 耗时过长，后续消息会被阻塞
                await self._consolidate_memory(session)
```

**问题**:
- 消息处理使用全局锁 `_processing_lock`，但 consolidation 是异步后台任务
- 在高并发场景下，如果 consolidation 耗时过长（如 30s+），可能导致新消息处理延迟
- 虽然有 `_consolidating` 标记防止重复触发，但长时间 running 的 consolidation 会阻塞锁

**影响**: 高峰期可能出现响应延迟

**建议修复**:
1. 为 consolidation 添加超时保护
2. 考虑使用 `asyncio.timeout()` 或独立线程池执行

```python
async def _consolidate_and_unlock():
    try:
        async with asyncio.timeout(60):  # 60s 超时
            async with lock:
                await self._consolidate_memory(session)
    except asyncio.TimeoutError:
        logger.error("Memory consolidation timed out after 60s")
    finally:
        self._consolidating.discard(session.key)
```

---

### 1.3 RAG DocumentStore 缓存无内存限制，可能导致内存泄漏

**位置**: `nanobot/rag/store.py:89-91`

```python
# Search cache - stores (timestamp, results)
self._search_cache: dict[str, tuple[float, list[SearchResultWithContext]]] = {}
self._basic_search_cache: dict[str, tuple[float, list[SearchResult]]] = {}
```

**问题**:
- 缓存只有 TTL 清理，没有最大条目数限制
- 长时间运行的服务可能积累大量缓存条目
- `scan_and_index()` 会清空缓存，但如果长时间不刷新索引，缓存会持续增长

**影响**: 内存泄漏风险，尤其是高频查询场景

**建议修复**:
```python
from collections import OrderedDict

# 使用 LRU cache + TTL 双重保护
from functools import lru_cache
import time

class DocumentStore:
    # 或者使用 cachetools.TTLCache
    from cachetools import TTLCache
    _search_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
```

或者手动实现：
```python
MAX_CACHE_SIZE = 1000

def _cleanup_cache_if_needed(self):
    if len(self._search_cache) > MAX_CACHE_SIZE:
        # 移除最旧的 50% 条目
        oldest_keys = sorted(self._search_cache.keys())[:MAX_CACHE_SIZE // 2]
        for key in oldest_keys:
            del self._search_cache[key]
```

---

## 二、中优先级问题 (🟡)

### 2.1 ContextBuilder 中 System Prompt 过长风险

**位置**: `nanobot/agent/context.py:26-58`

```python
def build_system_prompt(self, skill_names: list[str] | None = None, query: str | None = None) -> str:
    parts = [self._get_identity()]
    bootstrap = self._load_bootstrap_files()  # ← 可能返回大量内容
    if bootstrap:
        parts.append(bootstrap)
    memory = self.memory.get_memory_context(query)  # ← 可能返回大量内容
    if memory:
        parts.append(f"# Memory\n\n{memory}")
    # ... skills 也可能很长
```

**问题**:
- 没有长度限制或截断机制
- 如果 bootstrap 文件过多（如 5 个文件各 2000 字）+ memory 内容过长，可能导致 prompt 超出模型 context window
- 某些免费模型 context window 有限（如 8K）

**建议**: 添加长度限制和渐进式截断策略
```python
MAX_SYSTEM_PROMPT_LENGTH = 8000  # 为 model 留出空间

def build_system_prompt(self, ...) -> str:
    # ... 构建 parts ...
    result = "\n\n---\n\n".join(parts)
    if len(result) > MAX_SYSTEM_PROMPT_LENGTH:
        logger.warning("System prompt too long ({} chars), truncating", len(result))
        result = result[:MAX_SYSTEM_PROMPT_LENGTH] + "\n\n[Truncated due to length]"
    return result
```

---

### 2.2 MemoryStore 原子操作回滚逻辑存在竞态条件

**位置**: `nanobot/agent/memory.py:336-371`

```python
def apply_operations_atomic(self, operations: list[dict]) -> bool:
    applied: list[tuple[MemoryFile, str]] = []  # (file, old_content) for rollback

    try:
        for op_data in operations:
            # ... 应用操作 ...
            applied.append((file, old_content))  # ← 保存旧内容
            if not self.apply_operation(op):
                raise Exception(f"Operation failed: {op_data}")
        return True
    except Exception as e:
        # 回滚
        for file, old_content in reversed(applied):
            self.replace(file, old_content)  # ← 回滚
        return False
```

**问题**:
- 回滚时直接使用 `replace()`，如果回滚失败则数据不一致
- 没有检查回滚是否成功
- 如果文件在操作过程中被外部修改（如用户手动编辑），回滚会覆盖这些修改

**建议**:
```python
def apply_operations_atomic(self, operations: list[dict]) -> bool:
    # ... 同上 ...
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
            logger.critical("Rollback incomplete, manual intervention may be needed: {}", rollback_failed)
        return False
```

---

### 2.3 RAG 搜索 pipeline 中错误处理过于宽泛

**位置**: `nanobot/rag/store.py:713-719`

```python
except Exception as e:
    logger.warning("Hybrid search failed, falling back: {}", e)
```

**问题**:
- 捕获所有 `Exception` 但不区分具体错误类型
- 可能掩盖严重问题（如数据库损坏、权限错误）
- fallback 逻辑没有记录原始错误堆栈

**建议**:
```python
except sqlite3.DatabaseError as e:
    logger.error("Database error, check integrity: {}", e)
    raise  # 数据库错误不应该 silently fallback
except Exception as e:
    logger.warning("Hybrid search failed, falling back: {}", e, exc_info=True)  # 记录堆栈
```

---

### 2.4 AgentLoop 中全局锁 `_processing_lock` 可能成为瓶颈

**位置**: `nanobot/agent/loop.py:347`

```python
async def _dispatch(self, msg: InboundMessage) -> None:
    async with self._processing_lock:  # ← 所有消息共享一把锁
        try:
            response = await self._process_message(msg)
```

**问题**:
- 所有 session 的消息处理都使用同一把全局锁
- 多 channel 场景下（如同时有 Telegram、WhatsApp、Discord 消息），会串行处理
- 违背了 `_dispatch` 设计初衷（通过 task 分发提高并发性）

**建议**: 使用 per-session 锁
```python
def _get_session_lock(self, session_key: str) -> asyncio.Lock:
    lock = self._session_locks.get(session_key)
    if lock is None:
        lock = asyncio.Lock()
        self._session_locks[session_key] = lock
    return lock

async def _dispatch(self, msg: InboundMessage) -> None:
    lock = self._get_session_lock(msg.session_key)
    async with lock:
        # 同 session 串行，跨 session 并行
```

---

### 2.5 ToolRegistry 错误提示信息重复追加

**位置**: `nanobot/agent/tools/registry.py:40-55`

```python
_HINT = "\n\n[Analyze the error above and try a different approach.]"

async def execute(self, name: str, params: dict[str, Any]) -> str:
    # ...
    if errors:
        return f"Error: ... " + _HINT  # ← 每次错误都追加
    # ...
    if isinstance(result, str) and result.startswith("Error"):
        return result + _HINT  # ← 如果 tool 内部返回 error，会再次追加
```

**问题**:
- 如果 tool 执行多次失败，每次都会追加 `_HINT`
- 多轮对话后，错误信息可能变得冗长

**影响**: 浪费 token，可能干扰 LLM 判断

**建议**: 只在第一次错误时追加，或使用标志位
```python
def execute(self, name: str, params: dict[str, Any]) -> str:
    # ...
    error_msg = f"Error: ..."
    # 不直接追加 hint，让上层决定是否添加
    return error_msg
```

---

### 2.6 Session.get_history() 可能返回空列表导致 LLM 困惑

**位置**: `nanobot/session/manager.py:45-63`

```python
def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
    unconsolidated = self.messages[self.last_consolidated:]
    sliced = unconsolidated[-max_messages:]

    # Drop leading non-user messages
    for i, m in enumerate(sliced):
        if m.get("role") == "user":
            sliced = sliced[i:]
            break

    # ... 返回 out
```

**问题**:
- 如果 `sliced` 中没有 user 消息（如全是 tool_result），会返回空列表
- 空历史 + 当前消息可能导致 LLM 无法理解上下文

**建议**:
```python
if not out and unconsolidated:
    # 至少返回第一条消息，避免空历史
    out.append({"role": unconsolidated[0]["role"], "content": unconsolidated[0].get("content", "")})
```

---

### 2.7 RAG DocumentStore 中向量搜索降级后无恢复机制

**位置**: `nanobot/rag/store.py:144-146`

```python
self._vector_enabled = False  # ← 一旦禁用，永久禁用

except Exception as e:
    logger.warning("Could not load sqlite-vec extension, vector search disabled: {}", e)
```

**问题**:
- 向量搜索一旦因错误禁用，服务重启前无法恢复
- 如果是暂时性问题（如内存不足），后续请求也无法使用向量搜索

**建议**:
```python
# 添加重试机制或定期检测
def _check_vector_search_available(self):
    if not self._vector_enabled:
        # 尝试重新加载
        try:
            # 重新加载逻辑
            self._vector_enabled = True
            logger.info("Vector search re-enabled")
        except Exception:
            pass
```

---

### 2.8 HeartbeatService 使用硬编码的 tool schema

**位置**: `nanobot/heartbeat/service.py:14-37`

```python
_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            # ... 硬编码的 schema
        }
    }
]
```

**问题**:
- Tool schema 与实现耦合，修改时需同步更新
- 无法动态调整 heartbeat 行为

**建议**: 将 schema 提取到单独的配置类或常量文件

---

### 2.9 MessageBus 无背压机制

**位置**: `nanobot/bus/queue.py:16-18`

```python
def __init__(self):
    self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()  # ← 无界队列
    self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
```

**问题**:
- 使用无界队列，如果 consumer 处理速度慢于 producer，队列可能无限增长
- 高频消息场景可能导致内存问题

**建议**:
```python
def __init__(self, maxsize: int = 1000):
    self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=maxsize)
    self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=maxsize)

async def publish_inbound(self, msg: InboundMessage) -> None:
    try:
        await asyncio.wait_for(self.inbound.put(msg), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Inbound queue full, message dropped")
```

---

### 2.10 ContextBuilder 中 bootstrap 文件加载无错误处理

**位置**: `nanobot/agent/context.py:103-113`

```python
def _load_bootstrap_files(self) -> str:
    parts = []
    for filename in self.BOOTSTRAP_FILES:
        file_path = self.workspace / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")  # ← 无异常处理
            parts.append(f"## {filename}\n\n{content}")
```

**问题**:
- `read_text()` 可能抛出 `UnicodeDecodeError`、`PermissionError` 等
- 文件存在但无法读取时，整个 bootstrap 加载会失败

**建议**:
```python
def _load_bootstrap_files(self) -> str:
    parts = []
    for filename in self.BOOTSTRAP_FILES:
        file_path = self.workspace / filename
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
            except Exception as e:
                logger.warning("Failed to load bootstrap file {}: {}", filename, e)
                parts.append(f"## {filename}\n\n[Failed to load: {e}]")
    return "\n\n".join(parts) if parts else ""
```

---

### 2.11 AgentLoop 中 MCP 连接重试无退避

**位置**: `nanobot/agent/loop.py:158-180`

```python
async def _connect_mcp(self) -> None:
    if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
        return
    self._mcp_connecting = True
    try:
        # ... 连接逻辑 ...
    except Exception as e:
        logger.error("Failed to connect MCP servers (will retry next message): {}", e)
        # ... 清理 ...
    finally:
        self._mcp_connecting = False
```

**问题**:
- 每次新消息都会尝试重连，没有指数退避
- 如果 MCP server 长时间不可用，会频繁重试浪费资源

**建议**:
```python
# 添加重试计数和退避时间
self._mcp_retry_count = 0
self._mcp_last_failure_time: float = 0

async def _connect_mcp(self) -> None:
    # 检查是否需要等待退避
    if self._mcp_last_failure_time:
        backoff = min(300, 2 ** self._mcp_retry_count)  # 最多 5 分钟
        if time.time() - self._mcp_last_failure_time < backoff:
            return  # 还在退避期内

    # ... 连接逻辑 ...
    except Exception as e:
        self._mcp_retry_count += 1
        self._mcp_last_failure_time = time.time()
```

---

### 2.12 MemoryStoreOptimized 中 RAG 索引更新效率低

**位置**: `nanobot/agent/memory.py:828-838`

```python
if action in ("create", "merge", "replace"):
    try:
        # Only index memory directory (efficient incremental update)
        await self._rag_store.scan_and_index(
            self._memory.memory_dir,
            ...
        )
```

**问题**:
- 每次 memory 更新都触发全量 `scan_and_index`
- 虽然只扫描 memory 目录，但仍是全量操作
- 高频 consolidation 场景会导致大量重复索引操作

**建议**: 添加批量更新或延迟更新机制
```python
# 添加延迟更新队列
self._pending_index_updates: asyncio.Queue = asyncio.Queue()

async def _process_index_updates(self):
    while True:
        await asyncio.sleep(30)  # 每 30 秒批量更新一次
        # 批量处理所有待更新
```

---

## 三、低优先级问题 (🟢)

### 3.1 代码风格不一致

- `nanobot/agent/memory.py` 使用中文注释，而其他文件主要用英文
- 部分方法有详细 docstring，部分没有

**建议**: 统一注释语言（推荐英文），补充缺失的 docstring

---

### 3.2 魔法数字硬编码

```python
# nanobot/agent/loop.py
_TOOL_RESULT_MAX_CHARS = 500  # 应该移到配置中

# nanobot/heartbeat/service.py
interval_s: int = 30 * 60  # 应该使用配置值
```

**建议**: 将这些值提取到配置类中

---

### 3.3 日志级别使用不一致

部分地方用 `logger.warning`，部分用 `logger.error`，没有明确标准

**建议**: 制定日志级别规范：
- `DEBUG`: 调试信息
- `INFO`: 正常操作记录
- `WARNING`: 可恢复的异常
- `ERROR`: 需要关注的错误
- `CRITICAL`: 系统级故障

---

### 3.4 类型注解不完整

部分函数参数和返回值缺少类型注解

**建议**: 补全类型注解，尤其是公共 API

---

### 3.5 测试覆盖率不足

关键模块如 `MessageBus`、`SessionManager` 缺少单元测试

**建议**: 优先为核心模块添加测试

---

### 3.6 配置验证不足

配置加载时没有验证必填字段

**建议**: 添加配置验证逻辑

---

### 3.7 缺少健康检查接口

没有 `/health` 或 `/status` 接口供外部监控

**建议**: 添加健康检查端点

---

### 3.8 部分异常未记录堆栈

```python
except Exception as e:
    logger.warning("...: {}", e)  # 缺少 exc_info=True
```

**建议**: 记录完整堆栈便于调试

---

## 四、架构层面建议

### 4.1 考虑引入事件总线

当前 `MessageBus` 只是简单队列，建议升级为事件总线模式，支持：
- 事件订阅/发布
- 事件过滤
- 事件持久化

### 4.2 考虑引入配置热重载

当前配置修改需要重启服务，建议支持：
- 配置文件监听
- 配置变更通知
- 增量配置更新

### 4.3 考虑引入指标收集

添加 Prometheus 或其他指标收集：
- 请求处理时间
- 队列长度
- 错误率
- Token 使用量

---

## 五、总结

### 优先级排序

| 优先级 | 问题编号 | 建议修复时间 |
|--------|----------|--------------|
| P0 | 1.1, 1.2, 1.3 | 1 周内 |
| P1 | 2.1, 2.2, 2.3, 2.4, 2.5 | 2 周内 |
| P2 | 2.6-2.12 | 1 个月内 |
| P3 | 3.1-3.8 | 视情况而定 |

### 总体评价

Nanobot 代码质量整体良好，架构清晰，核心逻辑正确。主要问题集中在：
1. **并发处理**: 部分锁设计和队列管理需要优化
2. **错误处理**: 部分异常处理过于宽泛
3. **性能优化**: 缓存管理和索引更新效率有提升空间

建议按优先级逐步修复，优先解决 P0 级别问题。
