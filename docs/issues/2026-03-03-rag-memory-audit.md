# RAG & Memory 模块问题审计报告

**审计日期**: 2026-03-03
**审计范围**: RAG 模块 (nanobot/rag/)、Memory 模块 (nanobot/agent/memory.py)
**状态**: 修复中

---

## 修复进展

| 问题 | 状态 | 修复时间 |
|------|------|----------|
| P0-1: 统一默认模型 | ✅ 已修复 | 2026-03-03 |
| P0-2: Memory 索引效率优化 | ✅ 已修复 | 2026-03-03 |
| P0-3: 修复路径匹配 | ✅ 已修复 | 2026-03-03 |
| P0-4: 修复缓存 Key | ✅ 已修复 | 2026-03-03 |
| P0-5: 去重阈值配置混乱 | ✅ 已修复 | 2026-03-03 |
| P0-6: RAG-based consolidation 流程优化 | ✅ 已修复 | 2026-03-03 |
| P0-7: 原子操作回滚可能失败 | ✅ 已修复 | 2026-03-03 |
| P0-8: 余弦相似度重复实现 | ✅ 已修复 | 2026-03-03 |
| **P0-9: 固化触发时机生硬** | **✅ 已修复** | **2026-03-03** |

---

## 执行摘要

本次审计共发现 **9 个 P0 问题**、**8 个 P1 问题**、**7 个 P2 问题**。**已修复全部 9 个 P0 问题**。

主要问题集中在：
1. **配置不一致**（默认模型、阈值混乱）
2. **架构耦合**（RAG + Memory 耦合过紧）
3. **性能问题**（Memory 索引全量扫描）
4. **代码重复**（余弦相似度重复实现）
5. **边界处理**（路径匹配、原子操作回滚）

---

## 问题详细列表

### 🔴 P0 级问题（严重，需优先修复）

| ID | 问题 | 位置 | 影响 | 风险等级 |
|----|------|------|------|----------|
| P0-1 | 默认模型不一致 | `rag.py:50`, `schema.py:331` | 可能导致 embedding 维度不匹配 | 高 |
| P0-2 | Memory 索引更新效率低 | `memory.py:758-767` | 记忆更新频繁时开销巨大 | 高 |
| P0-3 | RAG 记忆搜索路径过滤脆弱 | `memory.py:811-823` | 可能搜到 docs/ 内容，污染记忆 | 高 |
| P0-4 | 搜索缓存 Key 设计不安全 | `search.py:134` | 哈希碰撞 + top_k 混用导致缓存污染 | 中 |
| P0-5 | 去重阈值配置混乱 | `rerank.py:130`, `schema.py:317` | 语义去重效果不可预测 | 中 |
| P0-6 | RAG-based consolidation 流程复杂 | `memory.py:652-778` | 3 次 LLM 调用，耗时高，效果未验证 | 中 |
| P0-7 | 原子操作回滚可能失败 | `memory.py:388-405` | 磁盘满/权限问题导致数据不一致 | 中 |
| P0-8 | 余弦相似度重复实现 | `memory.py:242-256`, `rerank.py:133-145` | 代码重复，维护成本高 | 低 |

### 🟡 P1 级问题（中等，建议尽快修复）

| ID | 问题 | 位置 | 影响 |
|----|------|------|------|
| P1-1 | 缺少索引版本管理 | `connection.py` | Schema 变更或模型更换时无法自动迁移 |
| P1-2 | 向量距离计算理解偏差 | `search.py:276` | `1.0 - (distance / 2.0)` 不一定适用于所有模型 |
| P1-3 | `get_memory_context()` 用关键词匹配 | `memory.py:441-465` | 关键词匹配不灵活，可能误匹配 |
| P1-4 | consolidation 两种策略代码重复 | `memory.py:525-650` | 前半部分几乎一样，维护成本高 |
| P1-5 | `update_section()` 正则有边界问题 | `memory.py:268-311` | 某些边缘情况可能匹配错误 section |
| P1-6 | 缺少记忆访问统计 | `memory.py` | 无法知道哪些记忆被频繁使用 |
| P1-7 | 批量 embedding 无错误处理 | `indexer.py` | 一个失败会导致整个索引失败 |
| P1-8 | 缺少搜索质量指标 | `search.py` | 无 precision/recall 日志，难以优化 |

### 🟢 P2 级问题（低优先级，有时间再优化）

| ID | 问题 | 位置 | 说明 |
|----|------|------|------|
| P2-1 | 缺少索引元数据表 | `connection.py` | 建议添加元数据表记录 schema_version、embedding_model |
| P2-2 | 融合策略单一 | `search.py` | 只有 RRF，可以考虑添加线性插值作为备选 |
| P2-3 | 缺少搜索 explain 模式 | `search.py` | 可以添加 debug 模式返回每一步中间结果 |
| P2-4 | 记忆文件没有标签 | `indexer.py` | 可以给 memory 分块加 `is_memory=true` 标签 |
| P2-5 | 缺少冷热数据分离 | `memory.py` | 长期不用的记忆可以考虑归档 |
| P2-6 | 缺少配置验证 | `schema.py` | 可以添加 validator 检查参数合理性 |
| P2-7 | 缺少压力测试 | - | 建议添加 1000+ 文档的索引/搜索测试 |

---

## P0 问题详细分析

### P0-1: 默认模型不一致

**问题描述：**
- `SearchKnowledgeTool.__init__()` 默认使用 `"all-MiniLM-L6-v2"` (384 维)
- `RAGConfig` 默认使用 `"BAAI/bge-m3"` (1024 维)
- 如果用户混用，可能导致 embedding 维度不匹配

**影响：**
- 数据库已有 384 维向量，换成 1024 维模型会导致搜索失败
- 或者相反

**建议修复：**
```python
# 方案 1：统一默认模型（推荐）
# SearchKnowledgeTool 也用 "BAAI/bge-m3"

# 方案 2：添加兼容性检查
# 初始化时检查数据库维度与当前模型维度是否匹配
```

---

### P0-2: Memory 索引更新效率低

**问题描述：**
```python
# memory.py:758-767
await self._rag_store.scan_and_index(
    self.memory_dir,  # 每次都全量扫描 memory_dir！
    chunk_size=self._rag_store.config.memory_chunk_size,
    chunk_overlap_ratio=self._rag_store.config.memory_chunk_overlap_ratio,
)
```

**影响：**
- 每次记忆巩固后都全量扫描 memory/
- 如果记忆文件很多，每次都要检查 mtime
- 其实只需要更新变化的那一个文件

**建议修复：**
```python
# 方案：提供单文件索引接口
DocumentStore.add_single_file(path)
# 或者
DocumentIndexer.index_file(path)
```

---

### P0-3: RAG 记忆搜索路径过滤脆弱

**问题描述：**
```python
# memory.py:811-823
memory_dir_str = str(self.memory_dir).replace("\\", "/").rstrip("/")
memory_dir_prefix = memory_dir_str + "/"
for r in results:
    doc_path = r.document.path.replace("\\", "/").rstrip("/")
    if doc_path.startswith(memory_dir_prefix) or doc_path == memory_dir_str:
        memory_results.append(r.combined_content)
```

**问题：**
- 字符串匹配在跨平台时不可靠
- 如果 memory_dir 是 `/home/user/memory`，而有个文档在 `/home/user/memory2/`，会误匹配
- Windows 路径分隔符问题

**建议修复：**
```python
# 使用 Path 对象的 is_relative_to() 方法
from pathlib import Path

memory_path = Path(self.memory_dir)
doc_path = Path(r.document.path)
if memory_path in doc_path.parents or doc_path == memory_path:
    # 是 memory 文件
```

---

### P0-4: 搜索缓存 Key 设计不安全

**问题描述：**
```python
# search.py:134
cache_key = f"{hash(query)}:{top_k}"
```

**问题：**
- `hash(query)` 可能碰撞（概率低但存在）
- 基础搜索用 `(hash(query), top_k)`，高级搜索只用 `hash(query)`
- 应该用更安全的哈希方式

**建议修复：**
```python
# 方案：使用 hashlib
import hashlib

def _get_cache_key(query: str, top_k: int | None = None) -> str:
    key = query.encode("utf-8")
    if top_k is not None:
        key += f":{top_k}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()
```

---

### P0-5: 去重阈值配置混乱

**问题描述：**
- `SemanticDeduplicator.__init__()` 默认 `0.9`
- `RAGDefaults.DEDUP_THRESHOLD = 0.7`
- 配置中没有说明哪个生效

**建议修复：**
统一阈值，或者在文档中明确说明优先级。

---

### P0-6: RAG-based consolidation 流程复杂

**问题描述：**
```
_consolidate_with_rag() 流程：
1. 压缩消息 → LLM #1
2. RAG 搜索相关记忆
3. LLM #2 决定 action
4. LLM #3 生成更新内容
总共 3 次 LLM 调用！
```

**问题：**
- 耗时是 direct 策略的 3 倍
- 费用也是 3 倍
- 效果是否真的更好？没有 A/B 测试验证

**建议修复：**
- 考虑简化流程，或者做 A/B 测试
- 如果效果差异不大，建议移除 RAG-based consolidation

---

### P0-7: 原子操作回滚可能失败

**问题描述：**
```python
# memory.py:388-405
for file, old_content in reversed(applied):
    try:
        self.replace(file, old_content)
    except Exception as rollback_e:
        # 回滚也失败了怎么办？
        logger.error("Rollback failed for {}: {}", file.value, rollback_e)
        rollback_failed.append(file)
```

**问题：**
- 如果磁盘满了，回滚也会失败
- 如果权限变了，回滚也会失败
- 这时候数据处于不一致状态

**建议修复：**
```python
# 方案：用临时文件 + rename（文件系统 rename 是原子的）
def safe_replace(file: MemoryFile, content: str):
    path = self.memory_dir / file.value
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    # rename 是原子操作
    tmp_path.rename(path)
```

---

### P0-8: 余弦相似度重复实现

**问题描述：**
- `memory.py:242-256` 有一个 `_cosine_similarity()`
- `rerank.py:133-145` 有一个 `_cosine_similarity()`
- 两个实现几乎一样

**建议修复：**
提取到 `nanobot/utils/helpers.py` 中共享。

---

## 架构问题分析

### RAG + Memory 耦合分析

```
当前架构：
┌─────────────────────────────────────────────────┐
│  MemoryStore                                    │
│  - _rag_store: DocumentStore  ← 紧耦合         │
│  - consolidate()  → scan_and_index(memory_dir) │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  DocumentStore                                 │
│  - scan_and_index()  全量扫描！                 │
└─────────────────────────────────────────────────┘
```

**问题：**
- MemoryStore 持有 DocumentStore 引用，耦合过紧
- 记忆更新触发全量扫描，效率低
- 没有明确的职责边界

**建议架构：**

```
方案 1：事件驱动（推荐）
┌─────────────────────────────────────────────────┐
│  MemoryStore                                    │
│  - on_memory_changed(file)                      │
│     ↓ publish event                              │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  MessageBus                                     │
│  MemoryFileUpdatedEvent                         │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  RAG Indexer (监听事件)                         │
│  - index_single_file(file)  ← 只索引变更文件  │
└─────────────────────────────────────────────────┘

方案 2：回调函数（简单）
┌─────────────────────────────────────────────────┐
│  MemoryStore                                    │
│  - set_index_callback(callback)                 │
│  - memory 变更时调用 callback(file)            │
└─────────────────────────────────────────────────┘
```

---

## Quick Wins（1 小时内能做的）

| 任务 | 预估时间 | 说明 |
|------|----------|------|
| 统一默认模型 | 10 分钟 | 让 SearchKnowledgeTool 和 RAGConfig 用同一个默认 |
| 修复缓存 Key | 15 分钟 | 用 hashlib 替换 hash() |
| 提取余弦相似度 | 15 分钟 | 移到 nanobot/utils/helpers.py |
| 修复路径匹配 | 15 分钟 | 用 Path.is_relative_to() |
| 统一去重阈值 | 5 分钟 | 明确配置优先级 |

**总计：约 1 小时**

---

## 改进建议优先级

### Phase 1（本周）- 修复 P0 问题
- [ ] P0-1: 统一默认模型
- [ ] P0-2: Memory 索引效率优化（单文件更新）
- [ ] P0-3: 修复路径匹配
- [ ] P0-4: 修复缓存 Key
- [ ] P0-8: 提取余弦相似度

### Phase 2（下周）- 修复 P1 问题
- [ ] P1-1: 添加索引元数据表
- [ ] P1-3: 改进 get_memory_context()
- [ ] P1-4: 消除 consolidation 代码重复
- [ ] P1-5: 修复 update_section() 正则

### Phase 3（后续）- 架构优化
- [ ] 解耦 RAG + Memory（事件驱动或回调）
- [ ] A/B 测试 RAG-based consolidation
- [ ] 添加搜索 explain 模式
- [ ] 添加记忆访问统计

---

## 附录

### 相关文件清单

**RAG 模块：**
- `nanobot/rag/__init__.py`
- `nanobot/rag/store.py`
- `nanobot/rag/search.py`
- `nanobot/rag/connection.py`
- `nanobot/rag/indexer.py`
- `nanobot/rag/rerank.py`
- `nanobot/rag/parser.py`
- `nanobot/rag/embeddings.py`
- `nanobot/rag/cache.py`
- `nanobot/rag/query.py`

**Memory 模块：**
- `nanobot/agent/memory.py`

**集成：**
- `nanobot/agent/tools/rag.py`
- `nanobot/agent/loop.py`
- `nanobot/config/schema.py`

---

## 优化方案（质量优先）

### 讨论总结

基于讨论，我们确定了**质量优先**的优化方向：
- ✅ 使用 `memory_model`（次要模型），LLM 调用成本不是问题
- ✅ 记忆质量最重要，速度其次
- ✅ 保留 RAG-based consolidation 的 3 次 LLM 调用，但提升质量
- ✅ 优化触发时机，不再生硬按窗口

---

### 一、RAG-based Consolidation 优化（质量优先）

#### 改进点

| 改进 | 说明 |
|------|------|
| 1. 结构化压缩输出 | 第 1 次 LLM 输出：`{summary, search_keywords, target_file_hint}` |
| 2. 关键词搜索 | 用专门提取的 `search_keywords` + `summary` 搜索，更精准 |
| 3. 细粒度决策 | 第 3 次 LLM 也用 `save_memory` tool，支持 `update_section` |
| 4. 代码清晰化 | 步骤清晰，注释详细 |

#### 优化后流程

```
旧消息
   ↓
[1/3] LLM #1: 深度压缩 + 提取关键词
   ↓
   ├─ summary（5-10 个关键点，高质量）
   ├─ search_keywords（用于搜索的关键词，提高搜索质量）
   └─ target_file_hint（papers/profile/projects 等）
   ↓
[2/3] RAG 搜索：用 keywords + summary 搜索
   ↓
相关记忆（高质量，因为用了关键词搜索）
   ↓
[3/3] LLM #2: 基于相关记忆做精细决策
   ↓
精细决策（包括 update_section！不只 create/merge/replace/skip）
   ↓
原子应用操作
   ↓
[可选] 更新 RAG 索引（只索引修改的文件，P0-2 已修复）
```

---

### 二、触发时机优化（方案 1：混合策略）

#### 触发条件（满足任一即触发）

```
┌─────────────────────────────────────────────────────────┐
│  触发条件（任一满足）                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [1] 消息窗口（保底）                                  │
│      └─ 消息数 ≥ 50（可配置）                           │
│                                                         │
│  [2] 会话暂停                                          │
│      └─ 超过 5 分钟无新消息 + 已有 10+ 条消息         │
│                                                         │
│  [3] 重要内容检测                                      │
│      └─ 最近 5 条消息含关键词 + 已有 5+ 条消息        │
│         关键词：决定/decision、结论/conclusion、        │
│                todo/任务、计划/plan、项目/project、    │
│                架构/architecture、记住/remember         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 实现思路

```python
async def _check_consolidation_trigger(self, session: Session) -> bool:
    """检查是否需要触发记忆固化。"""
    now = time.time()
    message_count = len(session.messages) - session.last_consolidated

    # 条件 1：消息窗口（保底）
    if message_count >= self.memory_window:
        logger.debug("Consolidation trigger: message window reached")
        return True

    # 条件 2：会话暂停（超过 N 分钟无新消息）
    PAUSE_THRESHOLD = 300  # 5 分钟
    last_message_time = session.messages[-1].get("timestamp", now) if session.messages else now
    if now - last_message_time > PAUSE_THRESHOLD and message_count > 10:
        logger.debug("Consolidation trigger: session pause")
        return True

    # 条件 3：重要内容关键词
    IMPORTANT_KEYWORDS = [
        "决定", "decision", "结论", "conclusion",
        "todo", "任务", "task", "计划", "plan",
        "项目", "project", "架构", "architecture",
        "记住", "remember", "note", "笔记"
    ]
    last_messages = session.messages[-5:]  # 检查最近 5 条
    for msg in last_messages:
        content = msg.get("content", "").lower()
        if any(kw in content for kw in IMPORTANT_KEYWORDS):
            if message_count > 5:  # 至少有几条消息才固化
                logger.debug("Consolidation trigger: important content detected")
                return True

    return False
```

---

### 三、整体架构图

```
新消息到来
    ↓
检查是否触发固化？
    ↓
    ├─ 是 → 进入固化流程
    │       ↓
    │   [1/3] LLM #1：深度压缩 + 提取关键词
    │       ↓
    │       ├─ summary（5-10 关键点）
    │       ├─ search_keywords（搜索关键词）
    │       └─ target_file_hint（目标文件提示）
    │       ↓
    │   [2/3] RAG 搜索
    │       ↓
    │       用 keywords + summary 搜索
    │       ↓
    │       相关记忆（高质量）
    │       ↓
    │   [3/3] LLM #2：细粒度决策
    │       ↓
    │       用 save_memory tool
    │       支持 update_section
    │       ↓
    │   原子应用操作
    │       ↓
    │   只索引修改的文件（P0-2 已修复）
    │
    └─ 否 → 继续，不固化
```

---

### 四、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `nanobot/agent/memory.py` | 1. 更新 `COMPRESSION_PROMPT` 输出结构化<br>2. 更新 `_consolidate_with_rag()` 流程<br>3. 更新 `_decide_with_context()` 用 tool<br>4. 新增 `_check_consolidation_trigger()` 方法 |
| `nanobot/agent/loop.py` | 在 `_run_agent_loop()` 中调用新的触发检查 |

---

### 五、预期收益

| 维度 | 改进 |
|------|------|
| 记忆质量 | ✅ 更高（关键词搜索 + 细粒度决策） |
| 触发时机 | ✅ 更智能（混合策略） |
| 索引效率 | ✅ 更高（P0-2 已修复，只索引修改的文件） |
| 代码优雅度 | ✅ 更高（步骤清晰） |

---

### 六、方案对比

| 方案 | 复杂度 | 智能度 | 质量 | 推荐度 |
|------|--------|--------|------|--------|
| 当前方案（按窗口） | 低 | 低 | 中 | ⭐⭐ |
| 质量优先方案（本文档） | 中 | 高 | 高 | ⭐⭐⭐⭐⭐ |

---

## 最新进展（2026-03-03）

### ✅ 已完成：固化触发时机优化（混合策略）

**实现内容**：
- 新增 `ConsolidationTrigger` 类，独立的触发检查逻辑
- 三种触发策略（满足任一即触发）：
  1. **消息窗口（保底）**：未固化消息数 ≥ threshold（默认 100）
  2. **会话暂停**：超过 5 分钟无新消息 + 已有 10+ 条消息
  3. **重要内容检测**：最近 5 条消息含关键词 + 已有 5+ 条消息
- 支持中英文关键词：决定/decision、结论/conclusion、todo/任务、plan/计划、project/项目、architecture/架构、remember/记住、note/笔记
- 所有阈值可通过配置调整

**修改文件**：
| 文件 | 修改内容 |
|------|----------|
| `nanobot/session/manager.py` | Session 类新增 `last_consolidation_check`、`consolidation_paused` 字段；更新 load/save/clear 方法 |
| `nanobot/agent/loop.py` | 新增 `ConsolidationTrigger` 类；AgentLoop 集成触发器 |
| `nanobot/config/schema.py` | AgentDefaults 新增 4 个配置项 |
| `nanobot/cli/commands.py` | 3 处 AgentLoop 初始化传递新配置 |

**验证结果**：
- ✅ 向后兼容：旧会话正常加载
- ✅ 所有现有测试通过（29 个）
- ✅ 集成测试通过（6 个）

---

### ✅ 已完成：P0-6 RAG-based consolidation 流程优化

**实现内容**：
- **核心改进**：将 RAG-based consolidation 从 2 次 LLM 调用降为 1 次
- **新增共享方法**：
  - `_get_messages_to_consolidate()` - 共享的消息选择逻辑
  - `_process_save_memory_tool_call()` - 共享的 tool call 处理
  - `_search_related_memory_enhanced()` - 增强的 RAG 搜索（用完整消息搜索）
  - `_apply_save_memory_operations()` - 共享的操作应用逻辑
- **重构 `_consolidate_with_rag()`**：
  - 单次 LLM 调用，直接使用 `save_memory` tool
  - 支持 `update_section` 细粒度操作
  - 支持多文件操作
  - 用完整消息搜索而非压缩摘要，信息更全
- **更新 `_consolidate_direct()`**：复用所有共享逻辑
- **删除冗余代码**：移除 `_compress_messages()`、`_search_related_memory()`、`_infer_target_file()`、`_decide_with_context()` 及相关提示词

**修改文件**：
| 文件 | 修改内容 |
|------|----------|
| `nanobot/agent/memory.py` | 重构 consolidation 逻辑，提取共享方法 |

**验证结果**：
- ✅ 29/29 测试通过
- ✅ 代码减少 62 行（971 行 → 909 行）
- ✅ 向后兼容：外部 API 完全不变
- ✅ 功能增强：RAG-based 现在支持 `update_section` 和多文件操作

**对比改进**：
| 对比项 | 优化前 | 优化后 |
|--------|--------|--------|
| LLM 调用次数（RAG 模式） | 2 次 | **1 次** |
| 支持 update_section | ❌ | **✅** |
| 支持多文件操作 | ❌ | **✅** |
| 代码重复 | 严重 | **消除** |

---

## 后续规划

### ✅ 已完成（全部 P0 问题）
- P0-1: 统一默认模型
- P0-2: Memory 索引效率优化（单文件更新）
- P0-3: 修复路径匹配
- P0-4: 修复缓存 Key
- P0-5: 去重阈值配置混乱
- P0-6: RAG-based consolidation 流程优化
- P0-7: 原子操作回滚可能失败
- P0-8: 余弦相似度重复实现
- P0-9: 固化触发时机生硬

### 中期（下一步）
1. **P1 问题修复**
   - P1-1: 添加索引版本管理
   - P1-2: 向量距离计算理解偏差
   - P1-3: 改进 `get_memory_context()` 关键词匹配
   - P1-4: 消除 consolidation 两种策略代码重复（✅ 已在 P0-6 中完成）
   - P1-5: 修复 `update_section()` 正则边界问题
   - P1-6: 缺少记忆访问统计
   - P1-7: 批量 embedding 无错误处理
   - P1-8: 缺少搜索质量指标

2. **配置完整传递**
   - 将更多配置项从 schema 传递到各模块

### 长期
1. **架构解耦**（RAG + Memory）
   - 事件驱动或回调机制
   - 明确职责边界

2. **A/B 测试框架**
   - 验证 RAG-based consolidation 效果
   - 数据驱动优化

---

## 2026-03-03: P1 问题 3 遍审查结论

**审查背景**：在修复完所有 P0 问题后，对 P1 问题进行重新评估。

**项目背景**：个人研究助手，未上线，非生产级，自己用。

---

### 审查结果

| ID | 问题 | 必要性 | 结论 |
|----|------|--------|------|
| P1-1 | 缺少索引版本管理 | 🔴 低 | 个人项目换模型删库重建就行 |
| P1-2 | 向量距离计算理解偏差 | 🔴 **不是问题** | 代码是正确的！sqlite-vec 余弦距离范围 [0,2]，`1.0 - (distance/2.0)` 正确转换成 [0,1] 相似度 |
| P1-3 | get_memory_context() 关键词匹配 | 🟡 低 | 个人用，误匹配影响不大 |
| P1-4 | consolidation 代码重复 | 🟢 **已完成** | ✅ 已在 P0-6 中修复 |
| P1-5 | update_section() 正则边界问题 | 🟡 低 | 代码有两层正则 + 验证回滚机制，已经很健壮 |
| P1-6 | 缺少记忆访问统计 | 🔴 低 | 个人项目不需要统计 |
| P1-7 | 批量 embedding 无错误处理 | 🔴 低 | 本地模型 + 已有降级策略（失败禁用 5 分钟，FTS 仍可用） |
| P1-8 | 缺少搜索质量指标 | 🔴 低 | 个人项目不需要指标 |

---

### 最终决定

**这 8 个 P1 问题，全部暂不修复！**

理由：
1. P1-2 不是问题，当前代码是正确的
2. P1-4 已经在 P0-6 中修复了
3. 其他问题对于个人研究助手来说，优先级很低，没有必要现在修

**建议**：先聚焦于核心功能的开发和使用，有实际需求时再考虑优化。