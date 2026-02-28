# Memory + RAG 集成问题修复报告

**创建日期**: 2026-03-01
**修复日期**: 2026-03-01
**相关计划**: `docs/plans/2026-02-28-memory-optimization-project.md`

---

## 一、背景

在 2026-02-28 的 RAG + Memory 集成优化中，实现了：
1. RAG 索引同时包含 `docs/` 和 `memory/` 目录
2. Consolidation 优化流程：压缩 → RAG 搜索 → LLM 决策 → 写入

经过代码审查，发现 6 个实际问题，本次修复全部解决。

---

## 二、问题清单与修复详情

### 🔴 P0 问题 1: RAG 索引在 consolidation 后不更新

**问题类型**: 功能缺陷

**问题位置**: `nanobot/agent/memory.py:810-813`

**问题描述**:
- Consolidation 写入的新记忆无法被后续 RAG 搜索检索到
- 原代码被注释掉：`# if self._rag_store: await self._update_memory_index()`
- 导致第二轮 consolidation 时搜不到第一轮写的记忆 → 去重失效

**修复方案**:
```python
# Step 6: Update RAG index for memory changes
if action in ("create", "merge", "replace"):
    try:
        # Only index memory directory (efficient incremental update)
        await self._rag_store.scan_and_index(
            self._memory.memory_dir,
            chunk_size=self._rag_store.config.memory_chunk_size,
            chunk_overlap=self._rag_store.config.memory_chunk_overlap,
        )
        logger.info("RAG memory index updated")
    except Exception as e:
        logger.warning("RAG memory index update failed: {}", e)
```

**修复位置**: `nanobot/agent/memory.py:827-838`

---

### 🔴 P0 问题 2: `_retrieve_tool` 属性可能未初始化

**问题类型**: 潜在崩溃

**问题位置**: `nanobot/agent/loop.py:139-154`

**问题描述**:
- 当 `rag_config.enabled = False` 时，`self._retrieve_tool` 未被赋值
- 后续访问该属性会导致 `AttributeError`

**修复方案**:

1. 在 `_register_default_tools()` 开头初始化：
```python
def _register_default_tools(self) -> None:
    """Register the default set of tools."""
    self._retrieve_tool = None  # 先初始化为 None
    # ... 后续代码
```

2. 简化 `stop()` 方法，移除 `hasattr` 检查：
```python
def stop(self) -> None:
    """Stop the agent loop."""
    self._running = False
    logger.info("Agent loop stopping")
    # Close RAG tool if it exists
    if self._retrieve_tool is not None:
        try:
            self._retrieve_tool.close()
            logger.debug("RAG tool closed")
        except Exception as e:
            logger.warning("Error closing RAG tool: {}", e)
```

**修复位置**:
- `nanobot/agent/loop.py:123` - 初始化
- `nanobot/agent/loop.py:378` - 简化检查

---

### 🔴 P1 问题 3: RAG 搜不到时 `target_file` 缺失

**问题类型**: 分类失效

**问题位置**: `nanobot/agent/memory.py:776-780`, `787`

**问题描述**:
- 当 RAG 搜索返回空时，决策逻辑缺少 `target_file` 字段
- 所有内容都写入默认的 `PROFILE.md`

**修复方案**:

1. 添加 `_infer_target_file()` 方法，根据关键词推断文件类型：
```python
def _infer_target_file(self, summary: str) -> str:
    """根据摘要内容推断目标文件类型。"""
    summary_lower = summary.lower()

    # Papers keywords
    if any(kw in summary_lower for kw in ["论文", "paper", "arxiv", "阅读", "研究", "文献"]):
        return "papers"
    # Projects keywords
    if any(kw in summary_lower for kw in ["项目", "project", "代码", "架构", "开发", "技术", "stack"]):
        return "projects"
    # Decisions keywords
    if any(kw in summary_lower for kw in ["决定", "决策", "选择", "why", "因为", "原因", "instead of"]):
        return "decisions"
    # Todos keywords
    if any(kw in summary_lower for kw in ["任务", "todo", "待办", "下一步", "计划", "plan", "task"]):
        return "todos"
    # Default to profile
    return "profile"
```

2. 修改 consolidation 逻辑，总是调用 LLM 决策，并添加 fallback：
```python
# Step 4: LLM decide (always call LLM, even without related memory)
decision = await self._decide_with_context(summary, related_memory, provider, model)

# Fallback: if LLM doesn't return target_file, infer it from summary
if "target_file" not in decision:
    decision["target_file"] = self._infer_target_file(summary)
```

**修复位置**:
- `nanobot/agent/memory.py:687-704` - 新增方法
- `nanobot/agent/memory.py:792-797` - 修改逻辑

---

### 🟡 P2 问题 4: DECISION_PROMPT 中 `create` 语义不清

**问题类型**: LLM 困惑

**问题位置**: `nanobot/agent/memory.py:610-625`

**问题描述**:
- `create` 时应该输出**新增内容**（用于 `append()`）
- `replace` 时才应该输出**完整内容**
- Prompt 中两者都写"完整记忆内容"，会让 LLM 困惑

**修复方案**:

修改 `memory_update` 字段说明：
```python
"memory_update": "新内容（create 时）或完整内容（replace/merge 时）"
```

关键说明：
```
关键：
- 必须指定 target_file 字段
- create 时输出要**追加**的新内容（不需要包含已有内容）
- merge 时输出**合并后的完整内容**（结合新旧内容）
- replace 时输出文件的**完整新内容**
- skip 时只需要 action 和 reason
```

**修复位置**: `nanobot/agent/memory.py:617`, `621-625`

---

### 🟡 P2 问题 5: 路径过滤可能误匹配

**问题类型**: 边界情况

**问题位置**: `nanobot/agent/memory.py:675-678`

**问题描述**:
- `memory_dir_str in doc_path` 可能误匹配
- 例：`memory_dir = /tmp/memory`，`doc_path = /tmp/memory_cache/docs/1.md` → ❌ 误匹配

**修复方案**:

使用严格的路径前缀匹配：
```python
memory_dir_prefix = memory_dir_str + "/"  # 确保是目录前缀
for r in results:
    doc_path = r.document.path.replace("\\", "/").rstrip("/")
    # 使用严格的路径前缀匹配，避免误匹配
    if doc_path.startswith(memory_dir_prefix) or doc_path == memory_dir_str:
        memory_results.append(r.combined_content)
```

**修复位置**: `nanobot/agent/memory.py:675-680`

---

### 🟢 P3 问题 6: consolidation 中 merge 逻辑重复

**问题类型**: 代码冗余

**问题位置**: `nanobot/agent/memory.py:799-808`

**问题描述**:
- `merge` 操作先用 `existing + decision["memory_update"]` 拼接，再调用 `replace()`
- 但如果 LLM 在 `memory_update` 中已经返回了合并后的完整内容，就会重复

**修复方案**:

简化 merge 逻辑，让 LLM 直接返回合并后的完整内容：
```python
if action in ("create", "merge", "replace") and decision.get("memory_update"):
    if action == "create":
        # append: 追加新内容
        self._memory.append(target_memory_file, decision["memory_update"])
    elif action == "merge":
        # LLM 应返回合并后的完整内容
        self._memory.replace(target_memory_file, decision["memory_update"])
    elif action == "replace":
        # LLM 应返回完整的新内容
        self._memory.replace(target_memory_file, decision["memory_update"])
```

**修复位置**: `nanobot/agent/memory.py:816-825`

---

## 三、变更文件清单

| 文件 | 变更行数 | 变更内容 |
|------|----------|----------|
| `nanobot/agent/memory.py` | ~50 行 | 1. `consolidate()`: 添加 RAG 索引更新 (Step 6)<br>2. `DECISION_PROMPT`: 修正 `create`/`merge` 语义<br>3. `_search_related_memory()`: 路径前缀匹配修复<br>4. 新增 `_infer_target_file()` 方法<br>5. `merge` 逻辑简化 |
| `nanobot/agent/loop.py` | ~5 行 | 1. `_register_default_tools()`: 初始化 `_retrieve_tool`<br>2. `stop()`: 移除 `hasattr` 检查 |
| `nanobot/agent/tools/rag.py` | ~5 行 | 1. `index_memory_only()`: 使用配置参数而非硬编码 |

---

## 四、验证结果

### 4.1 单元测试验证

```bash
$ pytest tests/test_memory_consolidation_types.py -xvs
============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-9.0.2, pluggy-1.6.0
collected 29 items

tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_string_arguments_work PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_dict_arguments_serialized_to_json PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_string_arguments_as_raw_json PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_no_tool_call_returns_false PASSED
# ... 共 29 个测试全部通过
============================== 29 passed in 2.14s ==============================
```

### 4.2 代码审查验证

| 问题 | 优先级 | 状态 | 关键验证点 |
|------|--------|------|------------|
| 1. RAG 索引更新 | P0 | ✅ | 使用 `self._rag_store.config.memory_chunk_size` |
| 2. `_retrieve_tool` 初始化 | P0 | ✅ | 第 123 行初始化，第 378 行无 `hasattr` |
| 3. `target_file` 推断 | P1 | ✅ | `_infer_target_file()` 存在，有 fallback |
| 4. DECISION_PROMPT 语义 | P2 | ✅ | create/merge/replace 语义清晰区分 |
| 5. 路径前缀匹配 | P2 | ✅ | 使用 `startswith()` 而非 `in` |
| 6. merge 逻辑简化 | P3 | ✅ | 直接调用 `replace()` |

### 4.3 功能验证脚本

```python
# Test 1: DECISION_PROMPT verification
assert 'create 时输出要**追加**的新内容' in DECISION_PROMPT
assert 'merge 时输出**合并后的完整内容**' in DECISION_PROMPT

# Test 2: _infer_target_file method verification
assert optimized._infer_target_file('这篇论文讲的是机器学习') == 'papers'
assert optimized._infer_target_file('项目架构设计') == 'projects'
assert optimized._infer_target_file('决定使用 Python') == 'decisions'
assert optimized._infer_target_file('下一步计划') == 'todos'
assert optimized._infer_target_file('今天天气不错') == 'profile'
```

---

## 五、注意事项

### 5.1 RAG 索引更新性能

- 每次 consolidation 都更新索引可能影响性能
- 当前实现：
  - 仅索引 `memory/` 目录（高效增量更新）
  - 失败时仅记录警告，不阻止 consolidation 完成
- 未来优化方向：
  - 添加配置开关 `enable_auto_rag_index_update`
  - 或仅在 memory 内容实际变化时更新

### 5.2 向后兼容性

- 修改 `DECISION_PROMPT` 可能影响 LLM 行为
- 已添加 fallback 逻辑，确保即使 LLM 未返回 `target_file` 也能正常工作

### 5.3 错误处理

- RAG 索引更新失败不应阻止 consolidation 完成（已处理）
- `_infer_target_file()` 失败时默认返回 `profile`

---

## 六、相关文档

- 实施计划：`docs/plans/2026-02-28-memory-optimization-project.md`
- Memory 系统：`nanobot/agent/memory.py`
- RAG 工具：`nanobot/agent/tools/rag.py`
- Agent 循环：`nanobot/agent/loop.py`