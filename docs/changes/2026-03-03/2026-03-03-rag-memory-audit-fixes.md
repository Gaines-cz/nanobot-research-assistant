# RAG & Memory 模块审计修复报告

**创建日期**: 2026-03-03
**修复日期**: 2026-03-03
**相关审计**: `docs/issues/2026-03-03-rag-memory-audit.md`

---

## 一、背景

基于审计报告 `docs/issues/2026-03-03-rag-memory-audit.md` 的综合分析，本次修复完成了 **9 个 P0 问题** 的全部修复，并对 P1 问题进行了重新评估。

### 问题统计

| 优先级 | 数量 | 状态 |
|--------|------|------|
| 🔴 P0 | 9 | ✅ 已完成 |
| 🟡 P1 | 8 | ⏸️ 重新评估后暂不修复 |
| 🟢 P2 | 7 | ⏸️ 暂不修复 |

---

## 二、P0 修复详情

### P0-1: 统一默认模型 ✅

**问题**: `SearchKnowledgeTool.__init__()` 默认使用 `"all-MiniLM-L6-v2"` (384 维)，而 `RAGConfig` 默认使用 `"BAAI/bge-m3"` (1024 维)，可能导致 embedding 维度不匹配。

**位置**: `nanobot/agent/tools/rag.py:50`, `nanobot/config/schema.py:331`

**修复方案**: 统一默认模型为 `"BAAI/bge-m3"`。

---

### P0-2: Memory 索引效率优化 ✅

**问题**: 每次 memory consolidation 后都调用 `scan_and_index(memory_dir)` 全量扫描，效率低。

**位置**: `nanobot/agent/memory.py:758-767`

**修复方案**:
- 新增 `DocumentIndexer.index_single_file()` 方法
- consolidation 时只索引修改的文件，不扫描整个目录

**修改文件**:
- `nanobot/rag/indexer.py` - 新增 `index_single_file()` 方法
- `nanobot/agent/memory.py` - 更新 consolidation 逻辑

---

### P0-3: 修复路径匹配 ✅

**问题**: RAG 记忆搜索使用字符串前缀匹配，跨平台不可靠，可能误匹配。

**位置**: `nanobot/agent/memory.py:811-823`

**修复方案**: 使用 `Path.is_relative_to()` 替代字符串匹配。

---

### P0-4: 修复缓存 Key ✅

**问题**: 搜索缓存使用 `hash(query)` 不安全，可能碰撞，且基础搜索和高级搜索缓存 Key 不一致。

**位置**: `nanobot/rag/search.py:134`

**修复方案**: 使用 `hashlib.sha256()` 替代 `hash()`，基础搜索和高级搜索都包含 `top_k` 参数。

---

### P0-5: 去重阈值配置混乱 ✅

**问题**: `SemanticDeduplicator.__init__()` 默认 `0.9`，`RAGDefaults.DEDUP_THRESHOLD = 0.7`，配置不明确。

**位置**: `nanobot/rag/rerank.py:130`, `nanobot/config/schema.py:317`

**修复方案**: 统一阈值为 `0.7`，明确配置优先级。

---

### P0-6: RAG-based consolidation 流程优化 ✅

**问题**: RAG-based consolidation 需要 2 次 LLM 调用，不支持 `update_section`，不支持多文件操作，代码重复严重。

**位置**: `nanobot/agent/memory.py:652-778`

**修复方案**:
- 提取共享方法：
  - `_get_messages_to_consolidate()` - 共享的消息选择逻辑
  - `_process_save_memory_tool_call()` - 共享的 tool call 处理
  - `_apply_save_memory_operations()` - 共享的操作应用逻辑
  - `_search_related_memory_enhanced()` - 增强的 RAG 搜索
- 重构 `_consolidate_with_rag()`：
  - 从 2 次 LLM 调用降为 1 次
  - 直接使用 `save_memory` tool
  - 支持 `update_section` 和多文件操作
- 更新 `_consolidate_direct()` 复用共享逻辑

**改进对比**:
| 对比项 | 优化前 | 优化后 |
|--------|--------|--------|
| LLM 调用次数（RAG 模式） | 2 次 | **1 次** |
| 支持 update_section | ❌ | **✅** |
| 支持多文件操作 | ❌ | **✅** |
| 代码行数 | ~971 行 | **~909 行** (-62) |

---

### P0-7: 原子操作回滚可能失败 ✅

**问题**: `apply_operations_atomic()` 回滚时可能失败（磁盘满/权限问题），导致数据不一致。

**位置**: `nanobot/agent/memory.py:388-405`

**修复方案**: 新增 `_safe_write()` 方法，使用临时文件 + rename（原子操作）。

---

### P0-8: 余弦相似度重复实现 ✅

**问题**: `memory.py` 和 `rerank.py` 各有一个 `_cosine_similarity()` 实现，代码重复。

**位置**: `nanobot/agent/memory.py:242-256`, `nanobot/rag/rerank.py:133-145`

**修复方案**:
- 提取到 `nanobot/utils/helpers.py` 中的 `cosine_similarity()` 函数
- 两个模块都使用共享函数

---

### P0-9: 固化触发时机生硬 ✅

**问题**: 只按消息窗口触发固化，不够智能。

**修复方案**:
- 新增 `ConsolidationTrigger` 类
- 三种触发策略（满足任一即触发）：
  1. **消息窗口（保底）**：未固化消息数 ≥ threshold（默认 100）
  2. **会话暂停**：超过 5 分钟无新消息 + 已有 10+ 条消息
  3. **重要内容检测**：最近 5 条消息含关键词 + 已有 5+ 条消息
- 支持中英文关键词：决定/decision、结论/conclusion、todo/任务、plan/计划、project/项目、architecture/架构、remember/记住、note/笔记

**修改文件**:
- `nanobot/session/manager.py` - Session 类新增字段
- `nanobot/agent/loop.py` - 新增 `ConsolidationTrigger` 类
- `nanobot/config/schema.py` - AgentDefaults 新增配置项
- `nanobot/cli/commands.py` - 配置传递

---

## 三、P1 问题重新评估

经过 3 遍仔细审查，结论是：**这 8 个 P1 问题全部暂不修复**。

### 详细评估

| ID | 问题 | 评估结论 |
|----|------|----------|
| P1-1 | 缺少索引版本管理 | 🔴 低 - 个人项目换模型删库重建就行 |
| P1-2 | 向量距离计算理解偏差 | 🔴 **不是问题** - 代码正确！sqlite-vec 余弦距离范围 [0,2]，`1.0 - (distance/2.0)` 正确转换成 [0,1] 相似度 |
| P1-3 | get_memory_context() 关键词匹配 | 🟡 低 - 个人用，误匹配影响不大 |
| P1-4 | consolidation 代码重复 | 🟢 **已完成** - P0-6 中已修复 |
| P1-5 | update_section() 正则边界问题 | 🟡 低 - 代码有两层正则 + 验证回滚机制，已经很健壮 |
| P1-6 | 缺少记忆访问统计 | 🔴 低 - 个人项目不需要统计 |
| P1-7 | 批量 embedding 无错误处理 | 🔴 低 - 本地模型 + 已有降级策略（失败禁用 5 分钟，FTS 仍可用） |
| P1-8 | 缺少搜索质量指标 | 🔴 低 - 个人项目不需要指标 |

---

## 四、变更文件清单

| 文件 | 变更内容 |
|------|----------|
| `nanobot/agent/memory.py` | P0-2: Memory 索引单文件更新; P0-3: 修复路径匹配; P0-6: RAG-based consolidation 重构; P0-7: 原子操作 _safe_write(); P0-8: 使用共享 cosine_similarity() |
| `nanobot/utils/helpers.py` | P0-8: 新增 cosine_similarity() 函数 |
| `nanobot/rag/indexer.py` | P0-2: 新增 index_single_file() 方法 |
| `nanobot/rag/rerank.py` | P0-8: 使用共享 cosine_similarity() |
| `nanobot/agent/loop.py` | P0-9: 新增 ConsolidationTrigger 类 |
| `nanobot/session/manager.py` | P0-9: Session 类新增 last_consolidation_check、consolidation_paused 字段 |
| `nanobot/config/schema.py` | P0-9: AgentDefaults 新增 4 个配置项 |
| `nanobot/cli/commands.py` | P0-9: 配置传递 |
| `nanobot/rag/search.py` | P0-4: 修复缓存 Key |
| `nanobot/rag/store.py` | 小改动 |
| `nanobot/agent/tools/rag.py` | P0-1: 统一默认模型 |
| `docs/issues/2026-03-03-rag-memory-audit.md` | 更新审计报告，记录 P0 完成和 P1 评估结论 |
| `docs/drafts/2026-03-03-rag-evaluation-design.md` | 新增草稿文档 |

---

## 五、验证结果

### 5.1 单元测试验证

```bash
$ python -m pytest tests/test_memory_consolidation_types.py -xvs
============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-9.0.2, pluggy-1.6.0
collected 29 items

tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_string_arguments_work PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_dict_arguments_serialized_to_json PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_string_arguments_as_raw_json PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_no_tool_call_returns_false PASSED
tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_skips_when_few_messages PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_append_to_empty_file PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_append_to_existing_file PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_prepend_to_empty_file PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_prepend_to_existing_file PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_replace_entire_file PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_update_section_creates_new PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_update_section_replaces_existing PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_update_section_preserves_other_sections PASSED
tests/test_memory_consolidation_types.py::TestMemoryIncrementalOperations::test_skip_action_does_nothing PASSED
tests/test_memory_consolidation_types.py::TestMemoryContextLoading::test_default_loads_profile_and_todos PASSED
tests/test_memory_consolidation_types.py::TestMemoryContextLoading::test_query_triggers_papers PASSED
tests/test_memory_consolidation_types.py::TestMemoryContextLoading::test_query_triggers_projects PASSED
tests/test_memory_consolidation_types.py::TestMemoryContextLoading::test_query_triggers_decisions PASSED
tests/test_memory_consolidation_types.py::TestMemoryContextLoading::test_empty_files_not_loaded PASSED
tests/test_memory_consolidation_types.py::TestMemoryContextLoading::test_get_memory_summary PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_exact_match PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_new_content PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_whitespace_normalized PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_empty_content PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_partial_match_not_blocked PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_semantic_similar PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_semantic_different PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_append_with_dedup_embedding_failure_graceful PASSED
tests/test_memory_consolidation_types.py::TestMemoryDeduplication::test_cosine_similarity PASSED

============================== 29 passed in 1.40s ==============================
```

### 5.2 修复完成率

| 优先级 | 数量 | 完成 | 完成率 |
|--------|------|------|--------|
| P0 | 9 | 9 | **100%** |
| P1 | 8 | 0 | 暂不修复 |
| P2 | 7 | 0 | 暂不修复 |

---

## 六、相关文档

- 审计报告：`docs/issues/2026-03-03-rag-memory-audit.md`
- 优化方案：`docs/drafts/2026-03-03-rag-evaluation-design.md`