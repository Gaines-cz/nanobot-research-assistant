# Memory 层优化方案

**Date:** 2026-02-28
**Status:** Draft

---

## 背景

当前 memory 系统已实现多文件分类存储和增量操作，但仍有几个可优化点，主要围绕**健壮性**、**可观测性**和**检索准确性**。

---

## 一、高优先级优化（低风险，高收益）

### 1.1 操作日志

**问题**: 记忆更新没有日志，难以调试和追溯。

**现状**: `apply_operation()` 执行操作时没有记录。

**方案**:
```python
def apply_operation(self, op: MemoryOperation) -> None:
    # 记录操作日志
    log_msg = f"[Memory] {op.action} on {op.file.value}"
    if op.section:
        log_msg += f" (section: {op.section})"
    logger.info(log_msg)

    # ... 执行操作
```

---

### 1.2 批量操作原子性

**问题**: 多个 operations 中，一个失败后面的不执行，但前面的已经改了，导致状态不一致。

**现状**:
```python
for op_data in operations:
    op = MemoryOperation(...)
    self.apply_operation(op)  # 失败的话，前面的已经改了
```

**方案**:
```python
async def apply_operations_atomic(
    self,
    ops: list[MemoryOperation]
) -> bool:
    """原子性地执行多个操作，失败则回滚。"""
    # 1. 先校验所有操作
    for op in ops:
        if not self._validate_operation(op):
            logger.warning("Operation validation failed: {}", op)
            return False

    # 2. 备份当前状态
    backup = self._backup_all_files()

    # 3. 执行操作
    try:
        for op in ops:
            self.apply_operation(op)
        return True
    except Exception:
        logger.exception("Batch operation failed, rolling back")
        self._restore_from_backup(backup)
        return False

def _backup_all_files(self) -> dict[MemoryFile, str]:
    """备份所有 memory 文件内容。"""
    backup = {}
    for file in MemoryFile:
        if file != MemoryFile.HISTORY:  # HISTORY 只追加，不回滚
            backup[file] = self.read_file(file)
    return backup

def _restore_from_backup(self, backup: dict[MemoryFile, str]) -> None:
    """从备份恢复。"""
    for file, content in backup.items():
        self.replace(file, content)
```

---

### 1.3 记忆更新后验证

**问题**: LLM 可能写错格式（比如 section 名不对），导致 update_section 实际上没生效。

**现状**: `update_section()` 执行后不验证结果。

**方案**:
```python
def update_section(self, file: MemoryFile, section: str, content: str) -> None:
    old_content = self.read_file(file)

    # 执行更新
    existing = old_content
    section_header = f"## {section}"
    new_section = f"{section_header}\n{content.strip()}\n"
    pattern = rf"##\s+{re.escape(section)}\s*(?:\n|$).*?(?=\n##\s|\Z)"

    if re.search(pattern, existing, re.DOTALL):
        new_content = re.sub(pattern, new_section.rstrip(), existing, flags=re.DOTALL)
    elif existing:
        new_content = existing.rstrip() + "\n\n" + new_section
    else:
        new_content = new_section

    path = self.memory_dir / file.value
    path.write_text(new_content, encoding="utf-8")

    # 验证：确认内容确实更新了
    verification = self.read_file(file)
    if content.strip() not in verification:
        logger.warning(
            "Section update verification failed, rolling back. "
            "Section: {}, File: {}",
            section, file.value
        )
        self.replace(file, old_content)
```

---

## 二、中优先级优化（需要设计）

### 2.1 记忆去重

**问题**: LLM 可能重复追加相似的内容到 PAPERS.md 等文件。

**方案**:
```python
def append_with_dedup(
    self,
    file: MemoryFile,
    content: str,
    similarity_threshold: float = 0.8
) -> None:
    """追加前检查是否已有相似内容。"""
    existing = self.read_file(file)

    # 简单去重：精确匹配
    if content.strip() in existing:
        logger.debug("Exact content already exists, skipping append")
        return

    # 语义去重（可选，需要 embedding）
    if self._embedding_provider:
        if self._is_semantically_similar(content, existing, similarity_threshold):
            logger.debug("Similar content already exists, skipping append")
            return

    self.append(file, content)
```

---

### 2.2 关键词匹配 → 语义检索

**问题**: 关键词太死板，比如 "这篇文章讲了什么" 不会触发 PAPERS.md。

**机会**: 项目已经有完整的 RAG 模块，可以复用！

**方案**:
```python
# 为 memory 创建单独的 DocumentStore 实例
# memory/
#   ├── PROFILE.md
#   ├── PROJECTS.md
#   ├── ...
#   └── memory.db  (memory 的向量索引)

def get_memory_context(self, query: str | None = None) -> str:
    if not query:
        # 默认加载 PROFILE + TODOS
        ...
        return parts

    # 用语义检索找到相关的记忆片段
    relevant_chunks = await self._memory_store.search_advanced(query, top_k=3)

    # 只注入相关的
    parts = []
    for chunk in relevant_chunks:
        parts.append(f"## Relevant Memory\n{chunk.combined_content}")

    return "\n\n---\n\n".join(parts)
```

---

### 2.3 记忆重要性评分 + 衰减

**问题**: 所有记忆同等重要，但有些应该被优先记住。

**方案**:
```python
@dataclass
class MemoryEntry:
    id: str
    content: str
    importance: float  # 0.0 - 1.0
    created_at: datetime
    access_count: int = 0

# 合并时让 LLM 给重要性评分
# 检索时按重要性排序
```

---

## 三、长期优化（架构级改动）

### 3.1 结构化记忆（JSONL）

**问题**: 当前是纯 Markdown 文件，难以做细粒度管理。

**方案**:
```
memory/
├── facts.jsonl
│   # {"id": "uuid", "content": "用户喜欢喝茶", "tags": ["偏好"], "importance": 0.9, ...}
│   # {"id": "uuid", "content": "项目用 Python 3.11", "tags": ["项目"], "importance": 0.7, ...}
└── history.jsonl
```

---

### 3.2 记忆编辑工具暴露给 LLM

**问题**: 只有合并时才能更新记忆，LLM 无法实时编辑。

**方案**:
```python
class EditMemoryTool(BaseTool):
    name = "edit_memory"
    description = "Edit memory files directly."

    parameters = {
        "action": {"type": "string", "enum": ["append", "update_section", "replace"]},
        "file": {"type": "string", "enum": ["profile", "projects", "papers", "decisions", "todos"]},
        "section": {"type": "string", "optional": True},
        "content": {"type": "string"},
    }

    async def execute(
        self,
        action: str,
        file: str,
        content: str,
        section: str | None = None,
    ) -> str:
        # 执行记忆编辑
        ...
```

---

## 四、实现优先级

| 优先级 | 优化项 | 工作量 | 理由 |
|--------|--------|--------|------|
| P0 | 操作日志 | 1h | 可观测性，容易调试 |
| P0 | 批量操作原子性 | 3h | 避免部分更新导致不一致 |
| P1 | 记忆更新后验证 | 2h | 防止 LLM 写错格式 |
| P1 | 记忆去重 | 2h | 避免冗余 |
| P2 | 语义检索（复用 RAG） | 8h | 最大的体验提升 |

---

## 五、验收标准

1. **操作日志**: 每次记忆更新都有 INFO 级别日志
2. **原子性**: 批量操作失败时，文件恢复到初始状态
3. **验证**: LLM 写错 section 名时，操作回滚并 warning
4. **去重**: 相同内容不会重复追加

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 原子性备份占用内存 | 低 | 只备份非 HISTORY 文件，且只在批量操作时 |
| 验证误判 | 低 | 只在内容完全没出现时才回滚 |
| 去重误判 | 低 | 先做精确匹配，语义匹配设为可选 |