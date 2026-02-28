# Memory System Refactor: Multi-file + Incremental Operations

**Date:** 2026-02-28

## Overview

Refactored the memory system from a single-file design (MEMORY.md) to a multi-file architecture with incremental operations. This change improves knowledge management for research scenarios and reduces the risk of information loss during consolidation.

## Problem Statement

The original memory system had several limitations:

1. **Single file**: All memories in MEMORY.md, full overwrite on each consolidation
2. **No categorization**: Not suitable for research knowledge management
3. **Wasted HISTORY.md**: Write-only, never used for retrieval
4. **Inefficient loading**: Full memory loaded into context every time

## Solution

### New File Structure

```
memory/
├── PROFILE.md      # User profile (stable)
│   - Research direction, preferences, tools
│   - Update frequency: Low
│
├── PROJECTS.md     # Project knowledge (semi-stable)
│   - Tech stack, architecture, progress
│   - Update frequency: Medium
│
├── PAPERS.md       # Paper notes (incremental)
│   - Papers read, key findings
│   - Update frequency: High
│
├── DECISIONS.md    # Decision records (incremental)
│   - Why A over B, trade-offs
│   - Update frequency: Medium
│
├── TODOS.md        # Current tasks (frequent)
│   - Tasks, next steps, blockers
│   - Update frequency: High
│
└── HISTORY.md      # Event log (append-only)
    - Timeline summary, grep-searchable
```

### Incremental Operations

| Action | Description | Use Case |
|--------|-------------|----------|
| `append` | Add to end of file | New paper note, new decision |
| `prepend` | Add to start of file | New urgent todo |
| `update_section` | Update specific section | Project progress update |
| `replace` | Full file replacement | Complete todo list |
| `skip` | No change | File doesn't need update |

### Selective Loading

Memory files are loaded based on context:

- **Always loaded**: PROFILE.md, TODOS.md
- **Keyword-triggered**:
  - `论文/paper/arxiv/研究` → PAPERS.md
  - `项目/project/代码/架构` → PROJECTS.md
  - `为什么/决策/why/decision` → DECISIONS.md

## Changes

### 1. `nanobot/agent/memory.py` (Major Refactor)

**New Classes:**

```python
class MemoryFile(Enum):
    PROFILE = "PROFILE.md"
    PROJECTS = "PROJECTS.md"
    PAPERS = "PAPERS.md"
    DECISIONS = "DECISIONS.md"
    TODOS = "TODOS.md"
    HISTORY = "HISTORY.md"

@dataclass
class MemoryOperation:
    file: MemoryFile
    action: str  # append | prepend | update_section | replace | skip
    content: Optional[str] = None
    section: Optional[str] = None
```

**New Tool Schema:**

```python
_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "parameters": {
                "properties": {
                    "history_entry": {"type": "string"},
                    "operations": {
                        "type": "array",
                        "items": {
                            "properties": {
                                "file": {"enum": ["profile", "projects", "papers", "decisions", "todos"]},
                                "action": {"enum": ["append", "prepend", "update_section", "replace", "skip"]},
                                "content": {"type": "string"},
                                "section": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
]
```

**New Methods:**

- `read_file(file: MemoryFile) -> str`
- `read_section(file: MemoryFile, section: str) -> Optional[str]`
- `append(file: MemoryFile, content: str)`
- `prepend(file: MemoryFile, content: str)`
- `update_section(file: MemoryFile, section: str, content: str)`
- `replace(file: MemoryFile, content: str)`
- `apply_operation(op: MemoryOperation)`
- `get_memory_context(query: str | None = None) -> str`
- `_get_memory_summary() -> str`

**Removed:**

- `memory_file` attribute (replaced by `MemoryFile` enum)
- `history_file` attribute (replaced by `MemoryFile.HISTORY`)
- `read_long_term()` (replaced by `read_file()`)
- `write_long_term()` (replaced by incremental operations)

### 2. `nanobot/agent/context.py`

**Updated `_get_identity()`:**

```python
## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/
  - PROFILE.md: User profile (preferences, research direction)
  - PROJECTS.md: Project knowledge (tech stack, architecture)
  - PAPERS.md: Paper notes and research findings
  - DECISIONS.md: Decision records (why A over B)
  - TODOS.md: Current tasks and next steps
  - HISTORY.md: Append-only event log (grep-searchable)
```

**Updated `build_system_prompt()`:**

```python
def build_system_prompt(self, skill_names: list[str] | None = None, query: str | None = None) -> str:
    # ...
    memory = self.memory.get_memory_context(query)  # Now accepts query for selective loading
```

**Updated `build_messages()`:**

```python
return [
    {"role": "system", "content": self.build_system_prompt(skill_names, query=current_message)},
    # ...
]
```

### 3. `nanobot/skills/memory/SKILL.md`

Updated documentation to reflect new structure and usage patterns.

### 4. `tests/test_memory_consolidation_types.py`

Added 15 new tests:

**Incremental Operations:**
- `test_append_to_empty_file`
- `test_append_to_existing_file`
- `test_prepend_to_empty_file`
- `test_prepend_to_existing_file`
- `test_replace_entire_file`
- `test_update_section_creates_new`
- `test_update_section_replaces_existing`
- `test_update_section_preserves_other_sections`
- `test_skip_action_does_nothing`

**Selective Loading:**
- `test_default_loads_profile_and_todos`
- `test_query_triggers_papers`
- `test_query_triggers_projects`
- `test_query_triggers_decisions`
- `test_empty_files_not_loaded`
- `test_get_memory_summary`

## Migration

No migration needed. This is a new project not yet in production. Users can start fresh with the new memory structure.

## Stats

| File | Changes |
|------|---------|
| `nanobot/agent/memory.py` | +288 lines (major refactor) |
| `nanobot/agent/context.py` | +59 lines (identity + query support) |
| `nanobot/skills/memory/SKILL.md` | +41 lines (documentation) |
| `tests/test_memory_consolidation_types.py` | +214 lines (15 new tests) |

## Testing

All 108 tests pass:

```
tests/test_memory_consolidation_types.py .....  (20 tests)
tests/test_consolidate_offset.py .........     (39 tests)
...other tests...
============================= 108 passed ==============================
```

---

## Post-Refinement (2026-02-28)

### Bug Fixes & Optimizations

After initial implementation, several issues were identified and fixed:

| Issue | Fix | File |
|-------|-----|------|
| **Regex edge case in `update_section`** | Section matching regex failed when section was at end of file without trailing newline. Added `(?:\n|$)` to handle both cases. | `nanobot/agent/memory.py:127, 160` |
| **Overly broad keyword triggers** | "研究" (research) was too broad, causing false positives. Replaced with more specific triggers: "文献", "read paper", "论文笔记", "paper note". Decisions also got more specific triggers: "why did i", "为什么选", "为什么决定". | `nanobot/agent/memory.py:233-249` |
| **Missing parameter validation** | Added validation in `apply_operation()` to check required parameters before execution. Skips operation with warning if parameters are missing. | `nanobot/agent/memory.py:184-190` |
| **Redundant default values** | Removed redundant `or ""` defaults after parameter validation (since validation already ensures non-None). | `nanobot/agent/memory.py:192-199` |
| **Legacy compatibility removed** | Removed `memory_update` legacy format support from tests and tool response handling. | `tests/test_memory_consolidation_types.py` |

### Updated Selective Loading Triggers

**PAPERS.md triggers:**
- Before: `["论文", "paper", "arxiv", "研究"]`
- After: `["论文", "paper", "arxiv", "文献", "read paper", "论文笔记", "paper note"]`

**DECISIONS.md triggers:**
- Before: `["为什么", "决策", "why", "decision"]`
- After: `["为什么", "决策", "why did i", "为什么选", "为什么决定"]`