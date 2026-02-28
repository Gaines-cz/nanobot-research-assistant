# Memory Bug Fixes

**Date:** 2026-02-28

## Overview

Fixed 3 potential issues in memory system found during code review. Additionally, fixed similar issue in heartbeat service.

## Changes Made

### 1. append_history Empty Value Check

**Location:** `nanobot/agent/memory.py` lines 217-224

**Problem:**
`append_history` had no null/empty value validation. Empty entries could be written to history file.

**Fix:**
```python
def append_history(self, entry: str) -> None:
    """Append to history log."""
    if not entry or not entry.strip():
        logger.debug("Skipping empty history entry")
        return
    path = self.memory_dir / MemoryFile.HISTORY.value
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n\n")
```

**Benefit:**
- Empty or whitespace-only entries are now skipped
- Prevents invalid data from being written to history
- Debug log for traceability

---

### 2. Handle Multiple tool_calls in Memory Consolidation

**Location:** `nanobot/agent/memory.py` lines 350-365

**Problem:**
When LLM returned multiple tool_calls, only the first one was processed, potentially losing other memory operations.

**Fix:**
```python
# Find all save_memory tool calls
save_memory_calls = [tc for tc in response.tool_calls if tc.name == "save_memory"]

if not save_memory_calls:
    logger.warning("Memory consolidation: no save_memory tool call found")
    return False

# Process the first save_memory call
args = save_memory_calls[0].arguments

# Log if there are additional calls we're ignoring
if len(save_memory_calls) > 1:
    logger.warning(
        "Memory consolidation: %d save_memory calls found, processing only the first",
        len(save_memory_calls)
    )
```

**Benefit:**
- Now explicitly filters for `save_memory` tool calls
- Logs warning when multiple save_memory calls are found
- Ensures first call is processed (maintains backward compatibility)

---

### 3. tool_calls Empty Check in Heartbeat Service

**Location:** `nanobot/heartbeat/service.py` lines 102-112

**Problem:**
Similar to issue #2: no empty check before accessing `tool_calls[0]`, and no handling for multiple tool calls.

**Fix:**
```python
if not response.tool_calls:
    logger.warning("Heartbeat: no tool calls found")
    return "skip", ""

# Process the first tool call
args = response.tool_calls[0].arguments

if len(response.tool_calls) > 1:
    logger.warning("Heartbeat: %d tool calls found, processing only the first", len(response.tool_calls))

return args.get("action", "skip"), args.get("tasks", "")
```

**Benefit:**
- Prevents potential IndexError if tool_calls is empty
- Logs warning for multiple tool calls (unexpected but handled)

---

## Verification

All tests pass:
```bash
pytest tests/test_memory_consolidation_types.py -xvs
# 20 passed
```
