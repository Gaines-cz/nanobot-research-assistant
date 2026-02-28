# RAG 数据库连接与错误处理改进

**Date:** 2026-02-28

## Overview

Optimizations for the research assistant RAG functionality: Fixed database connection management to prevent resource leaks during long-running research sessions; improved visibility into vector search status so users clearly know whether hybrid search (vector + full-text) or full-text-only search is available for querying local research papers, lab records, project proposals, technical documentation, and other documents.

## Issues Fixed

### 问题 1: 数据库连接未关闭 - 资源泄漏

**Location:** `nanobot/rag/store.py`

**Problem:**
- DocumentStore 没有提供关闭数据库连接的方法
- 长时间运行的进程（如 gateway 模式）可能导致连接泄漏
- CLI 命令执行后没有显式关闭连接

**Fix:**
```python
def close(self) -> None:
    """Close the database connection if open."""
    if self._db is not None:
        try:
            self._db.close()
            logger.debug("Database connection closed")
        except Exception as e:
            logger.warning("Error closing database connection: {}", e)
        finally:
            self._db = None
```

同时更新了 `get_stats()` 方法返回向量搜索状态：
```python
return {
    "documents": doc_count,
    "chunks": chunk_count,
    "by_file_type": by_type,
    "avg_chunk_size": round(avg_chunk_size, 2),
    "last_scan_at": last_scan_at,
    "vector_enabled": self._vector_enabled,  # 新增
}
```

新增 `is_vector_enabled()` 方法：
```python
def is_vector_enabled(self) -> bool:
    """Check if vector search is enabled."""
    return self._vector_enabled
```

---

### 问题 2: 错误处理不够健壮 - 向量搜索禁用无感知

**Location:** `nanobot/cli/commands.py`, `nanobot/agent/tools/rag.py`, `nanobot/agent/loop.py`

**Problem:**
- 用户不知道向量搜索是否启用
- 工具没有提供查询向量搜索状态的方法
- 进程退出时没有正确清理资源

**Fix:**

#### 2.1 更新 `SearchKnowledgeTool` 类 (`nanobot/agent/tools/rag.py`)

添加了三个新方法：
```python
def get_stats(self) -> dict[str, Any]:
    """Get statistics about the document store."""
    self._ensure_initialized()
    assert self._doc_store is not None
    return self._doc_store.get_stats()

def is_vector_enabled(self) -> bool:
    """Check if vector search is enabled."""
    self._ensure_initialized()
    assert self._doc_store is not None
    return self._doc_store.is_vector_enabled()

def close(self) -> None:
    """Close the document store connection."""
    if self._doc_store is not None:
        self._doc_store.close()
        self._doc_store = None
```

#### 2.2 更新 CLI 命令 (`nanobot/cli/commands.py`)

**rag refresh:**
- 显示向量搜索状态
- 调用 `store.close()` 关闭连接

**rag rebuild:**
- 显示向量搜索状态
- 调用 `store.close()` 关闭连接

**rag status:**
- 尝试加载 embedding provider 以检查向量搜索状态
- 显示 "Search Capabilities" 部分，包括向量搜索和全文搜索状态
- 调用 `store.close()` 关闭连接

**rag search:**
- 调用 `store.close()` 关闭连接

#### 2.3 更新 `AgentLoop` 类 (`nanobot/agent/loop.py`)

在 `stop()` 方法中添加 RAG 工具清理：
```python
def stop(self) -> None:
    """Stop the agent loop."""
    self._running = False
    logger.info("Agent loop stopping")
    # Close RAG tool if it exists
    if hasattr(self, '_retrieve_tool') and self._retrieve_tool is not None:
        try:
            self._retrieve_tool.close()
            logger.debug("RAG tool closed")
        except Exception as e:
            logger.warning("Error closing RAG tool: {}", e)
```

---

## Modified Files

1. `nanobot/rag/store.py` - 添加 close()、is_vector_enabled() 方法，更新 get_stats()
2. `nanobot/agent/tools/rag.py` - 添加 close()、is_vector_enabled() 方法
3. `nanobot/agent/loop.py` - 在 stop() 中关闭 RAG 工具
4. `nanobot/cli/commands.py` - 所有 rag 命令都调用 close() 并显示向量搜索状态

## Verification

- 所有 21 个单元测试通过: `test_rag_parser.py` (8), `test_rag_search.py` (6), `test_rag_rerank.py` (7)
- CLI 命令 `nanobot rag status` 现在显示向量搜索状态
- CLI 命令 `nanobot rag refresh` 和 `rebuild` 显示向量搜索状态
- 所有 CLI 命令正确关闭数据库连接

## Summary of Fixes

1. ✅ **数据库连接正确关闭** - 添加 close() 方法，CLI 和 AgentLoop 都正确调用
2. ✅ **向量搜索状态可见** - 用户可以清楚知道向量搜索是否启用
3. ✅ **更好的错误处理** - 关闭操作有异常处理，不会导致程序崩溃
4. ✅ **向后兼容** - 所有变更都是新增方法，不影响现有功能
