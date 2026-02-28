# RAG CLI 命令 Bug 修复

**Date:** 2026-02-28

## Overview

修复了 `nanobot rag` CLI 命令中的三个实际影响功能的 bug。

## Issues Fixed

### 🟡 P1 Medium

#### Issue 1: `rag search` 缺少 `rag_config` 参数

**Location:** `nanobot/cli/commands.py` line 1236

**Problem:**
```python
store = DocumentStore(db_path, embedding_provider)  # 缺少 rag_config!
```

对比 `rag_refresh` (line 1035) 和 `rag_rebuild` (line 1101) 都传了 `rag_config`。

**影响:**
- `DocumentStore.__init__` 中的 `_rerank_service` 初始化依赖 `self.config.enable_rerank`
- 如果不传 config，会用默认值 `RAGConfig()`，可能导致 rerank 不生效
- 配置文件中的设置（如 `enableRerank`, `rerankThreshold` 等）被忽略

**Fix:**
```python
store = DocumentStore(db_path, embedding_provider, rag_config)
```

---

#### Issue 2: `rag search` 用的是 `search()` 而不是 `search_advanced()`

**Location:** `nanobot/cli/commands.py` line 1245

**Problem:**
```python
return await store.search(query, top_k=top_k)  # 应该用 search_advanced!
```

**影响:**
- 没有用上 Phase 2-4 的优化（语义切分、上下文扩展、文档级优先、Cross-Encoder rerank）
- 只用了基础的混合搜索
- 结果质量不如预期

**Fix:**
```python
# 先用 search_advanced()
results = await store.search_advanced(query)
# 然后格式化输出
```

同时更新了结果展示逻辑，因为 `search_advanced()` 返回 `list[SearchResultWithContext]`：
- 使用 `result.combined_content` 而不是 `result.content`
- 使用 `result.final_score` 而不是 `result.score`
- 显示文档标题 `result.document.title or result.document.filename`
- 如果有 section title 也显示出来
- 支持 `--top-k` 参数来限制显示的结果数量

---

#### Issue 3: `rag status` 统计文件数不准确

**Location:** `nanobot/cli/commands.py` lines 1156-1160

**Problem:**
```python
count = sum(1 for _ in docs_dir.rglob("*") if _.is_file() and not _.name.startswith("."))
```

这会统计所有文件，包括临时文件、系统文件等，而不只是支持的文档类型。

**影响:**
- 显示的文件数可能比实际索引的多
- 用户困惑

**Fix:**
只统计支持的文件类型：
```python
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".docx", ".doc", ".txt"}
count = sum(1 for _ in docs_dir.rglob("*")
              if _.is_file() and not _.name.startswith(".")
              and _.suffix.lower() in SUPPORTED_EXTENSIONS)
```

---

## Modified Files

1. `nanobot/cli/commands.py` - 修复上述 3 个问题

## Verification Steps

1. 运行 `nanobot rag status` - 确认文件统计正确
2. 运行 `nanobot rag refresh` - 确认索引正常
3. 运行 `nanobot rag search "query"` - 确认用的是高级搜索，结果展示正确

## Summary

1. ✅ **`rag search` 现在使用配置文件中的 RAG 配置** - rerank 等设置现在会生效
2. ✅ **`rag search` 使用 `search_advanced()`** - 结果质量大幅提升，支持语义切分、上下文扩展、rerank
3. ✅ **`rag status` 文件统计准确** - 只统计支持的文档类型
