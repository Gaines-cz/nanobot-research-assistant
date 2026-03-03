# RAG 模块重构：DocumentStore 拆分

**创建日期**: 2026-03-01
**类型**: 代码重构
**影响范围**: RAG 模块

---

## 一、重构原因

`nanobot/rag/store.py` 文件过大（1214 行），职责过多：
- 数据库连接管理
- 文档索引
- 向量搜索
- 全文搜索
- 高级搜索流程
- 缓存管理

这导致代码难以维护、测试和理解。

---

## 二、重构方案

将 `store.py` 拆分为多个职责单一的模块：

| 新文件 | 职责 | 行数 |
|--------|------|------|
| `cache.py` | 搜索缓存管理（LRU + TTL） | ~100 行 |
| `connection.py` | 数据库连接、schema 初始化 | ~200 行 |
| `indexer.py` | 文档索引（add/update/delete/scan） | ~250 行 |
| `search.py` | 所有搜索功能（向量、全文、高级） | ~500 行 |
| `store.py` | Facade 主类（协调各模块） | ~150 行 |

---

## 三、模块职责

### 3.1 `cache.py` - 搜索缓存管理

```python
class SearchCache[T]:
    """泛型搜索缓存，支持 LRU  eviction 和 TTL 过期"""

class SearchCacheManager:
    """管理多个搜索缓存（advanced + basic）"""
```

### 3.2 `connection.py` - 数据库连接

```python
class DatabaseConnection:
    """
    数据库连接管理：
    - 连接创建和管理
    - sqlite-vec 扩展加载
    - Schema 初始化（向后兼容）
    - FTS5 triggers 设置
    """
```

### 3.3 `indexer.py` - 文档索引

```python
class DocumentIndexer:
    """
    文档索引操作：
    - 扫描目录查找文档
    - 添加新文档
    - 更新已修改文档
    - 删除已移除文档
    - 批量索引更新
    """
```

### 3.4 `search.py` - 搜索功能

```python
class DocumentSearch:
    """
    所有搜索操作：
    - 向量搜索（相似度）
    - 全文搜索（FTS5）
    - 混合搜索（RRF fusion）
    - 高级多步搜索流程
    """

# 数据类（从 store.py 迁移）
@dataclass
class SearchResult: ...

@dataclass
class DocumentInfo: ...

@dataclass
class ChunkInfo: ...

@dataclass
class SearchResultWithContext: ...
```

### 3.5 `store.py` - Facade 主类

```python
class DocumentStore:
    """
    统一的文档存储门面类。

    委托给：
    - DatabaseConnection: 连接管理和 schema
    - DocumentIndexer: 文档索引操作
    - DocumentSearch: 搜索操作
    """
```

---

## 四、公共 API 变更

### 保持不变（向后兼容）

`DocumentStore` 的公共 API 保持不变：

```python
# 创建实例
store = DocumentStore(db_path, embedding_provider, config)

# 索引操作
await store.scan_and_index(docs_dir, chunk_size, chunk_overlap_ratio)
await store.schedule_index_update(docs_dir, chunk_size, chunk_overlap_ratio)

# 搜索操作
results = await store.search(query, top_k)
results = await store.search_advanced(query)

# 统计
stats = store.get_stats()
vector_enabled = store.is_vector_enabled()

# 连接管理
store.close()
```

### 内部变更

- 数据类（`SearchResult`, `DocumentInfo`, `ChunkInfo`, `SearchResultWithContext`）现在从 `nanobot.rag.search` 导出
- 缓存管理现在由 `SearchCache` 和 `SearchCacheManager` 处理

---

## 五、文件清单

| 操作 | 文件 |
|------|------|
| 新增 | `nanobot/rag/cache.py` |
| 新增 | `nanobot/rag/connection.py` |
| 新增 | `nanobot/rag/indexer.py` |
| 新增 | `nanobot/rag/search.py` |
| 修改 | `nanobot/rag/store.py` (重构为 Facade) |
| 不变 | `nanobot/rag/__init__.py` (导出保持不变) |

---

## 六、验证

### 模块导入验证

```python
from nanobot.rag import DocumentStore, SearchResult, SearchResultWithContext
from nanobot.rag.cache import SearchCache, SearchCacheManager
from nanobot.rag.connection import DatabaseConnection
from nanobot.rag.indexer import DocumentIndexer
from nanobot.rag.search import DocumentSearch
# 所有模块导入成功
```

### 功能验证

```python
# 创建实例
store = DocumentStore(db_path, embedding_provider, config)

# 索引功能
stats = await store.scan_and_index(docs_dir)

# 搜索功能
results = await store.search("query")
advanced_results = await store.search_advanced("query")

# 统计
stats = store.get_stats()
```

---

## 七、检查中发现并修复的问题

### 第 2 遍检查时发现的问题：

1. **`search.py` 缺少 `sqlite3` 导入**
   - 问题：代码使用了 `sqlite3.DatabaseError` 但未导入 `sqlite3`
   - 修复：在文件开头添加 `import sqlite3`

2. **`search.py` 中 `SearchCache.MAX_CACHE_SIZE` 引用不存在**
   - 问题：代码使用了不存在的常量
   - 修复：直接使用固定值 `max_size=1000, ttl_seconds=self.config.cache_ttl_seconds`

### 第 3 遍检查时的验证项目：

- [x] RAG 模块所有导出项验证
- [x] Agent 模块所有导出项验证
- [x] `MemoryStore.consolidate` 方法签名验证（包含 `use_rag` 参数）
- [x] CLI 模块导入验证
- [x] `RAGConfig` 默认值验证
- [x] `DocumentStore` 公共 API 完整性验证
- [x] `MemoryStore` 公共 API 完整性验证

---

## 八、重构后功能变化说明

### MemoryStore 合并

| 项目 | 重构前 | 重构后 |
|------|--------|--------|
| 类数量 | 2 个 (`MemoryStore`, `MemoryStoreOptimized`) | 1 个 (`MemoryStore`) |
| 使用方式 | 创建不同的类实例 | 单一类，`use_rag` 参数选择策略 |
| 功能 | 分离 | 完整保留，无功能变化 |

### DocumentStore 拆分

| 项目 | 重构前 | 重构后 |
|------|--------|--------|
| 文件数量 | 1 个 (`store.py`, 1214 行) | 5 个 (~1200 行总计) |
| 公共 API | `DocumentStore` | 保持不变 |
| 功能 | 完整 | 完整保留，无功能变化 |

### 总结：**重构后功能无任何变化，完全向后兼容**

---

## 九、收益

1. **代码可读性**: 每个文件职责单一，易于理解
2. **可维护性**: 修改某个功能时影响范围清晰
3. **可测试性**: 各模块可以独立测试
4. **可扩展性**: 新增功能时可以在相应模块中添加

---

## 十、后续工作

1. 为各子模块添加单元测试
2. 考虑将 `search.py` 进一步拆分为 `vector.py`, `fulltext.py`, `hybrid.py`
3. 优化缓存策略（如支持分布式缓存）
