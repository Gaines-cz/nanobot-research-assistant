# RAG 模块改动报告

**Date:** 2026-02-28

---

## 一、Bug 修复

| Issue | 问题 | 修复位置 | 状态 |
|-------|------|---------|------|
| Issue 1 | `_step2_context_expansion` 只取单 chunk | `store.py:761-781` | ✅ |
| Issue 2 | `rag.py` 每次启动删除数据库 | `rag.py:75` | ✅ |
| Issue 3 | `chunk_with_section_awareness` overlap 逻辑错误 | `parser.py:449` | ✅ |
| Issue 4 | SECTION_PATTERNS 硬编码章节编号 | `parser.py:29-69` | ✅ |
| Issue 5 | `rerank_and_dedup` 注释误导 | `rerank.py:237` | ✅ |

---

## 二、模型升级

| 组件 | 原模型 | 新模型 | 说明 |
|-----|-------|-------|------|
| 召回 | `all-MiniLM-L6-v2` (384维) | `BAAI/bge-m3` (1024维) | 多语言、科研优化 |
| 重排 | `ms-marco-MiniLM-L-6-v2` | `BAAI/bge-reranker-v2-m3` | 多语言重排 |

**影响**: 需要重建索引 (`nanobot rag rebuild`)

---

## 三、新增功能

### 3.1 PDF 解析升级
- **文件**: `nanobot/rag/parser.py`
- **改动**: PyMuPDF 优先，pypdf fallback
- **优势**: 更好的布局保持、阅读顺序

### 3.2 Query 扩展
- **新文件**: `nanobot/rag/query.py`
- **功能**: 30+ 科研缩写自动扩展
- **示例**: `LLM` → `LLM large language model large language models`

---

## 四、配置变更

**文件**: `nanobot/config/schema.py`

```python
# 新增配置
enable_query_expand: bool = True      # Query 扩展开关
pdf_parser: str = "pymupdf"           # PDF 解析器选择

# 模型变更
embedding_model: str = "BAAI/bge-m3"
rerank_model: str = "BAAI/bge-reranker-v2-m3"
```

---

## 五、文件变更清单

| 文件 | 变更类型 | 说明 |
|-----|---------|------|
| `nanobot/rag/query.py` | 新增 | Query 扩展模块 |
| `nanobot/rag/store.py` | 修改 | 集成 QueryExpander、修复 RerankService |
| `nanobot/rag/parser.py` | 修改 | PyMuPDF 支持、SECTION_PATTERNS 通用化 |
| `nanobot/rag/rerank.py` | 修改 | 注释修正 |
| `nanobot/config/schema.py` | 修改 | 新配置项、模型升级 |
| `nanobot/agent/tools/rag.py` | 修改 | 删除删库逻辑 |

---

## 六、使用说明

```bash
# 1. 安装新依赖（可选，用于更好的 PDF 解析）
pip install pymupdf

# 2. 重建索引（向量维度变了）
nanobot rag rebuild

# 3. 测试搜索
nanobot rag search "transformer attention"
```

---

## 七、未实现项（后续可选）

| 功能 | 优先级 | 说明 |
|-----|-------|------|
| Issue 7: 文档级核心内容提取 | P2 | 功能增强 |
| arxiv 元数据提取 | P2 | 科研增强 |
| 引文关系构建 | P2 | 科研增强 |

---

**总结**: 所有 P0/P1 bug 已修复，模型已升级为 BGE-M3 系列，新增 Query 扩展和 PyMuPDF 支持。