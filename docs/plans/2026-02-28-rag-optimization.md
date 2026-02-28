# RAG 优化方案

**Date:** 2026-02-28
**Status:** Draft

---

## 背景

nanobot 定位为**本地个人科研助手**，RAG 模块是其核心功能之一。当前实现已具备：
- 混合检索 (Vector + BM25)
- 语义切分 (Section-aware chunking)
- Cross-Encoder 重排序
- 上下文扩展

本文档从**科研场景需求**和**RAG 专业性**角度，提出优化方向。

---

## 一、高优先级优化

### 1.1 Query 理解与改写

**问题**: 科研场景的查询往往模糊、口语化或不完整。

**现状**:
```python
# 当前: 直接用原始 query 检索
results = await store.search_advanced(query)
```

**典型场景**:
| 用户输入 | 期望改写 |
|---------|---------|
| "那篇讲 attention 的论文" | "transformer attention mechanism" |
| "怎么做实验验证" | "experimental setup methodology validation" |
| "跟 BERT 比怎么样" | "comparison BERT performance benchmark" |

**方案**:

```python
class QueryRewriter:
    """Query 改写器，使用 LLM 或规则增强查询。"""

    REWRITE_PROMPT = """你是一个科研助手。用户正在搜索论文库。
请将用户的查询改写为更适合检索的形式：
1. 补充专业术语
2. 移除口语化表达
3. 添加相关同义词

用户查询: {query}
改写结果:"""

    def __init__(self, llm_provider, enable: bool = True):
        self.llm = llm_provider
        self.enable = enable

    async def rewrite(self, query: str) -> str:
        if not self.enable:
            return query

        # 简单规则增强（无需 LLM）
        query = self._expand_abbreviations(query)

        # LLM 改写（可选）
        if self.llm:
            rewritten = await self._llm_rewrite(query)
            return rewritten

        return query

    def _expand_abbreviations(self, query: str) -> str:
        """展开常见缩写。"""
        ABBREVIATIONS = {
            "NN": "neural network",
            "DL": "deep learning",
            "ML": "machine learning",
            "NLP": "natural language processing",
            "LLM": "large language model",
            "CV": "computer vision",
            "RL": "reinforcement learning",
            "GAN": "generative adversarial network",
            "Transformer": "transformer attention",
        }
        for abbr, full in ABBREVIATIONS.items():
            if abbr.lower() in query.lower() and full.lower() not in query.lower():
                query = query + f" ({full})"
        return query
```

**配置**:
```python
class RAGConfig(Base):
    enable_query_rewrite: bool = True
    query_rewrite_mode: str = "rule"  # "rule" | "llm" | "both"
```

---

### 1.2 PDF 解析质量提升

**问题**: 当前使用 `pypdf` 简单提取文本，丢失重要信息。

**现状问题**:
- 表格结构丢失
- 数学公式乱码或丢失
- 双栏排版顺序错乱
- 图片 caption 未关联

**方案 A: 升级到 PyMuPDF (fitz)**

```python
@staticmethod
def parse_pdf(path: Path) -> str:
    """使用 PyMuPDF 解析 PDF，支持更好的布局保持。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required. Install: pip install pymupdf")

    doc = fitz.open(path)
    parts: list[str] = []

    for page_num, page in enumerate(doc):
        # 提取文本块（保持阅读顺序）
        blocks = page.get_text("blocks", sort=True)

        for block in blocks:
            text = block[4].strip()
            if text:
                parts.append(text)

        # 提取图片信息
        images = page.get_images()
        for img in images:
            # 可以添加图片描述占位符
            parts.append(f"[Figure on page {page_num + 1}]")

    return "\n\n".join(parts)
```

**方案 B: 支持 Nougat/Marker（科研论文专用）**

```python
@staticmethod
def parse_pdf_with_nougat(path: Path) -> str:
    """使用 Nougat 解析学术论文，保留公式和表格。"""
    # Nougat 是 Meta 开源的学术论文解析器
    # 能识别 LaTeX 公式、表格结构
    # 需要额外安装: pip install nougat-ocr
    pass
```

**建议**: 先实现方案 A（PyMuPDF），后续可选支持 Nougat。

---

### 1.3 Hybrid Search 权重可配置

**问题**: 当前 RRF fusion 使用固定参数，无法针对不同场景调优。

**现状**:
```python
k = 60  # 硬编码
rrf_scores[key] = 1.0 / (k + rank)
```

**方案**:

```python
class RAGConfig(Base):
    # Hybrid search 权重
    vector_weight: float = 0.6      # 向量检索权重
    bm25_weight: float = 0.4        # BM25 权重
    rrf_k: int = 60                 # RRF 参数

# store.py 中
def _rrf_fusion(self, vector_results, bm25_results) -> list[SearchResult]:
    k = self.config.rrf_k
    rrf_scores: dict[str, float] = {}

    for rank, result in enumerate(vector_results, 1):
        key = f"{result.path}:{result.chunk_index}"
        rrf_scores[key] = self.config.vector_weight / (k + rank)

    for rank, result in enumerate(bm25_results, 1):
        key = f"{result.path}:{result.chunk_index}"
        score = self.config.bm25_weight / (k + rank)
        if key in rrf_scores:
            rrf_scores[key] += score
        else:
            rrf_scores[key] = score

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 二、中优先级优化

### 2.1 Chunk 类型加权检索

**问题**: 已存储 `chunk_type` 但检索时未利用。

**方案**: 根据查询意图对不同类型 chunk 加权。

```python
class QueryIntentAnalyzer:
    """分析查询意图，确定应该关注哪些章节。"""

    INTENT_PATTERNS = {
        "method": [
            r"how (to|does|do)",
            r"method|approach|algorithm",
            r"怎么|如何|方法|算法",
        ],
        "experiment": [
            r"experiment|evaluation|setup",
            r"实验|评估|设置",
        ],
        "result": [
            r"result|performance|accuracy",
            r"结果|性能|准确率",
        ],
        "conclusion": [
            r"conclusion|summary|contribution",
            r"结论|总结|贡献",
        ],
        "introduction": [
            r"background|related work|previous",
            r"背景|相关工作|之前",
        ],
    }

    def analyze(self, query: str) -> dict[str, float]:
        """返回各 chunk_type 的权重 boost。"""
        boosts = {ct: 1.0 for ct in ["abstract", "introduction", "method",
                                      "experiment", "result", "conclusion", "other"]}

        query_lower = query.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(p, query_lower) for p in patterns):
                boosts[intent] = 1.5  # 匹配意图的类型加权

        return boosts
```

**在检索时应用**:
```python
async def _vector_search(self, query: str, top_k: int, boosts: dict = None) -> list[SearchResult]:
    # ... 获取结果后
    for result in results:
        chunk_type = self._get_chunk_type(result)
        if boosts and chunk_type in boosts:
            result.score *= boosts[chunk_type]

    return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
```

---

### 2.2 检索结果缓存

**问题**: 重复查询每次都重新计算。

**方案**:

```python
from functools import lru_cache
import hashlib

class DocumentStore:
    def __init__(self, ...):
        # ...
        self._search_cache: dict[str, tuple[float, list]] = {}
        self._cache_ttl: int = 300  # 5分钟

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    async def search_advanced(self, query: str) -> list[SearchResultWithContext]:
        cache_key = self._cache_key(query)

        # 检查缓存
        if cache_key in self._search_cache:
            cached_time, cached_results = self._search_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_results

        # 执行检索
        results = await self._do_search_advanced(query)

        # 存入缓存
        self._search_cache[cache_key] = (time.time(), results)
        return results
```

---

### 2.3 增量索引优化

**问题**: 当前仅用 mtime 判断文件变更，可能遗漏情况。

**方案**: 添加 content hash 校验。

```python
import hashlib

def _compute_file_hash(self, path: Path) -> str:
    """计算文件内容 hash。"""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]

async def scan_and_index(self, docs_dir: Path) -> dict[str, int]:
    # ...
    for file_path in docs_dir.rglob("*"):
        # ...
        content_hash = self._compute_file_hash(file_path)

        if abs_path in known_docs:
            stored_hash = known_docs[abs_path].get("content_hash")
            if stored_hash == content_hash:
                continue  # 内容未变，跳过

        # ... 索引文件
        # 存储 hash
        db.execute("UPDATE documents SET content_hash = ? WHERE path = ?",
                   (content_hash, abs_path))
```

---

## 三、科研场景特有优化

### 3.1 论文元数据增强

**问题**: arxiv 论文有丰富的元数据未利用。

**方案**: 从文件名/内容提取 arxiv ID，获取元数据。

```python
class ArxivMetadataExtractor:
    """从 arxiv 论文提取元数据。"""

    ARXIV_ID_PATTERN = r"(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+/\d{7})"

    def extract_from_filename(self, filename: str) -> dict | None:
        """从文件名提取 arxiv ID。"""
        match = re.search(self.ARXIV_ID_PATTERN, filename)
        if match:
            return self._fetch_metadata(match.group(1))
        return None

    def _fetch_metadata(self, arxiv_id: str) -> dict:
        """调用 arxiv API 获取元数据。"""
        import arxiv
        try:
            paper = next(arxiv.Search(id_list=[arxiv_id]).results())
            return {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": paper.published.isoformat(),
                "categories": paper.categories,
                "abstract": paper.summary,
                "arxiv_id": arxiv_id,
            }
        except Exception:
            return {}
```

---

### 3.2 引文关系构建

**问题**: 论文之间的引用关系未利用。

**方案**: 解析 References 章节，提取引用关系。

```python
class CitationExtractor:
    """从论文中提取引用关系。"""

    def extract_citations(self, content: str) -> list[str]:
        """提取 References 中的 arxiv ID / DOI。"""
        citations = []

        # 找到 References 章节
        refs_start = re.search(r"\n(References|参考文献)\s*\n", content, re.IGNORECASE)
        if not refs_start:
            return citations

        refs_content = content[refs_start.end():]

        # 提取 arxiv ID
        arxiv_pattern = r"arXiv[:\s]+(\d{4}\.\d{4,5})"
        citations.extend(re.findall(arxiv_pattern, refs_content))

        # 提取 DOI
        doi_pattern = r"doi[:\s]+(10\.\d{4,}/[^\s]+)"
        citations.extend(re.findall(doi_pattern, refs_content))

        return citations
```

**存储结构**:
```sql
CREATE TABLE citations (
    source_doc_id INTEGER,
    target_arxiv_id TEXT,
    target_doi TEXT,
    FOREIGN KEY (source_doc_id) REFERENCES documents(id)
);
```

---

### 3.3 术语词典

**问题**: 科研术语缩写未展开，影响检索。

**方案**: 建立术语词典，查询时自动扩展。

```python
SCIENTIFIC_TERMS = {
    # 缩写 -> 全称
    "NN": ["neural network", "neural networks"],
    "CNN": ["convolutional neural network", "卷积神经网络"],
    "RNN": ["recurrent neural network", "循环神经网络"],
    "LSTM": ["long short-term memory"],
    "GAN": ["generative adversarial network", "生成对抗网络"],
    "VAE": ["variational autoencoder"],
    "BERT": ["bidirectional encoder representations from transformers"],
    "GPT": ["generative pre-trained transformer"],
    "LLM": ["large language model", "大语言模型"],
    "NLP": ["natural language processing", "自然语言处理"],
    "CV": ["computer vision", "计算机视觉"],
    "RL": ["reinforcement learning", "强化学习"],
    # 中文术语
    "注意力": ["attention", "attention mechanism"],
    "嵌入": ["embedding", "representation"],
    "微调": ["fine-tuning", "fine tuning"],
}

def expand_query_terms(query: str) -> str:
    """扩展查询中的术语。"""
    for abbr, expansions in SCIENTIFIC_TERMS.items():
        if abbr.lower() in query.lower():
            # 添加最常见的扩展
            query += f" {expansions[0]}"
    return query
```

---

## 四、实现计划

### Phase 1: 核心优化（预计 2 周）

| 任务 | 工作量 | 依赖 |
|-----|-------|-----|
| Query 改写（规则版） | 4h | 无 |
| Hybrid 权重可配置 | 2h | 无 |
| 检索缓存 | 2h | 无 |
| PDF 解析升级到 PyMuPDF | 4h | 无 |
| Chunk 类型加权 | 4h | 无 |

### Phase 2: 科研增强（预计 1 周）

| 任务 | 工作量 | 依赖 |
|-----|-------|-----|
| arxiv 元数据提取 | 4h | 无 |
| 引文关系构建 | 6h | 元数据提取 |
| 术语词典 | 2h | 无 |

### Phase 3: 可选增强（按需）

| 任务 | 工作量 | 依赖 |
|-----|-------|-----|
| Query LLM 改写 | 4h | LLM 接口 |
| Nougat/Marker 支持 | 8h | GPU（可选）|
| 检索质量评估 | 4h | 测试集 |

---

## 五、配置汇总

```python
class RAGConfig(Base):
    # === 现有配置 ===
    enabled: bool = True
    chunk_strategy: str = "phase2b"
    # ...

    # === 新增配置 ===

    # Query 改写
    enable_query_rewrite: bool = True
    query_rewrite_mode: str = "rule"  # "rule" | "llm" | "both"

    # Hybrid 权重
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60

    # Chunk 加权
    enable_chunk_type_boost: bool = True

    # 缓存
    enable_search_cache: bool = True
    cache_ttl_seconds: int = 300

    # PDF 解析
    pdf_parser: str = "pymupdf"  # "pypdf" | "pymupdf" | "nougat"

    # 科研增强
    enable_arxiv_metadata: bool = True
    enable_citation_extraction: bool = True
    enable_term_expansion: bool = True
```

---

## 六、验收标准

1. **Query 改写**: 用户输入口语化查询，能检索到正确结果
2. **PDF 解析**: 能正确提取双栏论文的段落顺序
3. **Hybrid 权重**: 可通过配置调整向量/BM25 权重
4. **检索缓存**: 重复查询响应时间 < 10ms
5. **元数据提取**: 能从 arxiv 论文文件名提取标题、作者

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| PyMuPDF 依赖冲突 | 中 | 保留 pypdf 作为 fallback |
| Query 改写引入噪音 | 中 | 提供开关，可关闭 |
| 缓存内存占用 | 低 | 设置 TTL 和大小限制 |
| arxiv API 限流 | 低 | 添加重试和本地缓存 |