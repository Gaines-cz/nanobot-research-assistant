"""RAG Evaluation - Test data generator with LLM."""

import random
import re
from typing import List, Optional

from loguru import logger

from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.evaluation.base import EvalQuery
from nanobot.rag.store import DocumentStore


# Prompt templates for different question types (Chinese)
PROMPT_TEMPLATES_ZH = {
    "factoid": """你是一个研究助手。请根据以下文档片段，生成一个用户可能会问的事实型问题。

要求：
1. 难度：中高难度
2. 问题要自然，像真实用户提问
3. 不要直接复制文档内容
4. 使用与文档相同的语言（中文）
5. 问题答案应能从文档中找到

示例：
文档：nanobot 是一个基于 LLM 的 AI 助手框架，使用 Python 开发。
问题：nanobot 是用什么语言开发的？

文档：
{chunk_content}

生成的问题（仅输出问题本身）：""",

    "summary": """你是一个研究助手。请根据以下文档片段，生成一个用户可能会问的摘要型问题。

要求：
1. 难度：中高难度
2. 问题要自然，像真实用户提问
3. 不要直接复制文档内容
4. 使用与文档相同的语言（中文）
5. 问题答案应能从文档中找到

示例：
文档：RAG 是检索增强生成技术，结合了检索系统和生成模型的优势。
问题：RAG 技术结合了哪些系统的优势？

文档：
{chunk_content}

生成的问题（仅输出问题本身）：""",

    "analytical": """你是一个研究助手。请根据以下文档片段，生成一个用户可能会问的分析型问题。

要求：
1. 难度：中高难度
2. 问题要自然，像真实用户提问
3. 不要直接复制文档内容
4. 使用与文档相同的语言（中文）
5. 问题答案应能从文档中找到

示例：
文档：向量数据库通过嵌入向量进行语义检索，比关键词搜索更能理解语义。
问题：向量数据库为什么能更好地理解语义？

文档：
{chunk_content}

生成的问题（仅输出问题本身）：""",
}

# Prompt templates for different question types (English)
PROMPT_TEMPLATES_EN = {
    "factoid": """You are a research assistant. Generate a factoid question based on the following document.

Requirements:
1. Difficulty: medium to high
2. Natural, like a real user question
3. Don't copy the document content directly
4. Use the same language as the document (English)
5. The answer should be findable in the document

Example:
Document: nanobot is an LLM-based AI assistant framework written in Python.
Question: What programming language is nanobot written in?

Document:
{chunk_content}

Question (only output the question itself):""",

    "summary": """You are a research assistant. Generate a summary question based on the following document.

Requirements:
1. Difficulty: medium to high
2. Natural, like a real user question
3. Don't copy the document content directly
4. Use the same language as the document (English)
5. The answer should be findable in the document

Example:
Document: RAG is retrieval-augmented generation, combining the strengths of retrieval systems and generative models.
Question: What systems does RAG combine?

Document:
{chunk_content}

Question (only output the question itself):""",

    "analytical": """You are a research assistant. Generate an analytical question based on the following document.

Requirements:
1. Difficulty: medium to high
2. Natural, like a real user question
3. Don't copy the document content directly
4. Use the same language as the document (English)
5. The answer should be findable in the document

Example:
Document: Vector databases use embedded vectors for semantic search, understanding semantics better than keyword search.
Question: Why can vector databases better understand semantics?

Document:
{chunk_content}

Question (only output the question itself):""",
}

QUESTION_TYPES = ["factoid", "summary", "analytical"]


def detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English."""
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # Count English words (rough estimate)
    english_words = len(re.findall(r'[a-zA-Z]+', text))

    # If Chinese chars > 10% of English words, consider it Chinese
    if chinese_chars > english_words * 0.1:
        return "zh"
    return "en"


class DataGenerator:
    """
    Test data generator using LLM to generate realistic user queries.

    Supports generating factoid, summary, and analytical questions
    with language-aware prompts (Chinese/English).
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        llm_provider=None,  # LLM provider for generation
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.doc_store = doc_store
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    async def generate(
        self,
        num_samples: int = 50,
        min_chunk_length: int = 200,
        random_seed: Optional[int] = 42,
    ) -> List[EvalQuery]:
        """
        Generate test queries using LLM.

        Args:
            num_samples: Number of samples to generate
            min_chunk_length: Minimum chunk length to consider
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        # Get chunks from database - prefer large chunks for dual granularity
        db = self.doc_store.connection.db
        cursor = db.execute("""
            SELECT c.id, c.content, d.path, d.filename
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE LENGTH(c.content) >= ?
              AND (c.granularity = 'large' OR c.granularity IS NULL)
        """, (min_chunk_length,))

        all_chunks = cursor.fetchall()

        if not all_chunks:
            return []

        # Filter by content quality first
        good_chunks = []
        for chunk_id, content, doc_path, doc_filename in all_chunks:
            cleaned_content = self._clean_chunk_content(content)
            if self._is_good_quality_content(cleaned_content):
                good_chunks.append((chunk_id, cleaned_content, doc_path, doc_filename))

        if not good_chunks:
            logger.warning("No good quality chunks found after filtering")
            return []

        logger.info("Selected {}/{} chunks after quality filtering", len(good_chunks), len(all_chunks))

        # Shuffle and take num_samples
        random.shuffle(good_chunks)
        chunks = good_chunks[:num_samples]

        queries: List[EvalQuery] = []

        for idx, (chunk_id, content, doc_path, doc_filename) in enumerate(chunks):
            # Detect language and select appropriate question type
            lang = detect_language(content)
            question_type = random.choice(QUESTION_TYPES)

            # Generate query using LLM or fallback to basic method
            if self.llm_provider:
                query_text = await self._generate_with_llm(content, question_type, lang)
            else:
                query_text = self._generate_fallback(content)

            queries.append(EvalQuery(
                id=f"q_{idx}",
                query=query_text,
                golden_context=content,
                source_chunk_id=chunk_id,
                source_doc=doc_path,
                question_type=question_type,
                tags=[],
                golden_embedding=None,
            ))

            if (idx + 1) % 10 == 0:
                logger.info("Generated {}/{} queries", idx + 1, len(chunks))

        logger.info("Generated {} test queries", len(queries))
        return queries

    async def _generate_with_llm(
        self,
        content: str,
        question_type: str,
        lang: str,
    ) -> str:
        """Generate query using LLM."""
        templates = PROMPT_TEMPLATES_ZH if lang == "zh" else PROMPT_TEMPLATES_EN
        prompt = templates[question_type].format(chunk_content=content)

        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_provider.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )

        if response.content:
            # Clean up the response
            query = response.content.strip()
            # Remove quotes if present
            query = re.sub(r'^["\'](.*)["\']$', r'\1', query)
            return query

        return self._generate_fallback(content)

    def _generate_fallback(self, content: str) -> str:
        """Fallback query generation when LLM is not available."""
        # Preprocess: fix hyphenated word breaks
        content = self._fix_hyphenated_breaks(content)

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content.strip())
        if not paragraphs:
            return content[:100]

        # Use middle paragraph
        mid_idx = len(paragraphs) // 2
        target_para = paragraphs[mid_idx].strip()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', target_para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Pick a sentence from the middle
            sent_idx = len(sentences) // 2
            selected = sentences[sent_idx]
            # Clean up
            selected = re.sub(r'[.!?]+$', '', selected).strip()
            if 10 <= len(selected) <= 150:
                return selected

        # Fallback: clean slice of paragraph
        clean_para = re.sub(r'\s+', ' ', target_para).strip()
        return clean_para[:120]

    @staticmethod
    def _fix_hyphenated_breaks(text: str) -> str:
        """Fix hyphenated word breaks from PDF/text extraction."""
        text = re.sub(r'(\w+)-[\r\n\t]+(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-[\r\n\t ]+(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _clean_chunk_content(content: str) -> str:
        """Clean chunk content by removing PDF parsing noise."""
        # 1. Remove isolated number lines (table data residue)
        content = re.sub(r'^\s*\d+(\.\d+)?\s*$', '', content, flags=re.MULTILINE)

        # 2. Fix hyphenated word breaks
        content = re.sub(r'(\w+)-[\r\n\t]+(\w+)', r'\1\2', content)
        content = re.sub(r'(\w+)-[\r\n\t ]+(\w+)', r'\1\2', content)

        # 3. Remove excessive empty lines (more than 2 consecutive)
        content = re.sub(r'\n\s*\n\s*\n\s*\n', '\n\n\n', content)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # 4. Remove lines that are just punctuation/symbols
        content = re.sub(r'^\s*[^\w\s\u4e00-\u9fff]+\s*$', '', content, flags=re.MULTILINE)

        # 5. Normalize whitespace
        content = re.sub(r'[ \t]+', ' ', content)

        return content.strip()

    @staticmethod
    def _is_good_quality_content(content: str) -> bool:
        """Check if content is good quality for test generation."""
        if not content or len(content.strip()) < 100:
            return False

        # 1. Check valid text ratio (exclude mostly numbers/symbols)
        total_chars = len(content)
        text_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', content))
        if total_chars > 0 and text_chars / total_chars < 0.4:
            return False

        # 2. Check sentence count (at least 2 meaningful sentences)
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 30]
        if len(meaningful_sentences) < 2:
            return False

        # 3. Check for references/index/table of contents (skip these)
        start_content = content[:300].lower()
        skip_keywords = [
            'references', 'bibliography', 'index', 'table of contents',
            'contents', 'figures', 'tables', 'appendix',
            '参考文献', '目录', '索引', '附表', '附图'
        ]
        for keyword in skip_keywords:
            if keyword in start_content:
                return False

        # 4. Check if it's mostly citations (e.g., [1], [2], etc.)
        citation_markers = len(re.findall(r'\[\d+\]', content))
        if citation_markers > 10 and len(content) < 500:
            return False

        return True

    async def precompute_embeddings(
        self,
        queries: List[EvalQuery],
        batch_size: int = 4,
    ) -> List[EvalQuery]:
        """Precompute golden_embedding for queries."""
        if not self.embedding_provider:
            logger.warning("No embedding provider, skipping precompute")
            return queries

        all_embeddings = []
        contents = [q.golden_context for q in queries]

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            logger.info("Precomputing embeddings batch {}/{}",
                       i // batch_size + 1, (len(contents) + batch_size - 1) // batch_size)
            batch_embeddings = await self.embedding_provider.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        for query, emb in zip(queries, all_embeddings):
            query.golden_embedding = emb

        logger.info("Precomputed embeddings for {} queries", len(queries))
        return queries
