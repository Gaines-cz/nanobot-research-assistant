"""RAG Evaluation - Test data generator (basic + LLM-enhanced)."""

import random
import re
from typing import List, Optional

from loguru import logger

from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.evaluation.base import EvalQuery
from nanobot.rag.store import DocumentStore


class DataGenerator:
    """
    Test data generator for RAG evaluation.

    Supports two modes:
    - Basic: Extract from middle of chunk + keyword-to-question (zero cost)
    - LLM-enhanced: LLM generates real questions (higher quality, optional cost)
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.doc_store = doc_store
        self.embedding_provider = embedding_provider

    def generate_basic(
        self,
        num_samples: int = 50,
        min_chunk_length: int = 200,
        random_seed: Optional[int] = 42,
    ) -> List[EvalQuery]:
        """
        Generate basic test data (zero cost).

        Strategy to avoid inflated metrics:
        1. Prefer middle paragraph of chunk (not the beginning, which often has high similarity)
        2. Extract keywords and form simple questions

        Args:
            num_samples: Number of samples to generate
            min_chunk_length: Minimum chunk length to consider
            random_seed: Random seed for reproducibility
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Get chunks from database - prefer large chunks for dual granularity
        # Note: We don't use ORDER BY RANDOM() here because it doesn't respect Python's random.seed()
        db = self.doc_store.connection.db
        cursor = db.execute("""
            SELECT c.id, c.content, d.path, d.filename
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE LENGTH(c.content) >= ?
              AND (c.granularity = 'large' OR c.granularity IS NULL)
        """, (min_chunk_length,))

        all_chunks = cursor.fetchall()

        # Shuffle in Python (respects random_seed) and take top num_samples
        if not all_chunks:
            chunks = []
        else:
            random.shuffle(all_chunks)
            chunks = all_chunks[:num_samples]

        queries: List[EvalQuery] = []

        for idx, (chunk_id, content, doc_path, doc_filename) in enumerate(chunks):
            # Generate query from chunk (improved: use middle paragraph)
            query_text = self._generate_query_from_chunk_basic(content)

            queries.append(EvalQuery(
                id=f"q_{idx}",
                query=query_text,
                golden_context=content,
                source_chunk_id=chunk_id,
                source_doc=doc_path,
                golden_embedding=None,  # Will be set later by precompute_embeddings()
            ))

        logger.info("Generated {} basic test queries", len(queries))
        return queries

    def _generate_query_from_chunk_basic(self, content: str) -> str:
        """
        Generate query from chunk (improved basic version).

        Strategy:
        1. Split by paragraphs
        2. Use middle paragraph (not beginning)
        3. Extract a meaningful phrase or use sentence directly as query
        4. Avoid "What is <complete sentence>?" awkward format
        5. Prefer noun phrases and natural search queries
        """
        # Preprocess: fix hyphenated word breaks (like "limi-\ntations" → "limitations")
        content = self._fix_hyphenated_breaks(content)

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content.strip())
        if not paragraphs:
            return content[:100]

        # Prefer middle paragraph (avoid beginning)
        mid_idx = len(paragraphs) // 2
        target_para = paragraphs[mid_idx].strip()

        # Split into sentences (handle . ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', target_para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Pick a sentence from the middle
            sent_idx = len(sentences) // 2
            selected_sentence = sentences[sent_idx]

            # Clean up trailing punctuation
            selected_sentence = re.sub(r'[.!?]+$', '', selected_sentence).strip()

            # If it's already a question, use as-is
            if selected_sentence.endswith('?'):
                return selected_sentence

            # If reasonable length, try to extract a noun phrase or use as-is
            if 15 <= len(selected_sentence) <= 150:
                # Try to rephrase sentences that start with "This is"/"It is"/etc.
                lowered = selected_sentence.lower()
                if lowered.startswith(('this is', 'it is', 'that is')):
                    # Extract the main part
                    parts = selected_sentence.split(' ', 2)
                    if len(parts) > 2:
                        rest = parts[2]
                        # If the rest is short enough, make it a question
                        if len(rest) <= 100:
                            return f"What is {rest}?"
                        # Otherwise just use the rest
                        return rest
                elif lowered.startswith(('there is', 'there are')):
                    parts = selected_sentence.split(' ', 2)
                    if len(parts) > 2:
                        rest = parts[2]
                        if len(rest) <= 100:
                            if lowered.startswith('there is'):
                                return f"What is {rest}?"
                            else:
                                return f"What are {rest}?"
                        return rest

                # For normal sentences, try to extract a meaningful phrase first
                phrase = self._extract_meaningful_phrase(selected_sentence)
                if phrase:
                    return phrase

                # If no good phrase found, use the sentence as-is (it works fine for search)
                return selected_sentence

        # Fallback: extract meaningful word phrase
        phrase = self._extract_meaningful_phrase(target_para)
        if phrase:
            return phrase

        # Last resort: use a clean slice
        clean_para = re.sub(r'\s+', ' ', target_para).strip()
        return clean_para[:120]

    def _extract_meaningful_phrase(self, text: str) -> Optional[str]:
        """
        Extract a meaningful phrase from text (avoid complete sentence questions).

        Tries to find:
        - Consecutive nouns/terms (5-12 words)
        - Removes stop words from start/end
        """
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'this', 'that', 'these', 'those', 'it',
            'its', 'he', 'she', 'they', 'them', 'we', 'us', 'i', 'me', 'my',
            'your', 'his', 'her', 'their', 'our', 'what', 'which', 'who', 'whom',
            'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
            'there', 'about', 'because', 'while', 'until', 'unless', 'although',
            'though', 'if', 'since', 'so', 'therefore', 'thus', 'hence', 'however'
        }

        words = re.split(r'\s+', text.strip())
        if len(words) < 5:
            return None

        # Clean words (remove punctuation)
        cleaned_words = []
        for w in words:
            clean = re.sub(r'[^\w\-\']', '', w)
            if clean:
                cleaned_words.append(clean)

        if len(cleaned_words) < 5:
            return None

        # Try to find a good slice of 5-10 words
        # Start from the middle
        start_idx = max(0, len(cleaned_words) // 3)

        # Try different window sizes
        for window_size in [10, 8, 6, 5]:
            end_idx = min(len(cleaned_words), start_idx + window_size)
            candidate = cleaned_words[start_idx:end_idx]

            if len(candidate) >= 5:
                # Trim stop words from beginning
                while candidate and candidate[0].lower() in stop_words:
                    candidate.pop(0)
                # Trim stop words from end
                while candidate and candidate[-1].lower() in stop_words:
                    candidate.pop()

                if len(candidate) >= 4:
                    phrase = ' '.join(candidate)
                    if 20 <= len(phrase) <= 120:
                        return phrase

        # If nothing else works, just use the middle 6-8 words
        start_idx = max(0, (len(cleaned_words) - 7) // 2)
        end_idx = min(len(cleaned_words), start_idx + 7)
        phrase = ' '.join(cleaned_words[start_idx:end_idx])
        return phrase

    def _fix_hyphenated_breaks(self, text: str) -> str:
        """
        Fix hyphenated word breaks from PDF/text extraction.

        Examples:
            "limi-\ntations" → "limitations"
            "exam-\nple" → "example"
        """
        # Pattern: word- followed by newline and then more word characters
        # Handle both "-\n" and "-\r\n" and also "-\t" etc.
        text = re.sub(r'(\w+)-[\r\n\t]+(\w+)', r'\1\2', text)
        # Also handle cases with a space after hyphen: "limi- \ntations"
        text = re.sub(r'(\w+)-[\r\n\t ]+(\w+)', r'\1\2', text)
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def precompute_embeddings(
        self,
        queries: List[EvalQuery],
        batch_size: int = 4,
    ) -> List[EvalQuery]:
        """Precompute golden_embedding and cache in EvalQuery."""
        if not self.embedding_provider:
            logger.warning("No embedding provider, skipping precompute")
            return queries

        all_embeddings = []
        contents = [q.golden_context for q in queries]

        # Process in small batches to avoid OOM
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            logger.info("Precomputing embeddings batch {}/{}", i // batch_size + 1, (len(contents) + batch_size - 1) // batch_size)
            batch_embeddings = await self.embedding_provider.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        for query, emb in zip(queries, all_embeddings):
            query.golden_embedding = emb

        logger.info("Precomputed embeddings for {} queries", len(queries))
        return queries
