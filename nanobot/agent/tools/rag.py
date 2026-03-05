"""RAG retrieve tool."""

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.schema import RAGConfig
from nanobot.rag import (
    DocumentStore,
    SearchResult,
    SearchResultWithContext,
    SentenceTransformerEmbeddingProvider,
)


class SearchKnowledgeTool(Tool):
    """Tool for searching local documents."""

    name = "search_knowledge"
    description = """
    Semantic search across indexed research documents (PDFs, papers, notes).
    Use when you need to find information or answer questions from your knowledge base.
    NOT for reading specific files at known paths (use read_file).
    """
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you want to find in the documents",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Number of results to return",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        workspace: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "BAAI/bge-m3",
        rag_config: Optional[RAGConfig] = None,
    ):
        self.workspace = workspace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rag_config = rag_config or RAGConfig()
        # 如果没有显式传入 embedding_model，使用 rag_config 中的值
        if embedding_model == "BAAI/bge-m3":
            self.embedding_model = self.rag_config.embedding_model
        else:
            self.embedding_model = embedding_model

        self._doc_store: DocumentStore | None = None
        self._docs_dir: Path | None = None
        self._memory_dir: Path | None = None
        self._rag_dir: Path | None = None

    def _ensure_initialized(self) -> None:
        """Initialize the document store if not already initialized."""
        if self._doc_store is not None:
            return

        self._rag_dir = self.workspace / "rag"
        self._rag_dir.mkdir(parents=True, exist_ok=True)
        self._docs_dir = self.workspace / "docs"
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._memory_dir = self.workspace / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        db_path = self._rag_dir / "docs.db"

        # Don't delete database! Use existing one
        embedding_provider = SentenceTransformerEmbeddingProvider(self.embedding_model)
        self._doc_store = DocumentStore(db_path, embedding_provider, self.rag_config)

    async def scan_and_index(self, include_memory: bool = None) -> dict[str, int]:
        """
        Scan and index documents in docs and memory directories.

        Args:
            include_memory: If None, uses RAGConfig.enable_memory_index setting.
                           If True/False, overrides the config.
        """
        self._ensure_initialized()
        assert self._doc_store is not None
        assert self._docs_dir is not None
        assert self._memory_dir is not None

        # Determine whether to include memory
        if include_memory is None:
            include_memory = self.rag_config.enable_memory_index

        total_stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}

        # Index docs/ (convert legacy chunk_overlap to ratio)
        docs_stats = await self._doc_store.scan_and_index(
            self._docs_dir,
            chunk_size=self.rag_config.max_chunk_size,
            chunk_overlap_ratio=self.rag_config.chunk_overlap_ratio,
            root_path=self._docs_dir,
        )
        for key in docs_stats:
            total_stats[key] = total_stats.get(key, 0) + docs_stats[key]

        # Index memory/ (with smaller chunks since memory files are typically small)
        if include_memory:
            memory_stats = await self._doc_store.scan_and_index(
                self._memory_dir,
                chunk_size=self.rag_config.memory_chunk_size,
                chunk_overlap_ratio=self.rag_config.memory_chunk_overlap_ratio,
                root_path=self._memory_dir,
            )
            for key in memory_stats:
                total_stats[key] = total_stats.get(key, 0) + memory_stats[key]

        return total_stats

    async def index_memory_only(self) -> dict[str, int]:
        """Index only the memory directory (for consolidation updates)."""
        self._ensure_initialized()
        assert self._doc_store is not None
        assert self._memory_dir is not None

        # 使用配置中的参数，而非硬编码
        return await self._doc_store.scan_and_index(
            self._memory_dir,
            chunk_size=self.rag_config.memory_chunk_size,
            chunk_overlap_ratio=self.rag_config.memory_chunk_overlap_ratio,
            root_path=self._memory_dir,
        )

    async def execute(self, query: str, top_k: int = 5) -> str:
        """Execute the retrieve tool using the new advanced search flow."""
        self._ensure_initialized()
        assert self._doc_store is not None

        # Try advanced search first
        try:
            advanced_results = await self._doc_store.search_advanced(query)
            if advanced_results:
                return self._format_advanced_results(query, advanced_results)
        except Exception as e:
            logger.warning("Advanced search failed, falling back to basic search: {}", e)

        # Fallback to basic search
        results = await self._doc_store.search(query, top_k=top_k)

        if not results:
            stats = self._doc_store.get_stats()
            if stats["documents"] == 0:
                return (
                    "No documents found in your workspace/docs directory. "
                    "Add some PDF, Markdown, or Word documents there first."
                )
            return f"No results found for '{query}' in your documents."

        return self._format_results(query, results)

    def _format_results(self, query: str, results: list[SearchResult]) -> str:
        """Format search results nicely."""
        lines = [f"Results for \"{query}\":", ""]

        for i, result in enumerate(results, 1):
            content = result.content
            if len(content) > 500:
                content = content[:497] + "..."
            content = content.replace('"', '\\"')

            lines.append(f"[{i}] {result.filename} (score: {result.score:.2f}, {result.source})")
            lines.append(f'"{content}"')
            lines.append("")

        return "\n".join(lines)

    def _format_advanced_results(self, query: str, results: list[SearchResultWithContext]) -> str:
        """Format advanced search results with context."""
        lines = [f"Results for \"{query}\":", ""]

        for i, result in enumerate(results, 1):
            content = result.combined_content
            if len(content) > 800:
                content = content[:797] + "..."
            content = content.replace('"', '\\"')

            doc_title = result.document.title or result.document.filename
            lines.append(f"[{i}] {doc_title} (score: {result.final_score:.2f})")
            if result.chunk.section_title:
                lines.append(f"  Section: {result.chunk.section_title}")
            lines.append(f'"{content}"')
            lines.append("")

        return "\n".join(lines)

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

    @property
    def doc_store(self) -> "DocumentStore | None":
        """Get the underlying DocumentStore (read-only)."""
        self._ensure_initialized()
        return self._doc_store

    def close(self) -> None:
        """Close the document store connection."""
        if self._doc_store is not None:
            self._doc_store.close()
            self._doc_store = None

    @classmethod
    def from_shared_store(
        cls,
        doc_store: "DocumentStore",
        workspace: Path,
        rag_config: "RAGConfig",
    ) -> "SearchKnowledgeTool":
        """Create tool with shared DocumentStore (no new embedding model)."""
        # Use normal __init__ for proper initialization
        tool = cls(
            workspace=workspace,
            chunk_size=rag_config.max_chunk_size,
            chunk_overlap=int(rag_config.max_chunk_size * rag_config.chunk_overlap_ratio) if rag_config.max_chunk_size > 0 else 200,
            embedding_model=rag_config.embedding_model,
            rag_config=rag_config,
        )
        # Replace doc_store with shared instance
        tool._doc_store = doc_store
        # Set directory paths that would be created in _ensure_initialized
        tool._docs_dir = workspace / "docs"
        tool._memory_dir = workspace / "memory"
        tool._rag_dir = workspace / "rag"
        return tool
