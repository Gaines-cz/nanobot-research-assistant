"""RAG retrieve tool."""

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.rag import DocumentStore, SearchResult, SentenceTransformerEmbeddingProvider


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
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.workspace = workspace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        self._doc_store: DocumentStore | None = None
        self._docs_dir: Path | None = None
        self._rag_dir: Path | None = None

    def _ensure_initialized(self) -> None:
        """Initialize the document store if not already initialized."""
        if self._doc_store is not None:
            return

        self._rag_dir = self.workspace / "rag"
        self._rag_dir.mkdir(parents=True, exist_ok=True)
        self._docs_dir = self.workspace / "docs"
        self._docs_dir.mkdir(parents=True, exist_ok=True)

        db_path = self._rag_dir / "docs.db"
        embedding_provider = SentenceTransformerEmbeddingProvider(self.embedding_model)
        self._doc_store = DocumentStore(db_path, embedding_provider)

    async def scan_and_index(self) -> dict[str, int]:
        """Scan and index documents in the docs directory."""
        self._ensure_initialized()
        assert self._doc_store is not None
        assert self._docs_dir is not None
        return await self._doc_store.scan_and_index(
            self._docs_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    async def execute(self, query: str, top_k: int = 5) -> str:
        """Execute the retrieve tool."""
        self._ensure_initialized()
        assert self._doc_store is not None

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
            # Truncate content if too long
            content = result.content
            if len(content) > 500:
                content = content[:497] + "..."
            # Escape quotes
            content = content.replace('"', '\\"')

            lines.append(f"[{i}] {result.filename} (score: {result.score:.2f}, {result.source})")
            lines.append(f'"{content}"')
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the document store."""
        self._ensure_initialized()
        assert self._doc_store is not None
        return self._doc_store.get_stats()
