"""Memory tool: search and view memories."""

from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore, MemoryType
from nanobot.agent.tools.base import Tool


class SearchMemoryTool(Tool):
    """Tool to search memories using hybrid search (BM25 + Vector)."""

    def __init__(self, workspace: Path, embedding_model: str = "BAAI/bge-m3"):
        self._workspace = workspace
        self._embedding_model = embedding_model
        self._embedding_provider = None
        self._memory_store = None

    def _get_memory_store(self) -> MemoryStore:
        """Lazy initialization of MemoryStore with embedding provider."""
        if self._memory_store is None:
            # Import here to avoid circular imports and optional dependency issues
            try:
                from nanobot.rag.embeddings import SentenceTransformerEmbeddingProvider
                self._embedding_provider = SentenceTransformerEmbeddingProvider(self._embedding_model)
            except ImportError:
                # Fallback: vector search will be disabled
                pass

            self._memory_store = MemoryStore(
                self._workspace,
                embedding_provider=self._embedding_provider
            )
        return self._memory_store

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return """Search memories using hybrid search (BM25 + Vector).

Returns relevant memories sorted by comprehensive score (relevance 70%, frequency 20%, recency 10%).

Arguments:
- query: Search query string
- type: Optional memory type filter (history, knowledge, decisions, projects)
- limit: Number of results to return (default 5)"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                    "minLength": 1
                },
                "type": {
                    "type": "string",
                    "enum": ["history", "knowledge", "decisions", "projects"],
                    "description": "Optional memory type filter"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, type: str = None, limit: int = 5, **kwargs: Any) -> str:
        from datetime import datetime

        memory_type = None
        if type:
            type_map = {
                "history": MemoryType.HISTORY,
                "knowledge": MemoryType.KNOWLEDGE,
                "decisions": MemoryType.DECISIONS,
                "projects": MemoryType.PROJECTS,
            }
            memory_type = type_map.get(type.lower())

        try:
            memory_store = self._get_memory_store()
            results = await memory_store.search(query, type=memory_type, top_k=limit)

            if not results:
                return "No memories found."

            lines = [f"Found {len(results)} memory entries:\n"]
            for i, r in enumerate(results, 1):
                time_str = datetime.fromtimestamp(r["at_time"]).strftime("%Y-%m-%d")
                lines.append(f"--- Result {i} [{r['type']}] (score: {r['score']:.3f}, {time_str}) ---")
                lines.append(r["detail"])
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"Error searching memories: {str(e)}"
