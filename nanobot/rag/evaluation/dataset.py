"""RAG Evaluation - Test dataset persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from nanobot.rag.evaluation.base import EvalQuery, TestDataset


class TestDatasetManager:
    """Manages test dataset persistence and loading."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize dataset manager.

        Args:
            base_dir: Base directory for storing datasets.
                      Defaults to ~/.nanobot/workspace/rag/eval/
        """
        if base_dir is None:
            base_dir = Path.home() / ".nanobot" / "workspace" / "rag" / "eval"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        dataset: TestDataset,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save test dataset to file.

        Args:
            dataset: TestDataset to save
            output_path: Custom output path. If None, uses default location.

        Returns:
            Path where dataset was saved
        """
        if output_path is None:
            output_path = self.base_dir / f"{dataset.version}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        data = {
            "version": dataset.version,
            "created_at": dataset.created_at,
            "num_queries": dataset.num_queries,
            "metadata": dataset.metadata,
            "queries": [
                {
                    "id": q.id,
                    "query": q.query,
                    "golden_context": q.golden_context,
                    "source_chunk_id": q.source_chunk_id,
                    "source_doc": q.source_doc,
                    "question_type": q.question_type,
                    "tags": q.tags,
                    "golden_embedding": q.golden_embedding,
                }
                for q in dataset.queries
            ],
        }

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Saved dataset to {}", output_path)
        return output_path

    def load(self, path: Path) -> TestDataset:
        """
        Load test dataset from file.

        Args:
            path: Path to dataset file

        Returns:
            Loaded TestDataset
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        queries = [
            EvalQuery(
                id=q["id"],
                query=q["query"],
                golden_context=q["golden_context"],
                source_chunk_id=q.get("source_chunk_id"),
                source_doc=q.get("source_doc"),
                question_type=q.get("question_type"),
                tags=q.get("tags", []),
                golden_embedding=q.get("golden_embedding"),
            )
            for q in data["queries"]
        ]

        return TestDataset(
            version=data["version"],
            created_at=data["created_at"],
            num_queries=data["num_queries"],
            metadata=data.get("metadata", {}),
            queries=queries,
        )

    def list_datasets(self) -> List[dict]:
        """
        List all available datasets in the base directory.

        Returns:
            List of dataset info dicts with version, path, num_queries
        """
        datasets = []
        for f in self.base_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                datasets.append({
                    "version": data["version"],
                    "path": str(f),
                    "num_queries": data["num_queries"],
                    "created_at": data["created_at"],
                })
            except Exception as e:
                logger.warning("Failed to read dataset {}: {}", f, e)

        return sorted(datasets, key=lambda x: x["created_at"], reverse=True)

    @staticmethod
    def create_dataset(
        queries: List[EvalQuery],
        version: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> TestDataset:
        """
        Create a TestDataset from a list of queries.

        Args:
            queries: List of EvalQuery objects
            version: Dataset version string. If None, auto-generates from timestamp.
            metadata: Optional metadata dict

        Returns:
            TestDataset ready for saving
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        return TestDataset(
            version=version,
            created_at=datetime.now().isoformat(),
            num_queries=len(queries),
            queries=queries,
            metadata=metadata or {},
        )
