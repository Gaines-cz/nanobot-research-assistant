"""RAG document retrieval commands for nanobot CLI."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nanobot import __logo__

rag_app = typer.Typer(help="Manage RAG (document retrieval)")
console = Console()


@rag_app.command("refresh")
def rag_refresh():
    """Refresh RAG document index - scan for new/changed/deleted documents."""
    from nanobot.config.loader import load_config
    from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider

    config = load_config()
    workspace = config.workspace_path
    rag_config = config.rag

    console.print(f"{__logo__} Refreshing RAG index...\n")

    if not rag_config.enabled:
        console.print("[yellow]RAG is disabled in config[/yellow]")
        raise typer.Exit(1)

    docs_dir = workspace / "docs"
    db_path = workspace / "rag" / "docs.db"

    console.print(f"Workspace: {workspace}")
    console.print(f"Docs dir: {docs_dir}")
    console.print(f"Database: {db_path}\n")

    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    async def scan():
        """Scan docs for rag_scan command."""
        return await store.scan_and_index(
            docs_dir,
            min_chunk_size=rag_config.min_chunk_size,
            max_chunk_size=rag_config.max_chunk_size,
            chunk_overlap_ratio=rag_config.chunk_overlap_ratio,
        )

    with console.status("Scanning documents...", spinner="dots"):
        stats = asyncio.run(scan())

    console.print("[green]✓[/green] RAG refresh complete!")
    console.print(f"  Added: {stats['added']}")
    console.print(f"  Updated: {stats['updated']}")
    console.print(f"  Deleted: {stats['deleted']}")

    stats = store.get_stats()
    console.print(f"\nTotal: {stats['documents']} documents, {stats['chunks']} chunks")
    vector_status = "[green]enabled[/green]" if stats.get('vector_enabled', False) else "[yellow]disabled[/yellow]"
    console.print(f"Vector search: {vector_status}")

    store.close()


@rag_app.command("rebuild")
def rag_rebuild():
    """Delete existing index and rebuild from scratch."""
    from nanobot.config.loader import load_config
    from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider

    config = load_config()
    workspace = config.workspace_path
    rag_config = config.rag

    console.print(f"{__logo__} Rebuilding RAG index...\n")

    if not rag_config.enabled:
        console.print("[yellow]RAG is disabled in config[/yellow]")
        raise typer.Exit(1)

    docs_dir = workspace / "docs"
    db_path = workspace / "rag" / "docs.db"

    console.print(f"Workspace: {workspace}")
    console.print(f"Docs dir: {docs_dir}")
    console.print(f"Database: {db_path}\n")

    # Delete existing database if it exists
    if db_path.exists():
        console.print(f"[red]Deleting existing index: {db_path}[/red]")
        if not typer.confirm("Continue?"):
            console.print("Cancelled.")
            raise typer.Exit(0)
        db_path.unlink()
        console.print("[green]✓[/green] Deleted existing index\n")

    # Rebuild index
    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    async def scan():
        """Scan docs for rag_rebuild command."""
        return await store.scan_and_index(
            docs_dir,
            min_chunk_size=rag_config.min_chunk_size,
            max_chunk_size=rag_config.max_chunk_size,
            chunk_overlap_ratio=rag_config.chunk_overlap_ratio,
        )

    with console.status("Rebuilding index...", spinner="dots"):
        stats = asyncio.run(scan())

    console.print("[green]✓[/green] RAG rebuild complete!")
    console.print(f"  Added: {stats['added']}")
    console.print(f"  Updated: {stats['updated']}")
    console.print(f"  Deleted: {stats['deleted']}")

    stats = store.get_stats()
    console.print(f"\nTotal: {stats['documents']} documents, {stats['chunks']} chunks")
    vector_status = "[green]enabled[/green]" if stats.get('vector_enabled', False) else "[yellow]disabled[/yellow]"
    console.print(f"Vector search: {vector_status}")

    store.close()


@rag_app.command("status")
def rag_status():
    """Show RAG index status and statistics."""
    from nanobot.config.loader import load_config
    from nanobot.rag import DocumentStore

    config = load_config()
    workspace = config.workspace_path
    rag_config = config.rag

    console.print(f"{__logo__} RAG Status\n")

    if not rag_config.enabled:
        console.print("[yellow]RAG is disabled in config[/yellow]")
    else:
        console.print("RAG: [green]enabled[/green]")
        console.print(f"Embedding model: {rag_config.embedding_model}")
        console.print(f"Chunk size: {rag_config.chunk_size} (overlap ratio: {rag_config.chunk_overlap_ratio:.2f})")

    docs_dir = workspace / "docs"
    db_path = workspace / "rag" / "docs.db"

    console.print(f"\nDocs dir: {docs_dir}")
    if docs_dir.exists():
        # Only count supported document types
        supported_extensions = {".pdf", ".md", ".markdown", ".docx", ".doc", ".txt"}
        count = sum(1 for _ in docs_dir.rglob("*")
                      if _.is_file() and not _.name.startswith(".")
                      and _.suffix.lower() in supported_extensions)
        console.print(f"  Files in docs: {count}")
    else:
        console.print("  [yellow]Docs directory not found[/yellow]")

    console.print(f"Database: {db_path}")
    if db_path.exists():
        console.print(f"  Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        console.print("  [yellow]Database not found[/yellow]")

    if not rag_config.enabled or not db_path.exists():
        raise typer.Exit(0)

    try:
        # Try to load with embedding provider if possible to check vector status
        store = None
        try:
            from nanobot.rag import SentenceTransformerEmbeddingProvider
            embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
            store = DocumentStore(db_path, embedding_provider, rag_config)
        except (ImportError, Exception):
            # Fall back to no embedding provider
            store = DocumentStore(db_path)

        stats = store.get_stats()

        console.print("\n[bold]Index Statistics[/bold]")
        console.print(f"  Documents: {stats['documents']}")
        console.print(f"  Chunks: {stats['chunks']}")

        by_type = stats.get('by_file_type', {})
        if by_type:
            console.print("  By type:")
            for ft, count in by_type.items():
                console.print(f"    {ft}: {count}")

        vector_enabled = stats.get('vector_enabled', False)
        vector_status = "[green]enabled[/green]" if vector_enabled else "[yellow]disabled[/yellow]"
        console.print("\n[bold]Search Capabilities[/bold]")
        console.print(f"  Vector search: {vector_status}")
        if not vector_enabled:
            console.print("  Full-text search: [green]enabled[/green]")

        store.close()
    except Exception as e:
        console.print(f"\n[yellow]Could not load database: {e}[/yellow]")


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
):
    """Search indexed documents using semantic search."""
    from nanobot.config.loader import load_config
    from nanobot.rag import (
        DocumentStore,
        SentenceTransformerEmbeddingProvider,
    )

    config = load_config()
    workspace = config.workspace_path
    rag_config = config.rag

    console.print(f"{__logo__} Searching...\n")

    if not rag_config.enabled:
        console.print("[yellow]RAG is disabled in config[/yellow]")
        raise typer.Exit(1)

    db_path = workspace / "rag" / "docs.db"

    if not db_path.exists():
        console.print("[red]No index found. Run 'nanobot rag refresh' first.[/red]")
        raise typer.Exit(1)

    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    async def search():
        return await store.search_advanced(query, top_k=top_k)

    with console.status("Searching...", spinner="dots"):
        results = asyncio.run(search())

    if not results:
        console.print(f"[yellow]No results found for: {query}[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Results for:[/bold] {query}\n")

    for i, result in enumerate(results, 1):
        content = result.combined_content
        if len(content) > 400:
            content = content[:397] + "..."
        doc_title = result.document.title or result.document.filename
        console.print(f"[{i}] {doc_title} (score: {result.final_score:.2f})")
        if result.chunk.section_title:
            console.print(f"    Section: {result.chunk.section_title}")
        console.print(f"    {content}\n")

    store.close()


@rag_app.command("eval")
def rag_eval(
    num_samples: int = typer.Option(50, "--num-samples", "-n", help="Number of test samples to generate"),
    include_baseline: bool = typer.Option(True, "--baseline/--no-baseline", help="Include BM25 baseline comparison"),
    min_chunk_length: int = typer.Option(200, "--min-chunk-length", help="Minimum chunk length for sampling"),
    random_seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed results"),
):
    """Evaluate RAG retrieval performance."""
    from nanobot.config.loader import load_config
    from nanobot.rag import (
        DocumentStore,
        SentenceTransformerEmbeddingProvider,
    )
    from nanobot.rag.evaluation import (
        DataGenerator,
        EvalConfig,
        RAGEvaluator,
    )

    config = load_config()
    workspace = config.workspace_path
    rag_config = config.rag

    console.print(f"{__logo__} RAG Evaluation\n")

    if not rag_config.enabled:
        console.print("[yellow]RAG is disabled in config[/yellow]")
        raise typer.Exit(1)

    db_path = workspace / "rag" / "docs.db"

    if not db_path.exists():
        console.print("[red]No index found. Run 'nanobot rag refresh' first.[/red]")
        raise typer.Exit(1)

    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    # Store queries and baseline results for verbose output
    eval_queries = []
    baseline_results_cache = {}

    async def run_evaluation():
        nonlocal eval_queries, baseline_results_cache

        # Step 1: Generate test data
        console.print(f"Generating {num_samples} test queries...")
        generator = DataGenerator(store, embedding_provider)
        queries = generator.generate_basic(
            num_samples=num_samples,
            min_chunk_length=min_chunk_length,
            random_seed=random_seed,
        )

        eval_queries = queries

        if not queries:
            console.print("[yellow]No suitable chunks found for evaluation[/yellow]")
            return None

        # Precompute embeddings
        console.print(f"Precomputing embeddings for {len(queries)} queries...")
        queries = await generator.precompute_embeddings(queries)

        # Step 2: Run evaluation
        console.print("Running evaluation...")
        eval_config = EvalConfig(random_seed=random_seed)
        evaluator = RAGEvaluator(store, embedding_provider, eval_config)
        summary = await evaluator.evaluate(
            queries,
            include_baseline=include_baseline,
        )

        # If verbose, collect baseline results for comparison
        if verbose and include_baseline:
            console.print("Collecting detailed baseline results...")
            from nanobot.rag.evaluation.baseline import BaselineRetriever
            baseline = BaselineRetriever(store.connection)
            for query in queries:
                baseline_results_cache[query.id] = await baseline.search_bm25(query.query, top_k=5)

        return summary

    with console.status("Running evaluation...", spinner="dots"):
        summary = asyncio.run(run_evaluation())

    if summary is None:
        store.close()
        raise typer.Exit(1)

    # Print results
    console.print("\n[bold]Evaluation Results[/bold]\n")

    # Core metrics table
    table = Table(title="Core Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Our Score", justify="right")
    if include_baseline and summary.baseline_recall_at_5 is not None:
        table.add_column("Baseline", justify="right")
        table.add_column("Improvement", justify="right")

    # Recall@5
    if include_baseline and summary.baseline_recall_at_5 is not None:
        improvement = ""
        if summary.baseline_recall_at_5 > 0:
            imp_pct = (summary.recall_at_5 - summary.baseline_recall_at_5) / summary.baseline_recall_at_5 * 100
            improvement = f"{imp_pct:+.1f}%"
        table.add_row(
            "Recall@5",
            f"{summary.recall_at_5:.4f}",
            f"{summary.baseline_recall_at_5:.4f}",
            improvement,
        )
    else:
        table.add_row("Recall@5", f"{summary.recall_at_5:.4f}")

    # MRR
    if include_baseline and summary.baseline_mrr is not None:
        improvement = ""
        if summary.baseline_mrr > 0:
            imp_pct = (summary.mrr - summary.baseline_mrr) / summary.baseline_mrr * 100
            improvement = f"{imp_pct:+.1f}%"
        table.add_row(
            "MRR",
            f"{summary.mrr:.4f}",
            f"{summary.baseline_mrr:.4f}",
            improvement,
        )
    else:
        table.add_row("MRR", f"{summary.mrr:.4f}")

    # Hit Rate@5
    table.add_row("Hit Rate@5", f"{summary.hit_rate_at_5:.4f}")
    table.add_row("Avg Latency", f"{summary.avg_latency_ms:.2f}ms")

    console.print(table)

    # Difficulty breakdown
    if summary.difficulty_breakdown:
        console.print("\n[bold]Difficulty Breakdown[/bold]")
        for diff, data in summary.difficulty_breakdown.items():
            console.print(f"  {diff}: {data['hits']}/{data['total']} (recall: {data['recall']:.2%})")

    # Failure breakdown
    if summary.failure_breakdown:
        console.print("\n[bold]Failure Breakdown[/bold]")
        for reason, count in summary.failure_breakdown.items():
            console.print(f"  {reason}: {count}")

    # Verbose: detailed results
    if verbose and summary.details:
        console.print("\n[bold]Detailed Results[/bold]\n")
        for result in summary.details:
            status = "[green]✓[/green]" if result.hit else "[red]✗[/red]"
            reason = f" ({result.hit_reason})" if result.hit_reason else ""
            baseline_info = ""
            if result.baseline_hit is not None:
                baseline_status = "✓" if result.baseline_hit else "✗"
                baseline_rank = f"@{result.baseline_hit_rank}" if result.baseline_hit_rank else ""
                baseline_info = f" [baseline: {baseline_status}{baseline_rank}]"
            console.print(f"{status} {result.query}{reason}{baseline_info}")
            if result.failure_reason:
                console.print(f"    Reason: {result.failure_reason}")
            if result.found_chunk_ids:
                console.print(f"    Found chunk IDs: {result.found_chunk_ids}")
            # Find the query to show source chunk
            query = next((q for q in eval_queries if q.id == result.query_id), None)
            if query:
                console.print(f"    Expected chunk ID: {query.source_chunk_id}")
                console.print(f"    Source doc: {query.source_doc}")
            console.print()

    # Save to file
    if output:
        output_data = {
            "dataset_name": summary.dataset_name,
            "num_queries": summary.num_queries,
            "config": {
                "strong_threshold": summary.config.strong_threshold,
                "weak_threshold": summary.config.weak_threshold,
                "top_k": summary.config.top_k,
                "random_seed": summary.config.random_seed,
            },
            "rag_config": {
                "bm25_threshold": rag_config.bm25_threshold,
                "vector_threshold": rag_config.vector_threshold,
                "rrf_k": rag_config.rrf_k,
            },
            "metrics": {
                "recall_at_5": summary.recall_at_5,
                "mrr": summary.mrr,
                "hit_rate_at_5": summary.hit_rate_at_5,
                "avg_latency_ms": summary.avg_latency_ms,
                "baseline_recall_at_5": summary.baseline_recall_at_5,
                "baseline_mrr": summary.baseline_mrr,
            },
            "difficulty_breakdown": summary.difficulty_breakdown,
            "failure_breakdown": summary.failure_breakdown,
        }

        # Add detailed per-query data if verbose
        if verbose and summary.details:
            detailed_queries = []
            for result in summary.details:
                query = next((q for q in eval_queries if q.id == result.query_id), None)
                query_data = {
                    "query_id": result.query_id,
                    "query": result.query,
                    "hit": result.hit,
                    "hit_rank": result.hit_rank,
                    "hit_reason": result.hit_reason,
                    "failure_reason": result.failure_reason,
                    "best_similarity": result.best_similarity,
                    "found_chunk_ids": result.found_chunk_ids,
                    "latency_ms": result.latency_ms,
                    "baseline_hit": result.baseline_hit,
                    "baseline_hit_rank": result.baseline_hit_rank,
                }
                if query:
                    query_data["source_chunk_id"] = query.source_chunk_id
                    query_data["source_doc"] = query.source_doc
                    query_data["golden_context"] = query.golden_context
                # Add baseline results if available
                if result.query_id in baseline_results_cache:
                    query_data["baseline_results"] = [
                        {"chunk_id": r.chunk.id, "content": r.chunk.content[:100] if r.chunk.content else None}
                        for r in baseline_results_cache[result.query_id]
                    ]
                detailed_queries.append(query_data)
            output_data["detailed_queries"] = detailed_queries

        output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[green]✓[/green] Results saved to {output}")

    store.close()
