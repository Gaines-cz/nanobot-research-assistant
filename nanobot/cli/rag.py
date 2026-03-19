"""RAG document retrieval commands for nanobot CLI."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

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

    if db_path.exists():
        console.print(f"[red]Deleting existing index: {db_path}[/red]")
        if not typer.confirm("Continue?"):
            console.print("Cancelled.")
            raise typer.Exit(0)
        db_path.unlink()
        console.print("[green]✓[/green] Deleted existing index\n")

    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    async def scan():
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
        store = None
        try:
            from nanobot.rag import SentenceTransformerEmbeddingProvider
            embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
            store = DocumentStore(db_path, embedding_provider, rag_config)
        except (ImportError, Exception):
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
    from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider

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


@rag_app.command("eval-gen")
def rag_eval_gen(
    num_samples: int = typer.Option(50, "--num-samples", "-n", help="Number of test samples to generate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (default: ~/.nanobot/workspace/rag/eval/<timestamp>.json)"),
    min_chunk_length: int = typer.Option(200, "--min-chunk-length", help="Minimum chunk length for sampling"),
    random_seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
):
    """
    Generate test dataset using LLM to create realistic user queries.

    The generated queries will be saved to a JSON file and can be used
    with 'nanobot rag eval' for evaluation.
    """
    from datetime import datetime
    from nanobot.config.loader import load_config
    from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider
    from nanobot.rag.evaluation import DataGenerator, TestDatasetManager

    config = load_config()
    workspace = config.workspace_path
    rag_config = config.rag

    console.print(f"{__logo__} Test Dataset Generation\n")

    if not rag_config.enabled:
        console.print("[yellow]RAG is disabled in config[/yellow]")
        raise typer.Exit(1)

    db_path = workspace / "rag" / "docs.db"

    if not db_path.exists():
        console.print("[red]No index found. Run 'nanobot rag refresh' first.[/red]")
        raise typer.Exit(1)

    # Initialize LLM provider using _make_provider pattern
    try:
        from nanobot.cli.commands import _make_provider
        llm_provider = _make_provider(config)
        console.print(f"[green]✓[/green] LLM provider initialized")
    except Exception as e:
        console.print(f"[red]Failed to initialize LLM provider: {e}[/red]")
        console.print("Will use fallback generation method (basic keyword extraction)")
        llm_provider = None

    # Initialize embedding provider for precomputation
    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    try:
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except Exception as e:
        console.print(f"[red]Failed to open document store: {e}[/red]")
        raise typer.Exit(1)

    async def generate():
        # Generate queries using LLM
        generator = DataGenerator(store, llm_provider, embedding_provider)
        queries = await generator.generate(
            num_samples=num_samples,
            min_chunk_length=min_chunk_length,
            random_seed=random_seed,
        )

        if not queries:
            console.print("[yellow]No suitable chunks found for evaluation[/yellow]")
            return None

        # Precompute embeddings
        console.print(f"Precomputing embeddings for {len(queries)} queries...")
        queries = await generator.precompute_embeddings(queries)

        # Save dataset
        dataset_mgr = TestDatasetManager()
        dataset = TestDatasetManager.create_dataset(queries)

        save_path = dataset_mgr.save(dataset, output)

        return save_path, len(queries)

    with console.status("Generating test dataset...", spinner="dots"):
        result = asyncio.run(generate())

    if result is None:
        store.close()
        raise typer.Exit(1)

    save_path, count = result
    console.print(f"\n[green]✓[/green] Generated {count} test queries")
    console.print(f"  Saved to: {save_path}")
    console.print("\nRun evaluation with:")
    console.print(f"  nanobot rag eval --dataset {save_path}")

    store.close()


@rag_app.command("eval")
def rag_eval(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to test dataset JSON file"),
    include_baseline: bool = typer.Option(True, "--baseline/--no-baseline", help="Include BM25 baseline comparison"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed results"),
    ablation: bool = typer.Option(False, "--ablation", "-a", help="Run ablation study to test component contributions"),
):
    """
    Evaluate RAG retrieval performance using a pre-generated test dataset.

    Use 'nanobot rag eval-gen' first to generate a test dataset.
    """
    from nanobot.config.loader import load_config
    from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider
    from nanobot.rag.evaluation import (
        ABLATION_CONFIGS,
        AblationStudy,
        EvalConfig,
        ReportGenerator,
        RAGEvaluator,
        TestDatasetManager,
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

    if not dataset.exists():
        console.print(f"[red]Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Load test dataset
    try:
        dataset_mgr = TestDatasetManager()
        test_dataset = dataset_mgr.load(dataset)
        console.print(f"Loaded dataset: {test_dataset.version}")
        console.print(f"Queries: {test_dataset.num_queries}")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        raise typer.Exit(1)

    # Initialize embedding provider
    try:
        embedding_provider = SentenceTransformerEmbeddingProvider(rag_config.embedding_model)
    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install 'nanobot-ai[rag]'")
        raise typer.Exit(1)

    try:
        store = DocumentStore(db_path, embedding_provider, rag_config)
    except Exception as e:
        console.print(f"[red]Failed to open document store: {e}[/red]")
        raise typer.Exit(1)

    async def run_evaluation():
        nonlocal ablation_results

        # Clear cache and disable it for fresh evaluation
        store.clear_cache()
        original_cache_setting = rag_config.enable_search_cache
        rag_config.enable_search_cache = False

        # Precompute embeddings for queries
        queries = test_dataset.queries

        console.print("Precomputing embeddings...")
        for i in range(0, len(queries), 10):
            batch = queries[i:i + 10]
            batch_embeddings = await embedding_provider.embed_batch([q.golden_context for q in batch])
            for q, emb in zip(batch, batch_embeddings):
                q.golden_embedding = emb

        eval_config = EvalConfig(random_seed=42)

        # Run ablation study if requested
        ablation_results = None
        if ablation:
            console.print("\n[bold]Running Ablation Study[/bold]")
            console.print(f"Testing {len(ABLATION_CONFIGS)} configurations...")
            ablation_study = AblationStudy(
                store,
                embedding_provider,
                eval_config=eval_config,
                rag_config=rag_config,
            )
            ablation_results = await ablation_study.run_ablation(
                queries,
                ablation_configs=ABLATION_CONFIGS,
                include_baseline=False,
            )

        # Run main evaluation
        evaluator = RAGEvaluator(store, embedding_provider, eval_config)
        summary = await evaluator.evaluate(queries, include_baseline=include_baseline)

        # Restore cache setting
        rag_config.enable_search_cache = original_cache_setting

        return summary

    ablation_results = None
    with console.status("Running evaluation...", spinner="dots"):
        summary = asyncio.run(run_evaluation())

    # Generate and print report
    report_gen = ReportGenerator()
    full_report = report_gen.generate_full_report(
        summary,
        test_dataset.queries,
        ablation_results=ablation_results if ablation else None,
        ablation_configs=ABLATION_CONFIGS if ablation else None,
    )
    console.print(full_report)

    # Verbose output
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
            console.print()

    # Save results
    if output:
        output_data = {
            "dataset_version": test_dataset.version,
            "dataset_path": str(dataset),
            "num_queries": summary.num_queries,
            "metrics": {
                "recall_at_k": summary.recall_at_k,
                "mrr": summary.mrr,
                "hit_rate_at_k": summary.hit_rate_at_k,
                "ndcg_at_k": summary.ndcg_at_k,
                "avg_latency_ms": summary.avg_latency_ms,
                "baseline_recall_at_k": summary.baseline_recall_at_k,
                "baseline_mrr": summary.baseline_mrr,
                "baseline_ndcg_at_k": summary.baseline_ndcg_at_k,
            },
            "question_type_breakdown": summary.question_type_breakdown,
        }

        if ablation and ablation_results:
            output_data["ablation_study"] = {
                name: {
                    "recall_at_k": s.recall_at_k,
                    "mrr": s.mrr,
                    "ndcg_at_k": s.ndcg_at_k,
                    "avg_latency_ms": s.avg_latency_ms,
                }
                for name, s in ablation_results.items()
            }

        if verbose and summary.details:
            detailed_results = []
            queries_map = {q.id: q for q in test_dataset.queries}
            for result in summary.details:
                query = queries_map.get(result.query_id)
                data = {
                    "query_id": result.query_id,
                    "query": result.query,
                    "hit": result.hit,
                    "hit_rank": result.hit_rank,
                    "latency_ms": result.latency_ms,
                    "baseline_hit": result.baseline_hit,
                }
                if query:
                    data["source_chunk_id"] = query.source_chunk_id
                    data["question_type"] = query.question_type
                detailed_results.append(data)
            output_data["details"] = detailed_results

        output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[green]✓[/green] Results saved to {output}")

    store.close()
