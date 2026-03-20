"""Memory commands for nanobot CLI."""


import typer
from rich.console import Console
from rich.table import Table

from nanobot import __logo__

memory_app = typer.Typer(help="Manage memory")
console = Console()


@memory_app.command("status")
def memory_status():
    """Show memory status."""
    from nanobot.agent.memory import MemoryStore
    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} Memory Status\n")
    console.print(f"Database: {workspace / 'memory' / 'memory.db'}\n")

    with MemoryStore(workspace) as memory_store:
        # Table header
        table = Table(title="Memory Types")
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Total Size", justify="right")

        # Get counts per type

        conn = memory_store.connection
        cursor = conn.execute("""
            SELECT type, COUNT(*) as count, SUM(LENGTH(detail)) as total_size
            FROM memories WHERE deleted_at IS NULL
            GROUP BY type
        """)
        rows = cursor.fetchall()

        if not rows:
            console.print("[yellow]No memories found in database[/yellow]")
            raise typer.Exit(0)

        for row in rows:
            table.add_row(row[0], str(row[1]), f"{row[2] or 0:,}")

        console.print(table)

        # Show total
        cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL")
        total = cursor.fetchone()[0]
        console.print(f"Total: {total} memory entries\n")


@memory_app.command("view")
def memory_view(
    type: str = typer.Argument(..., help="Memory type to view (e.g., history, knowledge, decisions, projects)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent entries to show"),
):
    """View recent memory entries of a specific type."""
    from nanobot.agent.memory import MemoryStore, MemoryType
    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path

    # Map common names to MemoryType enum
    type_map = {
        "history": MemoryType.HISTORY,
        "knowledge": MemoryType.KNOWLEDGE,
        "decisions": MemoryType.DECISIONS,
        "projects": MemoryType.PROJECTS,
    }

    type_lower = type.lower()
    if type_lower not in type_map:
        console.print(f"[red]Unknown memory type: {type}[/red]")
        console.print(f"Available: {', '.join(type_map.keys())}")
        raise typer.Exit(1)

    memory_type = type_map[type_lower]

    with MemoryStore(workspace) as memory_store:
        # Query recent entries from database
        conn = memory_store.connection
        cursor = conn.execute("""
            SELECT detail, at_time, read_times
            FROM memories
            WHERE type = ? AND deleted_at IS NULL
            ORDER BY at_time DESC
            LIMIT ?
        """, (memory_type.value, limit))

        rows = cursor.fetchall()

        if not rows:
            console.print(f"[yellow]No {memory_type.value} memories found[/yellow]")
            raise typer.Exit(0)

        console.print(f"\n[bold]{memory_type.value}[/bold] (recent {len(rows)} entries)\n")

        for i, row in enumerate(rows, 1):
            detail, at_time, read_times = row
            from datetime import datetime
            time_str = datetime.fromtimestamp(at_time).strftime("%Y-%m-%d %H:%M")
            console.print(f"[cyan]--- Entry {i} ({time_str}, read: {read_times}) ---[/cyan]")
            console.print(detail if detail.strip() else "[dim](empty)[/dim]")
            console.print()


