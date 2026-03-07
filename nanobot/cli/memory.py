"""Memory file commands for nanobot CLI."""

import time

import typer
from rich.console import Console
from rich.table import Table

from nanobot import __logo__

memory_app = typer.Typer(help="Manage memory files")
console = Console()


@memory_app.command("status")
def memory_status():
    """Show memory file status."""
    from nanobot.agent.memory import MemoryFile
    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path
    memory_dir = workspace / "memory"

    console.print(f"{__logo__} Memory Status\n")
    console.print(f"Directory: {memory_dir}\n")

    if not memory_dir.exists():
        console.print("[yellow]Memory directory not found[/yellow]")
        raise typer.Exit(1)

    # Table header
    table = Table(title="Memory Files")
    table.add_column("File", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Last Modified")

    for mf in MemoryFile:
        path = memory_dir / mf.value
        if path.exists():
            stat = path.stat()
            content = path.read_text(encoding="utf-8")
            lines = len(content.splitlines())
            size = stat.st_size
            modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
            table.add_row(mf.value, str(lines), f"{size:,}", modified)
        else:
            table.add_row(mf.value, "-", "-", "[dim]not found[/dim]")

    console.print(table)


@memory_app.command("view")
def memory_view(
    file: str = typer.Argument(..., help="Memory file to view (e.g., profile, projects, papers)"),
):
    """View content of a specific memory file."""
    from nanobot.agent.memory import MemoryFile
    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path
    memory_dir = workspace / "memory"

    # Map common names to MemoryFile enum
    file_map = {
        "profile": MemoryFile.PROFILE,
        "projects": MemoryFile.PROJECTS,
        "papers": MemoryFile.PAPERS,
        "decisions": MemoryFile.DECISIONS,
        "todos": MemoryFile.TODOS,
        "history": MemoryFile.HISTORY,
    }

    file_lower = file.lower()
    if file_lower not in file_map:
        console.print(f"[red]Unknown memory file: {file}[/red]")
        console.print(f"Available: {', '.join(file_map.keys())}")
        raise typer.Exit(1)

    memory_file = file_map[file_lower]
    path = memory_dir / memory_file.value

    if not path.exists():
        console.print(f"[yellow]{memory_file.value} is empty or not found[/yellow]")
        raise typer.Exit(0)

    content = path.read_text(encoding="utf-8")
    console.print(f"\n[bold]{memory_file.value}[/bold]\n")
    console.print(content if content.strip() else "[dim](empty)[/dim]")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
):
    """Search across all memory files."""
    from nanobot.agent.memory import MemoryFile
    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path
    memory_dir = workspace / "memory"

    console.print(f"{__logo__} Searching memory for: [bold]{query}[/bold]\n")

    results = []

    for mf in MemoryFile:
        path = memory_dir / mf.value
        if not path.exists():
            continue

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            if query.lower() in line.lower():
                results.append((mf.value, i, line.strip()))

    if not results:
        console.print("[yellow]No results found[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(results)} match(es):\n")

    for filename, line_num, line_content in results:
        # Truncate long lines
        if len(line_content) > 100:
            line_content = line_content[:97] + "..."
        console.print(f"[cyan]{filename}:{line_num}[/cyan]")
        console.print(f"  {line_content}\n")
