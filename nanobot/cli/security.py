"""Security configuration commands for nanobot CLI."""

import shutil

import typer
from rich.console import Console

from nanobot import __logo__

security_app = typer.Typer(help="Security configuration helpers")
console = Console()


@security_app.command("setup")
def security_setup():
    """Show instructions for setting up secure nanobot deployment."""
    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} Security Setup Guide\n")

    console.print("[bold]Step 1: Create dedicated user[/bold]")
    console.print("  [dim]# Linux[/dim]")
    console.print("  sudo useradd -r -s /bin/false -M nanobot")
    console.print()
    console.print("  [dim]# macOS[/dim]")
    console.print("  sudo sysadminctl -addUser nanobot -shell /usr/bin/false -home /var/empty")
    console.print()

    console.print("[bold]Step 2: Set workspace permissions[/bold]")
    console.print(f"  sudo mkdir -p {workspace}")
    console.print(f"  sudo chown -R nanobot:nanobot {workspace}")
    console.print(f"  sudo chmod 750 {workspace}")
    console.print()

    console.print("[bold]Step 3: Secure sensitive directories[/bold]")
    console.print("  chmod 700 ~/.ssh")
    console.print("  chmod 700 ~/.aws")
    console.print("  chmod 700 ~/.config")
    console.print()

    console.print("[bold]Step 4: Run nanobot as nanobot user[/bold]")
    console.print("  sudo -u nanobot nanobot agent -m \"Hello\"")
    console.print()

    console.print("[bold]Optional: Install firejail (Linux only)[/bold]")
    console.print("  [dim]# Debian/Ubuntu[/dim]")
    console.print("  sudo apt install firejail")
    console.print()
    console.print("  [dim]# RHEL/CentOS[/dim]")
    console.print("  sudo dnf install firejail")
    console.print()

    console.print("[bold]Current Configuration[/bold]")
    restrict_status = "[green]enabled[/green]" if config.tools.restrict_to_workspace else "[yellow]disabled[/yellow]"
    console.print(f"  restrict_to_workspace: {restrict_status}")
    console.print(f"  workspace: {workspace}")


@security_app.command("status")
def security_status():
    """Check current security configuration."""

    from nanobot.config.loader import load_config

    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} Security Status\n")

    # Workspace restrictions
    restrict_status = "[green]enabled[/green]" if config.tools.restrict_to_workspace else "[yellow]disabled[/yellow]"
    console.print(f"Workspace restriction: {restrict_status}")
    console.print(f"  Workspace path: {workspace}")

    # Workspace directory
    if workspace.exists():
        stat = workspace.stat()
        console.print("  Exists: [green]✓[/green]")
        console.print(f"  Permissions: {oct(stat.st_mode)[-3:]}")
    else:
        console.print("  Exists: [red]✗[/red]")

    # Firejail availability
    console.print()
    firejail_available = shutil.which("firejail") is not None
    firejail_status = "[green]available[/green]" if firejail_available else "[yellow]not found[/yellow]"
    console.print(f"Firejail sandbox: {firejail_status}")

    use_firejail = getattr(config.tools.exec, "use_firejail", True)
    firejail_config = "[green]enabled[/green]" if use_firejail else "[yellow]disabled[/yellow]"
    console.print(f"Firejail in config: {firejail_config}")

    # Summary
    console.print()
    console.print("[bold]Security Assessment[/bold]")
    if config.tools.restrict_to_workspace and firejail_available and use_firejail:
        console.print("  [green]Good: Workspace restriction + firejail enabled[/green]")
    elif config.tools.restrict_to_workspace:
        console.print("  [yellow]Fair: Workspace restriction enabled, firejail not available[/yellow]")
    else:
        console.print("  [red]Poor: Workspace restriction disabled[/red]")
