"""Onboard/setup commands for nanobot CLI."""


import typer
from rich.console import Console

from nanobot import __logo__

console = Console()


def onboard():
    """Initialize nanobot configuration and workspace."""
    from nanobot.cli.commands import _create_workspace_templates
    from nanobot.config.loader import get_config_path, load_config, save_config
    from nanobot.config.schema import Config
    from nanobot.utils.helpers import get_workspace_path

    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            # 修复：合并旧配置和新默认配置
            old_config = load_config()
            new_config = Config()
            # 用旧配置的值更新新配置（保留旧值，添加新字段）
            old_data = old_config.model_dump(by_alias=True, exclude_unset=True)
            new_data = new_config.model_dump(by_alias=True)
            # 合并：旧值覆盖新默认值
            merged_data = {**new_data, **old_data}
            config = Config.model_validate(merged_data)
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved, new fields added)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")

    # Create workspace
    workspace = get_workspace_path()

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")

    # Create default bootstrap files
    _create_workspace_templates(workspace)

    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.nanobot/config.json[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]nanobot agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")
