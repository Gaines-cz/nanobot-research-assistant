# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**⚠️ Important: This project is a personal research assistant built on nanobot (the base framework). Development is in progress - no backward compatibility needed for now.**

Since this project is not yet released in production, breaking changes are acceptable. If a feature needs a major redesign or rebuild, it can be done directly without maintaining backward compatibility - but large rebuilds should be discussed and confirmed before implementation.

nanobot is an ultra-lightweight personal AI assistant framework (~4,000 lines of core code) built in Python. It provides a complete agent system with LLM integration, chat channels, tools, memory, and scheduling.

## Documentation Convention

**Documentation language:** Documents created during development (especially in `/docs/` directory) should be written in **Chinese (中文)** unless otherwise specified.

This includes:
- Implementation plans (实施计划)
- Design documents (设计文档)
- Change logs (改动说明)
- Bug fix records (Bug 修复记录)

## Key Architecture

### High-Level Flow

```
Channels (Telegram/Discord/etc.) → MessageBus → AgentLoop → LLM → Tools → Response
```

### Core Components

**`nanobot/agent/loop.py` - AgentLoop**
- The core processing engine that orchestrates the agent's behavior
- Receives messages from the bus, builds context, calls LLM, executes tools
- Handles session management, memory consolidation, and MCP server connections

**`nanobot/bus/queue.py` - MessageBus**
- Async queue that decouples channels from the agent
- Has `inbound` and `outbound` queues for message routing

**`nanobot/providers/` - LLM Providers**
- `registry.py` - Single source of truth for provider metadata (ProviderSpec)
- Supports 20+ providers via LiteLLM, plus custom direct providers
- Adding a new provider: 1) add to PROVIDERS in registry.py, 2) add to ProvidersConfig in schema.py

**`nanobot/channels/` - Chat Channels**
- Base channel interface in `base.py`
- Implementations: Telegram, Discord, WhatsApp, Feishu, Slack, Email, QQ, DingTalk, Matrix, Mochat
- ChannelManager in `manager.py` coordinates all enabled channels

**`nanobot/config/schema.py` - Configuration**
- Pydantic-based config schema with camelCase alias support
- Root `Config` model contains all sub-configs (agents, channels, providers, tools, gateway)

**`nanobot/agent/tools/` - Built-in Tools**
- Filesystem: read_file, write_file, edit_file, list_dir
- Shell: exec
- Web: web_search (Serper), web_fetch
- Message: send messages back to channels
- Spawn: create subagents
- Cron: manage scheduled tasks
- MCP: dynamic tools from Model Context Protocol servers

### Message Flow

1. A channel receives a message from a user
2. Channel creates an `InboundMessage` and publishes to `MessageBus.inbound`
3. `AgentLoop` consumes the inbound message
4. `ContextBuilder` builds the prompt with history, memory, skills
5. `LLMProvider` calls the LLM with tools
6. If tool calls are returned, `ToolRegistry` executes them
7. Loop continues until LLM returns a final response
8. `AgentLoop` creates an `OutboundMessage` and publishes to `MessageBus.outbound`
9. Channel consumes the outbound message and delivers to user

## Development Commands

**Installation**
```bash
pip install -e .
```

**Running Tests**
```bash
pytest tests/
pytest tests/test_commands.py -xvs  # Run specific test file
```

**Linting**
```bash
ruff check nanobot/
ruff check --fix nanobot/
```

**CLI Commands**
```bash
nanobot onboard                    # Initialize config & workspace
nanobot agent -m "Hello"           # Single message
nanobot agent                       # Interactive mode
nanobot gateway                     # Start gateway with channels
nanobot status                      # Show status
nanobot cron add --name daily --message "Hi" --every 3600
nanobot cron list
nanobot channels status
nanobot provider login openai-codex
```

## Configuration

Config file location: `~/.nanobot/config.json`

Minimal config:
```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  }
}
```

## Key Files to Know

- `nanobot/cli/commands.py` - Typer CLI entry point
- `nanobot/agent/context.py` - Prompt building
- `nanobot/agent/memory.py` - Memory system with consolidation
- `nanobot/session/manager.py` - Conversation session storage
- `nanobot/cron/service.py` - Scheduled tasks
- `nanobot/heartbeat/service.py` - Periodic heartbeat tasks
- `nanobot/agent/tools/mcp.py` - Model Context Protocol integration

## Adding New Features

**Adding a new tool**:
1. Create a class in `nanobot/agent/tools/` that inherits from `BaseTool`
2. Register it in `AgentLoop._register_default_tools()`

**Adding a new channel**:
1. Create a file in `nanobot/channels/` implementing the channel interface
2. Add config to `ChannelsConfig` in `config/schema.py`
3. Register in `ChannelManager.start_all()`

**Adding a new provider**:
1. Add `ProviderSpec` to `PROVIDERS` in `providers/registry.py`
2. Add field to `ProvidersConfig` in `config/schema.py`

## Project Stats

- ~4,000 lines of core agent code
- Python ≥3.11
- Uses asyncio throughout
- Dependencies: typer, litellm, pydantic, websockets, httpx, rich, prompt-toolkit