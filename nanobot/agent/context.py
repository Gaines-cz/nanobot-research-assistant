"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    MAX_SYSTEM_PROMPT_LENGTH = 12000  # 为 model 留出空间

    def __init__(
        self,
        workspace: Path,
        embedding_provider: Optional[Any] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.workspace = workspace
        # Use provided MemoryStore or create a new one
        self.memory = memory_store if memory_store is not None else MemoryStore(workspace, embedding_provider)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None, query: str | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills.

        Args:
            skill_names: Optional list of skill names to load.
            query: Optional query for selective memory loading.
        """
        parts = []
        part_names = []

        # 1. Identity
        identity = self._get_identity()
        parts.append(identity)
        part_names.append("identity")

        # 2. Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
            part_names.append("bootstrap")

        # 3. Memory
        memory = self.memory.get_memory_context(query)
        if memory:
            memory_part = f"# Memory\n\n{memory}"
            parts.append(memory_part)
            part_names.append("memory")

        # 4. Always skills
        always_skills = self.skills.get_always_skills()
        always_content = ""
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                always_part = f"# Active Skills\n\n{always_content}"
                parts.append(always_part)
                part_names.append("always_skills")

        # 5. Skills summary
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            skills_part = f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}"""
            parts.append(skills_part)
            part_names.append("skills_summary")

        result = "\n\n---\n\n".join(parts)

        # 记录各部分长度（用于调试）
        if logger.level("DEBUG").no <= logger._core.min_level:
            separator_len = len("\n\n---\n\n") * (len(parts) - 1) if len(parts) > 1 else 0
            debug_msg = "System prompt parts breakdown: "
            breakdown = []
            for name, part in zip(part_names, parts):
                breakdown.append(f"{name}={len(part)}")
            debug_msg += ", ".join(breakdown)
            debug_msg += f" | separators={separator_len} | total={len(result)}"
            logger.debug(debug_msg)

        # Length check and truncation
        if len(result) > self.MAX_SYSTEM_PROMPT_LENGTH:
            # 记录详细的警告信息，包括各部分长度
            separator_len = len("\n\n---\n\n") * (len(parts) - 1) if len(parts) > 1 else 0
            warning_msg = f"System prompt too long ({len(result)} chars > {self.MAX_SYSTEM_PROMPT_LENGTH} limit), truncating. Breakdown: "
            breakdown = []
            for name, part in zip(part_names, parts):
                breakdown.append(f"{name}={len(part)}")
            warning_msg += ", ".join(breakdown)
            warning_msg += f" | separators={separator_len}"
            logger.warning(warning_msg)
            result = self._truncate_system_prompt(parts, self.MAX_SYSTEM_PROMPT_LENGTH)

        return result

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/
  - PROFILE.md: User profile (preferences, research direction)
  - PROJECTS.md: Project knowledge (tech stack, architecture)
  - PAPERS.md: Paper notes and research findings
  - DECISIONS.md: Decision records (why A over B)
  - TODOS.md: Current tasks and next steps
  - HISTORY.md: Append-only event log (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def _truncate_system_prompt(self, parts: list[str], max_length: int) -> str:
        """Truncate system prompt while preserving priority parts.

        Priority: identity > bootstrap > skills > memory
        """
        result = parts[0]  # Always keep identity

        for part in parts[1:]:
            separator = "\n\n---\n\n"
            if len(result) + len(part) + len(separator) < max_length:
                result += separator + part
            else:
                # Partial add with truncation notice
                remaining = max_length - len(result) - 50
                if remaining > 100:
                    result += separator + part[:remaining] + "\n\n...[truncated]"
                break

        return result

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        Args:
            history: Conversation history.
            current_message: The current user message.
            skill_names: Optional list of skill names to load.
            media: Optional list of media file paths.
            channel: Optional channel name for runtime context.
            chat_id: Optional chat ID for runtime context.
        """
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names, query=current_message)},
            *history,
            {"role": "user", "content": self._build_runtime_context(channel, chat_id)},
            {"role": "user", "content": self._build_user_content(current_message, media)},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages
