"""Multi-file memory system with incremental operations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.rag.embeddings import EmbeddingProvider
    from nanobot.rag.store import DocumentStore
    from nanobot.session.manager import Session


class MemoryFile(Enum):
    """Memory file types."""

    PROFILE = "PROFILE.md"
    PROJECTS = "PROJECTS.md"
    PAPERS = "PAPERS.md"
    DECISIONS = "DECISIONS.md"
    TODOS = "TODOS.md"
    HISTORY = "HISTORY.md"


@dataclass
class MemoryOperation:
    """A single memory operation."""

    file: MemoryFile
    action: str  # append | prepend | update_section | replace | skip
    content: Optional[str] = None
    section: Optional[str] = None


# Tool schema for LLM to call
_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save memory updates with incremental operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events. Start with [YYYY-MM-DD HH:MM].",
                    },
                    "operations": {
                        "type": "array",
                        "description": "List of memory operations to perform.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "enum": ["profile", "projects", "papers", "decisions", "todos"],
                                    "description": "Target memory file.",
                                },
                                "action": {
                                    "type": "string",
                                    "enum": ["append", "prepend", "update_section", "replace", "skip"],
                                    "description": "Operation type.",
                                },
                                "section": {
                                    "type": "string",
                                    "description": "Section name for update_section action.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to write.",
                                },
                            },
                            "required": ["file", "action"],
                        },
                    },
                },
                "required": ["history_entry", "operations"],
            },
        },
    }
]

MEMORY_FILES_DESC = """## Memory Files

- **profile**: User profile (research direction, preferences). Stable, rarely changes.
- **projects**: Project knowledge (tech stack, architecture, progress). Semi-stable.
- **papers**: Paper notes (papers read, key findings). Incremental, append new entries.
- **decisions**: Decision records (why A over B). Incremental.
- **todos**: Todo list. Frequently updated, use replace action."""


class MemoryStore:
    """Multi-file memory store with incremental operations."""

    MEMORY_FILES = {
        "profile": MemoryFile.PROFILE,
        "projects": MemoryFile.PROJECTS,
        "papers": MemoryFile.PAPERS,
        "decisions": MemoryFile.DECISIONS,
        "todos": MemoryFile.TODOS,
    }

    def __init__(
        self,
        workspace: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        rag_store: Optional["DocumentStore"] = None,
    ):
        self.memory_dir = ensure_dir(workspace / "memory")
        self._embedding_provider = embedding_provider
        self._rag_store = rag_store  # RAG store for memory search

    # === Read Operations ===

    def read_file(self, file: MemoryFile) -> str:
        """Read a memory file."""
        path = self.memory_dir / file.value
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def read_section(self, file: MemoryFile, section: str) -> Optional[str]:
        """Read a specific section from a memory file."""
        content = self.read_file(file)
        # Match: ## Section Name\n...content...\n(?=##|$)
        pattern = rf"##\s+{re.escape(section)}\s*(?:\n|$)(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    # === Write Operations ===

    def append(self, file: MemoryFile, content: str) -> None:
        """Append content to file."""
        path = self.memory_dir / file.value
        existing = self.read_file(file)
        if existing:
            new_content = existing.rstrip() + "\n\n" + content.strip() + "\n"
        else:
            new_content = content.strip() + "\n"
        path.write_text(new_content, encoding="utf-8")

    async def append_with_dedup(
        self,
        file: MemoryFile,
        content: str,
        *,
        similarity_threshold: float = 0.8,
    ) -> bool:
        """
        追加内容，自动跳过重复内容。

        先检查精确匹配，再检查语义相似（需要 embedding_provider）。

        Args:
            file: 目标记忆文件
            content: 要追加的内容
            similarity_threshold: 语义相似度阈值，默认 0.8

        Returns:
            True: 成功追加
            False: 跳过（重复内容）
        """
        if not content or not content.strip():
            return False

        existing = self.read_file(file)
        content_stripped = content.strip()

        # Layer 1: 精确匹配
        if content_stripped in existing:
            logger.debug("Exact content already exists in {}, skipping", file.value)
            return False

        # Layer 2: 语义相似（可选）
        if self._embedding_provider:
            try:
                if await self._is_semantically_similar(content_stripped, existing, similarity_threshold):
                    logger.debug("Similar content already exists in {}, skipping", file.value)
                    return False
            except Exception as e:
                logger.warning("Semantic similarity check failed: {}", e)
                # 失败时不阻止追加

        self.append(file, content)
        return True

    async def _is_semantically_similar(
        self,
        content: str,
        existing: str,
        threshold: float,
    ) -> bool:
        """检查内容是否与已有内容语义相似。"""
        if not existing.strip():
            return False

        try:
            # 获取新内容的 embedding
            content_embedding = await self._embedding_provider.embed(content)

            # 将已有内容按段落分割，检查每个段落
            paragraphs = [p.strip() for p in existing.split("\n\n") if p.strip()]

            for para in paragraphs:
                para_embedding = await self._embedding_provider.embed(para)
                similarity = self._cosine_similarity(content_embedding, para_embedding)
                if similarity >= threshold:
                    return True

            return False
        except Exception as e:
            logger.warning("Semantic similarity check failed: {}", e)
            return False  # 失败时不阻止追加

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度。"""
        import math

        if len(a) != len(b):
            logger.warning("Embedding length mismatch: {} vs {}", len(a), len(b))
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def prepend(self, file: MemoryFile, content: str) -> None:
        """Prepend content to file."""
        path = self.memory_dir / file.value
        existing = self.read_file(file)
        if existing:
            new_content = content.strip() + "\n\n" + existing
        else:
            new_content = content.strip() + "\n"
        path.write_text(new_content, encoding="utf-8")

    def update_section(self, file: MemoryFile, section: str, content: str) -> None:
        """Update or add a section in file."""
        old_content = self.read_file(file)
        section_header = f"## {section}"
        content_stripped = content.strip()
        new_section = f"{section_header}\n{content_stripped}\n"

        # 检查 section 是否已存在相同内容（去重）
        if content_stripped:
            pattern = rf"##\s+{re.escape(section)}\s*(?:\n|$)(.*?)(?=\n##\s|\Z)"
            match = re.search(pattern, old_content, re.DOTALL)
            if match:
                existing_content = match.group(1).strip()
                # 跳过空 content 或看起来像 section header 的情况（regex 边界）
                if existing_content and not existing_content.startswith("##"):
                    if existing_content == content_stripped:
                        logger.debug("Same content already exists in section {}, skipping", section)
                        return

        # Check if section exists
        pattern = rf"##\s+{re.escape(section)}\s*(?:\n|$).*?(?=\n##\s|\Z)"
        if re.search(pattern, old_content, re.DOTALL):
            # Replace existing section
            new_content = re.sub(pattern, new_section.rstrip(), old_content, flags=re.DOTALL)
        elif old_content:
            # Add new section to existing file
            new_content = old_content.rstrip() + "\n\n" + new_section
        else:
            # First section in empty file
            new_content = new_section

        path = self.memory_dir / file.value
        path.write_text(new_content, encoding="utf-8")

        # Verify the update worked: check that section header was actually written
        verification = self.read_file(file)
        # 验证 section header 存在（比验证 content 更健壮）
        if section_header not in verification:
            logger.warning(
                "Section update verification failed: header not found after write. "
                "Section: {}, File: {}",
                section, file.value
            )
            self.replace(file, old_content)

    def replace(self, file: MemoryFile, content: str) -> None:
        """Replace entire file content."""
        path = self.memory_dir / file.value
        path.write_text(content.strip() + "\n", encoding="utf-8")

    def apply_operation(self, op: MemoryOperation) -> bool:
        """Apply a single memory operation. Returns True on success."""
        if op.action == "skip":
            return True

        # Validate required parameters
        if op.action == "update_section" and not op.section:
            logger.warning("update_section requires section parameter, skipping")
            return False
        if op.action in ("append", "prepend", "update_section", "replace") and op.content is None:
            logger.warning(f"{op.action} requires content parameter, skipping")
            return False

        # Log operation
        log_msg = f"[Memory] {op.action} on {op.file.value}"
        if op.section:
            log_msg += f" (section: {op.section})"
        logger.info(log_msg)

        if op.action == "append":
            # 精确匹配去重（同步版本，适用于 consolidation）
            existing = self.read_file(op.file)
            if op.content.strip() in existing:
                logger.debug("Exact content already exists in {}, skipping", op.file.value)
                return True
            self.append(op.file, op.content)
        elif op.action == "prepend":
            # 精确匹配去重（同步版本，适用于 consolidation）
            existing = self.read_file(op.file)
            if op.content.strip() in existing:
                logger.debug("Exact content already exists in {}, skipping", op.file.value)
                return True
            self.prepend(op.file, op.content)
        elif op.action == "update_section":
            self.update_section(op.file, op.section, op.content)
        elif op.action == "replace":
            self.replace(op.file, op.content)

        return True

    def apply_operations_atomic(self, operations: list[dict]) -> bool:
        """Apply multiple operations atomically. All succeed or all fail."""
        applied: list[tuple[MemoryFile, str]] = []  # (file, old_content) for rollback

        try:
            for op_data in operations:
                file_name = op_data.get("file")
                action = op_data.get("action", "skip")

                if file_name not in self.MEMORY_FILES:
                    logger.warning("Unknown memory file: {}", file_name)
                    raise ValueError(f"Unknown file: {file_name}")

                # 保存旧内容用于回滚
                file = self.MEMORY_FILES[file_name]
                old_content = self.read_file(file)
                applied.append((file, old_content))

                op = MemoryOperation(
                    file=file,
                    action=action,
                    content=op_data.get("content"),
                    section=op_data.get("section"),
                )

                if not self.apply_operation(op):
                    raise Exception(f"Operation failed: {op_data}")

            return True

        except Exception as e:
            logger.error("Atomic operation failed, rolling back: {}", e)
            rollback_failed = []

            for file, old_content in reversed(applied):
                try:
                    self.replace(file, old_content)
                    logger.debug("Rollback succeeded for {}", file.value)
                except Exception as rollback_e:
                    logger.error("Rollback failed for {}: {}", file.value, rollback_e)
                    rollback_failed.append(file)

            if rollback_failed:
                logger.critical(
                    "Rollback incomplete - manual intervention may be needed: {}",
                    [f.value for f in rollback_failed]
                )
            return False

    def append_history(self, entry: str) -> None:
        """Append to history log."""
        # 类型防御 + 空值检查
        if not isinstance(entry, str):
            logger.warning("append_history received non-string type: {}", type(entry).__name__)
            entry = str(entry)
        if not entry or not entry.strip():
            logger.debug("Skipping empty history entry")
            return
        path = self.memory_dir / MemoryFile.HISTORY.value
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    # === Context Loading ===

    def get_memory_context(self, query: str | None = None) -> str:
        """
        Get memory context for system prompt.

        If query is provided, load relevant files based on keywords.
        Otherwise, load PROFILE + TODOS by default.
        """
        parts = []

        # Always load profile
        profile = self.read_file(MemoryFile.PROFILE)
        if profile:
            parts.append(f"## Profile\n{profile}")

        # Load todos by default (current tasks)
        todos = self.read_file(MemoryFile.TODOS)
        if todos:
            parts.append(f"## Current Tasks\n{todos}")

        # Query-based loading
        if query:
            query_lower = query.lower()

            # Papers: more specific triggers to avoid false positives
            paper_kws = [
                "论文", "paper", "papers", "arxiv", "文献",
                "read paper", "论文笔记", "paper note", "文章"
            ]
            if any(kw in query_lower for kw in paper_kws):
                papers = self.read_file(MemoryFile.PAPERS)
                if papers:
                    parts.append(f"## Paper Notes\n{papers}")

            # Projects
            if any(kw in query_lower for kw in ["项目", "project", "代码", "架构"]):
                projects = self.read_file(MemoryFile.PROJECTS)
                if projects:
                    parts.append(f"## Projects\n{projects}")

            # Decisions
            if any(kw in query_lower for kw in ["为什么", "决策", "why did i", "为什么选", "为什么决定"]):
                decisions = self.read_file(MemoryFile.DECISIONS)
                if decisions:
                    parts.append(f"## Decisions\n{decisions}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def _get_memory_summary(self) -> str:
        """Get summary of all memory files for LLM prompt."""
        summaries = []
        for name, file in self.MEMORY_FILES.items():
            content = self.read_file(file)
            if content:
                lines = content.strip().split("\n")
                summaries.append(f"- {name}: {len(lines)} lines")
            else:
                summaries.append(f"- {name}: (empty)")
        return "\n".join(summaries)

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate with incremental operations.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation started (archive_all): total_messages={}", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug("Memory consolidation skipped: {} messages <= keep_count {}", len(session.messages), keep_count)
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                logger.debug("Memory consolidation skipped: no unconsolidated messages")
                return True
            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                logger.debug("Memory consolidation skipped: empty old_messages slice")
                return True
            logger.info("Memory consolidation started: total_messages={}, old_messages={}, keep_count={}",
                        len(session.messages), len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}"
            )

        prompt = f"""Process this conversation and call the save_memory tool.

{MEMORY_FILES_DESC}

## Current Memory State
{self._get_memory_summary()}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            logger.debug("Calling LLM for memory consolidation with {} messages", len(old_messages))
            response = await provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            logger.debug("LLM response tool_calls: {}", len(response.tool_calls))

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            # Find all save_memory tool calls
            save_memory_calls = [tc for tc in response.tool_calls if tc.name == "save_memory"]

            if not save_memory_calls:
                logger.warning("Memory consolidation: no save_memory tool call found")
                return False

            # Process the first save_memory call
            args = save_memory_calls[0].arguments

            # Log if there are additional calls we're ignoring
            if len(save_memory_calls) > 1:
                logger.warning(
                    "Memory consolidation: {} save_memory calls found, processing only the first",
                    len(save_memory_calls)
                )

            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning(
                    "Memory consolidation: unexpected arguments type {}", type(args).__name__
                )
                return False

            # 1. Always append history
            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
                logger.debug("History entry appended: {}", entry[:100])

            # 2. Apply incremental operations atomically
            operations = args.get("operations", [])
            logger.debug("Applying {} memory operations", len(operations))

            for i, op in enumerate(operations):
                op_file = op.get("file", "unknown")
                op_action = op.get("action", "unknown")
                op_content = op.get("content", "")
                content_preview = (op_content or "")[:50] if op_content else ""
                logger.debug("Operation {}/{}: file={}, action={}, content_preview={}",
                            i + 1, len(operations), op_file, op_action, content_preview)

            if not self.apply_operations_atomic(operations):
                logger.warning("Memory consolidation: operations failed")
                return False

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info(
                "Memory consolidation completed: history_entries={}, operations_applied={}, last_consolidated={}",
                len(session.messages), len(operations), session.last_consolidated,
            )
            return True
        except Exception:
            logger.exception("Memory consolidation failed with exception")
            return False


# === Consolidation with RAG Search Prompts ===

COMPRESSION_PROMPT = """将以下对话压缩为关键信息点（5-10 条）。

要求：
- 每条以 "-" 开头
- 聚焦：用户偏好、项目事实、人物关系、重要决定、技术细节
- 忽略：闲聊、感谢、重复内容、礼貌性回复

对话：
{messages}

输出格式：
- 关键点1
- 关键点2
- ..."""

DECISION_PROMPT = """判断如何处理新信息。

## 摘要（压缩后）
{summary}

## 相关记忆（RAG 搜索结果）
{memory_context}

## 判断规则

| action | 场景 |
|--------|------|
| create | 全新事实，与现有记忆无关 |
| merge | 相关内容，补充到现有记忆末尾 |
| replace | 用新信息完全替换旧信息（冲突或过时） |
| skip | 无新信息，或完全重复 |

## 目标文件
根据内容类型选择目标文件：
- profile: 用户偏好、个人习惯、身份信息
- projects: 项目知识、技术栈、开发进度
- papers: 论文笔记、研究记录、学习内容
- decisions: 决策记录、为什么选 A 不选 B
- todos: 当前任务、下一步计划、待办事项

## 冲突处理
- "用户偏好 A" vs "用户偏好 B" → replace，用新的
- "项目用 X 技术" vs "项目改用 Y 技术" → replace，用新的
- "用户喜欢 X" + "用户也喜欢 Y" → merge，追加

## 输出格式（JSON，必须包含在 ```json 和 ``` 之间）
```json
{{
    "action": "create" | "merge" | "replace" | "skip",
    "target_file": "profile" | "projects" | "papers" | "decisions" | "todos",
    "reason": "判断理由（1-2句话）",
    "history_entry": "HISTORY.md 条目（时间戳+事件摘要）",
    "memory_update": "新内容（create 时）或完整内容（replace/merge 时）"
}}
```

关键：
- 必须指定 target_file 字段
- create 时输出要**追加**的新内容（不需要包含已有内容）
- merge 时输出**合并后的完整内容**（结合新旧内容）
- replace 时输出文件的**完整新内容**
- skip 时只需要 action 和 reason"""


class MemoryStoreOptimized:
    """Optimized MemoryStore with RAG search for consolidation."""

    def __init__(self, memory_store: MemoryStore, rag_store: "DocumentStore"):
        """Initialize with existing MemoryStore and RAG store."""
        self._memory = memory_store
        self._rag_store = rag_store

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for LLM processing."""
        lines = []
        for m in messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        return "\n".join(lines)

    async def _compress_messages(self, messages: list[dict], provider: "LLMProvider", model: str) -> str:
        """Step 2: LLM compress messages into summary."""
        formatted = self._format_messages(messages)
        prompt = COMPRESSION_PROMPT.format(messages=formatted)

        response = await provider.chat(
            messages=[
                {"role": "system", "content": "You are a memory consolidation assistant. Compress the conversation into key points."},
                {"role": "user", "content": prompt}
            ],
            model=model,
        )

        return response.content

    async def _search_related_memory(self, summary: str, top_k: int = 3) -> list[str]:
        """Step 3: RAG search related memory."""
        if not self._rag_store:
            return []

        try:
            results = await self._rag_store.search_advanced(summary)
            # Get memory_dir path for filtering (handle cross-platform)
            memory_dir_str = str(self._memory.memory_dir)
            # Normalize path separators for cross-platform compatibility
            memory_dir_str = memory_dir_str.replace("\\", "/").rstrip("/")

            memory_results = []
            memory_dir_prefix = memory_dir_str + "/"  # 确保是目录前缀
            for r in results:
                doc_path = r.document.path.replace("\\", "/").rstrip("/")
                # 使用严格的路径前缀匹配，避免误匹配
                if doc_path.startswith(memory_dir_prefix) or doc_path == memory_dir_str:
                    memory_results.append(r.combined_content)

            return memory_results[:top_k]
        except Exception as e:
            logger.warning("Memory search failed: {}", e)
            return []

    def _infer_target_file(self, summary: str) -> str:
        """根据摘要内容推断目标文件类型。"""
        summary_lower = summary.lower()

        # Papers keywords
        if any(kw in summary_lower for kw in ["论文", "paper", "arxiv", "阅读", "研究", "文献"]):
            return "papers"
        # Projects keywords
        if any(kw in summary_lower for kw in ["项目", "project", "代码", "架构", "开发", "技术", "stack"]):
            return "projects"
        # Decisions keywords
        if any(kw in summary_lower for kw in ["决定", "决策", "选择", "why", "因为", "原因", "instead of"]):
            return "decisions"
        # Todos keywords
        if any(kw in summary_lower for kw in ["任务", "todo", "待办", "下一步", "计划", "plan", "task"]):
            return "todos"
        # Default to profile
        return "profile"

    async def _decide_with_context(
        self,
        summary: str,
        related_memory: list[str],
        provider: "LLMProvider",
        model: str,
    ) -> dict:
        """Step 4: LLM decide with RAG search results."""
        memory_context = "\n\n---\n\n".join([
            f"[相关记忆 {i+1}]\n{m}"
            for i, m in enumerate(related_memory)
        ]) if related_memory else "（无相关记忆）"

        prompt = DECISION_PROMPT.format(
            summary=summary,
            memory_context=memory_context,
        )

        response = await provider.chat(
            messages=[
                {"role": "system", "content": "You are a memory consolidation assistant. Decide how to handle the new information based on related memories."},
                {"role": "user", "content": prompt}
            ],
            model=model,
        )

        # Parse JSON from response
        content = response.content
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the whole response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM decision response")
            return {"action": "skip", "reason": "Parse failed"}

    async def consolidate(
        self,
        session: "Session",
        provider: "LLMProvider",
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """
        Optimized consolidation: compress → RAG search → LLM decide → write.

        Flow:
        1. Get old messages to consolidate
        2. LLM compress into summary (5-10 key points)
        3. RAG search related memory
        4. LLM decide (create/merge/replace/skip)
        5. Write to memory files
        """
        # Step 1: Get old messages
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation started (optimized, archive_all): total_messages={}", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug("Memory consolidation skipped (optimized): {} messages <= keep_count {}", len(session.messages), keep_count)
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                logger.debug("Memory consolidation skipped (optimized): no unconsolidated messages")
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                logger.debug("Memory consolidation skipped (optimized): empty old_messages slice")
                return True
            logger.info("Memory consolidation started (optimized): total_messages={}, old_messages={}, keep_count={}",
                        len(session.messages), len(old_messages), keep_count)

        try:
            # Step 2: LLM compress
            logger.debug("Step 2: Compressing {} messages into summary", len(old_messages))
            summary = await self._compress_messages(old_messages, provider, model)
            logger.debug("Compressed summary: {}", summary[:200])

            # Step 3: RAG search related memory
            logger.debug("Step 3: Searching for related memories")
            related_memory = await self._search_related_memory(summary, top_k=3)
            logger.debug("Found {} related memories", len(related_memory))

            # Step 4: LLM decide (always call LLM, even without related memory)
            logger.debug("Step 4: Deciding memory action with context")
            decision = await self._decide_with_context(summary, related_memory, provider, model)
            logger.debug("LLM decision: action={}, target_file={}", decision.get("action"), decision.get("target_file"))

            # Fallback: if LLM doesn't return target_file, infer it from summary
            if "target_file" not in decision:
                decision["target_file"] = self._infer_target_file(summary)

            # Step 5: Write to memory
            if decision.get("history_entry"):
                self._memory.append_history(decision["history_entry"])
                logger.debug("History entry appended: {}", decision["history_entry"][:100])

            action = decision.get("action", "skip")
            target_file = decision.get("target_file", "profile")

            # Map target_file string to MemoryFile enum
            file_map = {
                "profile": MemoryFile.PROFILE,
                "projects": MemoryFile.PROJECTS,
                "papers": MemoryFile.PAPERS,
                "decisions": MemoryFile.DECISIONS,
                "todos": MemoryFile.TODOS,
            }
            target_memory_file = file_map.get(target_file, MemoryFile.PROFILE)

            if action in ("create", "merge", "replace") and decision.get("memory_update"):
                if action == "create":
                    # append: 追加新内容
                    self._memory.append(target_memory_file, decision["memory_update"])
                    logger.debug("Memory created: appended to {}", target_file)
                elif action == "merge":
                    # LLM 应返回合并后的完整内容
                    self._memory.replace(target_memory_file, decision["memory_update"])
                    logger.debug("Memory merged: replaced {} with merged content", target_file)
                elif action == "replace":
                    # LLM 应返回完整的新内容
                    self._memory.replace(target_memory_file, decision["memory_update"])
                    logger.debug("Memory replaced: {} with new content", target_file)

            # Step 6: Update RAG index for memory changes
            if action in ("create", "merge", "replace"):
                try:
                    # Only index memory directory (efficient incremental update)
                    await self._rag_store.scan_and_index(
                        self._memory.memory_dir,
                        chunk_size=self._rag_store.config.memory_chunk_size,
                        chunk_overlap=self._rag_store.config.memory_chunk_overlap,
                    )
                    logger.info("RAG memory index updated")
                except Exception as e:
                    logger.warning("RAG memory index update failed: {}", e)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info(
                "Memory consolidation completed (optimized): action={}, target_file={}, last_consolidated={}",
                action, target_file, session.last_consolidated,
            )
            return True

        except Exception:
            logger.exception("Memory consolidation failed with exception")
            return False
