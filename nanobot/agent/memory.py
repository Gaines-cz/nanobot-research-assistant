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
        self._rag_store = rag_store  # RAG store for memory search (enables RAG-based consolidation)

    # === Safe Write Helper ===

    def _safe_write(self, path: Path, content: str) -> None:
        """
        安全写入文件（临时文件 + rename，原子操作）。

        Args:
            path: 目标文件路径
            content: 要写入的内容

        Raises:
            Exception: 写入失败时抛出
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            # 先写临时文件
            tmp_path.write_text(content, encoding="utf-8")
            # 原子替换
            tmp_path.rename(path)
        except Exception:
            # 清理临时文件
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass  # 清理失败不影响主错误
            raise

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
        self._safe_write(path, new_content)

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
            # 将已有内容按段落分割
            paragraphs = [p.strip() for p in existing.split("\n\n") if p.strip()]
            if not paragraphs:
                return False

            # 性能优化：限制最大段落数，避免批量过大
            # 如果段落过多，只检查前 100 个段落（覆盖绝大多数场景）
            max_paragraphs = 100
            if len(paragraphs) > max_paragraphs:
                logger.debug(
                    "Paragraph count {} exceeds limit {}, truncating for performance",
                    len(paragraphs), max_paragraphs
                )
                paragraphs = paragraphs[:max_paragraphs]

            # 批量 embedding：一次性获取所有段落和内容的 embedding
            # 性能提升：从 O(n) 次 API 调用降至 O(1) 次
            all_texts = [content] + paragraphs
            embeddings = await self._embedding_provider.embed_batch(all_texts)

            content_embedding = embeddings[0]
            para_embeddings = embeddings[1:]

            # 并行计算相似度
            for para_embedding in para_embeddings:
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
        from nanobot.utils.helpers import cosine_similarity
        return cosine_similarity(a, b)

    def prepend(self, file: MemoryFile, content: str) -> None:
        """Prepend content to file."""
        path = self.memory_dir / file.value
        existing = self.read_file(file)
        if existing:
            new_content = content.strip() + "\n\n" + existing
        else:
            new_content = content.strip() + "\n"
        self._safe_write(path, new_content)

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
        self._safe_write(path, new_content)

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
        self._safe_write(path, content.strip() + "\n")

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
        loaded_files = []

        # Always load profile
        profile = self.read_file(MemoryFile.PROFILE)
        if profile:
            parts.append(f"## Profile\n{profile}")
            loaded_files.append(f"profile={len(profile)}")

        # Load todos by default (current tasks)
        todos = self.read_file(MemoryFile.TODOS)
        if todos:
            parts.append(f"## Current Tasks\n{todos}")
            loaded_files.append(f"todos={len(todos)}")

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
                    loaded_files.append(f"papers={len(papers)}")

            # Projects
            if any(kw in query_lower for kw in ["项目", "project", "代码", "架构"]):
                projects = self.read_file(MemoryFile.PROJECTS)
                if projects:
                    parts.append(f"## Projects\n{projects}")
                    loaded_files.append(f"projects={len(projects)}")

            # Decisions
            if any(kw in query_lower for kw in ["为什么", "决策", "why did i", "为什么选", "为什么决定"]):
                decisions = self.read_file(MemoryFile.DECISIONS)
                if decisions:
                    parts.append(f"## Decisions\n{decisions}")
                    loaded_files.append(f"decisions={len(decisions)}")

        result = "\n\n---\n\n".join(parts) if parts else ""

        # 记录加载的 memory 文件及其长度
        if loaded_files:
            logger.debug("Memory context loaded: {}, total={}", ", ".join(loaded_files), len(result))

        return result

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

    def _get_messages_to_consolidate(
        self,
        session: Session,
        archive_all: bool,
        memory_window: int,
    ) -> tuple[list[dict], int]:
        """
        获取需要固化的消息。

        Returns:
            (old_messages, keep_count): 要固化的消息列表，保留的消息数量
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation started (archive_all): total_messages={}", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug("Memory consolidation skipped: {} messages <= keep_count {}", len(session.messages), keep_count)
                return [], 0
            if len(session.messages) - session.last_consolidated <= 0:
                logger.debug("Memory consolidation skipped: no unconsolidated messages")
                return [], 0
            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                logger.debug("Memory consolidation skipped: empty old_messages slice")
                return [], 0
            logger.info("Memory consolidation started: total_messages={}, old_messages={}, keep_count={}",
                        len(session.messages), len(old_messages), keep_count)

        return old_messages, keep_count

    def _process_save_memory_tool_call(
        self,
        response,
    ) -> tuple[dict, list[dict]] | None:
        """
        处理 save_memory tool call，提取 arguments。

        Returns:
            (args, operations) 或 None（如果没有有效 tool call）
        """
        if not response.has_tool_calls:
            content_preview = (response.content[:200] + "...") if response.content else "None"
            logger.warning(
                "Memory consolidation: LLM did not call save_memory, skipping. "
                "finish_reason={}, content_preview={}",
                response.finish_reason,
                content_preview
            )
            return None

        # Find all save_memory tool calls
        save_memory_calls = [tc for tc in response.tool_calls if tc.name == "save_memory"]

        if not save_memory_calls:
            logger.warning("Memory consolidation: no save_memory tool call found")
            return None

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
            return None

        operations = args.get("operations", [])
        return args, operations

    async def _apply_save_memory_operations(
        self,
        args: dict,
        operations: list[dict],
        session: Session,
        keep_count: int,
        archive_all: bool,
        is_rag_mode: bool = False,
    ) -> bool:
        """
        应用 save_memory tool 的操作。

        Args:
            args: tool call 的 arguments
            operations: 要应用的操作列表
            session: Session 对象
            keep_count: 保留的消息数量
            archive_all: 是否归档所有消息
            is_rag_mode: 是否是 RAG 模式（需要更新 RAG 索引）

        Returns:
            True 成功，False 失败
        """
        # 1. Always append history
        modified_files = set()
        if entry := args.get("history_entry"):
            if not isinstance(entry, str):
                entry = json.dumps(entry, ensure_ascii=False)
            self.append_history(entry)
            logger.debug("History entry appended: {}", entry[:100])

        # 2. Apply incremental operations atomically
        logger.debug("Applying {} memory operations", len(operations))

        for i, op in enumerate(operations):
            op_file = op.get("file", "unknown")
            op_action = op.get("action", "unknown")
            op_content = op.get("content", "")
            content_preview = (op_content or "")[:50] if op_content else ""
            logger.debug("Operation {}/{}: file={}, action={}, content_preview={}",
                        i + 1, len(operations), op_file, op_action, content_preview)
            if op_action != "skip":
                modified_files.add(op_file)

        if not self.apply_operations_atomic(operations):
            logger.warning("Memory consolidation: operations failed")
            return False

        # 3. 更新 RAG 索引（只索引修改的文件，仅 RAG 模式）
        rag_update_failed = False
        if is_rag_mode and modified_files and self._rag_store:
            for file_name in modified_files:
                if file_name in self.MEMORY_FILES:
                    target_memory_file = self.MEMORY_FILES[file_name]
                    if not self._rag_store.config.enable_memory_index:
                        logger.debug("Memory index update skipped (enable_memory_index=False)")
                    else:
                        try:
                            file_path = self.memory_dir / target_memory_file.value
                            await self._rag_store.index_single_file(
                                file_path,
                                chunk_size=self._rag_store.config.memory_chunk_size,
                                chunk_overlap_ratio=self._rag_store.config.memory_chunk_overlap_ratio,
                            )
                            logger.info("RAG memory index updated for: {}", target_memory_file.value)
                        except Exception as e:
                            logger.warning("RAG memory index update failed: {}", e)
                            rag_update_failed = True  # 标记失败

        # 4. 更新 session 状态（仅当 RAG 更新成功时）
        if not is_rag_mode or not rag_update_failed:
            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info(
                "Memory consolidation completed: operations={}, modified_files={}, last_consolidated={}",
                len(operations), len(modified_files), session.last_consolidated,
            )
        else:
            logger.warning(
                "Memory consolidation completed but RAG index update failed, "
                "last_consolidated not updated (will retry next time)"
            )
        return True

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
        use_rag: bool = True,
    ) -> bool:
        """
        Consolidate memory with incremental operations.

        Two strategies available:
        - **RAG-based** (use_rag=True): RAG search → LLM with save_memory tool.
          More accurate for avoiding duplicates and maintaining consistency.
        - **Direct** (use_rag=False): LLM directly decides operations.
          Faster but may create duplicates.

        Args:
            session: Session to consolidate
            provider: LLM provider
            model: Model to use
            archive_all: If True, archive all messages (default: keep recent half)
            memory_window: Number of recent messages to keep
            use_rag: If True, use RAG-based consolidation (requires _rag_store)

        Returns:
            True on success, False on failure
        """
        # Choose consolidation strategy
        if use_rag and self._rag_store is not None:
            return await self._consolidate_with_rag(
                session, provider, model,
                archive_all=archive_all, memory_window=memory_window,
            )
        else:
            if use_rag and self._rag_store is None:
                logger.warning("RAG-based consolidation requested but _rag_store is None, falling back to direct method")
            return await self._consolidate_direct(
                session, provider, model,
                archive_all=archive_all, memory_window=memory_window,
            )

    async def _consolidate_direct(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Direct consolidation: LLM directly decides operations without RAG search."""
        # Step 1: 获取要固化的消息（复用共享逻辑）
        old_messages, keep_count = self._get_messages_to_consolidate(
            session, archive_all, memory_window
        )
        if not old_messages:
            return True

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
            logger.debug("Calling LLM for memory consolidation: model={}, messages={}, prompt_len={}",
                       model, len(old_messages), len(prompt))
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

            logger.debug("LLM response: tool_calls={}, finish_reason={}, has_content={}",
                       len(response.tool_calls), response.finish_reason, response.content is not None)

            # 复用共享的 tool call 处理逻辑
            result = self._process_save_memory_tool_call(response)
            if result is None:
                return False
            args, operations = result

            # 复用共享的 apply 操作逻辑
            return await self._apply_save_memory_operations(
                args, operations, session, keep_count, archive_all, is_rag_mode=False
            )
        except Exception:
            logger.exception("Memory consolidation failed with exception")
            return False

    async def _search_related_memory_enhanced(
        self,
        messages: list[dict],
        top_k: int = 3,
    ) -> tuple[list[str], str]:
        """
        增强的 RAG 搜索：
        1. 格式化消息
        2. 用完整消息搜索（信息更全）
        3. 返回相关记忆 + 格式化后的消息

        Returns:
            (related_memory, formatted_messages): 相关记忆列表，格式化后的消息
        """
        if not self._rag_store:
            return [], ""

        # 格式化消息用于搜索和 LLM
        formatted = self._format_messages(messages)

        # 用完整消息搜索（不压缩，信息更全）
        try:
            results = await self._rag_store.search_advanced(formatted)
            memory_path = self.memory_dir.resolve()

            memory_results = []
            for r in results:
                doc_path = Path(r.document.path).resolve()
                if doc_path == memory_path or memory_path in doc_path.parents:
                    memory_results.append(r.combined_content)

            return memory_results[:top_k], formatted
        except Exception as e:
            logger.warning("Memory search failed: {}", e)
            return [], formatted

    async def _consolidate_with_rag(
        self,
        session: "Session",
        provider: "LLMProvider",
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """
        优化后的 RAG-based consolidation：
        1. 获取要固化的消息
        2. RAG 搜索相关记忆
        3. LLM 单次调用，直接用 save_memory tool

        优势：
        - 1 次 LLM 调用（之前 2 次）
        - 支持 update_section
        - 支持多文件操作
        - 代码更简洁
        """
        # Step 1: 获取要固化的消息
        old_messages, keep_count = self._get_messages_to_consolidate(
            session, archive_all, memory_window
        )
        if not old_messages:
            return True

        try:
            # Step 2: RAG 搜索相关记忆
            logger.debug("Step 2: Searching for related memories")
            related_memory, formatted_messages = await self._search_related_memory_enhanced(
                old_messages, top_k=3
            )
            logger.debug("Found {} related memories", len(related_memory))

            # Step 3: 构建带相关记忆的 prompt，单次 LLM 调用
            logger.debug("Step 3: Calling LLM with save_memory tool")

            memory_context = "\n\n---\n\n".join([
                f"[Related Memory {i+1}]\n{m}"
                for i, m in enumerate(related_memory)
            ]) if related_memory else "(No related memories)"

            prompt = f"""Process this conversation and call the save_memory tool.

{MEMORY_FILES_DESC}

## Current Memory State
{self._get_memory_summary()}

## Related Memories (from RAG search)
{memory_context}

## Conversation to Process
{formatted_messages}

Important: Use the related memories to avoid duplication and ensure consistency.
You can use update_section for granular updates, and multiple operations if needed."""

            logger.debug("Calling LLM for RAG memory consolidation: model={}, prompt_len={}",
                       model, len(prompt))
            response = await provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation assistant. Use the save_memory tool with the conversation and related memories.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            logger.debug("LLM response: tool_calls={}, finish_reason={}, has_content={}",
                       len(response.tool_calls), response.finish_reason, response.content is not None)

            # Step 4: 处理 tool call
            result = self._process_save_memory_tool_call(response)
            if result is None:
                return False
            args, operations = result

            # Step 5: 应用操作
            return await self._apply_save_memory_operations(
                args, operations, session, keep_count, archive_all, is_rag_mode=True
            )

        except Exception:
            logger.exception("Memory consolidation failed with exception")
            return False

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for LLM processing."""
        lines = []
        for m in messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        return "\n".join(lines)
