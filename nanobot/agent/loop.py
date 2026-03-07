"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.rag import SearchKnowledgeTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        ExecToolConfig,
        RAGConfig,
        ToolsConfig,
    )
    from nanobot.cron.service import CronService


class ConsolidationTrigger:
    """
    记忆固化触发检查器（混合策略）。

    触发条件（满足任一即可）：
    1. 消息窗口（保底）：未固化消息数 ≥ threshold
    2. 会话暂停：超过 N 分钟无新消息 + 已有 M 条消息
    3. 重要内容：最近 N 条消息含关键词 + 已有 K 条消息
    """

    # 默认配置
    DEFAULT_WINDOW_THRESHOLD = 100
    DEFAULT_PAUSE_THRESHOLD_SECONDS = 300  # 5 分钟
    DEFAULT_PAUSE_MIN_MESSAGES = 10
    DEFAULT_IMPORTANT_KEYWORDS = [
        "决定", "decision", "结论", "conclusion",
        "todo", "任务", "task", "计划", "plan",
        "项目", "project", "架构", "architecture",
        "记住", "remember", "note", "笔记",
        # 新增：偏好表达关键词
        "我喜欢", "我不喜欢", "我想要", "我不想要",
        "我希望", "我不希望", "prefer", "I like", "I don't like",
        "changed my mind", "my preference",
    ]
    DEFAULT_IMPORTANT_CHECK_WINDOW = 5  # 检查最近 5 条
    DEFAULT_IMPORTANT_MIN_MESSAGES = 2  # 至少有 2 条才触发

    def __init__(
        self,
        window_threshold: int = DEFAULT_WINDOW_THRESHOLD,
        pause_threshold_seconds: float = DEFAULT_PAUSE_THRESHOLD_SECONDS,
        pause_min_messages: int = DEFAULT_PAUSE_MIN_MESSAGES,
        important_keywords: list[str] | None = None,
        important_check_window: int = DEFAULT_IMPORTANT_CHECK_WINDOW,
        important_min_messages: int = DEFAULT_IMPORTANT_MIN_MESSAGES,
    ):
        self.window_threshold = window_threshold
        self.pause_threshold_seconds = pause_threshold_seconds
        self.pause_min_messages = pause_min_messages
        self.important_keywords = important_keywords or self.DEFAULT_IMPORTANT_KEYWORDS
        self.important_check_window = important_check_window
        self.important_min_messages = important_min_messages

    def should_trigger(
        self,
        session: Session,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """
        检查是否应该触发固化。

        Args:
            session: 会话对象
            now: 当前时间戳（用于测试）

        Returns:
            (should_trigger: bool, reason: str)
        """
        now = now or time.time()
        unconsolidated = len(session.messages) - session.last_consolidated

        # 条件 1：消息窗口（保底）
        if unconsolidated >= self.window_threshold:
            return True, f"message window reached ({unconsolidated} messages)"

        # 没有未固化消息，直接返回
        if unconsolidated == 0:
            return False, "no unconsolidated messages"

        # 条件 2：会话暂停
        pause_reason = self._check_pause(session, unconsolidated, now)
        if pause_reason:
            return True, pause_reason

        # 条件 3：重要内容检测
        important_reason = self._check_important_content(session, unconsolidated)
        if important_reason:
            return True, important_reason

        return False, "no trigger condition met"

    def _check_pause(self, session: Session, unconsolidated: int, now: float) -> str | None:
        """检查会话暂停条件。"""
        if unconsolidated < self.pause_min_messages:
            return None

        # 获取最后一条消息的时间
        last_message_time = self._get_last_message_time(session, now)
        if now - last_message_time > self.pause_threshold_seconds:
            return f"session pause ({int(now - last_message_time)}s idle)"

        return None

    def _check_important_content(self, session: Session, unconsolidated: int) -> str | None:
        """检查重要内容条件。"""
        if unconsolidated < self.important_min_messages:
            return None

        # 检查最近 N 条消息
        start_idx = max(session.last_consolidated, len(session.messages) - self.important_check_window)
        recent_messages = session.messages[start_idx:]

        for msg in recent_messages:
            content = msg.get("content", "").lower()
            for kw in self.important_keywords:
                if kw.lower() in content:
                    return f"important content detected (keyword: {kw})"

        return None

    def _get_last_message_time(self, session: Session, now: float) -> float:
        """获取最后一条消息的时间戳。"""
        if not session.messages:
            return now

        last_msg = session.messages[-1]
        ts_str = last_msg.get("timestamp")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                return dt.timestamp()
            except (ValueError, TypeError):
                pass

        return now


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        memory_model: str | None = None,  # Optional: separate model for memory consolidation
        subagent_model: str | None = None,  # Optional: separate model for subagents
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        serper_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = True,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        rag_config: RAGConfig | None = None,
        tools_config: ToolsConfig | None = None,
        # 新增：固化触发配置
        consolidation_pause_threshold_seconds: float = ConsolidationTrigger.DEFAULT_PAUSE_THRESHOLD_SECONDS,
        consolidation_pause_min_messages: int = ConsolidationTrigger.DEFAULT_PAUSE_MIN_MESSAGES,
        consolidation_important_check_window: int = ConsolidationTrigger.DEFAULT_IMPORTANT_CHECK_WINDOW,
        consolidation_important_min_messages: int = ConsolidationTrigger.DEFAULT_IMPORTANT_MIN_MESSAGES,
        consolidation_timeout_seconds: int = 90,
    ):
        from nanobot.config.schema import ExecToolConfig, RAGConfig, ToolsConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        # memory_model defaults to model if not specified
        self.memory_model = memory_model or self.model
        # subagent_model defaults to model if not specified
        self.subagent_model = subagent_model or self.model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.serper_api_key = serper_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.rag_config = rag_config or RAGConfig()
        self.tools_config = tools_config or ToolsConfig()
        self.consolidation_timeout_seconds = consolidation_timeout_seconds

        # Create embedding provider for memory dedup (if RAG dependencies available)
        self._embedding_provider = None
        if self.rag_config.enabled:
            try:
                from nanobot.rag import SentenceTransformerEmbeddingProvider
                self._embedding_provider = SentenceTransformerEmbeddingProvider(
                    self.rag_config.embedding_model
                )
            except ImportError:
                logger.debug("RAG dependencies not installed, semantic dedup disabled")

        # Create MemoryStore instance (shared between context builder and consolidation)
        self._memory_store = MemoryStore(workspace, self._embedding_provider)

        self.context = ContextBuilder(workspace, self._embedding_provider, memory_store=self._memory_store)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry(default_timeout=self.tools_config.default_tool_timeout)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.subagent_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            serper_api_key=serper_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            default_tool_timeout=self.tools_config.default_tool_timeout,
            rag_config=self.rag_config,
            shared_doc_store=None,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._mcp_retry_count = 0
        self._mcp_retry_delay = 1.0  # Initial delay
        self._mcp_max_retry_delay = 300.0  # Max 5 minutes
        self._mcp_last_failure_time: float = 0
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._consolidating_set_lock = asyncio.Lock()  # Protects _consolidating set
        self._session_locks: dict[str, asyncio.Lock] = {}  # Per-session locks for message processing
        self._session_locks_lock = asyncio.Lock()  # Protects _session_locks dict
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._rag_initialized = False

        # 新增：固化触发配置
        self.consolidation_pause_threshold_seconds = consolidation_pause_threshold_seconds
        self.consolidation_pause_min_messages = consolidation_pause_min_messages
        self.consolidation_important_check_window = consolidation_important_check_window
        self.consolidation_important_min_messages = consolidation_important_min_messages

        # 新增：触发检查器
        self._consolidation_trigger = ConsolidationTrigger(
            window_threshold=self.memory_window,
            pause_threshold_seconds=self.consolidation_pause_threshold_seconds,
            pause_min_messages=self.consolidation_pause_min_messages,
            important_check_window=self.consolidation_important_check_window,
            important_min_messages=self.consolidation_important_min_messages,
        )

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        self._retrieve_tool = None  # 先初始化为 None
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.tools_config.default_tool_timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
            use_firejail=getattr(self.exec_config, "use_firejail", True),
            firejail_strict=getattr(self.exec_config, "firejail_strict", True),
            firejail_options=getattr(self.exec_config, "firejail_options", None),
            firejail_net=getattr(self.exec_config, "firejail_net", "unrestricted"),
        ))
        self.tools.register(WebSearchTool(api_key=self.serper_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        # Register RAG tool if enabled
        if self.rag_config.enabled:
            try:
                retrieve_tool = SearchKnowledgeTool(
                    workspace=self.workspace,
                    chunk_size=self.rag_config.max_chunk_size,
                    chunk_overlap=int(self.rag_config.max_chunk_size * self.rag_config.chunk_overlap_ratio) if self.rag_config.max_chunk_size > 0 else 200,
                    embedding_model=self.rag_config.embedding_model,
                    rag_config=self.rag_config,
                )
                self.tools.register(retrieve_tool)
                self._retrieve_tool = retrieve_tool
            except ImportError as e:
                logger.warning("RAG dependencies not installed, skipping retrieve tool: {}", e)
                self._retrieve_tool = None
        else:
            self._retrieve_tool = None

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers with exponential backoff."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return

        # Check if we should wait before retrying
        if self._mcp_last_failure_time > 0:
            elapsed = time.time() - self._mcp_last_failure_time
            if elapsed < self._mcp_retry_delay:
                logger.debug("MCP connection retry cooldown: {:.1f}s remaining",
                            self._mcp_retry_delay - elapsed)
                return

        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
            self._mcp_retry_count = 0  # Reset on success
            logger.info("MCP servers connected successfully")
        except Exception as e:
            self._mcp_retry_count += 1
            self._mcp_last_failure_time = time.time()
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 300s...
            self._mcp_retry_delay = min(
                self._mcp_max_retry_delay,
                self._mcp_retry_delay * 2
            )
            logger.error(
                "MCP connection failed (retry {} in {:.1f}s): {}",
                self._mcp_retry_count, self._mcp_retry_delay, e
            )
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    async def _init_rag(self) -> None:
        """Initialize RAG and scan documents on first use."""
        if self._rag_initialized:
            return
        if not self._retrieve_tool:
            return

        self._rag_initialized = True

        # Always ensure DocumentStore is created (for sharing with subagents)
        # Accessing doc_store property ensures initialization
        _ = self._retrieve_tool.doc_store

        if not self.rag_config.auto_scan_on_startup:
            # Share DocumentStore with subagents even if auto_scan is disabled
            if self._retrieve_tool.doc_store:
                self.subagents.set_shared_doc_store(self._retrieve_tool.doc_store)
            return

        try:
            logger.info("Scanning documents for RAG...")
            stats = await self._retrieve_tool.scan_and_index()
            total = stats["added"] + stats["updated"]
            if total > 0:
                logger.info(
                    "RAG scan complete: +{} added/updated, -{} deleted",
                    total,
                    stats["deleted"],
                )
            else:
                logger.info("RAG scan complete: no changes")
            # Share DocumentStore with subagents after successful initialization
            if self._retrieve_tool.doc_store:
                self.subagents.set_shared_doc_store(self._retrieve_tool.doc_store)
        except Exception as e:
            logger.error("Failed to scan documents for RAG: {}", e)

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        await self._init_rag()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(self._make_task_cleanup(msg.session_key))

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the per-session lock."""
        lock = await self._get_session_lock(msg.session_key)
        async with lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections and cleanup session locks."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

        # Cleanup session locks
        self._session_locks.clear()

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
        # Close RAG tool if it exists
        if self._retrieve_tool is not None:
            try:
                self._retrieve_tool.close()
                logger.debug("RAG tool closed")
            except Exception as e:
                logger.warning("Error closing RAG tool: {}", e)

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    async def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Get or create a lock for the given session."""
        async with self._session_locks_lock:
            if session_key not in self._session_locks:
                self._session_locks[session_key] = asyncio.Lock()
            return self._session_locks[session_key]

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    def _make_task_cleanup(self, session_key: str) -> Callable[[asyncio.Task], None]:
        """Create a cleanup callback for task completion."""
        def _cleanup(task: asyncio.Task) -> None:
            """Remove task from active tasks list."""
            tasks = self._active_tasks.get(session_key, [])
            if task in tasks:
                tasks.remove(task)
                if not tasks:
                    # Clean up empty list
                    self._active_tasks.pop(session_key, None)
        return _cleanup

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            if self._retrieve_tool:
                self._retrieve_tool._last_results = None
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            async with self._consolidating_set_lock:
                self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                async with self._consolidating_set_lock:
                    self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        trigger_consolidation = False
        async with self._consolidating_set_lock:
            if session.key not in self._consolidating:
                should_trigger, reason = self._consolidation_trigger.should_trigger(session)
                if should_trigger:
                    logger.debug("Consolidation trigger: {}", reason)
                    self._consolidating.add(session.key)
                    trigger_consolidation = True

        if trigger_consolidation:
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with asyncio.timeout(self.consolidation_timeout_seconds):
                        async with lock:
                            await self._consolidate_memory(session)
                    logger.info("Memory consolidation completed for session {}", session.key)
                except asyncio.TimeoutError:
                    logger.error("Memory consolidation timed out after {}s for session {}",
                                 self.consolidation_timeout_seconds, session.key)
                except Exception as e:
                    logger.error("Memory consolidation failed for session {}: {}",
                                 session.key, e, exc_info=True)
                finally:
                    async with self._consolidating_set_lock:
                        self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)

            _task = asyncio.create_task(_consolidate_and_unlock())
            _task.add_done_callback(lambda task: self._consolidation_tasks.discard(task))
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        if self._retrieve_tool:
            self._retrieve_tool._last_results = None
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    _TOOL_RESULT_MAX_CHARS = 500

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            if entry.get("role") == "user" and isinstance(entry.get("content"), list):
                entry["content"] = [
                    {"type": "text", "text": "[image]"} if (
                        c.get("type") == "image_url"
                        and c.get("image_url", {}).get("url", "").startswith("data:image/")
                    ) else c
                    for c in entry["content"]
                ]
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session: Session, archive_all: bool = False) -> bool:
        """
        Delegate to MemoryStore.consolidate() with RAG-based strategy.

        Returns True on success, False on failure.
        """
        # Try to use RAG-based consolidation if RAG tool is available
        if self._retrieve_tool is not None:
            try:
                # Ensure RAG tool is initialized (doc_store property handles this)
                rag_store = self._retrieve_tool.doc_store

                if rag_store is not None:
                    # Update the memory store's rag_store reference for RAG-based consolidation
                    self._memory_store._rag_store = rag_store
                    return await self._memory_store.consolidate(
                        session, self.provider, self.memory_model,
                        archive_all=archive_all, memory_window=self.memory_window,
                        use_rag=True,  # Use RAG-based consolidation
                    )
            except Exception as e:
                logger.warning("RAG-based consolidation failed, falling back to direct: {}", e)

        # Fallback to direct consolidation (without RAG search)
        return await self._memory_store.consolidate(
            session, self.provider, self.memory_model,
            archive_all=archive_all, memory_window=self.memory_window,
            use_rag=False,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        await self._init_rag()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
