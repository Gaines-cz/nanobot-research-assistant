"""Test MemoryStore.consolidate() handles non-string tool call arguments.

Regression test for https://github.com/HKUDS/nanobot/issues/1042
When memory consolidation receives dict values instead of strings from the LLM
tool call response, it should serialize them to JSON instead of raising TypeError.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore, MemoryFile
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_session(message_count: int = 30, memory_window: int = 50):
    """Create a mock session with messages."""
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]
    session.last_consolidated = 0
    return session


def _make_tool_response(history_entry, operations=None):
    """Create an LLMResponse with a save_memory tool call."""
    args = {"history_entry": history_entry}
    if operations:
        args["operations"] = operations

    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments=args,
            )
        ],
    )


class TestMemoryConsolidationTypeHandling:
    """Test that consolidation handles various argument types correctly."""

    @pytest.mark.asyncio
    async def test_string_arguments_work(self, tmp_path: Path) -> None:
        """Normal case: LLM returns string arguments."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                history_entry="[2026-01-01] User discussed testing.",
                operations=[{"file": "profile", "action": "replace", "content": "# Memory\nUser likes testing."}],
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        history_path = store.memory_dir / MemoryFile.HISTORY.value
        profile_path = store.memory_dir / MemoryFile.PROFILE.value
        assert history_path.exists()
        assert "[2026-01-01] User discussed testing." in history_path.read_text()
        assert "User likes testing." in profile_path.read_text()

    @pytest.mark.asyncio
    async def test_dict_arguments_serialized_to_json(self, tmp_path: Path) -> None:
        """Issue #1042: LLM returns dict instead of string — must not raise TypeError."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                history_entry={"timestamp": "2026-01-01", "summary": "User discussed testing."},
                operations=[{"file": "profile", "action": "replace", "content": "# Memory\nUser likes testing."}],
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        history_path = store.memory_dir / MemoryFile.HISTORY.value
        assert history_path.exists()
        history_content = history_path.read_text()
        parsed = json.loads(history_content.strip())
        assert parsed["summary"] == "User discussed testing."

    @pytest.mark.asyncio
    async def test_string_arguments_as_raw_json(self, tmp_path: Path) -> None:
        """Some providers return arguments as a JSON string instead of parsed dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        # Simulate arguments being a JSON string (not yet parsed)
        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=json.dumps({
                        "history_entry": "[2026-01-01] User discussed testing.",
                        "operations": [{"file": "profile", "action": "replace", "content": "# Memory\nUser likes testing."}],
                    }),
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        history_path = store.memory_dir / MemoryFile.HISTORY.value
        assert "User discussed testing." in history_path.read_text()

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_false(self, tmp_path: Path) -> None:
        """When LLM doesn't use the save_memory tool, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(content="I summarized the conversation.", tool_calls=[])
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is False
        history_path = store.memory_dir / MemoryFile.HISTORY.value
        assert not history_path.exists()

    @pytest.mark.asyncio
    async def test_skips_when_few_messages(self, tmp_path: Path) -> None:
        """Consolidation should be a no-op when messages < keep_count."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        session = _make_session(message_count=10)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        provider.chat.assert_not_called()


class TestMemoryIncrementalOperations:
    """Test incremental memory operations."""

    def test_append_to_empty_file(self, tmp_path: Path) -> None:
        """Append to empty file should create content without extra newlines."""
        store = MemoryStore(tmp_path)
        store.append(MemoryFile.PAPERS, "## Attention Is All You Need\nKey finding: Transformers work.")

        content = store.read_file(MemoryFile.PAPERS)
        assert content.startswith("## Attention Is All You Need")
        assert "Transformers work" in content
        assert not content.startswith("\n")

    def test_append_to_existing_file(self, tmp_path: Path) -> None:
        """Append should add separator between existing and new content."""
        store = MemoryStore(tmp_path)
        store.append(MemoryFile.PAPERS, "## Paper 1\nContent 1")
        store.append(MemoryFile.PAPERS, "## Paper 2\nContent 2")

        content = store.read_file(MemoryFile.PAPERS)
        assert "## Paper 1" in content
        assert "## Paper 2" in content
        # Should have separator between entries
        assert "Content 1\n\n## Paper 2" in content

    def test_prepend_to_empty_file(self, tmp_path: Path) -> None:
        """Prepend to empty file should create content without extra newlines."""
        store = MemoryStore(tmp_path)
        store.prepend(MemoryFile.TODOS, "- Task 1")

        content = store.read_file(MemoryFile.TODOS)
        assert content.strip() == "- Task 1"

    def test_prepend_to_existing_file(self, tmp_path: Path) -> None:
        """Prepend should add new content at the beginning."""
        store = MemoryStore(tmp_path)
        store.prepend(MemoryFile.TODOS, "- Task 2")
        store.prepend(MemoryFile.TODOS, "- Task 1 (urgent)")

        content = store.read_file(MemoryFile.TODOS)
        assert content.startswith("- Task 1 (urgent)")
        assert "Task 2" in content

    def test_replace_entire_file(self, tmp_path: Path) -> None:
        """Replace should overwrite entire file content."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.TODOS, "- Old task")
        store.replace(MemoryFile.TODOS, "- New task 1\n- New task 2")

        content = store.read_file(MemoryFile.TODOS)
        assert "Old task" not in content
        assert "- New task 1" in content
        assert "- New task 2" in content

    def test_update_section_creates_new(self, tmp_path: Path) -> None:
        """update_section should create section if it doesn't exist."""
        store = MemoryStore(tmp_path)
        store.update_section(MemoryFile.PROJECTS, "nanobot", "Python agent framework\nUses LiteLLM")

        content = store.read_file(MemoryFile.PROJECTS)
        assert "## nanobot" in content
        assert "Python agent framework" in content

    def test_update_section_replaces_existing(self, tmp_path: Path) -> None:
        """update_section should replace existing section content."""
        store = MemoryStore(tmp_path)
        store.update_section(MemoryFile.PROJECTS, "nanobot", "Old description")
        store.update_section(MemoryFile.PROJECTS, "nanobot", "New description")

        content = store.read_file(MemoryFile.PROJECTS)
        assert "Old description" not in content
        assert "New description" in content
        # Should only appear once
        assert content.count("## nanobot") == 1

    def test_update_section_preserves_other_sections(self, tmp_path: Path) -> None:
        """update_section should not affect other sections."""
        store = MemoryStore(tmp_path)
        store.update_section(MemoryFile.PROJECTS, "Project A", "Description A")
        store.update_section(MemoryFile.PROJECTS, "Project B", "Description B")
        store.update_section(MemoryFile.PROJECTS, "Project A", "Updated A")

        content = store.read_file(MemoryFile.PROJECTS)
        assert "Updated A" in content
        assert "Description B" in content
        assert "Description A" not in content

    def test_skip_action_does_nothing(self, tmp_path: Path) -> None:
        """Skip action should not modify the file."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.PROFILE, "Original content")

        from nanobot.agent.memory import MemoryOperation
        store.apply_operation(MemoryOperation(file=MemoryFile.PROFILE, action="skip"))

        content = store.read_file(MemoryFile.PROFILE)
        assert content.strip() == "Original content"


class TestMemoryContextLoading:
    """Test selective memory loading based on query."""

    def test_default_loads_profile_and_todos(self, tmp_path: Path) -> None:
        """Without query, should load PROFILE and TODOS by default."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.PROFILE, "User preferences")
        store.replace(MemoryFile.TODOS, "Current tasks")
        store.replace(MemoryFile.PAPERS, "Paper notes")

        context = store.get_memory_context()
        assert "User preferences" in context
        assert "Current tasks" in context
        assert "Paper notes" not in context

    def test_query_triggers_papers(self, tmp_path: Path) -> None:
        """Query with paper keywords should load PAPERS."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.PROFILE, "User preferences")
        store.replace(MemoryFile.PAPERS, "Transformer paper notes")

        context = store.get_memory_context("我读过什么论文？")
        assert "Transformer paper notes" in context

        context = store.get_memory_context("What papers have I read?")
        assert "Transformer paper notes" in context

    def test_query_triggers_projects(self, tmp_path: Path) -> None:
        """Query with project keywords should load PROJECTS."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.PROJECTS, "nanobot architecture")

        context = store.get_memory_context("介绍一下项目架构")
        assert "nanobot architecture" in context

        context = store.get_memory_context("Tell me about my project")
        assert "nanobot architecture" in context

    def test_query_triggers_decisions(self, tmp_path: Path) -> None:
        """Query with decision keywords should load DECISIONS."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.DECISIONS, "Chose Python over Node.js")

        context = store.get_memory_context("为什么选择 Python？")
        assert "Chose Python over Node.js" in context

        context = store.get_memory_context("Why did I make this decision?")
        assert "Chose Python over Node.js" in context

    def test_empty_files_not_loaded(self, tmp_path: Path) -> None:
        """Empty files should not be included in context."""
        store = MemoryStore(tmp_path)
        # Don't create any files

        context = store.get_memory_context()
        assert context == ""

    def test_get_memory_summary(self, tmp_path: Path) -> None:
        """Memory summary should show line counts."""
        store = MemoryStore(tmp_path)
        store.replace(MemoryFile.PROFILE, "Line 1\nLine 2\nLine 3")

        summary = store._get_memory_summary()
        assert "profile: 3 lines" in summary
        assert "papers: (empty)" in summary


class TestMemoryDeduplication:
    """Test append_with_dedup for exact and semantic deduplication."""

    @pytest.mark.asyncio
    async def test_append_with_dedup_exact_match(self, tmp_path: Path) -> None:
        """精确匹配去重：相同内容不会重复追加"""
        store = MemoryStore(tmp_path)
        store.append(MemoryFile.PAPERS, "## Paper 1\nContent here")

        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "## Paper 1\nContent here"
        )

        assert result is False
        content = store.read_file(MemoryFile.PAPERS)
        assert content.count("Paper 1") == 1

    @pytest.mark.asyncio
    async def test_append_with_dedup_new_content(self, tmp_path: Path) -> None:
        """新内容正常追加"""
        store = MemoryStore(tmp_path)
        store.append(MemoryFile.PAPERS, "## Paper 1\nContent")

        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "## Paper 2\nNew content"
        )

        assert result is True
        content = store.read_file(MemoryFile.PAPERS)
        assert "Paper 1" in content
        assert "Paper 2" in content

    @pytest.mark.asyncio
    async def test_append_with_dedup_whitespace_normalized(self, tmp_path: Path) -> None:
        """空白字符差异不影响精确匹配"""
        store = MemoryStore(tmp_path)
        store.append(MemoryFile.PAPERS, "## Paper 1\nContent here")

        # 内容相同但有前后空白
        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "  ## Paper 1\nContent here  \n"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_append_with_dedup_empty_content(self, tmp_path: Path) -> None:
        """空内容不追加"""
        store = MemoryStore(tmp_path)

        result = await store.append_with_dedup(MemoryFile.PAPERS, "")
        assert result is False

        result = await store.append_with_dedup(MemoryFile.PAPERS, "   \n  ")
        assert result is False

    @pytest.mark.asyncio
    async def test_append_with_dedup_partial_match_not_blocked(self, tmp_path: Path) -> None:
        """部分匹配不应该阻止追加"""
        store = MemoryStore(tmp_path)
        store.append(MemoryFile.PAPERS, "## Paper 1\nThis is a paper about transformers.")

        # 内容包含已有内容的子串，但不是完全相同
        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "This is a paper about transformers.\n\nAdditional content."
        )

        # 因为不是精确匹配，应该追加成功
        assert result is True

    @pytest.mark.asyncio
    async def test_append_with_dedup_semantic_similar(self, tmp_path: Path) -> None:
        """语义相似去重（需要 embedding provider）"""
        from unittest.mock import AsyncMock

        # 创建 mock embedding provider
        mock_provider = AsyncMock()
        # 相似内容的 embedding（余弦相似度 > 0.8）
        similar_embedding = [0.5, 0.5, 0.5, 0.5]

        async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
            # 批量返回所有文本的 embedding
            # 第一个是新内容，其余是已有段落
            return [similar_embedding] * len(texts)

        mock_provider.embed_batch = mock_embed_batch

        store = MemoryStore(tmp_path, embedding_provider=mock_provider)
        store.append(MemoryFile.PAPERS, "这篇论文介绍了 Transformer 架构")

        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "The paper describes the Transformer architecture",
            similarity_threshold=0.8,
        )

        # 语义相似，应跳过
        assert result is False

    @pytest.mark.asyncio
    async def test_append_with_dedup_semantic_different(self, tmp_path: Path) -> None:
        """语义不同应该正常追加"""
        from unittest.mock import AsyncMock

        mock_provider = AsyncMock()

        async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
            # 第一个是新内容，其余是已有段落
            # 返回不相似的 embedding
            return [[1.0, 0.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0, 0.0]] * (len(texts) - 1)

        mock_provider.embed_batch = mock_embed_batch

        store = MemoryStore(tmp_path, embedding_provider=mock_provider)
        store.append(MemoryFile.PAPERS, "这是一篇关于机器学习的论文")

        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "Today's weather is sunny and warm",
            similarity_threshold=0.8,
        )

        # 语义不相似，应追加
        assert result is True

    @pytest.mark.asyncio
    async def test_append_with_dedup_embedding_failure_graceful(self, tmp_path: Path) -> None:
        """embedding 失败时不阻止追加"""
        from unittest.mock import AsyncMock

        mock_provider = AsyncMock()
        mock_provider.embed_batch = AsyncMock(side_effect=Exception("Embedding failed"))

        store = MemoryStore(tmp_path, embedding_provider=mock_provider)
        store.append(MemoryFile.PAPERS, "Existing content")

        result = await store.append_with_dedup(
            MemoryFile.PAPERS, "New content",
            similarity_threshold=0.8,
        )

        # embedding 失败，但精确匹配未触发，应该追加
        assert result is True

    @pytest.mark.asyncio
    async def test_cosine_similarity(self) -> None:
        """测试余弦相似度计算"""
        # 相同向量
        assert MemoryStore._cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

        # 正交向量
        assert MemoryStore._cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

        # 相似向量
        sim = MemoryStore._cosine_similarity([1, 1, 0], [1, 0, 0])
        assert 0 < sim < 1

        # 零向量
        assert MemoryStore._cosine_similarity([0, 0, 0], [1, 1, 1]) == 0.0
