import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.heartbeat.service import HeartbeatService


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    async def on_execute(tasks: str) -> str:
        return f"Executed: {tasks}"

    async def on_notify(response: str) -> None:
        pass

    # Create mock provider
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.tool_calls = []
    mock_provider.chat = AsyncMock(return_value=mock_response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=mock_provider,
        model="test-model",
        on_execute=on_execute,
        on_notify=on_notify,
        interval_s=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_heartbeat_disabled(tmp_path) -> None:
    """Test that disabled heartbeat does not start."""
    mock_provider = MagicMock()
    service = HeartbeatService(
        workspace=tmp_path,
        provider=mock_provider,
        model="test-model",
        interval_s=30,
        enabled=False,
    )

    await service.start()

    assert service._task is None
    assert not service._running


@pytest.mark.asyncio
async def test_trigger_now_no_file(tmp_path) -> None:
    """Test trigger_now returns None when HEARTBEAT.md doesn't exist."""
    mock_provider = MagicMock()
    service = HeartbeatService(
        workspace=tmp_path,
        provider=mock_provider,
        model="test-model",
    )

    result = await service.trigger_now()

    assert result is None


@pytest.mark.asyncio
async def test_trigger_now_with_tasks(tmp_path) -> None:
    """Test trigger_now executes when HEARTBEAT.md exists and returns run."""
    # Create HEARTBEAT.md with content
    heartbeat_file = tmp_path / "HEARTBEAT.md"
    heartbeat_file.write_text("Review these tasks:\n- task 1\n- task 2")

    # Create mock provider that returns 'run' action
    mock_provider = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.arguments = {"action": "run", "tasks": "Active: task 1, task 2"}
    mock_response = MagicMock()
    mock_response.tool_calls = [mock_tool_call]
    mock_provider.chat = AsyncMock(return_value=mock_response)

    async def on_execute(tasks: str) -> str:
        return f"Completed: {tasks}"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=mock_provider,
        model="test-model",
        on_execute=on_execute,
    )

    result = await service.trigger_now()

    assert result == "Completed: Active: task 1, task 2"
