"""nanobot 轻量级监控模块

可插拔设计：
- 不需要在 loop.py 中埋点
- 只需要在 _process_message 入口设置 session_key
- TraceContext 自动从 ContextVar 获取 session_key
"""

import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# === ContextVars：协程间传播上下文 ===
_current_session_key: ContextVar[str] = ContextVar("session_key", default="")
_current_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


# === 数据模型 ===

class EventType(Enum):
    SESSION_TURN = "session_turn"       # 用户消息 -> Agent 处理
    LLM_CALL = "llm_call"              # LLM 调用
    TOOL_CALL = "tool_call"             # 工具调用
    MEMORY_CONSOLIDATE = "memory_consolidate"  # 内存固化


@dataclass
class TraceEvent:
    trace_id: str
    event_type: EventType
    session_key: str
    timestamp: float
    duration_ms: Optional[float]
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "event_type": self.event_type.value,
            "session_key": self.session_key,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


# === MonitorCollector ===

class MonitorCollector:
    """监控数据收集器（单例）"""
    _instance: Optional["MonitorCollector"] = None
    _lock = threading.Lock()

    def __init__(self, max_events: int = 5000):
        self._events: deque[TraceEvent] = deque(maxlen=max_events)  # 使用 deque 自动管理

    @classmethod
    def instance(cls) -> "MonitorCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_event(self, event: TraceEvent):
        """添加监控事件"""
        with self._lock:
            self._events.append(event)

    def get_events(
        self,
        session_key: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> list[TraceEvent]:
        """获取事件列表"""
        with self._lock:
            events = list(self._events)[-limit:]
        if session_key:
            events = [e for e in events if e.session_key == session_key]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events

    def get_stats(self, session_key: Optional[str] = None) -> dict:
        """获取统计信息"""
        with self._lock:
            events = list(self._events)

        if session_key:
            events = [e for e in events if e.session_key == session_key]

        stats = {
            "total_events": len(events),
            "llm_calls": 0,
            "tool_calls": 0,
            "avg_llm_latency_ms": 0.0,
            "total_tokens": 0,
        }
        llm_latencies = []
        for e in events:
            if e.event_type == EventType.LLM_CALL:
                stats["llm_calls"] += 1
                if e.duration_ms:
                    llm_latencies.append(e.duration_ms)
                stats["total_tokens"] += e.metadata.get("total_tokens", 0)
            elif e.event_type == EventType.TOOL_CALL:
                stats["tool_calls"] += 1
        if llm_latencies:
            stats["avg_llm_latency_ms"] = sum(llm_latencies) / len(llm_latencies)
        return stats


# === TraceContext ===

class TraceContext:
    """上下文管理器，自动从 ContextVar 获取 session_key 和 trace_id

    支持嵌套：保存旧值，退出时恢复。
    """

    def __init__(self, event_type: EventType, metadata: Optional[dict] = None):
        self.event_type = event_type
        self.metadata = metadata or {}
        self._start_time: Optional[float] = None
        self._trace_id: str = ""
        self._session_key: str = ""
        self._old_trace_id: str = ""

    def __enter__(self) -> "TraceContext":
        self._start_time = time.perf_counter()
        self._session_key = _current_session_key.get()
        # 保存旧值（支持嵌套）
        self._old_trace_id = _current_trace_id.get()
        # 只有为空时才创建新的
        if not self._old_trace_id:
            self._old_trace_id = uuid.uuid4().hex
            _current_trace_id.set(self._old_trace_id)
        self._trace_id = self._old_trace_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start_time) * 1000 if self._start_time else None

        event = TraceEvent(
            trace_id=self._trace_id,
            event_type=self.event_type,
            session_key=self._session_key,
            timestamp=time.time(),
            duration_ms=duration_ms,
            success=exc_type is None,
            error=str(exc_val) if exc_val else None,
            metadata=self.metadata,
        )
        MonitorCollector.instance().add_event(event)

        # 退出时恢复旧的 trace_id（支持嵌套）
        _current_trace_id.set(self._old_trace_id)
        return False
