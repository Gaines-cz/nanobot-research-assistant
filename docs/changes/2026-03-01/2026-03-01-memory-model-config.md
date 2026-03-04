# Memory Model 配置支持

**创建日期**: 2026-03-01
**类型**: 功能增强
**影响范围**: 配置层、AgentLoop、CLI 命令

---

## 一、变更概述

为 Memory Consolidation 操作添加独立的模型配置支持，允许通过 `memory_model` 配置项指定与主对话不同的 LLM 模型。

### 主要收益
- **成本优化**: 可为 memory consolidation 配置更便宜的模型
- **灵活性**: 主模型和 memory 模型可独立选择
- **向后兼容**: 未配置时自动使用主模型，行为不变

---

## 二、变更详情

### 2.1 配置层 (`nanobot/config/schema.py`)

在 `AgentDefaults` 类中新增 `memory_model` 字段：

```python
class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    memory_model: str | None = None  # Optional: separate model for memory consolidation
    provider: str = "auto"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
```

### 2.2 AgentLoop 层 (`nanobot/agent/loop.py`)

**初始化变更**:
```python
def __init__(
    self,
    bus: MessageBus,
    provider: LLMProvider,
    workspace: Path,
    model: str | None = None,
    memory_model: str | None = None,  # 新增参数
    # ... 其他参数不变
):
    self.model = model or provider.get_default_model()
    # memory_model 默认与 model 相同
    self.memory_model = memory_model or self.model
```

**Memory Consolidation 变更**:
```python
async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
    # 使用 self.memory_model 替代 self.model
    return await MemoryStore(...).consolidate(
        session, self.provider, self.memory_model,  # ← 使用 memory_model
        archive_all=archive_all, memory_window=self.memory_window,
    )
```

### 2.3 CLI 命令层 (`nanobot/cli/commands.py`)

三个 `AgentLoop` 实例化位置均添加 `memory_model` 参数传递：

1. `gateway()` 命令 (line 311)
2. `agent()` 命令 (line 470)
3. `cron_run()` 命令 (line 963)

---

## 三、配置示例

### 默认配置（未指定 memory_model）

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```
→ Memory consolidation 使用 `claude-opus-4-5`

### 配置独立 memory_model

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "memory_model": "deepseek/deepseek-chat-v3-1226"
    }
  }
}
```
→ 主对话使用 `claude-opus-4-5`，Memory consolidation 使用 `deepseek-chat-v3`

### 推荐配置组合

| 主模型 | memory_model (推荐) | 说明 |
|--------|---------------------|------|
| claude-opus-4-5 | deepseek-chat-v3 | 高质量主模型 + 经济型 memory 模型 |
| claude-sonnet-4 | gpt-4o-mini | 平衡方案 |
| gemini-2.0-pro | qwen2.5-72b | 性价比方案 |
| llama-3.3-70b-instruct:free | llama-3.3-70b-instruct:free | 免费方案 |

---

## 四、变更文件清单

| 文件 | 变更内容 | 行数变化 |
|------|----------|----------|
| `nanobot/config/schema.py` | `AgentDefaults` 新增 `memory_model` 字段 | +1 行 |
| `nanobot/agent/loop.py` | `__init__()` 新增参数和属性 | +3 行 |
| `nanobot/agent/loop.py` | `_consolidate_memory()` 使用 `memory_model` | 修改 2 行 |
| `nanobot/cli/commands.py` | `gateway()` 传递 `memory_model` | +1 行 |
| `nanobot/cli/commands.py` | `agent()` 传递 `memory_model` | +1 行 |
| `nanobot/cli/commands.py` | `cron_run()` 传递 `memory_model` | +1 行 |

**总计**: ~9 行代码变更

---

## 五、验证步骤

### 配置层验证
```python
from nanobot.config.schema import AgentDefaults

# 默认行为
defaults = AgentDefaults()
assert defaults.memory_model is None  # 未配置时为 None

# 配置不同模型
defaults2 = AgentDefaults(memory_model="deepseek/deepseek-chat-v3-1226")
assert defaults2.memory_model == "deepseek/deepseek-chat-v3-1226"
```

### AgentLoop 层验证
```python
# memory_model=None 时，自动使用主模型
agent = AgentLoop(..., model="claude-opus", memory_model=None)
assert agent.memory_model == "claude-opus"

# 配置不同 memory_model
agent2 = AgentLoop(..., model="claude-opus", memory_model="deepseek-chat")
assert agent2.memory_model == "deepseek-chat"
```

### 端到端验证
- ✅ 现有记忆 consolidation 测试全部通过（29 个测试）
- ✅ 配置文件可正确加载 `memory_model` 字段

---

## 六、注意事项

### 模型能力要求

Memory consolidation 需要 LLM 具备：
- **Tool calling 能力**: 必须支持 function call
- **上下文理解**: 理解对话内容并提取关键信息
- **JSON 输出**: 输出符合格式的 JSON

**推荐模型**:
- DeepSeek Chat V3（性价比高）
- GPT-4o-mini（成本低，速度快）
- Qwen2.5-72B（开源模型，性价比高）

### 未来扩展

可能的后续增强：
```python
class AgentDefaults(Base):
    # ... 现有字段 ...
    memory_model: str | None = None
    memory_temperature: float | None = None  # 独立的 temperature
    memory_max_tokens: int | None = None     # 独立的 max_tokens
```

---

## 七、相关文档

- 实施计划：`docs/plans/2026-02-28-memory-optimization-project.md`
- Memory 系统重构：`docs/changes/2026-02-28-memory-system-refactor.md`
- RAG Memory 集成：`docs/changes/2026-02-28-rag-memory-integration.md`
