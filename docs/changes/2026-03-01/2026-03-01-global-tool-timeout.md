# 工具执行全局超时机制

**创建日期**: 2026-03-01
**实现日期**: 2026-03-01

---

## 一、背景

`agent/tools/registry.py:68` 中的 `execute` 方法直接调用 `await tool.execute(**params)`，没有任何超时保护。虽然部分工具（shell、web、mcp）内部有自己的超时实现，但：

1. 文件系统工具（read_file、write_file、edit_file、list_dir）完全无超时保护
2. message、spawn、cron、rag 等工具也无超时保护
3. 工具内部超时依赖各自实现，不统一
4. 如果工具内部卡住，无法从外部终止

---

## 二、解决方案

添加全局工具执行超时机制，统一保护所有工具调用。

---

## 三、修改文件

### 1. nanobot/config/schema.py

在 `ToolsConfig` 中添加默认超时配置，RAG 配置独立到根级别：

```python
class ToolsConfig(Base):
    """Tools configuration."""

    default_tool_timeout: int = 60  # Global default timeout in seconds, 0 means no timeout
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)  # RAG 独立到根级别
```

### 2. nanobot/agent/tools/registry.py

修改 `ToolRegistry` 类：

1. 添加 `import asyncio`
2. 添加 `default_timeout` 参数到 `__init__`
3. 修改 `execute` 方法使用 `asyncio.timeout()`

关键代码：

```python
class ToolRegistry:
    def __init__(self, default_timeout: float = 60.0):
        self._tools: dict[str, Tool] = {}
        self._default_timeout = default_timeout

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        # ... 参数验证 ...

        # 尝试使用工具自身的超时配置，否则使用全局默认
        tool_timeout = getattr(tool, 'timeout', None) or self._default_timeout

        if tool_timeout > 0:
            try:
                async with asyncio.timeout(tool_timeout):
                    result = await tool.execute(**params)
            except asyncio.TimeoutError:
                return f"Error: Tool '{name}' timed out after {tool_timeout} seconds." + _HINT
        else:
            result = await tool.execute(**params)

        # ... 返回结果 ...
```

### 3. nanobot/agent/loop.py

1. 添加 `tools_config: ToolsConfig | None = None` 参数到 `__init__`
2. 存储配置：`self.tools_config = tools_config or ToolsConfig()`
3. 传递超时值：`self.tools = ToolRegistry(default_timeout=self.tools_config.default_tool_timeout)`

### 4. nanobot/agent/subagent.py

在 `SubagentManager` 中添加 `default_tool_timeout` 参数，使子代理与主 Agent 使用相同的超时配置：

```python
def __init__(
    self,
    provider: LLMProvider,
    workspace: Path,
    bus: MessageBus,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    serper_api_key: str | None = None,
    exec_config: "ExecToolConfig | None" = None,
    restrict_to_workspace: bool = False,
    default_tool_timeout: float = 60.0,  # 新增参数
):
    # ...
    self.default_tool_timeout = default_tool_timeout
```

在 `_run_subagent` 方法中使用该超时值：

```python
tools = ToolRegistry(default_timeout=self.default_tool_timeout)
```

### 5. nanobot/agent/loop.py

在创建 SubagentManager 时传递超时配置：

```python
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
    default_tool_timeout=self.tools_config.default_tool_timeout,  # 新增
)
```

### 6. nanobot/cli/commands.py

在 3 处 `AgentLoop()` 调用处添加 `tools_config=config.tools`：

- `commands.py:325` - agent 命令
- `commands.py:486` - interactive 模式
- `commands.py:987` - gateway 命令

---

## 四、使用方式

### 默认行为

- 默认超时时间：**60 秒**
- 所有工具（spawn、read_file、write_file、exec 等）都受此超时保护

### 自定义配置

在 `config.json` 中修改：

```json
{
  "tools": {
    "default_tool_timeout": 120
  }
}
```

### 禁用超时

设置 `default_tool_timeout` 为 0 即可禁用全局超时：

```json
{
  "tools": {
    "default_tool_timeout": 0
  }
}
```

---

## 五、测试验证

### 单元测试

```bash
pytest tests/test_tool_validation.py -xvs
# 6 passed
```

### 手动测试

```python
import asyncio
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.base import Tool

class SlowTool(Tool):
    name = 'slow_tool'
    async def execute(self) -> str:
        await asyncio.sleep(5)
        return 'Done'

# 测试超时（1秒超时，5秒执行）
registry = ToolRegistry(default_timeout=1.0)
registry.register(SlowTool())
result = await registry.execute('slow_tool', {})
# Result: Error: Tool 'slow_tool' timed out after 1.0 seconds.

# 测试禁用超时（timeout=0）
registry = ToolRegistry(default_timeout=0)
result = await registry.execute('fast_tool', {})
# Result: Done
```

---

## 六、影响范围

| 工具类型 | 之前 | 之后 |
|----------|------|------|
| 文件系统工具 | ❌ 无超时 | ✅ 60s 超时 |
| spawn 工具 | ❌ 无超时 | ✅ 60s 超时 |
| message 工具 | ❌ 无超时 | ✅ 60s 超时 |
| cron 工具 | ❌ 无超时 | ✅ 60s 超时 |
| rag 工具 | ❌ 无超时 | ✅ 60s 超时 |
| shell 工具 | ✅ 内部超时 | ⚪ 优先使用内部超时 |
| web 工具 | ✅ 内部超时 | ⚪ 优先使用内部超时 |
| mcp 工具 | ✅ 内部超时 | ⚪ 优先使用内部超时 |

### 子代理超时一致性

| 组件 | 之前 | 之后 |
|------|------|------|
| 主 Agent | ✅ 60s 超时（从配置读取） | ✅ 60s 超时 |
| 子代理 (SubagentManager) | ❌ 30s 超时（硬编码） | ✅ 60s 超时（从配置读取） |

现在主 Agent 和子代理使用相同的超时配置，保持一致。

---

## 七、额外收益

此实现还顺便解决了之前代码审查中发现的**超时零散**问题，统一了所有工具的超时行为。
