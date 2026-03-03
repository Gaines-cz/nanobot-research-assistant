# Config 模块近期改动总结

**创建日期**: 2026-03-01
**类型**: 配置优化
**影响范围**: 全局

---

## 一、概述

本文档总结近期对 nanobot 配置模块的主要改动，包括：
1. 全局工具超时机制
2. 超时配置收拢
3. RAG 配置独立
4. Memory Model 配置支持

---

## 二、详细变更

### 2.1 全局工具超时机制

**问题**: 部分工具（shell、web、mcp）内部有超时实现，但文件系统工具、spawn、message、cron、rag 等工具无超时保护。

**解决方案**: 在 `ToolsConfig` 中添加 `default_tool_timeout` 字段，统一保护所有工具调用。

```python
class ToolsConfig(Base):
    """Tools configuration."""

    default_tool_timeout: int = 60  # 全局默认超时（秒），0 表示禁用超时
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
```

**影响文件**:
- `nanobot/config/schema.py` - 添加 `default_tool_timeout` 字段
- `nanobot/agent/tools/registry.py` - ToolRegistry 添加超时保护
- `nanobot/agent/loop.py` - 传递超时配置
- `nanobot/agent/subagent.py` - 子代理使用相同超时配置
- `nanobot/cli/commands.py` - 3 处 AgentLoop 调用添加配置

### 2.2 超时配置收拢

**问题**: 超时配置分散在多个地方，难以理解优先级：
- `default_tool_timeout` (ToolsConfig)
- `exec.timeout` (ExecToolConfig)
- `mcp.tool_timeout` (MCPServerConfig)

**解决方案**: 移除分散的超时配置，统一使用 `ToolsConfig.default_tool_timeout`。

**配置变更**:
| 之前 | 之后 |
|------|------|
| `ExecToolConfig.timeout` | ❌ 已移除 |
| `MCPServerConfig.tool_timeout` | ❌ 已移除 |
| `ToolsConfig.default_tool_timeout` | ✅ 保留（唯一全局超时） |

**影响文件**:
- `nanobot/config/schema.py` - 移除 `ExecToolConfig.timeout` 和 `MCPServerConfig.tool_timeout`
- `nanobot/agent/loop.py` - ExecTool 使用 `tools_config.default_tool_timeout`
- `nanobot/agent/subagent.py` - ExecTool 使用 `default_tool_timeout`
- `nanobot/agent/tools/mcp.py` - MCP 工具使用 `default_tool_timeout`

### 2.3 RAG 配置独立

**问题**: RAG 是独立功能模块，但嵌套在 `tools.rag` 下，不符合模块化设计。

**解决方案**: 将 RAG 配置从 `ToolsConfig` 移到 `Config` 根级别。

```python
class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)  # 独立到根级别
```

**配置访问变更**:
| 之前 | 之后 |
|------|------|
| `config.tools.rag.enabled` | `config.rag.enabled` |
| `config.tools.rag.embedding_model` | `config.rag.embedding_model` |

**影响文件**:
- `nanobot/config/schema.py` - 将 `rag` 从 ToolsConfig 移到 Config
- `nanobot/cli/commands.py` - 7 处 `config.tools.rag` 改为 `config.rag`

### 2.4 Memory Model 配置支持

**问题**: Memory Consolidation 使用与主对话相同的模型，无法针对内存操作优化成本。

**解决方案**: 在 `AgentDefaults` 中添加 `memory_model` 字段，允许独立配置。

```python
class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    memory_model: str | None = None  # 独立配置 memory consolidation 模型
    subagent_model: str | None = None  # 独立配置 subagent 模型
    provider: str = "auto"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
```

**影响文件**:
- `nanobot/config/schema.py` - 添加 `memory_model` 和 `subagent_model` 字段
- `nanobot/agent/loop.py` - 传递 memory_model 到 consolidation
- `nanobot/cli/commands.py` - 3 处 AgentLoop 调用添加 memory_model

---

## 三、配置示例

### 完整配置示例

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "memory_model": "deepseek/deepseek-chat-v3-1226",
      "subagent_model": "anthropic/claude-sonnet-4-20250514"
    }
  },
  "tools": {
    "default_tool_timeout": 120,
    "exec": {
      "path_append": "/usr/local/bin"
    },
    "mcp_servers": {
      "filesystem": {
        "command": "uvx",
        "args": ["mcp-server-filesystem", "--allowed-directory", "/path/to/docs"]
      }
    }
  },
  "rag": {
    "enabled": true,
    "embedding_model": "BAAI/bge-m3",
    "rerank_model": "BAAI/bge-reranker-v2-m3"
  }
}
```

---

## 四、变更文件清单

| 文件 | 变更内容 |
|------|----------|
| `nanobot/config/schema.py` | 添加 default_tool_timeout、memory_model、subagent_model；移除 exec.timeout、mcp.tool_timeout；rag 独立到根级别 |
| `nanobot/agent/loop.py` | 传递工具超时配置，更新 rag_config 来源 |
| `nanobot/agent/subagent.py` | 使用 default_tool_timeout |
| `nanobot/agent/tools/registry.py` | ToolRegistry 添加全局超时保护 |
| `nanobot/agent/tools/mcp.py` | MCP 工具使用 default_tool_timeout |
| `nanobot/cli/commands.py` | 更新 rag_config 来源，添加 memory_model、tools_config 参数 |

---

## 五、验证

### 配置层验证

```python
from nanobot.config.schema import Config

c = Config()

# 超时配置收拢
print(c.tools.default_tool_timeout)  # 60
print(hasattr(c.tools.exec, 'timeout'))  # False

# RAG 配置独立
print(c.rag.enabled)  # True
print(hasattr(c.tools, 'rag'))  # False

# Memory Model
print(c.agents.defaults.memory_model)  # None
print(c.agents.defaults.subagent_model)  # None
```

### 测试验证

```bash
pytest tests/ -x --ignore=tests/test_matrix_channel.py -q
# 140 passed
```

---

## 六、相关文档

- 全局工具超时机制: `docs/changes/2026-03-01-global-tool-timeout.md`
- Memory Model 配置: `docs/changes/2026-03-01-memory-model-config.md`
