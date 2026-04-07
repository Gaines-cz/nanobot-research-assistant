# nanobot-research-assistant

**轻量级个人 AI 助手**，基于 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 框架深度定制，专为个人研究场景优化。

> 灵感来自 [OpenClaw](https://github.com/openclaw/openclaw)，核心代码仅 ~4000 行，比 Clawdbot 430k+ 行轻量化 99%。

## 核心特性

### 基础特性（nanobot 原生）

- **多频道支持**: Telegram、飞书、Discord、Slack、WhatsApp、Email、QQ、钉钉、Matrix、Mochat 等
- **多 Provider 支持**: OpenRouter、DeepSeek、Anthropic、OpenAI、Moonshot、硅基流动等 20+ 模型
- **内置工具**: 文件读写、Shell 执行、网页搜索、消息发送、子 Agent、Cron 定时任务
- **MCP 支持**: 可接入 Model Context Protocol 服务器扩展工具能力
- **记忆系统**: 会话历史自动存档，支持长期记忆
- **会话管理**: JSONL 格式持久化，支持多会话并行

### 新增功能（本项目定制）

#### 1. 强化记忆系统

混合搜索 + 智能固化的下一代记忆系统：

- **混合检索**: BM25 全文搜索 + 向量相似度搜索 + RRF 融合 + CrossEncoder 重排
- **综合打分**: `model_score × 0.7 + freq_score × 0.2 + recency_score × 0.1`
- **2 步固化流程**:
  1. LLM 提取结构化记忆（HISTORY / KNOWLEDGE / DECISIONS / PROJECTS）
  2. 按类型处理：直接插入或搜索整合
- **智能触发**: 支持暂停检测（长时间对话不打断）、重要内容优先固化

#### 2. RAG 知识检索

完整的检索增强生成模块：

- **多格式支持**: PDF、Markdown、Word、TXT
- **双粒度分块**: Small chunks（向量检索）+ Large chunks（BM25 + 上下文扩展）
- **混合检索**: BM25 + Vector + 加权融合
- **智能重排**: CrossEncoder 模型重排序
- **文档级搜索**: 支持按文档聚合结果
- **查询扩展**: 自动优化搜索query
- **搜索缓存**: 5 分钟 TTL 缓存加速重复查询

#### 3. 可插拔监控模块

零埋点设计，通过 ContextVar 实现协程间上下文传播：

- **事件类型**: SESSION_TURN / LLM_CALL / TOOL_CALL / MEMORY_CONSOLIDATE
- **统计指标**: LLM 调用次数、平均延迟、Token 消耗
- **Trace 追踪**: 每个会话独立 trace_id，支持嵌套

#### 4. 飞书流式响应

打字机效果的流式卡片消息：

- **WebSocket 长连接**: 无需公网 IP / Webhook
- **流式卡片**: AI 助手"正在输入"体验
- **智能降级**: 有工具调用时自动切换普通消息

## 架构

```
Channels (Telegram/飞书/Discord...) → MessageBus → AgentLoop → LLM → Tools
                                         ↓
                                   SessionManager
                                         ↓
                                   MemoryStore (混合搜索)
                                         ↓
                                   RAG (文档检索)
```

**核心组件**:

| 组件 | 路径 | 职责 |
|------|------|------|
| AgentLoop | `nanobot/agent/loop.py` | 消息处理核心引擎 |
| MemoryStore | `nanobot/agent/memory.py` | 混合搜索记忆系统 |
| ContextBuilder | `nanobot/agent/context.py` | 提示词构建 |
| RAG | `nanobot/rag/` | 文档检索模块 |
| Monitor | `nanobot/agent/monitor.py` | 监控追踪 |
| FeishuChannel | `nanobot/channels/feishu.py` | 飞书频道 |

## 快速使用

### 安装

```bash
pip install -e .
```

### 配置

编辑 `~/.nanobot/config.json`:

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "your_app_id",
      "appSecret": "your_app_secret",
      "streaming": true
    }
  }
}
```

### 运行

```bash
# 启动网关（运行所有已启用的频道）
nanobot gateway

# 单条消息测试
nanobot agent -m "你好"

# 交互模式
nanobot agent
```

### 索引知识库

```bash
# 初始化工作空间
nanobot onboard

# 索引文档
nanobot rag index
```

## 目录结构

```
nanobot/
├── agent/              # Agent 核心
│   ├── loop.py         # 主循环
│   ├── memory.py       # 记忆系统
│   ├── context.py      # 上下文构建
│   └── monitor.py      # 监控模块
├── channels/           # 频道实现
│   └── feishu.py       # 飞书（含流式）
├── rag/                # RAG 模块
│   ├── indexing/       # 文档解析分块
│   ├── retrieval/      # 检索层
│   └── store.py        # 存储
├── session/            # 会话管理
├── config/              # 配置
└── providers/          # LLM Provider
```

## 致谢

本项目 fork 自 **[HKUDS/nanobot](https://github.com/HKUDS/nanobot) v0.1.4.post2**，在此基础上进行了大量定制开发：

- 记忆系统重写（混合搜索 + 2步固化）
- 新增完整 RAG 模块
- 新增监控模块
- 飞书流式响应

感谢 nanobot 团队提供了清晰易懂的超轻量级 Agent 框架！