# Token 监控与提醒功能设计

**Date:** 2026-02-28

## 需求确认

| 功能 | 选择 |
|------|------|
| 范围 | 完整版（统计 + 每日/每周 + CLI显示 + 智能提醒） |
| 存储 | JSON 文件 (`workspace/usage.json`) |
| 可视化 | CLI 文字统计 |
| 入口 | 独立命令 `nanobot usage` |

---

## 功能 1: Token 使用统计

### 数据结构

```json
{
  "records": [
    {
      "timestamp": "2026-02-28T10:30:00",
      "session_id": "abc123",
      "model": "gpt-4o",
      "prompt_tokens": 1500,
      "completion_tokens": 500,
      "total_tokens": 2000,
      "cost_usd": 0.03
    }
  ],
  "stats": {
    "total_tokens": 1234567,
    "total_cost_usd": 12.34,
    "last_updated": "2026-02-28T10:30:00"
  }
}
```

---

## 功能 2: 智能提醒

### 提醒策略

| 触发条件 | 提醒方式 | 说明 |
|----------|----------|------|
| 每消耗 10,000 token | 日志打印 | 不打扰用户，仅记录 |
| 会话结束 | 消息回复用户 | 本次会话消耗汇总 |
| 达到预算 80% | 消息回复用户 | 需配置预算 |
| 达到预算 100% | 消息回复用户 + 警告 | 可选：阻止继续 |

### 提醒内容示例

**会话结束时的提醒：**
```
💬 本次对话消耗: 2,500 tokens (约 $0.02)
📊 今日累计: 45,000 tokens
```

**预算警告：**
```
⚠️ 警告: 已消耗预算的 80% (8,000 / 10,000 tokens)
```

### 配置项

```json
{
  "tools": {
    "usage": {
      "enabled": true,
      "log_interval": 10000,
      "budget_tokens": 10000,
      "warn_at_percent": 80,
      "notify_on_session_end": true
    }
  }
}
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | true | 是否启用 Token 监控功能 |
| `log_interval` | 10000 | 每消耗 N token 打印一条日志（不打扰用户，仅用于排查） |
| `budget_tokens` | null | 预算上限（单位：token），设为 null 则不限制 |
| `warn_at_percent` | 80 | 达到预算百分比时发送警告消息 |
| `notify_on_session_end` | true | 会话结束时是否回复用户本次消耗 |

### 配置项详解

**1. `enabled`**
```json
"enabled": true
```
- 是否启用 Token 监控
- 设为 `false` 完全关闭，不记录、不提醒

**2. `log_interval`**
```json
"log_interval": 10000
```
- 每消耗 N 个 token 打印一条日志
- 用于在终端/日志中了解消耗速度
- 推荐值：10000（每消耗 1 万 token 打印一次）

**3. `budget_tokens`**
```json
"budget_tokens": 10000
// 或
"budget_tokens": null
```
- 每月/每周的 token 预算上限
- 设为 `null` 则不限制（不触发预算提醒）
- 常见值：10000（根据需求调整）

**4. `warn_at_percent`**
```json
"warn_at_percent": 80
```
- 当消耗达到预算的 X% 时发送警告
- 80 = 消耗 80% 时提醒一次
- 100 = 消耗 100% 时再次提醒

**5. `notify_on_session_end`**
```json
"notify_on_session_end": true
```
- 会话结束时自动回复用户本次消耗
- 示例：`"本次对话消耗 2,500 tokens (约 $0.02)"`

---

## 架构设计

### 1. 核心模块 (nanobot/usage/)

```
nanobot/usage/
├── __init__.py
├── config.py      # 配置模型 (UsageConfig)
├── recorder.py   # 记录 token 使用 + 提醒逻辑
├── storage.py    # JSON 文件读写
├── stats.py      # 统计计算
└── cli.py        # CLI 命令
```

### 2. 模块职责

| 模块 | 职责 |
|------|------|
| `config.py` | UsageConfig 配置类 |
| `recorder.py` | 记录 usage + 触发提醒 + 预算检查 |
| `storage.py` | 读写 workspace/usage.json（含 warned_sessions） |
| `stats.py` | 计算累计/今日/本周统计 |
| `cli.py` | `nanobot usage` 命令 |

### 3. 集成点

在 `AgentLoop` 中，每次 LLM 调用完成后记录：

```python
# nanobot/agent/loop.py
async def _make_completion(...):
    response = await self.provider.acompletion(...)

    # 记录 token 使用
    if response.usage:
        await record_usage(
            session_id=session.id,
            model=self.model,
            usage=response.usage,
            workspace=self.workspace,
            message_queue=self.bus.publish_outbound,  # 用于发送提醒
        )

    return response
```

### 4. 提醒触发逻辑

```python
async def record_usage(...):
    # 1. 写入记录
    await write_record(workspace, record)

    # 2. 检查是否需要提醒
    cumulative = get_cumulative_usage(workspace)

    # 3. 日志提醒 (每 N token)
    if cumulative.total % config.log_interval < last_record.total % config.log_interval:
        logger.info(f"Token usage: {cumulative.total} tokens")

    # 4. 预算提醒（需要持久化 warned 状态）
    if config.budget_tokens:
        percent = cumulative.total / config.budget_tokens * 100
        # 读取已警告状态（从 usage.json 的 warned_sessions 字段）
        warned_sessions = load_warned_sessions(workspace)
        if percent >= config.warn_at_percent and session_id not in warned_sessions:
            await send_message(f"⚠️ 已消耗预算的 {percent:.0f}%")
            warned_sessions.append(session_id)
            save_warned_sessions(workspace, warned_sessions)
```

**注意**：`warned_sessions` 需要持久化到 `usage.json`，否则重启后可能重复提醒：

```json
{
  "records": [...],
  "stats": {...},
  "warned_sessions": ["session_1", "session_2"]
}
```

## CLI 命令设计

### `nanobot rag refresh`

```
$ nanobot rag refresh
✓ RAG refresh complete!
  Added: 3
  Updated: 1
  Deleted: 0
```

### `nanobot rag status`

```
$ nanobot rag status

RAG: enabled
Embedding model: all-MiniLM-L6-v2
Chunk size: 1000 (overlap: 200)

Docs dir: /home/user/.nanobot/workspace/docs
  Files in docs: 5

Index Statistics:
  Documents: 5
  Chunks: 42
```

### `nanobot rag search`

```
$ nanobot rag search "transformer attention"

Results for "transformer attention":
[1] attention-is-all-you-need.pdf (score: 0.85, hybrid)
"The Transformer follows the overall architecture..."

[2] bert-paper.pdf (score: 0.72, vector)
"We propose a new language representation model..."
```

### `nanobot rag rebuild`

```
$ nanobot rag rebuild
This will delete existing index: /home/user/.nanobot/workspace/rag/docs.db
Continue? [y/N]: y
✓ Deleted existing index
✓ RAG rebuild complete!
  Added: 5
  Updated: 0
  Deleted: 0
```

## 实现步骤

### Step 1: 创建模块

1. `nanobot/usage/__init__.py`
2. `nanobot/usage/storage.py` - JSON 读写
3. `nanobot/usage/recorder.py` - 记录函数
4. `nanobot/usage/stats.py` - 统计计算
5. `nanobot/usage/cli.py` - CLI 命令

### Step 2: 集成到 AgentLoop

在 `loop.py` 中添加记录调用

### Step 3: 添加 CLI 命令

在 `commands.py` 中添加 `nanobot usage` 命令

## 文件修改清单

| 文件 | 操作 |
|------|------|
| `nanobot/usage/__init__.py` | 新增 |
| `nanobot/usage/storage.py` | 新增 |
| `nanobot/usage/recorder.py` | 新增 |
| `nanobot/usage/stats.py` | 新增 |
| `nanobot/usage/cli.py` | 新增 |
| `nanobot/usage/config.py` | 新增 - 配置模型 |
| `nanobot/agent/loop.py` | 修改 - 添加记录调用和会话结束提醒 |
| `nanobot/cli/commands.py` | 修改 - 添加 usage 命令 |
| `nanobot/config/schema.py` | 修改 - 添加 UsageConfig |

## 验证

```bash
# 测试命令
nanobot usage

# 检查文件生成
cat ~/.nanobot/workspace/usage.json
```
