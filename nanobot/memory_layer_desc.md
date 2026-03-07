# Memory 模块完整设计文档

**日期**: 2026-03-05
**模块**: `nanobot/agent/memory.py`, `nanobot/agent/context.py`, `nanobot/agent/loop.py`, `nanobot/session/manager.py`, `nanobot/cli/commands.py`
**最近更新**: 新增记忆固化重试机制、扩展偏好关键词、降低触发阈值

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentLoop                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐ │
│  │  Session     │    │ContextBuilder│    │   MemoryStore    │ │
│  │  (消息历史)   │───▶│ (构建prompt) │───▶│  (记忆文件管理)   │ │
│  └──────────────┘    └──────────────┘    └──────────────────┘ │
│         │                                    │                   │
│         │                                    ▼                   │
│         │                          ┌──────────────────┐        │
│         │                          │  6个 MemoryFile  │        │
│         │                          │  (PROFILE等)     │        │
│         │                          └──────────────────┘        │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────────┐                                          │
│  │ConsolidationTrigger│ (何时固化记忆)                          │
│  │  • 窗口阈值        │                                          │
│  │  • 会话暂停        │                                          │
│  │  • 重要内容(关键词) │                                          │
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. MemoryStore 类 (`nanobot/agent/memory.py`)

### 2.1 6 个记忆文件 (MemoryFile Enum)

| 文件名 | 用途 | 更新频率 |
|--------|------|----------|
| `PROFILE.md` | 用户画像（研究方向、偏好、技术背景） | 稳定，很少变化 |
| `PROJECTS.md` | 项目知识（技术栈、架构、进度） | 半稳定 |
| `PAPERS.md` | 论文笔记（读过的论文、关键发现） | 增量，追加新条目 |
| `DECISIONS.md` | 决策记录（为什么选 A 不选 B） | 增量 |
| `TODOS.md` | Todo 列表 | 频繁更新，用 replace |
| `HISTORY.md` | 事件日志（append-only） | 每次固化都追加 |

**重要提醒**: 所有记忆文件只用于存储 USER-related content，**禁止存储 AI 助手配置**（模型名、版本号等）。

### 2.2 支持的操作 (MemoryOperation)

| 操作 | 说明 |
|------|------|
| `append` | 追加内容到文件末尾 |
| `prepend` | 前置内容到文件开头 |
| `update_section` | 更新或创建一个 section（需要 section name） |
| `replace` | 替换整个文件内容 |
| `delete_section` | 删除一个 section |
| `skip` | 跳过（什么都不做） |

### 2.3 关键方法

**读取**:
```python
read_file(file: MemoryFile) -> str                    # 读取整个文件
read_section(file: MemoryFile, section: str) -> str  # 读取特定 section
```

**写入**:
```python
append(file, content)                                 # 追加
prepend(file, content)                                # 前置
update_section(file, section, content)               # 更新 section
delete_section(file, section) -> bool                # 删除 section
replace(file, content)                                # 替换整个文件
```

**高级功能**:
```python
append_with_dedup(file, content, similarity_threshold=0.8)  # 带去重的追加
# - 先精确匹配
# - 再语义相似度匹配（需要 embedding_provider）
```

**原子操作**:
```python
apply_operations_atomic(operations: list[dict]) -> bool
# 所有操作要么全成功，要么全回滚
```

**安全写入**:
```python
_safe_write(path, content)
# 临时文件 + rename，保证原子性
# 失败时自动清理临时文件
```

---

## 3. 记忆如何注入 Context (`nanobot/agent/context.py`)

### 3.1 `ContextBuilder.build_system_prompt()` 流程

```
build_system_prompt(skill_names, query)
│
├─ 1. Identity (固定)
│   └─ Runtime, Workspace, Guidelines
│
├─ 2. Bootstrap files (AGENTS.md, SOUL.md, USER.md 等)
│
├─ 3. Memory (关键部分) ← MemoryStore.get_memory_context(query)
│   ├─ 总是加载: PROFILE.md
│   ├─ 默认加载: TODOS.md
│   └─ 根据 query 关键词加载:
│       ├─ 论文相关关键词 → PAPERS.md
│       ├─ 项目相关关键词 → PROJECTS.md
│       └─ 决策相关关键词 → DECISIONS.md
│
├─ 4. Active Skills (always 技能)
│
└─ 5. Skills Summary
```

### 3.2 `get_memory_context(query)` 的选择性加载策略

```python
# 总是加载
PROFILE.md → "## Profile"

# 默认加载
TODOS.md → "## Current Tasks"

# 根据 query 关键词条件加载
query 含 "论文", "paper", "arxiv" → PAPERS.md → "## Paper Notes"
query 含 "项目", "project", "架构" → PROJECTS.md → "## Projects"
query 含 "为什么", "决策", "why" → DECISIONS.md → "## Decisions"
```

### 3.3 完整的消息构建 (`build_messages()`)

```python
[
  {"role": "system", "content": build_system_prompt(query=current_message)},
  *history,  # 来自 Session.get_history()
  {"role": "user", "content": runtime_context},  # 时间、channel、chat_id
  {"role": "user", "content": user_content}      # 当前消息 + 图片
]
```

---

## 4. Session 与消息历史 (`nanobot/session/manager.py`)

### 4.1 Session 数据结构

```python
@dataclass
class Session:
    key: str                                    # "channel:chat_id"
    messages: list[dict] = field(default_factory=list)  # 所有消息（append-only）
    last_consolidated: int = 0                 # 已固化到 memory 文件的消息数

    # 触发优化字段
    last_consolidation_check: float            # 上次检查时间
    consolidation_paused: bool                  # 是否暂停检测
```

### 4.2 `get_history(max_messages)` - 获取对话历史

```python
def get_history(max_messages: int = 500) -> list[dict]:
    # 1. 只取未固化的消息
    unconsolidated = self.messages[self.last_consolidated:]

    # 2. 切片最近 N 条
    sliced = unconsolidated[-max_messages:]

    # 3. 删除开头的非 user 消息（避免孤立的 tool_result）
    for i, m in enumerate(sliced):
        if m.get("role") == "user":
            sliced = sliced[i:]
            break

    # 4. 返回（只保留必要字段: role, content, tool_calls, tool_call_id, name）
    return out
```

**关键点**: Session 的 messages 列表**永远不会被修改或删除**，只有 `last_consolidated` 指针向前移动。

---

## 5. 记忆固化机制详解

### 5.1 固化触发机制 (`ConsolidationTrigger`)

**文件**: `nanobot/agent/loop.py:38-158`

#### 三个触发条件（满足任一即可）

| 条件 | 说明 | 阈值 |
|------|------|------|
| **1. 消息窗口（保底）** | 未固化消息达到一定数量 | 默认 100 条 |
| **2. 会话暂停** | 会话空闲一段时间 + 已有一定消息 | 5 分钟 + 10 条 |
| **3. 重要内容检测** | 最近消息含关键词 + 已有一定消息 | **2 条（2026-03-05 优化）** |

#### 重要内容关键词（2026-03-05 扩展）

```python
DEFAULT_IMPORTANT_KEYWORDS = [
    # 原有
    "决定", "decision", "结论", "conclusion",
    "todo", "任务", "task", "计划", "plan",
    "项目", "project", "架构", "architecture",
    "记住", "remember", "note", "笔记",
    # 新增：偏好表达关键词（2026-03-05）
    "我喜欢", "我不喜欢", "我想要", "我不想要",
    "我希望", "我不希望", "prefer", "I like", "I don't like",
    "changed my mind", "my preference",
]
```

**优化说明**:
- 新增 11 个偏好表达关键词，确保用户表达偏好变更时能被及时检测到
- 触发阈值从 5 条降至 2 条，用户表达偏好时能更快触发固化

#### 触发检查流程

```python
def should_trigger(session, now):
    unconsolidated = len(session.messages) - session.last_consolidated

    # 条件 1：消息窗口（保底）
    if unconsolidated >= window_threshold:
        return True, "message window reached"

    # 没有未固化消息，直接返回
    if unconsolidated == 0:
        return False, "no unconsolidated messages"

    # 条件 2：会话暂停
    if pause_reason := _check_pause(session, unconsolidated, now):
        return True, pause_reason

    # 条件 3：重要内容检测
    if important_reason := _check_important_content(session, unconsolidated):
        return True, important_reason

    return False, "no trigger condition met"
```

---

### 5.2 固化执行入口 (`AgentLoop._process_message()`)

```python
# 在处理消息前检查是否需要固化
if session.key not in self._consolidating:
    should_trigger, reason = self._consolidation_trigger.should_trigger(session)
    if should_trigger:
        # 后台任务执行固化（不阻塞当前消息处理）
        _task = asyncio.create_task(_consolidate_and_unlock())
```

**设计要点**:
- 固化在后台任务执行，不阻塞当前消息响应
- 使用 `_consolidating` set + 锁防止并发固化同一个 session
- 固化失败不影响主流程（降级策略）

---

### 5.3 固化核心策略 (`MemoryStore.consolidate()`)

**文件**: `nanobot/agent/memory.py:772-814`

支持两种策略：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **RAG-based** (优先) | 先搜索相关记忆 → LLM 决策 | 有 RAG Store，需要避免重复 |
| **Direct** (fallback) | 直接让 LLM 决策 | 无 RAG Store，追求速度 |

```python
async def consolidate(session, provider, model, *, use_rag=True):
    if use_rag and self._rag_store is not None:
        return await self._consolidate_with_rag(...)
    else:
        return await self._consolidate_direct(...)
```

---

### 5.4 Direct 固化策略 (`_consolidate_direct`)

**文件**: `nanobot/agent/memory.py:818-908`

```
步骤 1: 获取要固化的消息
         └─ old_messages = session.messages[last_consolidated : -keep_count]
         └─ keep_count = memory_window // 2 (保留最近一半)
         ↓
步骤 2: 构建 prompt（已加强制调用工具说明）
         └─ "IMPORTANT: You MUST call the save_memory tool..."
         ↓
步骤 3: 调用 LLM + save_memory 工具
         ↓
步骤 4: 检查是否调用了工具 ──否──→ 重试 2 次（2026-03-05 新增）
         ↓ 是
步骤 5: 应用操作，更新记忆文件
```

**加强后的 Prompt 开头**:
```
IMPORTANT: You MUST call the save_memory tool with your consolidation.
Do NOT just describe the changes in your response - actually call the tool.
If there is nothing worth saving, call the tool with empty operations.
```

---

### 5.5 RAG-based 固化策略 (`_consolidate_with_rag`)

**文件**: `nanobot/agent/memory.py:946-1059`

```
步骤 1: 获取要固化的消息
         ↓
步骤 2: RAG 搜索相关记忆（避免重复）
         ↓
步骤 3: 构建 prompt（含相关记忆 + 强制调用工具）
         ↓
步骤 4: 调用 LLM + save_memory 工具
         ↓
步骤 5: 检查是否调用了工具 ──否──→ 重试 2 次（2026-03-05 新增）
         ↓ 是
步骤 6: 应用操作，更新记忆文件 + 更新 RAG 索引
```

---

### 5.6 新增重试机制详解（2026-03-05）

#### 重试触发时机

当 `_process_save_memory_tool_call()` 返回 `None` 时（即 LLM 没有调用 `save_memory` 工具），触发重试。

#### 重试流程

```
初始调用失败
    ↓
第 1 次重试 (retry_count=1)
    ├─ Prompt: "IMPORTANT: You MUST call..."
    └─ 成功？→ 退出，继续执行
    ↓ 否
第 2 次重试 (retry_count=2)
    ├─ Prompt: "CRITICAL: This is your second retry..."
    └─ 成功？→ 退出，继续执行
    ↓ 否
全部失败 → 返回 False，记录错误日志
```

#### `_add_retry_instruction()` 方法

**文件**: `nanobot/agent/memory.py:1061-1074`

根据重试次数返回不同强度的提示：

| 重试次数 | 提示级别 | 内容 |
|---------|---------|------|
| 1 | IMPORTANT | 强调必须调用工具，不要只描述变更 |
| 2 | CRITICAL | 强调这是第二次重试，即使没有内容也要调用空操作 |

```python
def _add_retry_instruction(self, original_prompt: str, retry_count: int) -> str:
    if retry_count == 1:
        retry_instruction = """
IMPORTANT: You MUST call the save_memory tool with your consolidation.
Do not just describe the changes in your response - actually call the tool.
"""
    else:  # retry_count == 2
        retry_instruction = """
CRITICAL: This is your second retry. You MUST call the save_memory tool.
If you cannot identify any memory-worthy content, call the tool with empty operations.
DO NOT respond without calling the tool.
"""
    return retry_instruction + "\n\n" + original_prompt
```

#### 重试日志示例

```python
WARNING  | Memory consolidation: LLM did not call save_memory, skipping...
WARNING  | Memory consolidation: no save_memory tool call, retrying (attempt 1/2)...
WARNING  | Memory consolidation: no save_memory tool call, retrying (attempt 2/2)...
ERROR    | Memory consolidation failed after 3 attempts (1 initial + 2 retries)
```

---

### 5.7 `save_memory` 工具定义

**文件**: `nanobot/agent/memory.py:45-91`

```python
_SAVE_MEMORY_TOOL = [{
    "name": "save_memory",
    "parameters": {
        "history_entry": "总结段落（2-5 句话，以 [YYYY-MM-DD HH:MM] 开头）",
        "operations": [
            {
                "file": "profile|projects|papers|decisions|todos",
                "action": "append|prepend|update_section|replace|delete_section|skip",
                "section": "section name (可选)",
                "content": "内容 (可选)"
            }
        ]
    }
}]
```

---

### 5.8 操作应用流程 (`_apply_save_memory_operations`)

**文件**: `nanobot/agent/memory.py:689-770`

```
1. 追加历史记录 (HISTORY.md)
         ↓
2. 原子应用所有操作（全部成功或全部回滚）
         ├─ 保存每个文件的旧内容用于回滚
         ├─ 逐个应用操作
         └─ 任一失败则回滚所有文件
         ↓
3. 更新 RAG 索引（仅 RAG 模式，且只更新修改的文件）
         ↓
4. 更新 session.last_consolidated
```

---

## 6. Loop 如何读/存记忆 (`nanobot/agent/loop.py`)

### 6.1 初始化阶段 (`AgentLoop.__init__`)

```python
# 1. 创建 MemoryStore（共享实例）
self._memory_store = MemoryStore(workspace, self._embedding_provider)

# 2. 创建 ContextBuilder，传入共享的 MemoryStore
self.context = ContextBuilder(workspace, self._embedding_provider,
                              memory_store=self._memory_store)

# 3. 创建 ConsolidationTrigger（固化触发检查器）
self._consolidation_trigger = ConsolidationTrigger(
    window_threshold=self.memory_window,              # 默认 100 条
    pause_threshold_seconds=300,                       # 5 分钟暂停
    pause_min_messages=10,
    important_check_window=5,
    important_min_messages=2,  # 2026-03-05: 5 → 2
)
```

### 6.2 读取记忆（每次消息处理）

```python
async def _process_message(msg):
    # 1. 获取或创建 Session
    session = self.sessions.get_or_create(key)

    # 2. 从 Session 获取历史消息（未固化的部分）
    history = session.get_history(max_messages=self.memory_window)

    # 3. 构建消息（内部调用 context.build_system_prompt）
    initial_messages = self.context.build_messages(
        history=history,
        current_message=msg.content,
        query=msg.content,  # ← 用于 memory 选择性加载
        ...
    )

    # 4. 运行 agent loop
    final_content, _, all_msgs = await self._run_agent_loop(initial_messages)

    # 5. 保存新消息到 Session
    self._save_turn(session, all_msgs, 1 + len(history))
    self.sessions.save(session)
```

---

## 7. 完整数据流图

```
┌──────────┐
│  用户消息 │
└────┬─────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. SessionManager.get_or_create()                           │
│    └─ 从磁盘加载或创建新 Session                              │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 检查是否需要固化记忆?                                      │
│    └─ ConsolidationTrigger.should_trigger()                  │
│       ├─ 是 → 后台任务: _consolidate_memory()               │
│       │         └─ MemoryStore.consolidate()                 │
│       │             ├─ RAG 搜索相关记忆 (可选)               │
│       │             ├─ LLM 调用 save_memory tool             │
│       │             ├─ 重试机制 (最多 2 次，2026-03-05)     │
│       │             ├─ 原子操作更新 6 个记忆文件             │
│       │             └─ 更新 session.last_consolidated        │
│       └─ 否 → 继续                                            │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ContextBuilder.build_messages()                           │
│    ├─ build_system_prompt(query=当前消息)                    │
│    │  ├─ Identity                                            │
│    │  ├─ Bootstrap files                                     │
│    │  ├─ Memory (来自 MemoryStore)                           │
│    │  │  ├─ PROFILE.md (总是)                               │
│    │  │  ├─ TODOS.md (默认)                                 │
│    │  │  ├─ PAPERS.md (如 query 含论文关键词)                │
│    │  │  ├─ PROJECTS.md (如 query 含项目关键词)              │
│    │  │  └─ DECISIONS.md (如 query 含决策关键词)             │
│    │  ├─ Active Skills                                       │
│    │  └─ Skills Summary                                      │
│    └─ history = session.get_history()                        │
│       └─ 未固化的消息 (last_consolidated 之后)               │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LLM 调用 + Tool 执行                                       │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. _save_turn()                                              │
│    └─ 新消息追加到 session.messages                          │
│       (注意: 不修改 last_consolidated)                        │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SessionManager.save(session)                              │
│    └─ 写入磁盘 (JSONL 格式)                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 关键设计特点

| 特点 | 说明 |
|------|------|
| **Append-only** | Session.messages 永不删除，只移动 `last_consolidated` 指针 |
| **原子操作** | 记忆文件更新用临时文件 + rename，保证不损坏 |
| **选择性加载** | Memory 根据 query 关键词决定加载哪些文件 |
| **后台固化** | 记忆固化在后台任务执行，不阻塞当前消息 |
| **两种固化策略** | RAG-based（更准确）或 Direct（更快） |
| **去重机制** | 精确匹配 + 语义相似度匹配（可选） |
| **重试机制** | LLM 未调用工具时自动重试 2 次（2026-03-05 新增） |
| **偏好检测** | 扩展的关键词列表，及时捕获用户偏好变更（2026-03-05 新增） |

---

## 9. 记忆可视化 CLI 命令

新增了 `nanobot memory` 命令组，用于查看和管理记忆文件。

### 9.1 命令列表

| 命令 | 功能 |
|------|------|
| `nanobot memory status` | 显示所有记忆文件的状态（行数、大小、最后修改时间） |
| `nanobot memory view <file>` | 查看特定记忆文件的内容 |
| `nanobot memory search <query>` | 在所有记忆文件中搜索关键词 |

### 9.2 `memory status` - 状态概览

**功能**: 以表格形式展示所有 6 个记忆文件的状态。

**输出信息**:
- File: 文件名
- Lines: 行数
- Size: 文件大小（字节）
- Last Modified: 最后修改时间

**示例输出**:
```
Memory Files
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ File       ┃ Lines ┃   Size ┃ Last Modified     ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ PROFILE.md │    15 │    456 │ 2026-03-05 14:23 │
│ PROJECTS.md │    42 │  1,234 │ 2026-03-04 16:12 │
│ ...        │   ... │    ... │ ...               │
└────────────┴───────┴────────┴───────────────────┘
```

**实现位置**: `commands.py:1475-1513`

---

### 9.3 `memory view <file>` - 查看文件内容

**功能**: 查看指定记忆文件的完整内容。

**支持的文件名**:
- `profile` → `PROFILE.md`
- `projects` → `PROJECTS.md`
- `papers` → `PAPERS.md`
- `decisions` → `DECISIONS.md`
- `todos` → `TODOS.md`
- `history` → `HISTORY.md`

**使用示例**:
```bash
nanobot memory view profile
nanobot memory view todos
```

**实现位置**: `commands.py:1516-1553`

---

### 9.4 `memory search <query>` - 搜索记忆

**功能**: 在所有 6 个记忆文件中搜索关键词（不区分大小写）。

**特点**:
- 搜索所有记忆文件（包括 HISTORY.md）
- 显示匹配的文件名、行号、内容
- 长行自动截断（超过 100 字符显示省略号）

**使用示例**:
```bash
nanobot memory search "RAG"
nanobot memory search "论文"
```

**示例输出**:
```
Found 3 match(es):

PROJECTS.md:15
  RAG 配置: chunk_size=512, overlap=0.2

PAPERS.md:23
  这篇论文提出了一种新的 RAG 方法...
```

**实现位置**: `commands.py:1556-1604`

---

### 9.5 Skill 集成

`nanobot/skills/memory/SKILL.md` 已添加使用说明：

- 当用户询问记忆状态时，agent 会通过 `exec` 工具调用这些命令：
  ```bash
  nanobot memory status
  nanobot memory view profile
  nanobot memory search "关键词"
  ```

- 解析命令输出并自然总结给用户

---

## 10. 面试深度思考点

> **说明**: 以下是可以在面试中深入讨论的点，体现对 AI Agent Memory 系统的深度理解。

---

### 10.1 为什么设计成「分层记忆架构」?

**可以聊的点**:

1. **类比人类记忆系统**:
   - 短期记忆 (working memory): `Session.messages` + `get_history(max_messages)`
   - 长期记忆 (long-term memory): 6 个 MemoryFile
   - 这个设计借鉴了 **Atkinson-Shiffrin 记忆模型**（多重存储模型）
   - 短期记忆容量有限，需要通过复述（rehearsal）转化为长期记忆——正如我们通过固化（consolidation）把会话消息转化为记忆文件

2. **Context Window 约束**:
   - LLM 的 context window 是有限的（如 GPT-4 是 8K/32K/128K，Claude 是 200K）
   - 不能把所有历史都塞进去，否则：
     - 可能超过模型最大限制
     - 即使不超过，也会稀释当前 query 的注意力
     - Token 成本指数级增长
   - 需要区分「最近需要的」和「长期记住的」

3. **Token 成本考量**:
   - 每次都把完整历史发给 LLM 很贵
   - 固化后只加载相关记忆，显著节省 token
   - 例如：PROFILE + TODOS 通常只有几百 token，而完整历史可能有几万 token

4. **我的设计权衡**:
   - 保留最近一半消息 (`keep_count = memory_window // 2`)
   - 这样既有连续性（避免上下文突然断裂），又避免 context 过快增长
   - 这是「连续性」和「简洁性」的平衡

---

### 10.2 为什么用「结构化文件」而不是向量数据库存所有记忆?

**可以聊的点**:

1. **可解释性与可调试性**:
   - Markdown 文件人可以直接读、直接改
   - Vector DB 是黑盒，你不知道它存了什么、检索为什么准/不准
   - 这对个人助理特别重要——用户想知道 AI「记住了什么」，需要透明性
   - Debug 时，直接看 Markdown 文件就能定位问题，不需要写查询脚本

2. **不同类型记忆有不同访问模式**:

   | 记忆类型 | 访问模式 | 最佳存储 |
   |---------|---------|---------|
   | PROFILE | 总是全量加载 | 小文件，全量读 |
   | TODOS | 频繁全量替换 | 小文件，全量写 |
   | PAPERS | 增量追加，按主题检索 | 可选向量化 |
   | HISTORY | 只写不读 | append-only 日志 |

   - 用单一存储方案（向量数据库）去适配所有访问模式，本质是「过度设计」
   - 我的方案：**按需选择存储方式**，而不是为了技术炫酷而技术

3. **结构化 vs 非结构化**:
   - 我的设计：**语义理解 + 结构化存储**
   - 用 LLM 理解对话内容，决定更新哪个文件的哪个 section
   - 而不是简单地把所有内容都向量化
   - 这是「LLM 作为大脑，经典 CS 作为手脚」的设计哲学

4. **面试反问**: "你觉得全量向量存储在什么场景下更合适?"
   - 答案：企业级知识库、大量文档（10k+）、查询模式不确定的场景
   - 个人助理场景：记忆量可控，查询模式可预测，结构化文件更合适

---

### 10.3 记忆固化的触发策略——为什么是这三个条件?

**可以聊的点**:

1. **三个触发条件的设计意图**:

   | 条件 | 解决的问题 | 类比 |
   |------|-----------|------|
   | 消息窗口 (100条) | 保底机制，防止 context 爆了 | "记忆满了，整理一下" |
   | 会话暂停 (5分钟) | 用户可能走了，趁空闲整理 | "停下来反思一下" |
   | 重要内容 (关键词+2条) | 重要信息要及时记住 | "这个很重要，赶紧记下来" |

   - 这是「**混合策略**」，避免单一策略的缺陷

2. **为什么不只靠时间触发?**
   - 用户可能正在连续对话，中途打断不好（用户体验问题）
   - 重要决策需要立即记录，不能等 5 分钟（时效性问题）
   - 时间触发本质是「兜底」，不是「主力」

3. **为什么不只靠消息数触发?**
   - 如果都是废话，不需要记那么多（效率问题）
   - 如果有关键决策，10 条就该记了（时效性问题）
   - 2026-03-05 优化：把阈值从 5 降到 2，就是为了更快捕获重要信息

4. **我的工程细节**:
   - 固化在后台任务执行 (`asyncio.create_task`)
   - 用锁防止并发固化同一个 session
   - 固化失败不影响主流程 (降级策略)
   - 2026-03-05 新增：**重试机制**，当 LLM 忘记调用工具时重试 2 次

---

### 10.4 记忆选择性加载——关键词匹配 vs 语义检索?

**可以聊的点**:

1. **我为什么先用关键词匹配?**
   - 快，零延迟，不用调 embedding（性能）
   - 确定性，可预测（用户能理解为什么加载了这个文件）
   - 个人助理场景，query 通常比较明确（场景适配）
   - 这是「**过早优化是万恶之源**」的体现——先做简单够用的

2. **关键词匹配的局限性**:
   - "这篇文章讲了什么?" → 不会触发 PAPERS.md
   - "我之前想怎么做来着?" → 不会触发 DECISIONS.md
   - 缺少语义理解，只能做字面匹配

3. **我的 RAG 增强方案**:
   - 固化时用 RAG 搜索相关记忆，避免重复
   - 但**注入 context 时**还是用关键词（避免每次都 embedding）
   - 这是「**性价比**」权衡——在最需要的地方用 RAG，其他地方保持简单

4. **可以深入的方向**:
   - **Hybrid Search**: 关键词 + 语义混合，取长补短
   - **ReRank**: 先用 embedding 粗排，再用 LLM 精排（兼顾效率和准确）
   - **记忆重要性打分**: 给记忆加权重，优先加载高分的
   - **小 LLM 路由**: 用轻量模型（如 Claude Haiku）决定加载哪些文件

---

### 10.5 原子性与一致性——为什么这么重要?

**可以聊的点**:

1. **可能的故障场景**:
   - 写文件到一半程序 crash（电源断了、被 kill 了）
   - 写了 PROFILE 但没写 PROJECTS（部分失败）
   - LLM 返回了一半 tool call 就超时（网络问题）
   - 任何一个故障都可能导致记忆文件损坏或不一致

2. **我的解决方案**:
   - **临时文件 + rename**: 要么全成功，要么全回滚（单个文件的原子性）
   - **apply_operations_atomic**: 多个操作要么全成功，要么全回滚（多文件的原子性）
   - **回滚机制**: 保存旧内容，失败时恢复（容错）

3. **工程细节**:
   ```python
   # 这不是简单的 write_text
   tmp_path = path.with_suffix(path.suffix + ".tmp")
   tmp_path.write_text(content)
   tmp_path.rename(path)  # 原子操作（POSIX 系统保证）
   ```
   - rename 在 POSIX 系统是原子的
   - 临时文件确保崩溃时不会破坏原文件
   - 这是**经典文件系统设计模式**，被数据库、Git 等广泛使用

4. **面试可以问的点**:
   "如果让你实现分布式环境下的记忆一致性，你会怎么做?"
   - 答案: 分布式锁（Redis/ZK）、WAL（Write-Ahead Log）、版本号、幂等性、CRDT...

---

### 10.6 LLM 作为记忆处理器——把 save_memory 设计成 tool call 而不是单纯的文本生成?

**可以聊的点**:

1. **两种方案对比**:

   **方案 A: 纯文本生成**
   ```
   请把下面内容整理成记忆:
   ...
   输出格式:
   PROFILE:
   ...
   PROJECTS:
   ...
   ```
   - 缺点: 解析不稳定，容易格式错误
   - LLM 容易「自由发挥」，不严格遵守格式
   - 错误处理复杂，需要写很多解析和容错代码

   **方案 B: Tool Call (我的方案)**
   ```json
   {
     "history_entry": "...",
     "operations": [
       {"file": "projects", "action": "update_section", ...}
     ]
   }
   ```
   - 优点: 结构化，有 schema 校验，失败可重试
   - LLM 受到 schema 约束，输出更可靠
   - 可以单独验证每个 operation

2. **Tool Call 的优势**:
   - LLM 更不容易「幻觉」（受到 schema 约束）
   - 可以单独验证每个 operation
   - 可以部分失败、部分成功（虽然我做了原子）
   - 可以在 schema 层面做参数校验

3. **我的设计细节**:
   - `skip` 操作: LLM 可以决定什么都不做（灵活性）
   - `section` 参数: 支持细粒度更新，不用重写整个文件（效率）
   - 支持多文件操作: 一次可以更新多个记忆文件（表达力）
   - 2026-03-05 新增：**重试机制**，当 LLM 忘记调用工具时，有 2 次补救机会

---

### 10.7 记忆去重——为什么需要精确匹配 + 语义匹配两层?

**可以聊的点**:

1. **两层去重的设计**:

   ```
   精确匹配 (快, O(1), 无成本)
       ↓ 没命中?
   语义匹配 (慢, O(n), 需要 embedding)
       ↓ 没命中?
   真正追加
   ```

   - 这是「**瀑布式策略**」，先试便宜的，再试贵的

2. **为什么不只做语义匹配?**
   - 成本高: 每次都要 embedding（金钱成本）
   - 延迟大: 要等 embedding API 返回（时间成本）
   - 没必要: 很多情况精确匹配就够了（80/20 法则）

3. **性能优化**:
   - 限制最多检查 100 个段落（避免太大的文件）
   - 批量 embedding: 一次 API call 拿所有 embedding（O(1) API 调用）
   - 失败降级: embedding 失败就直接追加（不阻塞主流程）

4. **可以深入的方向**:
   - 本地 embedding 模型（如 Sentence-Transformers）减少延迟和成本
   - 缓存 embedding 结果（LRU Cache）
   - SimHash / MinHash 做快速相似性筛选（不用计算完整余弦相似度）

---

### 10.8 记忆可视化——为什么需要 CLI 命令?

**可以聊的点（新增）**:

1. **可观测性的重要性**:
   - 用户需要知道 AI「记住了什么」
   - 黑盒的向量数据库做不到这一点
   - 可读的 Markdown 文件 + CLI 工具 = 透明的记忆系统
   - 这是**用户信任**的基础——用户需要能看到 AI 的记忆

2. **为什么设计成这三个命令?**
   - `status`: 全局概览（仪表盘思维）——知道整体状态
   - `view`: 深入细节（放大镜思维）——看具体内容
   - `search`: 精准定位（搜索引擎思维）——找特定信息
   - 这三个命令覆盖了 90% 的使用场景

3. **Agent 自助使用**:
   - Skill 文档告诉 agent 如何调用这些命令
   - 用户问「我记住了什么?」时，agent 可以自己跑命令
   - 这是「**工具增强的智能**」——agent 用工具来回答问题
   - 不需要硬编码这种逻辑，agent 自己会学习使用工具

4. **工程细节**:
   - 用 Rich Table 做美化输出（用户体验）
   - 长行截断（避免屏幕污染）
   - 不区分大小写搜索（用户友好）

5. **可以深入的方向**:
   - 记忆可视化 dashboard（Web UI）
   - 记忆编辑工具（让用户直接修改）
   - 记忆导入导出（备份/迁移）

---

### 10.9 重试机制——为什么 LLM 会忘记调用工具，我们怎么应对?

**可以聊的点（2026-03-05 新增）**:

1. **问题分析**:
   - LLM 有时会在文本回应中描述「我已经更新了记忆」，但实际上没有调用 `save_memory` 工具
   - 这不是「幻觉」，而是「工具调用遗漏」——LLM 理解了任务，但选择了错误的执行方式
   - 原因可能是：prompt 不够明确、模型分心、随机采样的不确定性

2. **我的解决方案**:
   - **加强 Prompt**: 在 prompt 开头用醒目的 IMPORTANT 强调必须调用工具
   - **重试机制**: 当检测到没有工具调用时，自动重试 2 次
   - **渐进式提示**: 第一次重试用 IMPORTANT，第二次用 CRITICAL，强度递增
   - **空操作许可**: 明确告诉 LLM，如果没有内容可记，可以调用空操作，不要不调用工具

3. **设计权衡**:
   - 为什么只重试 2 次? 不是 5 次或 10 次?
     - 2 次是「性价比」平衡点——大多数情况前 2 次就能成功
     - 太多次会造成延迟和成本增加
   - 重试失败怎么办?
     - 返回 False，记录错误日志，不阻塞主流程
     - 下次有触发条件时会再次尝试固化（最终一致性）

4. **可以深入的方向**:
   - **RLHF**: 用强化学习让模型更愿意调用工具
   - **Prompt 优化**: 用 few-shot examples 展示正确和错误的做法
   - **Validation Layer**: 在 LLM 输出后加一层校验，确保工具被调用

---

### 10.10 这个设计的局限性和未来改进方向?

**可以聊的点（展示你有批判思维）**:

1. **当前设计的局限**:
   - 关键词匹配太简单，容易漏匹配
   - 没有记忆的「重要性评分」和「过期机制」
   - 没有跨 session 的记忆关联
   - 固化策略是硬编码的，不能自适应
   - 没有记忆编辑冲突处理（用户手动编辑 vs AI 自动更新）

2. **我的改进思路**:

   | 方向 | 方案 |
   |------|------|
   | 智能检索 | 用小 LLM 做「路由决策」，决定加载哪些记忆 |
   | 记忆重要性 | 让 LLM 给每次对话打重要性分，优先加载高分的 |
   | 记忆衰减 | 旧记忆降低权重，或者自动归档 |
   | 记忆关联 | 用 graph 存储记忆之间的关联 |
   | 自适应触发 | 用强化学习学习最佳固化时机 |

3. **面试官可能追问**: "这些改进里，你觉得哪个优先级最高?"
   - 我的答案: 智能检索（因为最容易落地，收益最大）
   - 可以用 Claude Haiku / GPT-4o-mini 这样的小模型，成本低、速度快
   - 比关键词匹配准确得多，又比全量语义检索可控

---

### 10.11 从这个项目中学到了什么关于 Agent 设计的经验?

**可以总结的点**:

1. **LLM 不是万能的，系统设计依然重要**:
   - LLM 很强大，但不能解决所有问题
   - 好的系统设计可以大幅降低 LLM 的负担
   - 比如：结构化记忆比自由文本更可靠
   - 比如：重试机制比完全依赖 LLM 一次做对更可靠

2. **权衡无处不在**:
   - 准确 vs 速度
   - 功能强大 vs 简单易用
   - 通用方案 vs 特定场景优化
   - 我的设计里到处都是权衡的痕迹——这才是工程的本质

3. **可观测性是生产系统的必需品**:
   - 我能快速 debug 记忆系统，因为 Markdown 文件是可读的
   - 如果全是向量数据库，会困难得多
   - CLI 工具让可观测性更进一步

4. **从用户体验出发**:
   - 用户希望 AI「记住」的方式和技术实现可能不一样
   - 比如：用户说「你要记住这个」，不是要你存在向量数据库里，而是要「真的记住」——下次能准确引用
   - 2026-03-05 优化：用户表达偏好时，我们需要更快地检测和固化，这就是从用户体验出发的改进

5. **最后的思考**:
   > "好的 AI Agent 设计，是让 LLM 做它最擅长的事——理解、推理、决策，而把存储、检索、一致性这些脏活累活，交给经典的计算机科学来解决。"

---

## 11. 记忆层改进历史

### 11.1 2026-03-05: 记忆固化优化

**问题**:
1. 用户表达偏好变更时，记忆没有及时更新
2. LLM 有时只描述变更但不调用 `save_memory` 工具
3. 重要内容关键词列表缺少偏好表达相关关键词

**改动**:
1. **扩展关键词**: 新增 11 个偏好表达关键词（"我喜欢"、"prefer"、"changed my mind" 等）
2. **降低阈值**: `DEFAULT_IMPORTANT_MIN_MESSAGES` 从 5 降至 2
3. **加强 Prompt**: 在 `_consolidate_direct()` 和 `_consolidate_with_rag()` 的 prompt 开头添加强制调用工具的说明
4. **重试机制**: 当 LLM 未调用工具时自动重试 2 次，包含 `_add_retry_instruction()` 辅助方法

**效果**:
- 用户偏好变更能被更快检测和固化
- 记忆固化的可靠性显著提升（有 2 次重试机会）
- 完全向后兼容

---

## 12. 记忆层改进建议

### 12.1 当前记忆层评价

**总体评价**: 这是一个**设计扎实、工程质量高**的个人助理记忆系统。

| 维度 | 评价 |
|------|------|
| **架构设计** | 分层清晰（Session ↔ ContextBuilder ↔ MemoryStore），职责明确 |
| **可观测性** | Markdown 文件 + CLI 可视化工具，非常透明 |
| **持久化** | append-only + 原子写入，数据安全 |
| **固化策略** | 三触发条件（窗口/暂停/重要内容），考虑周全 |
| **选择性加载** | 基于关键词的懒加载，节省 token |
| **去重机制** | 精确匹配 + 语义匹配，两层设计合理 |
| **重试机制** | 2 次重试 + 渐进式提示，可靠性高（2026-03-05 新增） |
| **用户体验** | Skill 集成让 agent 可以自助查询记忆 |

---

### 12.2 改进点清单（按优先级排序）

| 优先级 | 问题 | 影响 | 建议方案 |
|--------|------|------|----------|
| 🔴 P0 | 关键词匹配太简单 | 容易漏加载相关记忆 | 智能检索：小 LLM 路由决策 / Hybrid Search |
| 🟡 P1 | 没有记忆重要性评分 | 新旧记忆一视同仁 | 让 LLM 给每次对话打重要性分 |
| 🟡 P1 | `memory search` 只有关键词 | 语义相关内容搜不到 | 结合 RAG 做语义搜索 |
| 🟢 P2 | 没有记忆过期机制 | 记忆只会增长，不会清理 | 基于时间/访问频率的衰减/归档 |
| 🟢 P2 | 固化触发是硬编码 | 不能自适应不同用户 | 学习用户使用模式 |
| 🟢 P2 | 没有跨 session 关联 | 不同会话记忆是孤立的 | Graph 存储记忆关联 |

---

### 12.3 P0 改进：智能检索（优先级最高）

**问题**: 当前关键词匹配太简单，例如：
- 用户问："这篇文章讲了什么？" → 不会触发 `PAPERS.md`
- 用户问："我之前想怎么做来着？" → 不会触发 `DECISIONS.md`

**方案 A：小 LLM 路由决策**
```
用户 Query → 小 LLM (如 Claude Haiku / GPT-4o-mini) → 决定加载哪些记忆文件
```
- 成本低、速度快
- 可以用 few-shot learning 提升准确率
- 实现简单，收益大

**方案 B：Hybrid Search**
```
关键词搜索 + 语义搜索 → 结果融合 → rerank
```
- 更准确，但复杂度更高
- 需要维护记忆的 vector index

**推荐**: 先实施方案 A，性价比最高。

---

### 12.4 P1 改进：记忆重要性评分

**问题**: 当前所有记忆同等对待，但实际上：
- 有些对话很重要（技术决策、项目方向）
- 有些对话不重要（闲聊、临时调试）

**方案**:
1. **固化时打分**: LLM 在固化时给每条记忆打 0-1 分
2. **存储分数**: 在记忆文件中用 frontmatter 或特殊标记存储分数
3. **加载时排序**: 加载记忆时优先展示高分内容
4. **低分归档**: 分数低于阈值的记忆自动归档

**示例格式**:
```markdown
---
importance: 0.9
timestamp: 2026-03-05
---

决定用 RAG 来改进记忆检索，因为...
```

---

### 12.5 P1 改进：`memory search` 语义搜索

**问题**: 当前 `memory search` 只有关键词匹配，例如：
- 搜 "RAG" → 能找到
- 搜 "检索增强" → 找不到（如果记忆里写的是 "RAG"）

**方案**: 结合现有的 RAG 基础设施
1. 记忆文件也纳入 RAG 索引
2. `memory search` 命令支持 `--semantic` 选项做语义搜索
3. 或者默认混合搜索（关键词 + 语义）

**代码思路**:
```python
# memory_search 命令
if semantic:
    results = await rag_store.search_advanced(query)
else:
    results = keyword_search(query)
```

---

### 12.6 P2 改进：记忆过期/衰减机制

**问题**: 记忆只会无限增长，不会自动清理。

**方案**: 分层过期策略
```
活跃记忆 (最近 N 天) → 正常加载
归档记忆 (N~M 天) → 降低权重，需要时才加载
冷存储 (M 天以上) → 自动归档到 `memory/archive/`
```

**触发时机**:
- 每次 `memory status` 时检查
- 或者后台定期任务

---

### 12.7 P2 改进：自适应固化触发

**问题**: 当前固化触发是硬编码的（100 条 / 5 分钟），但不同用户习惯不同：
- 用户 A：话多，100 条可能 1 小时就到了
- 用户 B：话少，100 条可能要一周

**方案**: 简单的自适应策略
1. 统计用户的平均消息频率
2. 动态调整 `window_threshold`
3. 或者让用户在配置里自定义

---

### 12.8 P2 改进：跨 Session 记忆关联

**问题**: 不同 session 的记忆是孤立的，例如：
- Session 1：讨论了项目架构
- Session 2：问"我们之前怎么讨论架构的？" → 可能找不到

**方案**: 轻量级关联图
1. 每次固化时，用 LLM 提取关键词/实体
2. 建立简单的关联：`[记忆A] ←关联→ [记忆B]`
3. 检索时，如果找到记忆 A，也展示相关的记忆 B

---

### 12.9 总结

**当前状态**: ✅ 对于个人使用场景，已经**够用且好用**

**改进路径**:
1. **短期** (1-2 天): P0 智能检索（小 LLM 路由）
2. **中期** (1 周): P1 记忆重要性评分 + 语义搜索
3. **长期** (按需): P2 其他改进

**核心原则**: 保持当前架构的**简单性和可观测性**，不要为了改进而过度设计。
