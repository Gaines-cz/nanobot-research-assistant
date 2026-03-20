---
name: memory
description: Multi-file memory system with incremental operations and selective loading.
always: true
---

# Memory

## 记忆类型

| 类型 | 用途 |
|------|------|
| HISTORY | 记录发生的事件、讨论、动作 |
| KNOWLEDGE | 学到的知识、论文笔记、研究发现 |
| DECISIONS | 做过的决定及原因 |
| PROJECTS | 项目进度、任务、技术方案 |

## 工具

当问题涉及过往事件、知识、决定或项目时，使用 search_memory 工具搜索相关记忆：

search_memory(query, type?, limit?)
- query: 搜索内容
- type: 可选（history/knowledge/decisions/projects）
- limit: 返回数量，默认5

