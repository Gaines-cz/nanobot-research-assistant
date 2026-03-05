---
name: memory
description: Multi-file memory system with incremental operations and selective loading.
always: true
---

# Memory

## Structure

```
memory/
├── PROFILE.md     # User profile (research direction, preferences). Stable.
├── PROJECTS.md    # Project knowledge (tech stack, architecture). Semi-stable.
├── PAPERS.md      # Paper notes (papers read, key findings). Incremental.
├── DECISIONS.md   # Decision records (why A over B). Incremental.
├── TODOS.md       # Current tasks and next steps. Frequently updated.
└── HISTORY.md     # Append-only event log. NOT loaded into context.
```

## Selective Loading

Memory files are loaded selectively based on context:

- **Always loaded**: PROFILE.md, TODOS.md
- **Keyword-triggered**:
  - `论文/paper/arxiv/研究` → PAPERS.md
  - `项目/project/代码/架构` → PROJECTS.md
  - `为什么/决策/why/decision` → DECISIONS.md

## View Memory Status

When users ask about their current memory status (e.g., "我现在记住了什么？", "查看记忆现状"), use the `exec` tool to run the CLI command:

```bash
nanobot memory status
```

For viewing a specific file:
```bash
nanobot memory view profile
nanobot memory view todos
```

For searching memory content:
```bash
nanobot memory search "RAG"
nanobot memory search "论文"
```

Parse the command output and summarize the results naturally for the user.

## Search Past Events

```bash
grep -i "keyword" memory/HISTORY.md
```

Use the `exec` tool to run grep. Combine patterns: `grep -iE "meeting|deadline" memory/HISTORY.md`

## When to Update Memory

Update memory files directly using `edit_file` or `write_file`:

- **PROFILE.md**: User preferences, research direction, tools used
- **PROJECTS.md**: Project architecture, tech decisions, progress
- **PAPERS.md**: Paper summaries, key findings, citations
- **DECISIONS.md**: Why you chose A over B, trade-offs considered
- **TODOS.md**: Current tasks, next steps, blockers

## Auto-consolidation

Old conversations are automatically processed via the `save_memory` tool:
- History entry is appended to HISTORY.md
- Incremental operations update specific files (append/update_section/replace)
- You don't need to manage this manually.