# 代码质量修复 (2026-03-01)

## 概述

修复了 5 个代码质量问题，主要涉及测试文件更新、Linter 警告修复和日志改进。

---

## 修复详情

### 1. test_heartbeat_service.py (Critical)

**问题**:
- 导入了已删除的 `HEARTBEAT_OK_TOKEN`
- 使用旧参数 `on_heartbeat`，新 API 是 `on_execute`/`on_notify`
- 缺少必需的 `provider` 和 `model` 参数

**修复内容**:
- 删除 `HEARTBEAT_OK_TOKEN` 相关导入和测试
- 重写测试以匹配新 API
- 添加 mock provider 和 model
- 使用正确的 `on_execute` 和 `on_notify` 回调参数

**文件**: `tests/test_heartbeat_service.py`

---

### 2. Linter 警告修复

**问题**: W293 空白字符、I001 导入排序

**修复内容**:
- 修复 `nanobot/agent/__init__.py` 导入排序 (I001)
- 运行 `ruff check --fix` 自动修复 449 处 W293 空白字符问题

**文件**:
- `nanobot/agent/__init__.py`
- `nanobot/agent/skills.py`
- 以及其他多个文件

---

### 3. test_store_fix.py 移动

**问题**: 测试文件放在项目根目录

**修复内容**: 从根目录移动到 `tests/` 目录

**文件**: `test_store_fix.py` → `tests/test_store_fix.py`

---

### 4. print() 改为 logger

**问题**: `nanobot/config/loader.py` 使用 print 而非 logger

**修复内容**:
- 添加 `from loguru import logger` 导入
- 将 `print(f"Warning:...")` 改为 `logger.warning(...)`

**文件**: `nanobot/config/loader.py:39-42`

---

## 验证结果

| 检查项 | 状态 |
|--------|------|
| 测试导入无错误 | ✅ |
| heartbeat 测试通过 (4/4) | ✅ |
| test_store_fix.py 位置正确 | ✅ |
| loader.py 使用 logger | ✅ |
| agent/__init__.py 导入排序 | ✅ |

---

## 备注

仍存在约 64 个未修复的问题 (主要是 docstring 中的 W293 和部分预存在的 F821 错误)，这些不在本次修复范围内。
