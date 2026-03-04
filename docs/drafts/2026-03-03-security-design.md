# Nanobot 安全防护方案

**日期**: 2026-03-03
**状态**: 设计定稿（已审核）
**版本**: v1.1

---

## 1. 设计目标

为 nanobot 构建一套**纵深防御体系**，确保 AI 助手在用户系统上安全运行，即使 AI 被诱导执行恶意命令，也无法造成实质性的系统危害。

---

## 2. 安全架构概述

### 2.1 四层防护体系

```
┌─────────────────────────────────────────────────────────────────┐
│                    四层防护体系                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  第1层：workspace 限制      ← 逻辑层                            │
│         • 执行目录限制在 ~/.nanobot/workspace/                   │
│         • 阻止 ../ 路径遍历                                        │
│         • 阻止绝对路径访问 (/etc/passwd)                        │
│                                                                 │
│  第2层：firejail 沙箱       ← 进程层                            │
│         • 文件系统隔离：只能读写 whitelist 目录                  │
│         • 能力限制：无法挂载、无法修改内核                       │
│         • 网络隔离：可选，防止内网探测                            │
│                                                                 │
│  第3层：低权限用户          ← 系统层                            │
│         • 专用低权限用户运行 nanobot                             │
│         • 仅授权访问工作目录                                     │
│         • 即使突破前两层，仍是低权限                             │
│                                                                 │
│  第4层：系统权限            ← 物理层                            │
│         • 用户组权限控制                                         │
│         • 文件系统 ACL                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 攻击场景分析

| 攻击尝试 | 涉及工具 | workspace | firejail | 低权限用户 | 结果 |
|----------|----------|-----------|-----------|------------|------|
| `rm -rf /` | exec | ❌ 不拦截 | ✅ 只能在 whitelist 内删除 | ✅ 无权限修改系统 | ✅ 安全 |
| `cat ~/.ssh/id_rsa` | exec | ❌ 不拦截 | ✅ 无法访问 whitelist 外 | ✅ 无法读取 | ✅ 安全 |
| `read_file("~/.ssh/id_rsa")` | read_file | ✅ 阻止路径 | ⚠️ 不适用（Python 内调用） | ✅ 无权限读取 | ✅ 安全 |
| `write_file("/etc/passwd", "...")` | write_file | ✅ 阻止路径 | ⚠️ 不适用（Python 内调用） | ✅ 无权限写入 | ✅ 安全 |
| `cd / && rm -rf *` | exec | ⚠️ `cd` 可能漏检 | ✅ 只能在 whitelist 内操作 | ✅ 无权限 | ✅ 安全 |
| `python -c "os.system('rm -rf /')"` | exec | ❌ 不拦截 | ✅ 只能在 whitelist 内操作 | ✅ 无权限 | ✅ 安全 |
| `curl http://192.168.1.1` | exec | ❌ 不拦截 | ✅ `--net=none` 可选禁用网络 | ✅ 无网络访问 | ✅ 安全 |

**⚠️ 重要提示**:
- `read_file` / `write_file` / `edit_file` / `list_dir` 在 Python 进程内直接调用，**不经过 firejail**，因此必须依赖：
  1. **第1层**：workspace 路径限制（`_resolve_path` + `allowed_dir`）
  2. **第3层**：低权限用户运行 nanobot
  3. **第4层**：文件系统权限控制

---

## 3. 详细设计

### 3.1 第1层：workspace 限制

**实现位置**:
- `nanobot/agent/tools/shell.py` - ExecTool
- `nanobot/agent/tools/filesystem.py` - ReadFileTool, WriteFileTool, EditFileTool, ListDirTool

**⚠️ 重要**: 所有文件系统工具都需要实施相同的 workspace 限制，而不仅仅是 shell 命令。

```python
class ExecTool:
    def __init__(
        self,
        # ...
        restrict_to_workspace: bool = True,  # 默认启用（需从 False 修改）
    ):
        # ...
```

**功能 (ExecTool)**:
- 执行目录限制在 `~/.nanobot/workspace/`
- 阻止 `../` 路径遍历
- 阻止绝对路径访问（如 `/etc/passwd`）

**功能 (Filesystem Tools)**:
- 通过 `_resolve_path()` 函数统一验证路径
- `allowed_dir` 参数设置为 `~/.nanobot/workspace/`
- 使用 `resolved.relative_to(allowed_dir)` 确保路径在允许目录内
- 自动解析相对路径、`~` 展开等

**限制**:
- 不能完全拦截 shell 中的 `~` 或 `$HOME` 引用（需要 firejail 配合）
- Filesystem 工具不经过 firejail，必须依赖第1、3、4层防护

---

### 3.2 第2层：firejail 沙箱

**实现位置**: `nanobot/agent/tools/shell.py`

```python
import shlex

class ExecTool:
    def __init__(
        self,
        # ...
        use_firejail: bool = True,  # 新增配置
        firejail_options: list[str] = None,  # 自定义选项
    ):
        self.use_firejail = use_firejail
        self.firejail_options = firejail_options or [
            "--quiet",
            "--noprofile",
            "--shell=/bin/bash",
        ]

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        # ...

        # 构建 firejail 命令
        if self.use_firejail:
            workspace_path = Path(cwd).resolve()
            cmd_parts = [
                "firejail",
                *self.firejail_options,
                f"--whitelist={workspace_path}",
                "--readonly=/home",           # 用户目录只读
                "--readonly=/etc",            # 系统配置只读
                "--private-tmp",              # 私有临时目录
                "sh", "-c", shlex.quote(command),
            ]
            final_command = " ".join(cmd_parts)
        else:
            final_command = command

        process = await asyncio.create_subprocess_shell(
            final_command,
            # ...
        )
```

**firejail 选项说明**:

| 选项 | 作用 |
|------|------|
| `--quiet` | 静默模式，减少输出 |
| `--noprofile` | 不加载默认配置，完全自定义 |
| `--shell=/bin/bash` | 指定 shell |
| `--whitelist={workspace}` | 允许访问的工作目录 |
| `--readonly=/home` | 用户目录只读 |
| `--readonly=/etc` | 系统配置只读 |
| `--private-tmp` | 私有临时目录 |

**macOS 兼容性**:
- firejail 主要针对 Linux
- macOS 可选方案：`sandbox-exec`（内置）或 Docker

---

### 3.3 第3层：低权限用户

**创建专用用户**:

```bash
# 创建专用用户（无登录权限）
sudo useradd -r -s /bin/false -M nanobot

# 创建工作目录
sudo mkdir -p /home/user/.nanobot/workspace
sudo chown -R nanobot:nanobot /home/user/.nanobot/workspace
sudo chmod 755 /home/user/.nanobot/workspace

# 限制敏感目录访问
sudo chmod 700 /home/user/.ssh
sudo chmod 700 /home/user/.aws
sudo chmod 700 /home/user/.config
```

**用户权限设计**:

| 权限 | 允许 | 拒绝 |
|------|------|------|
| 读取 | workspace 目录 | ~/.ssh、~/.aws、/etc |
| 写入 | workspace 目录 | 系统目录 |
| 执行 | workspace 内的脚本 | sudo、su |
| 网络 | 可选限制 | - |

**配置 nanobot 使用低权限用户**:

> ⚠️ **注意**: Python 中切换用户需要 root 权限，且会影响整个进程。
> 建议使用以下方式之一：

**方式 1: sudo 启动（推荐）**
```bash
sudo -u nanobot nanobot agent -m "Hello"
```

**方式 2: systemd 服务**
```ini
# /etc/systemd/system/nanobot.service
[Service]
User=nanobot
Group=nanobot
ExecStart=/usr/bin/nanobot gateway
```

**方式 3: Docker 容器**
```dockerfile
USER nanobot
```

**方式 4: systemd 服务配置**（可选）:

```ini
# /etc/systemd/system/nanobot.service
[Service]
User=nanobot
Group=nanobot
ExecStart=/usr/bin/nanobot gateway

[Install]
WantedBy=multi-user.target
```

---

### 3.4 第4层：系统权限

**文件权限设置**:

```bash
# 工作目录权限
chmod 755 ~/.nanobot
chmod 755 ~/.nanobot/workspace
chmod 644 ~/.nanobot/config.json  # 配置可读

# 敏感目录权限
chmod 700 ~/.ssh
chmod 700 ~/.aws
chmod 700 ~/.config

# 禁止其他用户访问
chmod -R o-rwx ~/.nanobot/
```

---

## 4. 配置方案

### 4.1 配置文件

```python
# nanobot/config/schema.py
class ToolsConfig(Base):
    # ... 现有配置 ...

    # 新增安全配置
    restrict_to_workspace: bool = True          # 默认启用工作区限制（exec + filesystem 工具）
    use_firejail: bool = True                    # 默认启用 firejail
    firejail_net: str = "unrestricted"          # "unrestricted" | "none" | "whitelist"
```

**⚠️ 已存在的配置** (无需新增):
- `AgentDefaults.workspace`: 工作区路径，默认值 `"~/.nanobot/workspace"` (schema.py:207)
- `Config.workspace_path`: 通过 property 访问已展开的工作区路径 (schema.py:432-435)

**⚠️ Filesystem 工具配置** (已实现):
- `ReadFileTool`、`WriteFileTool`、`EditFileTool`、`ListDirTool` 已通过 `allowed_dir` 参数限制
- `AgentLoop._register_default_tools()` 已传入 `allowed_dir` (loop.py:289-291)
- `_resolve_path()` 函数已实现路径验证 (filesystem.py:10-21)

### 4.2 配置组合

| 场景 | restrict_to_workspace | use_firejail | run_as_user | 推荐度 |
|------|----------------------|--------------|-------------|--------|
| 开发测试 | False | False | 当前用户 | ⭐ |
| 个人使用 | True | True | 当前用户 | ⭐⭐⭐⭐ |
| 生产环境 | True | True | nanobot | ⭐⭐⭐⭐⭐ |

---

## 5. 使用方式

### 5.1 快速启用

```bash
# 方式1：直接运行（自动使用 workspace + firejail）
nanobot agent -m "Hello"

# 方式2：使用低权限用户（需要先创建用户）
sudo -u nanobot nanobot agent -m "Hello"

# 方式3：Docker 方式
docker run -v ~/.nanobot:/home/user/.nanobot nanobot agent -m "Hello"
```

### 5.2 验证安全配置

```bash
# 验证 firejail 是否生效
nanobot agent -m "exec whoami"

# 应该输出: nanobot (如果是低权限用户模式)
# 或者: [username] (如果是当前用户模式)
```

---

## 6. 实现计划

| 阶段 | 任务 | 预估时间 | 状态 |
|------|------|----------|------|
| Phase 1 | 修改 `restrict_to_workspace` 默认值为 `True` <br> - schema.py:410 (ToolsConfig) <br> - shell.py:21 (ExecTool) <br> - loop.py:187 (AgentLoop) | 0.5 小时 | 待实现 |
| Phase 2 | 添加 `use_firejail` 配置项和执行逻辑 | 1 小时 | 待实现 |
| Phase 3 | **确认** filesystem 工具 `allowed_dir` 机制已就绪 <br> - `_resolve_path()` 已实现 (filesystem.py:10-21) <br> - `allowed_dir` 已在 `AgentLoop` 传入 (loop.py:289-291) | - | 已存在 |
| Phase 4 | **补充** `run_as_user` 作为部署建议（非代码实现）<br> - Python 切换用户需要 root 权限 <br> - 建议通过 `sudo -u nanobot` 或 systemd 配置实现 | - | 部署建议 |
| Phase 5 | 添加 `firejail_net` 网络控制选项 | 0.5 小时 | 待实现 |
| Phase 6 | 编写安全配置脚本（创建用户、设置权限） | 0.5 小时 | 待实现 |
| Phase 7 | 测试验证（exec + filesystem 工具） | 1 小时 | 待实现 |

---

## 7. 安全性总结

| 层级 | 防护 | 威胁 |
|------|------|------|
| 1 | workspace | 误操作、路径遍历 |
| 2 | firejail | 漏洞利用、权限提升 |
| 3 | 低权限用户 | 沙箱突破 |
| 4 | 系统权限 | 极限防御 |

**即使攻击者突破了 firejail 沙箱**，它仍然只是一个低权限用户，无法：
- 修改系统文件
- 安装恶意软件
- 访问其他用户数据
- 提权到 root

---

## 8. 参考资料

- [Firejail 官方文档](https://firejail.wordpress.com/)
- [Linux 用户权限最佳实践](https://www.cyberciti.biz/tips/linux-security.html)
- [OWASP AI Security](https://owasp.org/www-project-ai-security/)

---

## 附录：代码审核记录

**审核日期**: 2026-03-04

### 已存在的功能

| 功能 | 位置 | 说明 |
|------|------|------|
| `_resolve_path()` | `filesystem.py:10-21` | 路径解析与验证函数，已支持 `allowed_dir` 限制 |
| `allowed_dir` 传入 | `loop.py:289-291` | `AgentLoop._register_default_tools()` 已根据 `restrict_to_workspace` 传入 `allowed_dir` |
| `AgentDefaults.workspace` | `schema.py:207` | 工作区路径配置，默认值 `"~/.nanobot/workspace"` |
| `Config.workspace_path` | `schema.py:432-435` | Property 方式访问已展开的工作区路径 |

### 需要修改的默认值

| 配置项 | 当前值 | 目标值 | 位置 |
|--------|--------|--------|------|
| `ToolsConfig.restrict_to_workspace` | `False` | `True` | `schema.py:410` |
| `ExecTool.restrict_to_workspace` | `False` | `True` | `shell.py:21` |
| `AgentLoop.restrict_to_workspace` | `False` | `True` | `loop.py:187` |

### 关于 `run_as_user` 的说明

经审核，在 Python 中实现运行时切换用户存在以下限制：
1. `os.setuid()` 需要 root 权限
2. 切换用户会影响整个进程，不适合仅用于 exec 工具
3. 更安全的方式是在进程启动时通过 `sudo -u nanobot` 或 systemd `User=` 配置完成

因此，`run_as_user` 应作为**部署建议**而非代码实现。