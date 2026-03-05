"""Shell execution tool."""

import asyncio
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = True,
        path_append: str = "",
        use_firejail: bool = True,
        firejail_strict: bool = True,
        firejail_options: list[str] | None = None,
        firejail_net: str = "unrestricted",
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
            # Pipe download execution: curl/wget/fetch | sh/bash/zsh
            r"(?:(?:^|[;&|])\s*(?:curl|wget|fetch)\b[^|]*\|[^|]*(?:sh|bash|zsh)\b)",
            # Download to file then execute: curl/wget -o file; sh file
            r"(?:(?:curl|wget|fetch)\b[^;]*?(?:-o\s|--output\s|>\s*)\S+.*?(?:&&|;)\s*(?:sh|bash|zsh)\b)",
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.use_firejail = use_firejail
        self.firejail_strict = firejail_strict
        self.firejail_options = firejail_options or [
            "--quiet",
            "--noprofile",
            "--shell=/bin/sh",
        ]
        self.firejail_net = firejail_net

    def _is_firejail_available(self) -> bool:
        """Check if firejail is available on the system."""
        return shutil.which("firejail") is not None

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                }
            },
            "required": ["command"]
        }

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()

        # Check cwd exists
        if not os.path.isdir(cwd):
            return f"Error: Working directory does not exist: {cwd}"

        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        # Determine final command (with or without firejail)
        use_firejail = self.use_firejail and self._is_firejail_available()
        final_command = command

        if use_firejail:
            try:
                workspace_path = Path(cwd).resolve()
                # Build firejail arguments as a list first
                args = ["firejail"]
                args.extend(self.firejail_options)
                args.append(f"--whitelist={workspace_path}")
                # Platform-specific paths: /home and /etc are Linux-specific
                if sys.platform == "linux":
                    args.append("--readonly=/home")
                    args.append("--readonly=/etc")
                args.append("--private-tmp")
                if self.firejail_net == "none":
                    args.append("--net=none")
                # The command to run inside firejail - consistent shell
                args.extend(["sh", "-c", command])
                # Now properly quote each argument for use in a shell
                final_command = " ".join(shlex.quote(str(arg)) for arg in args)
            except Exception as e:
                if self.firejail_strict:
                    return f"Error: Firejail setup failed (strict mode): {str(e)}"
                # Fall back to no firejail if strict is False
                logger.warning("Firejail setup failed, falling back to no sandbox: {}", str(e))
                final_command = command

        try:
            process = await asyncio.create_subprocess_shell(
                final_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                # Graceful timeout handling: try terminate first, then kill
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    # If terminate fails or process is already gone, kill it
                    try:
                        process.kill()
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except (asyncio.TimeoutError, ProcessLookupError):
                        pass
                return f"Error: Command timed out after {self.timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            cwd_path = Path(cwd).resolve()

            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
            # Enhanced POSIX path regex: matches absolute paths at start, after whitespace, or after redirection/operators
            # Also handles quoted paths with both single and double quotes
            posix_paths = re.findall(r'(?:^|[\s|;&<>()$`])(/[^\s\"\';|&<>()$`]+)', cmd)
            # Also extract quoted absolute paths
            posix_paths.extend(re.findall(r'(?:^|[\s|>])"(/[^"]+)"', cmd))
            posix_paths.extend(re.findall(r"(?:^|[\s|>])'(/[^']+)'", cmd))

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None
