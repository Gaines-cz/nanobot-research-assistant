"""Tool registry for dynamic tool management."""

from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.
    
    Allows dynamic registration and execution of tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            # More helpful error with available tools count
            available = self.tool_names
            return (
                f"Error: Tool '{name}' not found.\n"
                f"Available tools ({len(available)}): {', '.join(available[:10])}"
                + (f" and {len(available)-10} more..." if len(available) > 10 else "")
            )

        try:
            errors = tool.validate_params(params)
            if errors:
                # Add parameter hints based on error type
                hints = []
                for err in errors:
                    if "required" in err:
                        hints.append("Missing required parameter. Check tool schema.")
                    elif "type" in err:
                        hints.append("Parameter type mismatch. Expected format may differ.")
                return (
                    f"Error: Invalid parameters for tool '{name}': "
                    + "; ".join(errors)
                    + (f"\nHints: {'; '.join(hints)}" if hints else "")
                    + _HINT
                )
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
