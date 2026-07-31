"""app.tools - deterministic tool calling (Phase 11 foundation, preview).

Tools give Aria real actions - reading the clock, computing arithmetic,
persisting notes - that run BEFORE the LLM in the answer chain because
they are deterministic, not guessed. See app/tools/base.py for the
design contract and app/tools/builtin.py for the built-ins.
"""

from app.tools.base import FunctionTool, Tool, ToolRegistry
from app.tools.builtin import default_tools

__all__ = ["FunctionTool", "Tool", "ToolRegistry", "default_tools"]
