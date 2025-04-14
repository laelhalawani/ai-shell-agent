"""
Tool Registry for AI Shell Agent.

A centralized registry for tools to break circular dependencies 
between modules that both define and use tools.
"""
from typing import List, Dict
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
import threading
from . import logger

# Global registry storage
_tools_registry: List[BaseTool] = []
_tools_dict: Dict[str, BaseTool] = {}
_tools_functions: List[Dict] = []
_registry_lock = threading.Lock()

def register_tool(tool: BaseTool):
    """Registers a tool."""
    with _registry_lock:
        if tool.name not in _tools_dict:
            _tools_registry.append(tool)
            _tools_dict[tool.name] = tool
            _tools_functions.append(convert_to_openai_function(tool))
            # Consider adding logging here if desired
            # print(f"Registered tool: {tool.name}") # DEBUG
            logger.debug(f"Registered tool: {tool.name}")

def register_tools(tools_list: List[BaseTool]):
    """Registers multiple tools."""
    with _registry_lock:
        for tool in tools_list:
            if tool.name not in _tools_dict:
                _tools_registry.append(tool)
                _tools_dict[tool.name] = tool
                _tools_functions.append(convert_to_openai_function(tool))
                # print(f"Registered tool: {tool.name}") # DEBUG
                logger.debug(f"Registered tool: {tool.name}")

def get_tool(name: str) -> BaseTool | None:
    """Gets a tool by name."""
    with _registry_lock:
        return _tools_dict.get(name)

def get_all_tools() -> List[BaseTool]:
    """Gets a copy of the list of all registered tools."""
    with _registry_lock:
        return list(_tools_registry)  # Return a copy

def get_all_tools_dict() -> Dict[str, BaseTool]:
    """Gets a copy of the dictionary of all registered tools."""
    with _registry_lock:
        return dict(_tools_dict)  # Return a copy

def get_all_openai_functions() -> List[Dict]:
    """Gets a copy of the list of OpenAI function definitions."""
    with _registry_lock:
        return list(_tools_functions)  # Return a copy

def clear_registry():
    """Clears the registry (useful for testing)."""
    global _tools_registry, _tools_dict, _tools_functions
    with _registry_lock:
        _tools_registry = []
        _tools_dict = {}
        _tools_functions = []