"""
Central registry for discovering and managing toolset metadata.
"""
import os
import importlib
import pkgutil
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from pathlib import Path # Added Path

from .. import logger, ROOT_DIR # Use relative import for logger/ROOT_DIR
from ..tool_registry import get_tool # Import get_tool for validation

TOOLSETS_DIR = ROOT_DIR / 'ai_shell_agent' / 'toolsets'

@dataclass
class ToolsetMetadata:
    id: str # Use the directory name as ID
    name: str
    description: str
    start_tool: Optional[BaseTool] = None
    tools: List[BaseTool] = field(default_factory=list)
    prompt_content: Optional[str] = None
    module_path: Optional[str] = None # Store module path for reference
    # --- NEW FIELDS ---
    config_defaults: Dict[str, Any] = field(default_factory=dict)
    # Signature: configure_func(config_file_path: Path, current_config: Optional[Dict]) -> Dict
    configure_func: Optional[Callable[[Path, Optional[Dict]], Dict]] = None

# Global registry for toolset metadata
_toolsets_registry: Dict[str, ToolsetMetadata] = {}
_discovery_done = False # Flag to prevent repeated discovery

def discover_and_register_toolsets():
    """
    Scans the toolsets directory, imports toolset.py modules,
    extracts metadata, configuration info, and validates tools.
    """
    global _toolsets_registry, _discovery_done
    if _discovery_done:
        return # Avoid re-discovery

    logger.debug(f"Discovering toolsets in: {TOOLSETS_DIR}")
    for item in os.listdir(TOOLSETS_DIR):
        toolset_path = TOOLSETS_DIR / item
        toolset_module_file = toolset_path / "toolset.py"
        toolset_id = item # Use directory name as ID

        if toolset_path.is_dir() and toolset_module_file.exists() and toolset_id != "__pycache__":
            toolset_module_name = f"ai_shell_agent.toolsets.{toolset_id}.toolset"
            prompt_module_name = f"ai_shell_agent.toolsets.{toolset_id}.prompts"
            prompt_content_variable_name = f"{toolset_id.upper()}_TOOLSET_PROMPT" # Convention

            try:
                logger.debug(f"Importing toolset module: {toolset_module_name}")
                module = importlib.import_module(toolset_module_name)

                # --- Extract Metadata (including new config parts) ---
                name = getattr(module, "toolset_name", None)
                description = getattr(module, "toolset_description", None)
                start_tool_instance = getattr(module, "toolset_start_tool", None)
                tools_list = getattr(module, "toolset_tools", []) # Should be list of instances
                config_defaults = getattr(module, "toolset_config_defaults", {}) # Get defaults dict
                configure_func = getattr(module, "configure_toolset", None) # Get config function

                # --- Validation ---
                if not (name and description):
                    logger.warning(f"Skipping toolset '{toolset_id}'. Missing required metadata: 'toolset_name' or 'toolset_description'.")
                    continue
                # Validate config_defaults is a dict
                if not isinstance(config_defaults, dict):
                     logger.warning(f"Skipping toolset '{toolset_id}'. Invalid 'toolset_config_defaults' (must be a dict).")
                     continue
                # Validate configure_func is callable
                if configure_func is not None and not callable(configure_func):
                     logger.warning(f"Skipping toolset '{toolset_id}'. Invalid 'configure_toolset' (must be a function).")
                     continue

                # Validate tools against tool registry
                validated_tools = []
                all_tools_valid = True
                for tool_instance in tools_list:
                    if isinstance(tool_instance, BaseTool) and get_tool(tool_instance.name):
                        validated_tools.append(tool_instance)
                    else:
                        logger.error(f"Tool '{getattr(tool_instance, 'name', 'UNKNOWN')}' in '{toolset_module_name}' not registered or invalid. Skipping toolset.")
                        all_tools_valid = False; break
                if not all_tools_valid: continue

                start_tool_valid = True
                if start_tool_instance:
                    if not (isinstance(start_tool_instance, BaseTool) and get_tool(start_tool_instance.name)):
                        logger.error(f"Start tool '{getattr(start_tool_instance, 'name', 'UNKNOWN')}' in '{toolset_module_name}' not registered or invalid. Disabling.")
                        start_tool_instance = None; start_tool_valid = False

                # Get prompt content from prompts module
                prompt_content = None
                try:
                    prompts_module = importlib.import_module(prompt_module_name)
                    prompt_content = getattr(prompts_module, prompt_content_variable_name, None)
                    if not isinstance(prompt_content, str): prompt_content = None
                except Exception: pass # Ignore prompt errors


                # --- Register Validated Metadata ---
                if toolset_id in _toolsets_registry:
                    logger.warning(f"Toolset ID '{toolset_id}' already registered. Overwriting.")

                _toolsets_registry[toolset_id] = ToolsetMetadata(
                    id=toolset_id,
                    name=name,
                    description=description,
                    start_tool=start_tool_instance,
                    tools=validated_tools,
                    prompt_content=prompt_content,
                    module_path=toolset_module_name,
                    config_defaults=config_defaults, # Store defaults
                    configure_func=configure_func     # Store function reference
                )
                logger.info(f"Registered toolset: '{name}' (ID: {toolset_id}) with config function {'✓' if configure_func else '✗'}")

            except ImportError as e:
                logger.error(f"Failed to import toolset module {toolset_module_name}: {e}")
            except AttributeError as e:
                 logger.error(f"Attribute error while processing toolset module {toolset_module_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error registering toolset from {toolset_module_name}: {e}")

    _discovery_done = True # Mark discovery as complete

# --- Accessor Functions ---
def get_registered_toolsets() -> Dict[str, ToolsetMetadata]:
    """Returns a copy of the registered toolsets metadata."""
    if not _discovery_done: discover_and_register_toolsets()
    return dict(_toolsets_registry) # Return a copy

def get_toolset(id: str) -> Optional[ToolsetMetadata]:
    """Gets metadata for a specific toolset by ID."""
    if not _discovery_done: discover_and_register_toolsets()
    return _toolsets_registry.get(id)

def get_toolset_names() -> List[str]:
    """Returns a list of registered toolset names (display names)."""
    if not _discovery_done: discover_and_register_toolsets()
    return sorted([info.name for info in _toolsets_registry.values()])

def get_toolset_ids() -> List[str]:
    """Returns a list of registered toolset IDs (directory names)."""
    if not _discovery_done: discover_and_register_toolsets()
    return sorted(list(_toolsets_registry.keys()))

# Call discovery when the module is loaded
discover_and_register_toolsets()