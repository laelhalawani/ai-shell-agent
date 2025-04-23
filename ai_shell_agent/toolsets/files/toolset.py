# =========================================================================
# File: ai_shell_agent/toolsets/files/toolset.py
# =========================================================================
"""
File Manager toolset: Provides tools for direct file and directory manipulation.
Includes tools for create, read, delete, copy, move, rename, find, edit, history,
and backup management. Handles user confirmation for destructive/modifying actions.
"""
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Local Imports
from ... import logger, ROOT_DIR # Import ROOT_DIR for config fallback path
from ...tool_registry import register_tools
from ...utils.file_io import read_json, write_json
from ...utils.env import ensure_dotenv_key
from ...errors import PromptNeededError
from ...console_manager import get_console_manager
from ...chat_state_manager import (
    get_current_chat,
    get_toolset_data_path,
)
from ...texts import get_text as get_main_text
from .settings import (
    FILES_HISTORY_LIMIT, FIND_FUZZY_DEFAULT, FIND_THRESHOLD_DEFAULT,
    FIND_LIMIT_DEFAULT, FIND_WORKERS_DEFAULT
)
from .prompts import FILES_TOOLSET_PROMPT # Import the updated prompt text
from .texts import get_text # Toolset-specific texts
from .integration.find_logic import find_files_with_logic

# --- Import Tool Instances ---
# Imports tools which now import state helpers from .state
from .tools.tools import (
    file_manager_usage_guide_tool, create_tool, read_tool, delete_tool,
    overwrite_file_tool, find_replace_tool, copy_tool, move_tool,
    rename_tool, find_tool, exists_tool, history_tool, restore_tool,
    cleanup_tool
)
from langchain_core.tools import BaseTool # Keep this for the list type hint

console = get_console_manager()

# --- Toolset Metadata ---
toolset_id = "files"
toolset_name = get_text("toolset.name")
toolset_description = get_text("toolset.description")
toolset_required_secrets: Dict[str, str] = {}

# --- Configuration Function (Remains Here) ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path],
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
) -> Dict:
    """
    Configuration function for the File Manager toolset.
    Prompts user for history retrieval limit, using defaults from settings.
    """
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat"
    console.display_message(
        get_main_text("common.labels.system"),
        get_text("config.header"),
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )
    config_to_prompt = current_chat_config if current_chat_config is not None else {}
    final_config = {}
    try:
        default_limit = config_to_prompt.get("history_retrieval_limit", FILES_HISTORY_LIMIT)
        limit_str = console.prompt_for_input(
            get_text("config.prompt_limit"),
            default=str(default_limit)
        ).strip()
        try:
            limit = int(limit_str) if limit_str else default_limit
            if limit < 0: limit = 0
            final_config["history_retrieval_limit"] = limit
        except ValueError:
            console.display_message(
                get_main_text("common.labels.warning"),
                get_text("config.warn_invalid"),
                console.STYLE_WARNING_LABEL,
                console.STYLE_WARNING_CONTENT
            )
            final_config["history_retrieval_limit"] = default_limit
    except (KeyboardInterrupt, EOFError):
        console.display_message(
            get_main_text("common.labels.warning"),
            get_text("config.warn_cancel"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}
    except Exception as e:
        logger.error(f"Error during File Manager configuration: {e}", exc_info=True)
        console.display_message(
            get_main_text("common.labels.error"),
            get_text("config.error_generic", error=e),
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}
    save_success_global = True
    save_success_local = True
    try:
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(global_config_path, final_config)
        logger.info(f"File Manager configuration saved to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to save File Manager config to global path {global_config_path}: {e}")
    if local_config_path: # Check if local_config_path is provided
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.info(f"File Manager configuration saved to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to save File Manager config to local path {local_config_path}: {e}")
    if save_success_global and save_success_local:
        console.display_message(
            get_main_text("common.labels.info"),
            get_text("config.info_saved"),
            console.STYLE_INFO_LABEL,
            console.STYLE_INFO_CONTENT
        )
    else:
        console.display_message(
            get_main_text("common.labels.error"),
            get_text("config.error_save_failed"),
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )
    return final_config


# --- Define Toolset Structure (using imported instances) ---
toolset_tools: List[BaseTool] = [
    file_manager_usage_guide_tool,
    create_tool,
    read_tool,
    delete_tool,
    overwrite_file_tool,
    find_replace_tool,
    copy_tool,
    move_tool,
    rename_tool,
    find_tool,
    exists_tool,
    history_tool,
    restore_tool,
    cleanup_tool,
]

# --- Register Tools ---
# Ensure tools are registered using the updated instances from tools.py
register_tools(toolset_tools)
logger.debug(f"Registered File Manager toolset ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")
