# =========================================================================
# File: ai_shell_agent/toolsets/aider/toolset.py
# =========================================================================
"""
AI Code Copilot (Aider) toolset implementation.

Handles configuration and registration for tools that interact with the aider-chat library.
"""

import os
import threading
import traceback
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Langchain imports
from langchain_core.tools import BaseTool # Keep for list type hint
from rich.text import Text # Import Text for rich formatting

# Local imports
from ... import logger
from ...tool_registry import register_tools
from ...chat_state_manager import (
    get_current_chat,
    get_toolset_data_path
)
from ...config_manager import (
    get_current_model as get_agent_model,
    get_api_key_for_model,
    get_model_provider,
    normalize_model_name,
    ALL_MODELS
)
from ...utils.file_io import read_json, write_json
from ...utils.env import ensure_dotenv_key
from ...console_manager import get_console_manager
from .settings import (
    AIDER_DEFAULT_MAIN_MODEL, AIDER_DEFAULT_EDITOR_MODEL, AIDER_DEFAULT_WEAK_MODEL,
    AIDER_DEFAULT_EDIT_FORMAT, AIDER_DEFAULT_AUTO_COMMITS, AIDER_DEFAULT_DIRTY_COMMITS
)

from .prompts import AIDER_TOOLSET_PROMPT
from ...errors import PromptNeededError
from .texts import get_text

# --- Import Tool Instances ---
from .tools.tools import (
    aider_usage_guide_tool, add_code_file_tool, drop_code_file_tool,
    list_code_files_tool, edit_code_tool, submit_code_editor_input_tool,
    view_diff_tool, undo_last_edit_tool, close_code_editor_tool
)

# Get console manager instance
console = get_console_manager()

# Define edit formats supported by Aider (needed for config)
AIDER_EDIT_FORMATS = {
    "diff": "Traditional diff format",
    "edit_chunks": "Edit chunks format (easier to understand)",
    "whole_files": "Complete file replacements",
}

# --- Toolset metadata for discovery ---
toolset_name = get_text("toolset.name")
toolset_id = "aider"
toolset_description = get_text("toolset.description")
# Define required secrets here for the toolset registry
toolset_required_secrets: Dict[str, str] = {
    "OPENAI_API_KEY": "OpenAI API Key (used if GPT models are selected for Aider)",
    "GOOGLE_API_KEY": "Google AI API Key (used if Gemini models are selected for Aider)"
}

# --- Configuration Helpers (Remain Here) ---
def _prompt_for_single_model_config(role_name: str, current_value: Optional[str], default_value: Optional[str]) -> Optional[str]:
    """Helper to prompt for one of the coder models within configure_toolset."""
    console.display_message("SYSTEM:", get_text("config.model_header", role=role_name),
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    all_model_names = sorted(list(set(ALL_MODELS.values())))
    # Determine effective current model, considering agent default for Main model
    if role_name == 'Main/Architect':
        agent_default_model = get_agent_model() # Get the agent's model
        effective_current = current_value if current_value is not None else agent_default_model
    else:
        effective_current = current_value if current_value is not None else default_value

    current_marker_text = get_text("config.model_current_marker")

    # Build model list as a Text object
    model_list_text = Text()
    option_lines = []
    for model in all_model_names:
        marker = current_marker_text if model == effective_current else ""
        # Assemble the main line with potential marker
        line = Text.assemble(f"- {model}", marker)
        option_lines.append(line)
        # Add aliases on a new indented line if they exist
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        if aliases:
            alias_str = ', '.join(aliases)
            alias_line_text = get_text("config.model_aliases_suffix", alias_str=alias_str)
            # Indent alias line
            option_lines.append(Text("  ") + Text(alias_line_text, style=console.STYLE_SYSTEM_CONTENT))

    model_list_text = Text("\n").join(option_lines)

    # Print header and model list together
    console.display_message("SYSTEM:", get_text("config.model_available_title"),
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    console.console.print(model_list_text)

    # Determine display name for the prompt if keeping current
    current_display_name = effective_current or "Agent Default" if role_name == 'Main/Architect' else effective_current or "Aider Default"
    prompt_msg = get_text("config.model_prompt", role=role_name, current_setting=current_display_name)

    while True:
        try:
            selected = console.prompt_for_input(prompt_msg).strip()
            if not selected:
                console.display_message("INFO:", get_text("config.model_info_keep", setting=current_display_name),
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return current_value
            else:
                normalized_model = normalize_model_name(selected)
                if normalized_model in all_model_names:
                    # Return the selected normalized model name
                    console.display_message("INFO:", f"Selected '{normalized_model}' for {role_name}.",
                                          console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                    return normalized_model
                else:
                    # Simplified error message from text file
                    console.display_message("ERROR:", get_text("config.model_error_unknown", selected=selected),
                                          console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        except KeyboardInterrupt:
             console.display_message("WARNING:", get_text("config.model_warn_cancel"),
                                   console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
             return current_value

def _prompt_for_edit_format_config(current_value: Optional[str], default_value: Optional[str] = None) -> Optional[str]:
    """Prompts the user to select an Aider edit format using ConsoleManager."""
    console.display_message("SYSTEM:", get_text("config.format_header"),
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    valid_choices = {}
    default_display_name = get_text("config.format_default_name")
    current_marker_text = get_text("config.model_current_marker")

    # Build options list as Text object
    options_list_text = Text()
    option_lines = []

    # Add Default option
    default_marker = current_marker_text if current_value is None else ""
    option_lines.append(Text.assemble(
        "  ",
        ("0", console.STYLE_INPUT_OPTION),
        f": {default_display_name}",
        default_marker
    ))
    valid_choices['0'] = None

    # Add specific formats
    format_list = sorted(AIDER_EDIT_FORMATS.keys())
    for idx, fmt in enumerate(format_list, 1):
        description = AIDER_EDIT_FORMATS[fmt]
        marker = current_marker_text if fmt == current_value else ""
        option_lines.append(Text.assemble(
            "  ",
            (str(idx), console.STYLE_INPUT_OPTION),
            f": {fmt}",
            marker,
            f" - {description}"
        ))
        valid_choices[str(idx)] = fmt

    options_list_text = Text("\n").join(option_lines)
    console.console.print(options_list_text) # Print the assembled options

    max_idx = len(format_list)
    while True:
        try:
            # Prompt message fetched from texts.json
            prompt_msg = get_text("config.format_prompt", max_idx=max_idx)
            choice = console.prompt_for_input(prompt_msg).strip()

            if not choice:
                current_display = current_value if current_value is not None else default_display_name
                console.display_message("INFO:", get_text("config.format_info_keep", setting=current_display),
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return current_value
            elif choice in valid_choices:
                selected_format = valid_choices[choice]
                selected_display = selected_format if selected_format is not None else default_display_name
                console.display_message("INFO:", get_text("config.format_info_selected", setting=selected_display),
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return selected_format
            else:
                console.display_message("ERROR:", get_text("config.format_error_invalid"),
                                      console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        except KeyboardInterrupt:
            current_display = current_value if current_value is not None else default_display_name
            console.display_message("WARNING:", get_text("config.format_warn_cancel", setting=current_display),
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return current_value

# --- Configuration Function (Remains Here) ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path],
    dotenv_path: Path,
    current_config_for_prompting: Optional[Dict]
) -> Dict:
    """
    Configuration function for the File Editor (Aider) toolset.
    Prompts user based on current config (local or global), ensures secrets via .env utils,
    and saves the result appropriately (global-only or both). Uses defaults from settings.
    """
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat"
    logger.info(f"Configuring File Editor toolset ({context_name}). Global: {global_config_path}, Local: {local_config_path}, .env: {dotenv_path}")

    config_to_prompt = current_config_for_prompting or {}
    final_config = {}

    console.display_message("SYSTEM:", get_text("config.header", context=context_name),
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    save_location_info = get_text("config.save_location_global") if is_global_only else get_text("config.save_location_chat")
    console.display_message("SYSTEM:", get_text("config.instructions", save_location_info=save_location_info),
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    agent_model = get_agent_model()

    # --- Model Selection ---
    selected_main = _prompt_for_single_model_config("Main/Architect", config_to_prompt.get("main_model"), AIDER_DEFAULT_MAIN_MODEL)
    final_config["main_model"] = selected_main # Store None if user reset/cancelled/defaulted to None
    selected_editor = _prompt_for_single_model_config("Editor", config_to_prompt.get("editor_model"), AIDER_DEFAULT_EDITOR_MODEL)
    final_config["editor_model"] = selected_editor
    selected_weak = _prompt_for_single_model_config("Weak (Commits etc.)", config_to_prompt.get("weak_model"), AIDER_DEFAULT_WEAK_MODEL)
    final_config["weak_model"] = selected_weak

    # --- Edit Format Selection ---
    selected_format = _prompt_for_edit_format_config(config_to_prompt.get("edit_format"), AIDER_DEFAULT_EDIT_FORMAT)
    final_config["edit_format"] = selected_format # Store None if user chose default

    # --- Non-interactive settings ---
    final_config["auto_commits"] = config_to_prompt.get("auto_commits", AIDER_DEFAULT_AUTO_COMMITS)
    final_config["dirty_commits"] = config_to_prompt.get("dirty_commits", AIDER_DEFAULT_DIRTY_COMMITS)
    final_config["enabled"] = True # Assume enabled if configured

    # --- Ensure API Keys using ensure_dotenv_key ---
    console.display_message("SYSTEM:", get_text("config.api_key_check_header"),
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    actual_main_model = final_config.get("main_model")
    actual_editor_model = final_config.get("editor_model", AIDER_DEFAULT_EDITOR_MODEL) # Use default if None
    actual_weak_model = final_config.get("weak_model", AIDER_DEFAULT_WEAK_MODEL)       # Use default if None
    if actual_main_model is None:
        actual_main_model = agent_model
    models_to_check = {actual_main_model, actual_editor_model, actual_weak_model}

    checked_providers = set()
    required_keys_ok = True
    # Use the required_secrets defined at the module level
    api_key_descriptions = toolset_required_secrets

    for model_name in filter(None, models_to_check): # Filter out None values
        try:
             provider = get_model_provider(model_name)
             env_var = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"

             if env_var not in checked_providers:
                 logger.debug(f"Ensuring dotenv key: {env_var} for model {model_name}")
                 key_value = ensure_dotenv_key(
                     dotenv_path,
                     env_var,
                     api_key_descriptions.get(env_var) # Use description from toolset_required_secrets
                 )
                 if key_value is None:
                     required_keys_ok = False # Mark if user skips/cancels
                 checked_providers.add(env_var)
        except Exception as e:
            logger.error(f"Error checking API key for model '{model_name}': {e}", exc_info=True)
            console.display_message("ERROR:", get_text("config.api_key_error_check", model_name=model_name),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            required_keys_ok = False

    # --- Save Config Appropriately ---
    save_success_global = True
    save_success_local = True

    try:
        # Ensure directories exist before writing
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(global_config_path, final_config)
        logger.info(f"AI Code Copilot configuration saved to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to save AI Code Copilot config to global path {global_config_path}: {e}")

    if not is_global_only and local_config_path: # Check local_config_path exists
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.info(f"AI Code Copilot configuration saved to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to save AI Code Copilot config to local path {local_config_path}: {e}")

    # --- Confirmation Messages ---
    if save_success_global and (is_global_only or save_success_local):
        msg_key = "config.save_success_global" if is_global_only else "config.save_success_chat"
        console.display_message("INFO:", get_text(msg_key),
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        if not required_keys_ok:
             console.display_message("WARNING:", get_text("config.save_warn_missing_keys"),
                                   console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
    else:
        console.display_message("ERROR:", get_text("config.save_error_failed"),
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

    return final_config

# --- Define Toolset Structure (using imported instances) ---
toolset_tools: List[BaseTool] = [
    aider_usage_guide_tool, add_code_file_tool, drop_code_file_tool,
    list_code_files_tool, edit_code_tool, submit_code_editor_input_tool,
    view_diff_tool, undo_last_edit_tool, close_code_editor_tool
]

# --- Register Tools ---
register_tools(toolset_tools)
logger.debug(f"Registered AI Code Copilot toolset ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")