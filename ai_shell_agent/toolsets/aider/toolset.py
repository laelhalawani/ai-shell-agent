# ai_shell_agent/toolsets/aider/toolset.py
"""
File Editor toolset implementation.

Contains tools for interacting with the aider-chat library
to edit files and view changes.
"""

import os
import threading
import traceback
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Type

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor
from prompt_toolkit import prompt as prompt_toolkit_prompt
from pydantic import BaseModel, Field
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
from .integration.integration import (
    recreate_coder,
    update_aider_state_from_coder,
    get_active_coder_state,
    remove_active_coder_state,
    ensure_active_coder_state,
    create_active_coder_state,
    AiderIOStubWithQueues,
    _run_aider_in_thread,
    SIGNAL_PROMPT_NEEDED,
    TIMEOUT
)
from .prompts import AIDER_TOOLSET_PROMPT
from ...errors import PromptNeededError
from .texts import get_text

# Get console manager instance
console = get_console_manager()

# Import Aider repo-related classes conditionally
try:
    from aider.repo import ANY_GIT_ERROR
except ImportError:
    ANY_GIT_ERROR = Exception

# Define edit formats supported by Aider
AIDER_EDIT_FORMATS = {
    "diff": "Traditional diff format",
    "edit_chunks": "Edit chunks format (easier to understand)",
    "whole_files": "Complete file replacements",
}

# --- Toolset metadata for discovery ---
toolset_name = get_text("toolset.name")
toolset_id = "aider"
toolset_description = get_text("toolset.description")

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
    final_config["enabled"] = True

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
    api_key_descriptions = {
        "OPENAI_API_KEY": "OpenAI API Key (https://platform.openai.com/api-keys)",
        "GOOGLE_API_KEY": "Google AI API Key (https://aistudio.google.com/app/apikey)"
    }

    for model_name in filter(None, models_to_check): # Filter out None values
        try:
             provider = get_model_provider(model_name)
             env_var = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"

             if env_var not in checked_providers:
                 logger.debug(f"Ensuring dotenv key: {env_var} for model {model_name}")
                 key_value = ensure_dotenv_key(
                     dotenv_path, 
                     env_var,
                     api_key_descriptions.get(env_var)
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
        logger.info(f"File Editor configuration saved to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to save File Editor config to global path {global_config_path}: {e}")

    if not is_global_only and local_config_path: # Check local_config_path exists
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.info(f"File Editor configuration saved to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to save File Editor config to local path {local_config_path}: {e}")

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

# --- Tool Input Schemas ---
class NoArgsSchema(BaseModel):
    """Input schema for tools that require no arguments."""
    pass

class FilePathSchema(BaseModel):
    """Input schema for tools accepting a file path."""
    file_path: str = Field(description=get_text("schemas.file_path.path_desc"))

class InstructionSchema(BaseModel):
    """Input schema for tools accepting a natural language instruction."""
    instruction: str = Field(description=get_text("schemas.instruction.instruction_desc"))

class UserResponseSchema(BaseModel):
    """Input schema for tools accepting a user response to a prompt."""
    user_response: str = Field(description=get_text("schemas.user_response.response_desc"))


# --- Tool Classes (MODIFIED) ---

class AiderUsageGuideTool(BaseTool):
    name: str = get_text("tools.usage_guide.name")
    description: str = get_text("tools.usage_guide.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Returns the usage instructions for the Aider toolset."""
        logger.debug(f"AiderUsageGuideTool invoked.")
        return AIDER_TOOLSET_PROMPT

    async def _arun(self) -> str:
        return await run_in_executor(None, self._run)


class AddFileToConext(BaseTool):
    name: str = get_text("tools.add_file.name")
    description: str = get_text("tools.add_file.description")
    args_schema: Type[BaseModel] = FilePathSchema
    
    def _run(self, file_path: str) -> str:
        """Adds a file to the Aider context, recreating state if needed."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.add_file.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.add_file.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub

        try:
            abs_path_to_add = str(Path(file_path).resolve())

            if not os.path.exists(abs_path_to_add):
                return get_text("tools.add_file.error.not_exists", path=file_path, abs_path=abs_path_to_add)
            if coder.repo and coder.root != os.getcwd():
                try:
                    if hasattr(Path, 'is_relative_to'):
                        if not Path(abs_path_to_add).is_relative_to(Path(coder.root)):
                             return get_text("tools.add_file.error.outside_root", path=file_path, root=coder.root)
                    else:
                         rel_path_check = os.path.relpath(abs_path_to_add, coder.root)
                         if rel_path_check.startswith('..'):
                              return get_text("tools.add_file.error.outside_root", path=file_path, root=coder.root)
                except ValueError:
                     return get_text("tools.add_file.error.different_drive", path=file_path, root=coder.root)

            rel_path = coder.get_rel_fname(abs_path_to_add)
            coder.add_rel_fname(rel_path)

            update_aider_state_from_coder(aider_json_path, coder)
            logger.info(f"Added file {rel_path} and updated persistent state for {chat_id}")

            if abs_path_to_add in coder.abs_fnames:
                return get_text("tools.add_file.success", path=rel_path, output=io_stub.get_captured_output())
            else:
                logger.error(f"File {abs_path_to_add} not found in coder.abs_fnames after adding.")
                return get_text("tools.add_file.warn_confirm_failed", path=rel_path)

        except Exception as e:
            logger.error(f"Error in AddFileTool: {e}", exc_info=True)
            return get_text("tools.add_file.error.generic", path=file_path, error=e, output=io_stub.get_captured_output())
    
    async def _arun(self, file_path: str) -> str:
        return await run_in_executor(None, self._run, file_path)


class RemoveFileFromContext(BaseTool):
    name: str = get_text("tools.remove_file.name")
    description: str = get_text("tools.remove_file.description")
    args_schema: Type[BaseModel] = FilePathSchema
    
    def _run(self, file_path: str) -> str:
        """Removes a file from the Aider context."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.remove_file.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.remove_file.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub
            
        try:
            abs_path_to_drop = str(Path(file_path).resolve())
            rel_path_to_drop = coder.get_rel_fname(abs_path_to_drop)
            
            success = coder.drop_rel_fname(rel_path_to_drop)
            
            if success:
                update_aider_state_from_coder(aider_json_path, coder)
                return get_text("tools.remove_file.success", path=file_path, output=io_stub.get_captured_output())
            else:
                return get_text("tools.remove_file.info_not_found", path=file_path, output=io_stub.get_captured_output())
                
        except Exception as e:
            logger.error(f"Error in DropFileTool: {e}", exc_info=True)
            return get_text("tools.remove_file.error.generic", path=file_path, error=e, output=io_stub.get_captured_output())
            
    async def _arun(self, file_path: str) -> str:
        return await run_in_executor(None, self._run, file_path)


class ListFilesInContext(BaseTool):
    name: str = get_text("tools.list_files.name")
    description: str = get_text("tools.list_files.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    
    def _run(self) -> str:
        """Lists all files in the Aider context."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.list_files.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        aider_state = read_json(aider_json_path, default_value=None)
        if not aider_state or not aider_state.get("enabled", False):
            return get_text("tools.list_files.error.init_failed")
            
        try:
            files = aider_state.get("abs_fnames", [])
            if not files:
                return get_text("tools.list_files.info_empty")
                
            root = aider_state.get("git_root", os.getcwd())
            files_list = []
            for f in files:
                try:
                    rel_path = os.path.relpath(f, root)
                    files_list.append(rel_path)
                except ValueError:
                    files_list.append(f)
                    
            return get_text("tools.list_files.success", file_list="\n".join(f"- {f}" for f in sorted(files_list)))
            
        except Exception as e:
            logger.error(f"Error in ListFilesInEditorTool: {e}", exc_info=True)
            return get_text("tools.list_files.error.generic", error=e)
            
    async def _arun(self) -> str:
        return await run_in_executor(None, self._run)


class RequestEdits(BaseTool):
    name: str = get_text("tools.request_edit.name")
    description: str = get_text("tools.request_edit.description")
    args_schema: Type[BaseModel] = InstructionSchema
    
    def _run(self, instruction: str) -> str:
        """Runs Aider's main edit loop in a background thread."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.request_edit.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.request_edit.error.init_failed")

        if not state.coder.abs_fnames:
             return get_text("tools.request_edit.error.no_files")

        with state.lock:
            if state.thread and state.thread.is_alive():
                logger.warning(f"An edit is already in progress for {chat_id}. Please wait or submit input if needed.")
                return get_text("tools.request_edit.error.in_progress")

            if state.coder.io is not state.io_stub:
                 logger.warning("Correcting coder IO instance mismatch.")
                 state.coder.io = state.io_stub

            logger.info(f"Starting Aider worker thread for: {chat_id}")
            state.thread = threading.Thread(
                target=_run_aider_in_thread,
                args=(state.coder, instruction, state.output_q),
                daemon=True, 
                name=f"AiderWorker-{chat_id[:8]}"
            )
            state.thread.start()
            
            update_aider_state_from_coder(aider_json_path, state.coder)

        logger.debug(f"Main thread waiting for initial message from output_q for {chat_id}...")
        try:
             message = state.output_q.get(timeout=TIMEOUT)
             logger.debug(f"Main thread received initial message: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout waiting for initial Aider response ({chat_id}).")
             remove_active_coder_state(aider_json_path)
             return get_text("tools.request_edit.error.timeout")
        except Exception as e:
              logger.error(f"Exception waiting on output_q for {chat_id}: {e}")
              remove_active_coder_state(aider_json_path)
              return get_text("tools.request_edit.error.queue_exception", error=e)


        message_type = message.get('type')

        if message_type == 'prompt':
            prompt_data = message
            question = prompt_data.get('question', 'Input needed')
            subject = prompt_data.get('subject')
            default = prompt_data.get('default')
            allow_never = prompt_data.get('allow_never')

            response_guidance = get_text("tools.request_edit.prompt_guidance", prompt=question)
            if subject: response_guidance += get_text("tools.request_edit.prompt_subject", subject=subject[:100]+'...' if len(subject)>100 else subject)
            if default: response_guidance += get_text("tools.request_edit.prompt_default", default=default)
            if prompt_data.get('prompt_type', 'unknown') == 'confirm':
                 options = "(yes/no"
                 if prompt_data.get('group_id'): options += "/all/skip"
                 if allow_never: options += "/don't ask"
                 options += ")"
                 response_guidance += get_text("tools.request_edit.prompt_confirm_options", options=options)

            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)
                
            return get_text("tools.request_edit.prompt_needed", signal=SIGNAL_PROMPT_NEEDED, guidance=response_guidance)

        elif message_type == 'result':
            logger.info(f"Aider edit completed successfully for {chat_id}.")
            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)
                state.thread = None
            return get_text("tools.request_edit.success", output=message.get('content', 'No output captured.'))

        elif message_type == 'error':
            logger.error(f"Aider edit failed for {chat_id}.")
            error_content = message.get('message', 'Unknown error')
            
            try:
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
            except Exception:
                pass
            
            remove_active_coder_state(aider_json_path)
            return get_text("tools.request_edit.error.edit_failed", content=error_content)
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             remove_active_coder_state(aider_json_path)
             return get_text("tools.request_edit.error.unknown_response", type=message_type)
    
    async def _arun(self, instruction: str) -> str:
        return await run_in_executor(None, self._run, instruction)


class ViewDiffs(BaseTool):
    name: str = get_text("tools.view_diffs.name")
    description: str = get_text("tools.view_diffs.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    
    def _run(self) -> str:
        """Shows the diff of changes made by Aider."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.view_diffs.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.view_diffs.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub
            
        try:
            if not coder.repo:
                return get_text("tools.view_diffs.error.no_repo")
                
            if hasattr(coder, 'commands') and coder.commands:
                coder.commands.raw_cmd_diff("")
                captured = io_stub.get_captured_output()
                return get_text("tools.view_diffs.success", diff=captured) if captured else get_text("tools.view_diffs.info_no_changes")
            else:
                diff = coder.repo.get_unstaged_changes()
                
                if not diff:
                    return get_text("tools.view_diffs.info_no_changes")
                    
                return get_text("tools.view_diffs.success", diff=diff)
            
        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during diff: {e}")
            return get_text("tools.view_diffs.error.git_error", error=e, output=io_stub.get_captured_output()).strip()
            
        except Exception as e:
            logger.error(f"Error in ViewDiffTool: {e}")
            return get_text("tools.view_diffs.error.generic", error=e, output=io_stub.get_captured_output())
            
    async def _arun(self) -> str:
        return await run_in_executor(None, self._run())

class UndoLastEditTool(BaseTool):
    name: str = get_text("tools.undo_edit.name")
    description: str = get_text("tools.undo_edit.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    
    def _run(self) -> str:
        """Undoes the last edit commit made by Aider."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.undo_edit.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.undo_edit.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub
            
        try:
            if not coder.repo:
                return get_text("tools.undo_edit.error.no_repo")
                
            try:
                from aider.commands import Commands
                if not hasattr(coder, "commands"):
                    coder.commands = Commands(io=io_stub, coder=coder)
            except ImportError:
                return get_text("tools.undo_edit.error.cmd_module_missing")

            coder.commands.raw_cmd_undo(None)
            update_aider_state_from_coder(aider_json_path, coder)
            
            return get_text("tools.undo_edit.success", output=io_stub.get_captured_output()).strip()
            
        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during undo: {e}")
            return get_text("tools.undo_edit.error.git_error", error=e, output=io_stub.get_captured_output()).strip()
            
        except Exception as e:
            logger.error(f"Unexpected error during undo: {e}")
            return get_text("tools.undo_edit.error.generic", error=e, output=io_stub.get_captured_output()).strip()
            
    async def _arun(self) -> str:
        return await run_in_executor(None, self._run())

class CloseCodeCopilot(BaseTool):
    name: str = get_text("tools.close_editor.name")
    description: str = get_text("tools.close_editor.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Clears the Aider state and cleans up runtime state."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.close_editor.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        state_cleared_msg = ""
        try:
            remove_active_coder_state(aider_json_path)
            logger.info(f"Aider state cleared for {chat_id}")
            state_cleared_msg = get_text("tools.close_editor.info_success")
        except Exception as e:
            logger.error(f"Error clearing Aider state: {e}", exc_info=True)
            state_cleared_msg = get_text("tools.close_editor.warn_error")

        return state_cleared_msg.strip()
             
    async def _arun(self) -> str:
        return await run_in_executor(None, self._run())


class SubmitInput_HITL(BaseTool):
    name: str = get_text("tools.submit_input.name")
    description: str = get_text("tools.submit_input.description")
    args_schema: Type[BaseModel] = UserResponseSchema
    requires_confirmation: bool = True

    def _run(self, user_response: str, confirmed_input: Optional[str] = None) -> str:
        """Handles submitting input to Aider using the PromptNeededError approach."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.submit_input.error.no_chat")

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.submit_input.error.init_failed")

        if confirmed_input is None:
            logger.debug(f"SubmitInputTool: Raising PromptNeededError")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"user_response": user_response},
                edit_key="user_response",
                prompt_suffix=get_text("tools.submit_input.prompt_suffix")
            )

        edited_response = confirmed_input
        
        if not edited_response.strip():
             return get_text("tools.submit_input.error.empty_confirmed")

        processed_result = ""
        
        with state.lock:
            if not state.thread or not state.thread.is_alive():
                remove_active_coder_state(aider_json_path)
                return get_text("tools.submit_input.error.not_waiting")
            
            logger.debug(f"Putting confirmed response on input_q: '{edited_response[:50]}...' for {chat_id}")
            state.input_q.put(edited_response)
        
        logger.debug(f"Main thread waiting for response from output_q for {chat_id}...")
        try:
            message = state.output_q.get(timeout=TIMEOUT)
            logger.debug(f"Main thread received message from output_q: {message.get('type')}")
            
            message_type = message.get('type')
            if message_type == 'prompt':
                prompt_data = message
                question = prompt_data.get('question', 'Input needed')
                subject = prompt_data.get('subject')
                default = prompt_data.get('default')
                
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
                
                prompt_msg = question
                if subject:
                    prompt_msg += get_text("tools.submit_input.prompt_subject", subject=subject[:50]+'...' if len(subject) > 50 else subject)
                
                new_args = {"user_response": default or ""}
                
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args=new_args,
                    edit_key="user_response",
                    prompt_suffix=get_text("tools.submit_input.prompt_hitl_suffix", prompt=prompt_msg)
                )
            
            elif message_type == 'result':
                logger.info(f"Aider edit completed successfully for {chat_id} after input.")
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
                    state.thread = None
                processed_result = get_text("tools.submit_input.success", output=message.get('content', 'No output captured.'))
            
            elif message_type == 'error':
                logger.error(f"Aider edit failed for {chat_id} after input.")
                error_content = message.get('message', 'Unknown error')
                try:
                    with state.lock:
                        update_aider_state_from_coder(aider_json_path, state.coder)
                except Exception:
                    pass
                remove_active_coder_state(aider_json_path)
                processed_result = get_text("tools.submit_input.error.edit_failed", content=error_content)
            
            else:
                logger.error(f"Unknown message type from Aider thread: {message_type}")
                try:
                    with state.lock:
                        update_aider_state_from_coder(aider_json_path, state.coder)
                except Exception:
                    pass
                remove_active_coder_state(aider_json_path)
                processed_result = get_text("tools.submit_input.error.unknown_response", type=message_type)
                
        except queue.Empty:
            logger.error(f"Timeout waiting for Aider response ({chat_id}).")
            remove_active_coder_state(aider_json_path)
            processed_result = get_text("tools.submit_input.error.timeout")
        except PromptNeededError:
            raise
        except Exception as e:
            logger.error(f"Exception waiting on output_q for {chat_id}: {e}", exc_info=True)
            remove_active_coder_state(aider_json_path)
            processed_result = get_text("tools.submit_input.error.queue_exception", error=e)
            
        return processed_result
        
    async def _arun(self, user_response: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, user_response, confirmed_input)

# --- Instantiate Tools ---
aider_usage_guide_tool = AiderUsageGuideTool()
add_code_file_tool = AddFileToConext()
drop_code_file_tool = RemoveFileFromContext()
list_code_files_tool = ListFilesInContext()
edit_code_tool = RequestEdits()
submit_code_editor_input_tool = SubmitInput_HITL()
view_diff_tool = ViewDiffs()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseCodeCopilot()

# --- Define Toolset Structure ---
toolset_tools: List[BaseTool] = [
    aider_usage_guide_tool, add_code_file_tool, drop_code_file_tool,
    list_code_files_tool, edit_code_tool, submit_code_editor_input_tool,
    view_diff_tool, undo_last_edit_tool, close_code_editor_tool
]

# --- Register Tools ---
register_tools(toolset_tools)
logger.debug(f"Registered File Editor toolset with {len(toolset_tools)} tools")