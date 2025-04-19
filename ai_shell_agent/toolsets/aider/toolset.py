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
from typing import Dict, List, Optional, Any, Type # Add Type

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor  # Add import for run_in_executor
from prompt_toolkit import prompt as prompt_toolkit_prompt
from pydantic import BaseModel, Field # Import BaseModel and Field

# Local imports
from ... import logger
from ...tool_registry import register_tools
from ...chat_state_manager import (
    get_current_chat,
    get_toolset_data_path
    # Removed: _read_json, _write_json imports - use utils
    # Removed: get_active_toolsets, update_active_toolsets
)
# Import config manager helpers needed ONLY for model/provider info
from ...config_manager import (
    get_current_model as get_agent_model, # Rename to avoid confusion
    get_api_key_for_model, # For checking if keys EXIST (read-only)
    get_model_provider, # For model provider lookup
    # Removed: set_api_key_for_model
    normalize_model_name, # Keep for model name normalization
    ALL_MODELS # Keep models dictionary
)
# --- NEW: Import .env utilities ---
from ...utils import ensure_dotenv_key, read_json as _read_json, write_json as _write_json
# Remove console_io import
# Import console manager
from ...console_manager import get_console_manager

# Get console manager instance
console = get_console_manager()

# Import integration module for aider features
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
# Import the prompt content to be returned by the start tool
from .prompts import AIDER_TOOLSET_PROMPT

# Import Aider repo-related classes conditionally
try:
    from aider.repo import ANY_GIT_ERROR
except ImportError:
    # Define a fallback that will catch any exception when used with isinstance()
    ANY_GIT_ERROR = Exception

# Define edit formats supported by Aider
AIDER_EDIT_FORMATS = {
    "diff": "Traditional diff format",
    "edit_chunks": "Edit chunks format (easier to understand)",
    "whole_files": "Complete file replacements",
}

# --- Toolset metadata for discovery ---
toolset_name = "Aider Code Editor **EXPERIMENTAL**"
toolset_id = "aider" # Explicitly define ID for consistency
toolset_description = "Provides tools for interacting with the Aider code editor. "

# --- Configuration ---
DEFAULT_EDITOR_MODEL = "o3-mini" # Example default
DEFAULT_WEAK_MODEL = "gpt-4o-mini" # Example default

toolset_config_defaults = {
    "main_model": None, # Default to agent's main model
    "editor_model": DEFAULT_EDITOR_MODEL,
    "weak_model": DEFAULT_WEAK_MODEL,
    "edit_format": None, # Default to Aider's model-specific default
    # Add other potential Aider config keys if needed, e.g., auto_commits
    "auto_commits": True,
    "dirty_commits": True,
}

def _prompt_for_single_model_config(role_name: str, current_value: Optional[str], default_value: Optional[str]) -> Optional[str]:
    """Helper to prompt for one of the coder models within configure_toolset."""
    console.display_message("SYSTEM:", f"\n--- Select File Editor '{role_name}' Model ---", 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    console.display_message("SYSTEM:", "Available models:", 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    
    all_model_names = sorted(list(set(ALL_MODELS.values())))
    effective_current = current_value if current_value is not None else default_value
    
    for model in all_model_names:
        marker = " <- Current Setting" if model == effective_current else ""
        console.display_message("SYSTEM:", f"- {model}{marker}", 
                              console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        if aliases: 
            console.display_message("SYSTEM:", f"  (aliases: {', '.join(aliases)})", 
                                  console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    prompt_msg = (f"Enter model name for '{role_name}'"
                  f" (leave empty to keep '{effective_current or 'Default'}',"
                  f" enter 'none' to reset to default)")

    while True:
        try:
            selected = console.prompt_for_input(prompt_msg).strip()
            if not selected:
                console.display_message("INFO:", f"Keeping current setting: {effective_current or 'Default'}", 
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                # Return the value that *was* current (could be None)
                return current_value
            elif selected.lower() == 'none':
                console.display_message("INFO:", f"Resetting {role_name} model to default ('{default_value or 'Agent Default' if role_name == 'Main' else 'Aider Default'}')", 
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return None # Use None to signify reset/default
            else:
                normalized_model = normalize_model_name(selected)
                if normalized_model in all_model_names:
                    return normalized_model # Return the selected normalized name
                else:
                    console.display_message("ERROR:", f"Unknown model '{selected}'. Please choose from the list or enter 'none'", 
                                          console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        except KeyboardInterrupt:
            console.display_message("WARNING:", "Selection cancelled. Keeping current setting.", 
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return current_value

def _prompt_for_edit_format_config(current_value: Optional[str]) -> Optional[str]:
    """Prompts the user to select an Aider edit format using ConsoleManager."""
    console.display_message("SYSTEM:", "\n--- Select File Editor Edit Format ---", 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    
    i = 0
    valid_choices = {}
    default_display = "Aider Default (model-specific)"
    
    console.display_message("SYSTEM:", f"  0: {default_display} {'<- Current Setting' if current_value is None else ''}", 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    valid_choices['0'] = None

    format_list = sorted(AIDER_EDIT_FORMATS.keys())
    for idx, fmt in enumerate(format_list, 1):
        description = AIDER_EDIT_FORMATS[fmt]
        marker = " <- Current Setting" if fmt == current_value else ""
        console.display_message("SYSTEM:", f"  {idx}: {fmt}{marker} - {description}", 
                              console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        valid_choices[str(idx)] = fmt

    while True:
        try:
            # Use ConsoleManager for input
            choice = console.prompt_for_input(f"Enter choice (0-{len(format_list)}), leave empty to keep current: ").strip()
            
            if not choice:
                console.display_message("INFO:", f"Keeping current setting: {current_value or default_display}", 
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return current_value
            elif choice in valid_choices:
                selected_format = valid_choices[choice]
                console.display_message("INFO:", f"Selected format: {selected_format or default_display}", 
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return selected_format
            else:
                console.display_message("ERROR:", "Invalid choice. Please try again.", 
                                      console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        except KeyboardInterrupt:
            console.display_message("WARNING:", "Selection cancelled. Keeping current setting.", 
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return current_value

# Import utils for JSON I/O
from ...utils import read_json as _read_json, write_json as _write_json

# --- MODIFIED: configure_toolset signature and saving logic ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path], # Make local_config_path optional
    dotenv_path: Path,
    current_config_for_prompting: Optional[Dict] # Renamed for clarity
) -> Dict:
    """
    Configuration function for the File Editor (Aider) toolset.
    Prompts user based on current config (local or global), ensures secrets via .env utils,
    and saves the result appropriately (global-only or both).
    """
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat"
    logger.info(f"Configuring File Editor toolset ({context_name}). Global: {global_config_path}, Local: {local_config_path}, .env: {dotenv_path}")

    # Use current_config_for_prompting (read from local or global before calling)
    config_to_prompt = current_config_for_prompting or {}
    defaults = toolset_config_defaults
    final_config = {} # Build fresh

    print(f"\n--- Configure File Editor (Aider) Settings ({context_name}) ---")
    print("Select the models and edit format the File Editor should use.")
    if is_global_only:
        print("This configuration will be saved as the global default for new chats.")
    else:
        print("This configuration will be saved for the current chat AND as the global default for new chats.")
    print("Leave input empty at any step to keep the current setting.")
    print("Enter 'none' to reset a setting to its default value.")

    # --- Model Selection ---
    agent_model = get_agent_model() # Get the main agent model to use as default for 'main'

    # Main Model
    selected_main = _prompt_for_single_model_config(
        "Main/Architect",
        config_to_prompt.get("main_model"), # Use config_to_prompt for current value
        defaults["main_model"] # Default is None initially, _prompt.. handles agent fallback
    )
    final_config["main_model"] = selected_main

    # Editor Model
    selected_editor = _prompt_for_single_model_config(
        "Editor",
        config_to_prompt.get("editor_model"), # Use config_to_prompt for current value
        defaults["editor_model"]
    )
    final_config["editor_model"] = selected_editor

    # Weak Model
    selected_weak = _prompt_for_single_model_config(
        "Weak (Commits etc.)",
        config_to_prompt.get("weak_model"), # Use config_to_prompt for current value
        defaults["weak_model"]
    )
    final_config["weak_model"] = selected_weak

    # --- Edit Format Selection ---
    selected_format = _prompt_for_edit_format_config(config_to_prompt.get("edit_format")) # Use config_to_prompt
    final_config["edit_format"] = selected_format

    # Add other default config values if they aren't model/format related
    final_config["auto_commits"] = config_to_prompt.get("auto_commits", defaults["auto_commits"])
    final_config["dirty_commits"] = config_to_prompt.get("dirty_commits", defaults["dirty_commits"])
    final_config["enabled"] = True # Mark as enabled when configured

    # --- Ensure API Keys using ensure_dotenv_key ---
    print("\nChecking API keys for selected File Editor models...")
    # Determine the actual models being used (considering defaults)
    actual_main_model = final_config.get("main_model") or agent_model
    actual_editor_model = final_config.get("editor_model") or defaults["editor_model"]
    actual_weak_model = final_config.get("weak_model") or defaults["weak_model"]

    models_to_check = {
        actual_main_model,
        actual_editor_model,
        actual_weak_model
    }

    checked_providers = set()
    required_keys_ok = True
    # Define descriptions for keys
    api_key_descriptions = {
        "OPENAI_API_KEY": "OpenAI API Key (https://platform.openai.com/api-keys)",
        "GOOGLE_API_KEY": "Google AI API Key (https://aistudio.google.com/app/apikey)"
    }

    for model_name in filter(None, models_to_check): # Filter out potential None values
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
                     required_keys_ok = False
                 checked_providers.add(env_var)
        except Exception as e:
            logger.error(f"Error checking API key for model '{model_name}': {e}", exc_info=True)
            print(f"\nError checking API key for model '{model_name}'. Check logs.")
            required_keys_ok = False # Mark as problematic if check fails

    # --- Save Config Appropriately ---
    save_success_global = True
    save_success_local = True

    # Always save to global
    try:
        _write_json(global_config_path, final_config)
        logger.info(f"File Editor configuration saved to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to save File Editor config to global path {global_config_path}: {e}")
         print(f"\nError: Failed to save global default configuration: {e}")

    # Save to local only if local_config_path is provided (i.e., not global-only context)
    if not is_global_only:
        try:
            _write_json(local_config_path, final_config)
            logger.info(f"File Editor configuration saved to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to save File Editor config to local path {local_config_path}: {e}")
             print(f"\nError: Failed to save configuration for current chat: {e}")

    # --- Confirmation Messages ---
    if save_success_global and (is_global_only or save_success_local):
        if is_global_only: print("\nGlobal default File Editor configuration updated.")
        else: print("\nFile Editor configuration updated for this chat and globally.")
        if not required_keys_ok: print("Note: Some API keys missing/skipped. Ensure they are set in .env.")
    else:
        print("\nFile Editor configuration update failed. Check logs.")

    # Return the final config dict
    return final_config

# --- Tool Input Schemas ---
class NoArgsSchema(BaseModel):
    """Input schema for tools that require no arguments."""
    pass

class FilePathSchema(BaseModel):
    """Input schema for tools accepting a file path."""
    file_path: str = Field(description="The absolute or relative path to the file.")

class InstructionSchema(BaseModel):
    """Input schema for tools accepting a natural language instruction."""
    instruction: str = Field(description="The natural language instruction for the edit request.")

class UserResponseSchema(BaseModel):
    """Input schema for tools accepting a user response to a prompt."""
    user_response: str = Field(description="Your response to the editor's prompt.")


# --- Tool Classes (MODIFIED) ---

class AiderUsageGuideTool(BaseTool):
    name: str = "aider_usage_guide"
    description: str = "Displays usage instructions and context for the Aider Code Copilot toolset."
    args_schema: Type[BaseModel] = NoArgsSchema # Specify schema

    def _run(self) -> str:
        """Returns the usage instructions for the Aider toolset."""
        logger.debug(f"AiderUsageGuideTool invoked.")
        # Simply return the static prompt content
        return AIDER_TOOLSET_PROMPT

    async def _arun(self) -> str:
        return self._run()


class AddFileToConext(BaseTool):
    name: str = "add_file_to_copilot_context"
    description: str = "Adds a file to the Code Copilot context, files in context can be used to understand the codebase and perform edits. Files to be added can be specified by absolute or relative paths. The file must exist in the filesystem."
    args_schema: Type[BaseModel] = FilePathSchema # Specify schema
    
    def _run(self, file_path: str) -> str:
        """Adds a file to the Aider context, recreating state if needed."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        # Ensure active state exists
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            # If state is None after trying to recreate, it means editor isn't properly initialized
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."

        # Use the coder and IO stub from the active state
        coder = state.coder
        io_stub = state.io_stub

        try:
            abs_path_to_add = str(Path(file_path).resolve())

            # *** Checks (remain the same) ***
            if not os.path.exists(abs_path_to_add):
                return f"Error: File '{file_path}' (resolved to '{abs_path_to_add}') does not exist."
            if coder.repo and coder.root != os.getcwd():
                # ... (git root checks remain the same) ...
                try:
                    if hasattr(Path, 'is_relative_to'):
                        if not Path(abs_path_to_add).is_relative_to(Path(coder.root)):
                             return f"Error: Cannot add file '{file_path}' outside project root '{coder.root}'."
                    else:
                         rel_path_check = os.path.relpath(abs_path_to_add, coder.root)
                         if rel_path_check.startswith('..'):
                              return f"Error: Cannot add file '{file_path}' outside project root '{coder.root}'."
                except ValueError:
                     return f"Error: Cannot add file '{file_path}' on a different drive than project root '{coder.root}'."
            # *** End Checks ***

            rel_path = coder.get_rel_fname(abs_path_to_add)
            coder.add_rel_fname(rel_path)  # This modifies coder.abs_fnames internally

            # Update persistent state to reflect the added file
            update_aider_state_from_coder(aider_json_path, coder)
            logger.info(f"Added file {rel_path} and updated persistent state for {chat_id}")

            # Simple verification based on coder's internal state
            if abs_path_to_add in coder.abs_fnames:
                return f"Successfully added {rel_path}. {io_stub.get_captured_output()}"
            else:
                logger.error(f"File {abs_path_to_add} not found in coder.abs_fnames after adding.")
                return f"Warning: Failed to confirm {rel_path} was added successfully."

        except Exception as e:
            logger.error(f"Error in AddFileTool: {e}")
            logger.error(traceback.format_exc())
            # Pass io_stub output along with error
            return f"Error adding file {file_path}: {e}. {io_stub.get_captured_output()}"
    
    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


class RemoveFileFromContext(BaseTool):
    name: str = "remove_file_from_copilot_context"
    description: str = "Removes a file from the Code Copilot context. Files to be removed can be specified by absolute or relative paths. The file must have been added to the context previously."
    args_schema: Type[BaseModel] = FilePathSchema # Specify schema
    
    def _run(self, file_path: str) -> str:
        """Removes a file from the Aider context."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
            
        # Ensure active state exists
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."

        coder = state.coder
        io_stub = state.io_stub
            
        try:
            # Coder's drop_rel_fname expects relative path
            abs_path_to_drop = str(Path(file_path).resolve())
            rel_path_to_drop = coder.get_rel_fname(abs_path_to_drop)
            
            success = coder.drop_rel_fname(rel_path_to_drop)
            
            if success:
                # Update state after dropping the file
                update_aider_state_from_coder(aider_json_path, coder)
                return f"Successfully dropped {file_path}. {io_stub.get_captured_output()}"
            else:
                return f"File {file_path} not found in context. {io_stub.get_captured_output()}"
                
        except Exception as e:
            logger.error(f"Error in DropFileTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error dropping file {file_path}: {e}. {io_stub.get_captured_output()}"
            
    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


class ListFilesInContext(BaseTool):
    name: str = "list_files_in_copilot_context"
    description: str = "Lists all files in the Code Copilot context. This is useful to see what files are currently being edited or added for context."
    args_schema: Type[BaseModel] = NoArgsSchema # Specify schema
    
    def _run(self) -> str:
        """Lists all files in the Aider context."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat  
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        # Read the current state directly
        aider_state = _read_json(aider_json_path, default_value=None)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Editor not initialized. Use start_file_editor first."
            
        try:
            files = aider_state.get("abs_fnames", [])
            if not files:
                return "No files are currently added to the editing session."
                
            # Get the common root to show relative paths if possible
            root = aider_state.get("git_root", os.getcwd())
            
            files_list = []
            for f in files:
                try:
                    rel_path = os.path.relpath(f, root)
                    files_list.append(rel_path)
                except ValueError:
                    # If files are on different drives
                    files_list.append(f)
                    
            return "Files in editor:\n" + "\n".join(f"- {f}" for f in sorted(files_list))
            
        except Exception as e:
            logger.error(f"Error in ListFilesInEditorTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error listing files: {e}"
            
    async def _arun(self) -> str:
        return self._run()


class RequestEdits(BaseTool):
    name: str = "request_copilot_edit"
    description: str = "Explain to the copilot what needs to be done. The editor will complete the request based on own knowledge and submitted files. Use clear instructions using natural language, you can include specific code snippets."
    args_schema: Type[BaseModel] = InstructionSchema # Specify schema
    
    def _run(self, instruction: str) -> str:
        """Runs Aider's main edit loop in a background thread."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        # Ensure active state exists
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            # If state is None after trying to recreate, it means editor isn't properly initialized
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."

        if not state.coder.abs_fnames:
             return "Error: No files have been added to the editing session. Use include_file first."

        # --- Threading Logic ---
        # Acquire lock for this session before starting thread
        with state.lock:
            # Check if a thread is already running for this session
            if state.thread and state.thread.is_alive():
                logger.warning(f"An edit is already in progress for {chat_id}. Please wait or submit input if needed.")
                return "Error: An edit is already in progress for this session."

            # Ensure the coder's IO is the correct stub instance from the state
            if state.coder.io is not state.io_stub:
                 logger.warning("Correcting coder IO instance mismatch.")
                 state.coder.io = state.io_stub

            # Start the background thread
            logger.info(f"Starting Aider worker thread for: {chat_id}")
            state.thread = threading.Thread(
                target=_run_aider_in_thread,
                args=(state.coder, instruction, state.output_q),
                daemon=True, 
                name=f"AiderWorker-{chat_id[:8]}"
            )
            state.thread.start()
            
            # Update state before waiting - makes sure we have the latest before any edits
            update_aider_state_from_coder(aider_json_path, state.coder)
        # --- End Threading Logic ---

        # Release lock before waiting on queue
        # Wait for the *first* response from the Aider thread
        logger.debug(f"Main thread waiting for initial message from output_q for {chat_id}...")
        try:
             message = state.output_q.get(timeout=TIMEOUT)  # Add a timeout (e.g., 5 minutes)
             logger.debug(f"Main thread received initial message: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout waiting for initial Aider response ({chat_id}).")
             remove_active_coder_state(aider_json_path)  # Clean up state
             return "Error: Timed out waiting for Aider response."
        except Exception as e:
              logger.error(f"Exception waiting on output_q for {chat_id}: {e}")
              remove_active_coder_state(aider_json_path)
              return f"Error: Exception while waiting for Aider: {e}"


        # Process the *first* message received
        message_type = message.get('type')

        if message_type == 'prompt':
            # Aider needs input immediately
            prompt_data = message
            prompt_type = prompt_data.get('prompt_type', 'unknown')
            question = prompt_data.get('question', 'Input needed')
            subject = prompt_data.get('subject')
            default = prompt_data.get('default')
            allow_never = prompt_data.get('allow_never')

            response_guidance = f"Aider requires input. Please respond using 'submit_editor_input'. Prompt: '{question}'"
            if subject: response_guidance += f" (Regarding: {subject[:100]}{'...' if len(subject)>100 else ''})"
            if default: response_guidance += f" [Default: {default}]"
            if prompt_type == 'confirm':
                 options = "(yes/no"
                 if prompt_data.get('group_id'): options += "/all/skip"
                 if allow_never: options += "/don't ask"
                 options += ")"
                 response_guidance += f" Options: {options}"

            # Update state here too, in case the prompt interrupts mid-processing
            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)
                
            return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"

        elif message_type == 'result':
            # Aider finished without needing input
            logger.info(f"Aider edit completed successfully for {chat_id}.")
            with state.lock:  # Re-acquire lock briefly
                update_aider_state_from_coder(aider_json_path, state.coder)
                state.thread = None  # Clear the thread reference as it's done
            return f"Edit completed. {message.get('content', 'No output captured.')}"

        elif message_type == 'error':
            # Aider encountered an error immediately
            logger.error(f"Aider edit failed for {chat_id}.")
            error_content = message.get('message', 'Unknown error')
            
            # Even on error, update state if possible - might have partial changes
            try:
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
            except Exception:
                pass  # Ignore state update errors during cleanup
            
            remove_active_coder_state(aider_json_path)  # Clean up on error
            return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             remove_active_coder_state(aider_json_path)
             return f"Error: Unknown response type '{message_type}' from Aider process."
    
    async def _arun(self, instruction: str) -> str:
        # For simplicity in this stage, run synchronously.
        # Consider using asyncio.to_thread if true async is needed later.
        return self._run(instruction)


class ViewDiffs(BaseTool):
    name: str = "view_code_copilot_edit_diffs"
    description: str = "Shows the git diff of changes made by the Code Copilot based on 'request_edit' tool in the current session. This is useful to see what changes have been made to the files. Works only if there was a git repository initialized in the project root."
    args_schema: Type[BaseModel] = NoArgsSchema # Specify schema
    
    def _run(self) -> str:
        """Shows the diff of changes made by Aider."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
            
        # Ensure active state exists
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."

        coder = state.coder
        io_stub = state.io_stub
            
        try:
            # If there's no git repo, we can't show diffs
            if not coder.repo:
                return "Error: Git repository not available. Cannot show diff."
                
            # Use commands object if available
            if hasattr(coder, 'commands') and coder.commands:
                coder.commands.raw_cmd_diff("")  # Pass empty args for default diff behavior
                captured = io_stub.get_captured_output()
                return f"Diff:\n{captured}" if captured else "No changes detected in tracked files."
            else:
                # Fallback to direct repo method if commands not available
                diff = coder.repo.get_unstaged_changes()
                
                if not diff:
                    return "No changes detected in the tracked files."
                    
                return f"Changes in files:\n\n{diff}"
            
        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during diff: {e}")
            logger.error(traceback.format_exc())
            return f"Error viewing diff: {e}. {io_stub.get_captured_output()}".strip()
            
        except Exception as e:
            logger.error(f"Error in ViewDiffTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error viewing diff: {e}. {io_stub.get_captured_output()}"
            
    async def _arun(self) -> str:
        return self._run()

class UndoLastEditTool(BaseTool):
    name: str = "undo_last_code_copilot_edit"
    description: str = "Undoes the last edit commit made by the Code Copilot. This is useful to revert changes made to the files, might not work if the commit was made outside of the File Editor or if there is no git repository initialized."
    args_schema: Type[BaseModel] = NoArgsSchema # Specify schema
    
    def _run(self) -> str:
        """Undoes the last edit commit made by Aider."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
            
        # Ensure active state exists
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."

        coder = state.coder
        io_stub = state.io_stub
            
        try:
            # If there's no git repo, we can't undo commits
            if not coder.repo:
                return "Error: Cannot undo. Git repository not found or not configured."
                
            # Import commands module if needed
            try:
                from aider.commands import Commands
                if not hasattr(coder, "commands"):
                    coder.commands = Commands(io=io_stub, coder=coder)
            except ImportError:
                return "Error: Commands module not available in Aider."

            # Use the raw command to bypass interactive prompts from the standard command
            coder.commands.raw_cmd_undo(None)
            
            # Update the persistent state with potential changes to commit hashes
            update_aider_state_from_coder(aider_json_path, coder)
            
            return f"Undo attempt finished. {io_stub.get_captured_output()}".strip()
            
        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during undo: {e}")
            logger.error(traceback.format_exc())
            return f"Error during undo: {e}. {io_stub.get_captured_output()}".strip()
            
        except Exception as e:
            logger.error(f"Unexpected error during undo: {e}")
            logger.error(traceback.format_exc())
            return f"Error during undo: {e}. {io_stub.get_captured_output()}".strip()
            
    async def _arun(self) -> str:
        return self._run()

class CloseCodeCopilot(BaseTool):
    name: str = "close_code_copilot"
    description: str = "Closes the file editing session with AI Code Copilot. Changes are saved automatically. Close it after you verified with the user they don't want to edit the files anymore."
    args_schema: Type[BaseModel] = NoArgsSchema # Specify schema

    def _run(self) -> str:
        """Clears the Aider state and cleans up runtime state."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found to close the editor for."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        # --- Clear Aider Specific State ---
        state_cleared_msg = ""
        try:
            remove_active_coder_state(aider_json_path)  # This now handles both runtime and persistent state
            logger.info(f"Aider state cleared for {chat_id}")
            state_cleared_msg = "File Editor session context cleared."
        except Exception as e:
            logger.error(f"Error clearing Aider state: {e}", exc_info=True)
            state_cleared_msg = "File Editor session context possibly cleared (encountered error)."

        return state_cleared_msg.strip()
             
    async def _arun(self) -> str:
        # Simple enough to run synchronously
        return self._run()


class SubmitInput(BaseTool):
    name: str = "respond_to_code_copilot_input_request" # Keep the original intended name for LLM use
    description: str = (
         "Use to provide respond to Code Copilot only when requested (marked by '[CODE_COPILOT_INPUT_REQUEST]'). "
    )
    args_schema: Type[BaseModel] = UserResponseSchema # Specify schema
    is_hitl: bool = True # Mark this tool as requiring human-in-the-loop confirmation

    def _run(self, user_response: str) -> str:
        """Handles submitting input to Aider (user confirmation happens BEFORE this method is called)."""
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session found."

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state: return "Error: No active editor session found or state could not be restored."

        # The 'user_response' argument now contains the user-confirmed/edited response.
        edited_response = user_response # Use the argument directly

        # --- REMOVE HITL Prompt block ---
        # REMOVED: prompt_toolkit_prompt logic that was here before
        # --- END REMOVED BLOCK ---

        # Ensure the edited_response isn't empty after potential (now removed) prompt
        if not edited_response.strip():
             logger.warning("SubmitFileEditorInputTool._run called with empty response after potential HITL.")
             return "Error: Received empty response for submission."

        # Acquire lock for the specific session
        with state.lock:
             if not state.thread or not state.thread.is_alive():
                  remove_active_coder_state(aider_json_path)
                  return "Error: The editing process is not waiting for input."
             logger.debug(f"Putting user confirmed/edited response on input_q: '{edited_response}' for {chat_id}")
             state.input_q.put(edited_response)

        # Release lock before waiting on queue
        logger.debug(f"Main thread waiting for *next* message from output_q for {chat_id}...")
        try:
             message = state.output_q.get(timeout=TIMEOUT)
             logger.debug(f"Main thread received message from output_q: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout or queue error waiting for Aider response ({chat_id}).")
             remove_active_coder_state(aider_json_path)
             return "Error: Timed out waiting for Aider response after submitting input."
        except Exception as e:
             logger.error(f"Exception waiting on output_q for {chat_id}: {e}")
             remove_active_coder_state(aider_json_path)
             return f"Error: Exception while waiting for Aider after submitting input: {e}"

        # Process the message (logic remains the same)
        message_type = message.get('type')
        if message_type == 'prompt':
            prompt_data = message
            prompt_type = prompt_data.get('prompt_type', 'unknown')
            question = prompt_data.get('question', 'Input needed')
            subject = prompt_data.get('subject')
            default = prompt_data.get('default')
            allow_never = prompt_data.get('allow_never')
            # Update state before returning prompt signal
            with state.lock: update_aider_state_from_coder(aider_json_path, state.coder)
            response_guidance = f"Aider requires further input. Respond using 'submit_editor_input'. Prompt: '{question}'"
            if subject: response_guidance += f" (Regarding: {subject[:100]}{'...' if len(subject)>100 else ''})"
            if default: response_guidance += f" [Default: {default}]"
            if prompt_type == 'confirm':
                 options = "(yes/no"; options += "/all/skip" if prompt_data.get('group_id') else ""; options += "/don't ask" if allow_never else ""; options += ")"
                 response_guidance += f" Options: {options}"
            return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"
        elif message_type == 'result':
             logger.info(f"Aider edit completed successfully for {chat_id} after input.")
             with state.lock: update_aider_state_from_coder(aider_json_path, state.coder); state.thread = None
             return f"Edit completed. {message.get('content', 'No output captured.')}"
        elif message_type == 'error':
             logger.error(f"Aider edit failed for {chat_id} after input.")
             error_content = message.get('message', 'Unknown error')
             try:
                 with state.lock: update_aider_state_from_coder(aider_json_path, state.coder)
             except Exception: pass
             remove_active_coder_state(aider_json_path)
             return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             try:
                 with state.lock: update_aider_state_from_coder(aider_json_path, state.coder)
             except Exception: pass
             remove_active_coder_state(aider_json_path)
             return f"Error: Unknown response type '{message_type}' from Aider process."

    async def _arun(self, user_response: str) -> str:
        # Pass the already confirmed response
        return self._run(user_response)


from ...errors import PromptNeededError # Import the custom exception

# ... (in SubmitFileEditorInputTool_HITL class) ...

class SubmitInput_HITL(BaseTool):
    name: str = "respond_to_code_copilot_input_request" # Keep the original intended name for LLM use
    description: str = (
         "Use to provide respond to Code Copilot only when requested (marked by '[CODE_COPILOT_INPUT_REQUEST]'). "
    )
    args_schema: Type[BaseModel] = UserResponseSchema
    requires_confirmation: bool = True # Mark this tool as requiring HITL

    # Modify _run to use the PromptNeededError approach
    def _run(self, user_response: str, confirmed_input: Optional[str] = None) -> str:
        """Handles submitting input to Aider using the PromptNeededError approach."""
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return "Error: No active editor session found or state could not be restored."

        # --- HITL Prompt via PromptNeededError ---
        if confirmed_input is None:  # First call - needs confirmation
            logger.debug(f"SubmitFileEditorInputTool: Raising PromptNeededError for input")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"user_response": user_response},
                edit_key="user_response",
                prompt_suffix="(edit or confirm response) > "  # Custom suffix
            )

        # --- Handle Confirmed Input (second invocation) ---
        edited_response = confirmed_input
        
        # Check again after confirmation
        if not edited_response.strip():
             logger.warning("SubmitFileEditorInputTool: Received empty confirmed input.")
             return "Error: Confirmed response is empty."

        # --- Logic to send to Aider and get response ---
        processed_result = "" # Initialize result string
        
        # Acquire lock for the specific session
        with state.lock:
            if not state.thread or not state.thread.is_alive():
                remove_active_coder_state(aider_json_path)
                return "Error: The editing process is not waiting for input."
            
            logger.debug(f"Putting confirmed response on input_q: '{edited_response[:50]}...' for {chat_id}")
            state.input_q.put(edited_response)
        
        # Release lock before waiting
        logger.debug(f"Main thread waiting for response from output_q for {chat_id}...")
        try:
            message = state.output_q.get(timeout=TIMEOUT)
            logger.debug(f"Main thread received message from output_q: {message.get('type')}")
            
            # Process the message
            message_type = message.get('type')
            if message_type == 'prompt':
                # Another prompt needed - raise another PromptNeededError
                prompt_data = message
                question = prompt_data.get('question', 'Input needed')
                subject = prompt_data.get('subject')
                default = prompt_data.get('default')
                
                # Update state before returning prompt signal
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
                
                # Raise a new PromptNeededError with the details from Aider
                prompt_msg = question
                if subject:
                    prompt_msg += f" (Re: {subject[:50]}{'...' if len(subject) > 50 else ''})"
                
                new_args = {"user_response": default or ""}
                
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args=new_args,
                    edit_key="user_response",
                    prompt_suffix=f"({prompt_msg}) > "
                )
            
            elif message_type == 'result':
                logger.info(f"Aider edit completed successfully for {chat_id} after input.")
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
                    state.thread = None
                processed_result = f"Edit completed. {message.get('content', 'No output captured.')}"
            
            elif message_type == 'error':
                logger.error(f"Aider edit failed for {chat_id} after input.")
                error_content = message.get('message', 'Unknown error')
                try:
                    with state.lock:
                        update_aider_state_from_coder(aider_json_path, state.coder)
                except Exception:
                    pass
                remove_active_coder_state(aider_json_path)
                processed_result = f"Error during edit:\n{error_content}"
            
            else:
                logger.error(f"Received unknown message type from Aider thread: {message_type}")
                try:
                    with state.lock:
                        update_aider_state_from_coder(aider_json_path, state.coder)
                except Exception:
                    pass
                remove_active_coder_state(aider_json_path)
                processed_result = f"Error: Unknown response type '{message_type}'."
                
        except queue.Empty:
            logger.error(f"Timeout waiting for Aider response ({chat_id}).")
            remove_active_coder_state(aider_json_path)
            processed_result = "Error: Timed out waiting for Aider response after submitting input."
        except PromptNeededError:
            # Re-raise any PromptNeededError to be handled by chat_manager
            raise
        except Exception as e:
            logger.error(f"Exception waiting on output_q for {chat_id}: {e}", exc_info=True)
            remove_active_coder_state(aider_json_path)
            processed_result = f"Error: Exception while waiting for Aider after submitting input: {e}"
            
        # Return the result string
        return processed_result
        
    async def _arun(self, user_response: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, user_response, confirmed_input)

# Replace the old implementation
submit_code_editor_input_tool = SubmitInput_HITL()

# Remove the non-HITL version as it's no longer needed
# submit_code_editor_input_direct_tool = SubmitFileEditorInputTool()

# --- Create tool instances ---
aider_usage_guide_tool = AiderUsageGuideTool()
add_code_file_tool = AddFileToConext()
drop_code_file_tool = RemoveFileFromContext()
list_code_files_tool = ListFilesInContext()
edit_code_tool = RequestEdits()
view_diff_tool = ViewDiffs()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseCodeCopilot()

# Define the tools that belong to this toolset
toolset_tools = [
    aider_usage_guide_tool,
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    submit_code_editor_input_tool,
    view_diff_tool,
    undo_last_edit_tool,
    close_code_editor_tool
]

# Register all tools with the central registry
register_tools([
    aider_usage_guide_tool,
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    submit_code_editor_input_tool,
    view_diff_tool,
    undo_last_edit_tool,
    close_code_editor_tool
])

logger.debug(f"Registered File Editor toolset with {len(toolset_tools)} tools")