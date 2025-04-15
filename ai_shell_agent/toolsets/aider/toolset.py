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
from typing import Dict, List, Optional, Any

# Langchain imports
from langchain_core.tools import BaseTool
from prompt_toolkit import prompt

# Local imports
from ... import logger
from ...tool_registry import register_tools
from ...chat_state_manager import (
    get_current_chat,
    get_active_toolsets,
    update_active_toolsets,
    get_toolset_data_path,
    _read_json,
    _write_json
)
# Import config manager helpers needed for API keys during startup
from ...config_manager import (
    get_current_model as get_agent_model, # Rename to avoid confusion
    get_api_key_for_model, # For API key checks
    get_model_provider, # Added for model provider lookup
    set_api_key_for_model, # Added for setting API keys
    normalize_model_name, # Added for model name normalization
    ALL_MODELS # Added models dictionary
)

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
toolset_name = "File Editor"
toolset_id = "aider" # Explicitly define ID for consistency
toolset_description = "Provides tools for editing and managing code files using AI"

# --- Configuration ---
DEFAULT_EDITOR_MODEL = "gpt-4o-mini" # Example default
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
    print(f"\n--- Select File Editor '{role_name}' Model ---")
    print("Available models:")
    all_model_names = sorted(list(set(ALL_MODELS.values())))
    effective_current = current_value if current_value is not None else default_value
    for model in all_model_names:
        marker = " <- Current Setting" if model == effective_current else ""
        print(f"- {model}{marker}")
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        if aliases: print(f"  (aliases: {', '.join(aliases)})")

    prompt_msg = (f"Enter model name for '{role_name}'"
                  f" (leave empty to keep '{effective_current or 'Default'}',"
                  f" enter 'none' to reset to default): ")

    while True:
        selected = input(prompt_msg).strip()
        if not selected:
            print(f"Keeping current setting: {effective_current or 'Default'}")
            # Return the value that *was* current (could be None)
            return current_value
        elif selected.lower() == 'none':
            print(f"Resetting {role_name} model to default ('{default_value or 'Agent Default' if role_name == 'Main' else 'Aider Default'}').")
            return None # Use None to signify reset/default
        else:
            normalized_model = normalize_model_name(selected)
            if normalized_model in all_model_names:
                return normalized_model # Return the selected normalized name
            else:
                print(f"Error: Unknown model '{selected}'. Please choose from the list or enter 'none'.")

def _prompt_for_edit_format_config(current_value: Optional[str]) -> Optional[str]:
    """Prompts the user to select an Aider edit format."""
    print("\n--- Select File Editor Edit Format ---")
    i = 0
    valid_choices = {}
    default_display = "Aider Default (model-specific)"
    print(f"  0: {default_display} {'<- Current Setting' if current_value is None else ''}")
    valid_choices['0'] = None

    format_list = sorted(AIDER_EDIT_FORMATS.keys())
    for idx, fmt in enumerate(format_list, 1):
        description = AIDER_EDIT_FORMATS[fmt]
        marker = " <- Current Setting" if fmt == current_value else ""
        print(f"  {idx}: {fmt}{marker} - {description}")
        valid_choices[str(idx)] = fmt

    while True:
        try:
            choice = input(f"Enter choice (0-{len(format_list)}), leave empty to keep current: ").strip()
            if not choice:
                print(f"Keeping current setting: {current_value or default_display}")
                return current_value
            elif choice in valid_choices:
                selected_format = valid_choices[choice]
                print(f"Selected format: {selected_format or default_display}")
                return selected_format
            else:
                print("Invalid choice. Please try again.")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled. Keeping current setting.")
            return current_value

def configure_toolset(config_path: Path, current_config: Optional[Dict]) -> Dict:
    """Configuration function for the File Editor (Aider) toolset."""
    logger.info(f"Configuring File Editor toolset. Config path: {config_path}")
    config = current_config or {} # Start with current or empty dict
    defaults = toolset_config_defaults

    print("\n--- Configure File Editor (Aider) Settings ---")
    print("Select the models and edit format the File Editor should use.")
    print("Leave input empty at any step to keep the current setting.")
    print("Enter 'none' to reset a setting to its default value.")

    # --- Model Selection ---
    agent_model = get_agent_model() # Get the main agent model to use as default for 'main'

    # Main Model
    selected_main = _prompt_for_single_model_config(
        "Main/Architect",
        config.get("main_model"),
        defaults["main_model"] or agent_model # Default to agent model if not set
    )
    config["main_model"] = selected_main

    # Editor Model
    selected_editor = _prompt_for_single_model_config(
        "Editor",
        config.get("editor_model"),
        defaults["editor_model"]
    )
    config["editor_model"] = selected_editor

    # Weak Model
    selected_weak = _prompt_for_single_model_config(
        "Weak (Commits etc.)",
        config.get("weak_model"),
        defaults["weak_model"]
    )
    config["weak_model"] = selected_weak

    # --- Edit Format Selection ---
    selected_format = _prompt_for_edit_format_config(config.get("edit_format"))
    config["edit_format"] = selected_format

    # --- Ensure API Keys ---
    print("\nChecking API keys for selected File Editor models...")
    models_to_check = {
        config.get("main_model"),
        config.get("editor_model"),
        config.get("weak_model")
    }
    # Also check agent's main model if it's different and used as default
    if config.get("main_model") is None: models_to_check.add(agent_model)

    checked_providers = set()
    required_keys_ok = True
    for model_name in filter(None, models_to_check): # Filter out None values
        provider = get_model_provider(model_name)
        if provider not in checked_providers:
             api_key, env_var = get_api_key_for_model(model_name)
             if not api_key:
                 logger.warning(f"API key ({env_var}) needed for File Editor model ({model_name}) but not found.")
                 try:
                     set_api_key_for_model(model_name) # Prompt user
                     api_key, _ = get_api_key_for_model(model_name) # Check again
                     if not api_key:
                          print(f"Warning: API key ({env_var}) still not provided. File Editor may fail.")
                          required_keys_ok = False # Mark as potentially problematic
                 except (EOFError, KeyboardInterrupt):
                      print(f"\nAPI key entry skipped. File Editor may fail.")
                      required_keys_ok = False
             else:
                  logger.debug(f"API key ({env_var}) found for File Editor model {model_name}.")
             checked_providers.add(provider)

    # --- Save Config ---
    try:
        # Merge with defaults to ensure all keys are present? No, save only user-set or prompted values.
        # Let the toolset handle defaults when reading later.
        _write_json(config_path, config)
        logger.info(f"File Editor configuration saved to {config_path}")
        print("\nFile Editor configuration updated.")
        if not required_keys_ok:
             print("Note: Some API keys were missing or skipped. Ensure they are set in your environment.")
    except Exception as e:
         logger.error(f"Failed to save File Editor configuration to {config_path}: {e}")
         print(f"\nError: Failed to save configuration: {e}")
         return current_config or {} # Return old config on save error

    return config # Return the final config dict

# --- Tool Classes ---

class OpenFileEditor(BaseTool):
    name: str = "open_file_editor"
    description: str = "Use this to start the file editor, whenever asked to edit contents of any text file. The editor works for any text file including advanced code editing. You operate it using natural language commands. Useful only for modifying files."

    def _run(self, **kwargs) -> str:
        """
        Initializes the Aider state, activates the 'File Editor' toolset,
        and updates the system prompt.
        
        Handles both direct args and v__args wrapper format from LangChain.
        """
        # Extract real args if wrapped in v__args (for compatibility with some LLM bindings)
        args = kwargs.get("v__args", [])
        if isinstance(args, list) and len(args) > 0:
            # If we have positional args in v__args, use the first one
            args = args[0] if args else ""
        elif "args" in kwargs:
            # Try direct 'args' parameter
            args = kwargs.get("args", "")
        else:
            # Otherwise use empty string
            args = ""

        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        toolset_name = "File Editor"  # The name used in active_toolsets

        # --- Toolset Activation ---
        current_toolsets = get_active_toolsets(chat_id)
        activation_feedback = ""
        if (toolset_name not in current_toolsets) and (toolset_id not in current_toolsets):
            logger.debug(f"Activating '{toolset_name}' toolset for chat {chat_id}.")
            new_toolsets = list(current_toolsets)
            new_toolsets.append(toolset_name)
            update_active_toolsets(chat_id, new_toolsets)  # Save updated toolsets list
            activation_feedback = f"'{toolset_name}' toolset activated.\n\n"
            logger.debug(f"System prompt will be implicitly updated by LLM using new toolset state.")

        # --- Get json file path for this toolset ---
        aider_json_path = get_toolset_data_path(chat_id, toolset_id) 

        # --- Read current state or initialize empty ---
        aider_state = _read_json(aider_json_path, default_value={})

        # Check if already enabled
        if aider_state.get("enabled", False):
            # Try to resume existing session
            state = ensure_active_coder_state(aider_json_path)
            if state and state.coder:
                logger.info(f"Resuming File Editor session for chat {chat_id}.")
                files_str = ', '.join(state.coder.get_rel_fnames()) or 'None'
                status_message = f"Resumed existing File Editor session. Files: {files_str}."
                return f"{activation_feedback}{status_message}\n\n{AIDER_TOOLSET_PROMPT}"
            else:
                # Failed to resume - fall through to initialization code
                logger.warning(f"Failed to resume File Editor session for {chat_id}. Will reinitialize.")
        
        # --- Initialize new state ---
        # Check if we need to run configuration
        config_exists = bool(aider_state)  # Non-empty dict has some config
        if not config_exists:
            # Run configuration wizard (which will save the config to the path)
            logger.info(f"Running configuration wizard for new File Editor session ({chat_id}).")
            try:
                # Call the configuration function (which handles prompting and saving)
                configure_toolset(aider_json_path, aider_state)
                # Reload the state after configuration
                aider_state = _read_json(aider_json_path, default_value={})
            except (EOFError, KeyboardInterrupt) as e:
                logger.warning(f"Configuration cancelled for File Editor: {e}")
                return f"{activation_feedback}Error: Configuration cancelled. Please try again."
            except Exception as e:
                logger.error(f"Error during configuration: {e}", exc_info=True)
                return f"{activation_feedback}Error: Configuration failed: {e}"

        # --- Finalize and enable the state ---
        # Ensure state has minimal required fields
        aider_state["enabled"] = True
        if "abs_fnames" not in aider_state:
            aider_state["abs_fnames"] = []
        if "aider_done_messages" not in aider_state:
            aider_state["aider_done_messages"] = []

        # Save the initialized state
        try:
            _write_json(aider_json_path, aider_state)
        except Exception as e:
            logger.error(f"Failed to save Aider state: {e}", exc_info=True)
            return f"{activation_feedback}Error: Failed to initialize state: {e}."

        # Create the active state
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return f"{activation_feedback}Error: Failed to initialize File Editor. Check logs."

        status_message = "New File Editor session started. Use 'include_file' to add files."
        logger.info(f"New File Editor session initialized for {chat_id}")
        return f"{activation_feedback}{status_message}\n\n{AIDER_TOOLSET_PROMPT}"
        
    async def _arun(self, args: str = "") -> str:
        return self._run(args)


class OpenFileTool(BaseTool):
    name: str = "open_file"
    description: str = "Opens a file for editing in the File Editor, files can be added by absolute or relative paths. The file must exist in the filesystem."
    
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


class CloseFileTool(BaseTool):
    name: str = "close_file"
    description: str = "Closes a file from the File Editor. Files can be closed by relative or absolute paths, for files that were previously opened."
    
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


class ListOpenFilesTool(BaseTool):
    name: str = "list_files"
    description: str = "Lists all files currently in the File Editor's context. Can be used to preview what files are open for editing."
    
    def _run(self, **kwargs) -> str:
        """Lists all files in the Aider context."""
        # Extract real args if wrapped in v__args (for compatibility with LangChain bindings)
        args = kwargs.get("v__args", [])
        if isinstance(args, list) and len(args) > 0:
            args = args[0]
        elif "args" in kwargs:
            # Try direct 'args' parameter
            args = kwargs.get("args", "")
        else:
            # Otherwise use empty string
            args = ""
            
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
            
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class RequestEditsTool(BaseTool):
    name: str = "request_edit"
    description: str = "Using natural language, request an edit to the files opened in the File Editor. The editor is AI powered, the editor AI will respond with a plan and then execute it. Use this tool after adding files."
    
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


class ViewDiffTool(BaseTool):
    name: str = "view_changes"
    description: str = "Shows the git diff of changes made by the 'request_edit' tool in the current session. This is useful to see what changes have been made to the files. Works only if there was a git repository initialized in the project root."
    
    def _run(self, **kwargs) -> str:
        """Shows the diff of changes made by Aider."""
        # Extract real args if wrapped in v__args (for compatibility with LangChain bindings)
        args = kwargs.get("v__args", [])
        if isinstance(args, list) and len(args) > 0:
            args = args[0] if args else ""
        elif "args" in kwargs:
            # Try direct 'args' parameter
            args = kwargs.get("args", "")
        else:
            # Otherwise use empty string
            args = ""
            
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
            
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class UndoLastEditTool(BaseTool):
    name: str = "undo_last_edit"
    description: str = "Undoes the last edit commit made by the 'request_edit' tool. This is useful to revert changes made to the files, might not work if the commit was made outside of the File Editor or if there is no git repository initialized."
    
    def _run(self, **kwargs) -> str:
        """Undoes the last edit commit made by Aider."""
        # Extract real args if wrapped in v__args (for compatibility with LangChain bindings)
        args = kwargs.get("v__args", [])
        if isinstance(args, list) and len(args) > 0:
            args = args[0] if args else ""
        elif "args" in kwargs:
            # Try direct 'args' parameter
            args = kwargs.get("args", "")
        else:
            # Otherwise use empty string
            args = ""
            
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
            
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class CloseFileEditorTool(BaseTool):
    name: str = "close_file_editor"
    description: str = "Closes the File Editor and all the files. Changes are saved automatically. Close it as soon as you verified with the user they don't want to edit the files anymore, once verified close the File Editor right away."

    def _run(self, **kwargs) -> str:
        """Clears the Aider state, deactivates the toolset, and updates the prompt."""
        # Extract real args if wrapped in v__args (for compatibility with LangChain bindings)
        args = kwargs.get("v__args", [])
        if isinstance(args, list) and len(args) > 0:
            args = args[0] if args else ""
        elif "args" in kwargs:
            # Try direct 'args' parameter
            args = kwargs.get("args", "")
        else:
            # Otherwise use empty string
            args = ""
            
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found to close the editor for."

        toolset_name = "File Editor"
        
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

        # --- Deactivate Toolset ---
        current_toolsets = get_active_toolsets(chat_id)
        toolset_deactivated_msg = ""
        if toolset_name in current_toolsets:
            logger.info(f"Deactivating '{toolset_name}' toolset for chat {chat_id}.")
            new_toolsets = [ts for ts in current_toolsets if ts != toolset_name]
            update_active_toolsets(chat_id, new_toolsets)  # Save updated list
            toolset_deactivated_msg = f"'{toolset_name}' toolset deactivated."
            # No manual prompt update needed here
            logger.info(f"System prompt will be implicitly updated by LLM using new toolset state.")
        else:
             toolset_deactivated_msg = f"'{toolset_name}' toolset was already inactive."
        return f"{state_cleared_msg} {toolset_deactivated_msg}".strip()
             
    async def _arun(self, **kwargs) -> str:
        # Simple enough to run synchronously
        return self._run(**kwargs)


class SubmitFileEditorInputTool(BaseTool):
    """
    Tool to provide direct input to the File Editor when it requests it.
    WARNING: This version does NOT include Human-in-the-Loop confirmation.
    Use with caution, intended for specific internal use or future automated workflows.
    """
    name: str = "submit_editor_input_direct" # Changed name to avoid collision
    description: str = (
         "Directly submit input when the File Editor requests it (marked by '[FILE_EDITOR_INPUT_NEEDED]'). "
         "This version bypasses user confirmation."
    )
    def _run(self, user_response: str) -> str:
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # Get path to the aider.json file for this chat
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        # Ensure active state exists
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return "Error: No active editor session found or state could not be restored."

        # Acquire lock for the specific session
        with state.lock:
             # Check if the thread is actually running and waiting
             if not state.thread or not state.thread.is_alive():
                  remove_active_coder_state(aider_json_path)
                  return "Error: The editing process is not waiting for input."

             # --- Original Logic: Send the response directly ---
             logger.debug(f"Putting direct user response on input_q: '{user_response}' for {chat_id}")
             state.input_q.put(user_response)
             # --- End Original Logic ---

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

        # Process the message (identical logic to RunCodeEditTool's processing part)
        message_type = message.get('type')

        if message_type == 'prompt':
            prompt_data = message
            prompt_type = prompt_data.get('prompt_type', 'unknown')
            question = prompt_data.get('question', 'Input needed')
            subject = prompt_data.get('subject')
            default = prompt_data.get('default')
            allow_never = prompt_data.get('allow_never')

            # Update the state before returning the prompt
            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)

            response_guidance = f"Aider requires further input. Please respond using 'submit_editor_input'. Prompt: '{question}'"
            if subject: response_guidance += f" (Regarding: {subject[:100]}{'...' if len(subject)>100 else ''})"
            if default: response_guidance += f" [Default: {default}]"
            if prompt_type == 'confirm':
                 options = "(yes/no"
                 if prompt_data.get('group_id'): options += "/all/skip"
                 if allow_never: options += "/don't ask"
                 options += ")"
                 response_guidance += f" Options: {options}"

            return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"

        elif message_type == 'result':
             logger.info(f"Aider edit completed successfully for {chat_id} after input.")
             with state.lock:
                  update_aider_state_from_coder(aider_json_path, state.coder)
                  state.thread = None
             return f"Edit completed. {message.get('content', 'No output captured.')}"

        elif message_type == 'error':
             logger.error(f"Aider edit failed for {chat_id} after input.")
             error_content = message.get('message', 'Unknown error')
             try:
                 with state.lock:
                     update_aider_state_from_coder(aider_json_path, state.coder)
             except Exception: pass
             remove_active_coder_state(aider_json_path)
             return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             try:
                 with state.lock:
                     update_aider_state_from_coder(aider_json_path, state.coder)
             except Exception: pass
             remove_active_coder_state(aider_json_path)
             return f"Error: Unknown response type '{message_type}' from Aider process."

    async def _arun(self, user_response: str) -> str:
        # Consider running sync in threadpool if needed
        return self._run(user_response)


class SubmitFileEditorInputTool_HITL(BaseTool):
    name: str = "submit_editor_input" # Keep the original intended name for LLM use
    description: str = (
         "Use to provide input when the File Editor requests it (marked by '[FILE_EDITOR_INPUT_NEEDED]'). "
         "The proposed input will be shown to the user for confirmation or editing before being submitted."
    )
    def _run(self, user_response: str) -> str:
        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return "Error: No active editor session found or state could not be restored."
            
        # --- HITL Modification START ---
        logger.info(f"Proposing editor input: {user_response}")
        print(f"\n[Proposed response to File Editor]:")
        try:
            # Use the LLM's proposed response as the default
            edited_response = prompt("(Accept or Edit) > ", default=user_response)
        except EOFError:
             logger.warning("User cancelled input submission (EOF).")
             return "Input submission cancelled by user."

        # If the user provided an empty response, treat it as a cancellation
        if not edited_response.strip():
            logger.warning("User cancelled input submission by submitting empty input.")
            return "Input submission cancelled by user (empty input)."
        # --- HITL Modification END ---

        # Acquire lock for the specific session
        with state.lock:
             # Check if the thread is actually running and waiting
             if not state.thread or not state.thread.is_alive():
                  remove_active_coder_state(aider_json_path)
                  return "Error: The editing process is not waiting for input."

             # --- Use the potentially edited response ---
             logger.debug(f"Putting user confirmed/edited response on input_q: '{edited_response}' for {chat_id}")
             state.input_q.put(edited_response)
             # --- End Use edited response ---

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

        # Process the message (identical logic to RunCodeEditTool's processing part)
        message_type = message.get('type')

        if message_type == 'prompt':
            prompt_data = message
            prompt_type = prompt_data.get('prompt_type', 'unknown')
            question = prompt_data.get('question', 'Input needed')
            subject = prompt_data.get('subject')
            default = prompt_data.get('default')
            allow_never = prompt_data.get('allow_never')

            # Update the state before returning the prompt
            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)

            response_guidance = f"Aider requires further input. Please respond using 'submit_editor_input'. Prompt: '{question}'"
            if subject: response_guidance += f" (Regarding: {subject[:100]}{'...' if len(subject)>100 else ''})"
            if default: response_guidance += f" [Default: {default}]"
            if prompt_type == 'confirm':
                 options = "(yes/no"
                 if prompt_data.get('group_id'): options += "/all/skip"
                 if allow_never: options += "/don't ask"
                 options += ")"
                 response_guidance += f" Options: {options}"
                 
            return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"

        elif message_type == 'result':
             logger.info(f"Aider edit completed successfully for {chat_id} after input.")
             with state.lock:  # Re-acquire lock
                  update_aider_state_from_coder(aider_json_path, state.coder)
                  state.thread = None
             return f"Edit completed. {message.get('content', 'No output captured.')}"

        elif message_type == 'error':
             logger.error(f"Aider edit failed for {chat_id} after input.")
             error_content = message.get('message', 'Unknown error')
             
             # Even on error, try to update state to preserve any partial changes
             try:
                 with state.lock:
                     update_aider_state_from_coder(aider_json_path, state.coder)
             except Exception:
                 pass  # Ignore errors during final state update
                 
             remove_active_coder_state(aider_json_path)
             return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             
             # Try to update state before cleanup
             try:
                 with state.lock:
                     update_aider_state_from_coder(aider_json_path, state.coder)
             except Exception:
                 pass  # Ignore errors during final state update
                 
             remove_active_coder_state(aider_json_path)
             return f"Error: Unknown response type '{message_type}' from Aider process."
             
    async def _arun(self, user_response: str) -> str:
        # Consider running sync in threadpool if needed
        return self._run(user_response)


# --- Create tool instances ---
start_code_editor_tool = OpenFileEditor()
add_code_file_tool = OpenFileTool()
drop_code_file_tool = CloseFileTool()
list_code_files_tool = ListOpenFilesTool()
edit_code_tool = RequestEditsTool()
submit_code_editor_input_tool = SubmitFileEditorInputTool_HITL()
submit_code_editor_input_direct_tool = SubmitFileEditorInputTool()
view_diff_tool = ViewDiffTool()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseFileEditorTool()

# Define the tools that belong to this toolset
toolset_tools = [
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    submit_code_editor_input_tool,
    submit_code_editor_input_direct_tool,
    view_diff_tool,
    undo_last_edit_tool,
    close_code_editor_tool
]
toolset_start_tool = start_code_editor_tool

# Register all tools with the central registry
register_tools([
    start_code_editor_tool,
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    submit_code_editor_input_tool,
    submit_code_editor_input_direct_tool,
    view_diff_tool,
    undo_last_edit_tool,
    close_code_editor_tool
])

logger.debug(f"Registered File Editor toolset with {len(toolset_tools) + 1} tools")