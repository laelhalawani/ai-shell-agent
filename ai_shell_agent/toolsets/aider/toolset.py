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
    get_aider_state,
    save_aider_state,
    clear_aider_state,
    get_active_toolsets,
    update_active_toolsets,
)
from ...config_manager import (
    get_current_model, 
    get_api_key_for_model, 
    get_aider_edit_format, 
    get_aider_main_model, 
    get_aider_editor_model, 
    get_aider_weak_model
)
# Import from ai.py without creating circular imports
from ...ai import ensure_api_keys_for_coder_models
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

# --- Toolset metadata for discovery ---
toolset_name = "File Editor"
toolset_description = "Provides tools for editing and managing code files using AI"

# --- Tool Classes ---

class StartAIEditorTool(BaseTool):
    name: str = "start_file_editor"
    description: str = "Use this to start the file editor, whenever asked to edit contents of any file. The editor works for any text file including advanced code editing. You operate it using natural language commands. More information will be present upon startup."

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

        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        toolset_name = "File Editor"  # The name defined in llm.py

        # --- Toolset Activation & Prompt Update ---
        current_toolsets = get_active_toolsets(chat_file)
        activation_feedback = ""
        if toolset_name not in current_toolsets:
            logger.debug(f"Activating '{toolset_name}' toolset for chat {chat_file}.")
            new_toolsets = list(current_toolsets)
            new_toolsets.append(toolset_name)
            update_active_toolsets(chat_file, new_toolsets)  # Save updated toolsets list
            activation_feedback = f"'{toolset_name}' toolset activated.\n\n"
            logger.debug(f"System prompt will be implicitly updated by LLM using new toolset state.")
        # --- End Toolset Activation ---

        # --- Aider Initialization Logic ---
        try:
            ensure_api_keys_for_coder_models()
        except Exception as e:
            logger.error(f"Error checking API keys for coder models: {e}")
            # Return error message *including* the prompt content for usage info
            return f"{activation_feedback}Warning: Failed to validate API keys: {e}. Toolset may not function.\n\n{AIDER_TOOLSET_PROMPT}"

        state = ensure_active_coder_state(chat_file)
        if state and state.coder:
            # Resume existing session
            logger.info(f"Resuming File Editor session for {chat_file}.")
            files_str = ', '.join(state.coder.get_rel_fnames()) or 'None'
            status_message = f"Resumed existing File Editor session. Files: {files_str}."
            # Return combined feedback including the toolset prompt
            return f"{activation_feedback}{status_message}\n\n{AIDER_TOOLSET_PROMPT}"
        elif get_aider_state(chat_file) is None:  # No state, try creating fresh
            # Try to start fresh session
            fresh_state = ensure_active_coder_state(chat_file) # Should create if possible
            if fresh_state:
                status_message = "New File Editor session started. Use 'include_file' to add files."
                return f"{activation_feedback}{status_message}\n\n{AIDER_TOOLSET_PROMPT}"
            else:  # Failed to create fresh state
                clear_aider_state(chat_file)  # Mark as disabled
                return f"{activation_feedback}Error: Failed to initialize new File Editor session. Check config/logs.\n\n{AIDER_TOOLSET_PROMPT}"
        else:  # State exists but recreation failed
            clear_aider_state(chat_file)  # Mark as disabled
            return f"{activation_feedback}Error: Failed to restore File Editor session. Cleared state. Try starting again.\n\n{AIDER_TOOLSET_PROMPT}"
        
    async def _arun(self, args: str = "") -> str:
        return self._run(args)


class AddFileTool(BaseTool):
    name: str = "include_file"
    description: str = "Before the File Editor can edit any file, they need to be included in the editor's context. Argument must be the relative or absolute file path to add."
    
    def _run(self, file_path: str) -> str:
        """Adds a file to the Aider context, recreating state if needed."""
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        # --- Ensure active state exists ---
        state = ensure_active_coder_state(chat_file)
        if not state:
            # If state is None after trying to recreate, it means editor isn't properly initialized
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."
        # --- End ensure active state ---

        # Now use the ensured state
        coder = state.coder
        io_stub = state.io_stub  # Use the IO stub from the active state

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
            update_aider_state_from_coder(chat_file, coder)
            logger.info(f"Added file {rel_path} and updated persistent state for {chat_file}")

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


class DropFileTool(BaseTool):
    name: str = "exclude_file"
    description: str = "Removes a file from the File Editor's context. Argument must be the relative or absolute file path that was previously added."
    
    def _run(self, file_path: str) -> str:
        """Removes a file from the Aider context."""
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Editor not initialized."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate editor state. {io_stub.get_captured_output()}"
            
        try:
            # Coder's drop_rel_fname expects relative path
            abs_path_to_drop = str(Path(file_path).resolve())
            rel_path_to_drop = coder.get_rel_fname(abs_path_to_drop)
            
            success = coder.drop_rel_fname(rel_path_to_drop)
            
            if success:
                # Update state after dropping the file
                update_aider_state_from_coder(chat_file, coder)
                return f"Successfully dropped {file_path}. {io_stub.get_captured_output()}"
            else:
                return f"File {file_path} not found in context. {io_stub.get_captured_output()}"
                
        except Exception as e:
            logger.error(f"Error in DropFileTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error dropping file {file_path}: {e}. {io_stub.get_captured_output()}"
            
    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


class ListFilesInEditorTool(BaseTool):
    name: str = "list_files"
    description: str = "Lists all files currently in the File Editor's context."
    
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
            
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
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


class RunCodeEditTool(BaseTool):
    name: str = "request_edit"
    description: str = "Using natural language, request an edit to the files in the editor. The AI will respond with a plan and then execute it. Use this tool after adding files."
    
    def _run(self, instruction: str) -> str:
        """Runs Aider's main edit loop in a background thread."""
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        # --- Ensure active state exists ---
        state = ensure_active_coder_state(chat_file)
        if not state:
            # If state is None after trying to recreate, it means editor isn't properly initialized
            return "Error: File Editor not initialized or failed to restore state. Use start_file_editor first."
        # --- End ensure active state ---

        if not state.coder.abs_fnames:
             return "Error: No files have been added to the editing session. Use include_file first."

        # --- Threading Logic ---
        # Acquire lock for this session before starting thread
        with state.lock:
            # Check if a thread is already running for this session
            if state.thread and state.thread.is_alive():
                logger.warning(f"An edit is already in progress for {chat_file}. Please wait or submit input if needed.")
                return "Error: An edit is already in progress for this session."

            # Ensure the coder's IO is the correct stub instance from the state
            if state.coder.io is not state.io_stub:
                 logger.warning("Correcting coder IO instance mismatch.")
                 state.coder.io = state.io_stub

            # Start the background thread
            logger.info(f"Starting Aider worker thread for: {chat_file}")
            state.thread = threading.Thread(
                target=_run_aider_in_thread,
                args=(state.coder, instruction, state.output_q),
                daemon=True, 
                name=f"AiderWorker-{chat_file[:8]}"
            )
            state.thread.start()
            
            # Update state before waiting - makes sure we have the latest before any edits
            update_aider_state_from_coder(chat_file, state.coder)
        # --- End Threading Logic ---

        # Release lock before waiting on queue
        # Wait for the *first* response from the Aider thread
        logger.debug(f"Main thread waiting for initial message from output_q for {chat_file}...")
        try:
             message = state.output_q.get(timeout=TIMEOUT)  # Add a timeout (e.g., 5 minutes)
             logger.debug(f"Main thread received initial message: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout waiting for initial Aider response ({chat_file}).")
             remove_active_coder_state(chat_file)  # Clean up state
             return "Error: Timed out waiting for Aider response."
        except Exception as e:
              logger.error(f"Exception waiting on output_q for {chat_file}: {e}")
              remove_active_coder_state(chat_file)
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
                update_aider_state_from_coder(chat_file, state.coder)
                
            return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"

        elif message_type == 'result':
            # Aider finished without needing input
            logger.info(f"Aider edit completed successfully for {chat_file}.")
            with state.lock:  # Re-acquire lock briefly
                update_aider_state_from_coder(chat_file, state.coder)
                state.thread = None  # Clear the thread reference as it's done
            return f"Edit completed. {message.get('content', 'No output captured.')}"

        elif message_type == 'error':
            # Aider encountered an error immediately
            logger.error(f"Aider edit failed for {chat_file}.")
            error_content = message.get('message', 'Unknown error')
            
            # Even on error, update state if possible - might have partial changes
            try:
                with state.lock:
                    update_aider_state_from_coder(chat_file, state.coder)
            except Exception:
                pass  # Ignore state update errors during cleanup
            
            remove_active_coder_state(chat_file)  # Clean up on error
            return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             remove_active_coder_state(chat_file)
             return f"Error: Unknown response type '{message_type}' from Aider process."
    
    async def _arun(self, instruction: str) -> str:
        # For simplicity in this stage, run synchronously.
        # Consider using asyncio.to_thread if true async is needed later.
        return self._run(instruction)


class ViewDiffTool(BaseTool):
    name: str = "view_diff"
    description: str = "Shows the git diff of changes made by the 'request_edit' tool in the current session. This is useful to see what changes have been made to the files."
    
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
            
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Editor not initialized or is closed. Use start_file_editor first."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate editor state. {io_stub.get_captured_output()}"
            
        try:
            # If there's no git repo, we can't show diffs
            if not coder.repo:
                return "Error: Git repository not available. Cannot show diff."
                
            # *** MODIFIED: Use commands object if available ***
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
    description: str = "Undoes the last edit commit made by the 'request_edit' tool. This is useful to revert changes made to the files, might not work if the commit was made outside of the File Editor."
    
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
            
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Editor not initialized. Use start_file_editor first."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate editor state. {io_stub.get_captured_output()}"
            
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
            
            # Update the commit hashes in the state
            aider_state["aider_commit_hashes"] = list(coder.aider_commit_hashes)
            save_aider_state(chat_file, aider_state)
            
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


class CloseCodeEditorTool(BaseTool):
    name: str = "close_file_editor"
    description: str = "Closes the File Editor session, clearing its context AND deactivating the 'File Editor' toolset."

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
            
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found to close the editor for."

        toolset_name = "File Editor"

        # --- Clear Aider Specific State ---
        state_cleared_msg = ""
        try:
            clear_aider_state(chat_file)  # Mark aider state as disabled
            remove_active_coder_state(chat_file)  # Remove active coder instance
            logger.info(f"Aider state cleared and active coder removed for {chat_file}")
            state_cleared_msg = "File Editor session context cleared."
        except Exception as e:
            logger.error(f"Error clearing Aider state: {e}", exc_info=True)
            state_cleared_msg = "File Editor session context possibly cleared (encountered error)."

        # --- Deactivate Toolset ---
        current_toolsets = get_active_toolsets(chat_file)
        toolset_deactivated_msg = ""
        if toolset_name in current_toolsets:
            logger.info(f"Deactivating '{toolset_name}' toolset for chat {chat_file}.")
            new_toolsets = [ts for ts in current_toolsets if ts != toolset_name]
            update_active_toolsets(chat_file, new_toolsets)  # Save updated list
            toolset_deactivated_msg = f"'{toolset_name}' toolset deactivated."
            # No manual prompt update needed here
            logger.info(f"System prompt will be implicitly updated by LLM using new toolset state.")
        else:
             toolset_deactivated_msg = f"'{toolset_name}' toolset was already inactive."
        return f"{state_cleared_msg} {toolset_deactivated_msg}".strip()
             
    async def _arun(self, **kwargs) -> str:
        # Simple enough to run synchronously
        return self._run(**kwargs)


class SubmitCodeEditorInputTool(BaseTool):
    name: str = "submit_editor_input"
    description: str = (
         "Use to provide input to input request (e.g., 'yes', 'no', 'all', 'skip', 'don't ask', or text) "
         "when the File Editor signals '[FILE_EDITOR_INPUT_NEEDED]'."
    )

    def _run(self, user_response: str) -> str:
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        # --- Ensure active state exists ---
        state = ensure_active_coder_state(chat_file)
        if not state:
            # If state is None after trying to recreate, it means editor isn't properly initialized
            return "Error: No active editor session found or state could not be restored."
        # --- End ensure active state ---
            
        # HITL: Allow the user to review and edit the response before submitting
        print(f"\n[Proposed response to File Editor]:")
        edited_response = prompt("(Accept or Edit) > ", default=user_response)
        
        # If the user provided an empty response, treat it as a cancellation
        if not edited_response.strip():
            return "Input submission cancelled by user."

        # Acquire lock for the specific session
        with state.lock:
             # Check if the thread is actually running and waiting
             if not state.thread or not state.thread.is_alive():
                  # Could happen if thread finished unexpectedly or was closed
                  # Clean up just in case
                  remove_active_coder_state(chat_file)
                  return "Error: The editing process is not waiting for input."

             # Send the edited user response to the waiting Aider thread
             logger.debug(f"Putting user response on input_q: '{edited_response}' for {chat_file}")
             state.input_q.put(edited_response)

        # Release lock before waiting on queue
        logger.debug(f"Main thread waiting for *next* message from output_q for {chat_file}...")
        try:
            # Wait for the Aider thread's *next* action (could be another prompt, result, or error)
             message = state.output_q.get(timeout=TIMEOUT)  # Added timeout here too
             logger.debug(f"Main thread received message from output_q: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout or queue error waiting for Aider response ({chat_file}).")
             remove_active_coder_state(chat_file)
             return "Error: Timed out waiting for Aider response after submitting input."
        except Exception as e:
             logger.error(f"Exception waiting on output_q for {chat_file}: {e}")
             remove_active_coder_state(chat_file)
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
                update_aider_state_from_coder(chat_file, state.coder)

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
             logger.info(f"Aider edit completed successfully for {chat_file} after input.")
             with state.lock:  # Re-acquire lock
                  update_aider_state_from_coder(chat_file, state.coder)
                  state.thread = None
             return f"Edit completed. {message.get('content', 'No output captured.')}"

        elif message_type == 'error':
             logger.error(f"Aider edit failed for {chat_file} after input.")
             error_content = message.get('message', 'Unknown error')
             
             # Even on error, try to update state to preserve any partial changes
             try:
                 with state.lock:
                     update_aider_state_from_coder(chat_file, state.coder)
             except Exception:
                 pass  # Ignore errors during final state update
                 
             remove_active_coder_state(chat_file)
             return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             
             # Try to update state before cleanup
             try:
                 with state.lock:
                     update_aider_state_from_coder(chat_file, state.coder)
             except Exception:
                 pass  # Ignore errors during final state update
                 
             remove_active_coder_state(chat_file)
             return f"Error: Unknown response type '{message_type}' from Aider process."
             
    async def _arun(self, user_response: str) -> str:
        # Consider running sync in threadpool if needed
        return self._run(user_response)


# --- Create tool instances ---
start_code_editor_tool = StartAIEditorTool()
add_code_file_tool = AddFileTool()
drop_code_file_tool = DropFileTool()
list_code_files_tool = ListFilesInEditorTool()
edit_code_tool = RunCodeEditTool()
submit_code_editor_input_tool = SubmitCodeEditorInputTool()
view_diff_tool = ViewDiffTool()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseCodeEditorTool()

# Define the tools that belong to this toolset
toolset_tools = [
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    submit_code_editor_input_tool,
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
    view_diff_tool,
    undo_last_edit_tool,
    close_code_editor_tool
])

logger.debug(f"Registered File Editor toolset with {len(toolset_tools) + 1} tools")