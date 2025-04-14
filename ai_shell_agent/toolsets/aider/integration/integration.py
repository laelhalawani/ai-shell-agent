# ai_shell_agent/toolsets/aider/integration/integration.py
"""
Integration between AI Shell Agent and aider-chat library.

This module handles the runtime state and persistence of Aider sessions,
allowing the same Aider editing session to be maintained across multiple
interactions with the LLM.
"""

import os
import io
import re
import sys
import queue
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

# Logger from parent package
from .... import logger

# Import JSON helpers from utils instead of chat_state_manager
from ....utils import read_json, write_json
from ....config_manager import get_current_model as get_agent_model

# Constants for Aider integration
SIGNAL_PROMPT_NEEDED = "[FILE_EDITOR_INPUT_NEEDED]"
TIMEOUT = 300  # 5 minutes timeout for Aider operations

# Direct imports for Aider components
import aider
from aider.coders import Coder
from aider.coders.base_coder import ANY_GIT_ERROR
from aider.models import Model
from aider.repo import GitRepo

# --- I/O Class for Aider Integration ---
class AiderIOStubWithQueues:
    """
    I/O stub for Aider that captures output and input via queues.
    This allows AI Shell Agent to interact with Aider programmatically.
    """
    def __init__(self):
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.input_provided = threading.Event()
        self.input_needed = threading.Event()
        self.verbose = False
        self.quiet = False
        
    def write(self, text, end="\n"):
        """Write output to the queue."""
        if text:
            logger.debug(f"Aider IO: {text}")
            self.output_queue.put(text + end)
        
    def write_raw(self, text):
        """Write raw output (without newline)."""
        if text:
            self.output_queue.put(text)
    
    def input(self, prompt=""):
        """Get input from the queue."""
        if prompt:
            self.write_raw(prompt)
        
        # Signal that input is needed and wait for it
        self.input_needed.set()
        self.input_provided.clear()
        
        # Wait for input to be provided
        self.input_provided.wait()
        
        # Get input from the queue
        result = self.input_queue.get()
        logger.debug(f"Aider input: {result}")
        return result
        
    def provide_input(self, text):
        """Provide input to Aider."""
        self.input_queue.put(text)
        self.input_needed.clear()
        self.input_provided.set()
        
    def is_input_needed(self):
        """Check if input is needed."""
        return self.input_needed.is_set()
        
    def clear_input_needed(self):
        """Clear the input needed flag."""
        self.input_needed.clear()
    
    def get_accumulated_output(self):
        """Get all accumulated output as a single string."""
        outputs = []
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get())
        return "".join(outputs) if outputs else ""

    # Required methods for Aider compatibility
    def reuse_last_answer(self):
        return False
        
    def get_last_answer(self):
        return None
        
    def bump_tokens(self, val=0):
        pass
        
    def wrap_for_prompt(self, code):
        return f"```\n{code}\n```"

# --- Runtime State Management ---
class ActiveCoderState:
    """Runtime state for an active Aider coder instance."""
    def __init__(self, coder, io_stub):
        self.coder = coder
        self.io_stub = io_stub
        self.last_activity = threading.Event()
        self.last_activity.set()  # Mark as active initially
        
    def mark_activity(self):
        """Mark activity to prevent timeout."""
        self.last_activity.set()

# Dictionary of active coder states
active_coders = {}
_active_coders_dict_lock = threading.RLock()

# --- Helper Functions ---
def get_chat_id_from_path(json_path: Path) -> str:
    """Helper to get chat_id from toolset json path"""
    # Assumes structure .../chats/<chat_id>/toolsets/aider.json
    return json_path.parent.parent.name

# --- Coder State Management ---
def get_active_coder_state(chat_id_or_key: str) -> Optional[ActiveCoderState]:
    """Get active coder state for a chat or key."""
    with _active_coders_dict_lock:
        return active_coders.get(chat_id_or_key)

def create_active_coder_state(chat_id_or_key: str, coder) -> ActiveCoderState:
    """Create and store a new active coder state."""
    io_stub = getattr(coder, "io", None)
    if not isinstance(io_stub, AiderIOStubWithQueues):
        logger.warning(f"Coder has unexpected IO type: {type(io_stub)}")
        io_stub = AiderIOStubWithQueues()
        coder.io = io_stub
        
    new_state = ActiveCoderState(coder, io_stub)
    with _active_coders_dict_lock:
        active_coders[chat_id_or_key] = new_state
    return new_state

def remove_active_coder_state(aider_json_path: Path):
    """Removes the active coder state and marks as disabled in aider.json."""
    chat_id = get_chat_id_from_path(aider_json_path)
    with _active_coders_dict_lock:
        state = active_coders.pop(chat_id, None)
        if state:
            logger.info(f"Removed active Aider session for chat: {chat_id}")
            # Mark as disabled in the persistent file
            try:
                persistent_state = read_json(aider_json_path, default_value={})
                if persistent_state.get("enabled", False):
                    persistent_state["enabled"] = False
                    write_json(aider_json_path, persistent_state)
                    logger.info(f"Marked Aider state disabled in {aider_json_path}")
            except Exception as e:
                logger.error(f"Failed to mark Aider state disabled in {aider_json_path}: {e}")
        else:
            logger.debug(f"No active Aider session found to remove for chat {chat_id}")

def ensure_active_coder_state(aider_json_path: Path) -> Optional[ActiveCoderState]:
    """
    Gets the active coder state, recreating it from aider.json if necessary.
    """
    chat_id = get_chat_id_from_path(aider_json_path)
    state = get_active_coder_state(chat_id) # Check runtime dict first
    if state:
        logger.debug(f"Found existing active coder state for chat {chat_id}")
        return state

    logger.info(f"No active coder state found for chat {chat_id}, attempting recreation from {aider_json_path}.")
    persistent_state = read_json(aider_json_path, default_value=None)

    if not persistent_state or not persistent_state.get("enabled", False):
        logger.warning(f"Cannot recreate active state: Persistent Aider state not found or disabled in {aider_json_path}.")
        return None

    # Recreate the coder instance from persistent state
    temp_io_stub = AiderIOStubWithQueues()
    recreated_coder = recreate_coder(aider_json_path, persistent_state, temp_io_stub) # Pass loaded state

    if not recreated_coder:
        logger.error(f"Failed to recreate Coder from persistent state for {aider_json_path}.")
        # Mark as disabled
        persistent_state["enabled"] = False
        write_json(aider_json_path, persistent_state)
        return None

    # Create and store the new active state using the recreated coder
    new_state = create_active_coder_state(chat_id, recreated_coder)
    logger.info(f"Successfully recreated and stored active coder state for chat {chat_id}")
    return new_state

def recreate_coder(aider_json_path: Path, aider_state: Dict, io_stub: AiderIOStubWithQueues) -> Optional[Any]:
    """
    Recreates the Aider Coder instance from the loaded persistent state dict.
    """
    try:
        # --- Model and Config Setup from aider_state ---
        main_model_name_state = aider_state.get('main_model')
        agent_default_model = get_agent_model() # Global agent model

        main_model_name = main_model_name_state if main_model_name_state is not None else agent_default_model
        if not main_model_name: # Should not happen if agent has a model
            logger.error("Cannot determine main model name for Aider Coder recreation.")
            return None

        logger.debug(f"Using main model: {main_model_name} (Source: {'Aider Config' if main_model_name_state is not None else 'Agent Default'})")

        editor_model_name = aider_state.get('editor_model') # Defaults handled by Model class or config
        weak_model_name = aider_state.get('weak_model')
        edit_format_state = aider_state.get('edit_format')
        editor_edit_format = aider_state.get('editor_edit_format')

        try:
            main_model_instance = Model(
                main_model_name,
                weak_model=weak_model_name,
                editor_model=editor_model_name,
                editor_edit_format=editor_edit_format
            )
            # Determine final edit format (State override > Model default)
            edit_format = edit_format_state if edit_format_state is not None else main_model_instance.edit_format
            logger.debug(f"Using edit format: {edit_format} (Source: {'Aider Config' if edit_format_state is not None else 'Model Default'})")

        except Exception as e:
            logger.error(f"Failed to instantiate main_model '{main_model_name}': {e}", exc_info=True)
            return None

        # --- Load History, Files, Git from aider_state ---
        aider_done_messages = aider_state.get("aider_done_messages", [])
        abs_fnames = aider_state.get("abs_fnames", [])
        abs_read_only_fnames = aider_state.get("abs_read_only_fnames", [])
        git_root = aider_state.get("git_root")
        repo = None
        if git_root:
            try: # Simplified GitRepo setup
                repo = GitRepo(io=io_stub, fnames=abs_fnames + abs_read_only_fnames, git_dname=str(Path(git_root)))
                # Optional: verify repo.root matches git_root from state
            except Exception as e: # Catch ANY_GIT_ERROR or others
                logger.warning(f"GitRepo init failed for {git_root}: {e}. Proceeding without git.")
                repo = None
                git_root = None # Clear git_root if repo fails

        # --- Prepare Coder kwargs ---
        coder_kwargs = dict(
            main_model=main_model_instance,
            edit_format=edit_format,
            io=io_stub,
            repo=repo,
            fnames=abs_fnames,
            read_only_fnames=abs_read_only_fnames,
            done_messages=aider_done_messages,
            cur_messages=[],
            auto_commits=aider_state.get("auto_commits", True), # Get from state or default
            dirty_commits=aider_state.get("dirty_commits", True),
            use_git=bool(repo),
            map_tokens=aider_state.get("map_tokens", 0),
            verbose=False, stream=False, suggest_shell_commands=False,
        )

        coder = Coder.create(**coder_kwargs)
        coder.root = git_root or os.getcwd() # Set root

        # No need to call create_active_coder_state here, caller does it
        logger.info(f"Coder successfully recreated for {aider_json_path}")
        return coder

    except Exception as e:
        logger.error(f"Failed to recreate Coder for {aider_json_path}: {e}", exc_info=True)
        return None

def update_aider_state_from_coder(aider_json_path: Path, coder) -> None:
    """Update the aider.json state from a Coder instance."""
    try:
        # Read existing state to preserve other potential keys? Or just overwrite?
        # Let's overwrite with known keys for simplicity now.
        new_state = {"enabled": True} # Always mark enabled when saving active coder

        # --- Update fields based on the coder ---
        new_state["main_model"] = coder.main_model.name
        new_state["edit_format"] = coder.edit_format
        new_state["weak_model_name"] = getattr(coder.main_model.weak_model, 'name', None)
        new_state["editor_model_name"] = getattr(coder.main_model.editor_model, 'name', None)
        new_state["editor_edit_format"] = getattr(coder.main_model, 'editor_edit_format', None)
        new_state["abs_fnames"] = sorted(list(coder.abs_fnames))
        new_state["abs_read_only_fnames"] = sorted(list(getattr(coder, "abs_read_only_fnames", [])))
        new_state["auto_commits"] = getattr(coder, "auto_commits", True)
        new_state["dirty_commits"] = getattr(coder, "dirty_commits", True)
        # Add map_tokens etc. if needed

        if coder.repo:
            new_state["git_root"] = coder.repo.root
            new_state["aider_commit_hashes"] = sorted(list(map(str, coder.aider_commit_hashes)))
        else: # Ensure keys are removed if no repo
            new_state["git_root"] = None
            new_state["aider_commit_hashes"] = []

        # *** Save Aider's internal conversation history ***
        try:
            new_state["aider_done_messages"] = coder.done_messages
            logger.debug(f"Saving {len(coder.done_messages)} messages to aider_done_messages state for {aider_json_path}.")
        except Exception as e:
            logger.error(f"Failed to serialize aider_done_messages for {aider_json_path}: {e}")
            new_state["aider_done_messages"] = [] # Save empty list on error

        # --- Save the updated state ---
        write_json(aider_json_path, new_state)
        logger.debug(f"Aider state updated from coder to {aider_json_path}")

    except Exception as e:
        logger.error(f"Failed to update persistent Aider state in {aider_json_path}: {e}", exc_info=True)

# --- Run Aider Command in Thread ---
def _run_aider_in_thread(coder, instruction: str, output_q: queue.Queue):
    """Run an Aider command in a separate thread."""
    if not coder:
        output_q.put({"error": "No active Aider session."})
        return
        
    try:
        # Parse the instruction as input to the coder
        coder.io.clear_input_needed()
        
        # Reset output queue to start fresh
        while not coder.io.output_queue.empty():
            coder.io.output_queue.get()
        
        # Run the coder with the instruction
        logger.debug(f"Running Aider instruction: {instruction}")
        was_successful = coder.run_instruction(instruction)
        
        # Get any output
        output = coder.io.get_accumulated_output()
        
        # Prepare result
        result = {"output": output}
        if not was_successful:
            result["error"] = "Aider instruction failed."
            logger.error(f"Aider instruction failed: {instruction}")
        
        # Put the result in the output queue
        output_q.put(result)
        
    except Exception as e:
        logger.error(f"Error running Aider instruction: {e}", exc_info=True)
        output_q.put({"error": f"Error running Aider instruction: {e}", "output": ""})

# --- Run Aider Command ---
def run_aider_command(aider_json_path: Path, instruction: str, timeout: int = 60) -> Dict[str, str]:
    """Run an Aider command."""
    logger.info(f"Running Aider command: {instruction}")
    state = ensure_active_coder_state(aider_json_path)
    if not state:
        return {"error": "Aider session not active. Try starting the File Editor first.", "output": ""}
    
    coder = state.coder
    output_q = queue.Queue()
    
    # Run in a separate thread to avoid blocking
    thread = threading.Thread(
        target=_run_aider_in_thread, 
        args=(coder, instruction, output_q)
    )
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete
    try:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error(f"Aider command timed out after {timeout} seconds: {instruction}")
            return {"error": f"Command timed out after {timeout} seconds.", "output": ""}
    except Exception as e:
        logger.error(f"Error waiting for Aider command: {e}")
        return {"error": f"Error waiting for Aider command: {e}", "output": ""}
    
    # Check for additional input needed
    if state.io_stub.is_input_needed():
        logger.info("Aider is waiting for input.")
        # Get whatever output is available
        output = state.io_stub.get_accumulated_output()
        return {
            "input_needed": True,
            "output": output
        }
    
    # Get the result from the queue
    try:
        result = output_q.get(block=False)
    except queue.Empty:
        result = {"error": "No result from Aider command.", "output": ""}
    
    # Update persistent state if command was successful
    if "error" not in result:
        try:
            update_aider_state_from_coder(aider_json_path, coder)
        except Exception as e:
            logger.error(f"Failed to update Aider state after command: {e}")
    
    return result

def provide_input_to_aider(aider_json_path: Path, input_text: str) -> Dict[str, str]:
    """Provide input to an active Aider session."""
    logger.info(f"Providing input to Aider: {input_text}")
    state = ensure_active_coder_state(aider_json_path)
    if not state:
        return {"error": "Aider session not active.", "output": ""}
    
    # Check if input is needed
    if not state.io_stub.is_input_needed():
        logger.warning("Providing input to Aider when none was requested.")
    
    # Provide the input
    state.io_stub.provide_input(input_text)
    
    # Get any output that was generated after input
    output = state.io_stub.get_accumulated_output()
    
    # Check if more input is needed
    if state.io_stub.is_input_needed():
        return {
            "input_needed": True,
            "output": output
        }
    
    # Update persistent state
    try:
        update_aider_state_from_coder(aider_json_path, state.coder)
    except Exception as e:
        logger.error(f"Failed to update Aider state after input: {e}")
    
    return {"output": output}