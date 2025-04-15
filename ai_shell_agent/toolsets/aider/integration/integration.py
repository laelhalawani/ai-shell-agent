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
import traceback
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
from aider.io import InputOutput
from aider.commands import Commands

# --- I/O Class for Aider Integration ---
class AiderIOStubWithQueues(InputOutput):
    """
    An InputOutput stub for Aider that uses queues for interaction
    and captures output.
    """
    def __init__(self, *args, **kwargs):
        # Initialize with minimal necessary defaults for non-interactive use
        # Ensure 'yes' is True for automatic confirmations where possible internally
        # But external confirmations will be routed via queues.
        super().__init__(pretty=False, yes=True, fancy_input=False)
        self.input_q: Optional[queue.Queue] = None
        self.output_q: Optional[queue.Queue] = None
        self.group_preferences: Dict[int, str] = {}  # group_id -> preference ('yes'/'no'/'all'/'skip')
        self.never_prompts: Set[Tuple[str, Optional[str]]] = set()  # (question, subject)

        # Buffers for capturing output
        self.captured_output: List[str] = []
        self.captured_errors: List[str] = []
        self.captured_warnings: List[str] = []
        self.tool_output_lock = threading.Lock()  # Protect buffer access

    def set_queues(self, input_q: queue.Queue, output_q: queue.Queue):
        """Assign the input and output queues."""
        self.input_q = input_q
        self.output_q = output_q

    def tool_output(self, *messages, log_only=False, bold=False):
        """Capture tool output."""
        msg = " ".join(map(str, messages))
        with self.tool_output_lock:
             self.captured_output.append(msg)
        # Also call parent for potential logging if needed by aider internals
        super().tool_output(*messages, log_only=True)  # Ensure log_only=True for parent

    def tool_error(self, message="", strip=True):
        """Capture error messages."""
        msg = str(message).strip() if strip else str(message)
        with self.tool_output_lock:
            self.captured_errors.append(msg)
        super().tool_error(message, strip=strip)  # Call parent for potential logging

    def tool_warning(self, message="", strip=True):
        """Capture warning messages."""
        msg = str(message).strip() if strip else str(message)
        with self.tool_output_lock:
            self.captured_warnings.append(msg)
        super().tool_warning(message, strip=strip)  # Call parent for potential logging

    def get_captured_output(self, include_warnings=True, include_errors=True) -> str:
        """Returns all captured output, warnings, and errors as a single string and clears buffers."""
        with self.tool_output_lock:
            output = []
            if self.captured_output:
                output.extend(self.captured_output)
            if include_warnings and self.captured_warnings:
                for warn in self.captured_warnings: output.append(f"WARNING: {warn}")
            if include_errors and self.captured_errors:
                for err in self.captured_errors: output.append(f"ERROR: {err}")
            result = "\n".join(output)
            self.captured_output = []
            self.captured_errors = []
            self.captured_warnings = []
            return result

    # --- Intercept Blocking Methods ---
    def confirm_ask(self, question, default="y", subject=None, explicit_yes_required=False, group=None, allow_never=False):
        """Intercepts confirm_ask, sends prompt data via output_q, waits for input_q."""
        logger.debug(f"Intercepted confirm_ask: {question} (Subject: {subject})")
        if not self.input_q or not self.output_q:
            logger.error("Queues not set for AiderIOStubWithQueues confirm_ask.")
            raise RuntimeError("Aider IO Queues not initialized.")

        question_id = (question, subject)
        group_id = id(group) if group else None

        # 1. Check internal state for early exit (never/all/skip)
        if question_id in self.never_prompts:
            logger.debug(f"confirm_ask: Answering 'no' due to 'never_prompts' for {question_id}")
            return False
        if group_id and group_id in self.group_preferences:
            preference = self.group_preferences[group_id]
            logger.debug(f"confirm_ask: Using group preference '{preference}' for group {group_id}")
            if preference == 'skip': return False
            if preference == 'all' and not explicit_yes_required: return True

        # 2. Send prompt details to the main thread via output_q
        prompt_data = {
            'type': 'prompt', 'prompt_type': 'confirm', 'question': question,
            'default': default, 'subject': subject, 'explicit_yes_required': explicit_yes_required,
            'allow_never': allow_never, 'group_id': group_id
        }
        logger.debug(f"Putting prompt data on output_q: {prompt_data}")
        self.output_q.put(prompt_data)

        # 3. Block and wait for the response from the main thread via input_q
        logger.debug("Waiting for response on input_q...")
        raw_response = self.input_q.get() # Blocking get
        logger.debug(f"Received raw response from input_q: '{raw_response}'")

        # 4. Process the response
        response = str(raw_response).lower().strip()
        result = False  # Default to no

        if allow_never and ("never" in response or "don't ask" in response):
            self.never_prompts.add(question_id); logger.debug(f"Adding {question_id} to never_prompts"); return False
        if group_id:
            if 'all' in response: self.group_preferences[group_id] = 'all'; logger.debug(f"Setting preference 'all' for group {group_id}"); return True
            elif 'skip' in response: self.group_preferences[group_id] = 'skip'; logger.debug(f"Setting preference 'skip' for group {group_id}"); return False
        if response.startswith('y') or response == '1' or response == 'true' or response == 't': result = True
        return result

    def prompt_ask(self, question, default="", subject=None):
        """Intercepts prompt_ask, sends prompt data via output_q, waits for input_q."""
        logger.debug(f"Intercepted prompt_ask: {question} (Subject: {subject})")
        if not self.input_q or not self.output_q:
            logger.error("Queues not set for AiderIOStubWithQueues prompt_ask.")
            raise RuntimeError("Aider IO Queues not initialized.")

        prompt_data = {
            'type': 'prompt', 'prompt_type': 'input', 'question': question,
            'default': default, 'subject': subject
        }
        logger.debug(f"Putting prompt data on output_q: {prompt_data}")
        self.output_q.put(prompt_data)

        logger.debug("Waiting for response on input_q...")
        raw_response = self.input_q.get() # Blocking get
        logger.debug(f"Received raw response from input_q: '{raw_response}'")
        return raw_response if raw_response else default

    def get_input(self, *args, **kwargs):
        err = "AiderIOStubWithQueues.get_input() called unexpectedly in tool mode."
        logger.error(err)
        # In tool mode, input comes via RunCodeEditTool or SubmitCodeEditorInputTool -> prompt_ask/confirm_ask
        raise NotImplementedError(err)

    # Keep other original methods for compatibility if needed
    def user_input(self, *args, **kwargs): pass
    def ai_output(self, *args, **kwargs): pass
    def append_chat_history(self, *args, **kwargs): pass
    def rule(self, *args, **kwargs): pass # Add stubs for any other base methods if needed

# --- Runtime State Management ---
class ActiveCoderState:
    """Runtime state for an active Aider coder instance."""
    def __init__(self, coder, io_stub=None, input_q=None, output_q=None):
        self.coder = coder
        self.io_stub = io_stub or AiderIOStubWithQueues()
        self.input_q = input_q or queue.Queue()
        self.output_q = output_q or queue.Queue()
        self.thread = None
        self.lock = threading.RLock()
        self.last_activity = threading.Event()
        self.last_activity.set()  # Mark as active initially
        
    def __post_init__(self):
        """Initialize the IO stub with the queues."""
        if hasattr(self, 'io_stub') and hasattr(self, 'input_q') and hasattr(self, 'output_q'):
            self.io_stub.set_queues(self.input_q, self.output_q)
            # Also ensure the coder is using this io_stub
            if self.coder.io is not self.io_stub:
                self.coder.io = self.io_stub
        
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
    """Creates and stores a new active coder state."""
    with _active_coders_dict_lock:
        if chat_id_or_key in active_coders:
             logger.warning(f"Overwriting existing active coder state for {chat_id_or_key}")
             # Potentially add cleanup logic here if needed before overwriting
        
        # Create the state, which includes creating the IO stub and queues
        io_stub = getattr(coder, "io", None)
        if not isinstance(io_stub, AiderIOStubWithQueues):
            logger.warning(f"Coder has unexpected IO type: {type(io_stub)}")
            io_stub = AiderIOStubWithQueues()
            coder.io = io_stub
            
        state = ActiveCoderState(coder=coder, io_stub=io_stub)
        # Associate the coder with the io_stub explicitly
        coder.io = state.io_stub
        state.io_stub.set_queues(state.input_q, state.output_q)
        
        # Initialize Commands if not already done by Coder.create
        if not hasattr(coder, 'commands') or coder.commands is None:
            try:
                coder.commands = Commands(io=state.io_stub, coder=coder)
                logger.debug(f"Initialized Commands for coder {chat_id_or_key}")
            except Exception as e:
                logger.warning(f"Could not initialize Commands for coder {chat_id_or_key}: {e}")

        active_coders[chat_id_or_key] = state
        logger.info(f"Created active Aider session for: {chat_id_or_key}")
        return state

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
    recreated_coder = recreate_coder(aider_json_path, persistent_state, temp_io_stub)

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

        # Initialize Commands if needed
        if not hasattr(coder, 'commands') or coder.commands is None:
            try:
                coder.commands = Commands(io=io_stub, coder=coder)
                logger.debug(f"Initialized Commands for coder during recreation")
            except Exception as e:
                logger.warning(f"Could not initialize Commands for coder: {e}")

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

# --- Run Aider in Thread ---
def _run_aider_in_thread(coder, instruction: str, output_q: queue.Queue):
    """Run an Aider command in a separate thread."""
    thread_name = threading.current_thread().name
    logger.info(f"Aider worker thread '{thread_name}' started for instruction: {instruction[:50]}...")
    try:
        # Ensure the coder uses the stub with the correct queues assigned
        if not isinstance(coder.io, AiderIOStubWithQueues) or coder.io.output_q != output_q:
             logger.error(f"Thread {thread_name}: Coder IO setup is incorrect!")
             raise RuntimeError("Coder IO setup incorrect in thread.")

        # Clear any previous output in the stub before running
        coder.io.get_captured_output() # Use the restored method
        
        # The main blocking call - use coder.run
        coder.run(with_message=instruction)
        
        # Get accumulated output from the run
        final_output = coder.io.get_captured_output()
        logger.info(f"Thread {thread_name}: coder.run completed successfully.")
        # Put structured result on queue
        output_q.put({'type': 'result', 'content': final_output})
        
    except Exception as e:
        logger.error(f"Thread {thread_name}: Exception during coder.run: {e}", exc_info=True)
        # Capture any output accumulated before the error
        error_output = coder.io.get_captured_output()
        error_message = f"Error during edit: {e}\nOutput before error:\n{error_output}"
        # Put structured error on queue
        output_q.put({'type': 'error', 'message': error_message, 'traceback': traceback.format_exc()})
    finally:
        logger.info(f"Aider worker thread '{thread_name}' finished.")