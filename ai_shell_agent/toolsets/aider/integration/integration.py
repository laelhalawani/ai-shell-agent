# ai_shell_agent/toolsets/aider/integration/integration.py
"""
Aider integration for AI Shell Agent.

Handles the connection between AI Shell Agent and the aider-chat library,
including state management and inter-thread communication.
"""

import os
import sys
import queue
import threading
import traceback
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Tuple, Union

# External imports
from aider.coders import Coder
from aider.models import Model
from aider.repo import GitRepo, ANY_GIT_ERROR
from aider.io import InputOutput
from aider.utils import format_content
from aider.commands import Commands

# Local imports
from .... import logger
from ....config_manager import (
    get_current_model, 
    get_api_key_for_model, 
    normalize_model_name, 
    get_aider_edit_format, 
    get_aider_main_model, 
    get_aider_editor_model, 
    get_aider_weak_model
)
from ....chat_state_manager import (
    get_current_chat,
    get_aider_state,
    save_aider_state,
    clear_aider_state,
    get_active_toolsets,
    update_active_toolsets,
    _update_message_in_chat
)

# --- Constants ---
SIGNAL_PROMPT_NEEDED = "[FILE_EDITOR_INPUT_NEEDED]"
TIMEOUT = 10 * 60  # 10 minutes timeout for operations

# --- Custom IO Stub ---
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
            
            # Add main output first
            if self.captured_output:
                output.extend(self.captured_output)
            
            # Add warnings with prefix
            if include_warnings and self.captured_warnings:
                for warn in self.captured_warnings:
                    output.append(f"WARNING: {warn}")
            
            # Add errors with prefix
            if include_errors and self.captured_errors:
                for err in self.captured_errors:
                    output.append(f"ERROR: {err}")
                    
            # Clear buffers
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
            if preference == 'skip':
                return False
            if preference == 'all' and not explicit_yes_required:
                return True

        # 2. Send prompt details to the main thread via output_q
        prompt_data = {
            'type': 'prompt',
            'prompt_type': 'confirm',
            'question': question,
            'default': default,
            'subject': subject,
            'explicit_yes_required': explicit_yes_required,
            'allow_never': allow_never,
            'group_id': group_id  # Include group_id for context
        }
        logger.debug(f"Putting prompt data on output_q: {prompt_data}")
        self.output_q.put(prompt_data)

        # 3. Block and wait for the response from the main thread via input_q
        logger.debug("Waiting for response on input_q...")
        raw_response = self.input_q.get()
        logger.debug(f"Received raw response from input_q: '{raw_response}'")

        # 4. Process the response
        response = str(raw_response).lower().strip()
        result = False  # Default to no

        # Handle 'never'/'don't ask'
        if allow_never and ("never" in response or "don't ask" in response):
            self.never_prompts.add(question_id)
            logger.debug(f"Adding {question_id} to never_prompts")
            return False

        # Handle 'all' option for group
        if group_id:
            if 'all' in response:
                self.group_preferences[group_id] = 'all'
                logger.debug(f"Setting preference 'all' for group {group_id}")
                return True
            elif 'skip' in response:
                self.group_preferences[group_id] = 'skip'
                logger.debug(f"Setting preference 'skip' for group {group_id}")
                return False

        # Handle direct yes/no responses
        if response.startswith('y') or response == '1' or response == 'true' or response == 't':
            result = True
        
        return result

    def prompt_ask(self, question, default="", subject=None):
        """Intercepts prompt_ask, sends prompt data via output_q, waits for input_q."""
        logger.debug(f"Intercepted prompt_ask: {question} (Subject: {subject})")
        if not self.input_q or not self.output_q:
            logger.error("Queues not set for AiderIOStubWithQueues prompt_ask.")
            raise RuntimeError("Aider IO Queues not initialized.")

        prompt_data = {
            'type': 'prompt',
            'prompt_type': 'input',
            'question': question,
            'default': default,
            'subject': subject
        }
        logger.debug(f"Putting prompt data on output_q: {prompt_data}")
        self.output_q.put(prompt_data)

        logger.debug("Waiting for response on input_q...")
        raw_response = self.input_q.get()
        logger.debug(f"Received raw response from input_q: '{raw_response}'")

        # Return the raw response, or default if empty
        return raw_response if raw_response else default

    def get_input(self, *args, **kwargs):
        err = "AiderIOStubWithQueues.get_input() called unexpectedly in tool mode."
        logger.error(err)
        # In tool mode, input comes via RunCodeEditTool or SubmitCodeEditorInputTool
        raise NotImplementedError(err)

    def user_input(self, *args, **kwargs):
        """No-op as AI Shell Agent manages user input."""
        pass

    def ai_output(self, *args, **kwargs):
        """No-op as AI Shell Agent handles AI output."""
        pass

    def append_chat_history(self, *args, **kwargs):
        """No-op as AI Shell Agent manages chat history externally."""
        pass


# --- Active Coder State Management ---
@dataclass
class ActiveCoderState:
    """Holds the live state for an active Aider session."""
    coder: Coder
    io_stub: AiderIOStubWithQueues = field(default_factory=AiderIOStubWithQueues)
    input_q: queue.Queue = field(default_factory=queue.Queue)
    output_q: queue.Queue = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self):
        """Initialize the IO stub with the queues."""
        self.io_stub.set_queues(self.input_q, self.output_q)
        # Also ensure the coder is using this io_stub
        if self.coder.io is not self.io_stub:
            self.coder.io = self.io_stub


# Global dictionary to store active coder states, keyed by chat_file path
active_coders: Dict[str, ActiveCoderState] = {}
_active_coders_dict_lock = threading.Lock()


def get_active_coder_state(chat_file: str) -> Optional[ActiveCoderState]:
    """Gets the active coder state for a chat file if it exists."""
    with _active_coders_dict_lock:
        return active_coders.get(chat_file)


def create_active_coder_state(chat_file: str, coder: Coder) -> ActiveCoderState:
    """Creates and stores a new active coder state."""
    with _active_coders_dict_lock:
        if chat_file in active_coders:
             logger.warning(f"Overwriting existing active coder state for {chat_file}")
             # Potentially add cleanup logic here if needed before overwriting
        # Create the state, which includes creating the IO stub and queues
        state = ActiveCoderState(coder=coder)
        # Associate the coder with the io_stub explicitly if needed by Coder.create/internals
        coder.io = state.io_stub
         # Initialize Commands if not already done by Coder.create
        if not hasattr(coder, 'commands') or coder.commands is None:
            try:
                coder.commands = Commands(io=state.io_stub, coder=coder)
                logger.debug(f"Initialized Commands for coder {chat_file}")
            except Exception as e:
                logger.warning(f"Could not initialize Commands for coder {chat_file}: {e}")

        active_coders[chat_file] = state
        logger.info(f"Created active Aider session for: {chat_file}")
        return state


def remove_active_coder_state(chat_file: str):
    """Removes the active coder state and potentially cleans up resources."""
    with _active_coders_dict_lock:
        state = active_coders.pop(chat_file, None)
        if state:
            logger.info(f"Removed active Aider session for: {chat_file}")
            # Optional: Add logic to signal the thread to stop if it's running,
            # or wait for it if necessary, though letting it finish naturally might be okay.
            # if state.thread and state.thread.is_alive():
            #     logger.debug(f"Waiting for Aider thread to finish for {chat_file}...")
            #     state.thread.join(timeout=5.0) # Add a timeout
            #     if state.thread is_alive():
            #          logger.warning(f"Aider thread for {chat_file} did not terminate cleanly.")
        else:
             logger.debug(f"No active Aider session found to remove for {chat_file}")


def ensure_active_coder_state(chat_file: str) -> Optional[ActiveCoderState]:
    """
    Gets the active coder state, recreating it from persistent state if necessary.
    Returns the state or None if it cannot be obtained/recreated.
    """
    state = get_active_coder_state(chat_file)
    if (state):
        logger.debug(f"Found existing active coder state for {chat_file}")
        return state

    logger.info(f"No active coder state found for {chat_file}, attempting recreation.")
    aider_state_persistent = get_aider_state(chat_file)
    if not aider_state_persistent or not aider_state_persistent.get("enabled", False):
        logger.warning(f"Cannot recreate active state: Persistent Aider state not found or disabled for {chat_file}.")
        return None

    # Recreate the coder instance from persistent state
    # Need a temporary IO stub just for recreation process if coder needs it internally
    temp_io_stub = AiderIOStubWithQueues()
    recreated_coder = recreate_coder(chat_file, temp_io_stub)

    if not recreated_coder:
        logger.error(f"Failed to recreate Coder from persistent state for {chat_file}.")
        return None

    # Create and store the new active state using the recreated coder
    new_state = create_active_coder_state(chat_file, recreated_coder)
    logger.info(f"Successfully recreated and stored active coder state for {chat_file}")
    return new_state


def recreate_coder(chat_file: str, io_stub: AiderIOStubWithQueues) -> Optional[Coder]:
    """
    Recreates the Aider Coder instance from saved persistent state.
    This version is passive and primarily for use within StartCodeEditorTool.
    """
    try:
        # First check if there's an active coder state for this chat file
        active_state = get_active_coder_state(chat_file)
        if (active_state and active_state.coder):
            # If we have an active coder instance, just return it
            # But make sure it uses the provided io_stub
            if active_state.coder.io is not io_stub:
                logger.debug(f"Updating io_stub for existing active coder for {chat_file}")
                active_state.coder.io = io_stub
            logger.debug(f"Using existing active coder for {chat_file}")
            return active_state.coder
            
        # If no active coder state, recreate from persistent state
        aider_state = get_aider_state(chat_file)  # Use chat_state_manager function
        if not aider_state or not aider_state.get("enabled", False):
            logger.debug(f"Aider state not found, empty, or not enabled for {chat_file}")
            return None

        logger.debug(f"Recreating Coder for {chat_file} with state keys: {list(aider_state.keys())}")

        # --- Model and Config Setup ---
        # Priority: Agent Config -> Persistent State -> Agent Current Model
        main_model_name_cfg = get_aider_main_model()
        main_model_name_state = aider_state.get('main_model_name')
        main_model_name_agent = get_current_model()  # Agent's primary model

        main_model_name = main_model_name_cfg or main_model_name_state or main_model_name_agent
        if not main_model_name:
            logger.error("Cannot determine main model name for Aider Coder recreation.")
            return None

        logger.debug(f"Using main model: {main_model_name} (Source: {'Config' if main_model_name_cfg else 'State' if main_model_name_state else 'Agent'})")

        # Other models (Priority: Agent Config -> Persistent State -> None/Default)
        editor_model_name = get_aider_editor_model() or aider_state.get('editor_model_name')
        weak_model_name = get_aider_weak_model() or aider_state.get('weak_model_name')
        logger.debug(f"Using editor model: {editor_model_name or 'Default'}")
        logger.debug(f"Using weak model: {weak_model_name or 'Default'}")

        # Edit format (Priority: Agent Config -> Persistent State -> Model Default)
        edit_format_cfg = get_aider_edit_format()
        edit_format_state = aider_state.get('edit_format')
        # We'll determine the final edit_format after instantiating the model if needed
        editor_edit_format = aider_state.get('editor_edit_format')  # Editor format primarily from state for now

        # Ensure API key for the *determined* main_model_name
        api_key, env_var = get_api_key_for_model(main_model_name)
        if not api_key:
            logger.error(f"API Key ({env_var}) not found for model {main_model_name}. Cannot recreate Coder.")
            return None

        try:
            main_model_instance = Model(
                 main_model_name,
                 weak_model=weak_model_name,  # Pass potentially overridden models
                 editor_model=editor_model_name,
                 editor_edit_format=editor_edit_format  # Pass format from state
            )
            # Determine final edit format
            # If config is set, use it. If not, use state. If neither, use model default.
            edit_format = edit_format_cfg or edit_format_state or main_model_instance.edit_format
            logger.debug(f"Using edit format: {edit_format} (Source: {'Config' if edit_format_cfg else 'State' if edit_format_state else 'Model Default'})")

        except Exception as e:
             logger.error(f"Failed to instantiate main_model '{main_model_name}': {e}")
             return None

        # --- Load Aider History --- 
        aider_done_messages = aider_state.get("aider_done_messages", [])
        if not isinstance(aider_done_messages, list):
             logger.warning("aider_done_messages in state is not a list, using empty list.")
             aider_done_messages = []
        logger.debug(f"Loading {len(aider_done_messages)} messages from aider_state for Coder history.")

        # --- Git Repo Setup ---
        repo = None
        git_root = aider_state.get("git_root")
        abs_fnames = aider_state.get("abs_fnames", [])
        abs_read_only_fnames = aider_state.get("abs_read_only_fnames", [])
        # Use all known files for repo context if available
        fnames_for_repo = abs_fnames + abs_read_only_fnames
        if git_root:
            try:
                # Make sure GitRepo is available
                if 'GitRepo' not in globals():
                    raise ImportError("GitRepo not imported successfully.")

                repo_root_path = Path(git_root)
                if not repo_root_path.is_dir():
                    logger.warning(f"Git root directory specified in state does not exist or is not a directory: {git_root}. Attempting without git.")
                    git_root = None  # Invalidate git_root if path is gone/invalid
                else:
                    # Ensure fnames_for_repo contains absolute paths relative to the *correct* root
                    # It assumes saved paths were absolute or relative to the original root.
                    # Best practice is to save absolute paths.
                    abs_fnames_for_repo = [str(Path(p).resolve()) for p in fnames_for_repo]

                    repo = GitRepo(io=io_stub, fnames=abs_fnames_for_repo, git_dname=str(repo_root_path))
                    # Verify the repo root matches the state to catch potential mismatches
                    if str(Path(repo.root).resolve()) != str(repo_root_path.resolve()):
                         logger.warning(f"Detected Git root '{repo.root}' differs from state '{git_root}'. Using detected root.")
                         # Update git_root for Coder instantiation if needed, or decide policy
                         # For now, we'll trust the detected repo object's root
                         git_root = repo.root  # Use the root from the created repo object

            except ImportError:
                logger.warning("GitPython not installed or GitRepo not available, git features disabled.")
                git_root = None  # Ensure git_root is None if git can't be used
                repo = None
            except ANY_GIT_ERROR as e:
                logger.error(f"Error initializing GitRepo at {git_root}: {e}")
                git_root = None  # Invalidate on error
                repo = None
            except Exception as e:  # Catch other potential errors
                logger.error(f"Unexpected error initializing GitRepo at {git_root}: {e}")
                git_root = None
                repo = None
        else:
            logger.debug("No git_root found in state, proceeding without GitRepo.")

        # --- Prepare Explicit Config for Coder.create ---
        coder_kwargs = dict(
            main_model=main_model_instance,
            edit_format=edit_format,  # Use determined format
            io=io_stub,  # Use the provided stub
            repo=repo,
            fnames=abs_fnames,  # Use absolute paths from state
            read_only_fnames=abs_read_only_fnames,  # Use absolute paths from state
            done_messages=aider_done_messages,  # Use history from state
            cur_messages=[],  # Always start cur_messages fresh for a tool run

            # Pass other relevant settings from state or defaults
            auto_commits=aider_state.get("auto_commits", True),
            dirty_commits=aider_state.get("dirty_commits", True),
            use_git=bool(repo),
            # Set other flags based on state or agent defaults as needed
            map_tokens=aider_state.get("map_tokens", 0),  # Default to 0 if not in state
            verbose=False,  # Keep tool runs non-verbose unless specified
            stream=False,  # Tool runs are typically non-streaming
            suggest_shell_commands=False,  # Safer default for tool usage
        )

        # --- Instantiate Coder ---
        coder = Coder.create(**coder_kwargs)
        coder.root = git_root or os.getcwd()  # Set root explicitly

         # Initialize Commands instance for tools that might need it (like /diff, /undo)
        if not hasattr(coder, 'commands') or coder.commands is None:
            try:
                # Ensure Commands class is available
                if 'Commands' not in globals():
                     raise ImportError("Commands class not imported successfully.")
                coder.commands = Commands(io=io_stub, coder=coder)
                logger.debug(f"Initialized Commands for coder {chat_file} during recreation")
            except ImportError:
                logger.warning("Could not import Commands module from Aider.")
                # Proceed without commands if not critical for all tools
            except Exception as e:
                 logger.warning(f"Could not initialize Commands for coder {chat_file}: {e}")

        # Store the newly recreated coder in active state
        create_active_coder_state(chat_file, coder)
        logger.info(f"Coder successfully recreated for {chat_file} and stored in active state")
        return coder

    except Exception as e:
        logger.error(f"Failed to recreate Coder for {chat_file}: {e}")
        logger.error(traceback.format_exc())
        return None


def update_aider_state_from_coder(chat_file: str, coder: Coder) -> None:
    """Update the persistent Aider state from a Coder instance."""
    try:
        # Get the current state to update it, preserving unrelated fields
        aider_state = get_aider_state(chat_file)
        if not aider_state:
            aider_state = {}  # Initialize if it doesn't exist

        # Ensure 'enabled' flag is set (it should be if we have a coder)
        aider_state["enabled"] = True

        # --- Update fields based on the coder ---
        aider_state["main_model_name"] = coder.main_model.name
        aider_state["edit_format"] = coder.edit_format
        aider_state["weak_model_name"] = getattr(coder.main_model.weak_model, 'name', None)
        aider_state["editor_model_name"] = getattr(coder.main_model.editor_model, 'name', None)
        aider_state["editor_edit_format"] = getattr(coder.main_model, 'editor_edit_format', None)
        aider_state["abs_fnames"] = sorted(list(coder.abs_fnames))
        aider_state["abs_read_only_fnames"] = sorted(list(getattr(coder, "abs_read_only_fnames", [])))

        # Update git info if available
        if hasattr(coder, "repo") and coder.repo:
            aider_state["git_root"] = coder.repo.root
            # Ensure commit hashes are strings and sorted for consistency
            aider_state["aider_commit_hashes"] = sorted(list(map(str, coder.aider_commit_hashes)))
        else:
            # Clear git info if repo is no longer present
            aider_state.pop("git_root", None)
            aider_state.pop("aider_commit_hashes", None)

        # *** Crucially, save Aider's internal conversation history ***
        # Filter out any non-serializable content if necessary, though usually dicts are fine
        try:
             aider_state["aider_done_messages"] = coder.done_messages
             logger.debug(f"Saving {len(coder.done_messages)} messages to aider_done_messages state for {chat_file}.")
        except Exception as e:
             logger.error(f"Failed to serialize aider_done_messages for {chat_file}: {e}")
             # Decide on fallback: save empty list or raise error? Saving empty might be safer.
             aider_state["aider_done_messages"] = []

        # Update other relevant config flags if they were changed
        aider_state["auto_commits"] = getattr(coder, "auto_commits", True)
        aider_state["dirty_commits"] = getattr(coder, "dirty_commits", True)
        # Add other flags as needed, e.g., map_tokens

        # --- Save the updated state ---
        save_aider_state(chat_file, aider_state)

        logger.debug(f"Aider state updated from coder for {chat_file}")

    except Exception as e:
        logger.error(f"Failed to update persistent Aider state for {chat_file}: {e}")
        logger.error(traceback.format_exc())


def _run_aider_in_thread(coder: Coder, instruction: str, output_q: queue.Queue):
    """Target function for the Aider worker thread."""
    thread_name = threading.current_thread().name
    logger.info(f"Aider worker thread '{thread_name}' started for instruction: {instruction[:50]}...")
    try:
        # Ensure the coder uses the stub with the correct queues assigned
        if not isinstance(coder.io, AiderIOStubWithQueues) or coder.io.output_q != output_q:
             logger.error(f"Thread {thread_name}: Coder IO setup is incorrect!")
             raise RuntimeError("Coder IO setup incorrect in thread.")

        # Clear any previous output in the stub before running
        coder.io.get_captured_output()

        # The main blocking call
        coder.run(with_message=instruction)

        # Get accumulated output from the run
        final_output = coder.io.get_captured_output()
        logger.info(f"Thread {thread_name}: coder.run completed successfully.")
        output_q.put({'type': 'result', 'content': final_output})
    except Exception as e:
        logger.error(f"Thread {thread_name}: Exception during coder.run: {e}")
        logger.error(traceback.format_exc())
        # Capture any output accumulated before the error
        error_output = coder.io.get_captured_output()
        error_message = f"Error during edit: {e}\nOutput before error:\n{error_output}"
        output_q.put({'type': 'error', 'message': error_message, 'traceback': traceback.format_exc()})
    finally:
         logger.info(f"Aider worker thread '{thread_name}' finished.")