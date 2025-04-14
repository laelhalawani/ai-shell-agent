"""
File Editor integration for AI Shell Agent.

Handles communication with the Aider library, state management,
and provides Langchain Tools for agent interaction.
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
from prompt_toolkit import prompt
from aider.coders import Coder
from aider.models import Model
from aider.repo import GitRepo, ANY_GIT_ERROR
from aider.io import InputOutput
from aider.utils import format_content # If needed for formatting output, otherwise remove
from aider.commands import Commands


from langchain.tools import BaseTool

# Local imports (relative)
from . import logger
# Import necessary functions from chat_state_manager and config_manager
# These will be used by the tools and helper functions
from .config_manager import get_current_model, get_api_key_for_model, normalize_model_name, get_aider_edit_format, get_aider_main_model, get_aider_editor_model, get_aider_weak_model
# --- Import chat state functions from chat_state_manager instead of chat_manager ---
from .chat_state_manager import (
    get_current_chat,
    get_aider_state,
    save_aider_state,
    clear_aider_state,
    # --- NEW: Imports for toolset/prompt update ---
    get_active_toolsets,
    update_active_toolsets,
    _update_message_in_chat
)
# --- NEW: Import prompt builder ---
from .prompts.prompts import build_prompt
# --- Import the registry function ---
from .tool_registry import register_tools, get_all_tools_dict

# --- Constants ---
SIGNAL_PROMPT_NEEDED = "[FILE_EDITOR_INPUT_NEEDED]"
TIMEOUT = 10 * 60 
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
        self.group_preferences: Dict[int, str] = {} # group_id -> preference ('yes'/'no'/'all'/'skip')
        self.never_prompts: Set[Tuple[str, Optional[str]]] = set() # (question, subject)

        # Buffers for capturing output
        self.captured_output: List[str] = []
        self.captured_errors: List[str] = []
        self.captured_warnings: List[str] = []
        self.tool_output_lock = threading.Lock() # Protect buffer access

    def set_queues(self, input_q: queue.Queue, output_q: queue.Queue):
        """Assign the input and output queues."""
        self.input_q = input_q
        self.output_q = output_q

    def tool_output(self, *messages, log_only=False, bold=False):
        """Capture tool output."""
        msg = " ".join(map(str, messages))
        with self.tool_output_lock:
             self.captured_output.append(msg)
        # logger.debug(f"AiderIOStub Output: {msg}") # Optional: log if needed
        # Also call parent for potential logging if needed by aider internals
        super().tool_output(*messages, log_only=True) # Ensure log_only=True for parent

    def tool_error(self, message="", strip=True):
        """Capture error messages."""
        msg = str(message).strip() if strip else str(message)
        with self.tool_output_lock:
            self.captured_errors.append(msg)
        # logger.error(f"AiderIOStub Error: {msg}") # Optional: log if needed
        super().tool_error(message, strip=strip) # Call parent for potential logging

    def tool_warning(self, message="", strip=True):
        """Capture warning messages."""
        msg = str(message).strip() if strip else str(message)
        with self.tool_output_lock:
            self.captured_warnings.append(msg)
        # logger.warning(f"AiderIOStub Warning: {msg}") # Optional: log if needed
        super().tool_warning(message, strip=strip) # Call parent for potential logging

    def get_captured_output(self, include_warnings=True, include_errors=True) -> str:
        """Returns all captured output, warnings, and errors as a single string and clears buffers."""
        with self.tool_output_lock:
            output = "\n".join(self.captured_output)
            warnings = "\n".join([f"WARNING: {w}" for w in self.captured_warnings]) if include_warnings and self.captured_warnings else ""
            errors = "\n".join([f"ERROR: {e}" for e in self.captured_errors]) if include_errors and self.captured_errors else ""

            full_output = "\n".join(filter(None, [output, warnings, errors]))

            # Clear buffers after getting output
            self.captured_output = []
            self.captured_errors = []
            self.captured_warnings = []
            return full_output.strip()

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
            # If preference is 'yes' or 'no', we still need to ask the LLM/user
            # via the queue, but maybe we can pre-fill the answer?
            # For now, let's just proceed to asking via queue.

        # 2. Send prompt details to the main thread via output_q
        prompt_data = {
            'type': 'prompt',
            'prompt_type': 'confirm',
            'question': question,
            'default': default,
            'subject': subject,
            'explicit_yes_required': explicit_yes_required,
            'allow_never': allow_never,
            'group_id': group_id # Include group_id for context
        }
        logger.debug(f"Putting prompt data on output_q: {prompt_data}")
        self.output_q.put(prompt_data)

        # 3. Block and wait for the response from the main thread via input_q
        logger.debug("Waiting for response on input_q...")
        raw_response = self.input_q.get()
        logger.debug(f"Received raw response from input_q: '{raw_response}'")

        # 4. Process the response
        response = str(raw_response).lower().strip()
        result = False # Default to no

        # Handle 'never'/'don't ask'
        if allow_never and response.startswith('d'):
            self.never_prompts.add(question_id)
            logger.debug(f"Added {question_id} to never_prompts.")
            result = False # 'Don't ask' implies 'no' for this instance
        # Handle group preferences 'all'/'skip'
        elif group_id:
             if response.startswith('a') and not explicit_yes_required:
                 self.group_preferences[group_id] = 'all'
                 logger.debug(f"Set group {group_id} preference to 'all'.")
                 result = True
             elif response.startswith('s'):
                 self.group_preferences[group_id] = 'skip'
                 logger.debug(f"Set group {group_id} preference to 'skip'.")
                 result = False
             elif response.startswith('y'):
                 result = True
             elif response.startswith('n'):
                 result = False
             else: # Fallback to default if response is unclear for group
                 result = default.lower().startswith('y')
        # Handle regular yes/no
        elif response.startswith('y'):
            result = True
        elif response.startswith('n'):
            result = False
        else: # Use default if response is empty or unclear
            result = default.lower().startswith('y')

        # Override based on explicit_yes_required
        if explicit_yes_required and not response.startswith('y'):
            result = False

        logger.debug(f"confirm_ask returning: {result}")
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

    # --- Override other methods to be non-interactive or no-op ---
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
    thread: Optional[threading.Thread] = None
    input_q: queue.Queue = field(default_factory=queue.Queue)
    output_q: queue.Queue = field(default_factory=queue.Queue)
    io_stub: AiderIOStubWithQueues = field(default_factory=AiderIOStubWithQueues)
    lock: threading.Lock = field(default_factory=threading.Lock) # Lock for this specific session

    def __post_init__(self):
         # Ensure the io_stub has the queues associated with this state
         self.io_stub.set_queues(self.input_q, self.output_q)

# Global dictionary to store active coder states, keyed by chat_file path
active_coders: Dict[str, ActiveCoderState] = {}
# Lock to protect access to the active_coders dictionary itself
_active_coders_dict_lock = threading.Lock()

def get_active_coder_state(chat_file: str) -> Optional[ActiveCoderState]:
    """Safely retrieves the active coder state for a given chat file."""
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

# --- Helper function to recreate active state if missing ---
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
        aider_state = get_aider_state(chat_file) # Use chat_state_manager function
        if not aider_state or not aider_state.get("enabled", False):
            logger.debug(f"Aider state not found, empty, or not enabled for {chat_file}")
            return None

        logger.debug(f"Recreating Coder for {chat_file} with state keys: {list(aider_state.keys())}")

        # --- Model and Config Setup ---
        # Priority: Agent Config -> Persistent State -> Agent Current Model
        main_model_name_cfg = get_aider_main_model()
        main_model_name_state = aider_state.get('main_model_name')
        main_model_name_agent = get_current_model() # Agent's primary model

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
        
        editor_edit_format = aider_state.get('editor_edit_format') # Editor format primarily from state for now

        # Ensure API key for the *determined* main_model_name
        api_key, env_var = get_api_key_for_model(main_model_name)
        if not api_key:
            logger.error(f"API Key ({env_var}) not found for model {main_model_name}. Cannot recreate Coder.")
            return None

        try:
            main_model_instance = Model(
                 main_model_name,
                 weak_model=weak_model_name, # Pass potentially overridden models
                 editor_model=editor_model_name,
                 editor_edit_format=editor_edit_format # Pass format from state
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
                    git_root = None # Invalidate git_root if path is gone/invalid
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
                         git_root = repo.root # Use the root from the created repo object

            except ImportError:
                logger.warning("GitPython not installed or GitRepo not available, git features disabled.")
                git_root = None # Ensure git_root is None if git can't be used
                repo = None
            except ANY_GIT_ERROR as e:
                logger.error(f"Error initializing GitRepo at {git_root}: {e}")
                git_root = None # Invalidate on error
                repo = None
            except Exception as e: # Catch other potential errors
                logger.error(f"Unexpected error initializing GitRepo at {git_root}: {e}")
                git_root = None
                repo = None
        else:
            logger.debug("No git_root found in state, proceeding without GitRepo.")


        # --- Prepare Explicit Config for Coder.create ---
        coder_kwargs = dict(
            main_model=main_model_instance,
            edit_format=edit_format, # Use determined format
            io=io_stub, # Use the provided stub
            repo=repo,
            fnames=abs_fnames, # Use absolute paths from state
            read_only_fnames=abs_read_only_fnames, # Use absolute paths from state
            done_messages=aider_done_messages, # Use history from state
            cur_messages=[], # Always start cur_messages fresh for a tool run

            # Pass other relevant settings from state or defaults
            auto_commits=aider_state.get("auto_commits", True),
            dirty_commits=aider_state.get("dirty_commits", True),
            use_git=bool(repo),
            # Set other flags based on state or agent defaults as needed
            # Example: map_tokens, verbose, stream etc. might come from aider_state
            # For now, using sensible defaults for tool usage:
            map_tokens=aider_state.get("map_tokens", 0), # Default to 0 if not in state
            verbose=False, # Keep tool runs non-verbose unless specified
            stream=False, # Tool runs are typically non-streaming
            suggest_shell_commands=False, # Safer default for tool usage
        )

        # --- Instantiate Coder ---
        coder = Coder.create(**coder_kwargs)
        coder.root = git_root or os.getcwd() # Set root explicitly

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
            aider_state = {} # Initialize if it doesn't exist

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

# --- Threading Logic ---
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

# --- Tool Definitions ---
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
            
        # from .chat_manager import get_current_chat, clear_aider_state, save_aider_state # Already imported via state_manager
        # from .config_manager import get_current_model, get_api_key_for_model # Already imported globally
        from .ai import ensure_api_keys_for_coder_models

        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        toolset_name = "File Editor" # The name defined in llm.py

        # --- Toolset Activation & Prompt Update ---
        current_toolsets = get_active_toolsets(chat_file)
        toolsets_updated = False
        if (toolset_name not in current_toolsets):
            logger.info(f"Activating '{toolset_name}' toolset for chat {chat_file}.")
            new_toolsets = list(current_toolsets)
            new_toolsets.append(toolset_name)
            update_active_toolsets(chat_file, new_toolsets) # Save updated toolsets list

            # Re-build the system prompt with the new set of active toolsets
            new_system_prompt = build_prompt(active_toolsets=new_toolsets)

            # Update the system prompt message in the chat history (message 0)
            _update_message_in_chat(chat_file, 0, {"role": "system", "content": new_system_prompt})
            toolsets_updated = True
            logger.info(f"System prompt updated for chat {chat_file} due to toolset activation.")
        # --- End Toolset Activation ---

        # --- Aider Initialization Logic (largely unchanged) ---
        try:
            ensure_api_keys_for_coder_models()
        except Exception as e:
            logger.error(f"Error checking API keys for coder models: {e}")
            # Revert toolset activation if API keys fail? Maybe not, let user fix keys.
            return f"Error: Failed to validate API keys for required models: {e}. '{toolset_name}' toolset was activated but may not function."

        temp_io_stub = AiderIOStubWithQueues()
        coder = None
        resume_attempted = False
        if get_aider_state(chat_file): # Check if state exists before trying to resume
             resume_attempted = True
             try:
                 coder = recreate_coder(chat_file, temp_io_stub)
                 if coder:
                     logger.info(f"Resuming File Editor session for {chat_file} from persistent state.")
                     state = create_active_coder_state(chat_file, coder)
                     update_aider_state_from_coder(chat_file, coder) # Update state after recreation
                     recreation_output = temp_io_stub.get_captured_output()
                     if recreation_output: logger.warning(f"Output during coder recreation: {recreation_output}")

                     activation_message = f"'{toolset_name}' toolset activated. " if toolsets_updated else ""
                     return activation_message + f"Resumed existing File Editor session. Files in context: {', '.join(coder.get_rel_fnames()) or 'None'}. Ready for edits."

             except Exception as e:
                 logger.error(f"Error attempting to resume Aider session: {e}")
                 logger.error(traceback.format_exc())
                 # Fall through to fresh start

        # Start fresh if no state, or resume failed
        if not coder:
            logger.info(f"Starting fresh File Editor session for {chat_file}.")
            clear_aider_state(chat_file) # Clear any invalid old state first

            # Determine initial settings using config overrides
            main_model_name = get_aider_main_model() or get_current_model() # Use agent model as fallback
            editor_model_name = get_aider_editor_model()
            weak_model_name = get_aider_weak_model()
            edit_format = get_aider_edit_format()
            
            if not main_model_name:
                return "Error: Could not determine the main model for the agent or File Editor config."
            
            # Ensure API key for the main model
            api_key, env_var = get_api_key_for_model(main_model_name)
            if not api_key:
                return f"Error: API Key ({env_var}) not found for model {main_model_name}."
            
            # Instantiate the main model to get defaults if edit_format is still None
            final_edit_format = edit_format
            final_editor_edit_format = None # Let Model/Coder handle this initially
            try:
                temp_model = Model(
                    main_model_name,
                    weak_model=weak_model_name,
                    editor_model=editor_model_name
                )
                if final_edit_format is None: 
                    final_edit_format = temp_model.edit_format
                final_editor_edit_format = getattr(temp_model, 'editor_edit_format', None)
            except Exception as e:
                logger.warning(f"Could not get model defaults for {main_model_name}: {e}. Using basic defaults.")
                if final_edit_format is None: final_edit_format = 'whole'

            # Create initial persistent state
            initial_state = {
                "enabled": True, # Mark as enabled
                "main_model_name": main_model_name,
                "edit_format": final_edit_format,
                "weak_model_name": weak_model_name,
                "editor_model_name": editor_model_name,
                "editor_edit_format": final_editor_edit_format,
                "abs_fnames": [],
                "abs_read_only_fnames": [],
                "aider_done_messages": [],
                "aider_commit_hashes": [],
                "git_root": None,
                "auto_commits": True,
                "dirty_commits": True,
            }
            # Save state *before* creating coder to ensure 'enabled' flag is set
            save_aider_state(chat_file, initial_state)

            # Create the Coder instance
            try:
                fresh_io_stub = AiderIOStubWithQueues()
                fresh_main_model = Model(
                    main_model_name,
                    weak_model=weak_model_name,
                    editor_model=editor_model_name,
                    editor_edit_format=final_editor_edit_format
                )
                fresh_coder = Coder.create(
                    main_model=fresh_main_model,
                    edit_format=final_edit_format,
                    io=fresh_io_stub,
                    fnames=[], read_only_fnames=[], done_messages=[], cur_messages=[],
                    auto_commits=True, dirty_commits=True, use_git=True
                )

                if fresh_coder.repo: # If repo found, update state with git_root
                    initial_state["git_root"] = fresh_coder.repo.root
                    save_aider_state(chat_file, initial_state) # Save again with root

                # Create the active state (in-memory)
                create_active_coder_state(chat_file, fresh_coder)
                # Update persistent state *after* creating active state to reflect actual coder
                update_aider_state_from_coder(chat_file, fresh_coder)

                activation_message = f"'{toolset_name}' toolset activated. " if toolsets_updated else ""
                return activation_message + "New File Editor session started. Please add files using 'include_file' before requesting edits."

            except Exception as e:
                logger.error(f"Failed to create new Coder instance for {chat_file}: {e}")
                logger.error(traceback.format_exc())
                clear_aider_state(chat_file) # Clear state on failure
                remove_active_coder_state(chat_file) # Remove active state
                # Should we deactivate the toolset again on failure? Probably not.
                return f"Error: Failed to initialize File Editor. {e}. '{toolset_name}' toolset is active but unusable."

        # This part should ideally not be reached if logic above is correct
        return "Error: Unexpected state in StartAIEditorTool."


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
        io_stub = state.io_stub # Use the IO stub from the active state

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
            coder.add_rel_fname(rel_path) # This modifies coder.abs_fnames internally

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
        from .chat_state_manager import get_current_chat, get_aider_state
        
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
            
        from .chat_state_manager import get_current_chat, get_aider_state
        
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
             message = state.output_q.get(timeout=TIMEOUT) # Add a timeout (e.g., 5 minutes)
             logger.debug(f"Main thread received initial message: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout waiting for initial Aider response ({chat_file}).")
             remove_active_coder_state(chat_file) # Clean up state
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
            with state.lock: # Re-acquire lock briefly
                update_aider_state_from_coder(chat_file, state.coder)
                state.thread = None # Clear the thread reference as it's done
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
                
            remove_active_coder_state(chat_file) # Clean up on error
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
            
        from .chat_state_manager import get_current_chat, get_aider_state
        from aider.repo import ANY_GIT_ERROR
        
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
            
        from .chat_state_manager import get_current_chat, get_aider_state, save_aider_state
        from aider.repo import ANY_GIT_ERROR
        
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
        try:
            clear_aider_state(chat_file) # Mark aider state as disabled
            remove_active_coder_state(chat_file) # Remove active coder instance
            logger.info(f"Aider state cleared and active coder removed for {chat_file}")
        except Exception as e:
            logger.error(f"Error clearing Aider state: {e}")
            logger.error(traceback.format_exc())
            # Continue to deactivate toolset anyway? Yes.

        # --- Deactivate Toolset and Update Prompt ---
        current_toolsets = get_active_toolsets(chat_file)
        toolsets_updated = False
        if toolset_name in current_toolsets:
            logger.info(f"Deactivating '{toolset_name}' toolset for chat {chat_file}.")
            new_toolsets = [ts for ts in current_toolsets if ts != toolset_name]
            update_active_toolsets(chat_file, new_toolsets) # Save updated list

            # Re-build and update system prompt
            new_system_prompt = build_prompt(active_toolsets=new_toolsets)
            _update_message_in_chat(chat_file, 0, {"role": "system", "content": new_system_prompt})
            toolsets_updated = True
            logger.info(f"System prompt updated for chat {chat_file} due to toolset deactivation.")

        if toolsets_updated:
            return f"File Editor session closed and '{toolset_name}' toolset deactivated."
        else:
             # If toolset wasn't active but command was called
             return "File Editor session closed (it might have already been inactive)."

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
             message = state.output_q.get(timeout=TIMEOUT) # Added timeout here too
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
             with state.lock: # Re-acquire lock
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

# --- Tool Instances ---
start_code_editor_tool = StartAIEditorTool()
add_code_file_tool = AddFileTool()
drop_code_file_tool = DropFileTool()
list_code_files_tool = ListFilesInEditorTool()
edit_code_tool = RunCodeEditTool()
submit_code_editor_input_tool = SubmitCodeEditorInputTool()
view_diff_tool = ViewDiffTool()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseCodeEditorTool()

# --- List of Aider tools defined in THIS file ---
aider_tools_in_this_file = [
    start_code_editor_tool,
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    submit_code_editor_input_tool,
    view_diff_tool,
    undo_last_edit_tool,
    close_code_editor_tool,
]

# --- Register the Aider tools at the end of the file ---
register_tools(aider_tools_in_this_file)
logger.debug(f"Registered {len(aider_tools_in_this_file)} Aider tools.")

