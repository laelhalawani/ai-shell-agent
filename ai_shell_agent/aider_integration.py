"""
Aider integration for AI Shell Agent.

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

# Aider imports (ensure aider-chat is installed)
try:
    from aider.coders import Coder
    from aider.models import Model, models # Import the 'models' module itself for sanity_check_models
    from aider.repo import GitRepo, ANY_GIT_ERROR
    from aider.io import InputOutput
    from aider.utils import format_content # If needed for formatting output, otherwise remove
    from aider.commands import Commands
except ImportError as e:
    raise ImportError(
        "Aider package not found or incomplete. Please install it:"
        " pip install aider-chat"
    ) from e

# Langchain import for BaseTool
try:
    from langchain.tools import BaseTool
except ImportError:
     # Fallback for older langchain/langchain-core versions if needed
     from langchain_core.tools import BaseTool


# Local imports (relative)
from . import logger
# Import necessary functions from chat_manager and config_manager
# These will be used by the tools and helper functions
from .config_manager import get_current_model, get_api_key_for_model, normalize_model_name
from .chat_manager import (
    get_current_chat,
    get_aider_state,
    save_aider_state,
    clear_aider_state
)

# --- Constants ---
SIGNAL_PROMPT_NEEDED = "[CODE_EDITOR_INPUT_NEEDED]"

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
             logger.debug(f"No active Aider session found to remove for: {chat_file}")

# --- Aider Coder Recreation/Update ---
def recreate_coder(chat_file: str, io_stub: AiderIOStubWithQueues) -> Optional[Coder]:
    """
    Recreates the Aider Coder instance from saved persistent state.
    This version is passive and primarily for use within StartCodeEditorTool.
    """
    try:
        aider_state = get_aider_state(chat_file) # Use chat_manager function
        if not aider_state or not aider_state.get("enabled", False):
            logger.debug(f"Aider state not found, empty, or not enabled for {chat_file}")
            return None

        logger.debug(f"Recreating Coder for {chat_file} with state keys: {list(aider_state.keys())}")

        # --- Model and Config Setup ---
        # Prioritize state, then agent's current, then default
        main_model_name = aider_state.get('main_model_name')
        if not main_model_name:
            main_model_name = get_current_model() # Agent's current model
            logger.warning(f"Aider main_model_name not in state, using agent's current: {main_model_name}")
        if not main_model_name: # Should not happen if agent setup is correct
             logger.error("Cannot determine main model name for Aider Coder recreation.")
             return None

        weak_model_name = aider_state.get('weak_model_name')
        editor_model_name = aider_state.get('editor_model_name')
        # Ensure edit format comes from state or model default
        edit_format = aider_state.get('edit_format')
        editor_edit_format = aider_state.get('editor_edit_format')

        # Ensure API keys are set in the environment (Coder relies on this)
        api_key, env_var = get_api_key_for_model(main_model_name)
        if not api_key:
            logger.error(f"API Key ({env_var}) not found for model {main_model_name}. Cannot recreate Coder.")
            return None # Cannot proceed without API key

        # Instantiate the main model object
        # This might fail if the model name is invalid or requires specific setup not handled here
        try:
             main_model = Model(
                 main_model_name,
                 weak_model=weak_model_name,
                 editor_model=editor_model_name,
                 editor_edit_format=editor_edit_format
             )
             # If edit_format wasn't in state, get it from the instantiated model
             if edit_format is None:
                  edit_format = main_model.edit_format
                  logger.debug(f"Edit format not in state, using model default: {edit_format}")

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
            main_model=main_model,
            edit_format=edit_format,
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


        logger.info(f"Coder successfully recreated for {chat_file}")
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
        error_message = f"Error during code edit: {e}\nOutput before error:\n{error_output}"
        output_q.put({'type': 'error', 'message': error_message, 'traceback': traceback.format_exc()})
    finally:
         logger.info(f"Aider worker thread '{thread_name}' finished.")

# --- Tool Definitions ---
class StartCodeEditorTool(BaseTool):
    name = "start_code_editor"
    description = "Initializes or resets the code editing session for the current chat. Must be called before adding files or requesting edits."
    
    def _run(self, args: str = "") -> str:
        """Initializes the Aider state for the current chat."""
        from .chat_manager import get_current_chat, clear_aider_state, save_aider_state
        from .config_manager import get_current_model
        # Import Model and GitRepo locally if needed for default determination
        from aider.models import Model
        try:
            from aider.repo import GitRepo
        except ImportError:
            GitRepo = None  # Define dummy if missing
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        clear_aider_state(chat_file)  # Start fresh
        
        # Determine initial settings based on agent's current model
        current_main_model_name = get_current_model()
        try:
            # Temporarily instantiate model to get defaults
            temp_model = Model(current_main_model_name)
            default_edit_format = temp_model.edit_format
            # Handle potential AttributeError if editor_model is None
            default_editor_model_name = getattr(temp_model.editor_model, 'name', None) if hasattr(temp_model, 'editor_model') and temp_model.editor_model else None
            default_editor_edit_format = getattr(temp_model, 'editor_edit_format', None)
        except Exception as e:
            logger.warning(f"Could not get defaults for {current_main_model_name}: {e}. Using 'whole'.")
            default_edit_format = 'whole'
            default_editor_model_name = None
            default_editor_edit_format = None
        
        # Save comprehensive initial state with all necessary settings
        initial_state = {
            "enabled": True,
            "main_model_name": current_main_model_name,
            "edit_format": default_edit_format,
            "editor_model_name": default_editor_model_name,
            "editor_edit_format": default_editor_edit_format,
            "abs_fnames": [],
            "abs_read_only_fnames": [],
            "aider_done_messages": [],  # Initialize aider history
            "aider_commit_hashes": [],
            "git_root": None,  # Will be determined/set later if git is used
            "auto_commits": True,  # Default setting
            "dirty_commits": True,  # Explicitly match Coder default
        }
        
        # Attempt to find git root automatically
        try:
            if GitRepo:  # Check if GitRepo was imported successfully
                io_stub = AiderIOStubWithQueues()
                repo = GitRepo(io=io_stub, fnames=[], git_dname=os.getcwd())
                initial_state["git_root"] = repo.root
                logger.debug(f"Detected Git root: {repo.root}")
        except Exception as e:
            logger.debug(f"No Git repository found or error detecting it: {e}")
            pass  # Keep git_root as None if no repo found
            
        save_aider_state(chat_file, initial_state)
        return f"Code editor initialized (model: {current_main_model_name}, format: {default_edit_format}). Ready for files and edits."
        
    async def _arun(self, args: str = "") -> str:
        return self._run(args)

class AddFileTool(BaseTool):
    name = "add_code_file"
    description = "Adds a file to the code editing session context. Argument must be the relative or absolute file path."
    
    def _run(self, file_path: str) -> str:
        """Adds a file to the Aider context."""
        from .chat_manager import get_current_chat, get_aider_state, save_aider_state
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Code editor not initialized or is closed. Use start_code_editor first."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate code editor state. {io_stub.get_captured_output()}"
            
        try:
            abs_path_to_add = str(Path(file_path).resolve())
            
            # *** ADDED CHECKS ***
            if not os.path.exists(abs_path_to_add):
                return f"Error: File '{file_path}' (resolved to '{abs_path_to_add}') does not exist."
                
            if coder.repo and coder.root != os.getcwd():  # Check only if repo exists and isn't just cwd
                # Use Path.is_relative_to for robust checking if available (Python 3.9+)
                try:
                    if hasattr(Path, 'is_relative_to'):
                        if not Path(abs_path_to_add).is_relative_to(Path(coder.root)):
                            return f"Error: Cannot add file '{file_path}' as it is outside the project root '{coder.root}' when using git."
                    else:
                        # Fallback for Python < 3.9
                        rel_path = os.path.relpath(abs_path_to_add, coder.root)
                        if rel_path.startswith('..'):
                            return f"Error: Cannot add file '{file_path}' as it is outside the project root '{coder.root}' when using git."
                except ValueError:
                    # Handle case where files are on different drives in Windows
                    return f"Error: Cannot add file '{file_path}' as it is on a different drive than the project root '{coder.root}'."
            # *** END CHECKS ***
            
            # Get relative path for better output
            rel_path = coder.get_rel_fname(abs_path_to_add)
            # Now safe to add
            coder.add_rel_fname(rel_path)
            
            # Update state - Coder modifies its own abs_fnames set
            update_aider_state_from_coder(chat_file, coder)
            
            return f"Successfully added {rel_path}. {io_stub.get_captured_output()}"
            
        except Exception as e:
            logger.error(f"Error in AddFileTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error adding file {file_path}: {e}. {io_stub.get_captured_output()}"
            
    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)

class DropFileTool(BaseTool):
    name = "drop_code_file"
    description = "Removes a file from the code editing session context. Argument must be the relative or absolute file path that was previously added."
    
    def _run(self, file_path: str) -> str:
        """Removes a file from the Aider context."""
        from .chat_manager import get_current_chat, get_aider_state
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Code editor not initialized."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate code editor state. {io_stub.get_captured_output()}"
            
        try:
            # Coder's drop_rel_fname expects relative path
            abs_path_to_drop = str(Path(file_path).resolve())
            rel_path_to_drop = coder.get_rel_fname(abs_path_to_drop)
            
            success = coder.drop_rel_fname(rel_path_to_drop)
            
            if success:
                # Update state
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
    name = "list_code_files"
    description = "Lists all files currently in the code editing session context."
    
    def _run(self, args: str = "") -> str:
        """Lists all files in the Aider context."""
        from .chat_manager import get_current_chat, get_aider_state
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Code editor not initialized. Use start_code_editor first."
            
        try:
            files = aider_state.get("abs_fnames", [])
            if not files:
                return "No files are currently added to the code editing session."
                
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
                    
            return "Files in code editor:\n" + "\n".join(f"- {f}" for f in sorted(files_list))
            
        except Exception as e:
            logger.error(f"Error in ListFilesInEditorTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error listing files: {e}"
            
    async def _arun(self, args: str = "") -> str:
        return self._run(args)

class RunCodeEditTool(BaseTool):
    name = "edit_code"
    description = "Requests code changes based on the provided natural language instruction. Edits files currently in the session context."
    
    def _run(self, instruction: str) -> str:
        """Runs Aider's main edit loop with the given instruction, handling architect mode if needed."""
        from .chat_manager import get_current_chat, get_aider_state, save_aider_state
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Code editor not initialized. Use start_code_editor first."
            
        if not aider_state.get("abs_fnames", []):
            return "Error: No files have been added to the code editing session. Use add_code_file first."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate code editor state. {io_stub.get_captured_output()}"
        
        edit_format = coder.edit_format  # Get format from recreated coder (which got it from state)
        final_coder_for_state_saving = coder  # By default, save state from the initial coder
        
        try:
            if edit_format == "architect":
                logger.info("Running in Architect mode...")
                # Step 1: Run the architect model
                # The initial `coder` is already configured as the architect
                coder.run(with_message=instruction)
                architect_plan = coder.partial_response_content  # Get the plan
                logger.debug(f"Architect plan:\n{architect_plan}")
                
                # Check if an editor model is configured
                editor_model_name = aider_state.get("editor_model_name")
                editor_edit_format = aider_state.get("editor_edit_format") 
                
                if not editor_model_name or not editor_edit_format or not architect_plan:
                    logger.warning("Architect mode specified, but no editor model/format configured or no plan generated. No edits applied.")
                    # Still save the architect's run history
                    aider_state["aider_done_messages"] = coder.done_messages
                    save_aider_state(chat_file, aider_state)
                    return f"Architect proposed a plan (no editor configured or plan empty):\n{architect_plan}\n{io_stub.get_captured_output()}"
                
                # Step 2: Create and run the editor model
                logger.info(f"Invoking editor model ({editor_model_name}, format: {editor_edit_format})...")
                # We need a *new* io_stub for the editor run to capture its specific output
                editor_io_stub = AiderIOStubWithQueues()
                
                # Create editor model
                try:
                    editor_model = Model(editor_model_name)
                except Exception as e:
                    logger.error(f"Failed to create editor model: {e}")
                    # Fallback to saving architect state
                    aider_state["aider_done_messages"] = coder.done_messages
                    save_aider_state(chat_file, aider_state)
                    return f"Failed to create editor model: {e}. Architect plan:\n{architect_plan}\n{io_stub.get_captured_output()}"
                
                # Create the editor coder - it needs the same file context but NOT the architect's chat history
                try:
                    editor_coder = Coder.create(
                        main_model=editor_model,  # Use the designated editor model
                        edit_format=editor_edit_format,
                        io=editor_io_stub,
                        repo=coder.repo,  # Share the repo object
                        fnames=list(coder.abs_fnames),  # Pass copies of file lists
                        read_only_fnames=list(getattr(coder, "abs_read_only_fnames", [])),
                        # Start with empty history for the editor task
                        done_messages=[],
                        cur_messages=[],
                        auto_commits=coder.auto_commits,  # Inherit settings
                        aider_commit_hashes=set(coder.aider_commit_hashes),  # Share commit hashes
                        verbose=False
                    )
                    editor_coder.root = coder.root  # Ensure root is set
                    
                    # Run the editor coder with the architect's plan
                    editor_coder.run(with_message=architect_plan, preproc=False)
                    
                    # The editor coder made the actual commits and file changes
                    final_coder_for_state_saving = editor_coder
                    # Combine outputs - architect's reasoning + editor's actions
                    combined_output = f"Architect Plan:\n{architect_plan}\n\nEditor Actions:\n{editor_io_stub.get_captured_output()}"
                    
                except Exception as e:
                    logger.error(f"Error running editor model: {e}")
                    # Still save the architect state on editor failure
                    aider_state["aider_done_messages"] = coder.done_messages
                    save_aider_state(chat_file, aider_state)
                    return f"Error running editor model: {e}. Architect plan:\n{architect_plan}\n{io_stub.get_captured_output()}"
                
            else:  # Standard (non-architect) mode
                logger.info(f"Running in standard mode (format: {edit_format})...")
                coder.run(with_message=instruction)
                combined_output = io_stub.get_captured_output()
            
            # --- State Synchronization ---
            # Use the state from the coder that performed the final actions/commits
            aider_state["aider_commit_hashes"] = list(final_coder_for_state_saving.aider_commit_hashes)
            aider_state["abs_fnames"] = list(final_coder_for_state_saving.abs_fnames)
            aider_state["abs_read_only_fnames"] = list(getattr(final_coder_for_state_saving, "abs_read_only_fnames", []))
            
            # *** Save Aider's potentially summarized history ***
            # In architect mode, save the *architect's* history as it contains the user instruction + plan
            # In standard mode, save the main coder's history
            aider_state["aider_done_messages"] = coder.done_messages
            logger.debug(f"Saving {len(coder.done_messages)} messages to aider_done_messages state.")
            
            save_aider_state(chat_file, aider_state)
            
            return f"Edit request processed. Output:\n{combined_output}"
            
        except Exception as e:
            logger.error(f"Error during code edit: {e}")
            logger.error(traceback.format_exc())
            
            # Try saving partial state (commit hashes might be important)
            try:
                aider_state["aider_commit_hashes"] = list(final_coder_for_state_saving.aider_commit_hashes)
                aider_state["aider_done_messages"] = coder.done_messages  # Save history even on error
                save_aider_state(chat_file, aider_state)
            except Exception as save_err:
                logger.error(f"Failed to save partial state after error: {save_err}")
                
            # Combine outputs from both stubs if architect mode failed mid-way
            combined_error_output = io_stub.get_captured_output()
            if 'editor_io_stub' in locals() and edit_format == "architect":
                combined_error_output += f"\nEditor Output (before error):\n{editor_io_stub.get_captured_output()}"
                
            return f"Error processing edit instruction: {e}. Captured output:\n{combined_error_output}"
            
    async def _arun(self, instruction: str) -> str:
        return self._run(instruction)

class ViewDiffTool(BaseTool):
    name = "view_code_diff"
    description = "Shows the git diff of changes made by the 'edit_code' tool since the start of the last edit request."
    
    def _run(self, args: str = "") -> str:
        """Shows the diff of changes made by Aider."""
        from .chat_manager import get_current_chat, get_aider_state
        from aider.repo import ANY_GIT_ERROR
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Code editor not initialized or is closed. Use start_code_editor first."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate code editor state. {io_stub.get_captured_output()}"
            
        try:
            # If there's no git repo, we can't show diffs
            if not coder.repo:
                return "Error: Git repository not available. Cannot show diff."
                
            # *** MODIFIED: Use commands object if available ***
            if hasattr(coder, 'commands') and coder.commands:
                coder.commands.raw_cmd_diff("")  # Pass empty args for default diff behavior
                captured = io_stub.get_captured_output()
                return f"Diff output:\n{captured}" if captured else "No changes detected in tracked files."
            else:
                # Fallback to direct repo method if commands not available
                diff = coder.repo.get_unstaged_changes()
                
                if not diff:
                    return "No changes detected in the tracked files."
                    
                return f"Changes in code files:\n\n{diff}"
            
        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during diff: {e}")
            logger.error(traceback.format_exc())
            return f"Error viewing diff: {e}. Captured output:\n{io_stub.get_captured_output()}"
            
        except Exception as e:
            logger.error(f"Error in ViewDiffTool: {e}")
            logger.error(traceback.format_exc())
            return f"Error viewing diff: {e}. {io_stub.get_captured_output()}"
            
    async def _arun(self, args: str = "") -> str:
        return self._run(args)

class UndoLastEditTool(BaseTool):
    name = "undo_last_code_edit"
    description = "Undoes the last code edit commit made by the 'edit_code' tool in the current session, if possible."
    
    def _run(self, args: str = "") -> str:
        """Undoes the last code edit commit made by Aider."""
        from .chat_manager import get_current_chat, get_aider_state
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
            
        aider_state = get_aider_state(chat_file)
        if not aider_state or not aider_state.get("enabled", False):
            return "Error: Code editor not initialized. Use start_code_editor first."
            
        io_stub = AiderIOStubWithQueues()
        coder = recreate_coder(chat_file, io_stub)
        if not coder:
            return f"Error: Failed to recreate code editor state. {io_stub.get_captured_output()}"
            
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
            from .chat_manager import save_aider_state
            save_aider_state(chat_file, aider_state)
            
            return f"Undo attempt finished. Output:\n{io_stub.get_captured_output()}"
            
        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during undo: {e}")
            logger.error(traceback.format_exc())
            return f"Error during undo: {e}. Captured output:\n{io_stub.get_captured_output()}"
            
        except Exception as e:
            logger.error(f"Unexpected error during undo: {e}")
            logger.error(traceback.format_exc())
            return f"Error during undo: {e}. Captured output:\n{io_stub.get_captured_output()}"
            
    async def _arun(self, args: str = "") -> str:
        return self._run(args)

class CloseCodeEditorTool(BaseTool):
    name = "close_code_editor"
    description = "Closes the code editor session for the current chat, clearing its specific context like added files and edit history. Call this when code editing tasks are fully complete."

    def _run(self, args: str = "") -> str:
        """Clears the Aider state for the current chat."""
        from .chat_manager import get_current_chat, get_aider_state, clear_aider_state
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found to close the editor for."

        # Check if state exists before clearing, though clear_aider_state is safe anyway
        aider_state = get_aider_state(chat_file)
        if not aider_state:
             # Even if no state was found, confirm closure conceptually
             logger.info("No active Aider state found, but confirming editor closure.")
             return "Code editor context was already clear or not initialized."

        try:
            clear_aider_state(chat_file)
            return "Code editor context cleared successfully."
        except Exception as e:
            logger.error(f"Error clearing Aider state: {e}")
            logger.error(traceback.format_exc())
            return f"Error closing code editor: {e}"

    async def _arun(self, args: str = "") -> str:
        # Simple enough to run synchronously
        return self._run(args)

class SubmitCodeEditorInputTool(BaseTool):
    name = "submit_code_editor_input"
    description = (
         "Provides the required input (e.g., 'yes', 'no', 'all', 'skip', 'don't ask', or text) "
         "when the code editor signals '[CODE_EDITOR_INPUT_NEEDED]'."
         " Sends the input back to the ongoing Aider process."
         " Will return the next prompt, the final result, or an error."
    )

    def _run(self, user_response: str) -> str:
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        state = get_active_coder_state(chat_file)
        if not state:
            # This indicates an agent logic error - trying to submit input when no session active
            return "Error: No active code editor session to submit input to."

        # Acquire lock for the specific session
        with state.lock:
             # Check if the thread is actually running and waiting
             if not state.thread or not state.thread.is_alive():
                  # Could happen if thread finished unexpectedly or was closed
                  # Clean up just in case
                  remove_active_coder_state(chat_file)
                  return "Error: The code editing process is not waiting for input."

             # Send the raw user response to the waiting Aider thread
             logger.debug(f"Putting user response on input_q: '{user_response}' for {chat_file}")
             state.input_q.put(user_response)

        # Release lock before waiting on queue
        logger.debug(f"Main thread waiting for *next* message from output_q for {chat_file}...")
        try:
            # Wait for the Aider thread's *next* action (could be another prompt, result, or error)
             message = state.output_q.get() # Blocks here
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

            response_guidance = f"Aider requires further input. Please respond using 'submit_code_editor_input'. Prompt: '{question}'"
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
             return f"Edit completed. Output:\n{message.get('content', 'No output captured.')}"

        elif message_type == 'error':
             logger.error(f"Aider edit failed for {chat_file} after input.")
             error_content = message.get('message', 'Unknown error')
             remove_active_coder_state(chat_file)
             return f"Error during edit:\n{error_content}"
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             remove_active_coder_state(chat_file)
             return f"Error: Unknown response type '{message_type}' from Aider process."

    async def _arun(self, user_response: str) -> str:
        # Consider running sync in threadpool if needed
        return self._run(user_response)

# Create tool instances for export
start_code_editor_tool = StartCodeEditorTool()
add_code_file_tool = AddFileTool()
drop_code_file_tool = DropFileTool()
list_code_files_tool = ListFilesInEditorTool()
edit_code_tool = RunCodeEditTool()
view_diff_tool = ViewDiffTool()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseCodeEditorTool()
submit_code_editor_input_tool = SubmitCodeEditorInputTool()

