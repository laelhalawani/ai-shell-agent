import os
import json
import tempfile
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import uuid

from .. import logger
from ..chat_manager import get_current_chat, _read_json, _write_json, get_data_dir, load_session

# Check if Aider is available
try:
    from aider.coders import Coder
    from aider.io import InputOutput
    AIDER_AVAILABLE = True
except ImportError:
    logger.warning("Aider not installed. Aider tools will be disabled.")
    AIDER_AVAILABLE = False

# Directory for storing Aider session states
AIDER_SESSION_DIR = os.path.join(get_data_dir(), "aider_sessions")
os.makedirs(AIDER_SESSION_DIR, exist_ok=True)

@dataclass
class AiderSessionState:
    """Represents the state of an Aider session for a given chat."""
    chat_id: str
    is_active: bool = False
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    selected_files: list = field(default_factory=list)
    chat_history: list = field(default_factory=list)
    coder_instance: Any = None  # Will hold the Coder instance (not serialized)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization, excluding the coder_instance."""
        result = asdict(self)
        result.pop('coder_instance', None)
        return result

class AgentInputOutput(InputOutput):
    """Custom InputOutput class that captures output and bypasses confirmation."""
    
    def __init__(self):
        super().__init__(yes=True)  # Always auto-confirm
        self.captured_output = []
        self.captured_errors = []
    
    def tool_output(self):
        """Returns the captured output and errors as a formatted string."""
        result = "\n".join(self.captured_output)
        if self.captured_errors:
            result += "\n\nErrors:\n" + "\n".join(self.captured_errors)
        return result
    
    def clear_output(self):
        """Clears the captured output and errors."""
        self.captured_output = []
        self.captured_errors = []
    
    def prompt_user(self, prompt):
        """Always returns empty string (no user input during tool execution)."""
        return ""
    
    def confirm_ask(self, prompt):
        """Always confirms. The agent should ask the user before calling tools that need confirmation."""
        logger.debug(f"Bypassed confirmation prompt: {prompt}")
        return True
    
    def prompt_ask(self, *args, **kwargs):
        """Always returns empty. Used for prompting."""
        return ""
    
    def write_captured(self, text, end="\n", skip_output=False):
        """Captures output for the tool's return value."""
        if text and not skip_output:
            self.captured_output.append(text)
    
    def write_error(self, text, end="\n", skip_output=False):
        """Captures errors for the tool's return value."""
        if text and not skip_output:
            self.captured_errors.append(text)
    
    def write(self, text, end="\n", skip_output=False):
        self.write_captured(text, end, skip_output)
    
    def error_write(self, text, end="\n", skip_output=False):
        self.write_error(text, end, skip_output)

class AiderAgentBridge:
    """
    Bridge between AI Shell Agent and Aider.
    Manages Aider sessions, state persistence, and tool execution.
    """
    
    def __init__(self):
        if not AIDER_AVAILABLE:
            logger.warning("Aider not available. Bridge initialized in disabled state.")
        self.input_output = AgentInputOutput()
    
    def _get_current_chat_id(self) -> Optional[str]:
        """Get the current chat ID from the session."""
        chat_file = get_current_chat()
        if not chat_file:
            return None
        # Extract chat ID from filename (assuming format: {id}.json)
        return os.path.basename(chat_file).split('.')[0]
    
    def _get_session_file(self, chat_id: str) -> str:
        """Get the path to the session file for the given chat ID."""
        return os.path.join(AIDER_SESSION_DIR, f"{chat_id}.json")
    
    def _get_session_state(self, chat_id: str) -> AiderSessionState:
        """Get or create a session state for the given chat ID."""
        session_file = self._get_session_file(chat_id)
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                try:
                    data = json.load(f)
                    return AiderSessionState(
                        chat_id=chat_id,
                        is_active=data.get('is_active', False),
                        init_kwargs=data.get('init_kwargs', {}),
                        selected_files=data.get('selected_files', []),
                        chat_history=data.get('chat_history', [])
                    )
                except json.JSONDecodeError:
                    logger.error(f"Failed to load Aider session state for chat {chat_id}. Creating new state.")
        
        # Create new state
        return AiderSessionState(chat_id=chat_id)
    
    def _save_session_state(self, state: AiderSessionState) -> None:
        """Save the session state to disk."""
        session_file = self._get_session_file(state.chat_id)
        with open(session_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
    
    def _ensure_api_key(self) -> None:
        """
        Ensure the API key is set in the environment for Aider to use.
        Uses ai_shell_agent's config_manager to get the current model and API key.
        """
        if not AIDER_AVAILABLE:
            return
        
        from ..config_manager import get_current_model, get_api_key_for_model, get_model_provider
        
        model = get_current_model()
        api_key, env_var_name = get_api_key_for_model(model)
        if api_key:
            os.environ[env_var_name] = api_key
    
    def _update_and_save_state(self, state: AiderSessionState) -> None:
        """Update state from coder instance and save it."""
        if state.coder_instance:
            # Update from coder instance
            coder = state.coder_instance
            state.selected_files = list(coder.abs_fnames)
            # We could also capture chat history here if needed
        
        self._save_session_state(state)
    
    def get_coder(self, chat_id: str) -> Optional[Coder]:
        """
        Get or create a Coder instance for the given chat ID.
        Returns None if the session is not active.
        """
        if not AIDER_AVAILABLE:
            return None
        
        state = self._get_session_state(chat_id)
        if not state.is_active:
            return None
        
        if state.coder_instance:
            return state.coder_instance
        
        # Need to create a new Coder instance
        self._ensure_api_key()
        
        # Create a temporary working dir for the command-line args
        # (Aider will search for .git from this directory)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Default init kwargs if not specified
            kwargs = {
                'model_name': state.init_kwargs.get('model_name', 'gpt-4'),
                'edit_format': state.init_kwargs.get('edit_format', 'diff'),
                'input_output': self.input_output,
                'verbose': False,
                'quiet': True,
                'auto_commits': True,
            }
            
            # Update with any saved init_kwargs
            kwargs.update(state.init_kwargs)
            
            # Ensure input_output is our custom version
            kwargs['input_output'] = self.input_output
            
            try:
                # Create the coder instance
                coder = Coder.create([], **kwargs)
                
                # Restore selected files
                for file in state.selected_files:
                    if os.path.exists(file):
                        coder.add_files([file])
                
                state.coder_instance = coder
                self._save_session_state(state)
                return coder
            except Exception as e:
                logger.error(f"Failed to create Aider Coder instance: {e}")
                return None
    
    def start_editor_session(self, chat_id: str) -> str:
        """
        Start a new editor session for the given chat ID.
        This resets any existing session and initializes with current agent settings.
        """
        if not AIDER_AVAILABLE:
            return "Aider integration is not available. Please install aider-chat."
        
        state = self._get_session_state(chat_id)
        
        # Reset state but preserve files
        selected_files = state.selected_files
        state = AiderSessionState(chat_id=chat_id)
        state.selected_files = selected_files
        state.is_active = True
        
        # Set init kwargs based on current agent settings
        from ..config_manager import get_current_model, get_model_provider
        
        model = get_current_model()
        provider = get_model_provider(model)
        
        # Map provider to Aider-compatible model
        if provider == "openai":
            aider_model = model  # OpenAI models can be used directly
        else:
            # Default to GPT-4 if using a non-OpenAI model (Aider primarily supports OpenAI)
            aider_model = "gpt-4"
            logger.warning(f"Aider may not fully support {provider} models. Using {aider_model} instead.")
        
        state.init_kwargs = {
            'model_name': aider_model,
            'edit_format': 'diff',  # Default to diff format
            'auto_commits': True,   # Auto-commit changes
        }
        
        self._save_session_state(state)
        return f"File editor session started. Use SelectFiles to add files to edit."
    
    def close_editor_session(self, chat_id: str) -> str:
        """
        Close the editor session for the given chat ID.
        This preserves the state but marks it as inactive.
        """
        if not AIDER_AVAILABLE:
            return "Aider integration is not available."
        
        state = self._get_session_state(chat_id)
        if not state.is_active:
            return "No active editor session to close."
        
        # Mark inactive but preserve state
        state.is_active = False
        state.coder_instance = None
        self._save_session_state(state)
        
        return "File editor session closed. Files and history preserved."
    
    def _execute_coder_method(self, method_name: str, **kwargs) -> Tuple[bool, str]:
        """
        Execute a method on the Coder instance for the current chat.
        
        Args:
            method_name: Name of the method to call
            **kwargs: Arguments to pass to the method
        
        Returns:
            Tuple of (success, output)
        """
        if not AIDER_AVAILABLE:
            return False, "Aider integration is not available."
        
        chat_id = self._get_current_chat_id()
        if not chat_id:
            return False, "No active chat session."
        
        coder = self.get_coder(chat_id)
        if not coder:
            return False, "File Editor not active. Use StartFileEditor to begin editing files."
        
        state = self._get_session_state(chat_id)
        
        # Clear previous output
        self.input_output.clear_output()
        
        try:
            # Get and call the method
            method = getattr(coder, method_name, None)
            if not method:
                return False, f"Method {method_name} not found on Coder."
            
            result = method(**kwargs)
            output = self.input_output.tool_output()
            
            return True, output or str(result)
        except Exception as e:
            logger.error(f"Error executing Coder method {method_name}: {e}")
            return False, f"Error: {str(e)}"
        finally:
            self._update_and_save_state(state)
    
    def _execute_command_method(self, method_name: str, args_str: str) -> Tuple[bool, str]:
        """
        Execute a command method on the Coder instance for the current chat.
        These are methods that handle specific commands like /add, /drop, etc.
        
        Args:
            method_name: Name of the command method to call
            args_str: String of arguments to pass to the method
        
        Returns:
            Tuple of (success, output)
        """
        if not AIDER_AVAILABLE:
            return False, "Aider integration is not available."
        
        chat_id = self._get_current_chat_id()
        if not chat_id:
            return False, "No active chat session."
        
        coder = self.get_coder(chat_id)
        if not coder:
            return False, "File Editor not active. Use StartFileEditor to begin editing files."
        
        state = self._get_session_state(chat_id)
        
        # Clear previous output
        self.input_output.clear_output()
        
        try:
            # Get and call the method
            method = getattr(coder, method_name, None)
            if not method:
                return False, f"Method {method_name} not found on Coder."
            
            # Parse args using Aider's own logic if it's a complex command
            result = method(args_str)
            output = self.input_output.tool_output()
            
            return True, output or str(result)
        except Exception as e:
            logger.error(f"Error executing Coder command method {method_name}: {e}")
            return False, f"Error: {str(e)}"
        finally:
            self._update_and_save_state(state)
    
    def clear_history(self) -> str:
        """Clear the chat history for the current chat session."""
        chat_id = self._get_current_chat_id()
        if not chat_id:
            return "No active chat session."
        
        state = self._get_session_state(chat_id)
        if not state.is_active:
            return "No active editor session. Use StartFileEditor first."
        
        # Clear history
        state.chat_history = []
        if state.coder_instance:
            state.coder_instance.history = []
        
        self._save_session_state(state)
        return "Editor chat history cleared."
    
    def reset_session(self) -> str:
        """
        Reset the editor session for the current chat.
        This closes and then starts a new session.
        """
        chat_id = self._get_current_chat_id()
        if not chat_id:
            return "No active chat session."
        
        self.close_editor_session(chat_id)
        return self.start_editor_session(chat_id)

# Create a singleton instance
aider_bridge = AiderAgentBridge()
