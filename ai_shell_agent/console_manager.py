# ai_shell_agent/console_manager.py
"""
Manages all console input and output using Rich and prompt_toolkit,
ensuring clean state transitions without using rich.Live.
"""
import sys
import io  # Added for StringIO capture
import threading
from threading import Lock
from typing import Dict, Optional, Any, List, Tuple

# Rich imports
from rich.console import Console
# REMOVED: from rich.live import Live
# REMOVED: from rich.spinner import Spinner
from rich.style import Style as RichStyle
from rich.text import Text
from rich.markup import escape
from rich.columns import Columns # Keep for potential future use
from rich.panel import Panel # Keep for potential future use

# Prompt Toolkit imports
from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style as PTKStyle

# Local imports
from . import logger
from .errors import PromptNeededError

# --- Define color constants with semantic names ---
AI_COLOR = "#B19CD9"           # Pastel purple for AI responses
USER_COLOR = "#FFAA99"         # Pastel tuna pink/orange for user interaction
INFO_COLOR = "#8FD9A8"         # Bleached pale warm green for informational messages
WARNING_COLOR = "#CCAA00"      # Yellow/amber for warnings (unchanged)
ERROR_COLOR = "#FF0000"        # Red for errors (unchanged)
SYSTEM_COLOR = "#7FDBCA"       # Pastel turquoise for system messages
TOOL_COLOR = "#FFC0CB"         # Pastel pink for tool-related items
COMMAND_COLOR = "#FFAA99"      # Same as USER_COLOR (pastel tuna pink/orange)
DIM_TEXT_COLOR = "#888888"     # Mid-gray for less important text (unchanged)
NEUTRAL_COLOR = "#FFFFFF"      # White/neutral for default text (unchanged)

# --- Rich Styles using semantic color constants ---
STYLE_AI_LABEL = RichStyle(color=AI_COLOR, bold=True)
STYLE_AI_CONTENT = RichStyle(color=AI_COLOR)
STYLE_USER_LABEL = RichStyle(color=USER_COLOR, bold=True)
STYLE_INFO_LABEL = RichStyle(color=INFO_COLOR, bold=True)
STYLE_INFO_CONTENT = RichStyle(color=INFO_COLOR)
STYLE_WARNING_LABEL = RichStyle(color=WARNING_COLOR, bold=True)
STYLE_WARNING_CONTENT = RichStyle(color=WARNING_COLOR)
STYLE_ERROR_LABEL = RichStyle(color=ERROR_COLOR, bold=True)
STYLE_ERROR_CONTENT = RichStyle(color=ERROR_COLOR)
STYLE_SYSTEM_LABEL = RichStyle(color=SYSTEM_COLOR, bold=True)
STYLE_SYSTEM_CONTENT = RichStyle(color=SYSTEM_COLOR)
STYLE_TOOL_NAME = RichStyle(color=TOOL_COLOR, bold=False)  # Now using defined tool color
STYLE_ARG_NAME = RichStyle(color=DIM_TEXT_COLOR)  # Use explicit color instead of dim
STYLE_ARG_VALUE = RichStyle(color=DIM_TEXT_COLOR)  # Use explicit color instead of dim
STYLE_THINKING = RichStyle(color=AI_COLOR)  # Same as AI color
STYLE_INPUT_OPTION = RichStyle(underline=True)
STYLE_COMMAND_LABEL = RichStyle(color=COMMAND_COLOR, bold=True)
STYLE_COMMAND_CONTENT = RichStyle(color=COMMAND_COLOR)
STYLE_TOOL_OUTPUT_DIM = RichStyle(color=DIM_TEXT_COLOR)  # Use explicit color instead of dim

# --- Define prompt_toolkit Styles using the same semantic color constants ---
PTK_STYLE = PTKStyle.from_dict({
    # Labels (match Rich style names for clarity)
    'style_ai_label':          f'bold fg:{AI_COLOR}',
    'style_user_label':        f'bold fg:{USER_COLOR}',
    'style_info_label':        f'bold fg:{INFO_COLOR}',
    'style_warning_label':     f'bold fg:{WARNING_COLOR}',
    'style_error_label':       f'bold fg:{ERROR_COLOR}',
    'style_system_label':      f'bold fg:{SYSTEM_COLOR}',
    'style_command_label':     f'bold fg:{COMMAND_COLOR}',
    'style_tool_name':         f'fg:{TOOL_COLOR}',  # Now using defined tool color

    # Content (match Rich style names)
    'style_ai_content':        f'fg:{AI_COLOR}',
    'style_info_content':      f'fg:{INFO_COLOR}',
    'style_warning_content':   f'fg:{WARNING_COLOR}',
    'style_error_content':     f'fg:{ERROR_COLOR}',
    'style_system_content':    f'fg:{SYSTEM_COLOR}',
    'style_command_content':   f'fg:{COMMAND_COLOR}',

    # Args/Specifics (use explicit colors instead of 'dim')
    'style_arg_name':          f'fg:{DIM_TEXT_COLOR}',
    'style_arg_value':         f'fg:{DIM_TEXT_COLOR}',
    'style_tool_output_dim':   f'fg:{DIM_TEXT_COLOR}',

    # Other UI
    'style_thinking':          f'fg:{AI_COLOR}',
    'style_input_option':      'underline',

    # Prompt-specific names (used by FormattedText construction)
    'prompt.prefix':           'bold',
    'prompt.suffix':           '',
    'prompt.argname':          f'fg:{DIM_TEXT_COLOR}',
    'prompt.argvalue':         f'fg:{DIM_TEXT_COLOR}',
    'prompt.toolname':         f'bold fg:{TOOL_COLOR}',  # Now using defined tool color

    # Default for input text
    '':                        '',  # Default text style
    'default':                 f'fg:{DIM_TEXT_COLOR}',  # Default value hint
})
# --- End prompt_toolkit Styles ---

# --- ConsoleManager Class (Refactored) ---

class ConsoleManager:
    """
    Centralized manager for console I/O operations without using rich.Live.
    Handles thinking indicator, messages, and prompts via direct printing
    and ANSI escape codes.
    """
    # --- Added constant for truncating output ---
    CONDENSED_OUTPUT_LENGTH = 150
    # --------------------------------------------

    def __init__(self, stderr_output: bool = True):
        """Initialize the ConsoleManager."""
        self.console = Console(stderr=stderr_output, force_terminal=True if not sys.stderr.isatty() else None)
        # REMOVED: self._live_context
        self._lock = Lock()
        # NEW: Flag to track if the spinner is currently displayed on a line
        self._spinner_active = False

        # Copy style constants
        self.STYLE_AI_LABEL = STYLE_AI_LABEL
        self.STYLE_AI_CONTENT = STYLE_AI_CONTENT
        self.STYLE_USER_LABEL = STYLE_USER_LABEL
        self.STYLE_INFO_LABEL = STYLE_INFO_LABEL
        self.STYLE_INFO_CONTENT = STYLE_INFO_CONTENT
        self.STYLE_WARNING_LABEL = STYLE_WARNING_LABEL
        self.STYLE_WARNING_CONTENT = STYLE_WARNING_CONTENT
        self.STYLE_ERROR_LABEL = STYLE_ERROR_LABEL
        self.STYLE_ERROR_CONTENT = STYLE_ERROR_CONTENT
        self.STYLE_SYSTEM_LABEL = STYLE_SYSTEM_LABEL
        self.STYLE_SYSTEM_CONTENT = STYLE_SYSTEM_CONTENT
        self.STYLE_TOOL_NAME = STYLE_TOOL_NAME
        self.STYLE_ARG_NAME = STYLE_ARG_NAME
        self.STYLE_ARG_VALUE = STYLE_ARG_VALUE
        self.STYLE_THINKING = STYLE_THINKING
        self.STYLE_INPUT_OPTION = STYLE_INPUT_OPTION
        self.STYLE_COMMAND_LABEL = STYLE_COMMAND_LABEL
        self.STYLE_COMMAND_CONTENT = STYLE_COMMAND_CONTENT
        self.STYLE_TOOL_OUTPUT_DIM = STYLE_TOOL_OUTPUT_DIM


    # REMOVED: _stop_live_display method

    def _clear_previous_line(self):
        """Clears the previous line if the spinner was active."""
        # No lock needed here as it will be called by methods that already hold the lock
        if self._spinner_active and self.console.is_terminal:
            try:
                # \r moves cursor to beginning of line
                # \x1b[K clears from cursor to end of line
                self.console.file.write('\r\x1b[K')
                self.console.file.flush()
                logger.debug("_clear_previous_line: Cleared line using ANSI codes.")
            except Exception as e:
                logger.error(f"Error clearing line: {e}", exc_info=True)
                print("\nWarning: Could not clear previous line.", file=sys.stderr)
        self._spinner_active = False # Always reset flag after attempt
        
    # --- ADDED METHOD ---
    def clear_current_line(self):
        """Clears the current line using ANSI escape codes."""
        # No lock needed here as it will be called by methods that already hold the lock
        if self.console.is_terminal:
            try:
                # \r moves cursor to beginning of line
                # \x1b[K clears from cursor to end of line
                self.console.file.write('\r\x1b[K')
                self.console.file.flush()
                logger.debug("clear_current_line: Cleared current line using ANSI codes.")
            except Exception as e:
                logger.error(f"Error clearing current line: {e}", exc_info=True)
                print("\nWarning: Could not clear current line.", file=sys.stderr)
    # --- END ADDED METHOD ---

    def start_thinking(self):
        """Displays the 'AI: thinking...' status indicator on the current line."""
        with self._lock:
            # Clear previous line first if needed (e.g., tool output was just printed)
            self._clear_previous_line()

            if self._spinner_active: # Avoid printing multiple spinners
                return

            prefix = Text("AI: ", style=STYLE_AI_LABEL)
            # Simple text spinner - could add rotating characters later if desired
            thinking_text = Text("⏳ Thinking...", style=STYLE_THINKING)

            # Print without a newline to keep it on the current line
            self.console.print(Text.assemble(prefix, thinking_text), end="")
            self._spinner_active = True
            logger.debug("ConsoleManager: Started thinking indicator.")

    def display_message(self, prefix: str, content: str, style_label: RichStyle, style_content: RichStyle):
        """Displays a standard formatted message (INFO, WARNING, ERROR, SYSTEM)."""
        logger.debug(f"Entering display_message for prefix: {prefix[:10]}...")
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active
            text = Text.assemble((prefix, style_label), (escape(content), style_content))
            try:
                self.console.print(text) # Prints with a newline by default
                logger.debug(f"ConsoleManager: Displayed message: {prefix}{content[:50]}...")
            except Exception as e:
                print(f"{prefix}{content}", file=sys.stderr)
                logger.error(f"ConsoleManager: Error during console.print: {e}", exc_info=True)

    def display_tool_output(self, tool_name: str, output: str):
        """
        Displays a condensed, dimmed version of the tool output/prompt.
        The full output should be saved in the message history.
        """
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active

            # Format the output - replace newlines and truncate if needed
            formatted_output = str(output).replace('\n', ' ').replace('\r', '')
            if len(formatted_output) > self.CONDENSED_OUTPUT_LENGTH:
                formatted_output = formatted_output[:self.CONDENSED_OUTPUT_LENGTH] + "..."

            # Create the condensed text with proper tool styles
            text = Text.assemble(
                ("TOOL: ", self.STYLE_TOOL_NAME), # Now using tool name style
                (f"({escape(tool_name)}) ", self.STYLE_TOOL_OUTPUT_DIM), # Dim tool name
                (escape(formatted_output), self.STYLE_TOOL_OUTPUT_DIM) # Dim content
            )
            self.console.print(text) # Print on a new line
            logger.debug(f"ConsoleManager: Displayed condensed tool output for '{tool_name}': {formatted_output[:50]}...")

    def display_ai_response(self, content: str):
        """Displays the final AI text response."""
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active
            text = Text.assemble(("AI: ", STYLE_AI_LABEL), (escape(content), STYLE_AI_CONTENT))
            self.console.print(text)
            logger.debug(f"ConsoleManager: Displayed AI response: {content[:50]}...")

    def display_tool_confirmation(self, tool_name: str, final_args: Dict[str, Any]):
        """Prints the 'AI: Used tool...' confirmation line, replacing the spinner."""
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active

            text = Text.assemble(
                ("AI: ", STYLE_AI_LABEL),
                ("Used tool '", STYLE_AI_CONTENT),
                (escape(tool_name), STYLE_TOOL_NAME),
                ("'", STYLE_AI_CONTENT)
            )
            if final_args:
                text.append(" with ", style=STYLE_AI_CONTENT)
                args_parts = []
                for i, (arg_name, arg_val) in enumerate(final_args.items()):
                    arg_text = Text()
                    arg_text.append(escape(str(arg_name)), style=STYLE_ARG_NAME)
                    arg_text.append(": ", style=STYLE_ARG_NAME)
                    val_str = escape(str(arg_val))
                    max_len = 150 # Keep truncation
                    display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                    arg_text.append(display_val, style=STYLE_ARG_VALUE)
                    args_parts.append(arg_text)
                text.append(Text(", ", style=STYLE_AI_CONTENT).join(args_parts))

            self.console.print(text) # Print the confirmation line (with newline)
            logger.debug(f"ConsoleManager: Displayed tool confirmation for '{tool_name}'.")

    def display_tool_prompt(self, error: PromptNeededError) -> Optional[str]:
        """
        Displays the prompt for a HITL tool using prompt_toolkit for the full line.
        Uses the format: SYSTEM: AI wants to perform an action 'tool_name', edit or confirm: value_to_edit
        """
        with self._lock:
            self._clear_previous_line() # Clear spinner if needed

            tool_name = error.tool_name
            proposed_args = error.proposed_args
            edit_key = error.edit_key
            # prompt_suffix = error.prompt_suffix # No longer used directly for the main prompt text

            if (edit_key not in proposed_args):
                # Use Rich-based display_message for this internal error
                # Ensure STYLE_ERROR_LABEL and STYLE_ERROR_CONTENT are accessible (defined globally or as self attributes)
                self.display_message(
                    "ERROR: ",
                    f"Internal error: edit_key '{edit_key}' not found in proposed arguments for tool '{tool_name}'.",
                    self.STYLE_ERROR_LABEL,
                    self.STYLE_ERROR_CONTENT
                )
                return None

            value_to_edit = proposed_args[edit_key]

            # --- Build prompt_toolkit FormattedText prefix ---
            # NEW FORMAT: SYSTEM: AI wants to perform an action 'tool_name', edit or confirm:
            prompt_prefix_parts: List[Tuple[str, str]] = [
                ('class:style_system_label', "SYSTEM:"), # Changed label and style
                ('class:style_system_content', " AI wants to perform an action '"), # Changed wording and style
                ('class:style_tool_name', escape(tool_name)), # Keep tool name style
                ('class:style_system_content', "', edit or confirm: ") # Changed wording, separator, and style
            ]
            # --- End building FormattedText ---

            # --- Prompt user using prompt_toolkit ---
            user_input: Optional[str] = None
            try:
                logger.debug(f"ConsoleManager: Prompting user for tool '{tool_name}', key '{edit_key}'.")
                user_input = prompt_toolkit_prompt(
                    FormattedText(prompt_prefix_parts), # Pass the new simplified prefix
                    default=str(value_to_edit),
                    style=PTK_STYLE, # Use PTK style object
                    multiline=(len(str(value_to_edit)) > 60 or '\n' in str(value_to_edit))
                )
                if user_input is None: raise EOFError("Prompt returned None.") # Use specific error

            except (EOFError, KeyboardInterrupt):
                 # Use Rich console to print cancel message on a new line
                self.console.print()
                # Ensure STYLE_WARNING_CONTENT is accessible (defined globally or as self attribute)
                self.console.print("Input cancelled.", style=self.STYLE_WARNING_CONTENT)
                logger.warning(f"ConsoleManager: User cancelled input for tool '{tool_name}'.")
                return None
            except Exception as e:
                logger.error(f"ConsoleManager: Error during prompt_toolkit prompt: {e}", exc_info=True)
                # Use Rich console to print error message on a new line
                self.console.print()
                # Ensure STYLE_ERROR_CONTENT is accessible (defined globally or as self attribute)
                self.console.print(f"Error during input prompt: {e}", style=self.STYLE_ERROR_CONTENT)
                return None

            logger.debug(f"ConsoleManager: Received input: '{user_input[:50]}...'")
            # Return input. The calling function (ChatManager) is responsible
            # for clearing this line if needed before printing confirmation.
            return user_input

    def prompt_for_input(self, prompt_text: str, default: Optional[str] = None, is_password: bool = False) -> str:
        """
        Prompts the user for input using prompt_toolkit, handling the prefix correctly.

        Args:
            prompt_text: The prompt text to display (without trailing colon/space).
            default: Optional default value.
            is_password: Whether to hide input as password.

        Returns:
            The user's input as a string, or raises KeyboardInterrupt on cancel.
        """
        with self._lock:
            self._clear_previous_line() # Clear spinner if needed

            # --- Build FormattedText for prompt_toolkit ---
            prompt_parts: List[Tuple[str, str]] = [
                ('', prompt_text) # Use default style for main prompt text
            ]
            if default:
                # Add default value hint with specific style
                prompt_parts.append(('class:default', f" [{escape(default)}]"))

            # Add the trailing colon and space
            prompt_parts.append(('', ": "))
            # --- END FormattedText construction ---

            try:
                # Pass FormattedText to prompt_toolkit
                user_input = prompt_toolkit_prompt(
                    FormattedText(prompt_parts), # Pass the constructed prompt parts
                    default=default or "",
                    is_password=is_password,
                    style=PTK_STYLE # Use the main style object
                )

                if user_input is None: # Handle case where prompt might return None unexpectedly
                    raise EOFError("Prompt returned None.")

                return user_input # Return directly
            except (EOFError, KeyboardInterrupt):
                # Print cancellation message using Rich console on a new line
                self.console.print("\nInput cancelled.", style=STYLE_WARNING_CONTENT)
                logger.warning(f"User cancelled input for prompt: '{prompt_text}'")
                raise KeyboardInterrupt("User cancelled input.")
            except Exception as e:
                logger.error(f"ConsoleManager: Error during prompt_for_input: {e}", exc_info=True)
                # Print error using Rich console on a new line
                self.console.print(f"\nError getting input: {e}", style=STYLE_ERROR_CONTENT)
                # Reraise as KeyboardInterrupt to signal failure upwards similarly to cancellation
                raise KeyboardInterrupt(f"Error getting input: {e}")

# --- Singleton Instance (Remains the same) ---
_console_manager_instance = None
_console_manager_lock = Lock()

def get_console_manager() -> ConsoleManager:
    """Gets the singleton ConsoleManager instance."""
    global _console_manager_instance
    if (_console_manager_instance is None):
        with _console_manager_lock:
            if _console_manager_instance is None:
                _console_manager_instance = ConsoleManager()
    return _console_manager_instance