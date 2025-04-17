# ai_shell_agent/console_manager.py
"""
Manages all console input and output using Rich and prompt_toolkit,
ensuring clean state transitions without using rich.Live.
"""
import sys
import threading
from threading import Lock
from typing import Dict, Optional, Any

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

# Local imports
from . import logger
from .errors import PromptNeededError

# --- Styles (Remain the same) ---
STYLE_AI_LABEL = RichStyle(color="blue", bold=True)
STYLE_AI_CONTENT = RichStyle(color="blue")
STYLE_USER_LABEL = RichStyle(color="purple", bold=True)
STYLE_INFO_LABEL = RichStyle(color="green", bold=True)
STYLE_INFO_CONTENT = RichStyle(color="green")
STYLE_WARNING_LABEL = RichStyle(color="yellow", bold=True)
STYLE_WARNING_CONTENT = RichStyle(color="yellow")
STYLE_ERROR_LABEL = RichStyle(color="red", bold=True)
STYLE_ERROR_CONTENT = RichStyle(color="red")
STYLE_SYSTEM_LABEL = RichStyle(color="cyan", bold=True)
STYLE_SYSTEM_CONTENT = RichStyle(color="cyan")
STYLE_TOOL_NAME = RichStyle(bold=True)
STYLE_ARG_NAME = RichStyle(dim=True)
STYLE_ARG_VALUE = RichStyle(dim=True)
STYLE_THINKING = RichStyle(color="blue")
STYLE_INPUT_OPTION = RichStyle(underline=True)
STYLE_COMMAND_LABEL = RichStyle(color="magenta", bold=True) # Added for consistency
STYLE_COMMAND_CONTENT = RichStyle(color="magenta") # Added for consistency

# --- ConsoleManager Class (Refactored) ---

class ConsoleManager:
    """
    Centralized manager for console I/O operations without using rich.Live.
    Handles thinking indicator, messages, and prompts via direct printing
    and ANSI escape codes.
    """

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

    def display_tool_output(self, output: str):
        """Displays the output received from a tool execution ON A NEW LINE."""
        with self._lock:
            # Do not clear the previous line here
            text = Text.assemble(("TOOL: ", STYLE_INFO_LABEL), (escape(output), STYLE_INFO_CONTENT))
            self.console.print(text) # Print on a new line
            logger.debug(f"ConsoleManager: Displayed tool output: {output[:50]}...")

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
        Displays the prompt for a HITL tool and gets user input on the same line.

        Args:
            error: The PromptNeededError containing prompt details.

        Returns:
            The confirmed/edited string input, or None if cancelled.
        """
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active

            tool_name = error.tool_name
            proposed_args = error.proposed_args
            edit_key = error.edit_key
            prompt_suffix = error.prompt_suffix

            if edit_key not in proposed_args:
                # Use display_message for consistency
                self.display_message(
                    "ERROR: ",
                    f"Internal error: edit_key '{edit_key}' not found in proposed arguments for tool '{tool_name}'.",
                    STYLE_ERROR_LABEL,
                    STYLE_ERROR_CONTENT
                )
                return None

            value_to_edit = proposed_args[edit_key]

            # --- Build and print the prompt prefix using Rich ---
            prompt_prefix = Text.assemble(
                ("AI: ", STYLE_AI_LABEL),
                ("Using tool '", STYLE_AI_CONTENT),
                (escape(tool_name), STYLE_TOOL_NAME),
                ("'", STYLE_AI_CONTENT)
            )

            # Add non-editable args to the prefix
            non_editable_args_parts = []
            for arg_name, arg_val in proposed_args.items():
                if arg_name != edit_key:
                    arg_text = Text()
                    arg_text.append(escape(str(arg_name)), style=STYLE_ARG_NAME)
                    arg_text.append(": ", style=STYLE_ARG_NAME)
                    val_str = escape(str(arg_val))
                    max_len = 50 # Shorter for prefix
                    display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                    arg_text.append(display_val, style=STYLE_ARG_VALUE)
                    non_editable_args_parts.append(arg_text)

            if non_editable_args_parts:
                prompt_prefix.append(" with ", style=STYLE_AI_CONTENT)
                prompt_prefix.append(Text(", ", style=STYLE_AI_CONTENT).join(non_editable_args_parts))
                prompt_prefix.append(" ", style=STYLE_AI_CONTENT) # Space before editable part

            # Add the editable key name and suffix
            prompt_prefix.append(escape(edit_key), style=STYLE_ARG_NAME)
            prompt_prefix.append(" ", style=STYLE_ARG_NAME)
            prompt_prefix.append(prompt_suffix, style=STYLE_AI_CONTENT) # Suffix like "(edit or confirm) > "

            # Print the prefix WITHOUT newline
            self.console.print(prompt_prefix, end="")
            # --- End Prefix Printing ---

            # --- Prompt user using prompt_toolkit on the same line ---
            user_input: Optional[str] = None
            try:
                logger.debug(f"ConsoleManager: Prompting user for tool '{tool_name}', key '{edit_key}'.")
                # Use empty message as Rich printed the prefix
                user_input = prompt_toolkit_prompt(
                    "",
                    default=str(value_to_edit),
                    # Simplified multiline check
                    multiline=(len(str(value_to_edit)) > 60 or '\n' in str(value_to_edit))
                )

                if user_input is None:
                    raise EOFError("Prompt returned None unexpectedly.")

            except (EOFError, KeyboardInterrupt):
                self.console.print("\nInput cancelled.", style=STYLE_WARNING_CONTENT)
                logger.warning(f"ConsoleManager: User cancelled input for tool '{tool_name}'.")
                return None
            except Exception as e:
                logger.error(f"ConsoleManager: Error during input prompt: {e}", exc_info=True)
                # Print error on a new line
                self.console.print(f"\nError during input prompt: {e}", style=STYLE_ERROR_CONTENT)
                return None

            logger.debug(f"ConsoleManager: Received input: '{user_input[:50]}...'")
            return user_input

    def prompt_for_input(self, prompt_text: str, default: Optional[str] = None, is_password: bool = False) -> str:
        """
        Prompts the user for input using prompt_toolkit on the same line.

        Args:
            prompt_text: The prompt text to display
            default: Optional default value
            is_password: Whether to hide input as password

        Returns:
            The user's input as a string, or raises KeyboardInterrupt on cancel.
        """
        with self._lock:
            self._clear_previous_line() # Clear spinner if needed

            # Construct prompt text using Rich Text
            prompt_message = Text(prompt_text)
            if default:
                prompt_message.append(f" [{escape(default)}]", style="dim")
            prompt_message.append(": ")

            # Print the prompt text WITHOUT newline
            self.console.print(prompt_message, end="")

            try:
                # Prompt toolkit only captures input
                user_input = prompt_toolkit_prompt(
                    "", # Empty message, as Rich printed it
                    default=default or "",
                    is_password=is_password,
                )
                return user_input # Return directly
            except (EOFError, KeyboardInterrupt):
                self.console.print("\nInput cancelled.", style=STYLE_WARNING_CONTENT)
                raise KeyboardInterrupt("User cancelled input.")
            except Exception as e:
                logger.error(f"ConsoleManager: Error during prompt_for_input: {e}", exc_info=True)
                self.console.print(f"\nError getting input: {e}", style=STYLE_ERROR_CONTENT)
                raise KeyboardInterrupt(f"Error getting input: {e}")

# --- Singleton Instance (Remains the same) ---
_console_manager_instance = None
_console_manager_lock = Lock()

def get_console_manager() -> ConsoleManager:
    """Gets the singleton ConsoleManager instance."""
    global _console_manager_instance
    if _console_manager_instance is None:
        with _console_manager_lock:
            if _console_manager_instance is None:
                _console_manager_instance = ConsoleManager()
    return _console_manager_instance