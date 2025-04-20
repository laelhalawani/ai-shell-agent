# ai_shell_agent/console_manager.py
"""
Manages all console input and output using Rich and prompt_toolkit,
ensuring clean state transitions without using rich.Live.
Loads styles dynamically from styles.py.
"""
import sys
import io
import threading
from threading import Lock
from typing import Dict, Optional, Any, List, Tuple

# Rich imports
from rich.console import Console
from rich.text import Text
from rich.markup import escape
from rich.columns import Columns # Keep for potential future use
from rich.panel import Panel # Keep for potential future use
from rich.style import Style as RichStyle # Keep for potential future use
# Prompt Toolkit imports
from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.formatted_text import FormattedText
# REMOVED: from prompt_toolkit.styles import Style as PTKStyle (will import the object)

# Local imports
from . import logger
from .errors import PromptNeededError
# Import style constants and objects from styles.py
from .styles import (
    STYLE_AI_LABEL, STYLE_AI_CONTENT, STYLE_USER_LABEL, STYLE_INFO_LABEL,
    STYLE_INFO_CONTENT, STYLE_WARNING_LABEL, STYLE_WARNING_CONTENT,
    STYLE_ERROR_LABEL, STYLE_ERROR_CONTENT, STYLE_SYSTEM_LABEL,
    STYLE_SYSTEM_CONTENT, STYLE_TOOL_NAME, STYLE_ARG_NAME, STYLE_ARG_VALUE,
    STYLE_THINKING, STYLE_INPUT_OPTION, STYLE_COMMAND_LABEL, STYLE_COMMAND_CONTENT,
    STYLE_TOOL_OUTPUT_DIM, STYLE_CODE, # Added STYLE_CODE
    PTK_STYLE # Import the fully processed PTKStyle object
)
# Import global settings loader (only for condensed output length)
from .settings import CONSOLE_CONDENSED_OUTPUT_LENGTH
from .texts import get_text # Import the text loading function


# --- ConsoleManager Class (Refactored for External Styles) ---

class ConsoleManager:
    """
    Centralized manager for console I/O operations without using rich.Live.
    Handles thinking indicator, messages, and prompts via direct printing
    and ANSI escape codes, using styles loaded externally.
    """

    def __init__(self, stderr_output: bool = True):
        """Initialize the ConsoleManager."""
        # Check if stderr is a TTY, force terminal if not (e.g., redirecting stderr)
        force_term = True if not sys.stderr.isatty() and stderr_output else None
        self.console = Console(stderr=stderr_output, force_terminal=force_term)
        self._lock = Lock()
        self._spinner_active = False # Track if spinner is on current line

        # Style objects are now imported directly, no need to copy them as attributes.
        # Keep references if it simplifies method calls, but prefer direct use.
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
        self.STYLE_CODE = STYLE_CODE
        self.STYLE_TOOL_OUTPUT_DIM = STYLE_TOOL_OUTPUT_DIM


    def _clear_previous_line(self):
        """Clears the previous line if the spinner was active."""
        if self._spinner_active and self.console.is_terminal:
            try:
                self.console.file.write('\r\x1b[K')
                self.console.file.flush()
                logger.debug("_clear_previous_line: Cleared line using ANSI codes.")
            except Exception as e:
                logger.error(f"Error clearing line: {e}", exc_info=True)
                # Avoid printing directly to console here to prevent lock issues
                # The effect is just that the spinner might remain visible
        self._spinner_active = False # Always reset flag

    def clear_current_line(self):
        """Clears the current line using ANSI escape codes."""
        if self.console.is_terminal:
            try:
                self.console.file.write('\r\x1b[K')
                self.console.file.flush()
                logger.debug("clear_current_line: Cleared current line using ANSI codes.")
            except Exception as e:
                logger.error(f"Error clearing current line: {e}", exc_info=True)

    def start_thinking(self):
        """Displays the 'AI: thinking...' status indicator on the current line."""
        with self._lock:
            self._clear_previous_line() # Clear previous spinner if active

            if self._spinner_active: # Avoid printing multiple spinners
                return

            prefix = Text(get_text("common.labels.ai"), style=self.STYLE_AI_LABEL)
            thinking_text = Text(get_text("console.thinking_text"), style=self.STYLE_THINKING)

            # Print without a newline to keep it on the current line
            try:
                self.console.print(Text.assemble(prefix, thinking_text), end="")
                self._spinner_active = True
                logger.debug("ConsoleManager: Started thinking indicator.")
            except Exception as e:
                 logger.error(f"ConsoleManager: Error printing thinking indicator: {e}", exc_info=True)


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
                # Fallback to basic print if Rich fails
                try:
                    print(f"{prefix}{content}", file=sys.stderr)
                except Exception: pass # Ignore secondary print errors
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
            # Use loaded setting for length
            if len(formatted_output) > CONSOLE_CONDENSED_OUTPUT_LENGTH:
                formatted_output = formatted_output[:CONSOLE_CONDENSED_OUTPUT_LENGTH] + get_text("common.truncation_marker")

            prefix = get_text("common.labels.tool")
            content = get_text("console.tool_output.format", tool_name=escape(tool_name), output=escape(formatted_output))
            
            text = Text.assemble(
                (prefix, self.STYLE_TOOL_NAME),
                (content, self.STYLE_TOOL_OUTPUT_DIM)
            )
            try:
                self.console.print(text) # Print on a new line
                logger.debug(f"ConsoleManager: Displayed condensed tool output for '{tool_name}': {formatted_output[:50]}...")
            except Exception as e:
                 logger.error(f"ConsoleManager: Error printing tool output: {e}", exc_info=True)


    def display_ai_response(self, content: str):
        """Displays the final AI text response."""
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active
            prefix = get_text("common.labels.ai")
            text = Text.assemble((prefix, self.STYLE_AI_LABEL), (escape(content), self.STYLE_AI_CONTENT))
            try:
                self.console.print(text)
                logger.debug(f"ConsoleManager: Displayed AI response: {content[:50]}...")
            except Exception as e:
                 logger.error(f"ConsoleManager: Error printing AI response: {e}", exc_info=True)


    def display_tool_confirmation(self, tool_name: str, final_args: Dict[str, Any]):
        """Prints the 'AI: Used tool...' confirmation line, replacing the spinner."""
        with self._lock:
            self._clear_previous_line() # Clear spinner if it was active

            ai_prefix = get_text("common.labels.ai")
            used_tool_fmt = get_text("console.tool_confirm.used_tool", tool_name=escape(tool_name))
            
            text = Text.assemble(
                (ai_prefix, self.STYLE_AI_LABEL),
                (used_tool_fmt, self.STYLE_AI_CONTENT)
            )
            if final_args:
                with_args_str = get_text("console.tool_confirm.with_args")
                arg_separator = get_text("console.tool_confirm.arg_separator")
                arg_list_separator = get_text("console.tool_confirm.arg_list_separator")
                trunc_marker = get_text("common.truncation_marker")
                
                text.append(with_args_str, style=self.STYLE_AI_CONTENT)
                args_parts = []
                for i, (arg_name, arg_val) in enumerate(final_args.items()):
                    arg_text = Text()
                    arg_text.append(escape(str(arg_name)), style=self.STYLE_ARG_NAME)
                    arg_text.append(arg_separator, style=self.STYLE_ARG_NAME)
                    val_str = escape(str(arg_val))
                    max_len = CONSOLE_CONDENSED_OUTPUT_LENGTH
                    display_val = (val_str[:max_len] + trunc_marker) if len(val_str) > max_len else val_str
                    arg_text.append(display_val, style=self.STYLE_ARG_VALUE)
                    args_parts.append(arg_text)
                text.append(Text(arg_list_separator, style=self.STYLE_AI_CONTENT).join(args_parts))

            try:
                self.console.print(text) # Print the confirmation line (with newline)
                logger.debug(f"ConsoleManager: Displayed tool confirmation for '{tool_name}'.")
            except Exception as e:
                 logger.error(f"ConsoleManager: Error printing tool confirmation: {e}", exc_info=True)


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

            if (edit_key not in proposed_args):
                error_msg = get_text("console.hitl_prompt.error_edit_key_missing", 
                                   edit_key=edit_key, tool_name=tool_name)
                self.display_message(
                    get_text("common.labels.error"),
                    error_msg,
                    self.STYLE_ERROR_LABEL,
                    self.STYLE_ERROR_CONTENT
                )
                return None

            value_to_edit = proposed_args[edit_key]

            # --- Build prompt_toolkit FormattedText prefix ---
            system_label = get_text("common.labels.system")
            prefix_action = get_text("console.hitl_prompt.prefix_action")
            suffix_edit_confirm = get_text("console.hitl_prompt.suffix_edit_confirm")

            prompt_prefix_parts: List[Tuple[str, str]] = [
                ('class:style_system_label', system_label.strip()),
                ('class:style_system_content', prefix_action),
                ('class:style_tool_name', escape(tool_name)),
                ('class:style_system_content', suffix_edit_confirm)
            ]
            # --- End building FormattedText ---

            # --- Prompt user using prompt_toolkit ---
            user_input: Optional[str] = None
            try:
                logger.debug(f"ConsoleManager: Prompting user for tool '{tool_name}', key '{edit_key}'.")
                # Use the imported PTK_STYLE object
                user_input = prompt_toolkit_prompt(
                    FormattedText(prompt_prefix_parts),
                    default=str(value_to_edit),
                    style=PTK_STYLE,
                    multiline=(len(str(value_to_edit)) > 60 or '\n' in str(value_to_edit))
                )
                if user_input is None: raise EOFError("Prompt returned None.") # Use specific error

            except (EOFError, KeyboardInterrupt):
                # Use Rich console to print cancel message on a new line
                self.console.print() # Ensure on new line
                cancel_msg = get_text("console.common.input_cancelled")
                self.console.print(cancel_msg, style=self.STYLE_WARNING_CONTENT)
                logger.warning(f"ConsoleManager: User cancelled input for tool '{tool_name}'.")
                return None
            except Exception as e:
                logger.error(f"ConsoleManager: Error during prompt_toolkit prompt: {e}", exc_info=True)
                self.console.print() # Ensure on new line
                error_msg = get_text("console.hitl_prompt.error_generic", error=e)
                self.console.print(error_msg, style=self.STYLE_ERROR_CONTENT)
                return None

            logger.debug(f"ConsoleManager: Received input: '{user_input[:50]}...'")
            return user_input

    def prompt_for_input(self, prompt_text: str, default: Optional[str] = None, is_password: bool = False) -> str:
        """
        Prompts the user for input using prompt_toolkit, handling the prefix correctly.
        """
        with self._lock:
            self._clear_previous_line() # Clear spinner if needed

            # --- Build FormattedText for prompt_toolkit ---
            prompt_parts: List[Tuple[str, str]] = [
                # Use PTK style defined for 'prompt.prefix'
                ('class:prompt.prefix', prompt_text)
            ]
            if default:
                default_hint = get_text("console.prompt.default_hint_format", default=escape(default))
                prompt_parts.append(('class:default', default_hint))

            # Add the trailing colon and space - use default style
            prompt_suffix = get_text("console.prompt.suffix")
            prompt_parts.append(('', prompt_suffix))
            # --- END FormattedText construction ---

            try:
                # Pass FormattedText and the main PTK_STYLE object
                user_input = prompt_toolkit_prompt(
                    FormattedText(prompt_parts),
                    default=default or "",
                    is_password=is_password,
                    style=PTK_STYLE
                )

                if user_input is None: # Handle case where prompt might return None unexpectedly
                    raise EOFError("Prompt returned None.")

                return user_input # Return directly
            except (EOFError, KeyboardInterrupt):
                # Print cancellation message using Rich console on a new line
                self.console.print() # Ensure on new line
                cancel_msg = get_text("console.common.input_cancelled")
                self.console.print(cancel_msg, style=self.STYLE_WARNING_CONTENT)
                logger.warning(f"User cancelled input for prompt: '{prompt_text}'")
                raise KeyboardInterrupt("User cancelled input.")
            except Exception as e:
                logger.error(f"ConsoleManager: Error during prompt_for_input: {e}", exc_info=True)
                self.console.print() # Ensure on new line
                error_msg = get_text("console.prompt.error_generic", error=e)
                self.console.print(error_msg, style=self.STYLE_ERROR_CONTENT)
                raise KeyboardInterrupt(f"Error getting input: {e}")

    def display_progress(self, current_item: int, total_items: int):
        """Displays or updates a progress indicator line."""
        with self._lock:
            # Don't clear previous spinner here, as progress updates the current line
            # if self._spinner_active: self._clear_previous_line() # Might cause flicker

            prefix = Text(get_text("common.labels.system"), style=self.STYLE_SYSTEM_LABEL)
            # Use get_text for the progress format string
            progress_text_str = get_text("localize.progress_format", current=current_item, total=total_items)
            progress_text = Text(progress_text_str, style=self.STYLE_SYSTEM_CONTENT)

            # Print with \r to return cursor to the start of the line, and no newline
            try:
                self.console.print(Text.assemble(prefix, progress_text), end="\r")
                # We don't set _spinner_active here, it's a different kind of status
            except Exception as e:
                 logger.error(f"ConsoleManager: Error printing progress: {e}", exc_info=True)

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