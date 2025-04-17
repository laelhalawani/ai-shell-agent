# ai_shell_agent/console_manager.py
"""
Manages all console input and output using Rich, ensuring clean state transitions.
"""
import sys
import threading
from typing import Dict, Optional, Any

# Rich imports
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.style import Style as RichStyle
from rich.text import Text
from rich.markup import escape
from rich.columns import Columns
from rich.panel import Panel

# Prompt Toolkit imports
from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.formatted_text import FormattedText

# Local imports
from . import logger
from .errors import PromptNeededError  # Import the custom exception

# --- Styles ---
STYLE_AI_LABEL = RichStyle(color="blue", bold=True)
STYLE_AI_CONTENT = RichStyle(color="blue")
STYLE_USER_LABEL = RichStyle(color="purple", bold=True)  # For prompt prefix
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
STYLE_INPUT_OPTION = RichStyle(underline=True)  # For selection prompts

# --- ConsoleManager Class ---

class ConsoleManager:
    """
    Centralized manager for console I/O operations.
    
    This class handles all interactions with the terminal, ensuring proper state
    transitions between different display modes (thinking spinner, tool output,
    user prompts, etc.).
    """
    
    def __init__(self, stderr_output: bool = True):
        """
        Initialize the ConsoleManager.
        
        Args:
            stderr_output: Whether to use stderr for output (default: True)
        """
        self.console = Console(stderr=stderr_output)
        self._live_context: Optional[Live] = None
        self._lock = threading.Lock()
        self._is_thinking: bool = False  # Track spinner state
        
        # Copy style constants to make them accessible as attributes
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

    def _stop_live_display(self):
        """Safely stops the current Live display."""
        with self._lock:
            if self._live_context:
                try:
                    self._live_context.stop()
                    # Clear the line where the live display was
                    if sys.stderr.isatty():  # Use the console's file check
                        self.console.file.write('\x1b[1A\x1b[K')
                        self.console.file.flush()
                except Exception as e:
                    # Use print directly as console might be unusable
                    print(f"\nWarning: Error stopping Rich Live display: {e}", file=sys.stderr)
                finally:
                    self._live_context = None
                    self._is_thinking = False  # Ensure thinking state is reset

    def start_thinking(self):
        """Starts the 'AI: thinking...' status indicator."""
        with self._lock:
            if self._is_thinking:  # Don't start if already thinking
                return
            self._stop_live_display()  # Stop any previous display first

            prefix = Text("AI: ", style=STYLE_AI_LABEL)
            spinner = Spinner("dots", text=Text(" Thinking...", style=STYLE_THINKING))
            renderable = Columns([prefix, spinner], padding=(0, 1), expand=False)

            try:
                # Create and start the new Live display
                self._live_context = Live(renderable, console=self.console, refresh_per_second=10, transient=False)
                self._live_context.start(refresh=True)
                self._is_thinking = True
                logger.debug("ConsoleManager: Started thinking spinner.")
            except Exception as e:
                self._live_context = None
                self._is_thinking = False
                # Fallback print if Live fails
                self.console.print(f"{prefix.plain} Thinking...", style=STYLE_THINKING)
                print(f"Warning: Failed to start Rich Live display: {e}", file=sys.stderr)

    def display_message(self, prefix: str, content: str, style_label: RichStyle, style_content: RichStyle):
        """Displays a standard formatted message (INFO, WARNING, ERROR, SYSTEM)."""
        with self._lock:
            self._stop_live_display()  # Ensure spinner/prompt is stopped
            text = Text.assemble((prefix, style_label), (escape(content), style_content))
            self.console.print(text)
            logger.debug(f"ConsoleManager: Displayed message: {prefix}{content[:50]}...")

    def display_tool_output(self, output: str):
        """Displays the output received from a tool execution."""
        with self._lock:
            self._stop_live_display()
            text = Text.assemble(("TOOL: ", STYLE_INFO_LABEL), (escape(output), STYLE_INFO_CONTENT))
            self.console.print(text)
            logger.debug(f"ConsoleManager: Displayed tool output: {output[:50]}...")

    def display_ai_response(self, content: str):
        """Displays the final AI text response."""
        with self._lock:
            self._stop_live_display()  # Stop spinner before printing final response
            text = Text.assemble(("AI: ", STYLE_AI_LABEL), (escape(content), STYLE_AI_CONTENT))
            self.console.print(text)
            logger.debug(f"ConsoleManager: Displayed AI response: {content[:50]}...")

    def display_tool_confirmation(self, tool_name: str, final_args: Dict[str, Any]):
        """Prints the 'Used tool...' confirmation line."""
        with self._lock:
            self._stop_live_display()  # Stop spinner/prompt

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
                    max_len = 150
                    display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                    arg_text.append(display_val, style=STYLE_ARG_VALUE)
                    args_parts.append(arg_text)
                # Join parts with commas
                text.append(Text(", ", style=STYLE_AI_CONTENT).join(args_parts))

            self.console.print(text)
            logger.debug(f"ConsoleManager: Displayed tool confirmation for '{tool_name}'.")

    def display_tool_prompt(self, error: PromptNeededError) -> Optional[str]:
        """
        Displays the prompt for a HITL tool and gets user input.

        Args:
            error: The PromptNeededError containing prompt details.

        Returns:
            The confirmed/edited string input, or None if cancelled.
        """
        with self._lock:
            self._stop_live_display()  # Stop spinner before prompting

            tool_name = error.tool_name
            proposed_args = error.proposed_args
            edit_key = error.edit_key
            prompt_suffix = error.prompt_suffix

            if edit_key not in proposed_args:
                self.display_message(
                    "ERROR: ", 
                    f"Internal error: edit_key '{edit_key}' not found in proposed arguments for tool '{tool_name}'.", 
                    STYLE_ERROR_LABEL, 
                    STYLE_ERROR_CONTENT
                )
                return None

            value_to_edit = proposed_args[edit_key]

            # --- Build FormattedText list using DIRECT style strings ---
            prompt_fragments = []
            ptk_style_ai_label = 'bold fg:blue'
            ptk_style_ai_content = 'fg:blue'
            ptk_style_tool_name = 'bold'
            ptk_style_arg_name = 'fg:#888888'
            ptk_style_arg_value = 'fg:#888888'
            ptk_style_suffix = ''

            def add_fragment(style_string, text):
                prompt_fragments.append((style_string, text))

            add_fragment(ptk_style_ai_label, "AI: ")
            add_fragment(ptk_style_ai_content, "Using tool '")
            add_fragment(ptk_style_tool_name, escape(tool_name))
            add_fragment(ptk_style_ai_content, "'")

            if proposed_args:
                add_fragment(ptk_style_ai_content, " with ")
                arg_fragments = []
                is_first_arg = True
                for arg_name, arg_val in proposed_args.items():
                    current_part = []
                    # Add comma before args other than the first non-editable one
                    if not is_first_arg and arg_name != edit_key:
                        add_fragment(ptk_style_ai_content, ", ")

                    # Key
                    current_part.append((ptk_style_arg_name, escape(str(arg_name)) + ": "))

                    # Value (skip for editable key)
                    if arg_name != edit_key:
                        val_str = escape(str(arg_val))
                        max_len = 50
                        display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                        current_part.append((ptk_style_arg_value, display_val))
                        is_first_arg = False  # Mark that we've printed a non-editable arg

                    arg_fragments.append(current_part)

                # Join the argument fragments
                for part_list in arg_fragments:
                     prompt_fragments.extend(part_list)

                # Add space before prompt suffix only if args were printed
                add_fragment(ptk_style_ai_content, " ")

            # Add the prompt suffix
            add_fragment(ptk_style_suffix, prompt_suffix)

            formatted_prompt_message = FormattedText(prompt_fragments)
            # --- End FormattedText Build ---

            # --- Prompt user using prompt_toolkit ---
            user_input: Optional[str] = None
            try:
                logger.debug(f"ConsoleManager: Prompting user for tool '{tool_name}', key '{edit_key}'.")
                user_input = prompt_toolkit_prompt(
                    message=formatted_prompt_message,
                    default=str(value_to_edit),
                    # Simplified multiline check
                    multiline=(len(str(value_to_edit)) > 60 or '\n' in str(value_to_edit))
                )  # strip() happens in caller if needed

                if user_input is None:  # Check for explicit None which might indicate issue
                    raise EOFError("Prompt returned None unexpectedly.")

                # We don't print confirmation here; caller does after getting result

            except (EOFError, KeyboardInterrupt):
                # Use console.print for cancellation message, ensures it's seen
                self.console.print("\nInput cancelled.", style=STYLE_WARNING_CONTENT)
                logger.warning(f"ConsoleManager: User cancelled input for tool '{tool_name}'.")
                return None  # Signal cancellation
            except Exception as e:
                logger.error(f"ConsoleManager: Error during input prompt: {e}", exc_info=True)
                self.display_message("ERROR: ", f"Error during input prompt: {e}", STYLE_ERROR_LABEL, STYLE_ERROR_CONTENT)
                return None  # Signal error/cancellation

            logger.debug(f"ConsoleManager: Received input: '{user_input[:50]}...'")
            return user_input  # Return the raw input

    def prompt_for_input(self, prompt_text: str, default: Optional[str] = None, is_password: bool = False) -> str:
        """
        Prompts the user for input using prompt_toolkit.
        
        Args:
            prompt_text: The prompt text to display
            default: Optional default value
            is_password: Whether to hide input as password
            
        Returns:
            The user's input as a string, or empty string on error/cancel
        """
        with self._lock:
            self._stop_live_display()
            
            display_prompt = f"{prompt_text}"
            if default:
                display_prompt = f"{prompt_text} [{escape(default)}]"
                
            try:
                user_input = prompt_toolkit_prompt(
                    display_prompt,
                    default=default or "",
                    is_password=is_password,
                )
                return user_input
            except (EOFError, KeyboardInterrupt):
                self.console.print("\nInput cancelled.", style=STYLE_WARNING_CONTENT)
                # Re-raise KeyboardInterrupt to be handled by the caller's except block
                raise KeyboardInterrupt("User cancelled input.")
            except Exception as e:
                logger.error(f"ConsoleManager: Error during prompt_for_input: {e}", exc_info=True)
                self.display_message(
                    "ERROR: ", 
                    f"Error getting input: {e}", 
                    STYLE_ERROR_LABEL, 
                    STYLE_ERROR_CONTENT
                )
                return ""  # Return empty string on other errors


# --- Singleton Instance ---
# Making it a singleton simplifies access across the application
_console_manager_instance = None
_console_manager_lock = threading.Lock()

def get_console_manager() -> ConsoleManager:
    """Gets the singleton ConsoleManager instance."""
    global _console_manager_instance
    if _console_manager_instance is None:
        with _console_manager_lock:
            if _console_manager_instance is None:
                _console_manager_instance = ConsoleManager()
    return _console_manager_instance