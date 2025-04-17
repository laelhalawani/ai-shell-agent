# ai_shell_agent/console_io.py
"""
Handles all console input and output for the AI Shell Agent using the 'rich' library.
Manages printing user prompts, AI responses, tool calls/results, status messages,
and the 'thinking...' indicator.
"""

from typing import Dict, Optional, Any, List
import sys
import threading # Import threading for lock

# Rich imports - directly import required modules
from rich.console import Console, Group 
from rich.columns import Columns # Add Columns import
from rich.live import Live
from rich.spinner import Spinner
# --- Change Rich Style import ---
from rich.style import Style as RichStyle # Rename to avoid conflict
from rich.text import Text
from rich.markup import escape
from rich.traceback import install as rich_traceback_install
rich_traceback_install(show_locals=False) # Install rich tracebacks

# Prompt Toolkit for input - directly import required modules
from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.shortcuts import confirm as prompt_toolkit_confirm
# --- ADD prompt_toolkit FormattedText and Style ---
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style as PromptToolkitStyle # Rename Style import
from prompt_toolkit.styles import merge_styles

# --- Console and Styles ---
console = Console(stderr=True) # Print status/errors to stderr to not interfere with stdout piping if needed

# Define styles (adjust colors as needed)
STYLE_AI_LABEL = RichStyle(color="blue", bold=True) # Renamed for clarity 
STYLE_AI_CONTENT = RichStyle(color="blue")          # New style for AI message content
STYLE_USER = RichStyle(color="purple", bold=True) # Note: User prefix not typically printed
STYLE_INFO_LABEL = RichStyle(color="green", bold=True)
STYLE_INFO_CONTENT = RichStyle(color="green")
STYLE_WARNING_LABEL = RichStyle(color="yellow", bold=True)
STYLE_WARNING_CONTENT = RichStyle(color="yellow")
STYLE_ERROR_LABEL = RichStyle(color="red", bold=True)
STYLE_ERROR_CONTENT = RichStyle(color="red")
STYLE_SYSTEM_LABEL = RichStyle(color="cyan", bold=True) # Add bold to system label
STYLE_SYSTEM_CONTENT = RichStyle(color="cyan")        # Style for system message content
STYLE_TOOL_NAME = RichStyle(bold=True) # Italic often not supported well
STYLE_ARG_NAME = RichStyle(dim=True) # Dim or grey
STYLE_ARG_VALUE = RichStyle(dim=True)
STYLE_THINKING = RichStyle(color="blue") # Style for the thinking spinner/text

# Add Input Option Style
STYLE_INPUT_OPTION = RichStyle(underline=True)

# --- Live Display State ---
_live_context: Optional[Live] = None # type: ignore
_live_lock = threading.Lock() # Lock for managing live context access

# --- Internal Helper ---
def _stop_live():
    """Safely stops the current Live display and resets related state."""
    global _live_context
    with _live_lock:
        live = _live_context # Get context inside lock
        if live:
            _live_context = None # Clear context variable *before* stopping
            try:
                # Ensure we stop and clear the display area
                live.stop()
                console.print("", end="") # Force a clear/redraw
                # Add a small flush just in case
                console.file.flush() # <<< ADDED FLUSH
            except Exception as e:
                 # Use print directly here as console might be unusable
                 print(f"\nWarning: Error stopping Rich Live display: {e}", file=sys.stderr) # Modified print

# --- Public API ---

def start_ai_thinking():
    """Starts the 'AI: thinking...' status indicator."""
    global _live_context
    _stop_live() # Ensure any previous live display is stopped

    prefix = Text("AI: ", style=STYLE_AI_LABEL)
    spinner = Spinner("dots", text=Text(" Thinking...", style=STYLE_THINKING))
    renderable_columns = Columns([prefix, spinner], padding=(0, 1), expand=False)

    with _live_lock:
        if (_live_context is None): # Prevent starting if already started
            try:
                # --- REMOVE transient=True ---
                _live_context = Live(renderable_columns, console=console, refresh_per_second=10) # Removed transient=True
                _live_context.start(refresh=True)
            except Exception as e:
                _live_context = None
                console.print(f"{prefix.plain} Thinking...", style=STYLE_THINKING) # Fallback print
                print(f"Warning: Failed to start Rich Live display: {e}", file=sys.stderr)

# --- request_tool_edit (MODIFIED for FormattedText) ---
def request_tool_edit(
    tool_name: str,
    proposed_args: Dict[str, Any],
    edit_key: str,
    prompt_suffix: str = "(edit or confirm) > "
) -> Optional[str]:
    """
    Stops live display, prepares prompt message using prompt_toolkit FormattedText
    with direct style strings, and prompts user for editable input.

    Args:
        tool_name: Name of the tool being used.
        proposed_args: Dictionary of arguments proposed by the LLM.
        edit_key: The key within proposed_args containing the value to be edited.
        prompt_suffix: The suffix to display before the editable input field.

    Returns:
        The confirmed or edited string value, or None if cancelled.
    """
    _stop_live() # Stop spinner

    if edit_key not in proposed_args:
        print_error(f"Internal error: edit_key '{edit_key}' not found in proposed arguments for tool '{tool_name}'.")
        return None

    value_to_edit = proposed_args[edit_key]

    # --- Build FormattedText list using DIRECT style strings ---
    # List of (style_string, text_fragment) tuples
    prompt_fragments = []

    # Define direct style strings (Map from Rich Styles)
    ptk_style_ai_label = 'bold fg:blue'
    ptk_style_ai_content = 'fg:blue'
    ptk_style_tool_name = 'bold'
    ptk_style_arg_name = 'fg:#888888' # Grey
    ptk_style_arg_value = 'fg:#888888' # Grey
    ptk_style_suffix = '' # Default terminal style

    # Helper to add fragment with direct style string
    def add_fragment(style_string, text):
        prompt_fragments.append((style_string, text))

    add_fragment(ptk_style_ai_label, "AI: ")
    add_fragment(ptk_style_ai_content, "Using tool '")
    add_fragment(ptk_style_tool_name, escape(tool_name))
    add_fragment(ptk_style_ai_content, "'")

    if proposed_args:
        add_fragment(ptk_style_ai_content, " with ")
        arg_fragments = []
        for i, (arg_name, arg_val) in enumerate(proposed_args.items()):
            current_part = []
            # Key
            current_part.append((ptk_style_arg_name, escape(str(arg_name)) + ": "))

            # Value (or skip for editable key)
            if arg_name != edit_key:
                val_str = escape(str(arg_val))
                max_len = 50
                display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                current_part.append((ptk_style_arg_value, display_val))

            arg_fragments.append(current_part)

        # Join the argument fragments with commas
        for i, part_list in enumerate(arg_fragments):
            prompt_fragments.extend(part_list)
            is_last_arg = (i == len(arg_fragments) - 1)
            arg_name_of_part = list(proposed_args.keys())[i]

            # Add comma and space if not last and not the editable key
            if not is_last_arg and arg_name_of_part != edit_key:
                add_fragment(ptk_style_ai_content, ", ")
            # Add space after the last non-editable arg or the editable key name's colon
            elif is_last_arg or arg_name_of_part == edit_key:
                 add_fragment(ptk_style_ai_content, " ") # Add space before prompt suffix

    # Add the prompt suffix with its own style
    add_fragment(ptk_style_suffix, prompt_suffix)

    # Create FormattedText object
    formatted_prompt_message = FormattedText(prompt_fragments)
    # --- End FormattedText Build ---

    # --- Prompt user using prompt_toolkit with FormattedText ---
    try:
        user_input = prompt_toolkit_prompt(
            message=formatted_prompt_message, # Pass the list of fragments
            default=str(value_to_edit),
            # REMOVED: style=PROMPT_TOOLKIT_STYLES, # Apply the custom styles
            multiline=(edit_key == 'query' or edit_key == 'user_response' or edit_key == 'instruction') # Removed 'cmd' from multiline keys
        ).strip()

        if not user_input:
             # Use console.print for consistency, writing to stderr
             console.print("\nInput cancelled (empty).", style=STYLE_WARNING_CONTENT, stderr=True) # Use Rich print
             return None

        return user_input

    except (EOFError, KeyboardInterrupt):
         # Use console.print for consistency, writing to stderr
         console.print("\nInput cancelled.", style=STYLE_WARNING_CONTENT, stderr=True) # Use Rich print
         return None
    except Exception as e:
         # Print error using the defined function
         print_error(f"Error during input prompt: {e}")
         # Optionally log the full traceback if needed
         # logger.error("Exception during prompt:", exc_info=True)
         return None

def print_tool_execution_info(tool_name: str, final_args: Dict[str, Any]):
    """
    Prints the 'Used tool...' line with the final arguments after confirmation/edit.
    This replaces the line where the prompt occurred.
    """
    # Ensure live is stopped, just in case
    _stop_live()

    # Move cursor up one line and clear it. This targets the line where prompt_toolkit rendered.
    # Using stderr ensures it doesn't interfere with potential stdout redirection.
    # Check if stderr is a TTY before writing control codes
    if sys.stderr.isatty():
        sys.stderr.write('\x1b[1A\x1b[K')
        sys.stderr.flush() # <<< ADDED FLUSH

    # Build the final Text object (logic remains the same)
    text = Text.assemble(
        ("AI: ", STYLE_AI_LABEL),
        ("Used tool '", STYLE_AI_CONTENT),
        (escape(tool_name), STYLE_TOOL_NAME),
        ("'", STYLE_AI_CONTENT)
    )
    if final_args:
        text.append(" with ", style=STYLE_AI_CONTENT)
        args_parts: List[Text] = []
        for i, (arg_name, arg_val) in enumerate(final_args.items()):
            arg_text = Text()
            arg_text.append(escape(str(arg_name)), style=STYLE_ARG_NAME)
            arg_text.append(": ", style=STYLE_ARG_NAME)
            val_str = escape(str(arg_val))
            # Increased max_len for better visibility in confirmation line
            max_len = 150 # Increased from 100
            display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
            arg_text.append(display_val, style=STYLE_ARG_VALUE)
            args_parts.append(arg_text)
        for i, part in enumerate(args_parts):
             text.append(part)
             if i < len(args_parts) - 1:
                  text.append(", ", style=STYLE_AI_CONTENT)

    # Print the final confirmation line
    console.print(text) # Print to stderr by default

# --- ADD print_tool_output ---
def print_tool_output(output: str):
    """
    Prints the output received from a tool execution.
    """
    # Ensure any lingering live display is stopped (belt and suspenders)
    _stop_live()
    # Escape the output content to prevent accidental Rich markup interpretation
    console.print(Text.assemble(("TOOL: ", STYLE_INFO_LABEL), (escape(output), STYLE_INFO_CONTENT)))


def update_ai_response(content: str):
    """
    Stops the 'thinking' status, clears its line, and shows the final AI text response.
    """
    # 1. Stop the Live spinner updates first
    _stop_live()

    # 2. Clear the line where the spinner was displayed
    # Check if stderr is a TTY before writing control codes
    if sys.stderr.isatty():
        # Move cursor up one line and clear the entire line
        sys.stderr.write('\x1b[1A\x1b[K')
        sys.stderr.flush() # Ensure the control codes are processed # <<< ADDED FLUSH

    # 3. Build and print the final AI response text
    text = Text.assemble(
        ("AI: ", STYLE_AI_LABEL),
        (escape(content), STYLE_AI_CONTENT) # Apply content style
    )
    console.print(text) # Print the final response to stderr

# --- Static Print Functions (No changes needed) ---

def print_info(message: str):
    """Prints an informational message."""
    _stop_live()
    console.print(Text.assemble(("INFO: ", STYLE_INFO_LABEL), (escape(message), STYLE_INFO_CONTENT)))

def print_warning(message: str):
    """Prints a warning message."""
    _stop_live()
    console.print(Text.assemble(("WARNING: ", STYLE_WARNING_LABEL), (escape(message), STYLE_WARNING_CONTENT)))

def print_error(message: str):
    """Prints an error message."""
    _stop_live()
    console.print(Text.assemble(("ERROR: ", STYLE_ERROR_LABEL), (escape(message), STYLE_ERROR_CONTENT)))

def print_system(message: str):
    """Prints a system message (e.g., config prompts)."""
    _stop_live()
    console.print(Text.assemble(("SYSTEM: ", STYLE_SYSTEM_LABEL), (escape(message), STYLE_SYSTEM_CONTENT)))

def print_message_block(title: str, content: str, style: Optional[RichStyle] = None): # type: ignore
     """Prints a block of text, often used for history or list output."""
     _stop_live()
     from rich.panel import Panel # Import locally
     panel = Panel(escape(content), title=escape(title), border_style=style or "dim", expand=False)
     console.print(panel)


# --- Input Functions (No changes needed) ---

def prompt_for_input(prompt_text: str, default: Optional[str] = None, is_password: bool = False) -> str:
    """
    Prompts the user for input using prompt_toolkit.

    Args:
        prompt_text: The text to display to the user.
        default: Optional default value for the prompt.
        is_password: If True, input will be masked.

    Returns:
        The string entered by the user.
    """
    _stop_live() # Ensure no live display interferes with input

    # Format the prompt slightly for clarity
    display_prompt = f"{prompt_text}: "
    if default:
         display_prompt = f"{prompt_text} [{escape(default)}]: "

    try:
        # Use prompt_toolkit directly
        user_input = prompt_toolkit_prompt(
            display_prompt,
            default=default or "", # prompt_toolkit needs empty string, not None
            is_password=is_password,
        )
        return user_input # Return raw input, stripping is responsibility of caller if needed

    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.", file=sys.stderr)
        # Decide on behavior: raise exception, return None, or return empty string?
        # Returning empty string might be safest for downstream code expecting a string.
        return ""