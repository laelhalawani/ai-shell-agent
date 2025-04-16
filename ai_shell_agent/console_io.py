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
from rich.style import Style
from rich.text import Text
from rich.markup import escape
from rich.traceback import install as rich_traceback_install
rich_traceback_install(show_locals=False) # Install rich tracebacks

# Prompt Toolkit for input - directly import required modules
from prompt_toolkit import prompt as prompt_toolkit_prompt
from prompt_toolkit.shortcuts import confirm as prompt_toolkit_confirm

# --- Console and Styles ---
console = Console(stderr=True) # Print status/errors to stderr to not interfere with stdout piping if needed

# Define styles (adjust colors as needed)
STYLE_AI_LABEL = Style(color="blue", bold=True) # Renamed for clarity 
STYLE_AI_CONTENT = Style(color="blue")          # New style for AI message content
STYLE_USER = Style(color="purple", bold=True) # Note: User prefix not typically printed
STYLE_INFO_LABEL = Style(color="green", bold=True)
STYLE_INFO_CONTENT = Style(color="green")
STYLE_WARNING_LABEL = Style(color="yellow", bold=True)
STYLE_WARNING_CONTENT = Style(color="yellow")
STYLE_ERROR_LABEL = Style(color="red", bold=True)
STYLE_ERROR_CONTENT = Style(color="red")
STYLE_SYSTEM_LABEL = Style(color="cyan", bold=True) # Add bold to system label
STYLE_SYSTEM_CONTENT = Style(color="cyan")        # Style for system message content
STYLE_TOOL_NAME = Style(bold=True) # Italic often not supported well
STYLE_ARG_NAME = Style(dim=True) # Dim or grey
STYLE_ARG_VALUE = Style(dim=True)
STYLE_THINKING = Style(color="blue") # Style for the thinking spinner/text

# Add Input Option Style
STYLE_INPUT_OPTION = Style(underline=True)

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
                console.file.flush()
            except Exception as e:
                 print(f"\nWarning: Error stopping Rich Live display: {e}", file=sys.stderr)

# --- Public API ---

def start_ai_thinking():
    """Starts the 'AI: thinking...' status indicator."""
    global _live_context
    _stop_live() # Ensure any previous live display is stopped

    prefix = Text("AI: ", style=STYLE_AI_LABEL)
    spinner = Spinner("dots", text=Text(" Thinking...", style=STYLE_THINKING))
    renderable_columns = Columns([prefix, spinner], padding=(0, 1), expand=False)

    with _live_lock:
        if _live_context is None: # Prevent starting if already started
            try:
                # --- REMOVE transient=True ---
                _live_context = Live(renderable_columns, console=console, refresh_per_second=10) # Removed transient=True
                _live_context.start(refresh=True)
            except Exception as e:
                _live_context = None
                console.print(f"{prefix.plain} Thinking...", style=STYLE_THINKING) # Fallback print
                print(f"Warning: Failed to start Rich Live display: {e}", file=sys.stderr)

def request_tool_edit(
    tool_name: str,
    proposed_args: Dict[str, Any],
    edit_key: str,
    prompt_suffix: str = "(edit or confirm) > "
) -> Optional[str]:
    """
    Stops live display, prepares prompt message, and prompts user for editable input using prompt_toolkit.

    Args:
        tool_name: Name of the tool being used.
        proposed_args: Dictionary of arguments proposed by the LLM.
        edit_key: The key within proposed_args containing the value to be edited.
        prompt_suffix: The suffix to display before the editable input field.

    Returns:
        The confirmed or edited string value, or None if cancelled.
    """
    _stop_live() # Crucial: Stop spinner before printing/prompting

    if edit_key not in proposed_args:
        print_error(f"Internal error: edit_key '{edit_key}' not found in proposed arguments for tool '{tool_name}'.")
        return None

    value_to_edit = proposed_args[edit_key]

    # --- Construct the prefix text OBJECT (for plain text extraction) ---
    prefix_text_obj = Text.assemble( # Renamed variable
        ("AI: ", STYLE_AI_LABEL),
        ("Using tool '", STYLE_AI_CONTENT),
        (escape(tool_name), STYLE_TOOL_NAME),
        ("'", STYLE_AI_CONTENT)
    )
    # Append non-editable arguments for context in the prefix string
    if proposed_args:
        prefix_text_obj.append(" with ", style=STYLE_AI_CONTENT)
        args_parts: List[str] = [] # Store plain strings for the message
        # Iterate through all args
        for i, (arg_name, arg_val) in enumerate(proposed_args.items()):
             arg_part = f"{escape(str(arg_name))}: "
             if arg_name == edit_key:
                 # For the editable key, just show the key name in the prefix part
                 args_parts.append(arg_part)
             else:
                 # For non-editable keys, show key: value (truncated)
                 val_str = escape(str(arg_val))
                 max_len = 50
                 display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                 args_parts.append(f"{arg_part}{display_val}") # Append formatted value

        # Join argument strings with commas
        prefix_text_obj.append(", ".join(args_parts), style=STYLE_AI_CONTENT) # Use the appropriate style

    # --- Get the plain text version of the prefix ---
    prefix_plain = prefix_text_obj.plain # Use .plain to get string representation

    # --- Construct the FULL message for prompt_toolkit ---
    # Example: "AI: Using tool 'terminal' with cmd: (edit or confirm) > "
    # Ensure there's a space between the prefix and the suffix
    full_prompt_message = f"{prefix_plain.rstrip()} {prompt_suffix}" # rstrip to remove potential trailing space before adding one

    # --- Prompt user using prompt_toolkit with the full message ---
    try:
        # Pass the combined message string to prompt_toolkit
        user_input = prompt_toolkit_prompt(
            message=full_prompt_message, # Use the constructed message
            default=str(value_to_edit), # Ensure default is a string
            # Determine multiline based on key - adjust as needed
            multiline=(edit_key == 'query' or edit_key == 'user_response' or edit_key == 'instruction')
        ).strip() # Strip whitespace from user input

        if not user_input:
             print("\nInput cancelled (empty).", file=sys.stderr)
             return None

        return user_input

    except (EOFError, KeyboardInterrupt):
         print("\nInput cancelled.", file=sys.stderr)
         return None
    except Exception as e:
         # Print error using Rich console for consistency
         print_error(f"Error during input prompt: {e}")
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
        sys.stderr.flush()

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
            max_len = 100
            display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
            arg_text.append(display_val, style=STYLE_ARG_VALUE)
            args_parts.append(arg_text)
        for i, part in enumerate(args_parts):
             text.append(part)
             if i < len(args_parts) - 1:
                  text.append(", ", style=STYLE_AI_CONTENT)

    # Print the final confirmation line
    console.print(text)

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
        sys.stderr.flush() # Ensure the control codes are processed

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

def print_message_block(title: str, content: str, style: Optional[Style] = None): # type: ignore
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