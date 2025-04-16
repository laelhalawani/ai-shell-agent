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
_last_tool_name: Optional[str] = None
_last_tool_args: Optional[Dict] = None
_live_lock = threading.Lock() # Lock for managing live context access

# --- Internal Helper ---
def _stop_live():
    """Safely stops the current Live display and resets related state."""
    global _live_context, _last_tool_name, _last_tool_args
    with _live_lock:
        if (_live_context):
            _live_context.stop()
            _live_context = None
        _last_tool_name = None
        _last_tool_args = None

# --- Public API ---

def start_ai_thinking():
    """Starts the 'AI: thinking...' status indicator."""
    global _live_context
    _stop_live() # Ensure any previous live display is stopped

    # Create the static prefix text
    prefix = Text("AI: ", style=STYLE_AI_LABEL)

    # Create the spinner renderable (which includes " Thinking...")
    spinner = Spinner("dots", text=Text(" Thinking...", style=STYLE_THINKING))

    # Use Columns to arrange prefix and spinner horizontally
    # padding=(0, 1) adds a single space between the columns
    renderable_columns = Columns([prefix, spinner], padding=(0, 1), expand=False)

    with _live_lock:
        try:
            # Pass the Columns renderable directly to Live
            _live_context = Live(renderable_columns, console=console, refresh_per_second=10)
            _live_context.start(refresh=True)
        except Exception as e:
             # Fallback if Live or Columns fails
             _live_context = None
             # Simpler fallback: print prefix and spinner text separately on error
             console.print(f"{prefix.plain} Thinking...", style=STYLE_THINKING) # Fallback print
             print(f"Warning: Failed to start Rich Live display: {e}", file=sys.stderr)

def update_ai_tool_call(tool_name: str, tool_args: Dict):
    """
    Updates the 'thinking' status to show the tool call details.
    Does NOT stop the Live display.
    """
    global _last_tool_name, _last_tool_args
    with _live_lock:
        if not _live_context: return # Ensure live context exists

        _last_tool_name = tool_name
        _last_tool_args = tool_args # Store args for finalize_ai_tool_result

        # Build the rich Text object with proper style assignment
        text = Text.assemble(
            ("AI: ", STYLE_AI_LABEL),
            ("Using tool '", STYLE_AI_CONTENT), 
            (escape(tool_name), STYLE_TOOL_NAME),
            ("'", STYLE_AI_CONTENT)
        )

        if tool_args:
            text.append(" with ", style=STYLE_AI_CONTENT)
            args_parts: List[Text] = []
            for i, (arg_name, arg_val) in enumerate(tool_args.items()):
                arg_text = Text()
                arg_text.append(escape(str(arg_name)), style=STYLE_ARG_NAME)
                arg_text.append(": ", style=STYLE_ARG_NAME)
                # Truncate value display
                val_str = escape(str(arg_val))
                max_len = 50
                display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                arg_text.append(display_val, style=STYLE_ARG_VALUE)
                args_parts.append(arg_text)

            # Join args with commas
            for i, part in enumerate(args_parts):
                text.append(part)
                if i < len(args_parts) - 1:
                    text.append(", ", style=STYLE_AI_CONTENT)

        try:
             _live_context.update(text, refresh=True)
        except Exception as e:
             # Fallback if update fails
             _stop_live() # Stop the potentially broken live context
             console.print(text)
             print(f"Warning: Failed to update Rich Live display: {e}", file=sys.stderr)

def update_ai_response(content: str):
    """
    Updates the 'thinking' status to show the final AI text response
    and stops the Live display.
    """
    # Apply label style to prefix, content style to the rest
    text = Text.assemble(
        ("AI: ", STYLE_AI_LABEL),
        (escape(content), STYLE_AI_CONTENT) # Apply content style
    )

    with _live_lock:
         if (_live_context):
              try:
                  _live_context.update(text, refresh=True)
              except Exception as e:
                   # If update fails, stop and print normally
                   _stop_live()
                   console.print(text)
                   print(f"Warning: Failed to update Rich Live display: {e}", file=sys.stderr)
                   return # Exit after printing
         else:
              # If there was no live context (e.g., thinking fallback), just print
              console.print(text)

    _stop_live() # Finalize the display and clear state

def finalize_ai_tool_result(tool_result: str):
    """
    Updates the status line (which should show the tool call)
    to show the final tool result and stops the Live display.
    Uses the stored _last_tool_name and _last_tool_args.
    """
    with _live_lock:
         if not _live_context:
              # If there's no live context, just print the result simply
              # This might happen if the tool call itself failed to update the live display
              tool_name_str = f"'{escape(_last_tool_name)}'" if _last_tool_name else "tool"
              print_info(f"Result from {tool_name_str}: {tool_result}") # Use print_info for fallback
              _stop_live() # Ensure state is cleared
              return

         if not _last_tool_name:
              # Should not happen if update_ai_tool_call was called, but handle defensively
              print_error("Internal Error: finalize_ai_tool_result called without tool name.")
              _stop_live()
              return

         # Build the final Text object using stored info with Text.assemble
         text = Text.assemble(
            ("AI: ", STYLE_AI_LABEL),
            ("Used tool '", STYLE_AI_CONTENT),
            (escape(_last_tool_name), STYLE_TOOL_NAME),
            ("'", STYLE_AI_CONTENT)
         )

         if _last_tool_args:
             # Reconstruct arg string similar to update_ai_tool_call for consistency
             text.append(" with ", style=STYLE_AI_CONTENT)
             args_parts: List[Text] = []
             for i, (arg_name, arg_val) in enumerate(_last_tool_args.items()):
                 arg_text = Text()
                 arg_text.append(escape(str(arg_name)), style=STYLE_ARG_NAME)
                 arg_text.append(": ", style=STYLE_ARG_NAME)
                 val_str = escape(str(arg_val))
                 max_len = 50
                 display_val = (val_str[:max_len] + '...') if len(val_str) > max_len else val_str
                 arg_text.append(display_val, style=STYLE_ARG_VALUE)
                 args_parts.append(arg_text)
             for i, part in enumerate(args_parts):
                 text.append(part)
                 if i < len(args_parts) - 1:
                     text.append(", ", style=STYLE_AI_CONTENT)

         text.append(": ", style=STYLE_AI_CONTENT)
         # Append the actual tool result (escape it)
         text.append(escape(tool_result), style=STYLE_AI_CONTENT)

         try:
              _live_context.update(text, refresh=True)
         except Exception as e:
               # If update fails, stop and print normally
               _stop_live()
               console.print(text)
               print(f"Warning: Failed to update Rich Live display with tool result: {e}", file=sys.stderr)
               return # Exit after printing

    _stop_live() # Finalize the display and clear state


# --- Static Print Functions ---

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


# --- Input Functions ---

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
        return user_input.strip() # Return stripped input
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.", file=sys.stderr)
        # Decide on behavior: raise exception, return None, or return empty string?
        # Returning empty string might be safest for downstream code expecting a string.
        return ""