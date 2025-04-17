import os
import json
import argparse
from dotenv import load_dotenv
import sys
from pathlib import Path # Import Path

# Get installation directory (keep this function)
def get_install_dir():
    # Assumes ai.py is in ai_shell_agent/
    return Path(__file__).parent.parent.resolve()

# Load environment variables from .env in the installation directory
env_path = get_install_dir() / '.env'
load_dotenv(env_path)

# Setup logger early
from . import logger, ROOT_DIR # Remove console_io import
# Import console manager
from .console_manager import get_console_manager

# Get console manager instance
console = get_console_manager()

# Config manager imports (keep necessary ones)
from .config_manager import (
    get_current_model, set_model, prompt_for_model_selection,
    ensure_api_key_for_current_model, get_api_key_for_model,
    set_api_key_for_model, # Added this back for CLI --set-api-key support
    get_model_provider, check_if_first_run,
    set_default_enabled_toolsets # Keep for first run / select tools
)

# --- Import state manager functions with updated names ---
from .chat_state_manager import (
    create_or_load_chat,
    save_session, get_current_chat, get_enabled_toolsets, update_enabled_toolsets,
    get_active_toolsets,
    get_current_chat_title,
    # --- Import path helpers and JSON read needed here ---
    get_toolset_data_path, get_toolset_global_config_path
)
# --- Import utils for JSON reading ---
from .utils import read_json as _read_json, read_dotenv, write_dotenv # Alias if preferred

# --- Import chat manager AFTER state manager ---
from .chat_manager import (
    get_chat_titles_list, rename_chat, delete_chat, send_message,
    edit_message, start_temp_chat, flush_temp_chats, execute, list_messages,
    list_toolsets # Keep this import
)
# --- Import Toolset registry ---
from .toolsets.toolsets import get_registered_toolsets, get_toolset_names
# --- Import system prompt ---
from .prompts.prompts import SYSTEM_PROMPT

# --- API Key/Setup Functions ---
def first_time_setup():
    # Add log right inside the function start
    logger.debug("Entering first_time_setup function.")
    is_first = check_if_first_run()
    logger.debug(f"check_if_first_run() returned: {is_first}")
    if is_first:
        # Add log immediately after the 'if'
        logger.debug("First run condition met. Preparing for setup prompts.")
        # Add log right before the console call
        logger.debug("Attempting to display 'Welcome...' message via console manager.")
        try:
            console.display_message("INFO:", "Welcome to AI Shell Agent! Performing first-time setup.", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            logger.debug("Successfully displayed 'Welcome...' message.") # Log success after call
        except Exception as e:
            logger.error(f"Error calling console.display_message for Welcome: {e}", exc_info=True)
            # Decide how to handle this - maybe exit? For now, just log.
            # sys.exit(1) # Or perhaps raise

        logger.debug("Attempting to call prompt_for_model_selection.")
        selected_model = prompt_for_model_selection() # This now uses console_manager internally
        logger.debug(f"prompt_for_model_selection returned: {selected_model}")

        if selected_model:
            set_model(selected_model) # set_model logs internally
        else:
            # Use console manager for critical error
            logger.critical("No model selected during first run. Exiting.")
            console.display_message("ERROR:", "No model selected during first run. Exiting.", 
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            sys.exit(1)

        logger.debug("Attempting to call ensure_api_key.")
        if not ensure_api_key(): # ensure_api_key uses console_manager internally
            logger.critical("API Key not provided. Exiting.")
            console.display_message("ERROR:", "API Key not provided. Exiting.", 
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            sys.exit(1)
        logger.debug("ensure_api_key successful.")

        # Add prompt for initial toolsets
        logger.debug("Attempting to call prompt_for_initial_toolsets.")
        prompt_for_initial_toolsets() # This uses console_manager internally
        logger.debug("prompt_for_initial_toolsets finished.")

        console.display_message("INFO:", "First-time setup complete.", 
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        logger.debug("Displayed 'First-time setup complete.'")
    else:
        logger.debug("Not the first run, skipping setup.")

def ensure_api_key() -> bool:
    # Only ensures key for the main agent model
    return ensure_api_key_for_current_model() # This still uses config_manager logic for agent key

def prompt_for_initial_toolsets():
    """Prompts user to select default enabled toolsets during first run."""
    # Combine introductory messages
    intro_text = """
--- Select Default Enabled Toolsets ---
These toolsets will be enabled by default when you create new chats.
"""
    console.display_message("SYSTEM:", intro_text.strip(), console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT) 

    from .toolsets.toolsets import get_registered_toolsets
    from .config_manager import set_default_enabled_toolsets

    all_toolsets = get_registered_toolsets()
    if not all_toolsets:
        console.display_message("WARNING:", "No toolsets found/registered.", 
                              console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        set_default_enabled_toolsets([])
        return

    console.display_message("SYSTEM:", "\nAvailable Toolsets:", 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    options = {}
    idx = 1
    from rich.text import Text # Import Text here
    toolset_lines = []
    for ts_id, meta in sorted(all_toolsets.items(), key=lambda item: item[1].name):
        # Create Text object for highlighting
        line_text = Text.assemble(
            "  ",
            (f"{idx}", console.STYLE_INPUT_OPTION), # Highlight number
            f": {meta.name.ljust(15)} - {meta.description}"
            # Apply base system style implicitly or explicitly if needed
        )
        toolset_lines.append(line_text)
        options[str(idx)] = meta.name
        idx += 1
    # Print all toolset lines at once
    for line in toolset_lines:
        console.console.print(line) # Use console.print directly for Text objects

    # Combine prompt instructions
    prompt_instructions = """
Enter comma-separated numbers TO ENABLE by default (e.g., 1,3).
To enable none by default, leave empty or enter 'none'.
"""
    console.display_message("SYSTEM:", prompt_instructions.strip(), 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    while True:
        try:
            # Use console manager for prompting
            choice_str = console.prompt_for_input("> ").strip()
            selected_names = []
            if not choice_str or choice_str.lower() == 'none':
                pass # selected_names remains empty
            else:
                selected_indices = {c.strip() for c in choice_str.split(',') if c.strip()}
                valid_selection = True
                for index in selected_indices:
                    if index in options:
                        selected_names.append(options[index])
                    else:
                        # Use console manager for error
                        console.display_message("ERROR:", f"Invalid selection '{index}'. Please use numbers from 1 to {idx-1}.", 
                                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                        valid_selection = False
                        break
                if not valid_selection: continue # Ask again

            # Remove duplicates and sort
            final_selection = sorted(list(set(selected_names)))

            # Save as global default
            set_default_enabled_toolsets(final_selection)
            # Use console manager for confirmation
            console.display_message("INFO:", f"Default enabled toolsets set to: {', '.join(final_selection) or 'None'}", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            return

        except (EOFError, KeyboardInterrupt):
            # console.prompt_for_input handles the KeyboardInterrupt print
            console.display_message("WARNING:", "Selection cancelled. Setting no default toolsets.", 
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            set_default_enabled_toolsets([])
            return
        except Exception as e: # Catch other potential errors
            console.display_message("ERROR:", f"An error occurred: {e}", 
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            logger.error(f"Error in toolset prompt: {e}", exc_info=True)
            return # Exit for now

# --- Toolset Selection Command ---
def select_tools_for_chat():
    """Interactive prompt for selecting enabled toolsets for the current chat."""
    chat_id = get_current_chat() # Use chat_id
    if not chat_id:
        console.display_message("ERROR:", "No active chat session.", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        console.display_message("SYSTEM:", "Please load or create a chat first (e.g., `ai -c <chat_title>`).", 
                              console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        return

    chat_title = get_current_chat_title() # Use new function name
    
    # Combine intro messages
    intro_text = f"""
--- Select Enabled Toolsets for Chat: '{chat_title}' ---
Toolsets determine which capabilities the agent can potentially use in this chat.
"""
    console.display_message("SYSTEM:", intro_text.strip(), 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    all_toolsets = get_registered_toolsets() # Dict[id, ToolsetMetadata]
    if not all_toolsets:
        console.display_message("WARNING:", "No toolsets found/registered.", 
                              console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        return

    current_enabled_names = get_enabled_toolsets(chat_id) # Pass chat_id
    console.display_message("SYSTEM:", "\nAvailable Toolsets:", 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    options = {}
    idx = 1
    # Sort toolsets by name for consistent display
    from rich.text import Text # Import Text here
    toolset_lines = []
    for ts_id, meta in sorted(all_toolsets.items(), key=lambda item: item[1].name):
        marker = "[ENABLED]" if meta.name in current_enabled_names else "[DISABLED]"
        # Create Text object for highlighting
        line_text = Text.assemble(
            "  ",
            (f"{idx}", console.STYLE_INPUT_OPTION), # Highlight number
            f": {meta.name.ljust(15)} {marker} - {meta.description}"
            # Apply base system style implicitly or explicitly if needed
        )
        toolset_lines.append(line_text)
        options[str(idx)] = meta.name # Map index to display name
        idx += 1
    # Print all toolset lines at once
    for line in toolset_lines:
        console.console.print(line) # Use console.print directly for Text objects

    # Combine prompt instructions
    prompt_instructions = """
Enter comma-separated numbers TO ENABLE (e.g., 1,3).
To disable all, enter 'none'.
Leave empty to keep current settings.
"""
    console.display_message("SYSTEM:", prompt_instructions.strip(), 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    while True:
        try:
            choice_str = console.prompt_for_input("> ").strip()
            if not choice_str:
                console.display_message("INFO:", f"Kept current enabled toolsets: {', '.join(sorted(current_enabled_names)) or 'None'}", 
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT) # Simplified
                return

            if choice_str.lower() == 'none':
                new_enabled_list_names = []
            else:
                selected_indices = {c.strip() for c in choice_str.split(',') if c.strip()}
                new_enabled_list_names = []
                valid_selection = True
                for index in selected_indices:
                    if index in options:
                        new_enabled_list_names.append(options[index])
                    else:
                        console.display_message("ERROR:", f"Invalid selection '{index}'. Please use numbers from 1 to {idx-1}.", 
                                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                        valid_selection = False
                        break
                if not valid_selection: continue # Ask again

            # Remove duplicates and sort
            new_enabled_list_names = sorted(list(set(new_enabled_list_names)))

            # Update state - this also handles deactivating toolsets
            update_enabled_toolsets(chat_id, new_enabled_list_names) # Pass chat_id and list of names
            console.display_message("INFO:", f"Enabled toolsets set to: {', '.join(new_enabled_list_names) or 'None'}", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT) # Simplified confirmation

            # Also update the global default setting
            from .config_manager import set_default_enabled_toolsets
            set_default_enabled_toolsets(new_enabled_list_names)
            console.display_message("INFO:", "Global default enabled toolsets also updated.", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            console.display_message("INFO:", "Changes apply on next interaction.", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            return

        except (EOFError, KeyboardInterrupt):
            console.display_message("WARNING:", "\nSelection cancelled. No changes made.", 
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return

# --- Toolset Configuration Command (MODIFIED) ---
def configure_toolset_cli(toolset_name: str):
    """Handles the --configure-toolset command."""
    chat_id = get_current_chat()
    if not chat_id:
        console.display_message("ERROR:", "No active chat session. Load or create one first.", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return

    chat_title = get_current_chat_title()
    if not chat_title: # Should not happen if chat_id exists, but check anyway
        console.display_message("ERROR:", f"Could not determine title for active chat {chat_id}.", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return

    # Find the toolset metadata
    from .toolsets.toolsets import get_registered_toolsets, get_toolset_names
    registered_toolsets = get_registered_toolsets()
    target_toolset_id = None
    target_metadata = None
    for ts_id, meta in registered_toolsets.items():
        if meta.name.lower() == toolset_name.lower():
            target_toolset_id = ts_id
            target_metadata = meta
            break

    if not target_metadata or not target_toolset_id:
        console.display_message("ERROR:", f"Toolset '{toolset_name}' not found.", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        console.display_message("SYSTEM:", "Available toolsets: " + ", ".join(get_toolset_names()), 
                              console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        return

    if not target_metadata.configure_func:
        console.display_message("ERROR:", f"Toolset '{target_metadata.name}' (ID: {target_toolset_id}) does not have a configuration function.", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        # Check if config exists, maybe inform user?
        # For now, just exit as per original logic.
        return

    # --- Get required paths ---
    # Use ROOT_DIR from __init__ for reliable .env path
    dotenv_path = ROOT_DIR / '.env'
    # Use chat_state_manager helpers for toolset paths
    local_config_path = get_toolset_data_path(chat_id, target_toolset_id)
    global_config_path = get_toolset_global_config_path(target_toolset_id)

    # --- Read current local (chat-specific) config to pass to the function ---
    # Use the utility function directly
    current_local_config = _read_json(local_config_path, default_value=None) # Pass None default

    # Combine system messages
    config_info = f"""
--- Configuring '{target_metadata.name}' for chat '{chat_title}' ---
(Applying settings to chat config: {local_config_path})
(Applying settings to global default: {global_config_path})
(Checking/Storing secrets in: {dotenv_path})
"""
    console.display_message("SYSTEM:", config_info.strip(), 
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    try:
        # Call the toolset's configure function with all necessary paths and current config
        # Signature: configure_func(global_path, local_path, dotenv_path, current_local_config) -> Dict
        # The function now handles prompting, secret checks, and saving to both paths.
        final_config = target_metadata.configure_func(
            global_config_path,
            local_config_path,
            dotenv_path,
            current_local_config # Pass the read local config (or None)
        )
        # Confirmation message should be printed within configure_func now.
        # Optional: Could check the returned dict 'final_config' if needed here.

    except (EOFError, KeyboardInterrupt):
         logger.warning(f"Configuration cancelled for {toolset_name} by user.")
         console.display_message("WARNING:", "\nConfiguration cancelled.", 
                               console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
    except Exception as e:
        logger.error(f"Error running configuration for {toolset_name}: {e}", exc_info=True)
        console.display_message("ERROR:", f"\nAn error occurred during configuration: {e}", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

# --- Main CLI Execution Logic ---
def main():
    env_path = os.path.join(get_install_dir(), '.env'); load_dotenv(env_path)
    parser = argparse.ArgumentParser(description="AI Shell Agent CLI", formatter_class=argparse.RawTextHelpFormatter)
    # --- Argument Groups ---
    model_group = parser.add_argument_group('Model Configuration')
    chat_group = parser.add_argument_group('Chat Management')
    tool_group = parser.add_argument_group('Toolset Configuration')
    msg_group = parser.add_argument_group('Messaging & Interaction')

    # --- Arguments ---
    model_group.add_argument("-llm", "--model", help="Set LLM model")
    model_group.add_argument("--select-model", action="store_true", help="Interactively select LLM model")
    model_group.add_argument("-k", "--set-api-key", nargs="?", const=True, metavar="API_KEY", 
                             help="Set API key for the main agent model") # Clarified help text

    chat_group.add_argument("-c", "--chat", metavar="TITLE", help="Create or load chat session")
    chat_group.add_argument("-lc", "--load-chat", metavar="TITLE", help="Load chat (same as -c)")
    chat_group.add_argument("-lsc", "--list-chats", action="store_true", help="List chats")
    chat_group.add_argument("-rnc", "--rename-chat", nargs=2, metavar=("OLD", "NEW"), help="Rename chat")
    chat_group.add_argument("-delc", "--delete-chat", metavar="TITLE", help="Delete chat")
    chat_group.add_argument("--temp-flush", action="store_true", help="Remove temp chats")
    chat_group.add_argument("-ct", "--current-chat-title", action="store_true", help="Print current chat title")

    tool_group.add_argument("--select-tools", action="store_true", help="Interactively select enabled toolsets for the CURRENT chat")
    tool_group.add_argument("--list-toolsets", action="store_true", help="List available toolsets and their status")
    tool_group.add_argument("--configure-toolset", metavar="TOOLSET_NAME", 
                            help="Manually run configuration wizard for a toolset in the current chat")
    
    msg_group.add_argument("-m", "--send-message", metavar='"MSG"', help="Send message")
    msg_group.add_argument("-tc", "--temp-chat", metavar='"MSG"', help="Start temporary chat")
    msg_group.add_argument("-e", "--edit", nargs="+", metavar="IDX|last \"MSG\"", help="Edit message & resend")
    msg_group.add_argument("-lsm", "--list-messages", action="store_true", help="Print chat history")
    msg_group.add_argument("-x", "--execute", metavar='"CMD"', help="Execute shell command directly")

    parser.add_argument("message", nargs="?", help="Send message (default action)")
    args = parser.parse_args()

    # --- Execution Order ---
    # 1. Model Selection
    if args.model:
        set_model(args.model)
        ensure_api_key()
        console.display_message("INFO:", f"Model set to {get_current_model()}.", 
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.select_model:
        m = prompt_for_model_selection()
        if m and m != get_current_model():
            set_model(m)
            ensure_api_key()
            console.display_message("INFO:", f"Model set to {get_current_model()}.", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        elif m:
            console.display_message("INFO:", f"Model remains {get_current_model()}.", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        else:
            console.display_message("INFO:", "Model selection cancelled.", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return

    # 2. First Time Setup
    first_time_setup()

    # 3. API Key Management
    if args.set_api_key:
        set_api_key_for_model(get_current_model(), args.set_api_key if isinstance(args.set_api_key, str) else None)
        return

    # 4. Ensure API Key (Exit if missing)
    if not ensure_api_key(): # Only checks agent key now
        logger.critical("Main agent API Key missing.")
        console.display_message("ERROR:", "Agent API Key missing. Set with --set-api-key.", 
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        sys.exit(1)

    # 5. Toolset Selection (needs chat context)
    if args.select_tools:
        select_tools_for_chat()
        return
    if args.list_toolsets: 
        from .chat_manager import list_toolsets
        list_toolsets() # Needs update in chat_manager
        return
    if args.configure_toolset:
        configure_toolset_cli(args.configure_toolset)
        return

    # 7. Direct Command Execution
    if args.execute:
        execute(args.execute)
        return

    # 8. Chat Management
    if args.chat:
        chat_id = create_or_load_chat(args.chat) # Use new function name
        if chat_id: console.display_message("INFO:", f"Switched to chat: '{args.chat}'.", 
                                         console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.load_chat:
        chat_id = create_or_load_chat(args.load_chat) # Use new function name
        if chat_id: console.display_message("INFO:", f"Switched to chat: '{args.load_chat}'.", 
                                         console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.current_chat_title:
        title = get_current_chat_title() # Use new function name
        console.display_message("INFO:", f"Current chat: {title}" if title else "No active chat.", 
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.list_chats:
        get_chat_titles_list()
        return
    if args.rename_chat:
        rename_chat(*args.rename_chat)
        return
    if args.delete_chat:
        delete_chat(args.delete_chat)
        return
    if args.temp_flush:
        flush_temp_chats()
        return

    # --- Operations requiring active chat ---
    active_chat_id = get_current_chat() # Use chat_id consistently

    # 9. Messaging / History
    if args.list_messages:
        if not active_chat_id:
            console.display_message("ERROR:", "No active chat. Use -c <title> first.", 
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
        list_messages() # chat_manager will get chat_id using get_current_chat()
        return
    if args.edit:
        if not active_chat_id:
            console.display_message("ERROR:", "No active chat. Use -c <title> first.", 
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
        idx_str, msg_parts = args.edit[0], args.edit[1:]
        new_msg = " ".join(msg_parts)
        idx = None
        if idx_str.lower() != "last":
            try:
                idx = int(idx_str)
            except ValueError:
                console.display_message("ERROR:", f"Invalid index '{idx_str}'. Must be integer or 'last'.", 
                                      console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                return
        edit_message(idx, new_msg) # chat_manager will get chat_id using get_current_chat()
        return

    # 10. Sending Messages (Default actions)
    msg_to_send = args.send_message or args.message # Prioritize -m
    if msg_to_send:
        if not active_chat_id and not args.temp_chat:
            # If no active chat and not explicitly temp
            console.display_message("INFO:", "No active chat. Starting temporary chat...", 
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            start_temp_chat(msg_to_send)
        elif active_chat_id:
            send_message(msg_to_send) # chat_manager will get chat_id using get_current_chat()
        # If args.temp_chat is set, it's handled below
        return
    if args.temp_chat: # Handles -tc explicitly
        start_temp_chat(args.temp_chat)
        return

    # 11. No arguments provided
    parser.print_help()

if __name__ == "__main__":
    main()
