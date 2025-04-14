import os
import json
import argparse
from dotenv import load_dotenv
import sys # Import sys for exit

# Get installation directory
def get_install_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env in the installation directory
env_path = os.path.join(get_install_dir(), '.env')
load_dotenv(env_path)

# Setup logger early
from . import logger

# Config manager imports
from .config_manager import (
    get_current_model, set_model, prompt_for_model_selection,
    ensure_api_key_for_current_model, get_api_key_for_model, set_api_key_for_model,
    get_model_provider, check_if_first_run, prompt_for_edit_mode_selection,
    prompt_for_coder_models_selection, get_aider_main_model, get_aider_editor_model,
    get_aider_weak_model, set_aider_main_model, set_aider_editor_model, set_aider_weak_model
)

# --- Import state manager FIRST ---
from .chat_state_manager import (
    create_or_load_chat as create_or_load_chat_state,
    save_session, get_current_chat, get_enabled_toolsets, update_enabled_toolsets,
    get_active_toolsets, _update_message_in_chat, # Keep for edit/select-tools
    get_current_chat_title as get_current_chat_title_state
)
# --- Import chat manager AFTER state manager ---
from .chat_manager import (
    get_chat_titles_list, rename_chat, delete_chat, send_message,
    edit_message, start_temp_chat, flush_temp_chats, execute, list_messages
)
# --- Import Toolset registry ---
from .toolsets.toolsets import get_registered_toolsets, get_toolset_names # Use get_registered_toolsets for select_tools
# --- Import system prompt ---
from .prompts.prompts import SYSTEM_PROMPT

# --- API Key/Setup Functions ---
def first_time_setup():
    if check_if_first_run():
        logger.info("Welcome to AI Shell Agent! Performing first-time setup.")
        selected_model = prompt_for_model_selection()
        if selected_model: set_model(selected_model)
        else: logger.critical("No model selected during first run. Exiting."); sys.exit(1)
        if not ensure_api_key(): logger.critical("API Key not provided. Exiting."); sys.exit(1)

        registered_ts = get_toolset_names()
        if "File Editor" in registered_ts:
            logger.info("Configure AI Code Editor (File Editor Toolset) models now?")
            try:
                setup_coder = input("Configure AI Code Editor models? (y/n, default: y): ").strip().lower()
                if not setup_coder or setup_coder.startswith('y'):
                    if not get_aider_main_model(): set_aider_main_model(selected_model)
                    if not get_aider_editor_model(): set_aider_editor_model("gpt-4o-mini")
                    if not get_aider_weak_model(): set_aider_weak_model("gpt-4o-mini")
                    prompt_for_coder_models_selection()
                    ensure_api_keys_for_coder_models()
            except (EOFError, KeyboardInterrupt): logger.warning("Coder model setup skipped.")
        else: logger.info("File Editor toolset not available. Skipping coder model setup.")
        logger.info("First-time setup complete.")

def ensure_api_key() -> bool:
    return ensure_api_key_for_current_model()

def ensure_api_keys_for_coder_models() -> None:
    logger.debug("Ensuring API keys for configured Coder models.")
    if "File Editor" not in get_toolset_names(): return
    checked_models = {get_current_model()}
    models_to_check = {
         "AI Code Editor Main": get_aider_main_model(),
         "AI Code Editor Editor": get_aider_editor_model(),
         "AI Code Editor Weak": get_aider_weak_model(),
    }
    for role, model_name in models_to_check.items():
         if model_name and model_name not in checked_models:
             logger.info(f"Checking API key for {role} model: {model_name}")
             api_key, env_var = get_api_key_for_model(model_name)
             if not api_key:
                 logger.warning(f"API key ({env_var}) needed for {role} model ({model_name}) but not found.")
                 set_api_key_for_model(model_name) # Prompt user
             checked_models.add(model_name)

# --- Toolset Selection Command ---
def select_tools_for_chat():
    """Interactive prompt for selecting enabled toolsets for the current chat."""
    chat_file = get_current_chat()
    if not chat_file:
        print("\nError: No active chat session.")
        print("Please load or create a chat first (e.g., `ai -c <chat_title>`).")
        return

    chat_title = get_current_chat_title_state()
    print(f"\n--- Select Enabled Toolsets for Chat: '{chat_title}' ---")
    print("Toolsets determine which capabilities the agent can potentially use in this chat.")

    all_toolsets = get_registered_toolsets() # Dict[id, ToolsetMetadata]
    if not all_toolsets:
        print("No toolsets found/registered.")
        return

    current_enabled_names = get_enabled_toolsets(chat_file) # List of display names
    print("\nAvailable Toolsets:")
    options = {}
    idx = 1
    # Sort toolsets by name for consistent display
    for ts_id, meta in sorted(all_toolsets.items(), key=lambda item: item[1].name):
        marker = "[ENABLED]" if meta.name in current_enabled_names else "[DISABLED]"
        print(f"  {idx}: {meta.name.ljust(15)} {marker} - {meta.description}")
        options[str(idx)] = meta.name # Map index to display name
        idx += 1

    print("\nEnter comma-separated numbers TO ENABLE (e.g., 1,3).")
    print("To disable all, enter 'none'.")
    print("Leave empty to keep current settings.")

    while True:
        try:
            choice_str = input("> ").strip()
            if not choice_str:
                print(f"Keeping current enabled toolsets: {', '.join(sorted(current_enabled_names)) or 'None'}")
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
                        print(f"Error: Invalid selection '{index}'. Please use numbers from 1 to {idx-1}.")
                        valid_selection = False
                        break
                if not valid_selection: continue # Ask again

            # Remove duplicates and sort
            new_enabled_list_names = sorted(list(set(new_enabled_list_names)))

            # Update state - this also handles deactivating toolsets & prompt update implicitly
            update_enabled_toolsets(chat_file, new_enabled_list_names) # Pass list of names
            print(f"\nEnabled toolsets for '{chat_title}' set to: {', '.join(new_enabled_list_names) or 'None'}")

            # --- Rebuild and save the system prompt ---
            # Use the SYSTEM_PROMPT constant directly
            success = _update_message_in_chat(chat_file, 0, {"role": "system", "content": SYSTEM_PROMPT})
            if success: logger.info(f"System prompt updated for chat {chat_file} after toolset selection.")
            else: logger.error(f"Failed to update system prompt for chat {chat_file} after toolset selection.")
            print("System prompt updated. Changes apply on next interaction.")
            return

        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled. No changes made.")
            return

# --- Main CLI Execution Logic ---
def main():
    env_path = os.path.join(get_install_dir(), '.env'); load_dotenv(env_path)
    parser = argparse.ArgumentParser(description="AI Shell Agent CLI", formatter_class=argparse.RawTextHelpFormatter)
    # --- Argument Groups ---
    model_group = parser.add_argument_group('Model Configuration')
    chat_group = parser.add_argument_group('Chat Management')
    tool_group = parser.add_argument_group('Toolset Configuration')
    editor_group = parser.add_argument_group('File Editor (Aider) Configuration')
    msg_group = parser.add_argument_group('Messaging & Interaction')

    # --- Arguments ---
    model_group.add_argument("-llm", "--model", help="Set LLM model")
    model_group.add_argument("--select-model", action="store_true", help="Interactively select LLM model")
    model_group.add_argument("-k", "--set-api-key", nargs="?", const=True, metavar="API_KEY", help="Set API key for current model")

    chat_group.add_argument("-c", "--chat", metavar="TITLE", help="Create or load chat session")
    chat_group.add_argument("-lc", "--load-chat", metavar="TITLE", help="Load chat (same as -c)")
    chat_group.add_argument("-lsc", "--list-chats", action="store_true", help="List chats")
    chat_group.add_argument("-rnc", "--rename-chat", nargs=2, metavar=("OLD", "NEW"), help="Rename chat")
    chat_group.add_argument("-delc", "--delete-chat", metavar="TITLE", help="Delete chat")
    chat_group.add_argument("--temp-flush", action="store_true", help="Remove temp chats")
    chat_group.add_argument("-ct", "--current-chat-title", action="store_true", help="Print current chat title")

    tool_group.add_argument("--select-tools", action="store_true", help="Interactively select enabled toolsets for the CURRENT chat") # KEPT
    tool_group.add_argument("--list-toolsets", action="store_true", help="List available toolsets and their status") 

    editor_group.add_argument("--select-edit-mode", action="store_true", help="Select File Editor edit format")
    editor_group.add_argument("--select-coder-models", action="store_true", help="Select File Editor models")

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
        print(f"Model set to {get_current_model()}.")
        return
    if args.select_model:
        m = prompt_for_model_selection()
        if m and m != get_current_model():
            set_model(m)
            ensure_api_key()
            print(f"Model set to {get_current_model()}.")
        elif m:
            print(f"Model remains {get_current_model()}.")
        else:
            print("Model selection cancelled.")
        return

    # 2. First Time Setup
    first_time_setup()

    # 3. API Key Management
    if args.set_api_key:
        set_api_key_for_model(get_current_model(), args.set_api_key if isinstance(args.set_api_key, str) else None)
        return

    # 4. Ensure API Key (Exit if missing)
    if not ensure_api_key():
        logger.critical("API Key missing.")
        print("\nError: API Key missing. Set with --set-api-key.")
        sys.exit(1)

    # 5. Toolset Selection (needs chat context)
    if args.select_tools:
        select_tools_for_chat()
        return
    if args.list_toolsets: 
        from .chat_manager import list_toolsets
        list_toolsets()
        return

    # 6. Editor Config Wizards
    if args.select_edit_mode:
        prompt_for_edit_mode_selection()
        return
    if args.select_coder_models:
        prompt_for_coder_models_selection()
        ensure_api_keys_for_coder_models()
        return

    # 7. Direct Command Execution
    if args.execute:
        execute(args.execute)
        return

    # 8. Chat Management
    if args.chat:
        chat_file = create_or_load_chat_state(args.chat)
        if chat_file:
            print(f"Switched to chat: '{args.chat}'.")
        return
    if args.load_chat:
        chat_file = create_or_load_chat_state(args.load_chat)
        if chat_file:
            print(f"Switched to chat: '{args.load_chat}'.")
        return
    if args.current_chat_title:
        title = get_current_chat_title_state()
        print(f"Current chat: {title}" if title else "No active chat.")
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
    active_chat_file = get_current_chat()

    # 9. Messaging / History
    if args.list_messages:
        if not active_chat_file:
            print("No active chat. Use -c <title> first.")
            return
        list_messages()
        return
    if args.edit:
        if not active_chat_file:
            print("No active chat. Use -c <title> first.")
            return
        idx_str, msg_parts = args.edit[0], args.edit[1:]
        new_msg = " ".join(msg_parts)
        idx = None
        if idx_str.lower() != "last":
            try:
                idx = int(idx_str)
            except ValueError:
                print(f"Error: Invalid index '{idx_str}'. Must be integer or 'last'.")
                return
        edit_message(idx, new_msg)
        return

    # 10. Sending Messages (Default actions)
    msg_to_send = args.send_message or args.message # Prioritize -m
    if msg_to_send:
        if not active_chat_file and not args.temp_chat:
            # If no active chat and not explicitly temp
            print("No active chat. Starting temporary chat...")
            start_temp_chat(msg_to_send)
        elif active_chat_file:
            send_message(msg_to_send)
        # If args.temp_chat is set, it's handled below
        return
    if args.temp_chat: # Handles -tc explicitly
        start_temp_chat(args.temp_chat)
        return

    # 11. No arguments provided
    parser.print_help()

if __name__ == "__main__":
    main()
