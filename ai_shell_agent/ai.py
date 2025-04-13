import os
import json
import argparse
from dotenv import load_dotenv

# Get installation directory
def get_install_dir():
    """Return the installation directory of the package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env in the installation directory
env_path = os.path.join(get_install_dir(), '.env')
load_dotenv(env_path)

from . import logger
from .config_manager import (
    get_current_model, 
    set_model, 
    prompt_for_model_selection, 
    ensure_api_key_for_current_model,
    get_api_key_for_model,
    set_api_key_for_model,
    get_model_provider,
    check_if_first_run,
    prompt_for_edit_mode_selection,  # New import for AI Code Editor edit format selection
    prompt_for_coder_models_selection,  # New import for AI Code Editor coder models selection
    get_aider_main_model,  # Added for API key check
    get_aider_editor_model,  # Added for API key check
    get_aider_weak_model,   # Added for API key check
    set_aider_main_model,   # Added for setting default models
    set_aider_editor_model, # Added for setting default models 
    set_aider_weak_model    # Added for setting default models
)

# First time setup to ensure model is selected before API key
def first_time_setup():
    """
    Initialize the application on first run by prompting for model selection.
    """
    if check_if_first_run():
        logger.info("Welcome to AI Shell Agent! Please select a model to use.")
        selected_model = prompt_for_model_selection()
        set_model(selected_model)
        
        # Also set up AI Code Editor models on first run
        logger.info("Would you like to configure AI Code Editor models now?")
        setup_coder = input("Configure AI Code Editor models? (y/n, default: y): ").strip().lower()
        if not setup_coder or setup_coder.startswith('y'):
            # Set default coder models if user wants to configure
            # Default to o3-mini for editor and weak models if not set
            if not get_aider_main_model():
                set_aider_main_model(selected_model)  # Use the same model as agent by default
            if not get_aider_editor_model():
                set_aider_editor_model("o3-mini")     # Default to o3-mini for editor
            if not get_aider_weak_model():
                set_aider_weak_model("o3-mini")       # Default to o3-mini for weak model
            
            # Run the full configuration wizard
            prompt_for_coder_models_selection()
            
            # After models are set, ensure API keys for all selected models
            ensure_api_keys_for_coder_models()

# Ensure the API key is set before any other operations
def ensure_api_key() -> None:
    """
    Ensure that the appropriate API key is set for the selected model.
    """
    ensure_api_key_for_current_model()

# Ensure API keys for all model types
def ensure_api_keys_for_coder_models() -> None:
    """
    Check and ensure API keys are set for all models used by the AI Code Editor.
    This includes the main model, editor model, and weak model if they're configured.
    """
    # Check main agent model first (already checked by ensure_api_key)
    main_agent_model = get_current_model()
    
    # Check main code editor model if different from agent model
    main_coder_model = get_aider_main_model()
    if main_coder_model and main_coder_model != main_agent_model:
        logger.info(f"Checking API key for AI Code Editor main model: {main_coder_model}")
        api_key, env_var = get_api_key_for_model(main_coder_model)
        if not api_key:
            logger.info(f"API key needed for {main_coder_model}")
            set_api_key_for_model(main_coder_model)
    
    # Check editor model if configured
    editor_model = get_aider_editor_model()
    if editor_model and editor_model != main_agent_model and editor_model != main_coder_model:
        logger.info(f"Checking API key for AI Code Editor editor model: {editor_model}")
        api_key, env_var = get_api_key_for_model(editor_model)
        if not api_key:
            logger.info(f"API key needed for {editor_model}")
            set_api_key_for_model(editor_model)
    
    # Check weak model if configured
    weak_model = get_aider_weak_model()
    if weak_model and weak_model != main_agent_model and weak_model != main_coder_model and weak_model != editor_model:
        logger.info(f"Checking API key for AI Code Editor weak model: {weak_model}")
        api_key, env_var = get_api_key_for_model(weak_model)
        if not api_key:
            logger.info(f"API key needed for {weak_model}")
            set_api_key_for_model(weak_model)

from .chat_manager import (
    create_or_load_chat,
    get_chat_titles_list,
    rename_chat,
    delete_chat,
    load_session,
    save_session,
    send_message,
    edit_message,
    start_temp_chat,
    set_default_system_prompt,
    update_system_prompt,
    flush_temp_chats,
    execute,
    list_messages,
    current_chat_title
)

# ---------------------------
# CLI Command Handling
# ---------------------------
def main():
    # Load environment variables from the installation directory
    env_path = os.path.join(get_install_dir(), '.env')
    load_dotenv(env_path)
    
    # Parse arguments first before any prompting
    parser = argparse.ArgumentParser(
        description="AI Command-Line Chat Application"
    )
    # API Key Management
    parser.add_argument("-k", "--set-api-key", nargs="?", const=True, help="Set or update the API key for the current model")
    
    # Model selection
    parser.add_argument("-llm", "--model", help="Set the LLM model to use")
    parser.add_argument("--select-model", action="store_true", help="Interactively select an LLM model")
    
    # AI Code Editor Configuration Wizards
    parser.add_argument("--select-edit-mode", action="store_true", help="Interactively select the edit format the AI Code Editor should use.")
    parser.add_argument("--select-coder-models", action="store_true", help="Interactively select the models the AI Code Editor should use for coding tasks.")
    
    # Chat management options
    parser.add_argument("-c", "--chat", help="Create or load a chat session with the specified title")
    parser.add_argument("-lc", "--load-chat", help="Load an existing chat session with the specified title")
    parser.add_argument("-lsc", "--list-chats", action="store_true", help="List all available chat sessions")
    parser.add_argument("-rnc", "--rename-chat", nargs=2, metavar=("OLD_TITLE", "NEW_TITLE"), help="Rename a chat session")
    parser.add_argument("-delc", "--delete-chat", help="Delete a chat session with the specified title")
    
    # System prompt management
    parser.add_argument("--default-system-prompt", help="Set the default system prompt for new chats")
    parser.add_argument("--system-prompt", help="Update the system prompt for the active chat session")
    
    # Messaging commands
    parser.add_argument("-m", "--send-message", help="Send a message to the active chat session")
    parser.add_argument("-tc", "--temp-chat", help="Start a temporary (in-memory) chat session with the initial message")
    parser.add_argument("-e", "--edit", nargs="+", metavar=("INDEX", "NEW_MESSAGE"), help="Edit a previous message at the given index")
    parser.add_argument("--temp-flush", action="store_true", help="Removes all temp chat sessions")
    
    # Add direct command execution
    parser.add_argument("-x", "--execute", help="Execute a shell command preserving its context for AI")
    
    # Print the chat history
    parser.add_argument("-lsm", "--list-messages", action="store_true", help="Print the chat history")
    
    parser.add_argument("-ct", "--current-chat-title", action="store_true", help="Print the current chat title")
    
    # Fallback: echo a simple message.
    parser.add_argument("message", nargs="?", help="Send a message (if no other options are provided)")

    args = parser.parse_args()

    # Model selection commands need to run before API key checks
    if args.model:
        set_model(args.model)
        ensure_api_key()
        return
    
    if args.select_model:
        selected_model = prompt_for_model_selection()
        if (selected_model):
            set_model(selected_model)
            ensure_api_key()
        return

    # Run first-time setup ONLY if no model selection arguments were provided
    first_time_setup()
    
    # Handle API key management
    if args.set_api_key:
        current_model = get_current_model()
        if isinstance(args.set_api_key, str):
            set_api_key_for_model(current_model, args.set_api_key)
        else:
            set_api_key_for_model(current_model)
        return

    # Check API key before executing any command
    ensure_api_key()
    
    # Handle Aider configuration wizards if arguments are present
    if args.select_edit_mode:
        prompt_for_edit_mode_selection()
        return
        
    if args.select_coder_models:
        prompt_for_coder_models_selection()
        # Check API keys for all selected models after configuration
        ensure_api_keys_for_coder_models()
        return
    
    # Handle direct command execution
    if args.execute:
        output = execute(args.execute)
        return

    # Chat session management
    if args.chat:
        chat_file = create_or_load_chat(args.chat)
        save_session(chat_file)
        return
    
    if args.current_chat_title:
        current_chat_title()
        return

    if args.load_chat:
        chat_file = create_or_load_chat(args.load_chat)
        save_session(chat_file)
        return

    if args.list_chats:
        get_chat_titles_list()
        return

    if args.rename_chat:
        old_title, new_title = args.rename_chat
        rename_chat(old_title, new_title)
        return

    if args.delete_chat:
        delete_chat(args.delete_chat)
        return

    # System prompt management
    if args.default_system_prompt:
        set_default_system_prompt(args.default_system_prompt)
        return

    if args.system_prompt:
        update_system_prompt(args.system_prompt)
        return

    # Messaging commands
    if args.send_message:
        send_message(args.send_message)
        return

    if args.temp_chat:
        start_temp_chat(args.temp_chat)
        return

    if args.edit:
        if len(args.edit) == 1:
            new_message = args.edit[0]
            edit_message(None, new_message)
        elif len(args.edit) == 2:
            index, new_message = args.edit
            if index.lower() == "last":
                edit_message(None, new_message)
            else:
                edit_message(int(index), new_message)
        else:
            logger.error("Invalid number of arguments for --edit")
        return

    if args.temp_flush:
        flush_temp_chats()
        return
    
    # Print chat history
    if args.list_messages:
        list_messages()
        return
    # Fallback: if a message is provided without other commands, send it to current chat
    if args.message:
        # Use send_message which handles chat history properly
        send_message(args.message)
        return
    else:
        logger.info("No command provided. Use --help for options.")
        return

if __name__ == "__main__":
    main()
