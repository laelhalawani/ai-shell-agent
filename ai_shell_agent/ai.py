import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env if available.
load_dotenv()

from . import logger

# Ensure the API key is set before any other operations
def get_api_key() -> str:
    """Retrieve the OpenAI API key from the environment."""
    return os.getenv("OPENAI_API_KEY")

def set_api_key(api_key: str = None) -> None:
    """
    Prompt the user for an OpenAI API key and save it to the .env file.
    Aborts if no key is entered.
    """
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    if not api_key:
        logger.warning("No API key entered. Aborting.")
        return
    os.environ["OPENAI_API_KEY"] = api_key

    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("")

    try:
        with open(env_path, "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        logger.info("API key saved successfully to .env")
    except Exception as e:
        logger.error(f"Failed to write to .env: {e}")

def ensure_api_key() -> None:
    """
    Ensure that the OpenAI API key is set. If not, prompt the user to enter it.
    """
    if not get_api_key():
        logger.warning("OpenAI API key not found. Please enter your API key.")
        set_api_key()

ensure_api_key()

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
# API Key Management
# ---------------------------
def get_api_key() -> str:
    """Retrieve the OpenAI API key from the environment."""
    return os.getenv("OPENAI_API_KEY")

def set_api_key(api_key: str = None) -> None:
    """
    Prompt the user for an OpenAI API key and save it to the .env file.
    Aborts if no key is entered.
    """
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    if not api_key:
        logger.warning("No API key entered. Aborting.")
        return
    os.environ["OPENAI_API_KEY"] = api_key

    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("")

    try:
        with open(env_path, "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        logger.info("API key saved successfully to .env")
    except Exception as e:
        logger.error(f"Failed to write to .env: {e}")

def ensure_api_key() -> None:
    """
    Ensure that the OpenAI API key is set. If not, prompt the user to enter it.
    """
    if not get_api_key():
        logger.warning("OpenAI API key not found. Please enter your API key.")
        set_api_key()

# ---------------------------
# CLI Command Handling
# ---------------------------
def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    ensure_api_key()
    parser = argparse.ArgumentParser(
        description="AI Command-Line Chat Application"
    )
    # API Key Management
    parser.add_argument("-k", "--set-api-key", nargs="?", const=True, help="Set or update the OpenAI API key")
    
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

    # Handle API key management
    if args.set_api_key:
        if isinstance(args.set_api_key, str):
            set_api_key(args.set_api_key)
        else:
            set_api_key()
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
