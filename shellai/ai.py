import os
import json
import argparse
from dotenv import load_dotenv
from .chat_manager import (
    create_or_load_chat,
    list_chats,
    rename_chat,
    delete_chat,
    load_session,
    save_session,
    send_message,
    edit_message,
    start_temp_chat,
    set_default_system_prompt,
    update_system_prompt,
)

# Load environment variables from .env if available.
load_dotenv()

# ---------------------------
# API Key Management
# ---------------------------
def get_api_key() -> str:
    """Retrieve the OpenAI API key from the environment."""
    return os.getenv("OPENAI_API_KEY")

def set_api_key() -> None:
    """
    Prompt the user for an OpenAI API key and save it to the .env file.
    Aborts if no key is entered.
    """
    api_key = input("Enter OpenAI API key: ").strip()
    if not api_key:
        print("No API key entered. Aborting.")
        return
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print("API key saved successfully to .env")
    except Exception as e:
        print(f"Failed to write to .env: {e}")

# ---------------------------
# CLI Command Handling
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AI Command-Line Chat Application"
    )
    # API Key Management
    parser.add_argument("-k", "--set-api-key", action="store_true", help="Set or update the OpenAI API key")
    
    # Chat management options
    parser.add_argument("-c", "--chat", help="Create or load a chat session with the specified title")
    parser.add_argument("--load-chat", help="Load an existing chat session with the specified title")
    parser.add_argument("--list-chats", action="store_true", help="List all available chat sessions")
    parser.add_argument("--rename-chat", nargs=2, metavar=("OLD_TITLE", "NEW_TITLE"), help="Rename a chat session")
    parser.add_argument("--delete-chat", help="Delete a chat session with the specified title")
    
    # System prompt management
    parser.add_argument("--default-system-prompt", help="Set the default system prompt for new chats")
    parser.add_argument("--system-prompt", help="Update the system prompt for the active chat session")
    
    # Messaging commands
    parser.add_argument("-m", "--send-message", help="Send a message to the active chat session")
    parser.add_argument("-t", "--temp-chat", help="Start a temporary (in-memory) chat session with the initial message")
    parser.add_argument("--edit", nargs=2, metavar=("INDEX", "NEW_MESSAGE"), help="Edit a previous message at the given index")
    
    # Fallback: echo a simple message.
    parser.add_argument("message", nargs="?", help="Send a message (if no other options are provided)")

    args = parser.parse_args()

    # Handle API key management
    if args.set_api_key:
        set_api_key()
        return

    # Chat session management
    if args.chat:
        chat_file = create_or_load_chat(args.chat)
        save_session(chat_file)
        print(f"Chat session '{args.chat}' is now active.")
        return

    if args.load_chat:
        chat_file = create_or_load_chat(args.load_chat)
        save_session(chat_file)
        print(f"Chat session '{args.load_chat}' loaded and is now active.")
        return

    if args.list_chats:
        chats = list_chats()
        print("Available chats:")
        for title in chats:
            print(f" - {title}")
        return

    if args.rename_chat:
        old_title, new_title = args.rename_chat
        if rename_chat(old_title, new_title):
            print(f"Chat '{old_title}' renamed to '{new_title}'.")
        else:
            print("Rename operation failed.")
        return

    if args.delete_chat:
        if delete_chat(args.delete_chat):
            print(f"Chat '{args.delete_chat}' deleted.")
        else:
            print("Delete operation failed.")
        return

    # System prompt management
    if args.default_system_prompt:
        set_default_system_prompt(args.default_system_prompt)
        print("Default system prompt updated.")
        return

    if args.system_prompt:
        update_system_prompt(args.system_prompt)
        print("Active chat session system prompt updated.")
        return

    # Messaging commands
    if args.send_message:
        response = send_message(args.send_message)
        print("Response:")
        print(response)
        return

    if args.temp_chat:
        response = start_temp_chat(args.temp_chat)
        print("Temporary chat response:")
        print(response)
        return

    if args.edit:
        index, new_message = args.edit
        if edit_message(int(index), new_message):
            print("Message edited successfully.")
        else:
            print("Editing message failed.")
        return

    # Fallback: echo message if provided
    if args.message:
        print(f"Message received: {args.message}")
    else:
        print("No command provided. Use --help for options.")

    # Post-installation message
    print("ai-shell-agent installed successfully. Use 'ai --help' for usage information.")

if __name__ == "__main__":
    main()
