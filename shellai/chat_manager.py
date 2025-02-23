import os
import json
import uuid

CHAT_DIR = os.path.join("chats")
CHAT_MAP_FILE = os.path.join(CHAT_DIR, "chat_map.json")
SESSION_FILE = "session.json"
CONFIG_FILE = "config.json"

# Ensure the chats directory exists.
os.makedirs(CHAT_DIR, exist_ok=True)

def _read_json(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def _write_json(file_path: str, data: dict) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# ---------------------------
# Chat Session Management
# ---------------------------
def create_or_load_chat(title: str) -> str:
    """
    Creates or loads a chat session file based on the title.
    Chat sessions are stored in the 'chats' directory with a UUID as the filename.
    
    Parameters:
      title (str): The chat session title.
      
    Returns:
      str: The filepath of the chat session JSON file.
    """
    chat_map = _read_json(CHAT_MAP_FILE)
    if title in chat_map:
        chat_id = chat_map[title]
    else:
        chat_id = str(uuid.uuid4())
        chat_map[title] = chat_id
        _write_json(CHAT_MAP_FILE, chat_map)
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if not os.path.exists(chat_file):
        _write_json(chat_file, [])
    return chat_file

def list_chats() -> list:
    """Returns a list of all chat session titles."""
    chat_map = _read_json(CHAT_MAP_FILE)
    return list(chat_map.keys())

def rename_chat(old_title: str, new_title: str) -> bool:
    """
    Renames an existing chat session.
    
    Parameters:
      old_title (str): The current chat title.
      new_title (str): The new chat title.
      
    Returns:
      bool: True if successful, False otherwise.
    """
    chat_map = _read_json(CHAT_MAP_FILE)
    if old_title in chat_map:
        chat_map[new_title] = chat_map.pop(old_title)
        _write_json(CHAT_MAP_FILE, chat_map)
        return True
    return False

def delete_chat(title: str) -> bool:
    """
    Deletes a chat session.
    
    Parameters:
      title (str): The title of the chat to delete.
      
    Returns:
      bool: True if successful, False otherwise.
    """
    chat_map = _read_json(CHAT_MAP_FILE)
    if title in chat_map:
        chat_id = chat_map.pop(title)
        _write_json(CHAT_MAP_FILE, chat_map)
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
        return True
    return False

def save_session(chat_file: str) -> None:
    """
    Saves the active chat session to session.json.
    
    Parameters:
      chat_file (str): The filepath of the active chat session.
    """
    _write_json(SESSION_FILE, {"current_chat": chat_file})

def load_session() -> str:
    """
    Loads the active chat session from session.json.
    
    Returns:
      str: The filepath of the active chat session, or None if not set.
    """
    data = _read_json(SESSION_FILE)
    return data.get("current_chat", None)

# ---------------------------
# Messaging Functions
# ---------------------------
def send_message(message: str) -> str:
    """
    Appends a user message to the active chat session and simulates an AI response.
    
    Parameters:
      message (str): The user message.
      
    Returns:
      str: A simulated AI response.
    """
    chat_file = load_session()
    if not chat_file:
        return "No active chat session. Please start a new chat using --chat."
    chat_history = _read_json(chat_file)
    chat_history.append({"role": "user", "content": message})
    _write_json(chat_file, chat_history)
    response = f"AI: You said, '{message}'"
    chat_history.append({"role": "ai", "content": response})
    _write_json(chat_file, chat_history)
    return response

def edit_message(index: int, new_message: str) -> bool:
    """
    Edits a previous message at the given index and truncates subsequent messages.
    
    Parameters:
      index (int): The index of the message to edit.
      new_message (str): The new content for the message.
      
    Returns:
      bool: True if successful, False otherwise.
    """
    chat_file = load_session()
    if not chat_file:
        return False
    chat_history = _read_json(chat_file)
    if index < 0 or index >= len(chat_history):
        return False
    chat_history[index] = {"role": chat_history[index]["role"], "content": new_message}
    chat_history = chat_history[: index + 1]
    _write_json(chat_file, chat_history)
    return True

def start_temp_chat(message: str) -> str:
    """
    Starts a temporary (in-memory) chat session, processes the initial message,
    and returns a simulated AI response.
    
    Parameters:
      message (str): The initial message.
      
    Returns:
      str: The simulated AI response.
    """
    chat_history = [{"role": "user", "content": message}]
    response = f"AI (temp): You said, '{message}'"
    chat_history.append({"role": "ai", "content": response})
    return response

# ---------------------------
# System Prompt Management
# ---------------------------
def set_default_system_prompt(prompt_text: str) -> None:
    """
    Sets the default system prompt in config.json.
    
    Parameters:
      prompt_text (str): The default system prompt.
    """
    config = _read_json(CONFIG_FILE)
    config["default_system_prompt"] = prompt_text
    _write_json(CONFIG_FILE, config)

def update_system_prompt(prompt_text: str) -> None:
    """
    Updates the system prompt for the active chat session.
    
    Parameters:
      prompt_text (str): The new system prompt.
    """
    chat_file = load_session()
    if not chat_file:
        print("No active chat session to update.")
        return
    chat_history = _read_json(chat_file)
    system_message = {"role": "system", "content": prompt_text}
    chat_history.insert(0, system_message)
    _write_json(chat_file, chat_history)

