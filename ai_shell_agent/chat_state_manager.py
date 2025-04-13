"""
Manages persistent state related to chat sessions, including
session tracking, chat file I/O, Aider state, chat mapping, and active toolsets.
"""
import os
import json
import uuid
import traceback
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from datetime import datetime, timezone # ADDED IMPORT

# Local imports
from . import logger
# Import the prompt builder
from .prompts.prompts import build_prompt # MODIFIED IMPORT

# --- Constants ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHATS_DIR = os.path.join(DATA_DIR, "chats")
SESSION_FILE = os.path.join(DATA_DIR, "session.json")
CHAT_MAP_FILE = os.path.join(CHATS_DIR, "chat_map.json")
AIDER_STATE_KEY = "_aider_state"
# NEW: Metadata key for toolsets
ACTIVE_TOOLSETS_KEY = "active_toolsets"
DEFAULT_TOOLSETS = ["Terminal"] # Start with Terminal active by default

# Ensure directories exist
os.makedirs(CHATS_DIR, exist_ok=True)

# --- Low-Level JSON Helpers ---
def _read_json(file_path: str, default_value=None) -> Any:
    """Reads a JSON file or returns a default value if not found."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if isinstance(e, FileNotFoundError):
            logger.debug(f"File not found: {file_path}")
        else:
            logger.error(f"Error parsing JSON from {file_path}: {e}")
        return {} if default_value is None else default_value

def _write_json(file_path: str, data: Any) -> None:
    """Writes data to a JSON file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        logger.error(traceback.format_exc())

# --- Chat File Data Helpers ---
def _read_chat_data(chat_file: str) -> Dict:
    """Reads the data for a specific chat session file."""
    if not chat_file: # Added check
         logger.error("Attempted to read chat data with empty chat_file.")
         return {"messages": [], "metadata": {ACTIVE_TOOLSETS_KEY: list(DEFAULT_TOOLSETS)}}
    chat_path = os.path.join(CHATS_DIR, f"{chat_file}.json")
    # Ensure default structure includes metadata with active toolsets
    default_data = {"messages": [], "metadata": {ACTIVE_TOOLSETS_KEY: list(DEFAULT_TOOLSETS)}}
    return _read_json(chat_path, default_data)

def _write_chat_data(chat_file: str, data: Dict) -> None:
    """Writes data for a specific chat session file."""
    if not chat_file: # Added check
         logger.error("Attempted to write chat data with empty chat_file.")
         return
    chat_path = os.path.join(CHATS_DIR, f"{chat_file}.json")
    _write_json(chat_path, data)

def _get_chat_metadata(chat_file: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Gets metadata for a specific chat session file."""
    chat_data = _read_chat_data(chat_file)
    metadata = chat_data.get("metadata", {})
    if key is None:
        return metadata
    return metadata.get(key, default)

def _update_metadata_in_chat(chat_file: str, key: str, value: Any) -> None:
    """Updates a metadata value for a chat file."""
    chat_data = _read_chat_data(chat_file)
    chat_data.setdefault("metadata", {})[key] = value
    _write_chat_data(chat_file, chat_data)

# --- Chat Map Helpers ---
def _read_chat_map() -> Dict[str, str]:
    """Reads the chat UUID to title mapping."""
    return _read_json(CHAT_MAP_FILE, {})

def _write_chat_map(chat_map: Dict[str, str]) -> None:
    """Writes the chat UUID to title mapping."""
    _write_json(CHAT_MAP_FILE, chat_map)

# --- Session Management ---
def get_current_chat() -> Optional[str]:
    """Gets the current chat session filename (UUID without extension)."""
    session_data = _read_json(SESSION_FILE, {})
    return session_data.get("current_chat")

def save_session(chat_file: str) -> None:
    """Saves the current session information (which chat file is active)."""
    _write_json(SESSION_FILE, {"current_chat": chat_file})

# --- Helper Functions ---
def _get_console_session_id() -> str:
    """Get a unique ID for single-use console sessions."""
    return "console_" + str(uuid.uuid4())

# --- Aider State Management ---
def get_aider_state(chat_file: str) -> Optional[Dict]:
    """Gets the Aider state dictionary for a chat session."""
    if not chat_file:
        return None
    return _get_chat_metadata(chat_file, AIDER_STATE_KEY)

def save_aider_state(chat_file: str, state: Dict) -> None:
    """Saves the Aider state dictionary for a chat session."""
    if not chat_file:
        logger.error("Cannot save Aider state: No active chat session.")
        return
    _update_metadata_in_chat(chat_file, AIDER_STATE_KEY, state)

def clear_aider_state(chat_file: str) -> None:
    """Clears the Aider state for a chat session by setting 'enabled' to False."""
    if not chat_file:
        logger.error("Cannot clear Aider state: No active chat session.")
        return
    # Get existing state or create new if not present
    current_state = get_aider_state(chat_file) or {}
    current_state['enabled'] = False # Mark as disabled
    # Remove potentially large fields to save space
    current_state.pop('aider_done_messages', None)
    current_state.pop('abs_fnames', None)
    current_state.pop('abs_read_only_fnames', None)
    _update_metadata_in_chat(chat_file, AIDER_STATE_KEY, current_state)
    logger.info(f"Cleared Aider state for chat {chat_file}")

# --- Chat Creation/Management ---
def create_or_load_chat(title: str) -> str:
    """
    Creates a new chat or loads an existing one by title.
    Initializes new chats with default toolsets and dynamically built prompt.
    Ensures existing chats have the active_toolsets metadata key.
    """
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}

    if title in title_to_id:
        chat_file = title_to_id[title]
        logger.info(f"Loading existing chat: {title} ({chat_file})")
        # Ensure existing chats have the toolset key and valid system prompt
        chat_data = _read_chat_data(chat_file)
        metadata = chat_data.get("metadata", {})
        messages = chat_data.get("messages", [])
        needs_update = False

        if ACTIVE_TOOLSETS_KEY not in metadata or not isinstance(metadata.get(ACTIVE_TOOLSETS_KEY), list):
             current_toolsets = list(DEFAULT_TOOLSETS)
             logger.warning(f"Chat '{title}' missing/invalid '{ACTIVE_TOOLSETS_KEY}'. Initializing with default: {current_toolsets}")
             metadata[ACTIVE_TOOLSETS_KEY] = current_toolsets
             needs_update = True
        else:
             current_toolsets = metadata[ACTIVE_TOOLSETS_KEY] # Use existing valid toolsets

        # Also check if system prompt exists and rebuild if necessary (e.g., if toolsets were missing)
        if not messages or messages[0].get("role") != "system" or needs_update:
             new_system_prompt = build_prompt(active_toolsets=current_toolsets)
             if not messages or messages[0].get("role") != "system":
                 logger.warning(f"Chat '{title}' missing system prompt at index 0. Prepending.")
                 messages.insert(0, {"role": "system", "content": new_system_prompt})
             else: # Update existing system prompt
                 logger.info(f"Updating system prompt for chat '{title}' based on toolsets.")
                 messages[0]["content"] = new_system_prompt
             needs_update = True

        if needs_update:
             chat_data["metadata"] = metadata
             chat_data["messages"] = messages
             _write_chat_data(chat_file, chat_data) # Save updated data

    else:
        # Create a new chat
        chat_file = str(uuid.uuid4())
        chat_map[chat_file] = title
        _write_chat_map(chat_map)
        logger.info(f"Creating new chat: {title} ({chat_file})")

        # Initialize chat data
        initial_toolsets = list(DEFAULT_TOOLSETS)
        initial_system_prompt = build_prompt(active_toolsets=initial_toolsets)

        chat_data = {
            "messages": [
                {"role": "system", "content": initial_system_prompt}
            ],
            "metadata": {
                ACTIVE_TOOLSETS_KEY: initial_toolsets,
                # Use ISO format for consistency with message timestamps
                "created_at": datetime.now(timezone.utc).isoformat(),
                "title": title
            }
        }
        _write_chat_data(chat_file, chat_data)
        logger.info(f"Created new chat {title} with toolsets {initial_toolsets}.")

    # Save this as the current session
    save_session(chat_file)
    return chat_file

def rename_chat(old_title: str, new_title: str) -> bool:
    """Renames a chat session. Returns True on success, False on failure."""
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}

    if old_title not in title_to_id:
        logger.error(f"Chat not found for renaming: {old_title}")
        return False

    chat_file = title_to_id[old_title]
    # Check if new title already exists
    if new_title in title_to_id and title_to_id[new_title] != chat_file:
         logger.error(f"Cannot rename: chat title '{new_title}' already exists.")
         return False

    chat_map[chat_file] = new_title
    _write_chat_map(chat_map)
    logger.info(f"Renamed chat: {old_title} -> {new_title}")
    return True

def delete_chat(title: str) -> bool:
    """Deletes a chat session. Returns True on success, False on failure."""
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}

    if title not in title_to_id:
        logger.error(f"Chat not found for deletion: {title}")
        return False

    chat_file = title_to_id[title]
    # Remove from chat map
    del chat_map[chat_file]
    _write_chat_map(chat_map)

    # Delete the chat file
    chat_path = os.path.join(CHATS_DIR, f"{chat_file}.json")
    try:
        os.remove(chat_path)
        logger.info(f"Deleted chat: {title}")
        # If the deleted chat was the current one, clear the session
        if get_current_chat() == chat_file:
             save_session(None) # Clear current session
        return True
    except Exception as e:
        logger.warning(f"Could not delete chat file {chat_path}: {e}")
        return False # Indicate failure if file deletion fails

def get_chat_titles() -> Dict[str, str]:
    """Returns the chat map (ID to Title)."""
    return _read_chat_map()

def get_current_chat_title() -> Optional[str]:
    """Gets the title of the current chat session."""
    chat_file = get_current_chat()
    if not chat_file:
        return None
    chat_map = _read_chat_map()
    return chat_map.get(chat_file)

def flush_temp_chats() -> int:
    """
    Removes all temporary chat sessions. Returns the number removed.
    """
    chat_map = _read_chat_map()
    temp_chats_to_remove = {
        chat_id: title
        for chat_id, title in chat_map.items()
        if title.startswith("Temp Chat ")
    }
    removed_count = 0

    if not temp_chats_to_remove:
        return 0

    current_session = get_current_chat()
    clear_current = False

    for chat_id, title in temp_chats_to_remove.items():
        del chat_map[chat_id]
        chat_path = os.path.join(CHATS_DIR, f"{chat_id}.json")
        try:
            os.remove(chat_path)
            logger.info(f"Removed temporary chat: {title} ({chat_id})")
            removed_count += 1
            if current_session == chat_id:
                 clear_current = True
        except OSError as e:
            logger.warning(f"Could not delete temporary chat file {chat_path}: {e}")

    _write_chat_map(chat_map)
    if clear_current:
        save_session(None) # Clear current session if it was a temp one

    logger.info(f"Flushed {removed_count} temporary chats.")
    return removed_count

# --- Active Toolsets Management ---
def get_active_toolsets(chat_file: str) -> List[str]:
    """
    Gets the list of active toolsets for a specific chat session.
    Returns default toolsets if not set or on error.
    """
    if not chat_file:
        logger.warning("get_active_toolsets called with empty chat_file. Returning default.")
        return list(DEFAULT_TOOLSETS) # Return a copy
    toolsets = _get_chat_metadata(chat_file, ACTIVE_TOOLSETS_KEY, default=None)
    if toolsets is None or not isinstance(toolsets, list):
        logger.warning(f"'{ACTIVE_TOOLSETS_KEY}' not found or invalid in chat {chat_file}. Returning default.")
        # Optionally update the state here? Or just return default. Let's return default for now.
        return list(DEFAULT_TOOLSETS) # Return a copy
    return toolsets

def update_active_toolsets(chat_file: str, toolsets: List[str]) -> None:
    """
    Updates the list of active toolsets for a specific chat session.
    Ensures the list contains unique elements.
    """
    if not chat_file:
        logger.error("update_active_toolsets called with empty chat_file.")
        return
    # Ensure unique toolsets
    unique_toolsets = sorted(list(set(toolsets)))
    _update_metadata_in_chat(chat_file, ACTIVE_TOOLSETS_KEY, unique_toolsets)
    logger.info(f"Active toolsets updated for chat {chat_file}: {unique_toolsets}")

# --- Message Access Functions ---
def _get_chat_messages(chat_file: str) -> List[Dict]:
    """Gets the messages for a specific chat session."""
    if not chat_file:
        logger.error("Cannot get messages: No chat file specified.")
        return []
    
    chat_data = _read_chat_data(chat_file)
    return chat_data.get("messages", [])

def _update_message_in_chat(chat_file: str, index: int, message: Dict) -> bool:
    """
    Updates a specific message in the chat history.
    Specifically used for system message updates when activating/deactivating toolsets.
    
    Args:
        chat_file: The chat file ID
        index: The zero-based index of the message to update
        message: The new message object with role and content
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not chat_file:
        logger.error("Cannot update message: No chat file specified.")
        return False
        
    try:
        chat_data = _read_chat_data(chat_file)
        messages = chat_data.get("messages", [])
        
        if not messages or index >= len(messages):
            logger.error(f"Cannot update message at index {index}: Chat {chat_file} has only {len(messages)} messages.")
            return False
            
        # Update the message
        messages[index] = message
        chat_data["messages"] = messages
        _write_chat_data(chat_file, chat_data)
        logger.debug(f"Updated message at index {index} in chat {chat_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating message at index {index} in chat {chat_file}: {e}")
        logger.error(traceback.format_exc())
        return False