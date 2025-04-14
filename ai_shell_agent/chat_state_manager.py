"""
Manages persistent state related to chat sessions, including
session tracking, chat file I/O, Aider state, chat mapping, toolsets.
"""
import os
import json
import uuid
import traceback
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from datetime import datetime, timezone

# Local imports
from . import logger
# Import the prompt builder (needed for initial prompt)
from .prompts.prompts import system_prompt
# Import toolset registry functions to get available toolsets
# This import depends on toolsets.toolsets running its discovery first
from .toolsets.toolsets import get_toolset_ids

# --- Constants ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHATS_DIR = os.path.join(DATA_DIR, "chats")
SESSION_FILE = os.path.join(DATA_DIR, "session.json")
CHAT_MAP_FILE = os.path.join(CHATS_DIR, "chat_map.json")
AIDER_STATE_KEY = "_aider_state" # Internal key name
# Use simple keys for metadata storage
ACTIVE_TOOLSETS_KEY = "active_toolsets"
ENABLED_TOOLSETS_KEY = "enabled_toolsets"

# --- Default Toolsets ---
# Active toolsets start empty, must be activated by LLM or user
DEFAULT_ACTIVE_TOOLSETS = []

# Enabled toolsets default to all discovered toolsets
# This relies on discovery having run in toolsets/toolsets.py via the __init__ import chain
try:
    DEFAULT_ENABLED_TOOLSETS = get_toolset_ids()
    logger.info(f"Default enabled toolsets initialized: {DEFAULT_ENABLED_TOOLSETS}")
except Exception as e:
    logger.error(f"Failed to get toolset names for defaults at init: {e}. Falling back.")
    DEFAULT_ENABLED_TOOLSETS = ["terminal"] # Basic fallback

# Ensure directories exist
os.makedirs(CHATS_DIR, exist_ok=True)

# --- Low-Level JSON Helpers (Keep _read_json, _write_json as before) ---
def _read_json(file_path: str, default_value=None) -> Any:
    """Reads a JSON file or returns a default value if not found."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if isinstance(e, FileNotFoundError):
            logger.debug(f"File not found: {file_path}. Returning default.")
        else:
            logger.error(f"Error parsing JSON from {file_path}: {e}. Returning default.")
        return json.loads(json.dumps(default_value)) if default_value is not None else {}

def _write_json(file_path: str, data: Any) -> None:
    """Writes data to a JSON file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}.tmp.{uuid.uuid4()}"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}", exc_info=True)
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception as rm_e: logger.error(f"Failed to remove temporary file {tmp_path}: {rm_e}")

# --- Chat File Data Helpers ---
def _read_chat_data(chat_file: str) -> Dict:
    """Reads the data for a specific chat session file, applying defaults."""
    if not chat_file:
         logger.error("Attempted to read chat data with empty chat_file.")
         return {"messages": [], "metadata": {
             ACTIVE_TOOLSETS_KEY: list(DEFAULT_ACTIVE_TOOLSETS),
             ENABLED_TOOLSETS_KEY: list(DEFAULT_ENABLED_TOOLSETS)
         }}
    chat_path = os.path.join(CHATS_DIR, f"{chat_file}.json")
    # Provide current defaults when reading
    default_data = {"messages": [], "metadata": {
        ACTIVE_TOOLSETS_KEY: list(DEFAULT_ACTIVE_TOOLSETS),
        ENABLED_TOOLSETS_KEY: list(DEFAULT_ENABLED_TOOLSETS)
    }}
    data = _read_json(chat_path, default_data)
    # Ensure metadata keys exist even if file exists but lacks them
    if "metadata" not in data: data["metadata"] = {}
    if ACTIVE_TOOLSETS_KEY not in data["metadata"]: data["metadata"][ACTIVE_TOOLSETS_KEY] = list(DEFAULT_ACTIVE_TOOLSETS)
    if ENABLED_TOOLSETS_KEY not in data["metadata"]: data["metadata"][ENABLED_TOOLSETS_KEY] = list(DEFAULT_ENABLED_TOOLSETS)
    return data

def _write_chat_data(chat_file: str, data: Dict) -> None:
    """Writes data for a specific chat session file."""
    if not chat_file:
         logger.error("Attempted to write chat data with empty chat_file.")
         return
    # Ensure essential keys exist before writing (already handled by _read_chat_data defaults)
    if "metadata" not in data: data["metadata"] = {}
    if ACTIVE_TOOLSETS_KEY not in data["metadata"]: data["metadata"][ACTIVE_TOOLSETS_KEY] = list(DEFAULT_ACTIVE_TOOLSETS)
    if ENABLED_TOOLSETS_KEY not in data["metadata"]: data["metadata"][ENABLED_TOOLSETS_KEY] = list(DEFAULT_ENABLED_TOOLSETS)

    chat_path = os.path.join(CHATS_DIR, f"{chat_file}.json")
    _write_json(chat_path, data)

def _get_chat_metadata(chat_file: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Gets metadata, returning default if key not found."""
    if not chat_file: return default if key else {}
    chat_data = _read_chat_data(chat_file)
    metadata = chat_data.get("metadata", {})
    if key is None: return metadata
    return metadata.get(key, default)

def _update_metadata_in_chat(chat_file: str, key: str, value: Any) -> None:
    """Updates a metadata value."""
    if not chat_file:
         logger.error(f"Attempted to update metadata '{key}' with empty chat_file.")
         return
    chat_data = _read_chat_data(chat_file)
    if "metadata" not in chat_data: chat_data["metadata"] = {}
    chat_data["metadata"][key] = value
    _write_chat_data(chat_file, chat_data)

# --- Chat Map Helpers (Unchanged) ---
def _read_chat_map() -> Dict[str, str]: return _read_json(CHAT_MAP_FILE, {})
def _write_chat_map(chat_map: Dict[str, str]) -> None: _write_json(CHAT_MAP_FILE, chat_map)

# --- Session Management (Unchanged) ---
def get_current_chat() -> Optional[str]: return _read_json(SESSION_FILE, {}).get("current_chat")
def save_session(chat_file: Optional[str]) -> None: _write_json(SESSION_FILE, {"current_chat": chat_file} if chat_file else {})

# --- Helper Functions (Unchanged) ---
def _get_console_session_id() -> str: return "console_" + str(uuid.uuid4())

# --- Aider State Management (Unchanged) ---
def get_aider_state(chat_file: str) -> Optional[Dict]:
    if not chat_file: return None
    return _get_chat_metadata(chat_file, AIDER_STATE_KEY)

def save_aider_state(chat_file: str, state: Dict) -> None:
    if not chat_file: logger.error("Cannot save Aider state: No active chat session."); return
    _update_metadata_in_chat(chat_file, AIDER_STATE_KEY, state)

def clear_aider_state(chat_file: str) -> None:
    if not chat_file: logger.error("Cannot clear Aider state: No active chat session."); return
    current_state = get_aider_state(chat_file) or {}
    current_state['enabled'] = False
    current_state.pop('aider_done_messages', None)
    current_state.pop('abs_fnames', None)
    current_state.pop('abs_read_only_fnames', None)
    _update_metadata_in_chat(chat_file, AIDER_STATE_KEY, current_state)
    logger.info(f"Cleared Aider state for chat {chat_file}")

# --- Toolset Management ---
def get_active_toolsets(chat_file: str) -> List[str]:
    """Gets the list of *active* toolsets for a specific chat session."""
    return _get_chat_metadata(chat_file, ACTIVE_TOOLSETS_KEY, default=list(DEFAULT_ACTIVE_TOOLSETS))

def update_active_toolsets(chat_file: str, toolsets: List[str]) -> None:
    """Updates the list of *active* toolsets. Ensures they are valid and enabled."""
    if not chat_file: logger.error("update_active_toolsets called with empty chat_file."); return
    enabled = get_enabled_toolsets(chat_file)
    valid_active = [ts for ts in toolsets if ts in enabled]
    invalid_attempt = [ts for ts in toolsets if ts not in enabled]
    if invalid_attempt:
        logger.warning(f"Attempted to activate non-enabled toolsets for chat {chat_file}: {invalid_attempt}. Ignoring.")
    unique_toolsets = sorted(list(set(valid_active)))
    _update_metadata_in_chat(chat_file, ACTIVE_TOOLSETS_KEY, unique_toolsets)
    logger.debug(f"Active toolsets updated for chat {chat_file}: {unique_toolsets}")

def get_enabled_toolsets(chat_file: str) -> List[str]:
    """Gets the list of *enabled* toolsets for a specific chat session."""
    # Refresh default value in case discovery was delayed
    global DEFAULT_ENABLED_TOOLSETS
    try: DEFAULT_ENABLED_TOOLSETS = get_toolset_ids()
    except Exception: pass # Ignore errors here, default already set at top level
    return _get_chat_metadata(chat_file, ENABLED_TOOLSETS_KEY, default=list(DEFAULT_ENABLED_TOOLSETS))

def update_enabled_toolsets(chat_file: str, toolsets: List[str]) -> None:
    """Updates the list of *enabled* toolsets. Deactivates any no longer enabled."""
    if not chat_file: logger.error("update_enabled_toolsets called with empty chat_file."); return
    # Validate against registered toolsets
    registered_names = get_toolset_ids()
    valid_toolsets = [ts for ts in toolsets if ts in registered_names]
    invalid_toolsets = [ts for ts in toolsets if ts not in registered_names]
    if invalid_toolsets:
         logger.warning(f"Attempted to enable invalid toolsets for chat {chat_file}: {invalid_toolsets}. Ignoring.")
    unique_toolsets = sorted(list(set(valid_toolsets)))
    _update_metadata_in_chat(chat_file, ENABLED_TOOLSETS_KEY, unique_toolsets)
    logger.info(f"Enabled toolsets updated for chat {chat_file}: {unique_toolsets}")
    # Deactivate any currently active toolsets that are no longer enabled
    current_active = get_active_toolsets(chat_file)
    new_active = [ts for ts in current_active if ts in unique_toolsets]
    if len(new_active) != len(current_active):
         logger.info(f"Deactivating toolsets no longer enabled: {list(set(current_active) - set(new_active))}")
         update_active_toolsets(chat_file, new_active) # This calls _update_metadata_in_chat again

# --- Chat Creation/Management ---
def create_or_load_chat(title: str) -> Optional[str]:
    """Creates/Loads chat, ensuring metadata keys and valid toolsets."""
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}
    chat_file: Optional[str] = None
    needs_save = False

    # Refresh default enabled toolsets value
    global DEFAULT_ENABLED_TOOLSETS
    try: DEFAULT_ENABLED_TOOLSETS = get_toolset_ids()
    except Exception: pass

    if title in title_to_id:
        chat_file = title_to_id[title]
        logger.debug(f"Loading existing chat: {title} ({chat_file})")
        chat_data = _read_chat_data(chat_file) # Reads with current defaults
        metadata = chat_data.get("metadata", {})
        messages = chat_data.get("messages", [])

        current_enabled = metadata.get(ENABLED_TOOLSETS_KEY, list(DEFAULT_ENABLED_TOOLSETS))
        registered_names = get_toolset_ids()
        valid_enabled = [ts for ts in current_enabled if ts in registered_names]
        if set(valid_enabled) != set(current_enabled):
            logger.warning(f"Correcting enabled toolsets for chat '{title}': {valid_enabled}")
            metadata[ENABLED_TOOLSETS_KEY] = valid_enabled; needs_save = True
            current_enabled = valid_enabled

        current_active = metadata.get(ACTIVE_TOOLSETS_KEY, list(DEFAULT_ACTIVE_TOOLSETS))
        valid_active = [ts for ts in current_active if ts in current_enabled]
        if set(valid_active) != set(current_active):
            logger.warning(f"Correcting active toolsets for chat '{title}': {valid_active}")
            metadata[ACTIVE_TOOLSETS_KEY] = valid_active; needs_save = True
            current_active = valid_active

        # Check/Rebuild system prompt ONLY IF necessary
        prompt_needs_update = False
        expected_prompt = system_prompt(enabled_toolsets=current_enabled, active_toolsets=current_active)
        if not messages or messages[0].get("role") != "system" or messages[0].get("content") != expected_prompt:
            prompt_needs_update = True

        if prompt_needs_update:
            new_sys_msg = {"role": "system", "content": expected_prompt, "timestamp": datetime.now(timezone.utc).isoformat()}
            if not messages or messages[0].get("role") != "system":
                 logger.warning(f"Prepending missing/invalid system prompt for chat '{title}'.")
                 messages.insert(0, new_sys_msg)
            else:
                 logger.debug(f"Updating system prompt content for chat '{title}'.")
                 messages[0] = new_sys_msg
            needs_save = True

        # Ensure other metadata
        if "title" not in metadata: metadata["title"] = title; needs_save = True
        if "created_at" not in metadata: metadata["created_at"] = datetime.now(timezone.utc).isoformat(); needs_save = True

        if needs_save:
             chat_data["metadata"] = metadata
             chat_data["messages"] = messages
             _write_chat_data(chat_file, chat_data)

    else:
        # Create a new chat
        chat_file = str(uuid.uuid4())
        chat_map[chat_file] = title
        _write_chat_map(chat_map)
        logger.debug(f"Creating new chat: {title} ({chat_file})")

        initial_enabled = list(DEFAULT_ENABLED_TOOLSETS)
        initial_active = list(DEFAULT_ACTIVE_TOOLSETS)
        initial_system_prompt = system_prompt(enabled_toolsets=initial_enabled, active_toolsets=initial_active)
        chat_data = {
            "messages": [{"role": "system", "content": initial_system_prompt, "timestamp": datetime.now(timezone.utc).isoformat()}],
            "metadata": {
                ENABLED_TOOLSETS_KEY: initial_enabled,
                ACTIVE_TOOLSETS_KEY: initial_active,
                AIDER_STATE_KEY: {"enabled": False},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "title": title
            }
        }
        _write_chat_data(chat_file, chat_data)
        logger.debug(f"Created new chat '{title}' with enabled={initial_enabled}, active={initial_active}.")

    if chat_file: save_session(chat_file); return chat_file
    else: logger.error(f"Failed to create or load chat '{title}'."); return None

# --- Rename/Delete/List Chats (Largely unchanged) ---
def rename_chat(old_title: str, new_title: str) -> bool:
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}
    if old_title not in title_to_id: logger.error(f"Chat '{old_title}' not found."); print(f"Error: Chat '{old_title}' not found."); return False
    chat_file = title_to_id[old_title]
    if new_title in title_to_id and title_to_id[new_title] != chat_file: logger.error(f"Chat '{new_title}' already exists."); print(f"Error: Chat '{new_title}' already exists."); return False
    chat_map[chat_file] = new_title; _write_chat_map(chat_map)
    _update_metadata_in_chat(chat_file, "title", new_title)
    logger.info(f"Renamed chat: {old_title} -> {new_title}"); return True

def delete_chat(title: str) -> bool:
    chat_map = _read_chat_map(); title_to_id = {v: k for k, v in chat_map.items()}
    if title not in title_to_id: logger.error(f"Chat '{title}' not found."); print(f"Error: Chat '{title}' not found."); return False
    chat_file = title_to_id[title]; del chat_map[chat_file]; _write_chat_map(chat_map)
    chat_path = os.path.join(CHATS_DIR, f"{chat_file}.json")
    try:
        if os.path.exists(chat_path): os.remove(chat_path)
        logger.info(f"Deleted chat: {title}");
        if get_current_chat() == chat_file: save_session(None); logger.info("Cleared current session.")
        return True
    except Exception as e: logger.warning(f"Could not delete chat file {chat_path}: {e}"); print(f"Warning: Chat '{title}' removed, but could not delete file: {e}"); return False

def get_chat_titles() -> Dict[str, str]: return _read_chat_map()

def get_current_chat_title() -> Optional[str]:
    chat_file = get_current_chat();
    if not chat_file: return None
    title = _get_chat_metadata(chat_file, "title")
    return title if title else _read_chat_map().get(chat_file) # Fallback

def flush_temp_chats() -> int:
    chat_map = _read_chat_map(); removed_count = 0
    temp_chats_to_remove = {cid: t for cid, t in chat_map.items() if t.startswith("Temp Chat ")}
    if not temp_chats_to_remove: return 0
    current_session = get_current_chat(); clear_current = False
    for chat_id, title in temp_chats_to_remove.items():
        if chat_id in chat_map: del chat_map[chat_id]
        else: logger.warning(f"Temp chat ID {chat_id} ('{title}') not in map during flush."); continue
        chat_path = os.path.join(CHATS_DIR, f"{chat_id}.json")
        try:
            if os.path.exists(chat_path): os.remove(chat_path); removed_count += 1; logger.debug(f"Removed temp chat: {title}")
            else: logger.warning(f"Temp chat file {chat_path} not found during flush.")
            if current_session == chat_id: clear_current = True
        except OSError as e: logger.warning(f"Could not delete temp chat file {chat_path}: {e}")
    if temp_chats_to_remove: _write_chat_map(chat_map)
    if clear_current: save_session(None)
    logger.debug(f"Flushed {removed_count} temporary chats."); return removed_count

# --- Message Access/Update Functions ---
def _get_chat_messages(chat_file: str) -> List[Dict]:
    if not chat_file: logger.error("Cannot get messages: No chat file specified."); return []
    return _read_chat_data(chat_file).get("messages", [])

def _update_message_in_chat(chat_file: str, index: int, message: Dict) -> bool:
    """Updates a specific message in the chat history (used by --edit)."""
    if not chat_file: logger.error("Cannot update message: No chat file specified."); return False
    try:
        chat_data = _read_chat_data(chat_file); messages = chat_data.get("messages", [])
        if not (0 <= index < len(messages)): logger.error(f"Index {index} out of bounds for chat {chat_file}."); return False
        if "timestamp" not in message: message["timestamp"] = datetime.now(timezone.utc).isoformat()
        messages[index] = message; chat_data["messages"] = messages; _write_chat_data(chat_file, chat_data)
        logger.debug(f"Updated message at index {index} in chat {chat_file}"); return True
    except Exception as e: logger.error(f"Error updating message {index} in chat {chat_file}: {e}", exc_info=True); return False