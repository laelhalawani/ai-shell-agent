"""
Manages persistent state related to chat sessions, including
session tracking, chat file I/O, chat mapping, toolsets.
"""
import os
import json
import uuid
import traceback
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from datetime import datetime, timezone
import shutil # Import shutil for directory removal

# Local imports
from . import logger, DATA_DIR, CHATS_DIR, TOOLSETS_GLOBAL_CONFIG_DIR
from .paths import ROOT_DIR
# Import JSON utility functions from utils
from .utils.file_io import read_json, write_json
# Import the static system prompt (now using static prompt)
from .prompts.prompts import SYSTEM_PROMPT
# Import toolset registry functions to get available toolsets
from .toolsets.toolsets import get_toolset_ids, get_toolset_names  # Added get_toolset_names
# Import default enabled toolsets from settings
from .settings import DEFAULT_ENABLED_TOOLSETS_NAMES
# Import console manager and texts
from .console_manager import get_console_manager # <--- ADDED IMPORT
from .texts import get_text # <--- ADDED IMPORT

# Get console manager instance
console = get_console_manager() # <--- ADDED INSTANCE

# --- Constants ---
SESSION_FILE = Path(DATA_DIR) / "session.json"
CHAT_MAP_FILE = Path(CHATS_DIR) / "chat_map.json"
# Metadata keys within chat config.json
MODEL_KEY = "agent_model" # Optional override for the agent model per chat
ENABLED_TOOLSETS_KEY = "enabled_toolsets"
TITLE_KEY = "title"
CREATED_AT_KEY = "created_at"

# Ensure directories exist
os.makedirs(CHATS_DIR, exist_ok=True)

# --- Low-Level JSON Helpers ---
# These functions have been moved to utils.py
# Using aliases for backward compatibility
def _read_json(file_path: Path, default_value=None) -> Any:
    """Alias for read_json from utils."""
    return read_json(file_path, default_value)

def _write_json(file_path: Path, data: Any) -> None:
    """Alias for write_json from utils."""
    write_json(file_path, data)

# --- Path Helper Functions ---
def _get_chat_dir_path(chat_id: str) -> Path:
    """Gets the Path object for a chat's directory."""
    if not chat_id: raise ValueError("chat_id cannot be empty")
    return Path(CHATS_DIR) / chat_id

def _get_chat_messages_path(chat_id: str) -> Path:
    """Gets the Path object for a chat's messages file."""
    return _get_chat_dir_path(chat_id) / "chat.json"

def _get_chat_config_path(chat_id: str) -> Path:
    """Gets the Path object for a chat's config file."""
    return _get_chat_dir_path(chat_id) / "config.json"

def get_toolset_data_path(chat_id: str, toolset_id: str) -> Path:
    """Gets the Path object for a toolset's data file within a chat."""
    if not toolset_id: raise ValueError("toolset_id cannot be empty")
    toolsets_dir = _get_chat_dir_path(chat_id) / "toolsets"
    toolsets_dir.mkdir(exist_ok=True) # Ensure it exists when path is requested
    return toolsets_dir / f"{toolset_id}.json"

# --- NEW: Function to get global toolset config path ---
def get_toolset_global_config_path(toolset_id: str) -> Path:
    """Gets the Path object for a toolset's GLOBAL default config file."""
    if not toolset_id: raise ValueError("toolset_id cannot be empty")
    # TOOLSETS_GLOBAL_CONFIG_DIR is imported from __init__
    # Ensure the directory exists (should be handled by __init__)
    TOOLSETS_GLOBAL_CONFIG_DIR.mkdir(exist_ok=True)
    return TOOLSETS_GLOBAL_CONFIG_DIR / f"{toolset_id}.json"

# --- Chat Data Access ---
def _read_chat_messages(chat_id: str) -> List[Dict]:
    """Reads the messages for a specific chat session file."""
    if not chat_id:
        logger.error("Attempted to read chat messages with empty chat_id.")
        return []
    messages_path = _get_chat_messages_path(chat_id)
    # Default to empty list if file not found or invalid
    data = _read_json(messages_path, {"messages": []})
    return data.get("messages", [])

def _write_chat_messages(chat_id: str, messages: List[Dict]) -> None:
    """Writes messages for a specific chat session file."""
    if not chat_id:
        logger.error("Attempted to write chat messages with empty chat_id.")
        return
    messages_path = _get_chat_messages_path(chat_id)
    _write_json(messages_path, {"messages": messages})

def _read_chat_config(chat_id: str) -> Dict:
    """Reads the config for a specific chat session, applying defaults."""
    if not chat_id:
        logger.error("Attempted to read chat config with empty chat_id.")
        return {} # Return empty dict, let caller handle defaults
    config_path = _get_chat_config_path(chat_id)
    # Provide DEFAULTS structure when reading
    defaults = {
        MODEL_KEY: None, # Default to global model
        ENABLED_TOOLSETS_KEY: [], # Default to EMPTY list now
        TITLE_KEY: "Untitled Chat",
        CREATED_AT_KEY: datetime.now(timezone.utc).isoformat()
    }
    config = _read_json(config_path, default_value=defaults)
    # Ensure all default keys exist
    needs_update = False
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
            needs_update = True # If a key was missing, write back the complete default structure
    
    # Validate toolset lists against registry (can stay, good practice)
    from .toolsets.toolsets import get_toolset_names # Import locally here is ok
    registered_names = get_toolset_names()
    current_enabled = config.get(ENABLED_TOOLSETS_KEY, [])
    valid_enabled = [name for name in current_enabled if name in registered_names]
    if set(valid_enabled) != set(current_enabled):
         logger.warning(f"Correcting enabled toolsets for chat '{chat_id}': {valid_enabled}")
         config[ENABLED_TOOLSETS_KEY] = valid_enabled
         needs_update = True

    if needs_update:
         logger.debug(f"Updating chat config {config_path} with defaults/corrections.")
         _write_chat_config(chat_id, config)

    return config

def _write_chat_config(chat_id: str, config: Dict) -> None:
    """Writes config for a specific chat session file."""
    if not chat_id:
        logger.error("Attempted to write chat config with empty chat_id.")
        return
    config_path = _get_chat_config_path(chat_id)
    _write_json(config_path, config)

def get_chat_config_value(chat_id: str, key: str, default: Any = None) -> Any:
    """Reads a specific configuration value for a chat."""
    config = _read_chat_config(chat_id) # Reads with defaults/validation
    return config.get(key, default)

def update_chat_config_value(chat_id: str, key: str, value: Any) -> None:
    """Updates a specific configuration value for a chat."""
    if not chat_id:
        logger.error(f"Attempted to update config key '{key}' with empty chat_id.")
        return
    # Read existing config to update it safely
    config = _read_chat_config(chat_id)
    config[key] = value
    _write_chat_config(chat_id, config)

# --- Chat Map Helpers ---
def _read_chat_map() -> Dict[str, str]: 
    return _read_json(CHAT_MAP_FILE, {})

def _write_chat_map(chat_map: Dict[str, str]) -> None: 
    _write_json(CHAT_MAP_FILE, chat_map)

# --- Chat Map Access ---
def get_chat_map() -> Dict[str, str]:
    """Gets the map of chat IDs to chat titles."""
    return _read_chat_map()

# --- Session Management ---
def get_current_chat() -> Optional[str]: 
    return _read_json(SESSION_FILE, {}).get("current_chat")

def save_session(chat_id: Optional[str]) -> None: 
    _write_json(SESSION_FILE, {"current_chat": chat_id} if chat_id else {})

# --- Helper Functions ---
def _get_console_session_id() -> str: 
    return "console_" + str(uuid.uuid4())

# --- Toolset Management ---
def get_enabled_toolsets(chat_id: str) -> List[str]:
    """Gets the list of *enabled* toolset names for a specific chat session."""
    # No need to refresh global DEFAULT anymore
    return get_chat_config_value(chat_id, ENABLED_TOOLSETS_KEY, default=[]) # Return empty list default

def update_enabled_toolsets(chat_id: str, toolset_names: List[str]) -> None:
    """Updates the list of *enabled* toolset names. Checks config for newly enabled."""
    if not chat_id: 
        logger.error("update_enabled_toolsets called with empty chat_id.")
        return

    # --- Get enabled toolsets BEFORE the update ---
    config_before_update = _read_chat_config(chat_id) # Read state before update
    enabled_before_update = set(config_before_update.get(ENABLED_TOOLSETS_KEY, []))
    # --- End get before update ---

    registered_names = get_toolset_names() # Use display names
    valid_toolsets = [name for name in toolset_names if name in registered_names]
    invalid_toolsets = [name for name in toolset_names if name not in registered_names]
    if invalid_toolsets:
        logger.warning(f"Attempted to enable invalid toolsets for chat {chat_id}: {invalid_toolsets}. Ignoring.")

    unique_toolsets = sorted(list(set(valid_toolsets)))
    update_chat_config_value(chat_id, ENABLED_TOOLSETS_KEY, unique_toolsets)
    logger.info(f"Enabled toolsets updated for chat {chat_id}: {unique_toolsets}")

    # --- Correctly calculate and check newly enabled toolsets ---
    enabled_after_update = set(unique_toolsets)
    newly_enabled_correct = enabled_after_update - enabled_before_update # Compare new enabled with old enabled

    if newly_enabled_correct:
        logger.info(f"Checking configuration for newly enabled toolsets: {newly_enabled_correct}")
        for toolset_name in newly_enabled_correct: # Iterate over the correctly identified set
            check_and_configure_toolset(chat_id, toolset_name)
    # --- End correction ---

# --- MODIFIED: check_and_configure_toolset ---
def check_and_configure_toolset(chat_id: str, toolset_name: str):
    """
    Checks if a toolset is configured for a chat. If not, attempts to copy
    global defaults. If no global defaults, runs the configuration wizard
    which saves the result to both chat-specific and global locations.
    """
    logger.debug(f"Checking configuration for toolset '{toolset_name}' in chat {chat_id}")

    from .toolsets.toolsets import get_registered_toolsets # Import locally

    registered_toolsets = get_registered_toolsets()
    target_metadata = None
    target_id = None
    for ts_id, meta in registered_toolsets.items():
        if meta.name == toolset_name:
             target_metadata = meta
             target_id = ts_id
             break

    if not target_metadata or not target_id:
        logger.error(f"Cannot check configuration: Toolset '{toolset_name}' not found in registry.")
        return

    # Get paths
    local_config_path = get_toolset_data_path(chat_id, target_id)
    global_config_path = get_toolset_global_config_path(target_id)
    # --- Get .env path ---
    dotenv_path = ROOT_DIR / '.env'

    # Read current chat-specific config
    # Use None as default to distinguish between empty dict and file not found
    local_config = _read_json(local_config_path, default_value=None)

    # --- MODIFIED CHECK ---
    # Check if the local config file EXISTS *and* contains a dictionary
    is_local_configured = False
    if local_config_path.exists() and isinstance(local_config, dict):
        is_local_configured = True
        logger.debug(f"Found existing local config file for '{toolset_name}' at {local_config_path}.")
    # --- END MODIFIED CHECK ---

    if is_local_configured:
        logger.debug(f"Toolset '{toolset_name}' already configured locally for chat {chat_id}.")
        # --- Check for required secrets even if configured ---
        if target_metadata.required_secrets:
             logger.debug(f"Checking required secrets for already configured toolset '{toolset_name}'...")
             from .utils.env import ensure_dotenv_key # Import locally
             all_secrets_ok = True
             for key, desc in target_metadata.required_secrets.items():
                  if ensure_dotenv_key(dotenv_path, key, desc) is None:
                       all_secrets_ok = False
             if not all_secrets_ok:
                  logger.warning(f"One or more required secrets missing for '{toolset_name}'. It might malfunction.")
        return # Already configured locally

    # --- Local is NOT configured (File doesn't exist or isn't a dict) ---
    logger.debug(f"Local config not found or invalid for '{toolset_name}' at {local_config_path}. Checking global defaults at {global_config_path}.")
    global_config = _read_json(global_config_path, default_value=None)

    # --- MODIFIED CHECK ---
    # Check if global config exists and is valid
    is_global_configured = False
    if global_config_path.exists() and isinstance(global_config, dict):
         is_global_configured = True
    # --- END MODIFIED CHECK ---

    if is_global_configured:
        # Global default exists, copy it to the local path
        logger.info(f"Found valid global default config for '{toolset_name}'. Copying to {local_config_path}.")
        try:
            _write_json(local_config_path, global_config)
            logger.debug(f"Successfully copied global config to local config for '{toolset_name}'.")
            # --- Check required secrets AFTER copying global default ---
            if target_metadata.required_secrets:
                 logger.debug(f"Checking required secrets for '{toolset_name}' after applying global default...")
                 from .utils.env import ensure_dotenv_key
                 all_secrets_ok = True
                 for key, desc in target_metadata.required_secrets.items():
                      if ensure_dotenv_key(dotenv_path, key, desc) is None:
                           all_secrets_ok = False
                 if not all_secrets_ok:
                      logger.warning(f"One or more required secrets missing for '{toolset_name}'. It might malfunction.")
            return # Now configured locally using global defaults
        except Exception as e:
            logger.error(f"Failed to copy global config {global_config_path} to {local_config_path}: {e}. Proceeding to manual config.")
            # Fall through to manual configuration if copy fails

    # --- Neither local nor global config is valid/exists, run configure_func ---
    if not target_metadata.configure_func:
         logger.debug(f"Toolset '{toolset_name}' has no configuration function and no valid defaults found. Cannot configure automatically.")
         # Create an empty config file locally to mark it as "checked"
         try:
             _write_json(local_config_path, {})
             logger.debug(f"Wrote empty local config for toolset '{toolset_name}' at {local_config_path} as no configure_func exists.")
         except Exception as e:
              logger.error(f"Failed to write empty local config for '{toolset_name}' at {local_config_path}: {e}")
         return # Nothing more to do

    # --- Run the configuration function ---
    # It handles prompting, secret checks (using ensure_dotenv_key), and saving to both paths.
    console.display_message(get_text("common.labels.info"), get_text("state.tool_config.notice_needs_config", toolset_name=toolset_name), 
                           console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT) # MODIFIED
    logger.info(f"Running configuration function for '{toolset_name}' (ID: {target_id}) for chat {chat_id}.")
    try:
        # Call configure_func, passing paths and the potentially invalid local_config (or None)
        # Signature: configure_func(global_path, local_path, dotenv_path, current_local_config) -> Dict
        final_config = target_metadata.configure_func(
            global_config_path,
            local_config_path,
            dotenv_path,
            local_config if isinstance(local_config, dict) else None # Pass None if not a dict
        )
        logger.info(f"Configuration function completed for {toolset_name}.")
        # Configuration function handles user confirmation messages.

    except (EOFError, KeyboardInterrupt):
         logger.warning(f"Configuration cancelled by user for {toolset_name}. Toolset may not work correctly.")
         console.display_message(get_text("common.labels.warning"), get_text("state.tool_config.warn_cancel", toolset_name=toolset_name),
                                console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT) # MODIFIED
    except Exception as e:
         logger.error(f"Error running configuration for {toolset_name}: {e}", exc_info=True)
         console.display_message(get_text("common.labels.error"), get_text("state.tool_config.error_failed", toolset_name=toolset_name, error=e),
                                console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT) # MODIFIED

# --- Chat Creation/Management ---
def create_or_load_chat(title: str) -> Optional[str]:
    """Creates/Loads chat directory, ensuring structure and base config/messages."""
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}
    chat_id: Optional[str] = None

    if title in title_to_id:
        chat_id = title_to_id[title]
        chat_dir = _get_chat_dir_path(chat_id)
        logger.debug(f"Loading existing chat: {title} ({chat_id})")

        if not chat_dir.is_dir():
            logger.error(f"Chat directory not found for existing chat {title} ({chat_id}). Removing from map.")
            del chat_map[chat_id]
            _write_chat_map(chat_map)
            # Force creation below
            chat_id = None # Reset chat_id to trigger creation logic
        else:
            # Directory exists, ensure files and validate config
            config = _read_chat_config(chat_id) # Reads/validates/updates config.json
            messages = _read_chat_messages(chat_id) # Reads chat.json

            # Check/Rebuild system prompt ONLY IF necessary
            if not messages or messages[0].get("role") != "system" or messages[0].get("content") != SYSTEM_PROMPT:
                new_sys_msg = {"role": "system", "content": SYSTEM_PROMPT, "timestamp": datetime.now(timezone.utc).isoformat()}
                if not messages or messages[0].get("role") != "system":
                    logger.warning(f"Prepending missing/invalid system prompt for chat '{title}'.")
                    messages.insert(0, new_sys_msg)
                else:
                    logger.debug(f"Updating system prompt content for chat '{title}'.")
                    messages[0] = new_sys_msg
                _write_chat_messages(chat_id, messages) # Write updated messages

            # Check toolset configuration status for all enabled toolsets
            enabled_names = config.get(ENABLED_TOOLSETS_KEY, [])
            logger.debug(f"Chat {chat_id} loaded. Checking config for enabled toolsets: {enabled_names}")
            for toolset_name in enabled_names:
                check_and_configure_toolset(chat_id, toolset_name)

    # If chat_id is still None, it means we need to create it
    if chat_id is None:
        chat_id = str(uuid.uuid4())
        chat_map[chat_id] = title
        _write_chat_map(chat_map)
        logger.debug(f"Creating new chat: {title} ({chat_id})")

        # Create directory structure
        chat_dir = _get_chat_dir_path(chat_id)
        toolsets_subdir = chat_dir / "toolsets"
        try:
            chat_dir.mkdir(parents=True, exist_ok=True)
            toolsets_subdir.mkdir(exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory structure for chat {chat_id}: {e}")
            # Clean up map entry
            del chat_map[chat_id]
            _write_chat_map(chat_map)
            return None

        # Read global defaults for new chat
        from .config_manager import get_default_enabled_toolsets
        initial_enabled = get_default_enabled_toolsets()
        logger.info(f"Applying global default enabled toolsets to new chat: {initial_enabled}")
        
        initial_config = {
            MODEL_KEY: None, # Use global default initially
            ENABLED_TOOLSETS_KEY: initial_enabled, # Use global defaults here
            TITLE_KEY: title,
            CREATED_AT_KEY: datetime.now(timezone.utc).isoformat()
        }
        _write_chat_config(chat_id, initial_config)

        # Write initial chat.json with static system prompt
        initial_messages = [{"role": "system", "content": SYSTEM_PROMPT, "timestamp": datetime.now(timezone.utc).isoformat()}]
        _write_chat_messages(chat_id, initial_messages)

        logger.debug(f"Created new chat '{title}' with enabled={initial_enabled}.")

        # For new chats, check configuration for any default-enabled toolsets
        if initial_enabled:  # This would be empty by default, but just in case
            logger.debug(f"Chat {chat_id} created. Checking config for default enabled toolsets: {initial_enabled}")
            for toolset_name in initial_enabled:
                check_and_configure_toolset(chat_id, toolset_name)

    if chat_id:
        save_session(chat_id)
        return chat_id
    else:
        logger.error(f"Failed to create or load chat '{title}'.")
        return None

def rename_chat(old_title: str, new_title: str) -> bool:
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}
    if old_title not in title_to_id: 
        logger.error(f"Chat '{old_title}' not found.")
        console.display_message(get_text("common.labels.error"), get_text("state.rename.error_old_not_found", old_title=old_title),
                               console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT) # MODIFIED
        return False
    chat_id = title_to_id[old_title]
    if new_title in title_to_id and title_to_id[new_title] != chat_id: 
        logger.error(f"Chat '{new_title}' already exists.")
        console.display_message(get_text("common.labels.error"), get_text("state.rename.error_new_exists", new_title=new_title),
                               console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT) # MODIFIED
        return False
    
    chat_map[chat_id] = new_title
    _write_chat_map(chat_map)
    # Update title within the chat's config file
    update_chat_config_value(chat_id, TITLE_KEY, new_title)
    logger.info(f"Renamed chat: {old_title} -> {new_title}")
    return True

def delete_chat(title: str) -> bool:
    chat_map = _read_chat_map()
    title_to_id = {v: k for k, v in chat_map.items()}
    if title not in title_to_id: 
        logger.error(f"Chat '{title}' not found.")
        console.display_message(get_text("common.labels.error"), get_text("state.delete.error_not_found", title=title),
                               console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT) # MODIFIED
        return False
    
    chat_id = title_to_id[title]
    chat_dir_path = _get_chat_dir_path(chat_id)

    # Remove from map first
    del chat_map[chat_id]
    _write_chat_map(chat_map)

    try:
        if chat_dir_path.exists() and chat_dir_path.is_dir():
            shutil.rmtree(chat_dir_path) # Remove the whole directory
            logger.info(f"Deleted chat directory: {chat_dir_path}")
        else:
            logger.warning(f"Chat directory not found during delete: {chat_dir_path}")

        if get_current_chat() == chat_id: 
            save_session(None)
            logger.info("Cleared current session.")
        return True
    except Exception as e: 
        logger.error(f"Could not delete chat directory {chat_dir_path}: {e}", exc_info=True)
        console.display_message(get_text("common.labels.error"), get_text("state.delete.error_failed", title=title, error=e),
                               console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT) # MODIFIED
        return False

def get_chat_titles() -> Dict[str, str]: 
    return _read_chat_map()

def get_current_chat_title() -> Optional[str]:
    chat_id = get_current_chat()
    if not chat_id: 
        return None
    # Read title directly from config.json
    title = get_chat_config_value(chat_id, TITLE_KEY)
    # Fallback to map only if config read fails (which shouldn't happen with _read_chat_config)
    return title if title else _read_chat_map().get(chat_id)

def flush_temp_chats() -> int:
    chat_map = _read_chat_map()
    removed_count = 0
    temp_chats_to_remove = {cid: t for cid, t in chat_map.items() if t.startswith("Temp Chat ")}
    if not temp_chats_to_remove: 
        return 0
    
    current_session = get_current_chat()
    clear_current = False
    ids_removed_from_map = []
    
    for chat_id, title in temp_chats_to_remove.items():
        chat_dir_path = _get_chat_dir_path(chat_id)
        try:
            if chat_dir_path.exists() and chat_dir_path.is_dir():
                shutil.rmtree(chat_dir_path)
                removed_count += 1
                logger.debug(f"Removed temp chat dir: {title}")
            else:
                logger.warning(f"Temp chat dir {chat_dir_path} not found during flush.")

            # Only remove from map if deletion was attempted (successful or not)
            ids_removed_from_map.append(chat_id)
            if current_session == chat_id: 
                clear_current = True
        except OSError as e:
            logger.warning(f"Could not delete temp chat dir {chat_dir_path}: {e}")
            # Keep it in the map if deletion failed

    if ids_removed_from_map:
        for chat_id in ids_removed_from_map:
            if chat_id in chat_map: 
                del chat_map[chat_id]
        _write_chat_map(chat_map)
        
    if clear_current: 
        save_session(None)
        
    logger.info(f"Flushed {removed_count} temporary chats.")
    return removed_count

# --- Message Access/Update Functions ---
def get_chat_messages(chat_id: str) -> List[Dict]:
    """Gets all messages for a chat."""
    if not chat_id: 
        logger.error("Cannot get messages: No chat_id specified.")
        return []
    return _read_chat_messages(chat_id)

def _update_message_in_chat(chat_id: str, index: int, message: Dict) -> bool:
    """Updates a specific message in the chat history."""
    if not chat_id: 
        logger.error("Cannot update message: No chat_id specified.")
        return False
        
    try:
        messages = _read_chat_messages(chat_id)
        if not (0 <= index < len(messages)): 
            logger.error(f"Index {index} out of bounds for chat {chat_id}.")
            return False
            
        if "timestamp" not in message: 
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            
        messages[index] = message
        _write_chat_messages(chat_id, messages) # Pass full list
        logger.debug(f"Updated message at index {index} in chat {chat_id}")
        return True
    except Exception as e: 
        logger.error(f"Error updating message {index} in chat {chat_id}: {e}", exc_info=True)
        return False