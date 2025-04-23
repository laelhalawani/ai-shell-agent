# ai_shell_agent/toolsets/files/state.py
"""
Manages the persistent state (history) for the File Manager toolset.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Use relative imports to access core components needed here
from ... import logger
from ...utils.file_io import read_json, write_json
from ...chat_state_manager import get_toolset_data_path
from .settings import FILES_HISTORY_LIMIT # Import the setting

# --- State Management Helpers ---

TOOLSET_ID = "files" # Define toolset ID consistently

def _get_history_path(chat_id: str) -> Path:
    """Gets the path to the toolset's state file for a chat."""
    return get_toolset_data_path(chat_id, TOOLSET_ID)

def _read_toolset_state(chat_id: str) -> Dict:
    """Reads the toolset state, ensuring 'history' key exists."""
    state_path = _get_history_path(chat_id)
    default = {"history": []}
    # Ensure read_json returns a mutable type if default is used
    state = read_json(state_path, default_value=None)
    if state is None:
        state = default.copy() # Use a copy of the default
    if "history" not in state or not isinstance(state.get("history"), list):
        state["history"] = [] # Initialize or fix history type
    return state

def _write_toolset_state(chat_id: str, state_data: Dict) -> bool:
    """Writes the toolset state."""
    state_path = _get_history_path(chat_id)
    state_data.setdefault("history", []) # Ensure history key exists before writing
    return write_json(state_path, state_data)

def _log_history_event(chat_id: str, event_data: Dict) -> None:
    """Logs an event to the toolset's history, respecting the limit."""
    if not chat_id: return
    try:
        state = _read_toolset_state(chat_id)
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Ensure history is a list before appending
        if not isinstance(state.get("history"), list):
            state["history"] = []

        state["history"].append(event_data)

        limit = FILES_HISTORY_LIMIT # Use loaded setting

        # Apply limit if history exists and limit is positive
        if limit >= 0 and len(state["history"]) > limit:
             state["history"] = state["history"][-limit:]

        if not _write_toolset_state(chat_id, state):
            logger.error(f"Failed to write state after logging history event for chat {chat_id}")
        else:
            logger.debug(f"Logged file history event for chat {chat_id}: {event_data.get('operation')}")
    except Exception as e:
        logger.error(f"Failed to log file history event for chat {chat_id}: {e}", exc_info=True)