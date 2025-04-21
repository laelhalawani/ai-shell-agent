# ai_shell_agent/toolsets/aider/settings.py
"""
Loads Aider toolset default settings. Assumes the JSON file exists and is valid.
"""
from pathlib import Path
from typing import Optional, Any
from ...utils.file_io import read_json
from ... import logger

_toolset_id = 'aider'
_settings_file = Path(__file__).parent / 'default_settings' / 'default_settings.json'

def _load_data(file_path):
    """Basic JSON loader."""
    data = read_json(file_path, default_value={})
    if not data:
        logger.error(f"Error loading settings for toolset '{_toolset_id}' from {file_path}. Defaults missing.")
    return data

_settings_data = _load_data(_settings_file)

# --- Define Constants by Direct Access ---
try:
    AIDER_DEFAULT_MAIN_MODEL: Optional[str] = _settings_data['main_model']
    AIDER_DEFAULT_EDITOR_MODEL: Optional[str] = _settings_data['editor_model']
    AIDER_DEFAULT_WEAK_MODEL: Optional[str] = _settings_data['weak_model']
    AIDER_DEFAULT_EDIT_FORMAT: Optional[str] = _settings_data['edit_format']
    AIDER_DEFAULT_AUTO_COMMITS: bool = _settings_data['auto_commits']
    AIDER_DEFAULT_DIRTY_COMMITS: bool = _settings_data['dirty_commits']
except KeyError as e:
    logger.critical(f"Missing expected key in aider default_settings.json: {e}. Toolset cannot function correctly.")
    raise

if __name__ == '__main__':
    print(f"--- Aider Settings Loader Test (Simplified) ---")
    print(f"Loaded Aider Settings: {_settings_data}")
    print(f"AIDER_DEFAULT_MAIN_MODEL: {AIDER_DEFAULT_MAIN_MODEL}")
    print(f"AIDER_DEFAULT_EDITOR_MODEL: {AIDER_DEFAULT_EDITOR_MODEL}")
    print(f"AIDER_DEFAULT_WEAK_MODEL: {AIDER_DEFAULT_WEAK_MODEL}")
    print(f"AIDER_DEFAULT_EDIT_FORMAT: {AIDER_DEFAULT_EDIT_FORMAT}")
    print(f"AIDER_DEFAULT_AUTO_COMMITS: {AIDER_DEFAULT_AUTO_COMMITS}")
    print(f"AIDER_DEFAULT_DIRTY_COMMITS: {AIDER_DEFAULT_DIRTY_COMMITS}")