# ai_shell_agent/toolsets/cognitive/settings.py
"""
Loads Cognitive toolset default settings (currently none). Assumes the JSON file exists.
"""
from pathlib import Path
from typing import Any
from ...utils.file_io import read_json
from ... import logger

_toolset_id = 'cognitive' # <--- Use the correct toolset ID
_settings_file = Path(__file__).parent / 'default_settings' / 'default_settings.json'

def _load_data(file_path):
    """Basic JSON loader."""
    data = read_json(file_path, default_value={})
    # Don't log error if it's just empty, as expected for this toolset
    if data is None: # Log only if read_json failed entirely
         logger.error(f"Error loading settings for toolset '{_toolset_id}' from {file_path}. Defaults missing.")
         return {}
    return data

_settings_data = _load_data(_settings_file)

# --- Define Constants (Currently None) ---
# Example (if needed later): COGNITIVE_DEFAULT_SETTING = _settings_data['some_setting']
# No constants needed currently, but keep structure

if __name__ == '__main__':
    print(f"--- Cognitive Settings Loader Test (Simplified) ---")
    print(f"Loaded Cognitive Settings: {_settings_data}")
    # print(f"Example Setting (if added later): COGNITIVE_DEFAULT_SETTING")