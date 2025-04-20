# ai_shell_agent/settings.py
"""
Loads global application default settings from default_settings.json
and provides them as constants. Assumes the JSON file exists and is valid.
"""
from pathlib import Path
from .utils.file_io import read_json
from . import logger # Keep logger for potential debug messages if needed elsewhere

_settings_file = Path(__file__).parent / 'default_settings' / 'default_settings.json'

# Simplified loading function (using existing utility)
def _load_data(file_path):
    """Basic JSON loader."""
    data = read_json(file_path, default_value={}) # read_json handles file not found/decode errors returning {}
    if not data:
         # Log an error but proceed; subsequent access will raise KeyError if keys missing
         logger.error(f"Critical error: Failed to load or parse JSON data from {file_path}. Application defaults will be missing.")
    return data

# Load the data into a module-level variable
_settings_data = _load_data(_settings_file)

# --- Define Constants by Direct Access ---
# These will raise KeyError if the keys don't exist in the loaded JSON data.
try:
    APP_DEFAULT_MODEL = _settings_data['app']['default_model']
    APP_DEFAULT_LANGUAGE = _settings_data['app']['language']
    CHAT_MAX_ITERATIONS = _settings_data['chat']['max_react_iterations']
    # Ensure the default enabled toolsets is always a list after loading
    _raw_default_toolsets = _settings_data['chat']['default_enabled_toolsets']
    DEFAULT_ENABLED_TOOLSETS_NAMES = _raw_default_toolsets if isinstance(_raw_default_toolsets, list) else [] # Basic type check
    CONSOLE_CONDENSED_OUTPUT_LENGTH = _settings_data['console']['condensed_output_length']
    # Add any other top-level constants needed from default_settings.json here
except KeyError as e:
    logger.critical(f"Missing expected key in global default_settings.json: {e}. Application cannot start correctly.")
    raise # Re-raise the error to halt execution if critical settings are missing


if __name__ == '__main__':
    # Example usage for testing the loader
    print(f"--- Global Settings Loader Test (Simplified) ---")
    print(f"Loaded Settings Data: {_settings_data}")
    print(f"APP_DEFAULT_MODEL: {APP_DEFAULT_MODEL}")
    print(f"APP_DEFAULT_LANGUAGE: {APP_DEFAULT_LANGUAGE}")
    print(f"CHAT_MAX_ITERATIONS: {CHAT_MAX_ITERATIONS}")
    print(f"DEFAULT_ENABLED_TOOLSETS_NAMES: {DEFAULT_ENABLED_TOOLSETS_NAMES}")
    print(f"CONSOLE_CONDENSED_OUTPUT_LENGTH: {CONSOLE_CONDENSED_OUTPUT_LENGTH}")