# ai_shell_agent/utils/config_reader.py
"""
Low-level utility to read the main application config file (data/config.json)
without triggering higher-level dependencies.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import the main application logger
from .. import logger # <-- Use the logger from __init__

# Define path relative to this file's location
# Assumes utils/ is inside ai_shell_agent/ which is inside the root
_CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / 'data' / 'config.json'

def _read_raw_config() -> Dict:
    """
    Reads the raw config file directly. Returns an empty dict if
    the file doesn't exist or is invalid JSON. Logs errors using the main app logger.
    """
    config_data = {} # Default to empty dict
    if _CONFIG_FILE_PATH.exists():
        try:
            with open(_CONFIG_FILE_PATH, "r", encoding='utf-8') as f:
                content = f.read()
                if content.strip(): # Check if file is not just whitespace
                    config_data = json.loads(content)
                else:
                    # Log using the main application logger
                    logger.warning(f"Config file exists but is empty: {_CONFIG_FILE_PATH}")
        except json.JSONDecodeError as e:
            # Log error using the main application logger
            logger.error(f"Failed to decode JSON from config file {_CONFIG_FILE_PATH}: {e}")
        except Exception as e:
            # Log other potential file reading errors using the main application logger
            logger.error(f"Error reading config file {_CONFIG_FILE_PATH}: {e}", exc_info=True)
    # else: File doesn't exist, silently return empty dict

    return config_data


def get_config_value(key: str, default: Any = None) -> Any:
    """Reads a specific value from the main config file."""
    config = _read_raw_config() # Now guaranteed to be a dict
    return config.get(key, default)