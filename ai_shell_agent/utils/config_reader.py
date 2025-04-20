# ai_shell_agent/utils/config_reader.py
"""
Low-level utility to read the main application config file (data/config.json)
without triggering higher-level dependencies.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Cannot import from .. import logger here due to potential cycles during init
# Use standard logging temporarily if needed for debugging this specific file
# import logging
# config_reader_logger = logging.getLogger(__name__) # Separate logger

# Define path relative to this file's location
# Assumes utils/ is inside ai_shell_agent/ which is inside the root
_CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / 'data' / 'config.json'

def _read_raw_config() -> Dict:
    """Reads the raw config file directly."""
    if _CONFIG_FILE_PATH.exists():
        with open(_CONFIG_FILE_PATH, "r", encoding='utf-8') as f:
            return json.load(f)


def get_config_value(key: str, default: Any = None) -> Any:
    """Reads a specific value from the main config file."""
    config = _read_raw_config()
    return config.get(key, default)