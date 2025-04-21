# File: ai_shell_agent/toolsets/files/settings.py

"""
Loads File Manager toolset default settings.
"""
from pathlib import Path
from typing import Any, Dict
from ...utils.file_io import read_json
from ... import logger

_toolset_id = 'files'
_settings_file = Path(__file__).parent / 'default_settings' / 'default_settings.json'

def _load_data(file_path):
    data = read_json(file_path, default_value={})
    if not data:
        logger.error(f"Error loading settings for toolset '{_toolset_id}' from {file_path}. Defaults missing.")
    return data

_settings_data: Dict[str, Any] = _load_data(_settings_file)

try:
    FILES_HISTORY_LIMIT: int = _settings_data['history_retrieval_limit']
    FIND_FUZZY_DEFAULT: bool = bool(_settings_data['find_fuzzy_default'])
    FIND_THRESHOLD_DEFAULT: int = _settings_data['find_threshold_default']
    FIND_LIMIT_DEFAULT: int = _settings_data['find_limit_default']
    FIND_WORKERS_DEFAULT: int = _settings_data['find_workers_default']
except KeyError as e:
    logger.critical(f"Missing expected key in files default_settings.json: {e}. Toolset cannot function correctly.")
    raise
