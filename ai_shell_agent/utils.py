"""
Utility functions used across the AI Shell Agent.
"""
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Any

from . import logger

def read_json(file_path: Path, default_value=None) -> Any:
    """Reads a JSON file or returns a default value if not found.
    
    Args:
        file_path: Path to the JSON file
        default_value: Value to return if the file doesn't exist or is invalid
        
    Returns:
        The parsed JSON data or the default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if not isinstance(e, FileNotFoundError):
            logger.error(f"Error parsing JSON from {file_path}: {e}. Returning default.")
        return json.loads(json.dumps(default_value)) if default_value is not None else {}

def write_json(file_path: Path, data: Any) -> None:
    """Writes data to a JSON file, creating directories if needed.
    
    Args:
        file_path: Path to the JSON file
        data: Data to write to the file
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_str = f"{file_path}.tmp.{uuid.uuid4()}"
        tmp_path = Path(tmp_path_str)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}", exc_info=True)
        if 'tmp_path' in locals() and tmp_path.exists():
            try: tmp_path.unlink()
            except Exception as rm_e: logger.error(f"Failed to remove temporary file {tmp_path}: {rm_e}")