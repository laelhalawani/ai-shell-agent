"""
Handles basic file input/output operations, specifically JSON reading/writing.
"""
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Any

# Import the logger setup from the main __init__
from .. import logger

def read_json(file_path: Path, default_value=None) -> Any:
    """Reads a JSON file or returns a default value if not found or invalid.

    Args:
        file_path: Path to the JSON file
        default_value: Value to return if the file doesn't exist or is invalid

    Returns:
        The parsed JSON data or the default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Don't log error for file not found, just return default
        pass
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {file_path}: {e}. Returning default.")
    except Exception as e:
        logger.error(f"Unexpected error reading JSON from {file_path}: {e}", exc_info=True)

    # Return a deep copy of the default value if it's mutable
    if isinstance(default_value, (dict, list)):
         try:
             return json.loads(json.dumps(default_value))
         except Exception: # Fallback if default_value itself is not JSON serializable
              return default_value
    return default_value


def write_json(file_path: Path, data: Any) -> bool:
    """Writes data to a JSON file atomically, creating directories if needed.

    Args:
        file_path: Path to the JSON file
        data: Data to write to the file

    Returns:
        True if write was successful, False otherwise.
    """
    tmp_path = None # Initialize in case of early error
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a more robust temp file name within the same directory
        tmp_path_str = f"{file_path.name}.tmp.{uuid.uuid4()}.json"
        tmp_path = file_path.parent / tmp_path_str

        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # Use replace for atomic write
        os.replace(tmp_path, file_path)
        logger.debug(f"Successfully wrote JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}", exc_info=True)
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.debug(f"Removed temporary file {tmp_path}")
            except Exception as rm_e:
                logger.error(f"Failed to remove temporary file {tmp_path}: {rm_e}")
        return False
    finally:
        # Ensure tmp_path is cleaned up if os.replace failed but file was created
        if tmp_path and tmp_path.exists():
             try:
                 tmp_path.unlink()
             except Exception: pass