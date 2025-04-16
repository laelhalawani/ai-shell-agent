"""
Utility functions used across the AI Shell Agent.
"""
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, Optional # Added Dict, Optional

from . import logger, console_io # Add console_io import

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
        # Return a deep copy of the default value if it's mutable
        if isinstance(default_value, (dict, list)):
             return json.loads(json.dumps(default_value))
        return default_value if default_value is not None else {}

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
        # Use replace for atomic write on Unix-like systems, fallback needed for others?
        # os.replace is generally cross-platform for files.
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}", exc_info=True)
        if 'tmp_path' in locals() and tmp_path.exists():
            try: tmp_path.unlink()
            except Exception as rm_e: logger.error(f"Failed to remove temporary file {tmp_path}: {rm_e}")

# --- NEW .env Utility Functions ---

def read_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """
    Reads a .env file and returns a dictionary of key-value pairs.
    Handles comments and empty lines.
    """
    env_vars = {}
    if dotenv_path.exists():
        try:
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip("'\"") # Remove potential quotes
                        if key: # Ensure key is not empty
                            env_vars[key] = value
        except Exception as e:
            logger.error(f"Error reading .env file {dotenv_path}: {e}", exc_info=True)
    return env_vars

def write_dotenv(dotenv_path: Path, env_vars: Dict[str, str]) -> None:
    """
    Writes a dictionary of key-value pairs to a .env file.
    Overwrites the file, ensuring proper formatting.
    """
    try:
        # Ensure parent directory exists
        dotenv_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file path
        tmp_path_str = f"{dotenv_path}.tmp.{uuid.uuid4()}"
        tmp_path = Path(tmp_path_str)

        with open(tmp_path, 'w', encoding='utf-8') as f:
            for key, value in sorted(env_vars.items()): # Write sorted for consistency
                 # Basic quoting for values with spaces or special chars, adjust if needed
                 if ' ' in value or '#' in value or '=' in value:
                      f.write(f'{key}="{value}"\n')
                 else:
                      f.write(f'{key}={value}\n')

        # Atomically replace the original file
        os.replace(tmp_path, dotenv_path)
        logger.debug(f"Successfully wrote to .env file: {dotenv_path}")

    except Exception as e:
        logger.error(f"Error writing to .env file {dotenv_path}: {e}", exc_info=True)
        if 'tmp_path' in locals() and tmp_path.exists():
            try: tmp_path.unlink()
            except Exception as rm_e: logger.error(f"Failed to remove temporary .env file {tmp_path}: {rm_e}")

def ensure_dotenv_key(dotenv_path: Path, key: str, description: Optional[str] = None) -> Optional[str]:
    """
    Ensures a key exists in the environment and .env file using console_io for prompts.

    Checks os.environ first. If not found, prompts the user.
    If the user provides a value, it's saved to the .env file and os.environ.

    Args:
        dotenv_path: Path to the .env file.
        key: The environment variable key to ensure.
        description: A description shown to the user when prompting (e.g., 'API Key for X service' or a URL).

    Returns:
        The value of the key if found or provided, otherwise None.
    """
    value = os.getenv(key)
    if value:
        logger.debug(f"Found key '{key}' in environment.")
        return value

    logger.warning(f"Environment variable '{key}' not found.")
    # Use console_io for system/info messages
    console_io.print_system(f"\nConfiguration required: Missing environment variable '{key}'.")
    if description:
        console_io.print_info(f"Description: {description}")

    try:
        # Use console_io prompt
        user_input = console_io.prompt_for_input(f"Please enter the value for {key}").strip()

        if not user_input:
            logger.warning(f"User skipped providing value for '{key}'.")
            console_io.print_warning("Input skipped.") # Use console_io
            return None

        # Value provided, save it
        current_env_vars = read_dotenv(dotenv_path)
        current_env_vars[key] = user_input
        write_dotenv(dotenv_path, current_env_vars)
        
        # Update os.environ for the current session
        os.environ[key] = user_input

        logger.info(f"Saved '{key}' to {dotenv_path} and updated environment.")
        console_io.print_info(f"Value for '{key}' saved.") # Use console_io
        return user_input

    except KeyboardInterrupt:
         # Handled by console_io.prompt_for_input
         # logger.warning(f"User cancelled input for key '{key}'.") # Already logged by console_io
         return None
    except Exception as e:
         logger.error(f"Error during ensure_dotenv_key for '{key}': {e}", exc_info=True)
         console_io.print_error(f"An unexpected error occurred while handling '{key}'. Check logs.") # Use console_io
         return None