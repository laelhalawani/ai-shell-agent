"""
Utility functions used across the AI Shell Agent.
Focuses on .env file handling and other non-I/O utilities.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .. import logger

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
    Overwrites the file, ensuring proper formatting. Uses atomic write.
    """
    tmp_path = None # Initialize
    try:
        # Ensure parent directory exists
        dotenv_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file path
        tmp_path_str = f"{dotenv_path.name}.tmp.{uuid.uuid4()}.env" # Need uuid here still
        import uuid
        tmp_path = dotenv_path.parent / tmp_path_str

        with open(tmp_path, 'w', encoding='utf-8') as f:
            for key, value in sorted(env_vars.items()): # Write sorted for consistency
                 # Basic quoting for values with spaces or special chars, adjust if needed
                 if ' ' in value or '#' in value or '=' in value or not value: # Also quote empty strings
                      f.write(f'{key}="{value}"\n')
                 else:
                      f.write(f'{key}={value}\n')

        # Atomically replace the original file
        os.replace(tmp_path, dotenv_path)
        logger.debug(f"Successfully wrote to .env file: {dotenv_path}")

    except Exception as e:
        logger.error(f"Error writing to .env file {dotenv_path}: {e}", exc_info=True)
        if tmp_path and tmp_path.exists():
            try: tmp_path.unlink()
            except Exception as rm_e: logger.error(f"Failed to remove temporary .env file {tmp_path}: {rm_e}")
    finally:
        if tmp_path and tmp_path.exists():
             try: tmp_path.unlink()
             except Exception: pass

def ensure_dotenv_key(dotenv_path: Path, key: str, description: Optional[str] = None) -> Optional[str]:
    """
    Ensures a key exists in the environment and .env file using ConsoleManager for prompts.
    Checks os.environ first. If not found, prompts the user.
    If the user provides a value, it's saved to the .env file and os.environ.
    """
    value = os.getenv(key)
    if value:
        logger.debug(f"Found key '{key}' in environment.")
        return value

    from ..console_manager import get_console_manager
    console = get_console_manager()

    logger.warning(f"Environment variable '{key}' not found.")
    console.display_message("SYSTEM:", f"\nConfiguration required: Missing environment variable '{key}'.",
                           console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    if description:
        console.display_message("INFO:", f"Description: {description}",
                               console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)

    try:
        user_input = console.prompt_for_input(f"Please enter the value for {key}", is_password=True).strip()

        if not user_input:
            logger.warning(f"User skipped providing value for '{key}'.")
            console.display_message("WARNING:", "Input skipped.",
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return None

        current_env_vars = read_dotenv(dotenv_path)
        current_env_vars[key] = user_input
        write_dotenv(dotenv_path, current_env_vars)

        os.environ[key] = user_input

        logger.info(f"Saved '{key}' to {dotenv_path} and updated environment.")
        console.display_message("INFO:", f"Value for '{key}' saved.",
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return user_input

    except KeyboardInterrupt:
         return None
    except Exception as e:
         logger.error(f"Error during ensure_dotenv_key for '{key}': {e}", exc_info=True)
         console.display_message("ERROR:", f"An unexpected error occurred while handling '{key}'. Check logs.",
                               console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
         return None