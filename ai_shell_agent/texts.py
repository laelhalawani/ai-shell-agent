# ai_shell_agent/texts.py
"""
Loads global user-facing text strings from texts/*.json files based on
selected language, falling back to English ('en').
"""
from pathlib import Path
from string import Formatter
from typing import Dict, Any

# Use relative imports for utils and config
from .utils.file_io import read_json
from .utils.dict_utils import deep_merge_dicts
from . import logger
# Import the low-level config reader utility
from .utils.config_reader import get_config_value
# Import default lang from settings for fallback
from .settings import APP_DEFAULT_LANGUAGE

# Define the directory containing the text files for the core agent
_texts_dir = Path(__file__).parent / 'texts'
_texts_data = {}  # This will hold the final merged data

def _load_texts():
    """Loads text strings, merging the selected language over English defaults."""
    global _texts_data
    # Get language using the low-level utility, providing the app default as fallback
    lang = get_config_value("language", default=APP_DEFAULT_LANGUAGE)
    # Ensure lang is a valid string, fallback again if needed
    if not lang or not isinstance(lang, str):
        lang = APP_DEFAULT_LANGUAGE

    en_file = _texts_dir / "en_texts.json"
    lang_file = _texts_dir / f"{lang}_texts.json"

    # 1. Load English (required fallback)
    english_data = read_json(en_file, default_value=None)
    if english_data is None:
        logger.critical(f"Core English language file missing or invalid: {en_file}. Text lookups will fail.")
        _texts_data = {}  # Ensure data is empty if base fails
        return
    else:
        # Start with English data
        _texts_data = english_data
        logger.debug(f"Loaded core English texts from {en_file}")

    # 2. If selected language is not English, try to load and merge it
    if lang != "en":
        if lang_file.exists():
            lang_data = read_json(lang_file, default_value=None)
            if lang_data is not None:
                # Deep merge selected language over English
                _texts_data = deep_merge_dicts(english_data, lang_data)
                logger.debug(f"Merged core '{lang}' language texts from {lang_file}")
            else:
                logger.warning(f"Core language file for '{lang}' found but invalid: {lang_file}. Using English only.")
                # _texts_data already contains english_data
        else:
            logger.warning(f"Core language file for '{lang}' not found: {lang_file}. Using English only.")
            # _texts_data already contains english_data
    # else: language is 'en', no merge needed

def get_text(key_path: str, **kwargs) -> str:
    """
    Retrieves a text string using a dot-separated key path (e.g., 'cli.errors.chat_not_found')
    and formats it using provided keyword arguments.

    Returns the key_path itself if the text is not found, to help identify missing strings.
    """
    keys = key_path.split('.')
    value = _texts_data
    try:
        for key in keys:
            value = value[key]

        if isinstance(value, str):
            try:
                formatter = Formatter()
                return formatter.format(value, **kwargs)
            except KeyError as e:
                logger.warning(f"Missing key '{e}' for formatting text '{key_path}'. Provided args: {kwargs}")
                return value # Return raw template
            except Exception as format_e:
                 logger.error(f"Error formatting text '{key_path}' with args {kwargs}: {format_e}")
                 return value # Return raw template
        else:
            logger.warning(f"Text key '{key_path}' did not resolve to a string. Found type: {type(value)}")
            return key_path # Return key path itself
    except (KeyError, TypeError):
        logger.error(f"Text key '{key_path}' not found in loaded texts.")
        return key_path # Return key path itself


# --- Load texts when module is imported ---
_load_texts()

if __name__ == '__main__':
    # Example usage for testing the loader
    print(f"--- Core Texts Loader Test ---")
    # Test reading language directly for the test
    test_lang = get_config_value("language", default=APP_DEFAULT_LANGUAGE)
    print(f"Current Language (from config): {test_lang}")
    print(f"Loaded Texts Keys (Top Level): {list(_texts_data.keys())}")
    # Assuming en_texts.json has {"cli": {"errors": {"chat_not_found": "Chat '{title}' not found."}}}
    print(f"Test get_text (cli.errors.chat_not_found): {get_text('cli.errors.chat_not_found', title='My Chat')}")
    print(f"Test get_text (common.labels.info): {get_text('common.labels.info')}")
    print(f"Test missing key: {get_text('a.b.c.missing_key')}")
    print(f"Test formatting missing arg: {get_text('cli.errors.chat_not_found')}") # Should show raw template