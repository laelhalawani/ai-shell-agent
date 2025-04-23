# ai_shell_agent/toolsets/cognitive/texts.py
"""
Loads Cognitive toolset text strings based on selected language.
"""
from pathlib import Path
from string import Formatter
# Use relative imports within the package
from ...utils.file_io import read_json
from ...utils.dict_utils import deep_merge_dicts
# Import the low-level config reader utility
from ...utils.config_reader import get_config_value
from ... import logger
# Import default lang from settings for fallback
from ...settings import APP_DEFAULT_LANGUAGE

_toolset_id = 'cognitive' # <--- Use the correct toolset ID
_texts_dir = Path(__file__).parent / 'texts'
_texts_data = {}

def _load_texts():
    """Loads text strings, merging the selected language over English defaults."""
    global _texts_data
    # Get language using the low-level utility, providing the app default as fallback
    lang = get_config_value("language", default=APP_DEFAULT_LANGUAGE)
    if not lang or not isinstance(lang, str):
        lang = APP_DEFAULT_LANGUAGE

    en_file = _texts_dir / "en_texts.json"
    lang_file = _texts_dir / f"{lang}_texts.json"

    # 1. Load English (required fallback)
    english_data = read_json(en_file, default_value=None)
    if english_data is None:
        logger.error(f"Required English language file missing or invalid for toolset '{_toolset_id}': {en_file}. Texts will be unavailable.")
        _texts_data = {}
        return
    else:
        _texts_data = english_data
        logger.debug(f"Loaded English texts for toolset '{_toolset_id}' from {en_file}")

    # 2. If selected language is not English, try to load and merge it
    if lang != "en":
        if lang_file.exists():
            lang_data = read_json(lang_file, default_value=None)
            if lang_data is not None:
                _texts_data = deep_merge_dicts(english_data, lang_data)
                logger.debug(f"Merged '{lang}' language texts for toolset '{_toolset_id}' from {lang_file}")
            else:
                logger.warning(f"Language file for '{lang}' found but invalid for toolset '{_toolset_id}': {lang_file}. Using English only.")
        else:
            logger.warning(f"Language file for '{lang}' not found for toolset '{_toolset_id}': {lang_file}. Using English only.")

def get_text(key_path: str, **kwargs) -> str:
    """
    Retrieves a text string for this toolset using a dot-separated key path
    and formats it using provided keyword arguments.

    Returns the key_path itself if the text is not found or formatting fails.
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
                logger.warning(f"Missing key '{e}' for formatting toolset '{_toolset_id}' text '{key_path}'. Provided args: {kwargs}")
                return value # Return raw template
            except Exception as format_e:
                 logger.error(f"Error formatting toolset '{_toolset_id}' text '{key_path}' with args {kwargs}: {format_e}")
                 return value # Return raw template
        else:
            logger.warning(f"Toolset '{_toolset_id}' text key '{key_path}' did not resolve to a string. Found type: {type(value)}")
            return key_path
    except (KeyError, TypeError):
        logger.error(f"Toolset '{_toolset_id}' text key '{key_path}' not found.")
        return key_path # Return key path itself

# --- Load texts when module is imported ---
_load_texts()

if __name__ == '__main__':
    print(f"--- Cognitive Texts Loader Test ---")
    test_lang = get_config_value("language", default=APP_DEFAULT_LANGUAGE)
    print(f"Current Language (from config): {test_lang}")
    print(f"Loaded Keys: {list(_texts_data.keys())}")
    # Example: print(f"Test get_text: {get_text('tools.analyse.name')}")