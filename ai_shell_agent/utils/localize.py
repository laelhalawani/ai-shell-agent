# ai_shell_agent/utils/localize.py
"""
Handles automated localization of text files using an LLM.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import copy
import fnmatch  # Added for pattern matching

from .. import logger, ROOT_DIR
from ..llm import get_translation_llm # Use the specific translation LLM getter
from ..config_manager import get_translation_model
from ..prompts.prompts import TRANSLATION_PROMPT_TEMPLATE # Import the template
from .file_io import read_json, write_json
from ..console_manager import get_console_manager
from ..texts import get_text # Import to get UI text for the localization process

# Get console manager instance
console = get_console_manager()

# --- Define Key Patterns to Exclude from Translation ---
# Uses fnmatch pattern matching (e.g., *, ?, [seq], [!seq])
# Add patterns for keys whose string values should NOT be translated.
EXCLUDED_KEY_PATTERNS: List[str] = [
    "tools.*" # Exclude all keys under 'tools' (e.g., tools.tool_name.*)
    # Add more patterns here as needed
]
logger.debug(f"Localization exclusion patterns: {EXCLUDED_KEY_PATTERNS}")
# --- END ---

def discover_english_text_files() -> List[Path]:
    """Finds all en_texts.json files within the ai_shell_agent directory."""
    search_dir = ROOT_DIR / 'ai_shell_agent'
    logger.debug(f"Searching for en_texts.json in {search_dir} and subdirectories...")
    en_files = list(search_dir.rglob('en_texts.json'))
    logger.info(f"Found {len(en_files)} English text files to process.")
    return en_files

def _count_string_keys(data: Any) -> int:
    """Recursively counts the number of string values in a nested dictionary."""
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            count += _count_string_keys(value)
    elif isinstance(data, list):
        for item in data:
            count += _count_string_keys(item)
    elif isinstance(data, str):
        count = 1
    return count

def _translate_string(
    llm,
    text_to_translate: str,
    target_language: str,
    key_path: str,
    top_level_key: str,
    top_level_dict: Dict
) -> Optional[str]:
    """Translates a single string using the LLM."""
    try:
        # Format the top-level dictionary context nicely for the prompt
        top_level_dict_str = json.dumps(top_level_dict, indent=2, ensure_ascii=False)

        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            target_language=target_language,
            translation_key=key_path.split('.')[-1], # Get the last part of the key
            top_level_key=top_level_key,
            top_level_keys_value_dict_str=top_level_dict_str,
            object_text_path=key_path,
            object_text_path_value=text_to_translate,
            # Re-add target_language for the final instruction if needed by template
            taget_language=target_language
        )

        logger.debug(f"Translation prompt for '{key_path}':\n{prompt[:200]}...") # Log truncated prompt
        ai_response = llm.invoke(prompt)
        translation = ai_response.content.strip()

        # Basic validation: Check if the result is empty or just whitespace
        if not translation:
             logger.warning(f"LLM returned empty translation for key '{key_path}'. Keeping original.")
             return text_to_translate # Return original if translation is empty

        # Optional: Add more validation here (e.g., check for placeholder preservation)

        logger.debug(f"Translation for '{key_path}': '{translation}'")
        return translation

    except Exception as e:
        logger.error(f"LLM translation failed for key '{key_path}': {e}", exc_info=True)
        return text_to_translate # Return original text on error

def _translate_recursive(
    data: Any,
    llm,
    target_language: str,
    progress_callback: callable,
    key_path_prefix: str = "",
    original_data: Optional[Dict] = None # Pass original full data for context
) -> Any:
    """
    Recursively traverses the data, translates strings (skipping excluded keys),
    and calls progress callback for every string key encountered.
    """
    if original_data is None:
        original_data = data # Use top-level data as original context initially

    if isinstance(data, dict):
        translated_dict = {}
        for key, value in data.items():
            current_key_path = f"{key_path_prefix}.{key}" if key_path_prefix else key
            # Determine top-level key and its dictionary for context prompt
            top_level_key = current_key_path.split('.')[0]
            top_level_dict_context = original_data.get(top_level_key, {})

            translated_dict[key] = _translate_recursive(
                value, llm, target_language, progress_callback, current_key_path, original_data
            )
        return translated_dict
    elif isinstance(data, list):
        # Lists usually don't contain localizable strings directly in this app's structure
        return data
    elif isinstance(data, str):
        # --- START EXCLUSION CHECK ---
        is_excluded = False
        for pattern in EXCLUDED_KEY_PATTERNS:
            if fnmatch.fnmatch(key_path_prefix, pattern):
                logger.debug(f"Skipping translation for key '{key_path_prefix}' due to pattern '{pattern}'")
                is_excluded = True
                break # No need to check other patterns
        # --- END EXCLUSION CHECK ---

        translation = data # Default to original value

        if not is_excluded:
            # This is a string leaf node NOT excluded, translate it
            translation = _translate_string(
                llm, data, target_language, key_path_prefix,
                key_path_prefix.split('.')[0], # top_level_key
                original_data.get(key_path_prefix.split('.')[0], {}) # top_level_dict
            )
            # Note: _translate_string returns original on error

        # Call progress callback for EVERY string key, even skipped ones
        progress_callback()
        return translation
    else:
        # Return non-string, non-dict, non-list types as is
        return data

def localize_all_texts(target_language: str):
    """
    Discovers all en_texts.json files, translates them using the configured translation LLM,
    and saves them as <lang>_texts.json.
    """
    # Get translation model name for display
    translation_model_name = get_translation_model()
    
    console.display_message(
        get_text("common.labels.system"),
        get_text("localize.start_detailed", target_language=target_language, model_name=translation_model_name), 
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )

    llm = get_translation_llm()
    if not llm:
        console.display_message(
            get_text("common.labels.error"),
            get_text("localize.error_llm_init_specific", model_name=translation_model_name), 
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )
        return

    en_files = discover_english_text_files()
    if not en_files:
        console.display_message(
            get_text("common.labels.warning"),
            get_text("localize.warn_no_files"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        return

    total_keys_to_translate = 0
    files_data = {}
    for en_file_path in en_files:
        data = read_json(en_file_path)
        if data:
            files_data[en_file_path] = data
            total_keys_to_translate += _count_string_keys(data)
        else:
            logger.warning(f"Skipping empty or invalid file: {en_file_path}")

    if total_keys_to_translate == 0:
        console.display_message(
            get_text("common.labels.warning"),
            get_text("localize.warn_no_keys"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        return
        
    # Display progress info before starting translation
    console.display_message(
        get_text("common.labels.system"),
        get_text("localize.translating_start",
                 count=total_keys_to_translate,
                 target_language=target_language,
                 model_name=translation_model_name),
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )

    processed_keys = 0

    def progress_callback():
        nonlocal processed_keys
        processed_keys += 1
        console.display_progress(processed_keys, total_keys_to_translate)

    all_translations_succeeded = True
    for en_file_path, original_data in files_data.items():
        logger.info(f"Processing file: {en_file_path}")
        # Use deepcopy to avoid modifying the original data used for context
        data_to_translate = copy.deepcopy(original_data)

        # Translate data recursively
        translated_data = _translate_recursive(
            data_to_translate, llm, target_language, progress_callback, original_data=original_data
        )

        # Save the translated data
        target_file_name = f"{target_language.lower()}_texts.json"
        target_file_path = en_file_path.parent / target_file_name

        logger.info(f"Saving translations to: {target_file_path}")
        if not write_json(target_file_path, translated_data):
            console.display_message(
                get_text("common.labels.error"),
                get_text("localize.error_save", target_file=str(target_file_path)),
                console.STYLE_ERROR_LABEL,
                console.STYLE_ERROR_CONTENT
            )
            all_translations_succeeded = False

    # Final completion message
    console.console.print() # Print a newline to clear the progress line fully
    if all_translations_succeeded:
        console.display_message(
            get_text("common.labels.info"),
            get_text("localize.complete", target_language=target_language, count=total_keys_to_translate),
            console.STYLE_INFO_LABEL,
            console.STYLE_INFO_CONTENT
        )
    else:
         console.display_message(
            get_text("common.labels.warning"),
            get_text("localize.complete_errors"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )