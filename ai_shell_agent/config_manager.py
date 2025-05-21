import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from . import logger
from .paths import ROOT_DIR, DEFAULT_DOTENV_PATH # Added DEFAULT_DOTENV_PATH
from .settings import APP_DEFAULT_MODEL, APP_DEFAULT_LANGUAGE, DEFAULT_ENABLED_TOOLSETS_NAMES, APP_DEFAULT_TRANSLATION_MODEL
from .console_manager import get_console_manager # Import ConsoleManager
from .texts import get_text # Added import for get_text

# Get console manager instance
console = get_console_manager()

# Define model mappings
OPENAI_MODELS = {
    "gpt-4o": "gpt-4o",
    "4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "4o-mini": "gpt-4o-mini",
    "o3-mini": "o3-mini",
    # Removed o1 and o1-mini as they don't support system messages
}

GOOGLE_MODELS = {
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-2.5-pro": "gemini-2.5-pro-exp-03-25",
}

ALL_MODELS = {**OPENAI_MODELS, **GOOGLE_MODELS}

# DEFAULT_MODEL removed, now using APP_DEFAULT_MODEL from settings.py

def get_data_dir():
    """Return the directory where configuration data should be stored."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

CONFIG_FILE = os.path.join(get_data_dir(), "config.json")

def _read_config() -> Dict:
    """Read the configuration from the config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Config file {CONFIG_FILE} is corrupted. Returning empty config.")
                return {}
    return {}

def _write_config(config: Dict) -> None:
    """Write the configuration to the config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def get_model_provider(model_name: str) -> str:
    """Determine the provider (OpenAI or Google) for a given model name."""
    normalized_name = ALL_MODELS.get(model_name, model_name)
    if normalized_name in OPENAI_MODELS.values():
        return "openai"
    elif normalized_name in GOOGLE_MODELS.values():
        return "google"
    else:
        # Default to OpenAI if the model is not recognized
        return "openai"

def normalize_model_name(model_name: str) -> str:
    """Convert shorthand model names to their full names."""
    return ALL_MODELS.get(model_name, model_name)

def get_current_model() -> str:
    """
    Get the currently configured model, prioritizing environment variable over config file.
    """
    # First check environment variable
    env_model = os.getenv("AI_SHELL_AGENT_MODEL")
    if env_model:
        return env_model
    
    # Then check config file
    config = _read_config()
    model = config.get("model")
    
    # If neither exists, use default from settings and initialize it
    if not model:
        model = APP_DEFAULT_MODEL # Use imported constant
        set_model(model) # This will save it to config
    
    return model

def set_model(model_name: str) -> None:
    """
    Set the model to use for AI interactions, saving to environment variable and config file.
    No longer saves to .env file - toolsets handle their own secrets.
    API keys are now handled by ensure_dotenv_key in utils.env, which uses DEFAULT_DOTENV_PATH
    """
    normalized_name = normalize_model_name(model_name)
    
    # Save to environment variable (for current session only)
    os.environ["AI_SHELL_AGENT_MODEL"] = normalized_name
    
    # Save to config file (for persistence between sessions)
    config = _read_config()
    config["model"] = normalized_name
    _write_config(config)
    
    logger.info(f"Model set to: {normalized_name}")

def prompt_for_model_selection() -> Optional[str]:
    """Prompts user for model selection using ConsoleManager with highlighting."""
    current_model = get_current_model()
    
    # Create a map of model names to their aliases
    model_aliases = {}
    for alias, full_name in ALL_MODELS.items():
        if full_name in model_aliases:
            model_aliases[full_name].append(alias)
        else:
            model_aliases[full_name] = [alias]
    
    # Remove the full names from the aliases list to avoid redundancy
    for full_name in model_aliases:
        if full_name in model_aliases[full_name]:
            model_aliases[full_name].remove(full_name)
    
    from rich.text import Text # Import Text

    console.display_message(get_text("common.labels.system"), get_text("config.model_select.available_title"), console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    console.display_message(get_text("common.labels.system"), get_text("config.model_select.openai_header"), console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    openai_lines = []
    current_marker = get_text("config.model_select.current_marker")
    for model in set(OPENAI_MODELS.values()):
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        # Prepare alias string with markup before passing to get_text
        alias_markup = ', '.join([f'[underline]{a}[/underline]' for a in aliases])
        alias_text = get_text("config.model_select.aliases_suffix", alias_str=alias_markup) if aliases else ""
        marker = current_marker if model == current_model else ""
        line_text = Text.from_markup(f"- [underline]{model}[/underline]{alias_text}{marker}")
        openai_lines.append(line_text)
    for line in openai_lines:
        console.console.print(line)
    
    console.display_message(get_text("common.labels.system"), get_text("config.model_select.google_header"), console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    google_lines = []
    for model in set(GOOGLE_MODELS.values()):
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        alias_markup = ', '.join([f'[underline]{a}[/underline]' for a in aliases])
        alias_text = get_text("config.model_select.aliases_suffix", alias_str=alias_markup) if aliases else ""
        marker = current_marker if model == current_model else ""
        line_text = Text.from_markup(f"- [underline]{model}[/underline]{alias_text}{marker}")
        google_lines.append(line_text)
    for line in google_lines:
        console.console.print(line)
    
    try:
        prompt_msg = get_text("config.model_select.prompt", current_model=current_model)
        selected_model = console.prompt_for_input(prompt_msg).strip()
        
        if not selected_model:
            return current_model
        
        normalized_model = normalize_model_name(selected_model)
        if normalized_model not in set(OPENAI_MODELS.values()) and normalized_model not in set(GOOGLE_MODELS.values()):
            console.display_message(get_text("common.labels.warning"), get_text("config.model_select.warn_unknown", selected_model=selected_model, current_model=current_model),
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            logger.warning(f"Unknown model selected: {selected_model}. Using current: {current_model}")
            return current_model
        
        return normalized_model
    except KeyboardInterrupt:
        # Handled by ConsoleManager.prompt_for_input
        return None # Indicate cancellation
    except Exception as e:
        console.display_message(get_text("common.labels.error"), get_text("config.model_select.error_generic", error=e),
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        logger.error(f"Model selection error: {e}", exc_info=True)
        return None

# --- Translation Model Functions ---

TRANSLATION_MODEL_CONFIG_KEY = "translation_model"
TRANSLATION_MODEL_ENV_VAR = "AI_SHELL_AGENT_TRANSLATION_MODEL"

def get_translation_model() -> str:
    """
    Get the currently configured translation model.
    Priority: Env Var -> Config File -> Default Setting.
    """
    # 1. Check environment variable
    env_model = os.getenv(TRANSLATION_MODEL_ENV_VAR)
    if env_model:
        normalized_env_model = normalize_model_name(env_model)
        if normalized_env_model in ALL_MODELS.values():
            logger.debug(f"Using translation model from env var {TRANSLATION_MODEL_ENV_VAR}: {normalized_env_model}")
            return normalized_env_model
        else:
            logger.warning(f"Invalid model name '{env_model}' in env var {TRANSLATION_MODEL_ENV_VAR}. Ignoring.")

    # 2. Check config file
    config = _read_config()
    config_model = config.get(TRANSLATION_MODEL_CONFIG_KEY)
    if config_model:
        normalized_config_model = normalize_model_name(config_model)
        if normalized_config_model in ALL_MODELS.values():
            logger.debug(f"Using translation model from config file: {normalized_config_model}")
            return normalized_config_model
        else:
             logger.warning(f"Invalid model name '{config_model}' in config key '{TRANSLATION_MODEL_CONFIG_KEY}'. Ignoring.")

    # 3. Use default from settings
    logger.debug(f"Using default translation model from settings: {APP_DEFAULT_TRANSLATION_MODEL}")
    # Ensure the default itself is valid before returning
    normalized_default = normalize_model_name(APP_DEFAULT_TRANSLATION_MODEL)
    if normalized_default not in ALL_MODELS.values():
         logger.error(f"Default translation model '{APP_DEFAULT_TRANSLATION_MODEL}' from settings is invalid! Falling back to main default '{APP_DEFAULT_MODEL}'.")
         # Fallback to the main default model if the translation default is broken
         return normalize_model_name(APP_DEFAULT_MODEL)
    return normalized_default

def set_translation_model(model_name: str) -> None:
    """
    Set the translation model in the config file.
    """
    normalized_name = normalize_model_name(model_name)
    if normalized_name not in ALL_MODELS.values():
        logger.error(f"Attempted to set invalid translation model: {model_name} (normalized: {normalized_name}). Aborting save.")
        return

    config = _read_config()
    config[TRANSLATION_MODEL_CONFIG_KEY] = normalized_name
    _write_config(config)
    logger.info(f"Translation model set to: {normalized_name} in config file.")

def prompt_for_translation_model_selection() -> Optional[str]:
    """Prompts user for translation model selection using ConsoleManager."""
    current_translation_model = get_translation_model()

    # Create a map of model names to their aliases (same logic as main model prompt)
    model_aliases = {}
    for alias, full_name in ALL_MODELS.items():
        if full_name in model_aliases:
            model_aliases[full_name].append(alias)
        else:
            model_aliases[full_name] = [alias]
    for full_name in model_aliases:
        if full_name in model_aliases[full_name]:
            model_aliases[full_name].remove(full_name)

    from rich.text import Text

    # Use specific texts for translation model selection
    console.display_message(get_text("common.labels.system"), get_text("config.trans_model_select.title"),
                           console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    console.display_message(get_text("common.labels.system"), get_text("config.model_select.openai_header"),
                           console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    openai_lines = []
    current_marker = get_text("config.model_select.current_marker")
    for model in set(OPENAI_MODELS.values()):
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        alias_markup = ', '.join([f'[underline]{a}[/underline]' for a in aliases])
        alias_text = get_text("config.model_select.aliases_suffix", alias_str=alias_markup) if aliases else ""
        marker = current_marker if model == current_translation_model else ""
        line_text = Text.from_markup(f"- [underline]{model}[/underline]{alias_text}{marker}")
        openai_lines.append(line_text)
    for line in openai_lines:
        console.console.print(line)

    console.display_message(get_text("common.labels.system"), get_text("config.model_select.google_header"),
                           console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    google_lines = []
    for model in set(GOOGLE_MODELS.values()):
        aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
        alias_markup = ', '.join([f'[underline]{a}[/underline]' for a in aliases])
        alias_text = get_text("config.model_select.aliases_suffix", alias_str=alias_markup) if aliases else ""
        marker = current_marker if model == current_translation_model else ""
        line_text = Text.from_markup(f"- [underline]{model}[/underline]{alias_text}{marker}")
        google_lines.append(line_text)
    for line in google_lines:
        console.console.print(line)

    try:
        # Use specific prompt text
        prompt_msg = get_text("config.trans_model_select.prompt", current_model=current_translation_model)
        selected_model_input = console.prompt_for_input(prompt_msg).strip()

        if not selected_model_input:
             # User wants to keep the current setting
             return current_translation_model

        normalized_model = normalize_model_name(selected_model_input)
        if normalized_model not in ALL_MODELS.values():
            # Use specific warning text
            console.display_message(get_text("common.labels.warning"),
                                  get_text("config.trans_model_select.warn_unknown",
                                           selected_model=selected_model_input,
                                           current_model=current_translation_model),
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            logger.warning(f"Unknown translation model selected: {selected_model_input}. Keeping current: {current_translation_model}")
            return current_translation_model

        # Return the valid, normalized model selected by the user
        return normalized_model
    except KeyboardInterrupt:
        # Handled by ConsoleManager.prompt_for_input
        return None # Indicate cancellation
    except Exception as e:
        # Use specific error text
        console.display_message(get_text("common.labels.error"),
                              get_text("config.trans_model_select.error_generic", error=e),
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        logger.error(f"Translation model selection error: {e}", exc_info=True)
        return None

def ensure_api_key_for_translation_model() -> bool:
    """
    Ensure that the API key for the configured TRANSLATION model is set.
    If not, prompt the user to enter it.
    Uses the centralized get_api_key function.
    
    Returns:
        bool: True if the API key is set, False otherwise
    """
    trans_model = get_translation_model()
    provider = get_model_provider(trans_model)
    
    # Use the centralized get_api_key function
    api_key = get_api_key(provider)
    
    return api_key is not None

# --- End Translation Model Functions ---

def check_if_first_run() -> bool:
    """
    Check if this is the first run of the application. More robust check.

    Returns:
        bool: True if this is the first run, False otherwise
    """
    # If the main config file doesn't exist, it's definitely a first run.
    if not os.path.exists(CONFIG_FILE):
        logger.info("First run detected - config file missing.")
        return True

    # If the config file exists, check if a model is set either
    # in the config file or the environment variable.
    # If neither is set, treat it as needing setup (first run or reset state).
    env_model = os.getenv("AI_SHELL_AGENT_MODEL")
    config = _read_config()
    config_model = config.get("model")

    if not config_model and not env_model:
        logger.info("First run detected - no model configured in existing config or environment.")
        return True
        
    # Also consider language setting for first run
    config_lang = config.get("language")
    if not config_lang:
        logger.info("First run (or reset) detected - language not configured.")
        return True

    # If we have a model configured somewhere, it's not the first run.
    return False

def get_api_key_for_model(model_name: str) -> Tuple[Optional[str], str]:
    """
    Get the appropriate API key for the selected model.
    
    Returns:
        Tuple containing:
        - The API key (or None if not set)
        - The environment variable name for the API key
    """
    provider = get_model_provider(model_name)
    
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY"
    else:  # Google
        return os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY"

def set_api_key_for_model(model_name: str, api_key: Optional[str] = None) -> None:
    """
    Prompt for and save the appropriate API key using ConsoleManager.
    Uses DEFAULT_DOTENV_PATH for storing the API key.
    """
    provider = get_model_provider(model_name)
    provider_name = "OpenAI" if provider == "openai" else "Google"
    env_var_name = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"
    api_key_link = "https://platform.openai.com/api-keys" if provider == "openai" else "https://aistudio.google.com/app/apikey"
    
    if not api_key:
        prompt_info = get_text("config.api_key.prompt_instructions", provider_name=provider_name, api_key_link=api_key_link)
        console.display_message(get_text("common.labels.system"), prompt_info.strip(), console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        try:
            prompt_input_msg = get_text("config.api_key.prompt_input", provider_name=provider_name)
            api_key = console.prompt_for_input(prompt_input_msg, is_password=True).strip()
        except KeyboardInterrupt:
            logger.warning(f"API key input cancelled for {provider_name}.")
            return # Abort
        except Exception as e:
            console.display_message(get_text("common.labels.error"), get_text("config.api_key.error_generic", error=e),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            logger.error(f"API key input error: {e}", exc_info=True)
            return # Abort
    
    if not api_key:
        console.display_message(get_text("common.labels.warning"), get_text("config.api_key.warn_none_provided", provider_name=provider_name),
                              console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        logger.warning(f"No {provider_name} API key provided.")
        return
    
    os.environ[env_var_name] = api_key
    
    # Use DEFAULT_DOTENV_PATH instead of ROOT_DIR / '.env'
    from .utils.env import read_dotenv, write_dotenv # Import locally
    env_vars = read_dotenv(DEFAULT_DOTENV_PATH)
    env_vars[env_var_name] = api_key
    write_dotenv(DEFAULT_DOTENV_PATH, env_vars)
    
    console.display_message(get_text("common.labels.info"), get_text("config.api_key.info_saved", provider_name=provider_name),
                          console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
    logger.info(f"{provider_name} API key saved successfully to .env")

def ensure_api_key_for_current_model() -> bool:
    """
    Ensure that the API key for the current model is set.
    If not, prompt the user to enter it.
    Uses the centralized get_api_key function.
    
    Returns:
        bool: True if the API key is set, False otherwise
    """
    current_model = get_current_model()
    provider = get_model_provider(current_model)
    
    # Use the centralized get_api_key function
    api_key = get_api_key(provider)
    
    return api_key is not None

# --- Language Configuration ---

def get_language() -> str:
    """Gets the configured language, defaulting to APP_DEFAULT_LANGUAGE."""
    config = _read_config()
    lang = config.get("language")
    if not lang or not isinstance(lang, str):
        lang = APP_DEFAULT_LANGUAGE
        # Optionally save the default if it wasn't set
        # set_language(lang) # Let's avoid writing just for a default read for now
    return lang

def set_language(lang_code: str) -> bool:
    """Sets the application language in the config file."""
    # Basic validation: only ensure it's a non-empty string
    if not lang_code or not isinstance(lang_code, str):
         logger.error(f"Attempted to set invalid or empty language code: {lang_code}")
         return False

    try:
        config = _read_config()
        config["language"] = lang_code.lower() # Store lowercase
        _write_config(config)
        logger.info(f"Language set to: {lang_code.lower()}")
        return True
    except Exception as e:
        logger.error(f"Failed to write language setting to config: {e}", exc_info=True)
        return False

def list_available_languages(texts_dir: Path) -> Dict[str, str]:
    """
    Scans the specified directory for language files (*_texts.json)
    and returns a dictionary mapping index number (str) to language code (str).
    Ensures 'en' is present if en_texts.json exists.
    """
    languages = {}
    found_en = False
    idx = 1
    try:
        if not texts_dir.is_dir():
             logger.error(f"Texts directory not found for language discovery: {texts_dir}")
             return {}

        for item in sorted(texts_dir.glob('*_texts.json')):
            if item.is_file():
                lang_code = item.name.split('_texts.json')[0].lower()
                if lang_code: # Ensure we got a code
                    languages[str(idx)] = lang_code
                    if lang_code == 'en': found_en = True
                    idx += 1

        # Ensure 'en' is first if found, otherwise add it if en_texts.json exists
        if found_en and languages.get("1") != "en":
             # Find 'en', remove it, add it back at key "1" shifting others
             current_en_idx = None
             for k, v in languages.items():
                  if v == 'en': current_en_idx = k; break
             if current_en_idx:
                  en_code = languages.pop(current_en_idx)
                  # Rebuild dict with 'en' first
                  shifted_languages = {"1": en_code}
                  shifted_languages.update({str(int(k)+1) : v for k,v in languages.items() if k != '1'}) # Shift others
                  languages = shifted_languages

        elif not found_en and (texts_dir / "en_texts.json").exists():
             # If 'en' wasn't found by glob but the file exists, prepend it
             logger.warning(f"en_texts.json exists but wasn't globbed? Adding 'en' to available languages.")
             shifted_languages = {"1": "en"}
             shifted_languages.update({str(int(k)+1) : v for k,v in languages.items()})
             languages = shifted_languages

    except Exception as e:
        logger.error(f"Error discovering languages in {texts_dir}: {e}", exc_info=True)
        return {} # Return empty on error

    # Final check if english exists at all
    if not languages or "en" not in languages.values():
        logger.error(f"Mandatory 'en_texts.json' seems missing or unreadable in {texts_dir}. Cannot determine languages.")
        return {}

    return languages

def prompt_for_language_selection() -> Optional[str]:
    """Prompts the user to select the application language."""
    # Define path to core agent's text directory
    core_texts_dir = ROOT_DIR / 'ai_shell_agent' / 'texts'
    available_langs = list_available_languages(core_texts_dir)

    if not available_langs:
         console.display_message(get_text("common.labels.error"), get_text("config.lang_select.error_discovery_failed"),
                                console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
         set_language("en") # Ensure default is set
         return "en"

    current_lang = get_language()

    # --- MODIFICATION START ---
    # Build the options list first as a Text object
    from rich.text import Text
    lang_options_text = Text()
    option_lines = []
    for idx, code in available_langs.items():
        marker = get_text("config.lang_select.current_marker") if code == current_lang else ""
        # Use Text.assemble for potential future styling if needed
        option_lines.append(Text.assemble("  ", (idx, console.STYLE_INPUT_OPTION), f": {code}", marker)) # Use input option style for index
    lang_options_text = Text("\n").join(option_lines)

    # Print title and options together
    console.display_message(
        get_text("common.labels.system"),
        get_text("config.lang_select.prompt_title"),
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )
    # Print the assembled options directly
    console.console.print(lang_options_text)
    # --- MODIFICATION END ---

    while True:
        try:
            prompt_msg = get_text("config.lang_select.prompt_input", max_lang=len(available_langs), current_lang=current_lang)
            choice = console.prompt_for_input(prompt_msg).strip()

            if not choice:
                 console.display_message(get_text("common.labels.info"), get_text("config.lang_select.info_kept_current", lang_code=current_lang),
                                        console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                 return current_lang
            elif choice in available_langs:
                selected_code = available_langs[choice]
                if set_language(selected_code):
                    console.display_message(get_text("common.labels.info"), get_text("config.lang_select.info_set_success", lang_code=selected_code),
                                            console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                    return selected_code
                else:
                    # Error handled in set_language, but inform user here too
                    console.display_message(get_text("common.labels.error"), get_text("config.lang_select.error_save_failed"),
                                            console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                    return None # Indicate failure
            else:
                 console.display_message(get_text("common.labels.error"), get_text("config.lang_select.error_invalid_input", max_lang=len(available_langs)),
                                       console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

        except KeyboardInterrupt:
            console.display_message(get_text("common.labels.warning"), get_text("config.lang_select.warn_cancel"),
                                   console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return None # Indicate cancellation/failure
        except Exception as e:
             logger.error(f"Error during language selection: {e}", exc_info=True)
             console.display_message(get_text("common.labels.error"), get_text("config.lang_select.error_generic", error=e),
                                   console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
             return None # Indicate failure

# --- Global Default Toolset Functions ---

DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY = "default_enabled_toolsets"

def get_default_enabled_toolsets() -> List[str]:
    """
    Gets the globally configured default enabled toolset names.
    Returns the default list from settings if not set or invalid in config.
    """
    config = _read_config()
    # Use imported constant as the fallback default
    default_toolsets = config.get(DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY, DEFAULT_ENABLED_TOOLSETS_NAMES)
    if not isinstance(default_toolsets, list):
        logger.warning(f"'{DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY}' in config is not a list. Returning default from settings.")
        # Return a copy of the default list from settings
        return list(DEFAULT_ENABLED_TOOLSETS_NAMES)
    # Optionally validate against registered toolsets? Maybe not here, let caller handle.
    return default_toolsets

def set_default_enabled_toolsets(toolset_names: List[str]) -> None:
    """
    Sets the globally configured default enabled toolset names.
    """
    config = _read_config()
    # Validate input is a list of strings
    if not isinstance(toolset_names, list) or not all(isinstance(name, str) for name in toolset_names):
         logger.error(f"Invalid input to set_default_enabled_toolsets: {toolset_names}. Must be a list of strings.")
         return

    config[DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY] = sorted(list(set(toolset_names))) # Store unique sorted names
    _write_config(config)
    logger.info(f"Global default enabled toolsets set to: {config[DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY]}")

# --- API Key Management (Moved from individual toolsets to be centralized) ---

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
GOOGLE_API_KEY_NAME = "GOOGLE_API_KEY"

def get_api_key(provider: str) -> Optional[str]:
    """Gets the API key for the given provider (openai or google)."""
    from .utils.env import ensure_dotenv_key # Local import to avoid circularity if not already imported

    key_name = ""
    description = ""
    if provider == "openai":
        key_name = OPENAI_API_KEY_NAME
        description = get_text("config.api_key.openai_description")
    elif provider == "google":
        key_name = GOOGLE_API_KEY_NAME
        description = get_text("config.api_key.google_description")
    else:
        logger.error(f"Unknown API key provider: {provider}")
        return None

    # ensure_dotenv_key will use DEFAULT_DOTENV_PATH by default
    api_key = ensure_dotenv_key(key=key_name, description=description) # Removed dotenv_path=DEFAULT_DOTENV_PATH
    return api_key

def ensure_api_keys_are_set() -> None:
    """
    Checks and prompts for API keys for the currently selected model's provider.
    This should be called early in the application startup.
    """
    current_model = get_current_model()
    provider = get_model_provider(current_model)

    if provider == "openai":
        if not get_api_key("openai"):
            # The get_api_key function already handles prompting if the key is missing.
            # If it returns None here, it means the user skipped or an error occurred.
            console.display_message(
                get_text("common.labels.warning"),
                get_text("config.api_key.openai_missing_after_prompt"),
                console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT
            )
            # Potentially raise an error or exit if the key is critical and not set.
    elif provider == "google":
        if not get_api_key("google"):
            console.display_message(
                get_text("common.labels.warning"),
                get_text("config.api_key.google_missing_after_prompt"),
                console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT
            )
    else:
        logger.warning(f"API key check not implemented for provider: {provider} of model {current_model}")

# Call this function when the application starts or when the model changes.
# For example, in your main application entry point:
# ensure_api_keys_are_set()

# --- Language and Translation Settings ---
