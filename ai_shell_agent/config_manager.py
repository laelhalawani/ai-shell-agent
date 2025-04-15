import os
import json
from typing import Dict, Optional, Tuple, List
from . import logger

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

DEFAULT_MODEL = "gpt-4o-mini"

# Define Aider edit formats with descriptions (added back for toolset configuration)
AIDER_EDIT_FORMATS = {
    "whole": "LLM sends back the entire file content.",
    "diff": "LLM sends back search/replace blocks.",
    "diff-fenced": "LLM sends back search/replace blocks within a fenced code block (good for Gemini).",
    "udiff": "LLM sends back simplified unified diff format (good for GPT-4 Turbo).",
    "architect": "High-level planning by main model, detailed edits by editor model.",
    # editor-* formats are typically set automatically with architect mode or --editor-model
    # but allow explicit selection if needed.
    "editor-diff": "Streamlined diff format for editor models.",
    "editor-whole": "Streamlined whole format for editor models.",
}

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
    
    # If neither exists, use default and initialize it
    if not model:
        model = DEFAULT_MODEL
        set_model(model)
    
    return model

def set_model(model_name: str) -> None:
    """
    Set the model to use for AI interactions, saving to environment variable and config file.
    No longer saves to .env file - toolsets handle their own secrets.
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
    """
    Prompt the user for model selection, showing available options with aliases.
    
    Returns:
        str: The selected normalized model name
    """
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
    
    print("Available models:")
    print("OpenAI:")
    for model in set(OPENAI_MODELS.values()):
        aliases = model_aliases.get(model, [])
        # Only show aliases if they exist and are different from the model name
        alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
        marker = " <- Current Model" if model == current_model else ""
        print(f"- {model}{alias_text}{marker}")
    
    print("Google:")
    for model in set(GOOGLE_MODELS.values()):
        aliases = model_aliases.get(model, [])
        # Only show aliases if they exist and are different from the model name
        alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
        marker = " <- Current Model" if model == current_model else ""
        print(f"- {model}{alias_text}{marker}")
    
    selected_model = input(f"\nPlease input the model you want to use, or leave empty to keep using the current model {current_model}.\n> ").strip()
    
    if not selected_model:
        return current_model
    
    normalized_model = normalize_model_name(selected_model)
    if normalized_model not in set(OPENAI_MODELS.values()) and normalized_model not in set(GOOGLE_MODELS.values()):
        logger.warning(f"Unknown model: {selected_model}. Using default model: {current_model}")
        return current_model
    
    return normalized_model

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
    Prompt for and save the appropriate API key for the selected model.
    
    Args:
        model_name: The name of the model
        api_key: The API key to set, or None to prompt the user
    """
    provider = get_model_provider(model_name)
    env_var_name = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"
    provider_name = "OpenAI" if provider == "openai" else "Google"
    
    api_key_link = "https://platform.openai.com/api-keys" if provider == "openai" else "https://aistudio.google.com/app/apikey"
    
    if not api_key:
        print(f"Please enter your {provider_name} API key.")
        print(f"You can get it from: {api_key_link}")
        api_key = input(f"Enter {provider_name} API key: ").strip()
    
    if not api_key:
        logger.warning(f"No {provider_name} API key provided. Aborting.")
        return
    
    os.environ[env_var_name] = api_key
    
    # Save to .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    # Read existing .env file
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    
    # Update or add the API key
    env_vars[env_var_name] = api_key
    
    # Write back to .env file
    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"{provider_name} API key saved successfully to .env")

def ensure_api_key_for_current_model() -> bool:
    """
    Ensure that the API key for the current model is set.
    If not, prompt the user to enter it.
    
    Returns:
        bool: True if the API key is set, False otherwise
    """
    current_model = get_current_model()
    api_key, env_var_name = get_api_key_for_model(current_model)
    
    if not api_key:
        provider = get_model_provider(current_model)
        provider_name = "OpenAI" if provider == "openai" else "Google"
        logger.warning(f"{provider_name} API key not found. Please enter your API key.")
        set_api_key_for_model(current_model)
        
        # Check again if the API key is set
        api_key, _ = get_api_key_for_model(current_model)
        if not api_key:
            return False
    
    return True

# --- Global Default Toolset Functions ---

DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY = "default_enabled_toolsets"

def get_default_enabled_toolsets() -> List[str]:
    """
    Gets the globally configured default enabled toolset names.
    Returns an empty list if not set.
    """
    config = _read_config()
    # Ensure it returns a list, default to empty list if key missing or not a list
    default_toolsets = config.get(DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY, [])
    if not isinstance(default_toolsets, list):
        logger.warning(f"'{DEFAULT_ENABLED_TOOLSETS_CONFIG_KEY}' in config is not a list. Returning empty list.")
        return []
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

# All Aider-specific functions have been removed:
# - get_aider_edit_format
# - set_aider_edit_format
# - prompt_for_edit_mode_selection
# - _get_aider_model
# - _set_aider_model
# - get_aider_main_model
# - set_aider_main_model
# - get_aider_editor_model
# - set_aider_editor_model
# - get_aider_weak_model
# - set_aider_weak_model
# - _prompt_for_single_coder_model
# - prompt_for_coder_models_selection
