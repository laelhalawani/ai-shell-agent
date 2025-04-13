import os
import json
from typing import Dict, Optional, Tuple
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

# Define Aider edit formats with descriptions
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
DEFAULT_AIDER_EDIT_FORMAT = None  # Let AI Code Editor/Model decide by default

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
    Set the model to use for AI interactions, saving to both env var and config file.
    """
    normalized_name = normalize_model_name(model_name)
    
    # Save to environment variable
    os.environ["AI_SHELL_AGENT_MODEL"] = normalized_name
    
    # Save to config file
    config = _read_config()
    config["model"] = normalized_name
    _write_config(config)
    
    # Also persist to .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    # Read existing .env file
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    
    # Update with new model
    env_vars["AI_SHELL_AGENT_MODEL"] = normalized_name
    
    # Write back to .env file
    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
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
    Check if this is the first run of the application.
    
    Returns:
        bool: True if this is the first run, False otherwise
    """
    # Check if model is set in environment or config
    env_model = os.getenv("AI_SHELL_AGENT_MODEL")
    if env_model:
        return False
    
    # Check config file
    config = _read_config()
    if config.get("model"):
        return False
    
    # If we get here, it's the first run
    logger.info("First run detected - model selection required")
    return True

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

def get_aider_edit_format() -> Optional[str]:
    """Gets the configured Aider edit format."""
    config = _read_config()
    # Return None if not set, allowing Aider's defaults to take precedence
    return config.get("aider_edit_format", DEFAULT_AIDER_EDIT_FORMAT)

def set_aider_edit_format(edit_format: Optional[str]) -> None:
    """Sets the Aider edit format in the config."""
    config = _read_config()
    if edit_format is None:
        config.pop("aider_edit_format", None) # Remove key if set to None (use default)
        logger.info("Aider edit format reset to default (model-specific).")
    elif edit_format in AIDER_EDIT_FORMATS:
        config["aider_edit_format"] = edit_format
        logger.info(f"Aider edit format set to: {edit_format}")
    else:
        logger.error(f"Invalid edit format '{edit_format}'. Not setting.")
        return
    _write_config(config)

def prompt_for_edit_mode_selection() -> Optional[str]:
    """Prompts the user to select an AI Code Editor edit format."""
    current_format = get_aider_edit_format()
    print("\nSelect AI Code Editor Format:")
    print("-------------------------")
    i = 0
    valid_choices = {}
    # Option 0: Use Default
    print(f"  0: Default (Let the AI Code Editor choose based on the main model) {'<- Current Setting' if current_format is None else ''}")
    valid_choices['0'] = None

    # List other formats
    format_list = sorted(AIDER_EDIT_FORMATS.keys())
    for idx, fmt in enumerate(format_list, 1):
        description = AIDER_EDIT_FORMATS[fmt]
        marker = " <- Current Setting" if fmt == current_format else ""
        print(f"  {idx}: {fmt}{marker} - {description}")
        valid_choices[str(idx)] = fmt

    while True:
        try:
            choice = input(f"Enter choice (0-{len(format_list)}): ").strip()
            if choice in valid_choices:
                selected_format = valid_choices[choice]
                set_aider_edit_format(selected_format)
                return selected_format
            elif not choice: # Allow empty input to keep current setting
                 print(f"Keeping current setting: {current_format or 'Default'}")
                 return current_format
            else:
                print("Invalid choice. Please try again.")
        except EOFError:
            print("\nSelection cancelled.")
            return current_format # Keep current on Ctrl+D
        except KeyboardInterrupt:
             print("\nSelection cancelled.")
             return current_format # Keep current on Ctrl+C

def _get_aider_model(config_key: str) -> Optional[str]:
    """Helper to get a specific aider model config."""
    config = _read_config()
    return config.get(config_key) # Returns None if not set

def _set_aider_model(config_key: str, model_name: Optional[str]) -> None:
    """Helper to set a specific aider model config."""
    config = _read_config()
    normalized_name = normalize_model_name(model_name) if model_name else None
    if normalized_name:
         config[config_key] = normalized_name
         logger.info(f"Set {config_key} to: {normalized_name}")
    elif config_key in config:
         del config[config_key] # Remove if set to None
         logger.info(f"Reset {config_key} to default.")
    _write_config(config)

def get_aider_main_model() -> Optional[str]:
    """Gets the configured main/architect model for Aider."""
    return _get_aider_model("aider_main_model")

def set_aider_main_model(model_name: Optional[str]) -> None:
    """Sets the main/architect model for Aider."""
    _set_aider_model("aider_main_model", model_name)

def get_aider_editor_model() -> Optional[str]:
    """Gets the configured editor model for Aider."""
    return _get_aider_model("aider_editor_model")

def set_aider_editor_model(model_name: Optional[str]) -> None:
    """Sets the editor model for Aider."""
    _set_aider_model("aider_editor_model", model_name)

def get_aider_weak_model() -> Optional[str]:
    """Gets the configured weak model for Aider."""
    return _get_aider_model("aider_weak_model")

def set_aider_weak_model(model_name: Optional[str]) -> None:
    """Sets the weak model for Aider."""
    _set_aider_model("aider_weak_model", model_name)

def _prompt_for_single_coder_model(role_name: str, current_value: Optional[str]) -> Optional[str]:
    """Helper to prompt for one of the coder models."""
    print(f"\n--- Select AI Code Editor {role_name} Model ---")
    
    print("Available models (same as agent models):")
    all_model_names = sorted(list(set(ALL_MODELS.values()))) # Use ALL_MODELS from config_manager
    for model in all_model_names:
         marker = " <- Current Value" if model == current_value else ""
         print(f"- {model}{marker}")
         # Add aliases maybe?
         aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
         if aliases: print(f"  (aliases: {', '.join(aliases)})")

    prompt_msg = (f"Enter model name for {role_name} role"
                  f" (leave empty to keep '{current_value or 'Default'}',"
                  f" enter 'none' to use default): ")

    while True:
        selected = input(prompt_msg).strip()
        if not selected:
            print(f"Keeping current setting: {current_value or 'Default'}")
            return current_value # Keep current
        elif selected.lower() == 'none':
             print(f"Resetting {role_name} model to default.")
             return None # Use None to signify default/reset
        else:
             normalized_model = normalize_model_name(selected)
             if normalized_model in all_model_names:
                  return normalized_model # Return the selected normalized name
             else:
                  print(f"Error: Unknown model '{selected}'. Please choose from the list or enter 'none'.")


def prompt_for_coder_models_selection() -> None:
    """Runs the multi-step wizard to select AI Code Editor models."""
    print("\n--- Configure AI Code Editor Models ---")
    print("Select the models the AI Code Editor should use for different coding tasks.")
    print("Leave input empty at any step to keep the current setting.")
    print("Enter 'none' to reset a model to its default behavior.")

    current_main = get_aider_main_model()
    selected_main = _prompt_for_single_coder_model("Main/Architect", current_main)
    if selected_main != current_main: # Only set if changed
        set_aider_main_model(selected_main)

    current_editor = get_aider_editor_model()
    selected_editor = _prompt_for_single_coder_model("Editor", current_editor)
    if selected_editor != current_editor:
        set_aider_editor_model(selected_editor)

    current_weak = get_aider_weak_model()
    selected_weak = _prompt_for_single_coder_model("Weak (Commits/Summaries)", current_weak)
    if selected_weak != current_weak:
        set_aider_weak_model(selected_weak)

    print("\nAI Code Editor models configuration updated.")
