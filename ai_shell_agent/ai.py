import os
import json
import argparse
from dotenv import load_dotenv
import sys
from pathlib import Path # Import Path

# Get installation directory (keep this function)
def get_install_dir():
    # Assumes ai.py is in ai_shell_agent/
    return Path(__file__).parent.parent.resolve()

# Load environment variables from .env in the installation directory
env_path = get_install_dir() / '.env'
load_dotenv(env_path)

# Setup logger early
from . import logger
from .paths import ROOT_DIR # Import ROOT_DIR from paths.py
# Import console manager
from .console_manager import get_console_manager
# Import text getter
from .texts import get_text # <--- ADDED IMPORT
# Import localization utility
from .utils.localize import localize_all_texts

# Get console manager instance
console = get_console_manager()

# Config manager imports (keep necessary ones)
from .config_manager import (
    get_current_model, set_model, prompt_for_model_selection,
    ensure_api_key_for_current_model, get_api_key_for_model,
    set_api_key_for_model, # Added this back for CLI --set-api-key support
    get_model_provider, check_if_first_run,
    set_default_enabled_toolsets, # Keep for first run / select tools
    get_language, prompt_for_language_selection, # Added language functions
    # --- ADD TRANSLATION MODEL FUNCTIONS ---
    get_translation_model, set_translation_model, prompt_for_translation_model_selection,
    ensure_api_key_for_translation_model
    # --- END TRANSLATION MODEL FUNCTIONS ---
)

# --- Import state manager functions with updated names ---
from .chat_state_manager import (
    create_or_load_chat,
    save_session, get_current_chat, get_enabled_toolsets, update_enabled_toolsets,
    get_current_chat_title,
    # --- Import path helpers and JSON read needed here ---
    get_toolset_data_path, get_toolset_global_config_path
    # REMOVED: DEFAULT_ENABLED_TOOLSETS_NAMES
)

# Import DEFAULT_ENABLED_TOOLSETS_NAMES from settings
from .settings import DEFAULT_ENABLED_TOOLSETS_NAMES

# --- Import utils for JSON reading ---
from .utils.env import read_dotenv, write_dotenv # Alias if preferred 
from .utils.file_io import read_json, write_json # Keep these imports for file operations

# --- Import chat manager AFTER state manager ---
from .chat_manager import (
    get_chat_titles_list, rename_chat, delete_chat, send_message,
    edit_message, start_temp_chat, flush_temp_chats, execute, list_messages,
    list_toolsets # Keep this import
)
# --- Import Toolset registry ---
from .toolsets.toolsets import get_registered_toolsets, get_toolset_names
# --- Import system prompt ---
from .prompts.prompts import SYSTEM_PROMPT

# --- API Key/Setup Functions ---
def first_time_setup() -> bool:
    """
    Performs first-time setup and returns True if language was changed.
    
    Returns:
        bool: True if language was changed, False otherwise
    """
    # Add language change flag
    language_changed = False
    
    # Add log right inside the function start
    logger.debug("Entering first_time_setup function.")
    is_first = check_if_first_run()
    logger.debug(f"check_if_first_run() returned: {is_first}")
    if is_first:
        # Add log immediately after the 'if'
        logger.debug("First run condition met. Preparing for setup prompts.")
        # Add log right before the console call
        logger.debug("Attempting to display 'Welcome...' message via console manager.")
        try:
            console.display_message(get_text("common.labels.info"), get_text("setup.welcome"),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            logger.debug("Successfully displayed 'Welcome...' message.") # Log success after call
        except Exception as e:
            logger.error(f"Error calling console.display_message for Welcome: {e}", exc_info=True)
            # Decide how to handle this - maybe exit? For now, just log.
            # sys.exit(1) # Or perhaps raise

        # --- Language Selection ---
        logger.debug("Attempting language selection prompt.")
        original_lang = get_language() # Get language before prompting
        selected_lang_code = prompt_for_language_selection()
        
        # --- BEGIN MODIFICATION ---
        from .settings import APP_DEFAULT_LANGUAGE
        effective_lang_code = APP_DEFAULT_LANGUAGE # Default to app setting ('en')

        if selected_lang_code:
            # User made a selection (didn't cancel)
            effective_lang_code = selected_lang_code
            if selected_lang_code != original_lang:
                language_changed = True
                logger.info(f"Language changed during first-time setup: {original_lang} -> {selected_lang_code}")
            else:
                logger.debug(f"User selected the current language: {selected_lang_code}")
        else:
            # User cancelled (prompt returned None)
            console.display_message(get_text("common.labels.warning"), get_text("config.lang_select.warn_cancel_default"),
                                    console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            # Keep effective_lang_code as APP_DEFAULT_LANGUAGE
            logger.warning(f"Language selection cancelled during first run. Defaulting to {APP_DEFAULT_LANGUAGE}.")

        # Explicitly set the language in the config *before* proceeding
        from .config_manager import set_language # Import set_language here
        if not set_language(effective_lang_code):
             logger.error(f"Failed to set language '{effective_lang_code}' during first time setup!")
             # Decide how to handle this - maybe warn the user or exit?
             # For now, log the error and continue, hoping the default is usable.
        else:
             logger.debug(f"Ensured language '{effective_lang_code}' is set in config during first time setup.")

        logger.debug(f"Language selection block finished. Effective language: {effective_lang_code}. Language changed flag: {language_changed}")
        # --- END MODIFICATION ---

        logger.debug("Attempting to call prompt_for_model_selection.")
        selected_model = prompt_for_model_selection() # This now uses console_manager internally
        logger.debug(f"prompt_for_model_selection returned: {selected_model}")

        if selected_model:
            set_model(selected_model) # set_model logs internally
        else:
            # Use console manager for critical error
            logger.critical("No model selected during first run. Exiting.")
            console.display_message(get_text("common.labels.error"), get_text("setup.errors.no_model_selected"),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            sys.exit(1)

        logger.debug("Attempting to call ensure_api_key.")
        if not ensure_api_key(): # ensure_api_key uses console_manager internally
            logger.critical("API Key not provided. Exiting.")
            console.display_message(get_text("common.labels.error"), get_text("setup.errors.api_key_missing"),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            sys.exit(1)
        logger.debug("ensure_api_key successful.")

        # Add prompt for initial toolsets
        logger.debug("Attempting to call prompt_for_initial_toolsets.")
        prompt_for_initial_toolsets() # This uses console_manager internally
        logger.debug("prompt_for_initial_toolsets finished.")

        console.display_message(get_text("common.labels.info"), get_text("setup.complete"),
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        logger.debug("Displayed 'First-time setup complete.'")
    else:
        logger.debug("Not the first run, skipping setup.")

    return language_changed

def ensure_api_key() -> bool:
    # Only ensures key for the main agent model
    return ensure_api_key_for_current_model() # This still uses config_manager logic for agent key

def prompt_for_initial_toolsets():
    """Prompts user to select default enabled toolsets during first run."""
    # Combine introductory messages
    intro_text = get_text("setup.toolsets.prompt_intro")
    console.display_message(get_text("common.labels.system"), intro_text.strip(), console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT) # MODIFIED LABEL

    from .toolsets.toolsets import get_registered_toolsets
    from .config_manager import set_default_enabled_toolsets
    from .settings import DEFAULT_ENABLED_TOOLSETS_NAMES

    all_toolsets = get_registered_toolsets()
    if not all_toolsets:
        console.display_message(get_text("common.labels.warning"), get_text("setup.toolsets.warn_none_found"), # MODIFIED LABEL
                              console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        set_default_enabled_toolsets([])
        return

    console.display_message(get_text("common.labels.system"), get_text("setup.toolsets.available_title"), # MODIFIED LABEL
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    options = {}
    idx = 1
    from rich.text import Text # Import Text here
    toolset_lines = []
    for ts_id, meta in sorted(all_toolsets.items(), key=lambda item: item[1].name):
        # Create Text object for highlighting
        line_text = Text.assemble(
            "  ",
            (f"{idx}", console.STYLE_INPUT_OPTION), # Highlight number
            f": {meta.name.ljust(15)} - {meta.description}"
            # Apply base system style implicitly or explicitly if needed
        )
        toolset_lines.append(line_text)
        options[str(idx)] = meta.name
        idx += 1
    # Print all toolset lines at once
    for line in toolset_lines:
        console.console.print(line) # Use console.print directly for Text objects

    # Combine prompt instructions
    default_toolsets_str = ", ".join(DEFAULT_ENABLED_TOOLSETS_NAMES) if DEFAULT_ENABLED_TOOLSETS_NAMES else "None"
    prompt_instructions = get_text("setup.toolsets.prompt_instructions", default_toolsets_str=default_toolsets_str)
    console.display_message(get_text("common.labels.system"), prompt_instructions.strip(), # MODIFIED LABEL
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    while True:
        try:
            # Use console manager for prompting
            choice_str = console.prompt_for_input(get_text("common.prompt_input_marker")).strip()
            selected_names = []
            if not choice_str:
                # Use DEFAULT_ENABLED_TOOLSETS_NAMES when no selection is made
                selected_names = list(DEFAULT_ENABLED_TOOLSETS_NAMES)
                toolset_list_str = ', '.join(selected_names) or 'None'
                console.display_message(get_text("common.labels.info"), get_text("setup.toolsets.info_using_defaults", toolset_list=toolset_list_str), # MODIFIED LABEL
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            elif choice_str.lower() == 'none':
                pass # selected_names remains empty
            else:
                selected_indices = {c.strip() for c in choice_str.split(',') if c.strip()}
                valid_selection = True
                for index in selected_indices:
                    if index in options:
                        selected_names.append(options[index])
                    else:
                        # Use console manager for error
                        console.display_message(get_text("common.labels.error"), get_text("setup.toolsets.error_invalid_selection", index=index, max_idx=idx-1), # MODIFIED LABEL
                                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                        valid_selection = False
                        break
                if not valid_selection: continue # Ask again

            # Remove duplicates and sort
            final_selection = sorted(list(set(selected_names)))
            final_selection_str = ', '.join(final_selection) or 'None'

            # Save as global default
            set_default_enabled_toolsets(final_selection)
            # Use console manager for confirmation with updated wording
            console.display_message(get_text("common.labels.info"), get_text("setup.toolsets.info_selection_saved", toolset_list=final_selection_str), # MODIFIED LABEL
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            return

        except (EOFError, KeyboardInterrupt):
            # console.prompt_for_input handles the KeyboardInterrupt print
            default_toolsets_str = ", ".join(DEFAULT_ENABLED_TOOLSETS_NAMES) if DEFAULT_ENABLED_TOOLSETS_NAMES else "None"
            console.display_message(get_text("common.labels.warning"), get_text("setup.toolsets.warn_cancel", default_toolsets_str=default_toolsets_str), # MODIFIED LABEL
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            set_default_enabled_toolsets(list(DEFAULT_ENABLED_TOOLSETS_NAMES))
            return
        except Exception as e: # Catch other potential errors
            console.display_message(get_text("common.labels.error"), get_text("setup.toolsets.error_generic", error=e), # MODIFIED LABEL
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            logger.error(f"Error in toolset prompt: {e}", exc_info=True)
            return # Exit for now

# --- Toolset Selection Command (RENAMED and MODIFIED) ---
def select_enabled_toolsets(): # Renamed from select_tools_for_chat
    """Interactive prompt for selecting enabled toolsets (chat-specific or global default)."""
    chat_id = get_current_chat() # Use chat_id

    all_toolsets = get_registered_toolsets() # Dict[id, ToolsetMetadata]
    if not all_toolsets:
        console.display_message(get_text("common.labels.warning"), get_text("select_tools.warn_none_found"), # MODIFIED LABEL
                              console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        return

    # --- Determine Context and Get Current Settings ---
    is_global_context = chat_id is None
    if is_global_context:
        from .config_manager import get_default_enabled_toolsets # Import here
        chat_title = get_text("select_tools.context_global")
        context_explanation = get_text("select_tools.explanation_global")
        current_enabled_names = get_default_enabled_toolsets()
        save_confirmation_message = get_text("select_tools.info_selection_saved") # key only
        keep_message = get_text("select_tools.keep_message_global")
        update_global_default = True
        update_chat_specific = False
    else:
        chat_title_from_state = get_current_chat_title()
        if not chat_title_from_state:
            console.display_message(get_text("common.labels.error"), get_text("select_tools.error_chat_title_missing", chat_id=chat_id), # MODIFIED LABEL
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
        chat_title = get_text("select_tools.context_chat", chat_title=chat_title_from_state)
        context_explanation = get_text("select_tools.explanation_chat")
        current_enabled_names = get_enabled_toolsets(chat_id)
        save_confirmation_message = get_text("select_tools.info_selection_saved") # key only
        keep_message = get_text("select_tools.keep_message_chat")
        update_global_default = True
        update_chat_specific = True

    # --- Display Header ---
    context_header_str = get_text("select_tools.context_global") if is_global_context else get_text("select_tools.context_chat", chat_title=get_current_chat_title())
    intro_text = get_text("select_tools.prompt_header", context=context_header_str, explanation=context_explanation)
    console.display_message(get_text("common.labels.system"), intro_text.strip(), # MODIFIED LABEL
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    # --- Display Toolset Options ---
    console.display_message(get_text("common.labels.system"), get_text("select_tools.available_title"), # MODIFIED LABEL
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    options = {}
    idx = 1
    from rich.text import Text # Import Text here
    toolset_lines = []
    enabled_marker = get_text("select_tools.marker_enabled")
    disabled_marker = get_text("select_tools.marker_disabled")
    # Sort toolsets by name for consistent display
    for ts_id, meta in sorted(all_toolsets.items(), key=lambda item: item[1].name):
        marker = enabled_marker if meta.name in current_enabled_names else disabled_marker
        line_text = Text.assemble(
            "  ",
            (f"{idx}", console.STYLE_INPUT_OPTION), # Highlight number
            f": {meta.name.ljust(25)} {marker} - {meta.description}"
        )
        toolset_lines.append(line_text)
        options[str(idx)] = meta.name
        idx += 1
    for line in toolset_lines:
        console.console.print(line)

    # --- Prompt Instructions ---
    prompt_instructions = get_text("select_tools.prompt_instructions")
    console.display_message(get_text("common.labels.system"), prompt_instructions.strip(), # MODIFIED LABEL
                          console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    # --- Prompt Loop ---
    while True:
        try:
            choice_str = console.prompt_for_input(get_text("common.prompt_input_marker")).strip()
            if not choice_str:
                toolset_list_str = ', '.join(sorted(current_enabled_names)) or 'None'
                console.display_message(get_text("common.labels.info"), get_text("select_tools.info_kept_settings", keep_message=keep_message, toolset_list=toolset_list_str), # MODIFIED LABEL
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                return

            if choice_str.lower() == 'none':
                new_enabled_list_names = []
            else:
                selected_indices = {c.strip() for c in choice_str.split(',') if c.strip()}
                new_enabled_list_names = []
                valid_selection = True
                for index in selected_indices:
                    if index in options:
                        new_enabled_list_names.append(options[index])
                    else:
                        console.display_message(get_text("common.labels.error"), get_text("select_tools.error_invalid_selection", index=index, max_idx=idx-1), # MODIFIED LABEL
                                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                        valid_selection = False
                        break
                if not valid_selection: continue # Ask again

            # Remove duplicates and sort
            final_selection = sorted(list(set(new_enabled_list_names)))
            final_selection_str = ', '.join(final_selection) or 'None'

            # --- Update State ---
            if update_chat_specific:
                update_enabled_toolsets(chat_id, final_selection)
                console.display_message(get_text("common.labels.info"), get_text("select_tools.info_selection_saved", toolset_list=final_selection_str), # MODIFIED LABEL
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)

            if update_global_default:
                from .config_manager import set_default_enabled_toolsets # Import here
                set_default_enabled_toolsets(final_selection)
                if not is_global_context:
                    console.display_message(get_text("common.labels.info"), get_text("select_tools.info_global_updated"), # MODIFIED LABEL
                                          console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
                elif update_chat_specific == False:
                     console.display_message(get_text("common.labels.info"), get_text("select_tools.info_selection_saved", toolset_list=final_selection_str), # MODIFIED LABEL
                                          console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)

            if update_chat_specific:
                 console.display_message(get_text("common.labels.info"), get_text("select_tools.info_apply_next"), # MODIFIED LABEL
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            return

        except (EOFError, KeyboardInterrupt):
            console.display_message(get_text("common.labels.warning"), get_text("select_tools.warn_cancel"), # MODIFIED LABEL
                                  console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            return

# --- Toolset Configuration Command (MODIFIED) ---
def configure_toolset_cli(toolset_name: str):
    """Handles the --configure-toolset command for global or chat-specific settings."""
    chat_id = get_current_chat()
    is_global_context = chat_id is None

    # --- Find Toolset Metadata (Common logic) ---
    from .toolsets.toolsets import get_registered_toolsets, get_toolset_names
    registered_toolsets = get_registered_toolsets()
    target_toolset_id = None
    target_metadata = None
    for ts_id, meta in registered_toolsets.items():
        if meta.name.lower() == toolset_name.lower():
            target_toolset_id = ts_id
            target_metadata = meta
            break

    if not target_metadata or not target_toolset_id:
        console.display_message(get_text("common.labels.error"), get_text("configure_toolset.error_not_found", toolset_name=toolset_name),
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        console.display_message(get_text("common.labels.system"), get_text("configure_toolset.info_available", toolset_list=", ".join(get_toolset_names())),
                              console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        return

    if not target_metadata.configure_func:
        console.display_message(get_text("common.labels.info"), get_text("configure_toolset.info_not_configurable", toolset_name=target_metadata.name),
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return

    # --- Get Paths ---
    dotenv_path = ROOT_DIR / '.env'
    global_config_path = get_toolset_global_config_path(target_toolset_id)

    # --- Determine context and load current config ---
    local_config_path = None
    current_config_for_prompting = None

    if is_global_context:
        current_config_for_prompting = read_json(global_config_path, default_value=None)
    else: # Chat context
        chat_title_from_state = get_current_chat_title()
        if not chat_title_from_state:
            console.display_message(get_text("common.labels.error"), get_text("configure_toolset.error_chat_title_missing", chat_id=chat_id),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
        local_config_path = get_toolset_data_path(chat_id, target_toolset_id)
        current_config_for_prompting = read_json(local_config_path, default_value=None)

    # --- Execute Configuration ---
    try:
        # Pass all required paths and config to the toolset's function
        final_config = target_metadata.configure_func(
            global_config_path,
            local_config_path,
            dotenv_path,
            current_config_for_prompting
        )
        # The configure_func itself is now responsible for printing context headers
    except (EOFError, KeyboardInterrupt):
        logger.warning(f"Configuration cancelled for {toolset_name} by user.")
        console.display_message(get_text("common.labels.warning"), get_text("configure_toolset.warn_cancel"),
                              console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
    except Exception as e:
        logger.error(f"Error running configuration for {toolset_name}: {e}", exc_info=True)
        console.display_message(get_text("common.labels.error"), get_text("configure_toolset.error_generic", error=e),
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

# --- Main CLI Execution Logic ---
def main():
    # Capture language at the start, *after* initial imports ensure texts are loaded once
    original_language = get_language()
    logger.debug(f"Initial language detected: {original_language}")

    env_path = os.path.join(get_install_dir(), '.env'); load_dotenv(env_path)
    parser = argparse.ArgumentParser(description=get_text("cli.parser.description"), formatter_class=argparse.RawTextHelpFormatter)
    # --- Argument Groups ---
    model_group = parser.add_argument_group(get_text("cli.groups.model"))
    chat_group = parser.add_argument_group(get_text("cli.groups.chat"))
    tool_group = parser.add_argument_group(get_text("cli.groups.toolset"))
    msg_group = parser.add_argument_group(get_text("cli.groups.message"))
    util_group = parser.add_argument_group(get_text("cli.groups.util"))

    # --- Arguments (using get_text for help) ---
    model_group.add_argument("-llm", "--model", help=get_text("cli.args.model.help"))
    model_group.add_argument("--select-model", action="store_true", help=get_text("cli.args.select_model.help"))
    model_group.add_argument("-k", "--set-api-key", nargs="?", const=True, metavar="API_KEY",
                             help=get_text("cli.args.set_api_key.help"))
    model_group.add_argument("--select-language", action="store_true", help=get_text("cli.args.select_language.help"))
    
    # Add translation model selection argument
    model_group.add_argument("--select-translation-model", action="store_true", 
                             help=get_text("cli.args.select_translation_model.help"))
    
    # --- MODIFICATION START ---
    util_group.add_argument("--localize", metavar="LANGUAGE_NAME", help=get_text("cli.args.localize.help"))
    # --- MODIFICATION END ---

    chat_group.add_argument("-c", "--chat", metavar="TITLE", help=get_text("cli.args.chat.help"))
    chat_group.add_argument("-lc", "--load-chat", metavar="TITLE", help=get_text("cli.args.load_chat.help"))
    chat_group.add_argument("-lsc", "--list-chats", action="store_true", help=get_text("cli.args.list_chats.help"))
    chat_group.add_argument("-rnc", "--rename-chat", nargs=2, metavar=("OLD", "NEW"), help=get_text("cli.args.rename_chat.help"))
    chat_group.add_argument("-delc", "--delete-chat", metavar="TITLE", help=get_text("cli.args.delete_chat.help"))
    chat_group.add_argument("--temp-flush", action="store_true", help=get_text("cli.args.temp_flush.help"))
    chat_group.add_argument("-ct", "--current-chat-title", action="store_true", help=get_text("cli.args.current_chat_title.help"))

    tool_group.add_argument("--select-tools", action="store_true",
                           help=get_text("cli.args.select_tools.help"))
    tool_group.add_argument("--list-toolsets", action="store_true",
                           help=get_text("cli.args.list_toolsets.help"))
    tool_group.add_argument("--configure-toolset", metavar="TOOLSET_NAME",
                           help=get_text("cli.args.configure_toolset.help"))

    msg_group.add_argument("-m", "--send-message", metavar='"MSG"', help=get_text("cli.args.send_message.help"))
    msg_group.add_argument("-tc", "--temp-chat", metavar='"MSG"', help=get_text("cli.args.temp_chat.help"))
    msg_group.add_argument("-e", "--edit", nargs="+", metavar="IDX|last \"MSG\"", help=get_text("cli.args.edit.help"))
    msg_group.add_argument("-lsm", "--list-messages", action="store_true", help=get_text("cli.args.list_messages.help"))
    msg_group.add_argument("-x", "--execute", metavar='"CMD"', help=get_text("cli.args.execute.help"))

    parser.add_argument("message", nargs="?", help=get_text("cli.args.message.help"))
    args = parser.parse_args()

    # --- Execution Order ---
    
    # --- Handle translation model selection ---
    if args.select_translation_model:
        selected_trans_model = prompt_for_translation_model_selection()
        if selected_trans_model:
            # Check if the selected model is different from the current one
            current_trans_model = get_translation_model()
            if selected_trans_model != current_trans_model:
                set_translation_model(selected_trans_model)
                console.display_message(get_text("common.labels.info"), 
                                      get_text("cli.info.translation_model_set", model_name=selected_trans_model),
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            else:
                console.display_message(get_text("common.labels.info"),
                                      get_text("cli.info.translation_model_unchanged", model_name=current_trans_model),
                                      console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        # If selection was cancelled, the prompt function handles the message
        return
    
    # --- Handle localization ---
    if args.localize:
        target_lang = args.localize
        
        # 1. First Time Setup (ensure config exists, etc.)
        first_time_setup()
        
        # 2. Prompt for Translation Model if needed
        # The localization process will get the translation model automatically

        # 3. Ensure API Key for translation model
        if not ensure_api_key_for_translation_model():
            logger.critical("API Key missing for translation model. Cannot proceed with localization.")
            console.display_message(get_text("common.labels.error"), get_text("cli.errors.api_key_missing"),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            sys.exit(1)
            
        # 4. Run Localization
        localize_all_texts(target_lang)
        return # Exit after localization
    
    # --- Handle language selection (now before first-time setup) ---
    if args.select_language:
        # Store current language setting
        original_lang = get_language()
        # Prompt for language selection
        selected_lang = prompt_for_language_selection()
        
        # Check if language was changed
        if selected_lang and selected_lang != original_lang:
            # Language was changed successfully
            console.display_message(get_text("common.labels.info"), 
                                  get_text("cli.info.language_set", lang_code=selected_lang),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            console.display_message(get_text("common.labels.info"), 
                                  get_text("cli.info.language_restart_needed"),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            logger.info(f"Language changed: {original_lang} -> {selected_lang}. Exiting for restart.")
            sys.exit(0) # Exit cleanly after language change
        elif selected_lang:
            # Language selected but it's the same as the original
            console.display_message(get_text("common.labels.info"),
                                  get_text("cli.info.language_unchanged", lang_code=selected_lang),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        # else: Cancellation message handled within prompt function
        return # Exit after language selection attempt

    # --- Original Execution Order ---
    # 1. Model Selection
    if args.model:
        set_model(args.model)
        ensure_api_key()
        current_model = get_current_model()
        console.display_message(get_text("common.labels.info"), get_text("cli.info.model_set", model_name=current_model),
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.select_model:
        m = prompt_for_model_selection()
        current_model = get_current_model()
        if m and m != current_model:
            set_model(m)
            ensure_api_key()
            console.display_message(get_text("common.labels.info"), get_text("cli.info.model_set", model_name=get_current_model()),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        elif m:
            console.display_message(get_text("common.labels.info"), get_text("cli.info.model_unchanged", model_name=current_model),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        else:
            console.display_message(get_text("common.labels.info"), get_text("cli.info.model_select_cancel"),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return

    # 2. First Time Setup (modified to check for language changes)
    language_changed_during_setup = first_time_setup()
    if language_changed_during_setup:
        console.display_message(get_text("common.labels.info"), 
                              get_text("cli.info.language_restart_needed_setup"),
                              console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        logger.info("Language changed during first-time setup. Exiting for restart.")
        sys.exit(0) # Exit cleanly after language change during setup
                              
    # 3. API Key Management
    if args.set_api_key:
        set_api_key_for_model(get_current_model(), args.set_api_key if isinstance(args.set_api_key, str) else None)
        return

    # 4. Ensure API Key (Exit if missing)
    if not ensure_api_key(): # Only checks agent key now
        logger.critical("Main agent API Key missing.")
        console.display_message(get_text("common.labels.error"), get_text("cli.errors.api_key_missing"),
                              console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        sys.exit(1)

    # 5. Toolset Selection / Listing / Configuration (BEFORE chat loading)
    # These can now run without an active chat
    if args.select_tools:
        select_enabled_toolsets() # MODIFIED: Call the renamed function
        return
    if args.list_toolsets:
        from .chat_manager import list_toolsets # Keep import location
        list_toolsets() # This function now handles global context
        return
    if args.configure_toolset:
        configure_toolset_cli(args.configure_toolset) # This function will be modified next
        return

    # 7. Direct Command Execution
    if args.execute:
        execute(args.execute)
        return

    # 8. Chat Management
    if args.chat:
        chat_id = create_or_load_chat(args.chat) # Use new function name
        if chat_id: console.display_message(get_text("common.labels.info"), get_text("cli.info.chat_switched", chat_title=args.chat),
                                         console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.load_chat:
        chat_id = create_or_load_chat(args.load_chat) # Use new function name
        if chat_id: console.display_message(get_text("common.labels.info"), get_text("cli.info.chat_loaded", chat_title=args.load_chat),
                                         console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.current_chat_title:
        title = get_current_chat_title() # Use new function name
        if title:
            console.display_message(get_text("common.labels.info"), get_text("cli.info.current_chat_is", chat_title=title),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        else:
            console.display_message(get_text("common.labels.info"), get_text("cli.info.no_active_chat"),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return
    if args.list_chats:
        get_chat_titles_list()
        return
    if args.rename_chat:
        rename_chat(*args.rename_chat)
        return
    if args.delete_chat:
        delete_chat(args.delete_chat)
        return
    if args.temp_flush:
        flush_temp_chats()
        return

    # --- Operations requiring active chat ---
    active_chat_id = get_current_chat() # Use chat_id consistently

    # 9. Messaging / History
    if args.list_messages:
        if not active_chat_id:
            console.display_message(get_text("common.labels.error"), get_text("cli.errors.no_active_chat_for_list"),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
        list_messages() # chat_manager will get chat_id using get_current_chat()
        return
    if args.edit:
        if not active_chat_id:
            console.display_message(get_text("common.labels.error"), get_text("cli.errors.no_active_chat_for_edit"),
                                  console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
        idx_str, msg_parts = args.edit[0], args.edit[1:]
        new_msg = " ".join(msg_parts)
        idx = None
        if idx_str.lower() != "last":
            try:
                idx = int(idx_str)
            except ValueError:
                console.display_message(get_text("common.labels.error"), get_text("cli.errors.edit_invalid_index", index=idx_str),
                                      console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                return
        edit_message(idx, new_msg) # chat_manager will get chat_id using get_current_chat()
        return

    # 10. Sending Messages (Default actions)
    msg_to_send = args.send_message or args.message # Prioritize -m
    if msg_to_send:
        if not active_chat_id and not args.temp_chat:
            # If no active chat and not explicitly temp
            console.display_message(get_text("common.labels.info"), get_text("cli.info.starting_temp_chat"),
                                  console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            start_temp_chat(msg_to_send)
        elif active_chat_id:
            send_message(msg_to_send) # chat_manager will get chat_id using get_current_chat()
        # If args.temp_chat is set, it's handled below
        return
    if args.temp_chat: # Handles -tc explicitly
        start_temp_chat(args.temp_chat)
        return

    # 11. No arguments provided
    parser.print_help()

if __name__ == "__main__":
    main()
