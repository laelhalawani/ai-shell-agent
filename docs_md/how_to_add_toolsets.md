# How to Add a New Toolset to AI Shell Agent

The `ai-shell-agent` uses a modular architecture where capabilities are grouped into "Toolsets." Each toolset is self-contained within its own directory, defining its tools, configuration, required secrets, and user-facing text. This guide explains how to create and integrate a new toolset based on the current architecture.

## 1. Create the Toolset Directory Structure

1.  Navigate to the `ai_shell_agent/toolsets/` directory.
2.  Create a new subdirectory for your toolset. The name of this directory becomes the unique **Toolset ID** (e.g., `web_search`, `database_query`, `code_analyzer`). Choose a concise, lowercase name using underscores if needed.
    *   Example: `ai_shell_agent/toolsets/my_new_toolset/`
3.  Inside this directory, create the following recommended file structure. Files marked "(Optional)" can be omitted if not needed for your specific toolset.

    ```
    ai_shell_agent/
    └── toolsets/
        └── my_new_toolset/
            ├── __init__.py             # Package marker (REQUIRED)
            ├── toolset.py              # Core logic: tools, metadata, config (REQUIRED)
            |
            ├── tools/                  # Directory for tool logic (Recommended Structure)
            │   ├── __init__.py         # Package marker
            │   ├── tools.py            # Tool class definitions
            │   └── schemas.py          # Pydantic input schemas for tools
            |
            ├── texts/                  # Directory for UI texts (REQUIRED)
            │   └── en_texts.json       # English UI texts (REQUIRED)
            │   └── <lang>_texts.json   # (Optional) Other language UI texts
            └── texts.py                # Loads UI texts for the toolset (REQUIRED)
            |
            ├── prompts.py              # (Optional) AI context prompt string(s)
            |
            ├── settings.py             # (Optional) Loads default settings constants
            ├── default_settings/       # (Optional) Directory for default settings JSON
            │   └── default_settings.json # (Optional) Default config values JSON
            |
            ├── state.py                # (Optional) Manages toolset-specific persistent state
            └── integration/            # (Optional) For complex external library integrations
                └── __init__.py
                └── ... (integration logic)
    ```

## 2. Implement Core Files

Implement the essential files for your toolset.

**a) `__init__.py` (REQUIRED)**

*   Marks the directory as a Python package.
*   Crucially, it **must** import the `toolset` module to ensure your toolset is discovered and registered when the application starts.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/__init__.py
    """
    My New Toolset package initialization.
    Provides tools for X, Y, and Z.
    """
    from . import toolset # Import toolset to ensure registration runs on discovery
    ```

**b) `texts/en_texts.json` (REQUIRED)**

*   Stores all user-facing strings for your toolset in English (UI elements, descriptions, prompts, error messages, etc.) in a nested JSON format.
*   Use clear, hierarchical keys (e.g., `toolset`, `config`, `schemas`, `tools`).
*   **Crucially, use this for `description` fields of tools and schemas, and for any text returned or displayed to the user by tools or the configuration function.**
*   **Do NOT define the tool `name` here.** The `name` used by the LLM must be hardcoded in `tools.py`.

    ```json
    // ai_shell_agent/toolsets/my_new_toolset/texts/en_texts.json
    {
      "toolset": {
        "name": "My New Toolset",
        "description": "Provides fantastic new capabilities."
      },
      "config": {
        "header": "--- Configure My New Toolset ---",
        "prompt_api_url": "Enter the API URL for My New Toolset",
        "info_saved": "My New Toolset configuration saved.",
        "warn_invalid_url": "Invalid URL entered. Please try again.",
        "error_save_failed": "Failed to save My New Toolset configuration."
        // ... other config-related texts
      },
      "secrets": { // Example: Add a key for secret descriptions
        "api_key_desc": "Your unique API key obtained from my-new-toolset.com/keys"
      },
      "schemas": {
        "perform_action": {
          "item_id_desc": "The ID of the item to process.",
          "parameters_desc": "Optional JSON string of parameters."
        },
        "risky_operation": {
           "target_desc": "The target resource for the risky operation.",
           "operation_summary_desc": "A brief summary of the risky operation (shown in confirmation)."
        }
        // ... other schema descriptions
      },
      "tools": {
        "usage_guide": {
          // "name" is NOT needed/used here
          "description": "Displays usage instructions for My New Toolset."
        },
        "perform_action": {
          // "name" is NOT needed/used here
          "description": "Performs the primary action of this toolset.",
          "success": "Action completed successfully on item {item_id}.",
          "error_generic": "Failed to perform action on item {item_id}: {error}"
        },
        "risky_operation": {
          // "name" is NOT needed/used here
          "description": "Performs a risky operation requiring user confirmation.",
          "confirm_prompt": "AI wants to perform action '{tool_name}'.\nSummary: {summary}\nPerform operation on target '{target}'?",
          "confirm_suffix_yes_no": "(confirm: yes/no) > ",
          "info_rejected": "Risky operation rejected by user.",
          "success_confirm": "Successfully performed confirmed risky operation on '{target}'.",
          "error_generic": "Failed risky operation on '{target}': {error}"
        }
        // ... texts for other tools
      }
    }
    ```

**c) `texts.py` (REQUIRED)**

*   Loads the text strings from the `texts/` directory based on the application's selected language (via `utils.config_reader`), falling back to English (`en_texts.json`).
*   Provides a `get_text(key_path, **kwargs)` function specific to this toolset for accessing localized strings.
*   You can copy the implementation from an existing toolset's `texts.py` (e.g., `files/texts.py`) and **update the `_toolset_id` variable** to match your toolset's directory name.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/texts.py
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

    _toolset_id = 'my_new_toolset' # <-- SET YOUR TOOLSET ID
    _texts_dir = Path(__file__).parent / 'texts'
    _texts_data = {}

    def _load_texts():
        # ... (copy implementation from files/texts.py or similar) ...
        # Ensure it uses _toolset_id in log messages for clarity
        global _texts_data
        lang = get_config_value("language", default=APP_DEFAULT_LANGUAGE)
        if not lang or not isinstance(lang, str): lang = APP_DEFAULT_LANGUAGE
        en_file = _texts_dir / "en_texts.json"
        lang_file = _texts_dir / f"{lang}_texts.json"
        english_data = read_json(en_file, default_value=None)
        if english_data is None:
            logger.error(f"Required English language file missing or invalid for toolset '{_toolset_id}': {en_file}. Texts will be unavailable.")
            _texts_data = {}; return
        else:
            _texts_data = english_data
            logger.debug(f"Loaded English texts for toolset '{_toolset_id}' from {en_file}")
        if lang != "en":
            if lang_file.exists():
                lang_data = read_json(lang_file, default_value=None)
                if lang_data is not None:
                    _texts_data = deep_merge_dicts(english_data, lang_data)
                    logger.debug(f"Merged '{lang}' language texts for toolset '{_toolset_id}' from {lang_file}")
                else: logger.warning(f"Language file for '{lang}' found but invalid for toolset '{_toolset_id}': {lang_file}. Using English only.")
            else: logger.warning(f"Language file for '{lang}' not found for toolset '{_toolset_id}': {lang_file}. Using English only.")

    def get_text(key_path: str, **kwargs) -> str:
        # ... (copy implementation from files/texts.py or similar) ...
        # Ensure it uses _toolset_id in log messages
        keys = key_path.split('.')
        value = _texts_data
        try:
            for key in keys: value = value[key]
            if isinstance(value, str):
                try: return Formatter().format(value, **kwargs)
                except KeyError as e: logger.warning(f"Missing key '{e}' for formatting toolset '{_toolset_id}' text '{key_path}'. Args: {kwargs}"); return value
                except Exception as format_e: logger.error(f"Error formatting toolset '{_toolset_id}' text '{key_path}'. Args: {kwargs}. Error: {format_e}"); return value
            else: logger.warning(f"Toolset '{_toolset_id}' text key '{key_path}' did not resolve to a string. Type: {type(value)}"); return key_path
        except (KeyError, TypeError): logger.error(f"Toolset '{_toolset_id}' text key '{key_path}' not found."); return key_path

    _load_texts() # Load texts when module is imported
    ```

**d) `default_settings/default_settings.json` (Optional)**

*   Define default values for your toolset's configuration options (the ones prompted for in `configure_toolset`) or internal parameters. These are used as fallbacks if no user configuration is found.

    ```json
    // ai_shell_agent/toolsets/my_new_toolset/default_settings/default_settings.json
    {
        "api_url": "https://api.example.com/v1",
        "request_timeout_seconds": 30,
        "feature_flag_x": false
    }
    ```

**e) `settings.py` (Optional)**

*   Loads the values from `default_settings.json` into Python constants for easy access within your toolset code, especially within the `configure_toolset` function. Copy the structure from an existing toolset `settings.py` and adapt it.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/settings.py
    from pathlib import Path
    from typing import Any, Dict
    from ...utils.file_io import read_json
    from ... import logger

    _toolset_id = 'my_new_toolset' # <-- SET YOUR TOOLSET ID
    _settings_file = Path(__file__).parent / 'default_settings' / 'default_settings.json'

    def _load_data(file_path):
        data = read_json(file_path, default_value={})
        if not data: logger.error(f"Error loading settings for toolset '{_toolset_id}' from {file_path}. Defaults missing.")
        return data

    _settings_data: Dict[str, Any] = _load_data(_settings_file)

    try:
        # Load your specific settings into constants
        DEFAULT_API_URL: str = _settings_data['api_url']
        DEFAULT_TIMEOUT: int = _settings_data['request_timeout_seconds']
        DEFAULT_FEATURE_X: bool = _settings_data['feature_flag_x']
    except KeyError as e:
        logger.critical(f"Missing expected key in {_toolset_id} default_settings.json: {e}.")
        raise
    ```

**f) `prompts.py` (Optional)**

*   Define a constant string named `{TOOLSET_ID_UPPER}_TOOLSET_PROMPT` (e.g., `MY_NEW_TOOLSET_PROMPT`).
*   This string provides context to the AI about the toolset's capabilities and how to use its tools effectively. It's injected into the chat history as a `ToolMessage` when the toolset is first activated or used in a chat session.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/prompts.py
    """
    Prompt content for My New Toolset.
    """
    MY_NEW_TOOLSET_PROMPT = """\
    You have activated My New Toolset.
    - Use `mytool_perform_action` to process items identified by their ID. Provide parameters as a JSON string if needed.
    - Use `mytool_risky_operation` carefully for irreversible actions, specifying the target resource. It requires user confirmation.
    - Always verify the item ID exists before calling `mytool_perform_action`.
    """
    ```

**g) `tools/schemas.py` (Recommended)**

*   Define Pydantic `BaseModel` subclasses for the input arguments of each tool.
*   Use `pydantic.Field` to provide descriptions for each argument, fetching the text using `get_text()` from your toolset's `texts.py`. This description is crucial for the LLM.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/tools/schemas.py
    from pydantic import BaseModel, Field
    from typing import Optional, Type
    from ..texts import get_text # Import toolset-specific get_text

    class PerformActionSchema(BaseModel):
        item_id: str = Field(..., description=get_text("schemas.perform_action.item_id_desc"))
        parameters: Optional[str] = Field(None, description=get_text("schemas.perform_action.parameters_desc"))

    class RiskyOperationSchema(BaseModel):
        target: str = Field(..., description=get_text("schemas.risky_operation.target_desc"))
        # Example: Include summary directly if needed by tool, although confirmation handles summary display
        summary: str = Field(..., description=get_text("schemas.risky_operation.operation_summary_desc"))

    class NoArgsSchema(BaseModel):
        """Input schema for tools that require no arguments."""
        pass
    ```

**h) `tools/tools.py` (Recommended)**

*   Define your actual tool classes here, inheriting from `langchain_core.tools.BaseTool`.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/tools/tools.py
    from typing import Optional, Type
    from pydantic import BaseModel, Field # Keep pydantic imports if schemas defined here
    from langchain_core.tools import BaseTool
    from langchain_core.runnables.config import run_in_executor

    # Use .. imports to navigate up
    from ... import logger
    from ...errors import PromptNeededError
    from ..texts import get_text # Toolset-specific text getter
    from .schemas import PerformActionSchema, RiskyOperationSchema, NoArgsSchema # Import schemas

    class MyNewToolsetUsageGuideTool(BaseTool):
        name: str = "mytool_usage_guide" # HARDCODED name
        description: str = get_text("tools.usage_guide.description")
        args_schema: Type[BaseModel] = NoArgsSchema

        def _run(self) -> str:
            from ..prompts import MY_NEW_TOOLSET_PROMPT # Import prompt content dynamically
            logger.debug(f"{self.name} invoked.")
            return MY_NEW_TOOLSET_PROMPT

        async def _arun(self) -> str: return await run_in_executor(None, self._run)

    class PerformActionTool(BaseTool):
        name: str = "mytool_perform_action" # HARDCODED name
        description: str = get_text("tools.perform_action.description")
        args_schema: Type[BaseModel] = PerformActionSchema

        def _run(self, item_id: str, parameters: Optional[str] = None) -> str:
            try:
                logger.info(f"Performing action on {item_id} with params: {parameters}")
                # ... implementation ...
                # Use get_text for success/error messages
                return get_text("tools.perform_action.success", item_id=item_id)
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}", exc_info=True)
                return get_text("tools.perform_action.error_generic", item_id=item_id, error=e)

        async def _arun(self, item_id: str, parameters: Optional[str] = None) -> str:
            return await run_in_executor(None, self._run, item_id, parameters)

    class RiskyOperationTool(BaseTool):
        name: str = "mytool_risky_operation" # HARDCODED name
        description: str = get_text("tools.risky_operation.description")
        args_schema: Type[BaseModel] = RiskyOperationSchema
        requires_confirmation: bool = True # Mark as needing HITL

        def _run(self, target: str, summary: str, confirmed_input: Optional[str] = None) -> str:
            if confirmed_input is None:
                # Proposal Phase: Raise PromptNeededError
                logger.debug(f"Raising PromptNeededError for risky operation on {target}")

                # Format the confirmation prompt using texts.py
                prompt_message = get_text(
                    "tools.risky_operation.confirm_prompt",
                    tool_name=self.name,
                    summary=summary,
                    target=target
                )
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"confirmation_prompt": prompt_message, "_target": target, "_summary": summary},
                    edit_key="confirmation_prompt", # Use standard key for Yes/No
                    prompt_suffix=get_text("tools.risky_operation.confirm_suffix_yes_no") # Use standard Yes/No suffix
                )
            else:
                # Execution Phase: Check confirmation
                if confirmed_input.lower().strip() not in ['yes', 'y', 'confirm']:
                     logger.info(f"User rejected risky operation on '{target}'")
                     return get_text("tools.risky_operation.info_rejected")

                # User confirmed, proceed with original target
                final_target = target # Use the original target passed
                logger.info(f"Executing confirmed risky operation on '{final_target}'")
                try:
                    # ... implementation of the risky action ...
                    return get_text("tools.risky_operation.success_confirm", target=final_target)
                except Exception as e:
                    logger.error(f"Error during confirmed risky operation on {final_target}: {e}", exc_info=True)
                    return get_text("tools.risky_operation.error_generic", target=final_target, error=e)

        async def _arun(self, target: str, summary: str, confirmed_input: Optional[str] = None) -> str:
             return await run_in_executor(None, self._run, target, summary, confirmed_input)

    # Instantiate tools defined above
    my_new_toolset_usage_guide_tool = MyNewToolsetUsageGuideTool()
    perform_action_tool = PerformActionTool()
    risky_operation_tool = RiskyOperationTool()
    ```

**i) `toolset.py` (REQUIRED)**

This file ties everything together: metadata, configuration, and tool registration.

*   **Imports:** Import needed modules: `BaseTool`, `List`, `Dict`, `Optional`, `Callable`, `Path`, `logger`, `register_tools`, `ensure_dotenv_key`, `read_json`, `write_json`, `get_console_manager`, `PromptNeededError`. Import `get_text` from *your* toolset's `texts.py`. Import the tool **instances** from your `tools/tools.py`.
*   **Metadata:** Define `toolset_id`, `toolset_name`, `toolset_description`, and `toolset_required_secrets` at the top level.
*   **`configure_toolset` Function:** Implement this function (even if it does nothing but write empty config files) as described previously. It handles user interaction for settings and ensures required secrets are present.
*   **Tool List:** Create a list `toolset_tools: List[BaseTool]` containing the **instances** of your tools imported from `tools.py`.
*   **Registration Call:** At the **very end** of the file, call `register_tools(toolset_tools)`.

    ```python
    # ai_shell_agent/toolsets/my_new_toolset/toolset.py
    from typing import List, Dict, Optional, Any, Type, Callable
    from pathlib import Path
    from langchain_core.tools import BaseTool

    # Use .. imports to navigate up
    from ... import logger
    from ...tool_registry import register_tools
    from ...utils.file_io import read_json, write_json
    from ...utils.env import ensure_dotenv_key
    from ...errors import PromptNeededError
    from ...console_manager import get_console_manager
    from .texts import get_text # Toolset-specific text getter
    # Import settings constants if needed for defaults in configure_toolset
    from .settings import DEFAULT_API_URL, DEFAULT_TIMEOUT
    # Import tool INSTANCES from tools/tools.py
    from .tools.tools import (
        my_new_toolset_usage_guide_tool,
        perform_action_tool,
        risky_operation_tool
    )

    console = get_console_manager()

    # --- Toolset Metadata ---
    toolset_id = "my_new_toolset"
    toolset_name = get_text("toolset.name") # "My New Toolset"
    toolset_description = get_text("toolset.description")
    toolset_required_secrets: Dict[str, str] = {
        "MYTOOLSET_API_KEY": get_text("secrets.api_key_desc")
    }

    # --- Configuration Function ---
    def configure_toolset(
        global_config_path: Path,
        local_config_path: Optional[Path],
        dotenv_path: Path,
        current_config_for_prompting: Optional[Dict]
    ) -> Dict:
        """Handles configuration for My New Toolset."""
        is_global_only = local_config_path is None
        context_name = "Global Defaults" if is_global_only else "Current Chat"
        logger.info(f"Configuring My New Toolset ({context_name}).")

        config_to_prompt = current_config_for_prompting or {}
        final_config = {}

        console.display_message("SYSTEM:", get_text("config.header"),
                              console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

        try:
            # 1. Prompt for API URL
            current_url = config_to_prompt.get("api_url", DEFAULT_API_URL)
            api_url = console.prompt_for_input(
                get_text("config.prompt_api_url"),
                default=current_url
            ).strip()
            # Basic URL validation example (improve as needed)
            if not api_url.startswith(("http://", "https://")):
                 api_url = current_url # Keep current if invalid
                 console.display_message("WARNING:", get_text("config.warn_invalid_url"),
                                       console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            final_config["api_url"] = api_url

            # 2. Ensure API Key Secret
            ensure_dotenv_key(
                dotenv_path,
                "MYTOOLSET_API_KEY",
                toolset_required_secrets["MYTOOLSET_API_KEY"] # Pass description
            )
            # Add prompts for other settings (e.g., timeout) similarly...

        except (KeyboardInterrupt, EOFError):
             logger.warning("Configuration cancelled by user.")
             # Return previous config or defaults if cancelled
             return current_config_for_prompting or {"api_url": DEFAULT_API_URL}
        except Exception as e:
             logger.error(f"Error during My New Toolset configuration: {e}", exc_info=True)
             return current_config_for_prompting or {"api_url": DEFAULT_API_URL}

        # Add a key to indicate configuration was done (useful for state checks)
        final_config["_configured_"] = True

        # Save config to global and local (if applicable) paths
        save_success_global = write_json(global_config_path, final_config)
        save_success_local = True
        if local_config_path:
            save_success_local = write_json(local_config_path, final_config)

        if save_success_global and save_success_local:
             console.display_message("INFO:", get_text("config.info_saved"),
                                   console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        else:
             console.display_message("ERROR:", get_text("config.error_save_failed"),
                                   console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

        return final_config


    # --- Toolset Tools List ---
    toolset_tools: List[BaseTool] = [
        my_new_toolset_usage_guide_tool,
        perform_action_tool,
        risky_operation_tool,
        # Add other tool instances here
    ]

    # --- Register Tool Instances ---
    register_tools(toolset_tools)
    logger.debug(f"Registered {toolset_name} ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")

    ```

## 3. Testing Your Toolset

Follow these steps rigorously to ensure your toolset integrates correctly:

1.  **Restart Application:** Make sure the agent picks up the new toolset module.
2.  **Discovery & Metadata:**
    *   Run `ai --list-toolsets`.
    *   Verify your toolset (`My New Toolset`) appears with the correct description from `texts.py`.
    *   Check `data/agent.log` for registration messages and any errors during discovery.
3.  **Default Enable (Optional):** Add your toolset's display name (`My New Toolset`) to the `default_enabled_toolsets` list in `ai_shell_agent/default_settings/default_settings.json` if you want it enabled by default for new chats.
4.  **Configuration:**
    *   Start a new chat: `ai -c test_my_toolset`.
    *   If your toolset is default enabled, the `configure_toolset` function (if defined) should run immediately.
    *   If not default enabled, enable it: `ai --select-tools` (while in the chat or without an active chat for global defaults) and select your toolset. Then, the next interaction should trigger configuration.
    *   Test the prompts defined in `configure_toolset`. Check default values are loaded correctly.
    *   Test secret handling: Temporarily remove the secret from `.env` and run the configuration again. Verify `ensure_dotenv_key` prompts correctly using the description from `toolset_required_secrets`. Test cancelling the prompt.
    *   Verify saved config: Check the contents of `data/chats/test_my_toolset/toolsets/my_new_toolset.json` and the global default `data/toolsets/my_new_toolset.json`.
5.  **Activation & Usage:**
    *   In your test chat (`ai -c test_my_toolset`), ask the AI to perform tasks that should use your tools. Be specific: "Use 'My New Toolset' to perform action X on item Y." or reference tool names directly if needed for testing "Use tool `mytool_perform_action`...".
    *   Check `data/agent.log` for `INFO` messages confirming tool execution and any errors.
    *   Verify the tool's return messages (using `get_text()`) are displayed correctly.
    *   Test argument passing and parsing via the Pydantic schemas.
6.  **HITL Testing (If Applicable):**
    *   Invoke tools where `requires_confirmation=True`.
    *   Verify the `PromptNeededError` is raised correctly.
    *   Check that `console_manager` displays the correct prompt (using `proposed_args['confirmation_prompt']` or the `edit_key` value as default text, plus the `prompt_suffix`). Use `get_text` in the tool for the suffix.
    *   Confirm the prompt ("yes"). Verify the tool executes using the original/confirmed arguments.
    *   Reject the prompt ("no" or Ctrl+C). Verify the tool does *not* execute and returns an appropriate rejection message (e.g., `get_text("tools.common.info_rejected")`).
    *   Test editing the value in the prompt (if not using `confirmation_prompt` as the `edit_key`). Verify the tool uses the *edited* value.
7.  **Localization (Optional but Recommended):**
    *   Run `ai --localize <lang_code>` (e.g., `ai --localize pl`).
    *   Check the generated `ai_shell_agent/toolsets/my_new_toolset/texts/<lang_code>_texts.json`. Verify tool names (`mytool_perform_action`) were **not** translated, but descriptions and UI messages were.
    *   Run `ai --select-language` to switch, **restart the agent**, and test your toolset's interactions (prompts, output messages) in the new language.

By following this guide, you can create robust and well-integrated toolsets, extending the capabilities of the AI Shell Agent. Remember to check the logs frequently during development.