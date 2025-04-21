Okay, let's update the contributing guide (`how_to_add_toolsets.md`) based on the current codebase structure, features, and the `files` toolset example.

```markdown
# How to Add a New Toolset

The `ai-shell-agent` uses a modular architecture where capabilities are grouped into "Toolsets." Each toolset is self-contained within its own directory, defining its tools, configuration, and user-facing text. This guide explains how to create and integrate a new toolset.

## 1. Create the Toolset Directory Structure

*   Navigate to the `ai_shell_agent/toolsets/` directory.
*   Create a new subdirectory for your toolset. The name of this directory becomes the unique **Toolset ID** (e.g., `web_search`, `database_query`, `files`). Choose a concise, lowercase name.
    *   Example: `ai_shell_agent/toolsets/my_toolset/`
*   Inside this directory, create the following structure:

    ```
    ai_shell_agent/
    └── toolsets/
        └── my_toolset/
            ├── __init__.py             # Package marker
            ├── toolset.py              # Core logic: tools, metadata, config
            ├── prompts.py              # (Optional) AI context prompt on activation
            ├── settings.py             # (Optional) Loads default settings
            ├── default_settings/       # (Optional) Directory for settings
            │   └── default_settings.json # (Optional) Default values
            ├── texts/                  # (Recommended) Directory for UI texts
            │   └── en_texts.json       # English UI texts (required if texts/ exists)
            │   └── <lang>_texts.json   # (Optional) Other language UI texts
            └── texts.py                # Loads UI texts for the toolset
    ```

## 2. Implement Core Files

**a) `__init__.py`**

*   Marks the directory as a Python package.
*   Crucially, it **must** import the `toolset` module to trigger registration during discovery.

    ```python
    # ai_shell_agent/toolsets/my_toolset/__init__.py
    """
    My Toolset package initialization.
    Provides tools for X and Y.
    """
    from . import toolset # Import toolset to ensure registration runs on discovery
    ```

**b) `texts/en_texts.json` (Recommended)**

*   Stores all user-facing strings for your toolset in English (descriptions, prompts, error messages, etc.) in a nested JSON format.
*   **Important:** Do **NOT** put internal tool names (like `mytool_do_something`) here. Tool names must be hardcoded in `toolset.py` to comply with API requirements.
*   Follow the structure of existing `en_texts.json` files (e.g., `files/texts/en_texts.json`). Define logical top-level keys (like `toolset`, `config`, `schemas`, `tools`).

    ```json
    // ai_shell_agent/toolsets/my_toolset/texts/en_texts.json
    {
      "toolset": {
        "name": "My Custom Toolset",
        "description": "Provides tools for doing custom things."
      },
      "config": {
        "header": "--- Configure My Toolset ---",
        "prompt_endpoint": "Enter API endpoint",
        "info_saved": "My Toolset configuration saved."
        // ... other config-related texts
      },
      "schemas": {
        "do_something": {
          "target_desc": "The target object.",
          "value_desc": "The value to set."
        }
        // ... other schema descriptions
      },
      "tools": {
        "usage_guide": {
           // "name" key is NOT needed here anymore
           "description": "Displays usage instructions for My Toolset."
        },
        "do_something": {
          // "name" key is NOT needed here anymore
          "description": "Performs a specific action on a target.",
          "success": "Successfully did something with {target}.",
          "error_generic": "Failed to do something with {target}: {error}"
        },
        "risky_action": {
          // "name" key is NOT needed here anymore
          "description": "Performs a potentially risky action that requires user confirmation.",
          "prompt_suffix": "(Confirm risky action on {target}) > ",
          "success": "Successfully performed confirmed risky action on {final_target}.",
          "error_generic": "Failed risky action on {target}: {error}"
        }
        // ... texts for other tools
      }
    }

    ```

**c) `texts.py` (Recommended)**

*   Loads the text strings from the `texts/` directory based on the application's selected language, falling back to English.
*   Provides a `get_text(key_path, **kwargs)` function specific to this toolset.
*   You can mostly copy the structure from `ai_shell_agent/toolsets/files/texts.py`, just update the `_toolset_id`.

    ```python
    # ai_shell_agent/toolsets/my_toolset/texts.py
    from pathlib import Path
    from string import Formatter
    from ...utils.file_io import read_json
    from ...utils.dict_utils import deep_merge_dicts
    from ...utils.config_reader import get_config_value
    from ... import logger
    from ...settings import APP_DEFAULT_LANGUAGE

    _toolset_id = 'my_toolset' # <-- SET YOUR TOOLSET ID
    _texts_dir = Path(__file__).parent / 'texts'
    _texts_data = {}

    def _load_texts():
        # ... (copy implementation from files/texts.py) ...
        pass

    def get_text(key_path: str, **kwargs) -> str:
        # ... (copy implementation from files/texts.py) ...
        pass

    _load_texts() # Load texts when module is imported
    ```

**d) `default_settings/default_settings.json` (Optional)**

*   If your toolset has specific default parameters (not user-configurable via the wizard, but potentially useful internally or as fallbacks), define them here.
    *   Example: `files` toolset defines defaults for its `find` tool.

    ```json
    // ai_shell_agent/toolsets/my_toolset/default_settings/default_settings.json
    {
        "default_timeout_ms": 5000,
        "internal_batch_size": 100
    }
    ```

**e) `settings.py` (Optional)**

*   Loads the values from `default_settings.json` into Python constants.
*   Copy the structure from `ai_shell_agent/toolsets/files/settings.py`, update `_toolset_id` and the constants you define.

    ```python
    # ai_shell_agent/toolsets/my_toolset/settings.py
    from pathlib import Path
    from typing import Any, Dict
    from ...utils.file_io import read_json
    from ... import logger

    _toolset_id = 'my_toolset' # <-- SET YOUR TOOLSET ID
    _settings_file = Path(__file__).parent / 'default_settings' / 'default_settings.json'

    def _load_data(file_path):
        # ... (copy implementation) ...
        pass

    _settings_data: Dict[str, Any] = _load_data(_settings_file)

    try:
        # Load your specific settings into constants
        MYTOOLSET_DEFAULT_TIMEOUT: int = _settings_data['default_timeout_ms']
        MYTOOLSET_BATCH_SIZE: int = _settings_data['internal_batch_size']
    except KeyError as e:
        logger.critical(f"Missing expected key in {_toolset_id} default_settings.json: {e}.")
        raise
    ```

**f) `prompts.py` (Optional)**

*   Define a constant string `MY_TOOLSET_PROMPT` (replace `MY_TOOLSET` with your toolset ID in uppercase). This text will be shown to the AI as a `ToolMessage` when the toolset is first activated in a chat, providing context on how to use it.

    ```python
    # ai_shell_agent/toolsets/my_toolset/prompts.py
    """
    Prompt content for My Toolset.
    """
    MY_TOOLSET_PROMPT = """\
    You have activated My Toolset.
    Use tool 'mytool_do_something' to achieve X by providing target and value.
    Use tool 'mytool_risky_action' for Y, but be aware it requires confirmation.
    Remember to check Z before proceeding.
    """
    ```

**g) `toolset.py` (Core Implementation)**

This file defines the toolset's metadata, tools, and configuration logic.

*   **Imports:** Import necessary modules (`BaseTool`, `BaseModel`, `Field`, `register_tools`, `logger`, utils, errors, `console_manager`, `get_text` from *your* `texts.py`).
*   **Metadata Variables:** Define these at the top level:
    *   `toolset_id`: String, must match the directory name (e.g., `"my_toolset"`).
    *   `toolset_name`: String, user-friendly name from your texts file (e.g., `get_text("toolset.name")`).
    *   `toolset_description`: String, description from your texts file (e.g., `get_text("toolset.description")`).
    *   `toolset_required_secrets`: Dict[str, str], mapping environment variable names to user-friendly descriptions/URLs (e.g., `{"MYTOOL_API_KEY": "API Key from example.com"}`). Leave as `{}` if none are needed.

*   **`configure_toolset` Function (Optional but Recommended):**
    *   Define this function if your toolset needs settings beyond simple secrets or requires custom validation/prompting.
    *   **Signature:** `configure_toolset(global_config_path: Path, local_config_path: Optional[Path], dotenv_path: Path, current_chat_config: Optional[Dict]) -> Dict:`
    *   **Responsibilities:**
        *   Use `current_chat_config` or toolset defaults to pre-fill prompts.
        *   Use `console.prompt_for_input` (from `ai_shell_agent.console_manager`) to interact with the user. Use `get_text()` for all prompts and messages.
        *   Use `ensure_dotenv_key` (from `ai_shell_agent.utils.env`) to prompt for and save any secrets defined in `toolset_required_secrets`.
        *   Build the final configuration dictionary.
        *   Save the `final_config` dictionary using `write_json` to **both** `local_config_path` (if not `None`) and `global_config_path`.
        *   Return the `final_config` dictionary.
    *   Refer to the `files` or `aider` toolsets for examples.

*   **Tool Schemas (Pydantic Models):** Define input schemas for your tools using `pydantic.BaseModel` and `Field`. Use `get_text()` for the `description` of each field.

*   **Tool Classes:**
    *   Inherit from `langchain_core.tools.BaseTool`.
    *   **`name`:** **Hardcode** the API-compliant tool name (letters, numbers, `_`, `-` only). **Do NOT use `get_text()` here.** (e.g., `name: str = "mytool_do_something"`)
    *   **`description`:** Use `get_text()` to provide a clear description for the AI and user (e.g., `description: str = get_text("tools.do_something.description")`).
    *   **`args_schema`:** Assign your Pydantic schema class (e.g., `args_schema: Type[BaseModel] = DoSomethingSchema`).
    *   **`_run` method:** Implement synchronous logic. Use `get_text()` for any return messages or error strings.
    *   **`_arun` method (optional):** Implement async logic or use `run_in_executor`.
    *   **HITL Tools:**
        *   Add `requires_confirmation: bool = True`.
        *   Add `confirmed_input: Optional[str] = None` to `_run`/`_arun` signature.
        *   If `confirmed_input is None`, `raise PromptNeededError(...)`. Provide `tool_name=self.name`, `proposed_args` (dict matching `args_schema`), and `edit_key` (which argument field the user should edit/confirm). Use `get_text()` for `prompt_suffix` if customizing the prompt.
        *   If `confirmed_input is not None`, execute the action using the `confirmed_input`.

*   **Instantiate and Register Tools:**
    *   Create instances of your tool classes.
    *   Create a list `toolset_tools: List[BaseTool] = [instance1, instance2, ...]`.
    *   Call `register_tools(toolset_tools)` at the **end of the file**.

**Example Snippet (`toolset.py`):**

```python
# ai_shell_agent/toolsets/my_toolset/toolset.py
from typing import List, Dict, Optional, Any, Type, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor

# Use .. imports to navigate up from toolsets/my_toolset/ to the main package level
from ... import logger
from ...tool_registry import register_tools
from ...utils.file_io import read_json, write_json # Example utils
from ...utils.env import ensure_dotenv_key
from ...errors import PromptNeededError
from ...console_manager import get_console_manager
from .texts import get_text # Import from local texts.py

console = get_console_manager()

# --- Toolset Metadata ---
toolset_id = "my_toolset"
toolset_name = get_text("toolset.name")
toolset_description = get_text("toolset.description")
toolset_required_secrets: Dict[str, str] = {
    "MYTOOL_API_KEY": get_text("secrets.api_key_desc") # Get desc from texts
}

# --- Toolset Configuration Function (Optional) ---
def configure_toolset(...) -> Dict:
    # ... (Implementation as described above, using get_text) ...
    pass

# --- Tool Schemas ---
class DoSomethingSchema(BaseModel):
    target: str = Field(description=get_text("schemas.do_something.target_desc"))
    value: int = Field(description=get_text("schemas.do_something.value_desc"))

class RiskyActionSchema(BaseModel):
     target: str = Field(description=get_text("schemas.risky_action.target_desc"))

# --- Tool Classes ---
class DoSomethingTool(BaseTool):
    name: str = "mytool_do_something" # HARDCODED name
    description: str = get_text("tools.do_something.description")
    args_schema: Type[BaseModel] = DoSomethingSchema

    def _run(self, target: str, value: int) -> str:
        try:
            logger.info(f"MyTool: Doing something with {target}={value}")
            # ... implementation ...
            return get_text("tools.do_something.success", target=target)
        except Exception as e:
            logger.error(f"MyTool Error: {e}", exc_info=True)
            return get_text("tools.do_something.error_generic", target=target, error=e)

    async def _arun(self, target: str, value: int) -> str:
        return await run_in_executor(None, self._run, target, value)

class RiskyActionTool(BaseTool):
    name: str = "mytool_risky_action" # HARDCODED name
    description: str = get_text("tools.risky_action.description")
    args_schema: Type[BaseModel] = RiskyActionSchema
    requires_confirmation: bool = True

    def _run(self, target: str, confirmed_input: Optional[str] = None) -> str:
        if confirmed_input is None:
            logger.debug(f"MyTool: Raising PromptNeededError for risky action on {target}")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"target": target},
                edit_key="target",
                prompt_suffix=get_text("tools.risky_action.prompt_suffix", target=target)
            )
        else:
            final_target = confirmed_input
            logger.info(f"MyTool: Performing confirmed risky action on {final_target}")
            # ... perform action ...
            return get_text("tools.risky_action.success", final_target=final_target)

    async def _arun(self, target: str, confirmed_input: Optional[str] = None) -> str:
         return await run_in_executor(None, self._run, target, confirmed_input)


# --- Instantiate Tools ---
do_something_tool_instance = DoSomethingTool()
risky_action_tool_instance = RiskyActionTool()

# --- Define Toolset Structure ---
toolset_tools: List[BaseTool] = [
    do_something_tool_instance,
    risky_action_tool_instance,
]

# --- Register Tools ---
register_tools(toolset_tools)
logger.debug(f"Registered {toolset_name} ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")

```

## 3. Testing Your Toolset

*   **Discovery & Metadata:** Run `ai --list-toolsets`. Verify your toolset appears with the correct name and description (from `texts.py`). Check `data/agent.log` for registration messages and errors.
*   **Configuration:**
    *   Enable the toolset: `ai -c test_chat --select-tools`.
    *   Trigger configuration: The first time you use the toolset in `test_chat`, the `configure_toolset` function should run (if defined). Test its prompts, defaults, and secret handling (`ensure_dotenv_key`). If no `configure_toolset` is defined, ensure secrets prompt correctly if missing from `.env`.
    *   Verify saved config: Check `data/chats/test_chat/toolsets/my_toolset.json` and `data/toolsets/my_toolset.json`.
*   **Activation & Usage:**
    *   Ask the AI to perform tasks that should use your tools (e.g., `ai -c test_chat "Use my toolset to do something with target=abc and value=123"`).
    *   Check logs for tool execution messages.
    *   Verify expected outputs are returned.
    *   Test HITL tools: Ensure confirmation prompts appear correctly (using `get_text` for the suffix), editing works, and the action executes only after confirmation. Check cancellation (Ctrl+C).
*   **Localization (Optional):**
    *   Run `ai --localize <lang_code>` to generate translations.
    *   Check the generated `my_toolset/texts/<lang_code>_texts.json`. Verify tool names were *not* translated, but descriptions were.
    *   Run `ai --select-language` to switch, restart, and test the UI in the new language.

By following these steps and referencing the existing toolsets, you can effectively extend the AI Shell Agent with new capabilities.```

## 3. Testing Your Toolset

*   **Discovery & Metadata:** Run `ai --list-toolsets`. Verify your toolset appears with the correct name and description (from `texts.py`). Check `data/agent.log` for registration messages and errors.
*   **Configuration:**
    *   Enable the toolset: `ai -c test_chat --select-tools`.
    *   Trigger configuration: The first time you use the toolset in `test_chat`, the `configure_toolset` function should run (if defined). Test its prompts, defaults, and secret handling (`ensure_dotenv_key`). If no `configure_toolset` is defined, ensure secrets prompt correctly if missing from `.env`.
    *   Verify saved config: Check `data/chats/test_chat/toolsets/my_toolset.json` and `data/toolsets/my_toolset.json`.
*   **Activation & Usage:**
    *   Ask the AI to perform tasks that should use your tools (e.g., `ai -c test_chat "Use my toolset to do something with target=abc and value=123"`).
    *   Check logs for tool execution messages.
    *   Verify expected outputs are returned.
    *   Test HITL tools: Ensure confirmation prompts appear correctly (using `get_text` for the suffix), editing works, and the action executes only after confirmation. Check cancellation (Ctrl+C).
*   **Localization (Optional):**
    *   Run `ai --localize <lang_code>` to generate translations.
    *   Check the generated `my_toolset/texts/<lang_code>_texts.json`. Verify tool names were *not* translated, but descriptions were.
    *   Run `ai --select-language` to switch, restart, and test the UI in the new language.

By following these steps and referencing the existing toolsets, you can effectively extend the AI Shell Agent with new capabilities.