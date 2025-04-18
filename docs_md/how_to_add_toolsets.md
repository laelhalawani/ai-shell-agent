# How to Add a New Toolset

The `ai-shell-agent` uses a modular architecture where capabilities are grouped into "Toolsets." Each toolset is self-contained within its own directory. This guide explains how to create and integrate a new toolset.

## 1. Create the Toolset Directory

*   Navigate to the `ai_shell_agent/toolsets/` directory within the project.
*   Create a new subdirectory for your toolset. The name of this directory will become the unique **Toolset ID**. Choose a concise, descriptive name (e.g., `web_search`, `database_query`).
    *   Example: `ai_shell_agent/toolsets/my_toolset/`

## 2. Create Required Files

Inside your new toolset directory (`ai_shell_agent/toolsets/my_toolset/`), create the following files:

*   **`__init__.py`**: An empty file to mark the directory as a Python package.
    ```python
    # ai_shell_agent/toolsets/my_toolset/__init__.py
    """
    My Toolset package initialization.
    """
    from . import toolset # Import toolset to ensure registration runs on discovery
    ```
*   **`toolset.py`**: This is the main file where you define the toolset's tools, metadata, and configuration logic.

*   **(Optional) `prompts.py`**: If your toolset activation should provide specific instructions or context to the AI, create this file. Define a string variable named `<TOOLSET_ID>_TOOLSET_PROMPT` (all uppercase).
    ```python
    # ai_shell_agent/toolsets/my_toolset/prompts.py
    """
    Prompt content for My Toolset.
    """
    MY_TOOLSET_PROMPT = """\
    You have activated My Toolset.
    Use tool 'do_something' to achieve X.
    Use tool 'get_info' to retrieve Y.
    Remember to check Z before proceeding.
    """
    ```

## 3. Implement `toolset.py`

This file is the heart of your toolset. It needs to define several components:

**a) Imports:** Import necessary modules, including `BaseTool` from `langchain_core.tools`, any schemas for your tools (`BaseModel`, `Field` from `pydantic`), helper functions from `ai_shell_agent.utils`, and crucially, `register_tools` from `ai_shell_agent.tool_registry`.

**b) Tool Classes:** Define your tool classes, inheriting from `langchain_core.tools.BaseTool`.
    *   Each tool needs a unique `name`, a `description` (this is crucial for the AI to understand when to use it), and an `args_schema` (using `pydantic.BaseModel`).
    *   Implement the `_run` method for synchronous execution.
    *   Optionally implement `_arun` for asynchronous execution (can delegate to `_run` using `run_in_executor` if needed).
    *   **HITL Tools:** If a tool requires user confirmation/editing before execution (like running a potentially risky command), add `requires_confirmation: bool = True` to the class definition and modify `_run` to accept an optional `confirmed_input: Optional[str] = None` argument. In the `_run` method:
        *   If `confirmed_input is None`, raise `PromptNeededError` (from `ai_shell_agent.errors`) with the proposed arguments.
        *   If `confirmed_input is not None`, use that value to perform the action.

    ```python
    # ai_shell_agent/toolsets/my_toolset/toolset.py
    from typing import List, Dict, Optional, Any, Type, Callable # Add Callable
    from pathlib import Path
    from pydantic import BaseModel, Field
    from langchain_core.tools import BaseTool
    from langchain_core.runnables.config import run_in_executor

    from ai_shell_agent import logger # Use the central logger
    from ai_shell_agent.tool_registry import register_tools
    from ai_shell_agent.utils import read_json, write_json, ensure_dotenv_key # Import utils
    from ai_shell_agent.errors import PromptNeededError # For HITL tools
    from ai_shell_agent.console_manager import get_console_manager # For configure_toolset prompting

    console = get_console_manager()

    # --- Tool Schemas ---
    class DoSomethingSchema(BaseModel):
        target: str = Field(description="The target object.")
        value: int = Field(description="The value to set.")

    # --- Tool Classes ---
    class DoSomethingTool(BaseTool):
        name: str = "mytool_do_something" # Use a unique prefix
        description: str = "Performs a specific action on a target."
        args_schema: Type[BaseModel] = DoSomethingSchema

        def _run(self, target: str, value: int) -> str:
            logger.info(f"MyTool: Doing something with {target}={value}")
            # ... implementation ...
            return f"Successfully did something with {target}."

        async def _arun(self, target: str, value: int) -> str:
            return await run_in_executor(None, self._run, target, value)

    # --- Example HITL Tool ---
    class RiskyActionTool(BaseTool):
        name: str = "mytool_risky_action"
        description: str = "Performs a potentially risky action that requires user confirmation."
        args_schema: Type[BaseModel] = DoSomethingSchema # Reuse schema for example
        requires_confirmation: bool = True

        def _run(self, target: str, value: int, confirmed_input: Optional[str] = None) -> str:
            # Note: For HITL, the args schema defines what the *LLM provides*.
            # The `confirmed_input` comes from the user via the wrapper.
            # We need to parse the `confirmed_input` if the user changed it.
            # For simplicity here, we assume the user confirms/edits the 'target'.

            if confirmed_input is None:
                # First call, request confirmation
                logger.debug(f"MyTool: Raising PromptNeededError for risky action on {target}")
                # The key in proposed_args MUST match a field in args_schema
                # The edit_key tells the system which arg the user should edit/confirm.
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"target": target, "value": value}, # Provide all args the LLM gave
                    edit_key="target" # Ask user to confirm/edit the 'target'
                )
            else:
                # Second call, input is confirmed/edited
                final_target = confirmed_input # User confirmed/edited the 'target' string
                final_value = value # Use the original value provided by LLM
                logger.info(f"MyTool: Performing confirmed risky action on {final_target} with value {final_value}")
                # ... perform the action using final_target and final_value ...
                return f"Successfully performed confirmed risky action on {final_target}."

        async def _arun(self, target: str, value: int, confirmed_input: Optional[str] = None) -> str:
             return await run_in_executor(None, self._run, target, value, confirmed_input)
    ```

**c) Toolset Metadata Variables:** Define these variables at the module level.

    ```python
    # ai_shell_agent/toolsets/my_toolset/toolset.py

    # --- Toolset Metadata ---
    toolset_id = "my_toolset" # Should match the directory name
    toolset_name = "My Custom Toolset" # User-friendly name
    toolset_description = "Provides tools for doing custom things." # Description for the user
    ```

**d) Configuration Defaults and Secrets:** Define default configuration values and any required environment variables (secrets).

    ```python
    # ai_shell_agent/toolsets/my_toolset/toolset.py

    # --- Toolset Configuration ---
    # Default values for configuration settings specific to this toolset
    toolset_config_defaults: Dict[str, Any] = {
        "api_endpoint": "https://api.example.com/v1",
        "retries": 3,
        "feature_flag": False
    }

    # Secrets required by this toolset
    # Key: Environment variable name
    # Value: Description/URL shown to user if key is missing
    toolset_required_secrets: Dict[str, str] = {
        "MYTOOL_API_KEY": "API Key for the MyTool service (get from example.com/keys)",
        "MYTOOL_SECRET_TOKEN": "Secret token for authentication"
    }
    ```

**e) `configure_toolset` Function (Optional but Recommended):** If your toolset needs configuration beyond secrets (or needs complex secret handling), implement this function. It's called automatically when needed (e.g., first time the toolset is enabled in a chat without existing config).

    ```python
    # ai_shell_agent/toolsets/my_toolset/toolset.py

    # --- Toolset Configuration Function ---
    def configure_toolset(
        global_config_path: Path, # Path to save/read global defaults (data/toolsets/<id>.json)
        local_config_path: Path,  # Path to save/read chat-specific config (data/chats/<chat>/toolsets/<id>.json)
        dotenv_path: Path,        # Path to the main .env file
        current_chat_config: Optional[Dict] # Existing config for this chat (if any)
    ) -> Dict:
        """
        Prompts user for My Toolset configuration and saves it.
        """
        logger.info(f"Configuring My Toolset. Global: {global_config_path}, Local: {local_config_path}")

        # Use current_chat_config or defaults for prompting
        config_to_prompt = current_chat_config or toolset_config_defaults
        final_config = {} # Build the results here

        console.display_message("SYSTEM:", "\n--- Configure My Toolset ---", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

        # Example: Prompt for API endpoint
        try:
            endpoint = console.prompt_for_input(
                "Enter API endpoint",
                default=config_to_prompt.get("api_endpoint")
            ).strip()
            final_config["api_endpoint"] = endpoint or config_to_prompt.get("api_endpoint") # Keep default if empty

            # Example: Prompt for retries (numeric)
            retries_str = console.prompt_for_input(
                "Enter number of retries",
                default=str(config_to_prompt.get("retries", 3)) # Ensure default is string
            ).strip()
            final_config["retries"] = int(retries_str) if retries_str else config_to_prompt.get("retries", 3)

        except (KeyboardInterrupt, ValueError, EOFError):
            console.display_message("WARNING:", "\nConfiguration cancelled or invalid input. Using previous/default values.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            # Return the existing config or defaults on cancellation/error
            return current_chat_config or toolset_config_defaults
        except Exception as e:
            logger.error(f"Error during My Toolset configuration: {e}", exc_info=True)
            console.display_message("ERROR:", f"\nConfiguration error: {e}", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return current_chat_config or toolset_config_defaults

        # Ensure required secrets are present using the utility function
        all_secrets_ok = True
        console.display_message("SYSTEM:", "\nChecking required secrets...", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        for key, description in toolset_required_secrets.items():
            if ensure_dotenv_key(dotenv_path, key, description) is None:
                # ensure_dotenv_key handles prompting if needed
                all_secrets_ok = False # Mark if any key was skipped/failed

        # Save the final configuration to BOTH local and global paths
        save_success = True
        try:
            write_json(local_config_path, final_config)
            logger.info(f"My Toolset configuration saved to local path: {local_config_path}")
        except Exception as e:
            save_success = False; logger.error(f"Failed to save config to {local_config_path}: {e}")
        try:
            write_json(global_config_path, final_config)
            logger.info(f"My Toolset configuration saved to global path: {global_config_path}")
        except Exception as e:
            save_success = False; logger.error(f"Failed to save config to {global_config_path}: {e}")

        # Provide feedback
        if save_success:
            console.display_message("INFO:", "\nMy Toolset configuration saved for this chat and globally.", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            if not all_secrets_ok:
                 console.display_message("WARNING:", "Note: One or more required secrets were skipped. The toolset might not function correctly.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        else:
             console.display_message("ERROR:", "\nFailed to save My Toolset configuration. Check logs.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

        return final_config # Return the configured dictionary
    ```
    *   **Important:** The `configure_toolset` function *must* save the resulting configuration dictionary to *both* `local_config_path` and `global_config_path` using `write_json`. It should also use `ensure_dotenv_key` (from `ai_shell_agent.utils`) to handle prompting for any secrets defined in `toolset_required_secrets`.

**f) Instantiate and Register Tools:** Create instances of your tool classes and register them using `register_tools`. Define which tool (if any) acts as the "starter" for the toolset.

    ```python
    # ai_shell_agent/toolsets/my_toolset/toolset.py

    # --- Instantiate Tools ---
    do_something_tool = DoSomethingTool()
    risky_action_tool = RiskyActionTool()
    # Add other tool instances...

    # --- Define Toolset Structure ---
    # Tool that activates the toolset (can be None)
    toolset_start_tool: Optional[BaseTool] = None # Or assign one of your tool instances

    # List of main tools available when the toolset is active
    toolset_tools: List[BaseTool] = [
        do_something_tool,
        risky_action_tool,
        # Add other tool instances...
    ]

    # --- Register Tools (MUST be done at the end of the file) ---
    # Combine start tool (if exists) and main tools for registration
    all_tool_instances = ([toolset_start_tool] if toolset_start_tool else []) + toolset_tools
    register_tools(all_tool_instances)

    logger.debug(f"Registered My Toolset ({toolset_id}) with tools: {[t.name for t in all_tool_instances]}")
    ```

## 4. How It Works (Summary)

1.  **Discovery:** When the agent starts, `ai_shell_agent/toolsets/toolsets.py` scans the `toolsets/` directory.
2.  **Import & Metadata:** It imports `toolset.py` from each subdirectory found. It reads the metadata variables (`toolset_name`, `toolset_description`, etc.) and the `configure_toolset` function reference (if defined).
3.  **Tool Validation:** It checks if the tools listed in `toolset_tools` and `toolset_start_tool` are valid `BaseTool` instances AND are registered in the central `tool_registry` (this is why the `register_tools` call in your `toolset.py` is essential).
4.  **Configuration Trigger:** When a toolset is enabled for a chat (either manually via `--select-tools` or by the AI activating its `start_tool`), the `chat_state_manager.check_and_configure_toolset` function runs.
5.  **Config Handling:**
    *   It checks for existing configuration (`local_config_path`, `global_config_path`).
    *   If no valid config exists and your toolset provided a `configure_toolset` function, that function is called.
    *   The `configure_toolset` function handles user interaction (via `console_manager`) and saves the config to *both* local and global paths.
    *   If no `configure_toolset` function exists, default values (from `toolset_config_defaults`) and secrets (from `toolset_required_secrets`) are handled automatically (secrets will prompt the user if missing via `ensure_dotenv_key`).
6.  **Tool Binding:** When the AI needs to respond in a chat, `llm.py` gets the list of currently *active* and *enabled* toolsets for that specific chat from `chat_state_manager`. It dynamically binds the appropriate tools (starter tool for enabled-but-inactive, main tools for active) to the LLM for that specific interaction.

## 5. Testing Your Toolset

*   **Discovery:** Run the agent (e.g., `ai --list-toolsets`). Check the output and logs (`data/agent.log` if configured) to ensure your toolset is listed correctly and no errors occurred during discovery.
*   **Configuration:**
    *   Enable your toolset in a chat (`ai -c test_chat --select-tools`, then select your toolset).
    *   If you have a `configure_toolset` function, it should run automatically. Test the prompts and secret handling.
    *   If you rely on defaults/secrets only, ensure `ensure_dotenv_key` prompts correctly if secrets are missing.
    *   Check the contents of `data/chats/test_chat/toolsets/my_toolset.json` and `data/toolsets/my_toolset.json` to verify configuration was saved.
*   **Activation & Usage:**
    *   If you have a `start_tool`, try asking the AI to use it (e.g., `ai -c test_chat "Start my toolset"`). Verify it becomes active (`ai --list-toolsets`). Check if the prompt content from `prompts.py` is returned.
    *   Ask the AI to use the main tools defined in `toolset_tools`. Check logs for execution details and errors.
    *   Test any HITL tools – ensure the confirmation prompt appears correctly and the tool executes after confirmation.

## Examples

Refer to the existing toolsets for practical examples:

*   `ai_shell_agent/toolsets/terminal/`: Simple toolset with HITL tools and minimal configuration.
*   `ai_shell_agent/toolsets/aider/`: Complex toolset with significant state management, background threads, and a detailed `configure_toolset` function.

By following these steps, you can effectively add new capabilities to the AI Shell Agent.

---