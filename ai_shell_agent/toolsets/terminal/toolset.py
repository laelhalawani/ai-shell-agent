# ai_shell_agent/toolsets/terminal/toolset.py
"""
Defines the tools and metadata for the Terminal toolset.
"""
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path # Keep Path import

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from pydantic import Field

# Import the sanitize_input function directly from the module
from langchain_experimental.tools.python.tool import sanitize_input, _get_default_python_repl

# External imports
from prompt_toolkit import prompt

# Local imports
from ... import logger
from ...tool_registry import register_tools
from ...chat_state_manager import (
    get_current_chat,
    get_active_toolsets,
    update_active_toolsets,
    # Removed _write_json, use utils
)
# Import the prompt content to be returned by the start tool
from .prompts import TERMINAL_TOOLSET_PROMPT

# Import utils for JSON I/O
from ...utils import write_json as _write_json, read_json as _read_json # Add read_json if needed

# --- Toolset Metadata ---
toolset_name = "Terminal"
toolset_id = "terminal" # Add explicit ID
toolset_description = "Provides tools to execute shell commands and Python code."
# --- NEW: Required Secrets ---
toolset_required_secrets: Dict[str, str] = {} # Terminal needs no secrets

# --- Configuration ---
toolset_config_defaults = {} # No specific config needed yet

# --- MODIFIED: configure_toolset signature and saving logic ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Path,
    dotenv_path: Path, # Accept dotenv_path even if unused
    current_chat_config: Optional[Dict] # Accept chat config even if unused
) -> Dict:
    """
    Configuration function for the Terminal toolset. Terminal currently needs
    no specific configuration, so this writes empty config files to both paths
    to mark it as configured.
    """
    logger.info(f"Terminal toolset requires no specific configuration. Writing empty config to local: {local_config_path} and global: {global_config_path}")
    final_config = {} # Empty config
    save_success = True
    try:
        # Ensure directory exists before writing
        local_config_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(local_config_path, final_config)
        logger.debug(f"Wrote empty config for Terminal toolset to local path: {local_config_path}")
    except Exception as e:
         save_success = False
         logger.error(f"Failed to write empty config for Terminal toolset to local path {local_config_path}: {e}")

    try:
        # Ensure directory exists before writing
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(global_config_path, final_config)
        logger.debug(f"Wrote empty config for Terminal toolset to global path: {global_config_path}")
    except Exception as e:
         save_success = False
         logger.error(f"Failed to write empty config for Terminal toolset to global path {global_config_path}: {e}")

    if not save_success:
        print(f"\nWarning: Failed to write configuration files for Terminal toolset. Check logs.")
    else:
        # Provide confirmation only if run interactively (e.g., via --configure-toolset)
        # We don't have that context here, so maybe skip confirmation?
        # Or print a generic one:
        # print("\nTerminal toolset configuration check complete (no settings required).")
        pass

    # Return the empty config
    return final_config

# --- Tool Classes (Migrated from terminal_tools.py) ---

class StartTerminalTool(BaseTool):
    name: str = "start_terminal"
    description: str = "Activates the Terminal toolset, enabling execution of shell commands and Python code."

    def _run(self, *args, **kwargs) -> str:
        """Activates the 'Terminal' toolset and returns usage instructions."""
        logger.info(f"StartTerminalTool called")

        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        # --- NEW: Ensure configuration check runs ---
        # Check/run configuration before activating if needed.
        # This ensures the empty config files are created, marking it "configured".
        from ...chat_state_manager import check_and_configure_toolset
        check_and_configure_toolset(chat_id, toolset_name) # Add this call
        # --- End configuration check ---

        current_toolsets = get_active_toolsets(chat_id)
        if toolset_name in current_toolsets:
            logger.debug(f"'{toolset_name}' toolset is already active.")
            return f"'{toolset_name}' toolset is already active.\n\n{TERMINAL_TOOLSET_PROMPT}"

        # Activate the toolset
        new_toolsets = list(current_toolsets)
        new_toolsets.append(toolset_name)
        update_active_toolsets(chat_id, new_toolsets) # Save updated toolsets list

        logger.debug(f"Successfully activated '{toolset_name}' toolset for chat {chat_id}")
        return TERMINAL_TOOLSET_PROMPT

    async def _arun(self, *args, **kwargs) -> str:
        """Run the tool asynchronously."""
        # For now, run synchronously using run_in_executor if needed, or just call sync
        return self._run(*args, **kwargs)

class TerminalTool_HITL(BaseTool):
    """
    Tool for interacting with the system's shell with human-in-the-loop confirmation.
    """
    name: str = "terminal"
    description: str = (
        "Executes a command in the system's terminal (e.g., bash, cmd, powershell). "
        "User will be prompted to confirm or edit the command before execution."
        "Use this for file operations, system checks, installations, etc."
    )

    def _run(self, cmd: str) -> str:
        """Run a command in a shell, prompting the user first."""
        cmd_to_execute = cmd.strip()
        if not cmd_to_execute:
            return "Error: Empty command proposed."

        # --- HITL Prompt (Tool's responsibility) ---
        # Proposal notification is printed by chat_manager BEFORE this runs
        try:
            edited_command = prompt(
                "(edit or confirm) > ",
                default=cmd_to_execute
            )
        except EOFError:
            logger.warning("User cancelled command input (EOF).")
            return "Command input cancelled by user."
        except KeyboardInterrupt:
            # prompt_toolkit might handle the print, add log just in case
            logger.warning("User cancelled command input (KeyboardInterrupt).")
            # Return cancellation message consistently
            return "Command input cancelled by user."

        if not edited_command.strip():
            logger.warning("User cancelled command input by submitting empty input.")
            return "Command input cancelled by user (empty input)."
        # --- End HITL Prompt ---

        logger.info(f"Executing terminal command: {edited_command}")
        try:
            result = subprocess.run(
                edited_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            output_parts = []
            # Include command info in the result for clarity
            output_parts.append(f"Executed: `{edited_command}`")

            if result.returncode != 0:
                output_parts.append(f"Exit Code: {result.returncode}")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(f"Output:\n---\n{stdout}\n---")
            if stderr:
                output_parts.append(f"Errors/Warnings:\n---\n{stderr}\n---")

            if not stdout and not stderr and result.returncode == 0:
                output_parts.append("Status: Command completed successfully with no output.")
            elif not stdout and not stderr and result.returncode != 0:
                 output_parts.append("Status: Command failed with no output.")

            return "\n".join(output_parts)

        except FileNotFoundError:
             logger.error(f"Error executing command '{edited_command}': Command not found.")
             return f"Error: Command not found: '{edited_command.split()[0]}'. Please ensure it's installed and in the system's PATH. (Attempted: `{edited_command}`)"
        except Exception as e:
            logger.error(f"Error executing command '{edited_command}': {e}", exc_info=True)
            return f"Error executing command: {str(e)}\n(Attempted: `{edited_command}`)"

    async def _arun(self, command: str = None, **kwargs) -> str:
        return await run_in_executor(None, self._run, command, **kwargs)

    # Removed invoke - standard agent framework calls _run/_arun directly with dict args

class PythonREPLTool_HITL(BaseTool):
    """
    Human-in-the-loop wrapper for Python REPL execution.
    """
    name: str = "python_repl"
    description: str = (
        "Evaluates Python code snippets in a REPL environment. "
        "User will be prompted to confirm or edit the code before execution."
    )
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True

    def _run(
        self,
        query: str = None,
        **kwargs
    ) -> str:
        """HITL version of the tool."""
        code_to_execute = query

        # Handle query passed via kwargs if needed
        if code_to_execute is None and 'query' in kwargs:
             code_to_execute = kwargs.get('query', '')
        elif code_to_execute is None and 'command' in kwargs:
            logger.warning("Received 'command' argument for python_repl, expected 'query'. Using 'command'.")
            code_to_execute = kwargs.get('command', '')
        elif code_to_execute is None and 'v__args' in kwargs:
             wrapped_args = kwargs.get('v__args', [])
             if isinstance(wrapped_args, list) and wrapped_args:
                 code_to_execute = wrapped_args[0] if isinstance(wrapped_args[0], str) else str(wrapped_args[0])

        if not code_to_execute:
            return "Error: No Python code provided to execute."

        # Keep sanitize logic
        if self.sanitize_input:
            original_code = code_to_execute
            code_to_execute = sanitize_input(code_to_execute)
            if original_code != code_to_execute:
                logger.debug(f"Sanitized Python code: {code_to_execute}")

        # --- HITL Prompt (Tool's responsibility) ---
        # Proposal notification is printed by chat_manager BEFORE this runs
        try:
            edited_query = prompt(
                "(edit or confirm) > ",
                default=code_to_execute,
                multiline=True # Allow multi-line editing for code
            )
        except EOFError:
            logger.warning("User cancelled code input (EOF).")
            return "Code input cancelled by user."
        except KeyboardInterrupt:
            logger.warning("User cancelled code input (KeyboardInterrupt).")
            return "Code input cancelled by user."

        if not edited_query.strip():
            logger.warning("User cancelled code execution by submitting empty input.")
            return "Code execution cancelled by user (empty input)."
        # --- End HITL Prompt ---

        logger.info(f"Executing Python code: {edited_query}")
        try:
            result = self.python_repl.run(edited_query)
            # Format result consistently
            return f"Executed: `python_repl(query='''{edited_query}''')`\nResult:\n---\n{str(result)}\n---"
        except Exception as e:
            logger.error(f"Error executing Python code '{edited_query}': {e}", exc_info=True)
            return f"Error executing Python code:\n---\n{str(e)}\n---\n(Attempted Code:\n'''\n{edited_query}\n'''\n)"

    async def _arun(
        self,
        query: str = None,
         **kwargs
    ) -> str:
        # Ensure 'query' kwarg is passed correctly if called via agent
        query_arg = query if query is not None else kwargs.get('query')
        if query_arg is None: return "Error: No query provided to python_repl tool."
        return await run_in_executor(None, self._run, query_arg)

# --- Tool Instances ---
start_terminal_tool = StartTerminalTool()
terminal_tool = TerminalTool_HITL()
python_repl_tool = PythonREPLTool_HITL()

# --- Toolset Definition ---
toolset_start_tool: BaseTool = start_terminal_tool
toolset_tools: List[BaseTool] = [
    terminal_tool,
    python_repl_tool,
]

# --- Register Tools ---
register_tools([toolset_start_tool] + toolset_tools)
logger.debug(f"Terminal toolset tools registered: {[t.name for t in [toolset_start_tool] + toolset_tools]}")

# --- Direct Execution Helper (Not part of toolset, kept for execute() in chat_manager) ---
# This tool is NOT registered for the LLM, only used internally by `ai -x`
class TerminalTool_Direct(BaseTool):
    name: str = "_internal_direct_terminal" # Internal name
    description: str = "Internal tool for direct command execution without HITL."

    def _run(self, command: str) -> str:
        logger.info(f"Executing direct command internally: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            output = ""
            if stdout:
                output += f"Output:\n{stdout}\n"
            if stderr:
                output += f"Errors/Warnings:\n{stderr}\n"
            if result.returncode != 0:
                 output += f"Exit Code: {result.returncode}\n"
            if not output:
                output = "Command executed with no output."
            return output.strip()
        except Exception as e:
            logger.error(f"Direct execution failed for '{command}': {e}", exc_info=True)
            return f"Error executing command: {str(e)}"

    async def _arun(self, command: str) -> str:
         return await run_in_executor(None, self._run, command)

# Instance for internal use
_direct_terminal_tool_instance = TerminalTool_Direct()

def run_direct_terminal_command(command: str) -> str:
    """Function to run command using the internal direct tool"""
    return _direct_terminal_tool_instance._run(command=command)