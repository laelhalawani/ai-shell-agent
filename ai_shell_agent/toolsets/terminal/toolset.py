# ai_shell_agent/toolsets/terminal/toolset.py
"""
Defines the tools and metadata for the Terminal toolset.
"""
import subprocess
from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from pydantic import Field, BaseModel

# Import the sanitize_input function directly from the module
from langchain_experimental.tools.python.tool import sanitize_input, _get_default_python_repl

# External imports
# Remove prompt_toolkit import from here, use console_io's version
# from prompt_toolkit import prompt

# Local imports
# --- ADD console_io import ---
from ... import logger, console_io
from ...tool_registry import register_tools
from ...chat_state_manager import (
    get_current_chat,
    get_active_toolsets,
    update_active_toolsets,
)
# Import the prompt content to be returned by the start tool
from .prompts import TERMINAL_TOOLSET_PROMPT

# Import utils for JSON I/O
from ...utils import write_json as _write_json, read_json as _read_json

# --- Toolset Metadata ---
toolset_name = "Terminal"
toolset_id = "terminal"
toolset_description = "Provides tools to execute shell commands and Python code."
toolset_required_secrets: Dict[str, str] = {}
toolset_config_defaults = {}

# --- configure_toolset ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Path,
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
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

    return final_config

# --- Tool Classes ---

# Define schema for StartTerminalTool if needed (even if empty)
class StartTerminalToolArgs(BaseModel):
    pass

class StartTerminalTool(BaseTool):
    name: str = "start_terminal"
    description: str = "Activates the Terminal toolset, enabling execution of shell commands and Python code."
    args_schema: Type[BaseModel] = StartTerminalToolArgs # Use empty schema

    def _run(self, *args, **kwargs) -> str:
        """Activates the 'Terminal' toolset and returns usage instructions."""
        logger.info(f"StartTerminalTool called")

        chat_id = get_current_chat()
        if not chat_id:
            return "Error: No active chat session found."

        from ...chat_state_manager import check_and_configure_toolset
        check_and_configure_toolset(chat_id, toolset_name)

        current_toolsets = get_active_toolsets(chat_id)
        if toolset_name in current_toolsets:
            logger.debug(f"'{toolset_name}' toolset is already active.")
            return f"'{toolset_name}' toolset is already active.\n\n{TERMINAL_TOOLSET_PROMPT}"

        new_toolsets = list(current_toolsets)
        new_toolsets.append(toolset_name)
        update_active_toolsets(chat_id, new_toolsets)

        logger.debug(f"Successfully activated '{toolset_name}' toolset for chat {chat_id}")
        return TERMINAL_TOOLSET_PROMPT

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# Define schema for TerminalTool_HITL
class TerminalToolArgs(BaseModel):
    cmd: str = Field(..., description="The command to execute in the terminal.")

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
    args_schema: Type[BaseModel] = TerminalToolArgs # Use specific schema

    # Make cmd the first argument to match schema
    def _run(self, cmd: str) -> str:
        """Run a command in a shell, prompting the user first via console_io."""
        cmd_to_execute = cmd.strip()
        if not cmd_to_execute:
            # Return error message, don't print directly
            return "Error: Empty command proposed."

        # --- HITL Prompt via console_io ---
        edited_command = console_io.request_tool_edit(
            tool_name=self.name,
            proposed_args={"cmd": cmd_to_execute},
            edit_key="cmd"
            # Default prompt suffix "(edit or confirm) > " is used
        )

        # --- Handle Cancellation ---
        if edited_command is None:
            logger.warning("User cancelled command execution.")
            # Return cancellation message for ToolMessage
            return "Command execution cancelled by user."
        # --- End HITL ---

        # --- Print Confirmation Line ---
        console_io.print_tool_execution_info(
            tool_name=self.name,
            final_args={"cmd": edited_command}
        )
        # --- End Confirmation Line ---

        logger.info(f"Executing terminal command: {edited_command}")
        formatted_result = "" # Initialize result string
        try:
            result = subprocess.run(
                edited_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300 # Add a reasonable timeout (e.g., 5 minutes)
            )

            output_parts = []
            # Include command info in the result for clarity
            output_parts.append(f"Executed: `{edited_command}`")

            if result.returncode != 0:
                output_parts.append(f"Exit Code: {result.returncode}")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                # Limit output length to avoid overwhelming the console/LLM
                max_out_len = 2000
                display_stdout = (stdout[:max_out_len] + "\n... (truncated)") if len(stdout) > max_out_len else stdout
                output_parts.append(f"Output:\n---\n{display_stdout}\n---")
            if stderr:
                max_err_len = 1000
                display_stderr = (stderr[:max_err_len] + "\n... (truncated)") if len(stderr) > max_err_len else stderr
                output_parts.append(f"Errors/Warnings:\n---\n{display_stderr}\n---")

            if not stdout and not stderr and result.returncode == 0:
                output_parts.append("Status: Command completed successfully with no output.")
            elif not stdout and not stderr and result.returncode != 0:
                 output_parts.append("Status: Command failed with no output.")

            formatted_result = "\n".join(output_parts)

        except subprocess.TimeoutExpired:
             logger.error(f"Command '{edited_command}' timed out.")
             formatted_result = f"Error: Command timed out after 300 seconds.\n(Attempted: `{edited_command}`)"
        except FileNotFoundError:
             logger.error(f"Error executing command '{edited_command}': Command not found.")
             formatted_result = f"Error: Command not found: '{edited_command.split()[0]}'. Please ensure it's installed and in the system's PATH.\n(Attempted: `{edited_command}`)"
        except Exception as e:
            logger.error(f"Error executing command '{edited_command}': {e}", exc_info=True)
            formatted_result = f"Error executing command: {str(e)}\n(Attempted: `{edited_command}`)"

        # --- Print Tool Output ---
        console_io.print_tool_output(formatted_result)
        # --- End Print Tool Output ---

        # Return the formatted result for the ToolMessage
        return formatted_result

    # Use run_in_executor for async calls if needed, adjust signature
    async def _arun(self, cmd: str) -> str:
        return await run_in_executor(None, self._run, cmd)


# Define schema for PythonREPLTool_HITL
class PythonREPLToolArgs(BaseModel):
    query: str = Field(..., description="The Python code snippet to execute.")

class PythonREPLTool_HITL(BaseTool):
    """
    Human-in-the-loop wrapper for Python REPL execution.
    """
    name: str = "python_repl"
    description: str = (
        "Evaluates Python code snippets in a REPL environment. "
        "User will be prompted to confirm or edit the code before execution."
    )
    args_schema: Type[BaseModel] = PythonREPLToolArgs # Use specific schema
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True # Keep sanitize option

    # Make query the first argument to match schema
    def _run(
        self,
        query: str
    ) -> str:
        """HITL version of the tool using console_io."""
        code_to_execute = query

        if not code_to_execute:
             # Return error message
             return "Error: No Python code provided to execute."

        # Keep sanitize logic
        if self.sanitize_input:
            original_code = code_to_execute
            code_to_execute = sanitize_input(code_to_execute)
            if original_code != code_to_execute:
                logger.debug(f"Sanitized Python code: {code_to_execute}")

        # --- HITL Prompt via console_io ---
        edited_query = console_io.request_tool_edit(
            tool_name=self.name,
            proposed_args={"query": code_to_execute},
            edit_key="query",
            prompt_suffix="(edit or confirm python code) > " # Custom suffix
        )
        # --- End HITL Prompt ---

        # --- Handle Cancellation ---
        if edited_query is None:
            logger.warning("User cancelled Python code execution.")
            # Return cancellation message
            return "Python code execution cancelled by user."
        # --- End Cancellation ---

        # --- Print Confirmation Line ---
        console_io.print_tool_execution_info(
            tool_name=self.name,
            final_args={"query": edited_query} # Show the final code being executed
        )
        # --- End Confirmation Line ---

        logger.info(f"Executing Python code: {edited_query}")
        formatted_result = "" # Initialize result string
        try:
            # Execute the CONFIRMED/EDITED code
            result = self.python_repl.run(edited_query)
            # Format result consistently
            # Limit result length
            max_res_len = 2000
            result_str = str(result)
            display_result = (result_str[:max_res_len] + "\n... (truncated)") if len(result_str) > max_res_len else result_str
            formatted_result = f"Executed: `python_repl(query='''{edited_query}''')`\nResult:\n---\n{display_result}\n---"
        except Exception as e:
            logger.error(f"Error executing Python code '{edited_query}': {e}", exc_info=True)
            formatted_result = f"Error executing Python code:\n---\n{str(e)}\n---\n(Attempted Code:\n'''\n{edited_query}\n'''\n)"

        # --- Print Tool Output ---
        console_io.print_tool_output(formatted_result)
        # --- End Print Tool Output ---

        # Return the formatted result for the ToolMessage
        return formatted_result

    async def _arun(self, query: str) -> str:
        """Runs the tool asynchronously."""
        return await run_in_executor(None, self._run, query)


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

# --- Direct Execution Helper (remains the same) ---
class TerminalTool_Direct(BaseTool):
    name: str = "_internal_direct_terminal" # Internal name
    description: str = "Internal tool for direct command execution without HITL."
    # Add args_schema for consistency, even if used internally
    class DirectArgs(BaseModel):
        command: str = Field(...)
    args_schema: Type[BaseModel] = DirectArgs

    def _run(self, command: str) -> str:
        logger.info(f"Executing direct command internally: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300 # Add timeout
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
            if not output.strip() and result.returncode == 0: # Check if essentially empty
                output = "Command executed successfully with no output."
            elif not output.strip() and result.returncode != 0:
                 output = f"Command failed with no output. Exit Code: {result.returncode}"

            return output.strip()
        except subprocess.TimeoutExpired:
             logger.error(f"Direct execution timed out for '{command}'")
             return f"Error: Command timed out after 300 seconds.\n(Attempted: `{command}`)"
        except Exception as e:
            logger.error(f"Direct execution failed for '{command}': {e}", exc_info=True)
            return f"Error executing command: {str(e)}"

    async def _arun(self, command: str) -> str:
         return await run_in_executor(None, self._run, command)

_direct_terminal_tool_instance = TerminalTool_Direct()

def run_direct_terminal_command(command: str) -> str:
    """Function to run command using the internal direct tool"""
    return _direct_terminal_tool_instance._run(command=command)