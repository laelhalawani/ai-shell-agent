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

# Local imports
from ... import logger
from ...tool_registry import register_tools
from ...chat_state_manager import (
    get_current_chat,
    get_active_toolsets,
    update_active_toolsets,
)
# Import console manager instead of console_io
from ...console_manager import get_console_manager
# Import the new custom exceptions
from ...errors import PromptNeededError

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

# --- Get console manager instance ---
console = get_console_manager()

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
        console.display_message("WARNING:", "Failed to write configuration files for Terminal toolset. Check logs.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)

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
    requires_confirmation: bool = True # Mark this tool as requiring HITL

    # Make cmd the first argument to match schema, add confirmed_input for HITL support
    def _run(self, cmd: str, confirmed_input: Optional[str] = None) -> str:
        """
        Run a command in a shell. Raises PromptNeededError if confirmation needed.
        Executes directly if confirmed_input is provided.
        """
        cmd_to_execute = cmd.strip()
        if not cmd_to_execute:
            return "Error: Empty command proposed."

        if confirmed_input is None:
            # First call: Raise error to request prompt from chat_manager loop
            logger.debug(f"TerminalTool: Raising PromptNeededError for cmd: '{cmd_to_execute}'")
            console.display_message("WARNING:", "The AI wants to run a shell command:", 
                                    console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            console.display_message("COMMAND:", cmd_to_execute, console.STYLE_COMMAND_LABEL, console.STYLE_COMMAND_CONTENT)
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"cmd": cmd_to_execute},
                edit_key="cmd"
            )
        else:
            # Second call: Input has been confirmed by the user
            final_command = confirmed_input.strip() # Use the confirmed input
            if not final_command:
                logger.warning("TerminalTool: Received empty confirmed input.")
                return "Error: Confirmed command is empty."

            logger.info(f"Executing confirmed terminal command: {final_command}")
            # --- Actual Execution Logic ---
            formatted_result = ""
            try:
                result = subprocess.run(
                    final_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300
                )
                # Format the result
                output_parts = []
                output_parts.append(f"Executed: `{final_command}`")
                if result.returncode != 0: output_parts.append(f"Exit Code: {result.returncode}")
                stdout = result.stdout.strip(); stderr = result.stderr.strip()
                if stdout:
                    max_out_len = 2000
                    display_stdout = (stdout[:max_out_len] + "\n... (truncated)") if len(stdout) > max_out_len else stdout
                    output_parts.append(f"Output:\n---\n{display_stdout}\n---")
                if stderr:
                    max_err_len = 1000
                    display_stderr = (stderr[:max_err_len] + "\n... (truncated)") if len(stderr) > max_err_len else stderr
                    output_parts.append(f"Errors/Warnings:\n---\n{display_stderr}\n---")
                if not stdout and not stderr:
                    status_msg = "Command completed successfully with no output." if result.returncode == 0 else "Command failed with no output."
                    output_parts.append(f"Status: {status_msg}")

                formatted_result = "\n".join(output_parts)

            except subprocess.TimeoutExpired:
                 logger.error(f"Command '{final_command}' timed out.")
                 formatted_result = f"Error: Command timed out after 300 seconds.\n(Attempted: `{final_command}`)"
            except FileNotFoundError:
                 logger.error(f"Error executing command '{final_command}': Command not found.")
                 formatted_result = f"Error: Command not found: '{final_command.split()[0]}'. Ensure it's installed/in PATH.\n(Attempted: `{final_command}`)"
            except Exception as e:
                logger.error(f"Error executing command '{final_command}': {e}", exc_info=True)
                formatted_result = f"Error executing command: {str(e)}\n(Attempted: `{final_command}`)"
            # --- End Execution Logic ---

            # Return the result string (ConsoleManager will display it)
            return formatted_result

    async def _arun(self, cmd: str, confirmed_input: Optional[str] = None) -> str:
        """
        Async version of _run that delegates to the synchronous implementation
        """
        return await run_in_executor(None, self._run, cmd, confirmed_input)

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
    requires_confirmation: bool = True # Mark this tool as requiring HITL

    # Make query the first argument to match schema, add confirmed_input for HITL support
    def _run(
        self,
        query: str,
        confirmed_input: Optional[str] = None
    ) -> str:
        """
        Evaluates Python code. Raises PromptNeededError if confirmation needed.
        Executes directly if confirmed_input is provided.
        """
        code_to_execute = query
        if not code_to_execute:
             return "Error: No Python code provided to execute."

        # Apply sanitization on the *initial* proposal
        if self.sanitize_input:
            original_code = code_to_execute
            code_to_execute = sanitize_input(code_to_execute)
            # Log only if changed
            if original_code != code_to_execute:
                 logger.debug(f"Sanitized Python code: {code_to_execute}")


        if confirmed_input is None:
            # First call: Raise error to request prompt
            logger.debug(f"PythonREPLTool: Raising PromptNeededError for query: '{code_to_execute[:50]}...'")
            console.display_message("WARNING:", "The AI wants to run a Python code snippet:", 
                                    console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            console.display_message("CODE:", code_to_execute, console.STYLE_COMMAND_LABEL, console.STYLE_COMMAND_CONTENT)
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"query": code_to_execute}, # Pass potentially sanitized code
                edit_key="query",
                prompt_suffix="(edit or confirm python code) > "
            )
        else:
            # Second call: Input has been confirmed
            final_query = confirmed_input # Use the raw confirmed input (user might bypass sanitization)
            if not final_query.strip():
                 logger.warning("PythonREPLTool: Received empty confirmed input.")
                 return "Error: Confirmed Python code is empty."

            logger.info(f"Executing confirmed Python code: {final_query[:100]}...")
            # --- Actual Execution Logic ---
            formatted_result = ""
            try:
                result = self.python_repl.run(final_query)
                # Format result
                max_res_len = 2000
                result_str = str(result)
                display_result = (result_str[:max_res_len] + "\n... (truncated)") if len(result_str) > max_res_len else result_str
                formatted_result = f"Executed: `python_repl(query='''{final_query}''')`\nResult:\n---\n{display_result}\n---"
            except Exception as e:
                logger.error(f"Error executing Python code '{final_query}': {e}", exc_info=True)
                formatted_result = f"Error executing Python code:\n---\n{str(e)}\n---\n(Attempted Code:\n'''\n{final_query}\n'''\n)"
            # --- End Execution Logic ---

            # Return the result string
            return formatted_result

    async def _arun(self, query: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, query, confirmed_input)


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