# ai_shell_agent/toolsets/terminal/toolset.py
"""
Defines the tools and metadata for the Terminal toolset.
"""
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path # Import Path

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from pydantic import Field # BaseModel removed as not directly used here

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
    _write_json # Import helper for configuration
    # No need for _update_message_in_chat here anymore
)
# Import the prompt content to be returned by the start tool
from .prompts import TERMINAL_TOOLSET_PROMPT

# --- Toolset Metadata ---
toolset_name = "Terminal"
toolset_description = "Provides tools to execute shell commands and Python code."

# --- Configuration ---
toolset_config_defaults = {} # No specific config needed yet

def configure_toolset(config_path: Path, current_config: Optional[Dict]) -> Dict:
    """Configuration function for the Terminal toolset."""
    # Terminal currently needs no configuration.
    # We still write an empty config file to mark it as configured.
    logger.info(f"Terminal toolset requires no specific configuration for path: {config_path}")
    new_config = {} # Empty config
    try:
        _write_json(config_path, new_config)
        logger.debug(f"Wrote empty config for Terminal toolset to {config_path}")
    except Exception as e:
         logger.error(f"Failed to write empty config for Terminal toolset to {config_path}: {e}")
         # Return current_config or empty dict if write fails? Let's return new_config anyway.
    return new_config # Return the (empty) config that was intended

# --- Tool Classes (Migrated from terminal_tools.py) ---

class StartTerminalTool(BaseTool):
    name: str = "start_terminal"
    description: str = "Activates the Terminal toolset, enabling execution of shell commands and Python code."

    def _run(self, *args, **kwargs) -> str:
        """Activates the 'Terminal' toolset and returns usage instructions."""
        logger.info(f"StartTerminalTool called")

        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."

        current_toolsets = get_active_toolsets(chat_file)
        if toolset_name in current_toolsets:
            logger.debug(f"'{toolset_name}' toolset is already active.")
            # Maybe return a shorter message if already active?
            return f"'{toolset_name}' toolset is already active.\n\n{TERMINAL_TOOLSET_PROMPT}"

        # Activate the toolset
        new_toolsets = list(current_toolsets)
        new_toolsets.append(toolset_name)
        update_active_toolsets(chat_file, new_toolsets) # Save updated toolsets list

        # System prompt update is handled implicitly by llm.py using the new active_toolsets state.
        # No need to manually update message 0 here.

        logger.debug(f"Successfully activated '{toolset_name}' toolset for chat {chat_file}")
        # Return the instructional prompt as the ToolMessage content
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

    def _run(self, command: str = None, **kwargs) -> str:
        """Execute a command in the terminal shell with human verification."""
        cmd_to_execute = command

        # Check for 'command' in kwargs (used by agent invoke)
        if cmd_to_execute is None and 'command' in kwargs:
            cmd_to_execute = kwargs.get('command', '')
        # Check for v__args format from LangChain (less common now with direct dict passing)
        elif cmd_to_execute is None and 'v__args' in kwargs:
             wrapped_args = kwargs.get('v__args', [])
             if isinstance(wrapped_args, list) and wrapped_args:
                 cmd_to_execute = wrapped_args[0] if isinstance(wrapped_args[0], str) else str(wrapped_args[0])


        if not cmd_to_execute:
            return "Error: No command provided to execute."

        logger.info(f"Proposing terminal command: {cmd_to_execute}")
        print(f"\n[Proposed terminal command]:")
        try:
            edited_command = prompt("(Accept or Edit) > ", default=cmd_to_execute)
        except EOFError:
             logger.warning("User cancelled command input (EOF).")
             return "Command input cancelled by user."

        if not edited_command.strip():
            logger.warning("User cancelled command by submitting empty input.")
            return "Command cancelled by user (empty input)."

        logger.info(f"Executing terminal command: {edited_command}")
        try:
            result = subprocess.run(
                edited_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace' # Handle potential encoding errors gracefully
            )

            output_parts = []
            # Add command info first
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
             return f"Error: Command not found: '{edited_command.split()[0]}'. Please ensure it's installed and in the system's PATH."
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
        query: str = None, # Renamed 'command' to 'query' for consistency with PythonREPL
        #run_manager: Optional[CallbackManagerForToolRun] = None, # Often unused directly
        **kwargs
    ) -> str: # Changed return type hint to str
        """HITL version of the tool."""
        code_to_execute = query

        # Check for 'query' in kwargs (used by agent invoke)
        if code_to_execute is None and 'query' in kwargs:
             code_to_execute = kwargs.get('query', '')
        # Check for 'command' for backward compatibility if needed? maybe not.
        elif code_to_execute is None and 'command' in kwargs:
            logger.warning("Received 'command' argument for python_repl, expected 'query'. Using 'command'.")
            code_to_execute = kwargs.get('command', '')
        elif code_to_execute is None and 'v__args' in kwargs:
             wrapped_args = kwargs.get('v__args', [])
             if isinstance(wrapped_args, list) and wrapped_args:
                 code_to_execute = wrapped_args[0] if isinstance(wrapped_args[0], str) else str(wrapped_args[0])


        if not code_to_execute:
            return "Error: No Python code provided to execute."

        logger.info(f"Proposing Python code: {code_to_execute}")

        if self.sanitize_input:
            original_code = code_to_execute
            code_to_execute = sanitize_input(code_to_execute)
            if original_code != code_to_execute:
                logger.debug(f"Sanitized Python code: {code_to_execute}")


        print(f"\n[Proposed Python code]:")
        try:
            edited_query = prompt("(Accept or Edit) > ", default=code_to_execute)
        except EOFError:
             logger.warning("User cancelled code input (EOF).")
             return "Code input cancelled by user."


        if not edited_query.strip():
            logger.warning("User cancelled code execution by submitting empty input.")
            return "Code execution cancelled by user (empty input)."

        logger.info(f"Executing Python code: {edited_query}")
        try:
            result = self.python_repl.run(edited_query)
            # Ensure result is string
            return f"Executed: `python_repl(query='''{edited_query}''')`\nResult:\n---\n{str(result)}\n---"
        except Exception as e:
            logger.error(f"Error executing Python code '{edited_query}': {e}", exc_info=True)
            # Provide more context in the error message
            return f"Error executing Python code:\n---\n{str(e)}\n---\n(Attempted Code:\n'''\n{edited_query}\n'''\n)"

    async def _arun(
        self,
        query: str = None,
        #run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
         **kwargs
    ) -> str: # Changed return type hint to str
        """Use the tool asynchronously."""
        return await run_in_executor(None, self._run, query=query, **kwargs) # Pass query explicitly

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