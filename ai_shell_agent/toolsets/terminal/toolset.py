# ai_shell_agent/toolsets/terminal/toolset.py
"""
Defines the tools and metadata for the Terminal toolset.
"""
import subprocess
from typing import Dict, List, Optional, Any, Union, Type # Add Type
from pathlib import Path # Keep Path import

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, Field # Import BaseModel and Field

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

# --- Tool Input Schemas ---
class NoArgsSchema(BaseModel):
    """Input schema for tools that require no arguments."""
    pass

class CommandSchema(BaseModel):
    """Input schema for tools accepting a shell command."""
    command: str = Field(description="The command to execute in the terminal.")

class QuerySchema(BaseModel):
    """Input schema for tools accepting a query (like Python code)."""
    query: str = Field(description="The query or code snippet to execute.")

# --- Tool Classes (MODIFIED) ---

class StartTerminalTool(BaseTool):
    name: str = "start_terminal"
    description: str = "Activates the Terminal toolset, enabling execution of shell commands and Python code."
    args_schema: Type[BaseModel] = NoArgsSchema # Specify schema

    # Removed *args, **kwargs from signature as schema defines no args
    def _run(self) -> str:
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
        return f"'{toolset_name}' toolset activated.\n\n{TERMINAL_TOOLSET_PROMPT}"

    # Removed *args, **kwargs from signature
    async def _arun(self) -> str:
        """Run the tool asynchronously."""
        return self._run()

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
    args_schema: Type[BaseModel] = CommandSchema # Specify schema

    # Use explicit 'command' arg based on schema, keep **kwargs for potential framework args
    def _run(self, command: str, **kwargs) -> str:
        """Execute a command in the terminal shell with human verification."""
        cmd_to_execute = command # Argument directly from schema

        if not cmd_to_execute:
            # This case should ideally not happen if schema is required
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
            # Execute the command
            result = subprocess.run(
                edited_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Format the output
            output_parts = []
            output_parts.append(f"Executed: `{edited_command}`")
            if result.returncode != 0:
                output_parts.append(f"Exit Code: {result.returncode}")
                
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if stdout:
                output_parts.append(f"Output:\n---\n{stdout}\n---")
            if stderr:
                output_parts.append(f"Errors/Warnings:\n---\n{stderr}\n---")
            
            if not stdout and not stderr:
                status = "Command completed successfully with no output." if result.returncode == 0 else "Command failed with no output."
                output_parts.append(f"Status: {status}")
                
            return "\n".join(output_parts)

        except FileNotFoundError:
             logger.error(f"Error executing command '{edited_command}': Command not found.")
             return f"Error: Command not found: '{edited_command.split()[0]}'. Please ensure it's installed and in the system's PATH."
        except Exception as e:
            logger.error(f"Error executing command '{edited_command}': {e}", exc_info=True)
            return f"Error executing command: {str(e)}\n(Attempted: `{edited_command}`)"

    # Use explicit 'command' arg based on schema
    async def _arun(self, command: str, **kwargs) -> str:
        return await run_in_executor(None, self._run, command=command, **kwargs)


class PythonREPLTool_HITL(BaseTool):
    """
    Human-in-the-loop wrapper for Python REPL execution.
    """
    name: str = "python_repl"
    description: str = (
        "Evaluates Python code snippets in a REPL environment. "
        "User will be prompted to confirm or edit the code before execution."
    )
    args_schema: Type[BaseModel] = QuerySchema # Specify schema
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True # Keep sanitize option

    # Use explicit 'query' arg based on schema
    def _run(
        self,
        query: str,
        **kwargs # Keep kwargs for potential framework args
    ) -> str:
        """HITL version of the tool."""
        code_to_execute = query # Argument directly from schema

        if not code_to_execute:
             # Should not happen if schema is required
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
            return f"Executed: `python_repl(query='''{edited_query}''')`\nResult:\n---\n{str(result)}\n---"
        except Exception as e:
            logger.error(f"Error executing Python code '{edited_query}': {e}", exc_info=True)
            return f"Error executing Python code:\n---\n{str(e)}\n---\n(Attempted Code:\n'''\n{edited_query}\n'''\n)"

    # Use explicit 'query' arg based on schema
    async def _arun(
        self,
        query: str,
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return await run_in_executor(None, self._run, query=query, **kwargs)

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
class TerminalTool_Direct:
    """
    Direct terminal execution without HITL prompts.
    Not a BaseTool as it's not exposed to the LLM.
    """
    def run(self, command: str) -> str:
        """Execute a command directly in the terminal shell."""
        if not command:
            return "Error: No command provided."
            
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Format the output like the HITL tool
            output_parts = []
            output_parts.append(f"Executed: `{command}`")
            if result.returncode != 0:
                output_parts.append(f"Exit Code: {result.returncode}")
                
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if stdout:
                output_parts.append(f"Output:\n---\n{stdout}\n---")
            if stderr:
                output_parts.append(f"Errors/Warnings:\n---\n{stderr}\n---")
                
            if not stdout and not stderr:
                output_parts.append(f"Status: Command completed with no output.")
                
            return "\n".join(output_parts)
            
        except FileNotFoundError:
            return f"Error: Command not found: '{command.split()[0]}'. Please ensure it's installed and in the system's PATH."
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}", exc_info=True)
            return f"Error executing command: {str(e)}"

def run_direct_terminal_command(command: str) -> str:
    """Execute a command directly, used by execute() in chat_manager."""
    return TerminalTool_Direct().run(command)