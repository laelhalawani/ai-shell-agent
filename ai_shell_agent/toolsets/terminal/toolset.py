# ai_shell_agent/toolsets/terminal/toolset.py
"""
Defines the tools and metadata for the Terminal toolset.
"""
import subprocess
import os
import shutil
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
    check_and_configure_toolset
)
from ...console_manager import get_console_manager # Import console manager
from ...errors import PromptNeededError # Import the new custom exceptions
from ...texts import get_text as get_main_text # Import main get_text function
from .prompts import TERMINAL_TOOLSET_PROMPT # Import the prompt content
from ...utils.file_io import write_json, read_json # Import utils for JSON I/O
from .texts import get_text # Import toolset-specific get_text

# --- Toolset Metadata ---
toolset_name = get_text("toolset.name")
toolset_id = "terminal"
toolset_description = get_text("toolset.description")
toolset_required_secrets: Dict[str, str] = {}

# --- Get console manager instance ---
console = get_console_manager()

# --- configure_toolset ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path],
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
) -> Dict:
    """
    Configuration function for the Terminal toolset. Terminal currently needs
    no specific configuration.
    """
    final_config = {}
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat"

    # --- MODIFICATION START: Print context header ---
    console.display_message(
        get_main_text("common.labels.system"),
        f"Configure Terminal settings ({context_name})",
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )
    # --- MODIFICATION END ---

    logger.info(f"Configuring Terminal toolset ({context_name}). No user-configurable settings currently.")
    # --- MODIFICATION START: Use main get_text for label ---
    console.display_message(
        get_main_text("common.labels.info"),
        get_text("config.info_no_settings"),
        console.STYLE_INFO_LABEL,
        console.STYLE_INFO_CONTENT
    )
    # --- MODIFICATION END ---

    save_success_global = True
    save_success_local = True

    try:
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(global_config_path, final_config)
        logger.debug(f"Wrote empty config for Terminal toolset to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to write empty config for Terminal toolset to global path {global_config_path}: {e}")

    if not is_global_only and local_config_path: # Check local_config_path exists
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.debug(f"Wrote empty config for Terminal toolset to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to write empty config for Terminal toolset to local path {local_config_path}: {e}")

    if not save_success_global or not save_success_local:
        # --- MODIFICATION START: Use main get_text for label ---
        console.display_message(
            get_main_text("common.labels.warning"),
            get_text("config.warn_write_failed"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        # --- MODIFICATION END ---

    return final_config

# --- Tool Classes ---

class StartTerminalToolArgs(BaseModel): pass # Keep schema definition

class TerminalUsageGuideTool(BaseTool):
    name: str = get_text("tools.usage_guide.name")
    description: str = get_text("tools.usage_guide.description")
    args_schema: Type[BaseModel] = StartTerminalToolArgs

    def _run(self) -> str:
        logger.debug(f"TerminalUsageGuideTool invoked.")
        return TERMINAL_TOOLSET_PROMPT

    async def _arun(self) -> str:
        return self._run()

class TerminalToolArgs(BaseModel):
    cmd: str = Field(..., description=get_text("schemas.terminal.cmd_desc"))

class TerminalTool_HITL(BaseTool):
    """
    Tool for interacting with the system's shell with human-in-the-loop confirmation.
    """
    name: str = get_text("tools.terminal_hitl.name")
    description: str = get_text("tools.terminal_hitl.description")
    args_schema: Type[BaseModel] = TerminalToolArgs
    requires_confirmation: bool = True

    # --- MODIFICATION: Change parameter name to 'cmd' to match schema ---
    def _run(self, cmd: str, confirmed_input: Optional[str] = None) -> str:
        """
        Run a command in a shell. Raises PromptNeededError if confirmation needed.
        """
        cmd_to_execute = cmd.strip() # Use the 'cmd' parameter
        if not cmd_to_execute:
            return get_text("tools.terminal_hitl.error_empty_cmd")

        if confirmed_input is None:
            logger.debug(f"TerminalTool: Raising PromptNeededError for cmd: '{cmd_to_execute}'")
            logger.debug("The AI wants to run a shell command.")
            # Display command for confirmation via logger/console handled by chat_manager now if needed
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"cmd": cmd_to_execute}, # Use 'cmd' key
                edit_key="cmd", # Edit the 'cmd' key
                # --- MODIFICATION: Add explicit suffix ---
                prompt_suffix=get_text("tools.terminal_hitl.confirm_suffix")
            )
        else:
            final_command = confirmed_input.strip()
            if not final_command:
                logger.warning("TerminalTool: Received empty confirmed input.")
                return get_text("tools.terminal_hitl.error_empty_confirmed")

            logger.info(f"Executing confirmed terminal command: {final_command}")
            formatted_result = ""
            details_parts = []
            try:
                # Determine shell based on OS
                shell_executable = None
                shell_flag = True # Use shell=True by default
                if os.name == 'nt': # Windows
                    # Prefer powershell if available, fallback to cmd
                    if shutil.which('powershell'):
                        shell_executable = shutil.which('powershell')
                    elif shutil.which('cmd'):
                        shell_executable = shutil.which('cmd')
                    # No need to set shell=False when executable is specified
                elif os.name == 'posix': # Linux, macOS, etc.
                    # Use default shell (usually bash or zsh)
                     pass # shell=True is sufficient

                result = subprocess.run(
                    final_command,
                    shell=shell_flag,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300,
                    executable=shell_executable # Pass specific shell if found
                )

                if result.returncode != 0:
                     details_parts.append(get_text("tools.terminal_hitl.details.exit_code", code=result.returncode))
                stdout = result.stdout.strip(); stderr = result.stderr.strip()
                if stdout:
                    max_out_len = 2000
                    trunc_marker = "\\n... (truncated)" # Default truncation marker
                    display_stdout = (stdout[:max_out_len] + trunc_marker) if len(stdout) > max_out_len else stdout
                    details_parts.append(get_text("tools.terminal_hitl.details.stdout", output=display_stdout))
                if stderr:
                    max_err_len = 1000
                    trunc_marker = "\\n... (truncated)" # Default truncation marker
                    display_stderr = (stderr[:max_err_len] + trunc_marker) if len(stderr) > max_err_len else stderr
                    details_parts.append(get_text("tools.terminal_hitl.details.stderr", output=display_stderr))
                if not stdout and not stderr:
                    status_msg = get_text("tools.terminal_hitl.details.no_output_success") if result.returncode == 0 else get_text("tools.terminal_hitl.details.no_output_fail")
                    details_parts.append(status_msg)

                formatted_result = get_text("tools.terminal_hitl.result_format", command=final_command, details="\\n".join(details_parts))

            except subprocess.TimeoutExpired:
                 logger.error(f"Command '{final_command}' timed out.")
                 formatted_result = get_text("tools.terminal_hitl.error_timeout", command=final_command)
            except FileNotFoundError:
                 logger.error(f"Error executing command '{final_command}': Command not found.")
                 # Try to get just the command name
                 command_name = final_command.split()[0] if final_command else final_command
                 formatted_result = get_text("tools.terminal_hitl.error_not_found", command_name=command_name, command=final_command)
            except Exception as e:
                logger.error(f"Error executing command '{final_command}': {e}", exc_info=True)
                formatted_result = get_text("tools.terminal_hitl.error_generic", error=str(e), command=final_command)

            return formatted_result

    async def _arun(self, cmd: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, cmd, confirmed_input)

class PythonREPLToolArgs(BaseModel):
    query: str = Field(..., description=get_text("schemas.python.query_desc"))

class PythonREPLTool_HITL(BaseTool):
    """
    Human-in-the-loop wrapper for Python REPL execution.
    """
    name: str = get_text("tools.python_repl_hitl.name")
    description: str = get_text("tools.python_repl_hitl.description")
    args_schema: Type[BaseModel] = PythonREPLToolArgs
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True
    requires_confirmation: bool = True

    def _run(
        self,
        query: str,
        confirmed_input: Optional[str] = None
    ) -> str:
        """
        Evaluates Python code. Raises PromptNeededError if confirmation needed.
        """
        code_to_execute = query
        if not code_to_execute:
             return get_text("tools.python_repl_hitl.error_no_code")

        if self.sanitize_input:
            original_code = code_to_execute
            code_to_execute = sanitize_input(code_to_execute)
            if original_code != code_to_execute:
                 logger.debug(f"Sanitized Python code: {code_to_execute}")

        if confirmed_input is None:
            logger.debug(f"PythonREPLTool: Raising PromptNeededError for query: '{code_to_execute[:50]}...'")
            logger.debug("The AI wants to run a Python code snippet.")
            # Display code for confirmation via logger/console handled by chat_manager now
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"query": code_to_execute},
                edit_key="query",
                # --- MODIFICATION: Add explicit suffix ---
                prompt_suffix=get_text("tools.python_repl_hitl.confirm_suffix")
            )
        else:
            final_query = confirmed_input
            if not final_query.strip():
                 logger.warning("PythonREPLTool: Received empty confirmed input.")
                 return get_text("tools.python_repl_hitl.error_empty_confirmed")

            logger.info(f"Executing confirmed Python code: {final_query[:100]}...")
            formatted_result = ""
            try:
                result = self.python_repl.run(final_query)
                max_res_len = 2000
                result_str = str(result)
                trunc_marker = "\\n... (truncated)" # Default truncation marker
                display_result = (result_str[:max_res_len] + trunc_marker) if len(result_str) > max_res_len else result_str
                formatted_result = get_text("tools.python_repl_hitl.result_format", query=final_query, result=display_result)
            except Exception as e:
                logger.error(f"Error executing Python code '{final_query}': {e}", exc_info=True)
                formatted_result = get_text("tools.python_repl_hitl.error_generic", error=str(e), query=final_query)

            return formatted_result

    async def _arun(self, query: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, query, confirmed_input)

# --- Tool Instances ---
terminal_usage_guide_tool = TerminalUsageGuideTool()
terminal_tool = TerminalTool_HITL()
python_repl_tool = PythonREPLTool_HITL()

# --- Toolset Definition ---
toolset_tools: List[BaseTool] = [
    terminal_usage_guide_tool,
    terminal_tool,
    python_repl_tool,
]

# --- Register Tools ---
register_tools(toolset_tools)
logger.debug(f"Terminal toolset tools registered: {[t.name for t in toolset_tools]}")

# --- Direct Execution Helper ---
class TerminalTool_Direct(BaseTool):
    name: str = get_text("tools.direct_terminal.name")
    description: str = get_text("tools.direct_terminal.description")

    class DirectArgs(BaseModel):
        command: str = Field(..., description=get_text("schemas.direct_terminal.cmd_desc"))

    args_schema: Type[BaseModel] = DirectArgs

    def _run(self, command: str) -> str:
        logger.info(f"Executing direct command internally: {command}")
        output_parts = []
        try:
            shell_executable = None
            shell_flag = True
            if os.name == 'nt':
                if shutil.which('powershell'): shell_executable = shutil.which('powershell')
                elif shutil.which('cmd'): shell_executable = shutil.which('cmd')
            elif os.name == 'posix': pass

            result = subprocess.run(
                command,
                shell=shell_flag,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
                executable=shell_executable
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(get_text("tools.direct_terminal.result_details.stdout", output=stdout))
            if stderr:
                output_parts.append(get_text("tools.direct_terminal.result_details.stderr", output=stderr))
            if result.returncode != 0:
                 output_parts.append(get_text("tools.direct_terminal.result_details.exit_code", code=result.returncode))

            if not output_parts: # Check if list is empty
                 if result.returncode == 0:
                      output_parts.append(get_text("tools.direct_terminal.result_details.no_output_success"))
                 else:
                      output_parts.append(get_text("tools.direct_terminal.result_details.no_output_fail", code=result.returncode))

            return "".join(output_parts).strip() # Join parts

        except subprocess.TimeoutExpired:
             logger.error(f"Direct execution timed out for '{command}'")
             return get_text("tools.direct_terminal.error_timeout", command=command)
        except Exception as e:
            logger.error(f"Direct execution failed for '{command}': {e}", exc_info=True)
            return get_text("tools.direct_terminal.error_generic", error=str(e))

    async def _arun(self, command: str) -> str:
         return await run_in_executor(None, self._run, command)

_direct_terminal_tool_instance = TerminalTool_Direct()

def run_direct_terminal_command(command: str) -> str:
    """Function to run command using the internal direct tool"""
    return _direct_terminal_tool_instance._run(command=command)
