"""
Tools for AI Shell Agent.

Contains tool classes and utilities for implementing commands
and functions available to the AI agent.
"""
import os
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Langchain imports
from langchain.tools import BaseTool
# Add missing imports from the original tool.py file
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
# Fix the Pydantic imports - Field should include default_factory
from pydantic import BaseModel, Field

# Import the sanitize_input function directly from the module
from langchain_experimental.tools.python.tool import sanitize_input, _get_default_python_repl

# External imports
from prompt_toolkit import prompt

# Local imports
from . import logger
from .tool_registry import register_tools
# --- NEW: Import state manager and prompt builder ---
from .chat_state_manager import (
    get_current_chat,
    get_active_toolsets,
    update_active_toolsets,
    _read_chat_data,
    _write_chat_data,
    _update_message_in_chat, # For updating system prompt
)
from .prompts.prompts import build_prompt

# --- Tool Classes ---

class StartTerminalTool(BaseTool):
    name: str = "start_terminal"
    description: str = "Starts a terminal session and enables executing commands on the user's system."
    
    def _run(self, args: str = "", **kwargs) -> str:
        """Start the command shell by activating the 'Terminal' toolset."""
        # Extract args from either positional args or v__args wrapping
        command_args = None
        if args:
            command_args = args
        elif 'v__args' in kwargs:
            # Handle LangChain's v__args wrapping
            wrapped_args = kwargs.get('v__args', [])
            if isinstance(wrapped_args, list) and wrapped_args:
                command_args = wrapped_args
        
        # Convert to string for logging
        args_str = ''
        if command_args:
            if isinstance(command_args, (list, tuple)):
                args_str = ' '.join(map(str, command_args))
            else:
                args_str = str(command_args)
        
        logger.info(f"StartTerminalTool called with args: {command_args}")
        
        chat_file = get_current_chat()
        if not chat_file:
            return "Error: No active chat session found."
        
        toolset_name = "Terminal"
        
        # --- Toolset Activation & Prompt Update ---
        current_toolsets = get_active_toolsets(chat_file)
        if toolset_name in current_toolsets:
            logger.debug(f"'{toolset_name}' toolset is already active, no action needed")
            return f"'{toolset_name}' toolset is already active."
            
        new_toolsets = list(current_toolsets)
        new_toolsets.append(toolset_name)
        update_active_toolsets(chat_file, new_toolsets) # Save updated toolsets list
        
        # Re-build the system prompt with the new set of active toolsets
        new_system_prompt = build_prompt(active_toolsets=new_toolsets)
        
        # Update the system prompt message in the chat history (message 0)
        _update_message_in_chat(chat_file, 0, {"role": "system", "content": new_system_prompt})
        
        logger.debug(f"Successfully activated '{toolset_name}' toolset for chat {chat_file}")
        return f"'{toolset_name}' toolset activated. Use the `terminal` command to run commands."
            
    async def _arun(self, *args, **kwargs) -> str:
        """Run the tool asynchronously."""
        return self._run(*args, **kwargs)

class TerminalTool_HITL(BaseTool):
    """
    Tool for interacting with the Windows shell with human-in-the-loop confirmation.
    This tool is safer as it requires user confirmation before execution.
    """
    name: str = "terminal"
    description: str = """Executes a command in the system's terminal."""
    
    def _run(self, command: str = None, **kwargs) -> str:
        """Execute a command in the terminal shell with human verification."""
        # Handle different argument formats
        cmd_to_execute = command
        
        # Check for v__args format from LangChain
        if command is None and 'v__args' in kwargs:
            wrapped_args = kwargs.get('v__args', [])
            if isinstance(wrapped_args, list) and wrapped_args:
                if isinstance(wrapped_args[0], str):
                    cmd_to_execute = wrapped_args[0]
                else:
                    cmd_to_execute = str(wrapped_args[0])
        
        # If still None, check if command is in kwargs (used by invoke)
        if cmd_to_execute is None and 'command' in kwargs:
            cmd_to_execute = kwargs.get('command', '')
            
        # Return early if no command provided
        if not cmd_to_execute:
            return "No command provided to execute."
            
        # Show proposed command and allow user to edit it
        print(f"\n[Proposed terminal command]:")
        edited_command = prompt("(Accept or Edit) > ", default=cmd_to_execute)
        
        # If user provided an empty command, consider it a cancellation
        if not edited_command.strip():
            return "Command cancelled by user."
            
        try:
            # Use subprocess with shell=True for Windows shell commands
            result = subprocess.run(
                edited_command, 
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # Format the output in a cleaner way
            output_parts = []
            
            # Only add return code if it's not 0 (indicating an error)
            if result.returncode != 0:
                output_parts.append(f"Command completed with exit code {result.returncode}")
            
            # Combine stdout and stderr into a single output string
            if result.stdout and result.stdout.strip():
                output_parts.append(result.stdout.strip())
            
            if result.stderr and result.stderr.strip():
                # If we have both stdout and stderr, add a small separator
                if output_parts and result.stdout and result.stdout.strip():
                    output_parts.append("---")
                output_parts.append(result.stderr.strip())
            
            # If no output was captured but command succeeded
            if not output_parts and result.returncode == 0:
                output_parts.append("Command completed successfully.")
                
            # Add command info to help with debugging
            command_info = f"Executed: {edited_command}"
            output_parts.append(f"\n({command_info})")
                
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error executing command: {str(e)}\n\nAttempted to run: {edited_command}"
    
    async def _arun(self, command: str = None, **kwargs) -> str:
        """Async implementation simply calls the sync version."""
        return self._run(command, **kwargs)
    
    def invoke(self, args: Dict) -> str:
        """Invoke the tool with the given arguments."""
        command = args.get("command", "")
        if not command:
            return "No command provided."
        return self._run(command)

class TerminalTool_Direct(BaseTool):
    """
    Tool for directly interacting with the Windows shell without confirmation.
    This is used for internal commands that don't need user confirmation.
    """
    name: str = "shell_windows_direct"
    description: str = """Executes a command in the system's terminal. Use this for cmd/terminal/console/shell commands. First run commands to gather information."""
    
    def _run(self, command: str) -> str:
        """Execute a command in the Windows shell directly."""
        try:
            # Use subprocess with shell=True for Windows shell commands
            result = subprocess.run(
                command, 
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            output = result.stdout
            error = result.stderr
            
            # Combine output and error if both exist
            if output and error:
                return f"{output}\n\nErrors/Warnings:\n{error}"
            elif error:
                return f"Command completed with messages:\n{error}"
            else:
                return output
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def invoke(self, args: Dict) -> str:
        """Invoke the tool with the given arguments."""
        command = args.get("command", "")
        if not command:
            return "No command provided."
        return self._run(command)

class PythonREPLTool_HITL(BaseTool):
    """
    Human-in-the-loop wrapper for Python REPL execution.
    Allows the user to review and modify Python code before execution.
    """
    name: str = "python_repl"
    description: str = (
        "Evaluates Python code in a REPL environment. "
    )
    
    # Create the attributes that match the original PythonREPLTool
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True
    
    # No need for original_tool now that we directly use python_repl
    
    def _run(
        self,
        query: str = None,
        #run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """
        HITL version of the tool that allows user to review and edit code before execution.
        
        Args:
            query: Python code string to execute
            run_manager: Optional callback manager
            **kwargs: Additional arguments including v__args
            
        Returns:
            The output from executing the Python code
        """
        # Handle different argument formats
        code_to_execute = query
        
        # Check for v__args format from LangChain
        if query is None and 'v__args' in kwargs:
            wrapped_args = kwargs.get('v__args', [])
            if isinstance(wrapped_args, list) and wrapped_args:
                code_to_execute = wrapped_args[0] if isinstance(wrapped_args[0], str) else str(wrapped_args[0])
        
        # Check for 'command' parameter (used in invoke)
        if code_to_execute is None and 'command' in kwargs:
            code_to_execute = kwargs.get('command', '')
        
        # Return early if no code provided
        if not code_to_execute:
            return "No Python code provided to execute."
            
        if self.sanitize_input:
            code_to_execute = sanitize_input(code_to_execute)
            
        # HITL: Show proposed code and allow user to edit
        print(f"\n[Proposed Python code]:")
        edited_query = prompt("(Accept or Edit) > ", default=code_to_execute)
        
        # If user provided an empty command, consider it a cancellation
        if not edited_query.strip():
            return "Code execution cancelled by user."
            
        # Execute the edited code using the python_repl
        try:
            return self.python_repl.run(edited_query)
        except Exception as e:
            return f"Error executing Python code: {str(e)}"
            
    async def _arun(
        self,
        query: str = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """Use the tool asynchronously."""
        # For async execution, delegate to the synchronous implementation
        return await run_in_executor(None, self._run, query, run_manager, **kwargs)
    
    def invoke(self, args: Dict) -> str:
        """High-level interface used by the agent."""
        command = args.get("command", "")
        if not command:
            return "No Python code provided."
        return self._run(command)

# --- Tool instances ---
start_terminal_tool = StartTerminalTool()
python_repl_tool = PythonREPLTool_HITL() # Use our new HITL version
terminal_tool = TerminalTool_HITL()
direct_terminal_tool = TerminalTool_Direct()

# Register tools with the registry
from .tool_registry import register_tools

# Register all tools in this module
register_tools([
    start_terminal_tool,
    python_repl_tool,
    terminal_tool,
    # Don't register direct_terminal_tool in normal tools list
    # as it's only for internal use
])

def run_python_code(code: str) -> str:
    """Helper function to execute Python code."""
    return python_repl_tool.invoke({"command": code})

# Log registration
logger.debug(f"Registered terminal tools: start_terminal, python_repl, terminal")