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
from langchain_experimental.tools.python.tool import PythonREPLTool

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
    """
    Activates the 'Terminal' toolset, enabling direct shell command execution tools.
    This might be called automatically or if the toolset was deactivated.
    """
    name: str = "start_terminal"
    description: str = "Activates or ensures the 'Terminal' toolset is active, enabling tools like 'terminal' (execute shell commands) and 'python_repl'."

    def _run(self, args: str = "") -> str:
        """Activates the Terminal toolset and updates the system prompt."""
        logger.info("StartTerminalTool called with args: %s", args)
        
        chat_file = get_current_chat()
        if not chat_file:
            logger.error("No active chat session found when activating Terminal toolset")
            return "Error: No active chat session found."

        current_toolsets = get_active_toolsets(chat_file)
        logger.debug(f"Current active toolsets before activation: {current_toolsets}")
        
        toolset_name = "Terminal" # The name defined in llm.py

        if toolset_name in current_toolsets:
            logger.debug(f"'{toolset_name}' toolset is already active, no action needed")
            return f"'{toolset_name}' toolset is already active. Tools like 'terminal' and 'python_repl' are available."

        # Activate the toolset
        new_toolsets = list(current_toolsets)
        new_toolsets.append(toolset_name)
        logger.debug(f"Updating active toolsets to: {new_toolsets}")
        update_active_toolsets(chat_file, new_toolsets) # Save updated toolsets

        # Re-build the system prompt with the new set of active toolsets
        logger.debug("Rebuilding system prompt with updated toolsets")
        new_system_prompt = build_prompt(active_toolsets=new_toolsets)

        # Update the system prompt message in the chat history (message 0)
        # Use the helper function from state manager for safety
        logger.debug("Updating system prompt in chat history")
        _update_message_in_chat(chat_file, 0, {"role": "system", "content": new_system_prompt})
        logger.info(f"Successfully activated '{toolset_name}' toolset for chat {chat_file}")

        return f"'{toolset_name}' toolset activated. You can now use tools like 'terminal' and 'python_repl'."

    async def _arun(self, args: str = "") -> str:
        # Simple enough to run synchronously
        return self._run(args)

class TerminalTool_HITL(BaseTool):
    """
    Tool for interacting with the Windows shell with human-in-the-loop confirmation.
    This tool is safer as it requires user confirmation before execution.
    """
    name: str = "terminal"
    description: str = """Executes a command in the system's terminal. Use this for cmd/terminal/console/shell commands."""
    
    def _run(self, command: str) -> str:
        """Execute a command in the Windows shell with confirmation."""
        print(f"\n[Command to execute]: {command}")
        confirmation = prompt("[Execute? (y/n)]: ").lower()
        if confirmation != 'y':
            return "Command cancelled by user."
        
        try:
            # Use subprocess with shell=True for Windows shell commands
            result = subprocess.run(
                command, 
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            return f"Return Code: {result.returncode}\nOutput:\n{result.stdout}\nError:\n{result.stderr}"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
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
    description: str = """Executes a command in the system's terminal. Use this for cmd/terminal/console/shell commands."""
    
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

# --- Tool instances ---
start_terminal_tool = StartTerminalTool() # NEW INSTANCE
python_repl_tool = PythonREPLTool()
interactive_terminal_tool = TerminalTool_HITL()
direct_terminal_tool = TerminalTool_Direct() # Still useful for direct -x execution maybe

def run_python_code(code: str) -> str:
    """Helper function to execute Python code."""
    return python_repl_tool.invoke({"command": code})

# --- List of base tools defined in THIS file ---
base_tools_in_this_file = [
    start_terminal_tool, # ADDED
    interactive_terminal_tool,
    python_repl_tool,
    # direct_terminal_tool, # Should this be exposed to LLM? Probably not by default.
]

# --- Register the base tools at the end of the file ---
# Ensure this runs after all tool instances are created
register_tools(base_tools_in_this_file)
logger.debug(f"Registered {len(base_tools_in_this_file)} base tools from tools.py.")