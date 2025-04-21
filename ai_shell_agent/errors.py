# ai_shell_agent/errors.py
"""
Custom exceptions for AI Shell Agent.
"""

from typing import Dict, Any, Optional

class AgentError(Exception):
    """Base class for agent exceptions."""
    pass

class PromptNeededError(AgentError):
    """
    Raised by HITL tools when user input is required.
    
    This exception is used to signal from tools back to the main chat loop
    that user input is needed before the tool can execute. The chat manager
    will catch this exception and display a prompt to the user.
    """
    def __init__(
        self, 
        tool_name: str, 
        proposed_args: Dict[str, Any], 
        edit_key: str, 
        prompt_suffix: str = "(edit or confirm) > "
    ):
        """
        Initialize a PromptNeededError.
        
        Args:
            tool_name: Name of the tool requiring input
            proposed_args: Dictionary of all arguments for the tool, including the one to edit
            edit_key: Key in proposed_args that needs user input
            prompt_suffix: Text to show after the prompt
        """
        self.tool_name = tool_name
        self.proposed_args = proposed_args
        self.edit_key = edit_key
        self.prompt_suffix = prompt_suffix
        message = f"Tool '{tool_name}' requires user input for key '{edit_key}'."
        super().__init__(message)

class ToolExecutionError(AgentError):
    """
    Raised when a tool execution fails with a known reason.
    
    This exception can be caught and handled more gracefully than
    generic exceptions in the chat manager.
    """
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Error executing tool '{tool_name}': {message}")