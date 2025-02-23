import subprocess
from typing import Literal
from langchain.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompt_toolkit import prompt
from .. import logger
import os
import platform

class WindowsCmdExeTool_HITL(BaseTool):
    name: str = "interactive_windows_shell_tool"
    description: str = (
        "Use this tool to run CMD.exe commands and view the output."
        "Args:"
        "command (str): The initial shell command proposed by the agent."
        "Returns:"
        "str: The output from executing the edited command."
    )

    def _run(self, command: str) -> str:
        """
        Runs the command after allowing the user to edit it.
        
        Args:
            command (str): The initial shell command proposed by the agent.
        
        Returns:
            str: The output from executing the edited command.
        """
        edited_command = prompt("(Accept or Edit) CMD> \n", default=command)
        logger.debug(f"Executing command: {edited_command}")
        
        try:
            result = subprocess.run(
                edited_command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            logger.info(f"{output}")
            return output
        except subprocess.CalledProcessError as e:
            error = f"Error: {e.stderr}"
            logger.error(error)
            return error

    async def _arun(self, command: str) -> str:
        """
        Asynchronous implementation of running a command.
        
        Args:
            command (str): The initial shell command proposed by the agent.
        
        Returns:
            str: The output from executing the edited command.
        """
        return self._run(command)


class WindowsCmdExeTool_Direct(BaseTool):
    name: str = "direct_windows_shell_tool"
    description: str = "Executes a shell command directly without user confirmation."

    def _run(self, command: str) -> str:
        """
        Runs the shell command directly without user confirmation.
        
        Args:
            command (str): The shell command to execute.
        
        Returns:
            str: The output from executing the command.
        """
        logger.info(f"CMD.exe> {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            logger.debug(f"Command output: {output}")
            return output
        except subprocess.CalledProcessError as e:
            error = f"Error: {e.stderr}"
            logger.error(error)
            return error

    async def _arun(self, command: str) -> str:
        """
        Asynchronous implementation of running a command.
        
        Args:
            command (str): The shell command to execute.
        
        Returns:
            str: The output from executing the command.
        """
        return self._run(command)



class TagTechTool(BaseTool):
    name: str = "tag_tech_query"
    description: str = "Use this tool to tag the user message as technical question or query. "

    def _run(self, message: str) -> Literal["tech_query"]:
        logger.debug(f"Tagging message as technical query.")
        """
        Forward the user to tech support for further assistance.
        
        Args:
            message (str): The message to forward to tech support.
        
        Returns:
            str: The message to forward to tech support.
        """
        return f"Forwarding to tech support: {message}"

    async def _arun(self, message: str) -> str:
        """
        Asynchronous implementation of forwarding the user to tech support.
        
        Args:
            message (str): The message to forward to tech support.
        
        Returns:
            str: The message to forward to tech support.
        """
        return self._run(message)

class CheckSystemTool(BaseTool):
    name: str = "check_system"
    description: str = "Use this tool to retrieve system brand and version."
    
    def _run(self) -> str:
        """
        Retrieves the system brand and version.
        
        Returns:
            str: The system brand and version.
        """
        return f"{platform.system()} {platform.version()}"
    
    async def _arun(self) -> str:
        """
        Asynchronous implementation of retrieving the system brand and version.
        
        Returns:
            str: The system brand and version.
        """
        return self._run()
    

# Initialize the built-in Python REPL tool
python_repl_tool = PythonREPLTool()
interactive_windows_shell_tool = WindowsCmdExeTool_HITL()
direct_windows_shell_tool = WindowsCmdExeTool_Direct()
tag_tech_tool = TagTechTool()

tech_tools = [
    interactive_windows_shell_tool,
    python_repl_tool,
]
tech_tools_functions = [convert_to_openai_function(t) for t in tech_tools]

router_tools = [
    tag_tech_tool,
]
router_tools_functions = [convert_to_openai_function(t) for t in router_tools]
    