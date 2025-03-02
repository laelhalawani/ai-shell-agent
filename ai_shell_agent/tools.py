import subprocess
from langchain.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompt_toolkit import prompt
from . import logger

class ConsoleTool_HITL(BaseTool):
    name: str = "interactive_windows_cmd_tool"
    description: str = (
        "Use this tool to run CMD.exe commands and view the output."
        "Args:"
        "command (str): The initial cmd command proposed by the agent."
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
        edited_command = prompt("(Accept or Edit) > \n", default=command)
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


class ConsoleTool_Direct(BaseTool):
    name: str = "direct_cmd_shell_tool"
    description: str = "Executes a console command directly without user confirmation."

    def _run(self, command: str) -> str:
        """
        Runs the CMD command directly without user confirmation.
        
        Args:
            command (str): The shell command to execute.
        
        Returns:
            str: The output from executing the command.
        """
        logger.debug(f"> {command}")
        try:
            result = subprocess.run(
                command,
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
            command (str): The shell command to execute.
        
        Returns:
            str: The output from executing the command.
        """
        return self._run(command)

# Initialize the built-in Python REPL tool
python_repl_tool = PythonREPLTool()
interactive_windows_shell_tool = ConsoleTool_HITL()
direct_windows_shell_tool = ConsoleTool_Direct()

tools = [
    interactive_windows_shell_tool,
    python_repl_tool,
    
]

tools_functions = [convert_to_openai_function(t) for t in tools]