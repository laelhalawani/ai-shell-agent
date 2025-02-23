import subprocess
from langchain.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompt_toolkit import prompt
from . import logger

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


class MessageUserTool(BaseTool):
    name: str = "message_user_tool"
    description: str = (
        "Use this tool to send a message to the user after you've run the necessary commands."
        "Args:"
        "message (str): The message to send to the user."
        "Returns:"
        "str: The message sent to the user."
    )

    def _run(self, message: str) -> str:
        """
        Sends a message to the user.
        
        Args:
            message (str): The message to send to the user.
        
        Returns:
            str: The message sent to the user.
        """
        logger.info(f"AI: {message}")
        return message

    async def _arun(self, message: str) -> str:
        """
        Asynchronous implementation of sending a message to the user.
        
        Args:
            message (str): The message to send to the user.
        
        Returns:
            str: The message sent to the user.
        """
        return self._run(message)

# Initialize the built-in Python REPL tool
python_repl_tool = PythonREPLTool()
interactive_windows_shell_tool = WindowsCmdExeTool_HITL()
direct_windows_shell_tool = WindowsCmdExeTool_Direct()
message_user_tool = MessageUserTool()

@tool
def run_python_code(code: str) -> str:
    """
    Executes a Python code snippet using the built-in Python REPL tool.
    
    Parameters:
      code (str): The Python code to execute.
      
    Returns:
      str: The output produced by executing the Python code.
    """
    logger.info(f"Python REPL tool called with code:\n{code}")
    result = python_repl_tool.run({"code": code})
    logger.info(f"Python code output: {result}")
    return result

tools = [
    interactive_windows_shell_tool,
    run_python_code,
    message_user_tool
    
]

tools_functions = [convert_to_openai_function(t) for t in tools]