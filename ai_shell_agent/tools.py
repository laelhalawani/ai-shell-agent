import subprocess
from langchain.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompt_toolkit import prompt
import logging

class InteractiveWindowsShellTool(BaseTool):
    name: str = "interactive_windows_shell_tool"
    description: str = (
        "Presents a prefilled, editable shell command prompt in the console using prompt_toolkit. "
        "The user can edit the command before it's executed."
    )

    def _run(self, command: str) -> str:
        """
        Runs the command after allowing the user to edit it.
        
        Args:
            command (str): The initial shell command proposed by the agent.
        
        Returns:
            str: The output from executing the edited command.
        """
        logging.info(f"Shell tool called with command: {command}")
        edited_command = prompt("Edit the command: ", default=command)
        logging.info(f"Executing command: {edited_command}")
        
        try:
            result = subprocess.run(
                edited_command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            logging.info(f"Command output: {output}")
            return output
        except subprocess.CalledProcessError as e:
            error = f"Error: {e.stderr}"
            logging.error(error)
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


class DirectWindowsShellTool(BaseTool):
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
        logging.info(f"Direct shell execution: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            logging.info(f"Command output: {output}")
            return output
        except subprocess.CalledProcessError as e:
            error = f"Error: {e.stderr}"
            logging.error(error)
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
interactive_windows_shell_tool = InteractiveWindowsShellTool()
direct_windows_shell_tool = DirectWindowsShellTool()

@tool
def run_python_code(code: str) -> str:
    """
    Executes a Python code snippet using the built-in Python REPL tool.
    
    Parameters:
      code (str): The Python code to execute.
      
    Returns:
      str: The output produced by executing the Python code.
    """
    logging.info(f"Python REPL tool called with code:\n{code}")
    result = python_repl_tool.run({"code": code})
    logging.info(f"Python code output: {result}")
    return result

tools = [
    interactive_windows_shell_tool,
    run_python_code,
]

tools_functions = [convert_to_openai_function(t) for t in tools]