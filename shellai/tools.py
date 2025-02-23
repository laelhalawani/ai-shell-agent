import subprocess
from langchain.tools import BaseTool, tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompt_toolkit import prompt

@tool
def interactive_windows_shell_tool(command: str) -> str:
    """
    Presents a prefilled, editable shell command prompt in the console using prompt_toolkit.
    The user can edit the command before it's executed.
    
    Parameters:
      command (str): The initial shell command proposed by the agent.
      
    Returns:
      str: The output from executing the edited command.
    """
    # Present the command in an editable prompt.
    edited_command = prompt("Edit the command: ", default=command)
    try:
        result = subprocess.run(
            edited_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# Initialize the built-in Python REPL tool.
python_repl_tool = PythonREPLTool()

@tool
def run_python_code(code: str) -> str:
    """
    Executes a Python code snippet using the built-in Python REPL tool.
    
    Parameters:
      code (str): The Python code to execute.
      
    Returns:
      str: The output produced by executing the Python code.
    """
    return python_repl_tool.run({"code": code})

# For testing purposes.
if __name__ == "__main__":
    # Test the interactive Windows shell tool.
    initial_command = "dir"
    print("Original Command:", initial_command)
    print("Interactive Shell Tool Output:")
    print(interactive_windows_shell_tool.run(initial_command))
    
    # Test the Python REPL tool.
    sample_code = "print('Hello from Python REPL!')"
    print("Python REPL Output:")
    print(run_python_code(sample_code))
