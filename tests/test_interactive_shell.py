import subprocess
import pytest
from prompt_toolkit import prompt
from langchain_experimental.tools.python.tool import PythonREPLTool
from ai_shell_agent.tools import interactive_terminal_tool, run_python_code

# Monkeypatch the prompt from prompt_toolkit to simulate user editing.

def test_interactive_windows_shell_tool(fake_prompt, fake_subprocess):
    # Test interactive tool with simulated user edit and fake subprocess run.
    initial_command = "dir"
    output = interactive_terminal_tool.run(initial_command)
    # Since our fake prompt appends " /A", we expect the executed command to be "dir /A"
    assert "Directory:" in output

def test_run_python_code():
    # Test the Python REPL tool wrapper.
    sample_code = "print('Test output')"
    output = run_python_code(sample_code)
    # Since we are calling the actual PythonREPLTool, we expect the output to contain our printed text.
    # Note: This test assumes PythonREPLTool works as expected without modification.
    assert "Test output" in output
