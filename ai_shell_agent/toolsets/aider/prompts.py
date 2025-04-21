# ai_shell_agent/toolsets/aider/prompts.py
"""
Contains prompt fragments related to using the File Editor (Aider) toolset.
This content is returned as a ToolMessage when the toolset is activated.
"""

AIDER_TOOLSET_PROMPT = """\
You have activated the Code Copilot, an AI coding assistant to help you with editing tasks.
It is particularly effective for code but works with any text file. 
If a git repository is present, it will be used automatically for change tracking and diffs.
Make sure to add any relevant files to editing, and for context, as well as properly explain the task you want to accomplish when submitting edit requests.
The editor will work autonomously and ask you for input if it needs to clarify something.
"""