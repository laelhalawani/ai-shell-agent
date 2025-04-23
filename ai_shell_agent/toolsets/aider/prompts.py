# ai_shell_agent/toolsets/aider/prompts.py
"""
Contains prompt fragments related to using the AI Code Copilot tools.
This content is returned as a ToolMessage when the toolset is activated.
"""

AIDER_TOOLSET_PROMPT = """\
You have activated the AI Code Copilot, an assistant to help you with editing code and scripts.
If a git repository is present, it will be used automatically for change tracking and diffs.
Make sure to add relevant files to the editing context and clearly explain the task you want to accomplish when submitting edit requests.
The AI Code Copilot will work autonomously and ask you for input if it needs clarification.
"""