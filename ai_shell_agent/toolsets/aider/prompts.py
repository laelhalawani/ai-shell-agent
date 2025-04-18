# ai_shell_agent/toolsets/aider/prompts.py
"""
Contains prompt fragments related to using the File Editor (Aider) toolset.
This content is returned as a ToolMessage when the toolset is activated.
"""

AIDER_TOOLSET_PROMPT = """\
You have activated the 'File Editor', an assistant to help you with edits for your projects. 
Use it only to create files and make edits to files in text formats. 
It is particularly effective for code but works with any text file. 
If a git repository is present, it will be used automatically for change tracking and diffs.
Open Files in the editor to submit them for edits or context, and then request edits to instruct how the files should be used and what needs to be changed.
The editor will work autonomously and ask you for input if it needs to clarify something.
Always make sure the files exist before trying to open them.
"""