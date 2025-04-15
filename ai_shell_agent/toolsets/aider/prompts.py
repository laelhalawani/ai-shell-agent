# ai_shell_agent/toolsets/aider/prompts.py
"""
Contains prompt fragments related to using the File Editor (Aider) toolset.
This content is returned as a ToolMessage when the toolset is activated.
"""

AIDER_TOOLSET_PROMPT = """\
You have activated the 'File Editor' tools, an advanced code editing toolset for your project. Use it only to make edits to text files including regular files, code, configuration, scripts etc.
It is particularly effective for code but works with any text file. If a git repository is present, it will be used automatically for change tracking and diffs.
The editor is AI powered, it can view any files you open in the editor. Always open all files relevant for context, and specify edits only to the files that need to be changed.
IMPORTANT:
- You must have at least one file included before attempting to `request_edit`, inclusion is persistent per session, if you don't add the file you want to edit.
- Be specific in your `request_edit` instructions, as if you were explaining the request to a developer who will implement it.
- Use `list_files` if unsure what's in the context.
- You can add files to the context if they are relevant to the task at hand, even if they don't strictly need to be edited.
- Occasionally additional clarification might be requested via `[FILE_EDITOR_INPUT_NEEDED]` signals, in such cases immediately verify the request and respond directly using `submit_editor_input`.
- Use `close_file_editor` when the editing task is fully complete to free up resources and context, ensure the user won't need to continue editing these files before closing the editor, otherwise you will need to re-configure the context and describe the task to the editor again.
- Autonomously perform all the necessary operations before reporting back to the user, this includes any potential confirmations and providing information to the File Editor if prompted.
"""