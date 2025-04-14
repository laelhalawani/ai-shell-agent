# ai_shell_agent/toolsets/aider/prompts.py
"""
Contains prompt fragments related to using the File Editor (Aider) toolset.
This content is returned as a ToolMessage when the toolset is activated.
"""

AIDER_TOOLSET_PROMPT = """\
You have activated the 'File Editor' toolset, powered by Aider.
This toolset enables advanced, chat-based code editing for files within the user's project. It is particularly effective for code but works with any text file. If a git repository is present, it will be used automatically for change tracking and diffs.

Available Tools:
- `include_file`: Adds a file (by relative or absolute path) to the editor's context. MUST be done before editing.
- `exclude_file`: Removes a file from the editor's context.
- `list_files`: Shows which files are currently included in the context.
- `request_edit`: Describes changes you want to make to the included files in natural language. The editor will process this request.
- `view_diff`: (Optional) After an edit request, shows the proposed changes before they are potentially committed (if git is used).
- `undo_last_edit`: (Optional) If using git and auto-commits are enabled, reverts the last change made by `request_edit`.
- `submit_editor_input`: If the editor needs confirmation or clarification during a `request_edit` (indicated by `[FILE_EDITOR_INPUT_NEEDED]`), use this tool to provide the necessary input (e.g., 'yes', 'no', 'all', 'skip', 'don't ask', or specific text).
- `close_file_editor`: When finished with all editing tasks for the current session, clears the editor's context (included files, history). This makes the other File Editor tools unavailable until you start it again.

IMPORTANT:
- You must have at least one file included before attempting to `request_edit`, inclusion is persistent per session.
- Be specific in your `request_edit` instructions, as if you were explaining the request to a developer who will implement it.
- Use `list_files` if unsure what's in the context.
- You can add files to the context if they are relevant to the task at hand, even if they don't strictly need to be edited.
- Occasionally additional clarification might be requested via `[FILE_EDITOR_INPUT_NEEDED]` signals, in such cases immediately verify the request and respond directly using `submit_editor_input`.
- Use `close_file_editor` when the editing task is fully complete to free up resources and context, ensure the user won't need to continue editing these files before closing the editor, otherwise you will need to re-configure the context and describe the task to the editor again.
- Autonomously perform all the necessary operations before reporting back to the user.
"""