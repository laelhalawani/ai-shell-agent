# ai_shell_agent/prompts/ai_editor_prompt.py
"""
Contains prompt fragments related to using the AI Editor toolset.
"""

AI_EDITOR_TOOLSET_INTRO = """\

You have access to an 'AI Editor' toolset for code and text file manipulation.
This toolset allows you to add files to a context, request edits using natural language, view changes (diffs), and undo edits.
It can work with most text files but is especally effective with code files (Python, JavaScript, etc.)
If you are in a directory containing a git repository, the editor will automatically detect it and use git for version control.

Key Tools & Workflow:
1.  **`include_file`**: Add a file (using its relative or absolute path) to the editor's context. You MUST include files before editing.
2.  **`exclude_file`**: Remove a file from the editor's context.
3.  **`list_files`**: See which files are currently included in the context.
4.  **`request_edit`**: Describe the changes you want to make to the included files in natural language. The editor will process this request.
5.  **`view_diff`**: (Optional) After an edit request, use this to see the proposed changes before they are potentially committed (if git is used).
6.  **`undo_last_edit`**: (Optional) If using git and auto-commits are enabled, this reverts the last change made by `request_edit`.
7.  **`submit_editor_input`**: If the editor needs confirmation or clarification during a `request_edit` (indicated by `[AI_EDITOR_INPUT_NEEDED]`), use this tool to provide the necessary input (e.g., 'yes', 'no', 'all', 'skip', 'don't ask', or specific text).
8.  **`close_ai_editor`**: When finished with all editing tasks for the current session, use this to clear the editor's context (included files, history). This makes the other AI Editor tools unavailable until you start it again.

IMPORTANT:
- You must have at least one file included before attempting to `request_edit`, inclusion is persistent per session.
- Be specific in your `request_edit` instructions, as if you were explaining the request to a developer who will implement it.
- Use `list_files` if unsure what's in the context.
- You can add files to the context if they are relevant to the task at hand, even if they don't strictly need to be edited.
- Ocassionally adtional clarifiation might be requested via `[AI_EDITOR_INPUT_NEEDED]` signals, in such cases immidiately verify the request and respond directly using `submit_editor_input`.
- Use `close_ai_editor` when the editing task is fully complete to free up resources and context, ensure the user won't need to continue editing these files before closing the editor, otherwise you will need to re-configure the context and describe the task to the editor again.
"""