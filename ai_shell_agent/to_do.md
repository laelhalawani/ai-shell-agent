Okay, let's refine the integration plan incorporating your specific requirements for tool naming, lifecycle management, and confirmation handling.

**Core Principles:**

1.  **Agent-Centric Control:** The `ai-shell-agent` drives the process, managing configuration (model, edit format) and deciding when to use the editor tools.
2.  **Aider as a Library:** Aider's `Coder` class is used under the hood, but its complexities are abstracted by the bridge and tools.
3.  **State Management:** The `AiderAgentBridge` maintains state per `ai-shell-agent` chat session, allowing recreation of the Aider `Coder` environment.
4.  **Generic Tooling:** Tool names and descriptions focus on the *action* (editing, selecting files) rather than mentioning "Aider".
5.  **Simplified Confirmation:** Tools requiring confirmation (like destructive Git operations or adding many files) will rely on the *agent* to decide whether to ask the user for confirmation *before* invoking the tool. The tools themselves will proceed assuming confirmation if called.

**Refined Implementation Plan:**

**Step 1: Update Dependencies (No Change)**

Ensure `aider-chat` is a requirement.

**Step 2: Refine `AiderAgentBridge` and `AiderSessionState` (`aider_bridge.py`)**

*   **`AiderSessionState`:**
    *   Will store *all* relevant `init_kwargs` needed to recreate the `Coder`. This includes parameters derived from `ai-shell-agent`'s settings at the time of `StartFileEditorTool` execution (model names, edit format, commit settings, lint/test settings, etc.).
    *   Add a flag `is_active` (boolean) to track if the editor is currently "started" for this chat ID.
*   **`AiderAgentBridge`:**
    *   `_get_session_state`: Remains mostly the same, creates state if needed.
    *   `_ensure_api_key`: Will fetch the key using `ai_shell_agent.config_manager` and set it in `os.environ` before calling Aider code that needs it.
    *   `get_coder`:
        *   Checks `state.is_active`. If `False`, returns `None`.
        *   If `state.coder_instance` exists, return it after updating state *from* it.
        *   If `state.coder_instance` is `None` but `state.is_active` is `True`, recreate it using `state.init_kwargs` and restored history/files. This handles cases where the agent process might restart but the state file persists.
        *   Uses `AgentInputOutput`.
    *   `start_editor_session`: (New method called by `StartFileEditorTool`)
        *   Gets/creates state for `chat_id`.
        *   Resets existing state (clears history, files, coder instance).
        *   Populates `state.init_kwargs` based on *current* `ai-shell-agent` settings (model, edit format, etc.). It should fetch these settings dynamically.
        *   Sets `state.is_active = True`.
        *   *Crucially, it does not create the `Coder` instance here.* Creation is lazy, happening when a tool first needs it via `get_coder`.
        *   Saves the config.
    *   `close_editor_session`: (New method called by `CloseFileEditorTool`)
        *   Gets state for `chat_id`.
        *   Sets `state.is_active = False`.
        *   Sets `state.coder_instance = None` (releases the object).
        *   Optionally clear other parts of the state if desired upon closing (like chat history?). *Decision:* Keep history/files unless explicitly reset, just mark inactive.
        *   Saves the config.
    *   `_execute_coder_method`/`_execute_command_method`:
        *   First call `get_coder`. If it returns `None`, return an error message ("File Editor not active...").
        *   Proceed with execution.
        *   Handle `AgentInputOutput` capturing.
        *   Call `_update_and_save_state` in `finally`.
*   **`AgentInputOutput`:**
    *   `confirm_ask`: Will *always* return `True` (or the value of `self.yes`, which is set to `True` in the bridge). It will log the prompt attempt for debugging. The responsibility of asking the *actual* user is pushed up to the agent *before* calling the tool.
    *   Other methods mostly unchanged, focusing on capturing output/errors.

**Step 3: Define Tools (`aider_tools.py`)**

Create `ai_shell_agent/tools/aider_tools.py`. Use `BaseTool` subclasses.

```python
# ai_shell_agent/tools/aider_tools.py
from langchain.tools import BaseTool
from pydantic import Field
from typing import List, Optional, Type

# Use try-except for optional dependency loading in __init__ might be cleaner
try:
    from ..integrations.aider_bridge import aider_bridge, AIDER_AVAILABLE
except ImportError:
    AIDER_AVAILABLE = False

# --- Helper Function for Disabled Tools ---
def create_disabled_tool(name: str, description: str, args_schema: Optional[Type[BaseModel]] = None) -> Type[BaseTool]:
    class DisabledTool(BaseTool):
        name: str = name
        description: str = description
        args_schema: Optional[Type[BaseModel]] = args_schema

        def _run(self, *args, **kwargs) -> str:
            return "Aider integration is not available. Please install aider-chat."

        async def _arun(self, *args, **kwargs) -> str:
            return "Aider integration is not available. Please install aider-chat."
    return DisabledTool

# --- Tool Schemas (using Pydantic for clarity) ---
from pydantic import BaseModel, Field

class EditFilesArgs(BaseModel):
    instructions: str = Field(description="Detailed, natural language instructions describing the desired code changes.")

class SelectFilesArgs(BaseModel):
    files: List[str] = Field(description="List of file paths (relative to project root or absolute) or glob patterns to select for editing context.")

class DeselectFilesArgs(BaseModel):
    files: List[str] = Field(description="List of file paths (relative or absolute) or glob patterns to deselect from the editing context.")

class CommitEditsArgs(BaseModel):
    commit_message: Optional[str] = Field(default=None, description="Optional commit message. If not provided, one will be generated based on the changes.")


# --- Tool Definitions ---
if not AIDER_AVAILABLE:
    StartFileEditorTool = create_disabled_tool("StartFileEditor", "Starts a new file editing session, clearing previous context.")
    CloseFileEditorTool = create_disabled_tool("CloseFileEditor", "Closes the current file editing session.")
    EditFilesTool = create_disabled_tool("EditFiles", "Sends instructions to edit the selected code files.", EditFilesArgs)
    SelectFilesTool = create_disabled_tool("SelectFiles", "Selects one or more files, making them available for editing.", SelectFilesArgs)
    DeselectFilesTool = create_disabled_tool("DeselectFiles", "Deselects one or more files from the editing context.", DeselectFilesArgs)
    ListSelectedFilesTool = create_disabled_tool("ListSelectedFiles", "Lists all files currently selected for editing.")
    ShowDiffSinceLastInstructionTool = create_disabled_tool("ShowDiffSinceLastInstruction", "Shows the git diff of changes made since the last edit instruction.")
    CommitEditsTool = create_disabled_tool("CommitEdits", "Commits all pending changes in the repository.", CommitEditsArgs)
    UndoLastEditTool = create_disabled_tool("UndoLastEdit", "Undoes the last set of edits applied in the current session.")
    GetContextTokenCountTool = create_disabled_tool("GetContextTokenCount", "Reports the estimated token usage for the current editing context.")
    ClearEditorChatHistoryTool = create_disabled_tool("ClearEditorChatHistory", "Clears the editor's chat history for the current session, keeping the files selected.")
    ResetEditorSessionTool = create_disabled_tool("ResetEditorSession", "Resets the editor session, clearing history and deselecting all files.")

else:
    class StartFileEditorTool(BaseTool):
        name: str = "StartFileEditor"
        description: str = (
            "Starts a new file editing session. This initializes the editor environment based on current agent settings"
            " (like selected model and edit format) and clears any previous editor context (selected files, history)."
            " Use 'SelectFiles' next to add files you want to edit."
        )

        def _run(self) -> str:
            chat_id = aider_bridge._get_current_chat_id()
            if not chat_id: return "No active chat session."
            return aider_bridge.start_editor_session(chat_id)

        async def _arun(self) -> str:
            return self._run()

    class CloseFileEditorTool(BaseTool):
        name: str = "CloseFileEditor"
        description: str = (
            "Closes the current file editing session, releasing associated resources."
            " The selected files and edit history are preserved until the next 'StartFileEditor'."
        )

        def _run(self) -> str:
            chat_id = aider_bridge._get_current_chat_id()
            if not chat_id: return "No active chat session."
            return aider_bridge.close_editor_session(chat_id)

        async def _arun(self) -> str:
            return self._run()

    class EditFilesTool(BaseTool):
        name: str = "EditFiles"
        description: str = (
            "Sends instructions to edit the currently selected code files. "
            "Provide clear, natural language instructions describing the desired changes. "
            "The editor will attempt to apply these changes and commit them. "
            "Returns a summary of actions, outputs, errors, or confirmation requests."
        )
        args_schema: Type[BaseModel] = EditFilesArgs

        def _run(self, instructions: str) -> str:
            _, output = aider_bridge._execute_coder_method('run_instruction', instruction=instructions)
            return output or "Edit instruction processed."

        async def _arun(self, instructions: str) -> str:
             # Basic async wrapper, consider true async implementation in bridge if needed
            return self._run(instructions)

    class SelectFilesTool(BaseTool):
        name: str = "SelectFiles"
        description: str = (
            "Selects one or more files or glob patterns, adding them to the current editing context. "
            "Use relative paths from the project root or absolute paths. "
            "Only selected files can be edited."
        )
        args_schema: Type[BaseModel] = SelectFilesArgs

        def _run(self, files: List[str]) -> str:
            args_str = " ".join([f'"{f}"' for f in files])
            _, output = aider_bridge._execute_command_method('cmd_add', args_str)
            return output or "Files processed for selection."

        async def _arun(self, files: List[str]) -> str:
            return self._run(files)

    class DeselectFilesTool(BaseTool):
        name: str = "DeselectFiles"
        description: str = (
            "Deselects one or more files or glob patterns from the current editing context. "
            "These files will no longer be modified unless re-selected."
        )
        args_schema: Type[BaseModel] = DeselectFilesArgs

        def _run(self, files: List[str]) -> str:
            args_str = " ".join([f'"{f}"' for f in files])
            _, output = aider_bridge._execute_command_method('cmd_drop', args_str)
            return output or "Files processed for deselection."

        async def _arun(self, files: List[str]) -> str:
            return self._run(files)

    class ListSelectedFilesTool(BaseTool):
        name: str = "ListSelectedFiles"
        description: str = "Lists all files currently selected and available for editing in the current session."

        def _run(self) -> str:
            _, output = aider_bridge._execute_command_method('cmd_ls', "")
            return output or "No files selected or editor not active."

        async def _arun(self) -> str:
            return self._run()

    class ShowDiffSinceLastInstructionTool(BaseTool):
        name: str = "ShowDiffSinceLastInstruction"
        description: str = "Shows the git diff of changes applied by the editor since the last edit instruction was successfully processed."

        def _run(self) -> str:
            _, output = aider_bridge._execute_command_method('cmd_diff', "")
            return output or "No diff to show or editor not active."

        async def _arun(self) -> str:
            return self._run()

    class CommitEditsTool(BaseTool):
        name: str = "CommitEdits"
        description: str = (
            "Commits all pending changes in the repository associated with the current editing session. "
            "Optionally provide a specific commit message; otherwise, one will be generated."
        )
        args_schema: Type[BaseModel] = CommitEditsArgs

        def _run(self, commit_message: Optional[str] = None) -> str:
            # Agent should ask user for confirmation *before* calling this tool if appropriate.
            _, output = aider_bridge._execute_command_method('cmd_commit', commit_message or "")
            return output or "Commit processed."

        async def _arun(self, commit_message: Optional[str] = None) -> str:
            return self._run(commit_message)

    class UndoLastEditTool(BaseTool):
        name: str = "UndoLastEdit"
        description: str = (
             "Reverts the last set of code changes applied by the 'EditFiles' tool during the current session. "
             "This relies on the git history created by the editor."
        )

        def _run(self) -> str:
            # Agent should ask user for confirmation *before* calling this tool.
            _, output = aider_bridge._execute_command_method('cmd_undo', "")
            return output or "Undo processed."

        async def _arun(self) -> str:
            return self._run()

    class GetContextTokenCountTool(BaseTool):
        name: str = "GetContextTokenCount"
        description: str = (
            "Reports the estimated number of tokens currently being used by the editor's context "
            "(selected files, chat history, repository map). Useful for managing context window limits."
        )

        def _run(self) -> str:
            _, output = aider_bridge._execute_command_method('cmd_tokens', "")
            return output or "Could not get token count or editor not active."

        async def _arun(self) -> str:
            return self._run()

    class ClearEditorChatHistoryTool(BaseTool):
        name: str = "ClearEditorChatHistory"
        description: str = (
            "Clears the internal chat history used by the file editor for the current session. "
            "This can help reduce token usage but may cause the editor to lose context about previous edits. "
            "Selected files remain selected."
        )

        def _run(self) -> str:
            return aider_bridge.clear_history()

        async def _arun(self) -> str:
            return self._run()

    class ResetEditorSessionTool(BaseTool):
        name: str = "ResetEditorSession"
        description: str = (
             "Completely resets the file editor session. Clears the internal chat history and deselects all files. "
             "Equivalent to closing and then starting the editor."
        )

        def _run(self) -> str:
            return aider_bridge.reset_session()

        async def _arun(self) -> str:
            return self._run()

# Combine all Aider tools into a list
aider_tools = [
    StartFileEditorTool(),
    CloseFileEditorTool(),
    EditFilesTool(),
    SelectFilesTool(),
    DeselectFilesTool(),
    ListSelectedFilesTool(),
    ShowDiffSinceLastInstructionTool(),
    CommitEditsTool(),
    UndoLastEditTool(),
    GetContextTokenCountTool(),
    ClearEditorChatHistoryTool(),
    ResetEditorSessionTool(),
] if AIDER_AVAILABLE else []
```

**Step 4: Integrate Tools into `ai-shell-agent` (`ai_shell_agent/tools.py`)**

This remains the same as the previous plan: import `aider_tools` and add it to the main `tools` list.

```python
# ai_shell_agent/tools.py
# ... (existing imports and tool classes) ...

# --- Import Aider Tools ---
try:
    from .aider_tools import aider_tools
except ImportError:
    logger.warning("Aider tools could not be imported.")
    aider_tools = []
# --- End Import Aider Tools ---

# Combine all tools
tools = [
    interactive_windows_shell_tool,
    python_repl_tool,
    # Potentially other non-Aider tools here
] + aider_tools # Add aider_tools here

# Convert all tools for function calling
tools_functions = [convert_to_openai_function(t) for t in tools]
```

**Step 5: Update Chat Manager (No Change)**

The previous cleanup logic in `delete_chat` using `chat_id` remains appropriate.

**Step 6: Agent Integration (Critical)**

The agent's prompting and logic need refinement:

1.  **Tool Loading:** Ensure the agent loads the updated `tools` list containing the `BaseTool` classes.
2.  **Workflow:**
    *   The agent must understand that code editing requires starting the editor (`StartFileEditorTool`) first.
    *   Files must be added using `SelectFilesTool` before `EditFilesTool` can modify them.
    *   After editing, the agent might use `ShowDiffSinceLastInstructionTool` or `CommitEditsTool`.
    *   The agent should use `CloseFileEditorTool` when the editing task is complete to release resources and signal the end of the editing context.
    *   If a tool returns "File Editor not active...", the agent should know to call `StartFileEditorTool`.
3.  **Configuration:** The agent must manage the desired `model` and `edit_format` settings. When `StartFileEditorTool` is called, the `AiderAgentBridge` will fetch these current settings from the agent's configuration (`ai_shell_agent.config_manager`) to initialize the Aider session state.
4.  **Confirmation:** The agent's core prompt *must* instruct it to check with the user before using potentially destructive tools like `CommitEditsTool` or `UndoLastEditTool`, especially if the agent is uncertain. The agent should use its normal chat capabilities to ask the user "Should I commit these changes?" or "Should I undo the last edit?" and *then*, based on the user's textual response, decide whether or not to invoke the corresponding tool.

**Example Agent Prompt Snippet (Conceptual):**

```text
[...]
Available Tools:
- StartFileEditor: Begins a code editing session. Call this before selecting or editing files.
- CloseFileEditor: Ends the current code editing session.
- SelectFiles(files): Selects files (paths or globs) to be edited.
- EditFiles(instructions): Applies edits to selected files based on your instructions.
- CommitEdits(commit_message): Commits changes. IMPORTANT: Ask the user for confirmation before using this unless explicitly told to commit.
- UndoLastEdit: Undoes the last edit. IMPORTANT: Ask the user for confirmation before using this unless explicitly told to undo.
[...]

When asked to edit code:
1. Ensure the editor is started using StartFileEditor if not already active.
2. Use SelectFiles to add the necessary files to the context.
3. Use EditFiles with detailed instructions.
4. If the edit is complex or potentially risky (e.g., involves committing or undoing), ask the user for confirmation *first* using natural language chat before invoking the relevant tool (CommitEdits, UndoLastEdit).
5. When the editing task is complete, call CloseFileEditor.
```

This refined approach aligns with your requirements for `BaseTool` classes, generic naming, agent-driven configuration, explicit lifecycle management, and simplified (agent-managed) confirmation.