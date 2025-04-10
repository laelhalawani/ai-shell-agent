from langchain.tools import BaseTool
from pydantic import Field, BaseModel
from typing import List, Optional, Type

# Use try-except for optional dependency loading
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

# --- Tool Schemas ---
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
