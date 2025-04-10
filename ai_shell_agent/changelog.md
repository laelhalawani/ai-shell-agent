# AI Shell Agent Changelog

## [1.0.1] - Added Aider Code Editor Integration

### Added
- Integrated Aider code editing capabilities through custom tools
- Added dependency on `aider-chat` package for code editing functionality
- Created bridging architecture between AI Shell Agent and Aider
- Added 12 code editing tools:
  - StartFileEditor: Starts a new file editing session
  - CloseFileEditor: Closes the current file editing session
  - EditFiles: Sends instructions to edit the selected code files
  - SelectFiles: Selects files for editing
  - DeselectFiles: Removes files from editing context
  - ListSelectedFiles: Shows files ready for editing
  - ShowDiffSinceLastInstruction: Displays changes made by the editor
  - CommitEdits: Commits changes to git
  - UndoLastEdit: Reverts the last set of changes
  - GetContextTokenCount: Shows token usage
  - ClearEditorChatHistory: Clears editor history
  - ResetEditorSession: Completely resets the editor state
- Added state persistence for editor sessions
- Implemented model compatibility between AI Shell Agent's models and Aider

### Modified
- Enhanced chat session deletion to clean up associated editor sessions
- Updated tools.py to dynamically include the Aider tools when available

### Completed Tasks
- ✅ Created `AiderAgentBridge` and `AiderSessionState` for state management
- ✅ Implemented `AgentInputOutput` for capturing editor output
- ✅ Created all 12 editor tools with BaseTool implementation
- ✅ Added tool loading and integration with AI Shell Agent
- ✅ Implemented session persistence between agent restarts
- ✅ Added cleanup logic in chat manager
- ✅ Made editor compatible with agent's model selection
