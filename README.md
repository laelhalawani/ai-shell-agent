# AI Shell Agent

**AI Shell Agent** is a command-line LLM-powered assistant designed to help you with tasks by interacting with your system through modular **Toolsets**. It can understand your requests, plan actions, utilize tools like a **Terminal** (to execute commands with your confirmation) and a **File Editor** (powered by `aider-chat`), and respond intelligently directly from your console.

It features a simple CLI, persistent chat sessions, and adapts its available actions based on the toolsets enabled for the current chat.

Works on Windows, Linux, and macOS. (Tested primarily on Windows, contributions for Linux/macOS testing are welcome!)

*   [Features](#features)
*   [Quickstart Guide](#quickstart-guide)
*   [Installation](#installation)
*   [Usage (Command Reference)](#usage-command-reference)
*   [Toolsets](#toolsets)
*   [Development & Contributing](#development--contributing)
*   [License](#license)

---

## Features

*   **Modular Toolsets**: Extensible architecture allowing the AI to use different sets of capabilities (e.g., Terminal, File Editor).
*   **Terminal Interaction (with HITL)**: The AI can propose and execute shell commands or Python snippets via the Terminal toolset, but **only after you review, edit, or confirm** them (Human-in-the-Loop).
*   **AI-Powered File Editing**: Integrates with `aider-chat` library via the File Editor toolset to create, edit, and manage text/code files based on your instructions.
*   **Multiple AI Model Support**: Choose between OpenAI models (e.g., `gpt-4o`, `gpt-4o-mini`) and Google AI models (e.g., `gemini-1.5-pro`).
*   **System-Aware Prompting**: Automatically detects your OS (Windows, Linux, macOS) and provides relevant context to the AI when using the Terminal toolset.
*   **Persistent Chat Management**: Create, load, rename, list, and delete chat sessions to organize different tasks or projects. Each chat maintains its own history and enabled toolsets.
*   **Toolset Configuration**: Configure toolset-specific settings (like models used by the File Editor) and manage which toolsets are enabled by default or for specific chats.
*   **API Key Management**: Securely stores API keys (OpenAI, Google) in a `.env` file in the installation directory. Prompts for keys when needed.
*   **Message Editing**: Easily edit your last message and resend it for correction or clarification.
*   **Temporary Chats**: Quickly start one-off chat sessions that can be easily flushed.

---

## Quickstart Guide

### 1. First-Time Setup

When you run `ai` for the first time, you'll be guided through a setup:

1.  **Select Default AI Model**: Choose the primary LLM you want the agent to use.
    ```
    SYSTEM: Available models:
    SYSTEM: OpenAI:
    - gpt-4o (aliases: 4o)
    - gpt-4o-mini (aliases: 4o-mini) <- Current Model
    SYSTEM: Google:
    - gemini-1.5-pro
    - gemini-2.5-pro-exp-03-25

    Please input the model you want to use, or leave empty to keep using 'gpt-4o-mini':
    > : [CURSOR_IS_HERE]
    ```
2.  **Provide API Key**: Enter the API key for the selected model provider (OpenAI or Google). It will be saved securely in a `.env` file within the agent's installation directory.
    ```
    SYSTEM: Configuration required: Missing environment variable 'OPENAI_API_KEY'.
    INFO: Description: OpenAI API Key (https://platform.openai.com/api-keys)
    Please enter the value for OPENAI_API_KEY: ****[INPUT HIDDEN]****
    ```
3.  **Select Default Enabled Toolsets**: Choose which toolsets (e.g., Terminal, File Editor) should be enabled by default whenever you create a *new* chat session. You can always change this per chat later.
    ```
    SYSTEM: --- Select Default Enabled Toolsets ---
    These toolsets will be enabled by default when you create new chats.
    SYSTEM: Available Toolsets:
      1: File Editor     - Provides tools for editing and managing code files using AI
      2: Terminal        - Provides tools to execute shell commands and Python code.
    SYSTEM: Enter comma-separated numbers TO ENABLE by default (e.g., 1,3).
    To enable none by default, leave empty or enter 'none'.
    > : [CURSOR_IS_HERE]
    ```

### 2. Basic Interaction

*   **Send a message / Ask a question:**
    ```bash
    ai "How can I list files in the current directory sorted by size?"
    ```
    The AI will respond. If it needs to use a tool (like the Terminal), it will activate it and propose actions (which you'll need to confirm).

*   **Create or Load a Chat:** Keep conversations organized.
    ```bash
    ai -c "my_project_debug" # Creates 'my_project_debug' if new, otherwise loads it
    ai "Help me find the error in main.py"
    ```

### 3. Using Toolsets

*   **List Available Toolsets:** See what capabilities are installed.
    ```bash
    ai --list-toolsets
    ```
*   **Select Enabled Toolsets for Current Chat:** Change which toolsets the AI can use *in this specific chat*.
    ```bash
    ai --select-tools
    ```
    *(This will present an interactive prompt similar to the first-time setup)*

*   **Example: Using the Terminal:**
    ```bash
    # Agent might activate Terminal itself, or you can ask:
    ai "Start the terminal"
    # Then, ask for a command:
    ai "List the python files here"
    # AI will propose a command (e.g., ls *.py or dir *.py)
    # You'll see a prompt like this:
    # SYSTEM: AI wants to perform an action 'run_terminal_command', edit or confirm: ls *.py
    # > : ls *.py[CURSOR_IS_HERE]
    # Press Enter to confirm, edit the command, or Ctrl+C to cancel.
    ```

*   **Example: Using the File Editor:**
    ```bash
    ai -c "edit_config"
    ai "Start the file editor"
    ai "Add the file config.yaml to the editor" # AI uses 'submit_file_to_editor' tool
    ai "Request edits to change the port number to 8080" # AI uses 'request_edits' tool
    # The editor will run and might ask for confirmation via HITL prompts.
    ai "Close the file editor"
    ```

### 4. Other Useful Commands

*   **Update your API Key:**
    ```bash
    ai -k # Prompts for the key corresponding to the *currently selected* model
    ```*   **Change the AI Model:**
    ```bash
    ai --select-model # Interactive selection
    # OR
    ai --model "gpt-4o" # Set directly
    ```
*   **Execute a command directly (Bypasses AI/HITL):**
    ```bash
    ai -x "git status" # Runs 'git status' immediately and shows output
    ```

---

## Installation

```bash
pip install ai-shell-agent
```
This installs the `ai` command. Requires Python 3.11+.

Ensure your Python scripts directory is in your system's PATH environment variable.

---

## Usage (Command Reference)

**Default Action:**

*   `ai "<MESSAGE>"`
    *   Sends `<MESSAGE>` to the currently active chat session. If no chat is active, starts a temporary chat. The AI will respond and may use enabled tools (potentially triggering HITL prompts).

**Model Configuration:**

*   `ai --model <MODEL_NAME_OR_ALIAS>` / `ai -llm <MODEL_NAME_OR_ALIAS>`
    *   Sets the default AI model (e.g., `gpt-4o`, `4o-mini`, `gemini-1.5-pro`).
*   `ai --select-model`
    *   Starts an interactive prompt to choose the default AI model.
*   `ai -k [OPTIONAL_API_KEY]` / `ai --set-api-key [OPTIONAL_API_KEY]`
    *   Sets or updates the API key for the *currently selected* model provider (OpenAI/Google). Prompts interactively if `[OPTIONAL_API_KEY]` is omitted. Saves to `.env`.

**Chat Management:**

*   `ai -c <TITLE>` / `ai --chat <TITLE>` / `ai --load-chat <TITLE>` / `ai -lc <TITLE>`
    *   Creates a new chat session named `<TITLE>` or loads an existing one, making it the active session.
*   `ai -lsc` / `ai --list-chats`
    *   Lists all saved chat session titles.
*   `ai -rnc <OLD_TITLE> <NEW_TITLE>` / `ai --rename-chat <OLD_TITLE> <NEW_TITLE>`
    *   Renames a chat session.
*   `ai -delc <TITLE>` / `ai --delete-chat <TITLE>`
    *   Deletes the chat session with the given `<TITLE>`.
*   `ai -ct` / `ai --current-chat-title`
    *   Prints the title of the currently active chat session.
*   `ai -tc "<INITIAL_MESSAGE>"` / `ai --temp-chat "<INITIAL_MESSAGE>"`
    *   Starts a new temporary chat session with an initial message. Temporary chats are named `Temp Chat ...` and can be removed later.
*   `ai --temp-flush`
    *   Deletes all saved temporary chat sessions (except the currently active one, if it's temporary).

**Messaging & Interaction:**

*   `ai -m "<MESSAGE>"` / `ai --send-message "<MESSAGE>"`
    *   Explicitly sends `<MESSAGE>` to the active chat (same as default action).
*   `ai -e <INDEX|last> "<NEW_MESSAGE>"` / `ai --edit <INDEX|last> "<NEW_MESSAGE>"`
    *   Edits the message at the specified `<INDEX>` (0-based) or the `last` *human* message in the active chat history. Replaces the message content with `<NEW_MESSAGE>` and triggers the AI to respond again from that point.
*   `ai -lsm` / `ai --list-messages`
    *   Prints the formatted message history for the active chat session.
*   `ai -x "<COMMAND>"` / `ai --execute "<COMMAND>"`
    *   **Directly executes** the shell `<COMMAND>` in your system's terminal and prints the output. **This bypasses the AI and the HITL confirmation.** Use with caution.

**Toolset Management:**

*   `ai --list-toolsets`
    *   Lists all available toolsets found in the installation, showing their status (Enabled/Disabled, Active/Inactive) *for the current chat*.
*   `ai --select-tools`
    *   Starts an interactive prompt to enable or disable toolsets *for the current chat session*. Changes apply immediately for subsequent interactions in that chat. This also updates the global default setting for new chats.
*   `ai --configure-toolset <TOOLSET_NAME>`
    *   Runs the interactive configuration wizard for the specified `<TOOLSET_NAME>` (e.g., "File Editor", "Terminal"). This allows setting toolset-specific options (like models for the File Editor) and ensures required secrets are present. Configuration is saved for the current chat *and* as the global default for new chats.

---

## Toolsets

AI Shell Agent uses toolsets to provide specific capabilities.

*   **Terminal (`terminal`)**:
    *   Allows the AI to propose running shell commands (`run_terminal_command`) and Python code snippets (`python_repl`).
    *   **Requires user confirmation (HITL)** before execution. You can edit the proposed command/code before confirming.
    *   Provides OS-specific context to the AI.
*   **File Editor (`aider`)**:
    *   Integrates `aider-chat` for AI-powered file editing.
    *   Tools: `open_file_editor`, `submit_file_to_editor`, `withdraw_file_from_editor`, `list_files_in_editor`, `request_edits`, `submit_editor_input` (HITL), `view_edits_diff`, `undo_last_edit`, `close_file_editor`.
    *   Requires configuration (`ai --configure-toolset "File Editor"`) to set models used by Aider and ensure API keys.

*(New toolsets can be added by contributors - see below)*

---

## Development & Contributing

Interested in contributing or adding new toolsets?

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/laelhalawani/ai-shell-agent.git
    cd ai-shell-agent
    ```
2.  **Install in editable mode:**
    ```bash
    pip install -e .
    ```
3.  **Adding New Toolsets:** Please see the detailed guide: [CONTRIBUTING_TOOLSETS.md](CONTRIBUTING_TOOLSETS.md) *(Assuming the guide is saved with this name)*

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request on the GitHub repository.

---

## Warning

**Please use AI Shell Agent at your own risk.** While the HITL mechanism adds a layer of safety for AI-proposed commands, LLMs can still generate incorrect or unexpected commands or code edits. **Always review commands and edits carefully before confirming execution.** If you encounter dangerous suggestions, please report them by opening an issue.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.