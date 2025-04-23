# AI Shell Agent

---

**🚀 Version 0.2.0 Released! 🚀**

We're excited to announce the release of `ai-shell-agent` version 0.2.0! This version brings greatly improved functionality, modularity, and new features including enhanced toolsets, better configuration management, and localization support.

**Upgrade is highly recommended for all users:**

```bash
pip install --upgrade ai-shell-agent
```

---

**AI Shell Agent** is a command-line LLM-powered assistant designed to streamline your development and system administration tasks directly within your terminal. It interacts with your system through modular **Toolsets**, understanding your requests, planning actions, and leveraging capabilities like **Terminal execution**, **File System Management** (both with user confirmation), and experimental **AI-powered file editing** (via `aider-chat`). It maintains conversation context, adapts its available actions based on enabled toolsets, and prioritizes safety through Human-in-the-Loop (HITL) verification for potentially impactful operations.

## Philosophy

*   **Safety First (HITL):** Critical operations like running terminal commands or deleting/overwriting files often require your explicit confirmation. Review and even edit proposed actions before they run.
*   **Modular & Extensible:** Easily enable/disable capabilities (Toolsets) like Terminal, File Management, or the experimental Aider Code Editor per chat session or globally. New toolsets can be added by developers.
*   **Context-Aware:** Organizes interactions into distinct chat sessions, each with its own history, enabled toolsets, and configurations, allowing you to manage different projects or tasks separately.
*   **Seamless File Editing (Experimental):** Integrates `aider-chat` for sophisticated, AI-driven code and text file manipulation. *Note: This feature is experimental, potentially less intuitive, may not be enabled by default, and requires careful usage.*
*   **Multi-LLM & Configurable:** Supports various OpenAI (GPT) and Google AI (Gemini) models for both the main agent and dedicated translation tasks. Configure models, API keys, default toolsets, and language preferences easily.
*   **Cross-Platform:** Designed to work on Windows, Linux, and macOS, providing OS-specific guidance to the AI for relevant terminal commands.
*   **Internationalization:** Supports multiple languages for the user interface and includes a feature to automatically generate translations using an LLM.

---

*   [Features](#features)
*   [Quickstart Guide](#quickstart-guide)
*   [Installation](#installation)
*   [Usage (Command Reference)](#usage-command-reference)
*   [Toolsets](#toolsets)
*   [Localization](#localization)
*   [Development & Contributing](#development--contributing)
*   [Warning](#warning)
*   [License](#license)

---

## Features

*   **Modular Toolsets**: Extensible architecture (Terminal, File Manager, AI Code Copilot [Aider, Experimental], Cognitive).
*   **Safe Terminal Interaction (HITL)**: Execute shell commands and Python snippets after user review and optional editing/confirmation.
*   **Safe File Management (HITL)**: Create, read, delete, copy, move, rename files/directories. Destructive/modifying actions require user confirmation. Includes file history and backup/restore for edits (overwrite, find/replace).
*   **AI-Powered File Editing (Aider, Experimental)**: Integrates `aider-chat` for complex, multi-file code editing. *Use with caution.*
*   **Cognitive Tools**: Internal tools (`analyse`, `plan`) for the agent to improve reasoning and task breakdown before execution.
*   **Multi-LLM Support**: Choose from compatible OpenAI (e.g., `gpt-4o`, `gpt-4o-mini`) or Google AI (e.g., `gemini-1.5-pro`) models.
*   **Separate Translation Model**: Configure a specific LLM for UI localization tasks.
*   **System-Aware Prompting**: Detects OS (Windows, Linux, macOS) and provides relevant Terminal context to the AI.
*   **Persistent Chat Management**: Create, load, rename, list, delete distinct chat sessions.
*   **Per-Chat Toolset Selection**: Activate specific toolsets for different tasks or projects, overriding global defaults.
*   **Global & Per-Chat Configuration**: Set application defaults and manage toolset-specific settings (e.g., Aider models, File Manager history limit).
*   **Secure API Key Management**: Stores keys in `.env` in the project root; prompts securely when needed.
*   **Multi-Language UI**: Select application language interactively.
*   **Automated Localization**: Command (`--localize`) to generate UI translations for new languages using an LLM.
*   **Message Editing**: Correct your last *human* message and resend (`--edit last "..."`).
*   **Temporary Chats**: Quick, disposable chat sessions that can be easily flushed (`-tc`, `--temp-flush`).
*   **Direct Command Execution**: Option to bypass AI/HITL for immediate command execution (`-x`, `--execute`).

---

## Quickstart Guide

### 1. First-Time Setup

Run `ai` in your terminal for the first time. You'll be guided through a setup wizard:

1.  **Select Language**: Choose the UI language (requires restart if changed).
    ```console
    SYSTEM: Please select the application language:
      1: en <- Current
      2: pl # Example - other languages may appear if localized
    Enter number (1-2) or leave empty to keep 'en'> : [CURSOR]
    ```
2.  **Select Default AI Model**: Choose the main LLM for the agent (e.g., `gpt-4o-mini`).
    ```console
    SYSTEM: Please select the ai-shell-agent llm model from the available:
    SYSTEM: OpenAI:
    - gpt-4o (aliases: 4o)
    - gpt-4o-mini (aliases: 4o-mini) <- Current Model
    SYSTEM: Google:
    - gemini-1.5-pro
    Please input the model you want to use, or leave empty to keep using 'gpt-4o-mini'
    > : [CURSOR]
    ```
3.  **Provide API Key**: Enter the API key for the chosen model's provider (e.g., OpenAI API Key). It will be securely saved to a `.env` file in the agent's installation directory.
    ```console
    SYSTEM: Configuration required: Missing environment variable 'OPENAI_API_KEY'.
    INFO: Description: OpenAI API Key (used if GPT models are selected for Aider)
    Please enter the value for OPENAI_API_KEY: ****[INPUT HIDDEN]****
    INFO: Value for 'OPENAI_API_KEY' saved.
    ```
4.  **Select Default Enabled Toolsets**: Choose which toolsets are active by default when you create *new* chats (e.g., Terminal, File Manager, Cognitive). Aider is available but likely not included in the defaults due to its experimental nature.
    ```console
    SYSTEM: Please select the default enabled toolsets.
    These toolsets will be enabled by default when you create new chats.
    SYSTEM: Available Toolsets:
      1: AI Code Copilot **EXPERIMENTAL** - Provides tools for interacting with the AI Code Copilot for editing code and scripts.
      2: Cognitive                    - Provides internal tools for agent analysis and planning.
      3: File Manager                 - Provides tools for direct file and directory manipulation (create, read, edit, delete, copy, move, find, history).
      4: Terminal                     - Provides tools to execute shell commands and Python code.
    Enter comma-separated numbers TO ENABLE by default (e.g., 1,3).
    To disable all - enter 'none'.
    Leave empty to use the current defaults: Cognitive, File Manager, Terminal. # Example default

    > : 2,3,4 [CURSOR] # Example: Enabling Cognitive, Files, Terminal
    ```

### 2. Basic Interaction

*   **Start a new chat or talk in the current one:**
    ```bash
    # Start a new chat session named "my-project" (or load if exists)
    ai -c "my-project"

    # Ask a question or give an instruction within the current chat
    ai "What files are in the current directory?"
    ```
    The AI will analyze your request. If it needs to run a command (like `ls` or `dir`), it will propose it using the **Terminal** toolset (if enabled).

*   **Human-in-the-Loop (HITL) Example (Terminal):**
    ```bash
    ai "Show current path"
    ```
    *(Agent's interaction might look like this)*
    ```console
    AI: Used tool 'terminal_usage_guide'
    AI: ⏳ Thinking...
    SYSTEM:  AI wants to perform an action 'run_terminal_command', edit or confirm command: pwd
    (edit or confirm command) > : pwd [CURSOR_IS_HERE]
    ```
    *   Press **Enter** to confirm the command `pwd`.
    *   **Edit** the command if needed (e.g., change to `ls -l`), then press Enter.
    *   Press **Ctrl+C** to cancel the action.

    *(If confirmed, the agent executes the command and responds)*
    ```console
    AI: Used tool 'run_terminal_command' with cmd: 'pwd'
    TOOL: (run_terminal_command) Executed: `pwd`\nOutput:\n---\n/home/user/your_project\n---
    AI: ⏳ Thinking...
    AI: The current path is /home/user/your_project.
    ```

### 3. Using Toolsets (Examples)

*   **File Manager:**
    ```bash
    # Create a file (no confirmation needed)
    ai "Create a file named notes.md with the content 'Initial thoughts.'"

    # Delete a file (requires confirmation)
    ai "Delete the file old_notes.txt"
    ```
    *(Agent's interaction)*
    ```console
    AI: Used tool 'file_manager_usage_guide'
    AI: ⏳ Thinking...
    SYSTEM: AI wants to perform action 'delete_file_or_dir'.
    Delete file: '/path/to/your/project/old_notes.txt'?
    (confirm: yes/no) > : [CURSOR]
    ```
    *   Type `yes` and press Enter to confirm deletion.

*   **AI Code Copilot (Aider, Experimental):**
    *Note: This toolset might need to be manually enabled (`ai --select-tools`) and configured (`ai --configure-toolset ...`). Its HITL flow is more complex and less polished.*
    ```bash
    # Start a chat specific to coding task
    ai -c "fix-bug-123"
    # (Ensure Aider toolset is enabled for this chat)
    ai --select-tools # Select "AI Code Copilot **EXPERIMENTAL**" if needed

    # Add files to Aider's context
    ai "Add main.py to the code editor context"
    # AI uses 'add_file_to_copilot_context'

    # Request a simple edit
    ai "In main.py, add a print statement 'Starting...' at the beginning of the main function."
    # AI uses 'request_copilot_edit'
    # Aider starts working. It might ask for clarification:
    ```
    *(Agent's interaction)*
    ```console
    AI: Used tool 'aider_usage_guide'
    AI: ⏳ Thinking...
    AI: [CODE_COPILOT_INPUT_REQUEST] AI Code Copilot requires input. Please respond using 'respond_to_code_copilot_input_request'. Prompt: 'Apply the changes? (yes/no/details)' Options: (yes/no/all/skip/don't ask)
    ```
    ```bash
    # Respond to Aider's prompt (requires confirmation of YOUR response)
    ai "Respond to the code copilot: yes"
    ```
    *(Agent's interaction)*
    ```console
    AI: ⏳ Thinking...
    SYSTEM:  AI wants to perform an action 'respond_to_code_copilot_input_request', edit or confirm response: yes
    (edit or confirm response) > : yes [CURSOR] # Press Enter
    ```
    *(Aider continues with the confirmed input. It might finish or ask again.)*

    ```bash
    # View the changes made by Aider
    ai "Show the diff of the changes made"
    # AI uses 'view_code_copilot_edit_diffs'

    # Close the Aider session when done
    ai "Close the code copilot"
    ```

### 4. Configuration & Management

*   **Change Main AI Model:** `ai --select-model`
*   **Change Translation Model:** `ai --select-translation-model`
*   **Update API Key (for current main model):** `ai -k` or `ai --set-api-key`
*   **List Available Toolsets & Status:** `ai --list-toolsets`
*   **Change Enabled Toolsets (Current Chat/Global):** `ai --select-tools`
*   **Configure Specific Toolset Settings:** `ai --configure-toolset "File Manager"` (Use the exact name from `--list-toolsets`)
*   **List Chat History:** `ai --list-messages`
*   **Edit Last Message:** `ai --edit last "My corrected message"`
*   **Execute Command Directly (Bypass AI/HITL):** `ai -x "git status"`

---

## Installation

Requires **Python 3.11+**.

```bash
pip install ai-shell-agent
```
*(Note: If you encounter issues related to specific toolsets like Aider, you might need to ensure its dependencies are met separately. Refer to the `aider-chat` documentation if necessary.)*

---

## Usage (Command Reference)

```
ai [OPTIONS] [MESSAGE]
```

**Main Interaction:**

*   `[MESSAGE]`: The message/prompt/instruction to send to the AI agent (if no other command is given).
*   `-m, --send-message "MSG"`: Explicitly send a message to the current chat.
*   `-tc, --temp-chat "MSG"`: Start a temporary chat session with the message.
*   `-e, --edit IDX|"last" "MSG"`: Edit message `IDX` (integer) or the `last` human message and resend.
*   `-x, --execute "CMD"`: Execute the shell command `CMD` directly using the Terminal tool's backend (no AI/HITL). Output is added to chat history.
*   `-lsm, --list-messages`: Show the history of the current chat.

**Chat Management:**

*   `-c, --chat TITLE`: Create a new chat session with `TITLE` or load an existing one. Sets it as the current chat.
*   `-lc, --load-chat TITLE`: Same as `-c`.
*   `-lsc, --list-chats`: List all available chat session titles.
*   `-rnc, --rename-chat OLD_TITLE NEW_TITLE`: Rename a chat session.
*   `-delc, --delete-chat TITLE`: Delete a chat session and its history.
*   `--temp-flush`: Delete all chats whose titles start with "Temp Chat ".
*   `-ct, --current-chat-title`: Print the title of the currently active chat.

**Model & Language Configuration:**

*   `-llm, --model MODEL_NAME`: Set the main agent LLM (e.g., `gpt-4o-mini`, `gemini-1.5-pro`). Prompts for API key if needed for the new model.
*   `--select-model`: Interactively select the main agent LLM from a list.
*   `-k, --set-api-key [API_KEY]`: Set/update the API key for the *currently configured* main agent model. Prompts if `API_KEY` is omitted.
*   `--select-translation-model`: Interactively select the LLM used for the `--localize` command.
*   `--select-language`: Interactively select the application's display language.

**Toolset Management:**

*   `--list-toolsets`: List available toolsets and their enabled status (global or for the current chat).
*   `--select-tools`: Interactively enable/disable toolsets (globally if no chat active, otherwise for the current chat *and* updates global defaults).
*   `--configure-toolset TOOLSET_NAME`: Run the interactive configuration wizard for a specific toolset (use the exact name from `--list-toolsets`). Affects global defaults or the current chat's settings.

**Utilities:**

*   `--localize LANGUAGE_NAME`: Generate localized UI text files for the target language (e.g., `"Polish"`, `"Spanish"`, `"Pirate Speak"`). The name provided is used in the filename (e.g., `polish_texts.json`). Uses the configured translation model.

---

## Toolsets

Toolsets provide the agent's capabilities. They can be managed per-chat or globally.

**Available Toolsets (Default enabled toolsets may vary based on setup):**

*   **`Terminal`**: Allows execution of shell commands and Python code snippets. Uses HITL for safety. Provides OS-specific context to the AI.
*   **`File Manager`**: Enables direct file system operations (create, read, delete, copy, move, rename, find). Uses HITL for destructive/modifying actions. Includes file history and backup/restore features.
*   **`AI Code Copilot **EXPERIMENTAL**`**: (May not be enabled by default) Integrates `aider-chat` for advanced, context-aware file editing and creation within a persistent session. Uses a specialized, complex HITL workflow. *Use with caution and expect potential usability issues.*
*   **`Cognitive`**: Provides internal tools (`analyse`, `plan`) for the agent to improve its reasoning and planning before taking action. Outputs are internal notes, not shown directly to the user.

**Managing Toolsets:**

*   Use `ai --list-toolsets` to see available and enabled toolsets.
*   Use `ai --select-tools` to enable/disable toolsets for the current chat (if active) or globally.
*   Use `ai --configure-toolset "Toolset Name"` to adjust settings specific to a toolset (like Aider's models).

---

## Localization

The AI Shell Agent interface can be displayed in multiple languages.

*   **Selecting Language:** Use `ai --select-language` for an interactive prompt. A restart is required after changing the language.
*   **Generating Translations:** For developers or users wanting to add/update translations:
    1.  Ensure the desired translation LLM model is selected (`ai --select-translation-model`).
    2.  Ensure the API key for the translation model provider is set in `.env`.
    3.  Run `ai --localize "LANGUAGE_NAME"` (e.g., `ai --localize "Polish"`, `ai --localize "Pirate Speak"`). The name provided dictates the generated filename (e.g., `polish_texts.json`, `pirate speak_texts.json`).
    4.  This will find all `en_texts.json` files (in the core agent and toolsets), translate their string values using the LLM, and save new `<language_name>_texts.json` files. Tool names and other specific keys are excluded from translation.

---

## Development & Contributing

*   **Structure:** The agent is built with Python, leveraging libraries like `langchain`, `rich`, and `prompt_toolkit`. Toolsets are located in `ai_shell_agent/toolsets/`. Core logic is in `ai.py`, `chat_manager.py`, `llm.py`. State is managed via `config_manager.py` and `chat_state_manager.py`. UI via `console_manager.py`.
*   **Adding Toolsets:** Follow the detailed guide: [How to Add a New Toolset](how_to_add_toolsets.md)
*   **Dependencies:** Install development dependencies if contributing: `pip install -r requirements-dev.txt` (if available).

---

## Warning

This tool interacts directly with your system (terminal, file system). While safety measures like Human-in-the-Loop (HITL) confirmation are implemented for many actions, **always review proposed commands and file operations carefully before confirming.** The developers are not responsible for any data loss or system damage caused by misuse or unexpected behavior. Use at your own risk, especially the **experimental** AI Code Copilot (Aider) integration which may have rough edges and a complex interaction model.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
