# AI Shell Agent

**AI Shell Agent** is a command-line LLM-powered assistant designed to streamline your development and system tasks. It interacts with your system through modular **Toolsets**, understanding your requests, planning actions, and leveraging capabilities like **Terminal execution** (with confirmation), **AI-powered file editing** (via `aider-chat`), and **file system management**. It operates directly within your console, learns from chat history, and adapts its available actions based on the toolsets you enable.

## Philosophy

*   **Safety First (HITL):** Critical operations like running terminal commands or editing files require your explicit confirmation (Human-in-the-Loop), preventing accidental execution. You can review and even *edit* proposed actions before they run.
*   **Modular & Extensible:** Easily enable/disable capabilities (Toolsets) like Terminal, File Management, or Code Editing (Aider) per chat session or globally. New toolsets can be added.
*   **Seamless File Editing:** Deep integration with `aider-chat` allows for sophisticated, AI-driven code and text file manipulation directly from the command line, maintaining context within a chat session.
*   **Multi-LLM & Configurable:** Supports various OpenAI and Google models for both the main agent and translation tasks. Configure models, API keys, default toolsets, and language preferences easily.
*   **Cross-Platform:** Designed to work on Windows, Linux, and macOS, with OS-specific guidance provided to the AI for terminal commands.
*   **Persistent Context:** Organizes interactions into distinct chat sessions, each with its own history and enabled toolsets, allowing you to pick up where you left off.
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

*   **Modular Toolsets**: Extensible architecture (Terminal, File Manager, Aider Code Editor).
*   **Safe Terminal Interaction (HITL)**: Execute shell commands/Python after user review and confirmation.
*   **AI-Powered File Editing (Aider)**: Integrates `aider-chat` for complex file creation and editing.
*   **Direct File Management (HITL)**: Create, read, edit (simple replace), delete, copy, move, find files/directories with user confirmation for destructive actions. Includes history and backup/restore for edits.
*   **Multi-LLM Support**: Choose OpenAI (`gpt-4o`, `gpt-4o-mini`, etc.) or Google AI (`gemini-1.5-pro`, etc.) models.
*   **Separate Translation Model**: Configure a specific LLM for UI localization tasks.
*   **System-Aware Prompting**: Detects OS (Windows, Linux, macOS) for relevant Terminal context.
*   **Persistent Chat Management**: Create, load, rename, list, delete chat sessions.
*   **Per-Chat Toolset Selection**: Activate specific toolsets for different tasks or projects.
*   **Global & Per-Chat Configuration**: Set defaults for toolsets and manage toolset-specific settings (e.g., Aider models).
*   **Secure API Key Management**: Stores keys in `.env`; prompts when needed.
*   **Multi-Language UI**: Select application language.
*   **Automated Localization**: Generate UI translations for new languages using `--localize`.
*   **Message Editing**: Correct your last message and resend.
*   **Temporary Chats**: Quick, disposable chat sessions.
*   **Direct Command Execution**: Option to bypass AI/HITL for immediate command execution (`-x`).

---

## Quickstart Guide

### 1. First-Time Setup

Run `ai` for the first time to configure essentials:

1.  **Select Language**: Choose the UI language.
    ```
    SYSTEM: Please select the application language:
    SYSTEM:   1: en <- Current
      2: pl
    Enter number (1-2) or leave empty to keep 'en': : [CURSOR]
    ```
    *(If you change the language, you'll be asked to restart)*
2.  **Select Default AI Model**: Choose the main LLM (e.g., `gpt-4o-mini`).
    ```
    SYSTEM: Available models:
    SYSTEM: OpenAI:
    - gpt-4o (...)
    - gpt-4o-mini (...) <- Current Model
    SYSTEM: Google:
    - gemini-1.5-pro
    Please input the MAIN AGENT model, or leave empty to keep 'gpt-4o-mini':
    > : [CURSOR]
    ```
3.  **Provide API Key**: Enter the key for the chosen model's provider (saved to `.env`).
    ```
    SYSTEM: Configuration required: Missing environment variable 'OPENAI_API_KEY'.
    INFO: Description: OpenAI API Key (https://platform.openai.com/api-keys)
    Please enter the value for OPENAI_API_KEY: ****[INPUT HIDDEN]****
    ```
4.  **Select Default Enabled Toolsets**: Pick toolsets active by default in new chats (e.g., Terminal, File Manager).
    ```
    SYSTEM: --- Select Default Enabled Toolsets ---
    (...)
    SYSTEM: Available Toolsets:
      1: Aider Code Editor **EXPERIMENTAL** (...)
      2: File Manager (...)
      3: Terminal (...)
    Enter comma-separated numbers TO ENABLE by default (e.g., 1,3):
    > : [CURSOR]
    ```

### 2. Basic Interaction

*   **Ask a question or give an instruction:**
    ```bash
    ai "What are the top 5 largest files in my Downloads folder?"
    ```
    The AI will analyze, possibly enable the Terminal or File Manager toolset, propose actions (requiring your confirmation if needed), and provide the answer.

*   **Work within a project context:**
    ```bash
    ai -c "my-web-project" # Creates/loads the chat
    ai "Refactor the main api route in routes.py to use async await"
    # AI will likely activate File Editor, add routes.py, request the edit.
    ```

### 3. Using Toolsets (Examples)

*   **Terminal:**
    ```bash
    ai "Show running python processes"
    # AI activates Terminal, proposes command (e.g., ps aux | grep python)
    # SYSTEM: AI wants to perform an action 'run_terminal_command', edit or confirm: ps aux | grep python
    # > : ps aux | grep python [CURSOR_IS_HERE]
    # Press Enter to confirm, edit, or Ctrl+C to cancel.
    ```

*   **File Manager:**
    ```bash
    ai "Create a directory named 'docs'"
    # AI activates File Manager, proposes action
    # SYSTEM: AI wants to perform an action 'create_file_or_dir', edit or confirm: {'path': 'docs', 'is_directory': True} # Simplified prompt
    # > : docs [CURSOR_IS_HERE]
    # Confirm the path 'docs'.

    ai "Read the requirements.txt file"
    # AI uses 'read_file_content' (no confirmation needed for read)
    ```

*   **File Editor (Aider):**
    ```bash
    ai -c "fix-bug-123"
    ai "Add the file src/utils.py to the code editor"
    # AI uses 'add_file_to_copilot_context'
    ai "In src/utils.py, find the function calculate_total and add error handling for zero division. Explain the changes."
    # AI uses 'request_copilot_edit'
    # Aider runs, potentially asking for clarification via HITL prompts using 'respond_to_code_copilot_input_request'
    ai "Show me the diff of the changes"
    # AI uses 'view_code_copilot_edit_diffs'
    ai "Close the code editor"
    ```

### 4. Configuration & Management

*   **Change Main AI Model:**
    ```bash
    ai --select-model
    ```
*   **Change Translation Model:**
    ```bash
    ai --select-translation-model
    ```
*   **Update API Key:** (For the *currently set* main agent model)
    ```bash
    ai -k
    ```
*   **Manage Toolsets for Current Chat:**
    ```bash
    ai --select-tools # Enable/disable for this chat
    ai --configure-toolset "Aider Code Editor **EXPERIMENTAL**" # Configure Aider models
    ```
*   **Execute Directly:**
    ```bash
    ai -x "npm install" # Runs the command and adds output to the conversation, so AI can see it too
    ```

---

## Installation

Requires **Python 3.11+**.

```bash
pip install ai-shell-agent
```

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.