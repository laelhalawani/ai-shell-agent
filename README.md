# AI Shell Agent

**AI Shell Agent** is a command-line chat application that lets you interact with OpenAI’s language models directly from your terminal. Unlike libraries meant to be imported into other codebases, AI Shell Agent is built to be used as a standalone CLI tool—designed for a seamless, interactive experience.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quickstart Guide](#quickstart-guide)
- [Installation](#installation)
- [Usage](#usage)
- [Development & Contributing](#development--contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

AI Shell Agent allows you to:
- Create, load, rename, and delete chat sessions.
- Manage your OpenAI API key easily.
- Send and edit messages within chat sessions.
- Run temporary, in-memory chat sessions.
- Execute shell commands (with both interactive and direct modes).
- Run Python code snippets using an integrated REPL tool.

This project is designed exclusively as a command-line tool. Its entire interface is built around terminal commands, interactive prompts, and text-based feedback.

---

## Features

- **Chat Session Management:**  
  Create new chats or load existing ones using a title.

- **API Key Management:**  
  Set and update your OpenAI API key via a dedicated command.

- **Message Handling:**  
  Send new messages or edit previous ones within an active session.

- **Temporary Sessions:**  
  Start in-memory sessions for quick, ephemeral conversations.

- **Shell Command Execution:**  
  Execute system commands either directly or after interactive editing.

- **Python Code Execution:**  
  Run Python code snippets using an integrated Python REPL tool.

---

## Quickstart Guide

### Setting Up the API Key

Upon launching AI Shell Agent for the first time, if no API key is detected, the application will prompt you to enter it:

```bash
$ ai
No OpenAI API key found. Please enter your OpenAI API key:
```

After entering the key, it will be saved in a `.env` file located in the project's installation directory. This ensures that your API key is securely stored and automatically loaded in future sessions.

### Managing the API Key

If you need to update or set a new API key at any time, use the following command:

```bash
ai --set-api-key
```

This command will prompt you to enter the new API key and update the `.env` file accordingly.

### Starting a Chat Session

Create a new chat session with a title:
```bash
ai --chat "My Chat Session"
```

### Sending a Message

To send a message to the active chat session:
```bash
ai "what is the time right now?"
```

### Executing Shell Commands

Run a shell command directly:
```bash
ai --cmd "dir"   # (or the equivalent command for your OS)
```

By automatically detecting your operating system (via Python’s `platform` library), AI Shell Agent customizes its console suggestions for Windows CMD, Linux bash, or macOS Terminal. This ensures the suggested commands follow the conventions of your environment.

### Temporary Chat Sessions

Start a temporary session (in-memory):
```bash
ai --temp-chat "Initial temporary message"
```

### Listing and Managing Sessions

- **List Sessions:**
  ```bash
  ai --list-chats
  ```
- **Load an Existing Session:**
  ```bash
  ai --load-chat "My Chat Session"
  ```
- **Rename a Session:**
  ```bash
  ai --rename-chat "Old Title" "New Title"
  ```
- **Delete a Session:**
  ```bash
  ai --delete-chat "Chat Title"
  ```

---

## Installation

### Installing from PyPI

```bash
pip install ai-shell-agent
```

### Installing from Source

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ai-shell-agent.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd ai-shell-agent
    ```
3. **Install the package:**
    ```bash
    pip install .
    ```

---

## Usage

### API Key Management
- **Set or Update API Key:**
  ```bash
  ai --set-api-key
  ```

### Chat Session Management
- **Create or Load a Chat Session:**
  ```bash
  ai --chat "Session Title"
  ```

### Messaging
- **Send a Message:**
  ```bash
  ai --send-message "Your message"
  ```
- **Edit a Message at a Given Index:**
  ```bash
  ai --edit 1 "Updated message"
  ```

### System Prompt Management
- **Set Default System Prompt:**
  ```bash
  ai --default-system-prompt "Your default system prompt"
  ```

### Shell Command Execution
- **Direct Execution (without confirmation):**
  ```bash
  ai --cmd "your shell command"
  ```

### Python Code Execution
- **Run Python Code:**
  ```bash
  ai --run-python "print('Hello, World!')"
  ```

---

## Development & Contributing

### Setting Up the Development Environment
1. **Fork and Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/ai-shell-agent.git
    cd ai-shell-agent
    ```
2. **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run Tests:**
    ```bash
    pytest
    ```

---

## Acknowledgements

We would like to thank the following:
- [OpenAI](https://openai.com) for providing the API.
- [Python](https://www.python.org) for being an awesome programming language.
- All contributors who have provided feedback, bug reports, and improvements to the AI Shell Agent project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
