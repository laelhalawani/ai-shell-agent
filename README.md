# AI Shell Agent

**AI Shell Agent** is a command-line LLM powered tool that can help you perform tasks by writing and executing terminal commands (with human confirmation or edit) and respond to questions, directly from the console.
It features a very simple CLI, and adjust the LLM prompts based on your detected system.
Works on Windows, Linux with Bash, and Mac. (Tested on Windows, please contribute:)

### Installation

```bash
pip install ai-shell-agent
```
Will automatically install the CLI tool in your current python environment.
Requires `python=3.11.x`
You can also classically clone and install from the repo.

### Quickly send messages

```bash
ai "your message here"
```
Will send a message to the AI in the active chat (and create a new chat if there isn't one active)

You will see the AI response or editable commands that the AI wants to run, which you can confirm by pressing Enter.

Output of the command is displayed in the console, and added to the chat messages. 
Once all the commands are run, the AI will provide it's interpretation of the results or try to run more commands.

If you haven't set your API key yet, you will be prompted.

### Titled chats

```bash
ai -c "tile of or existing chat"
ai "your message here"
```
Will create a new chat and set it active if it doesn't exist, and , then send a message to active chat.

### Temporary chats

```bash
ai -t "your first message in a temporary chat"
```
Will create a new temporary chat without the title and set it active.




## Table of Contents

- [Features](#features)
- [Warning](#warning)
- [Quickstart Guide](#quickstart-guide)
- [Installation](#installation)
- [Usage](#usage)
- [Development & Contributing](#development--contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Warning

**Please use at your own risk, AI can still generate wrong and possibly destructive commands. You always are able to view the command before sending, please be mindful, it shouldn't be too bad, but if you see some terrible commands please post a screenshot**

---

## Features

- **Chat Session Management:**  
  Create new chats or load existing ones using a title, have one chat as active, set to receive messages by default

- **API Key Management:**  
  Set and update your OpenAI API key via a dedicated command, you will be prompted to input the key if you have not provided it yet

- **Message Handling:**  
  Send new messages or edit previous ones within an active session with super easy `ai "your message"` command

- **Temporary Sessions:**  
  Start temp sessions for quick, ephemeral chats (currently saved as temp chats under uuid names for easier debugging and tracing)

- **Shell Command Execution:**  
  LLM can write your commands, and you can edit them or execute with one press of a button.

- **Python Code Execution:**  
  Our agent has also ability to run Python REPL, but not much development and testing was directed at this feature, and it might perform subpair.

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
