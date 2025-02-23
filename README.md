# AI Shell Agent

**AI Shell Agent** is a command-line LLM-powered tool that can help you perform tasks by writing and executing terminal commands (with human confirmation or edit) and responding to questions, directly from the console.  
It features a very simple CLI and adjusts the LLM prompts based on your detected system.  
Works on Windows, Linux with Bash, and Mac. (Tested on Windows, please contribute!)

### Installation

```bash
pip install ai-shell-agent
```
This will automatically install the CLI tool in your current Python environment.  
Requires `python=3.11.x`.  
You can also clone and install from the repository.

### Quickly send messages

```bash
ai "your message here"
```
This will send a message to the AI in the active chat (and create a new chat if there isn't one active).  

You will see the AI response or editable commands that the AI wants to run, which you can confirm by pressing Enter.  

Output of the command is displayed in the console and added to the chat messages.  
Once all the commands are run, the AI will provide its interpretation of the results or try to run more commands.

If you haven't set your API key yet, you will be prompted.

### Execute command yourself and ask about the outputs

```bash
ai -x "dir"
```
This will execute the command and add the output to the AI logs, as it can't see the whole console.

```bash
ai "tell me about these files"
```
Will present both the command output and the question to the AI.  

You can run multiple commands in a row and then ask your question too.  
Or even run a few commands yourself and then ask the AI to finish up.

### Titled chats

```bash
ai -c "title of new or existing chat"
ai "your message here"
```
Will create a new chat and set it active if it doesn't exist, then send a message to the active chat.

### Temporary chats

```bash
ai -tc "your first message in a temporary chat"
```
Will create a new temporary chat without a title and set it active.

### Edit last message

```bash
ai -e "updated last message"
```
Will update the last message and send the updated chat to the llm to reply. You can also specify the user message id you want to update. It's displayed after each message you send, and when you list messages with `ai -lsm`.

---

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

**Please use at your own risk. AI can still generate wrong and possibly destructive commands. You always can view the command before sending—please be mindful. If you see any dangerous commands, please post a screenshot.**

---

## Features

- **Chat Session Management:**  
  Create new chats or load existing ones using a title, have one active chat session set to receive messages by default.

- **API Key Management:**  
  Set and update your OpenAI API key via a dedicated command. You will be prompted to input the key if you have not provided it yet.

- **Message Handling:**  
  Send new messages or edit previous ones within an active session with the simple `ai "your message"` command.

- **Temporary Sessions:**  
  Start temporary sessions for quick, ephemeral chats (currently saved as temp chats under UUID names for easier debugging and tracing).

- **Shell Command Execution:**  
  The LLM can write your commands, and you can edit them or execute them with one press of a button.

- **Python Code Execution:**  
  The agent also has the ability to run Python REPL, though this feature hasn't undergone extensive development or testing.

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
ai -k
```

Shorthand:  
```bash
ai --set-api-key
```

### Starting a Chat Session

Create a new chat session with a title:

```bash
ai -c "My Chat Session"
```

Shorthand:  
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
ai -x "dir"
```

Shorthand:  
```bash
ai --execute "dir"
```

By automatically detecting your operating system (via Python’s `platform` library), AI Shell Agent customizes its console suggestions for Windows CMD, Linux bash, or macOS Terminal.

### Temporary Chat Sessions

Start a temporary session (untitled, currently saved to file but untitled):

```bash
ai -tc "Initial temporary message"
```

Shorthand:  
```bash
ai --temp-chat "Initial temporary message"
```

### Listing and Managing Sessions

- **List Sessions:**
  ```bash
  ai -lsc
  ```
  Shorthand:  
  ```bash
  ai --list-chats
  ```

- **Load an Existing Session:**
  ```bash
  ai -lc "My Chat Session"
  ```
  Shorthand:  
  ```bash
  ai --load-chat "My Chat Session"
  ```

- **Rename a Session:**
  ```bash
  ai -rnc "Old Title" "New Title"
  ```
  Shorthand:  
  ```bash
  ai --rename-chat "Old Title" "New Title"
  ```

- **Delete a Session:**
  ```bash
  ai -delc "Chat Title"
  ```
  Shorthand:  
  ```bash
  ai --delete-chat "Chat Title"
  ```

- **List messages:**
  ```bash
  ai -lsm
  ```
  Shorthand:  
  ```bash
  ai --list-messages
  ```

- **Show the current chat title:**
  ```bash
  ai -ct
  ```
  Shorthand:  
  ```bash
  ai --current-chat-title
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
  ai -k
  ```
  Shorthand:  
  ```bash
  ai --set-api-key
  ```

### Chat Session Management
- **Create or Load a Chat Session:**
  ```bash
  ai -c "Session Title"
  ```
  Shorthand:  
  ```bash
  ai --chat "Session Title"
  ```

### Messaging
- **Send a Message:**
  ```bash
  ai -m "Your message"
  ```
  Shorthand:  
  ```bash
  ai --send-message "Your message"
  ```

- **Edit a Message at a Given Index:**
  ```bash
  ai -e 1 "Updated message"
  ```
  Shorthand:  
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
  ai -x "your shell command"
  ```
  Shorthand:  
  ```bash
  ai --execute "your shell command"
  ```

---

## Development & Contributing

Follow the same steps as described earlier.

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
