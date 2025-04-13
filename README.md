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

Please make sure your python scripts are added to path correctly. 

### Quick Examples

#### Send a message to AI
```bash
ai "your message here"
```
The AI will respond and may suggest commands that can help with your request.

#### Execute command yourself and add to context
```bash
ai -x "dir"
ai "tell me about these files"
```
This will execute the command and add the output to the AI logs, then you can ask about it.

#### Create a new chat for a different task
```bash
ai -c "Project Deployment"
ai "help me deploy my Flask app to Heroku"
```
This creates a dedicated chat for your task, keeping the conversation focused.

#### Start a temporary chat
```bash
ai -tc "how do I check disk space in Windows?"
```
Creates a quick chat session for one-off questions.

#### Edit your last message
```bash
ai -e "updated question with more details"
```
Lets you refine your last message if you forgot important details.

https://github.com/user-attachments/assets/6df08410-37e5-4e21-b99c-4133c15192cc

### First-Time Setup

When you first use AI Shell Agent, you'll be prompted to:

1. **Select an AI Model**:
   ```
   Available models:
   OpenAI:
   - gpt-4o-mini (aliases: 4o-mini) <- Current Model
   - gpt-4o (aliases: 4o)
   - o3-mini
   Google:
   - gemini-1.5-pro
   - gemini-2.5-pro
   
   Please input the model you want to use, or leave empty to keep using the current model gpt-4o-mini.
   > 
   ```

2. **Enter the appropriate API key**:
   After selecting a model, you'll be prompted for the corresponding API key (OpenAI or Google).
   The key will be saved to a local `.env` file for future sessions.

You can later change your model or API key using:
```bash
ai --select-model  # Interactive model selection
ai --model "gpt-4o"  # Directly set model
ai -k  # Update your API key
```

### Main Features

- **AI Writes and Executes Commands**: The AI will suggest commands to accomplish your tasks, which you can review, edit, or approve.
  
- **Multiple AI Model Support**: Choose between OpenAI models (gpt-4o, gpt-4o-mini, o3-mini) and Google AI models (gemini-1.5-pro, gemini-2.5-pro).
  
- **System Detection**: Automatically detects your OS and tailors commands to work with Windows CMD, Linux bash, or macOS Terminal.
  
- **Chat Management System**: Create, rename, list, and delete chat sessions to organize different tasks or projects.
  
- **Python Code Execution**: Run and evaluate Python code snippets (experimental feature).

- **AI-Powered Code Editing**: Make natural language code changes to your files using the integrated code editing capabilities. Add files to the session and describe the changes you want to make.

https://github.com/user-attachments/assets/049e6e37-5a5d-4125-b891-e1bb1f2ecdbf

## Warning

**Please use at your own risk. AI can still generate wrong and possibly destructive commands. You always can view the command before sending—please be mindful. If you see any dangerous commands, please post a screenshot.**

## Table of Contents

- [Features](#features)
- [Quickstart Guide](#quickstart-guide)
- [Installation](#installation)
- [Usage](#usage)
- [Code Editing](#code-editing)
- [Development & Contributing](#development--contributing)
- [License](#license)

---

## Features

- **AI Command Generation and Execution**:  
  The LLM suggests and executes terminal commands to accomplish your tasks, with your review and approval.

- **Multiple AI Model Support**:
  Choose between OpenAI and Google AI models with simple model selection commands.

- **System-Aware Prompting**:
  Automatically detects your operating system and optimizes commands for Windows, Linux, or macOS.

- **Chat Session Management**:  
  Create new chats or load existing ones using a title, have one active chat session set to receive messages by default.

- **API Key Management**:  
  Set and update your API keys (OpenAI or Google) via a dedicated command. You will be prompted to input the key if you have not provided it yet.

- **Message Handling**:  
  Send new messages or edit previous ones within an active session with the simple `ai "your message"` command.

- **Temporary Sessions**:  
  Start temporary sessions for quick, ephemeral chats (currently saved as temp chats under UUID names for easier debugging and tracing).

- **Python Code Execution**:  
  The agent also has the ability to run Python REPL, though this feature hasn't undergone extensive development or testing.

- **AI-Powered Code Editing**:  
  Make natural language code changes to your files using the integrated code editing capabilities. Add files to the session and describe the changes you want to make.

---

## Quickstart Guide

### Selecting a Model

On first run, AI Shell Agent will prompt you to select your preferred model:

```
Available models:
OpenAI:
- gpt-4o-mini (aliases: 4o-mini) <- Current Model
- gpt-4o (aliases: 4o)
- o3-mini
Google:
- gemini-1.5-pro
- gemini-2.5-pro

Please input the model you want to use, or leave empty to keep using the current model gpt-4o-mini.
> 
```

You can also change the model at any time:

```bash
ai --model "gpt-4o"  # or any supported model name/alias
```

### Setting Up the API Key

After selecting a model, the application will prompt you for the appropriate API key:

```bash
$ ai "Hi"
No OpenAI API key found. Please enter your API key.
You can get it from: https://platform.openai.com/api-keys
Enter OpenAI API key:
```

After entering the key, it will be saved in a `.env` file located in the project's installation directory. This ensures that your API key is securely stored and automatically loaded in future sessions.

### Managing the API Key

If you need to update or set a new API key at any time, use the following command:

```bash
ai -k
```

Shorthand:  
```bash
ai -k
```

### Starting a Chat Session

Create a new chat session with a title:

```bash
ai -c "My Chat Session"
```

Shorthand:  
```bash
ai -c "My Chat Session"
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
ai -x "dir"
```

By automatically detecting your operating system (via Python’s `platform` library), AI Shell Agent customizes its console suggestions for Windows CMD, Linux bash, or macOS Terminal.

### Code Editing

AI Shell Agent integrates code editing capabilities that allow you to make changes to your files using natural language instructions:

#### Initialize the Code Editor
```bash
ai "start a code editing session"
```

#### Add Files to Edit
```bash
ai "add file.py to the editor"
```

#### Edit Code with Natural Language
```bash
ai "add a function that calculates the factorial of a number"
```

#### List Files in the Editor
```bash
ai "list the files in the editor"
```

#### Remove Files from the Editor
```bash
ai "remove file.py from the editor"
```

#### View Changes
```bash
ai "show me the changes made to the files"
```

### Temporary Chat Sessions

Start a temporary session (untitled, currently saved to file but untitled):

```bash
ai -tc "Initial temporary message"
```

Shorthand:  
```bash
ai -tc "Initial temporary message"
```

### Listing and Managing Sessions

- **List Sessions:**
  ```bash
  ai -lsc
  ```
  Shorthand:  
  ```bash
  ai -lsc
  ```

- **Load an Existing Session:**
  ```bash
  ai -lc "My Chat Session"
  ```
  Shorthand:  
  ```bash
  ai -lc "My Chat Session"
  ```

- **Rename a Session:**
  ```bash
  ai -rnc "Old Title" "New Title"
  ```
  Shorthand:  
  ```bash
  ai -rnc "Old Title" "New Title"
  ```

- **Delete a Session:**
  ```bash
  ai -delc "Chat Title"
  ```
  Shorthand:  
  ```bash
  ai -delc "Chat Title"
  ```

- **List messages:**
  ```bash
  ai -lsm
  ```
  Shorthand:  
  ```bash
  ai -lsm
  ```

- **Show the current chat title:**
  ```bash
  ai -ct
  ```
  Shorthand:  
  ```bash
  ai -ct
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
    git clone https://github.com/laelhalawani/ai-shell-agent.git
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

### Model Selection
- **Set Model:**
  ```bash
  ai --model "gpt-4o"
  ```
  Shorthand:  
  ```bash
  ai -llm "gpt-4o"
  ```

- **Interactive Model Selection:**
  ```bash
  ai --select-model
  ```

### API Key Management
- **Set or Update API Key:**
  ```bash
  ai --set-api-key
  ```
  Shorthand:  
  ```bash
  ai -k
  ```
  This will prompt for the appropriate API key based on your selected model.

### Chat Session Management
- **Create or Load a Chat Session:**
  ```bash
  ai --chat "Session Title"
  ```
  Shorthand:  
  ```bash
  ai -c "Session Title"
  ```

- **Load an Existing Chat Session:**
  ```bash
  ai --load-chat "Session Title"
  ```
  Shorthand:
  ```bash
  ai -lc "Session Title"
  ```

- **List All Chat Sessions:**
  ```bash
  ai --list-chats
  ```
  Shorthand:
  ```bash
  ai -lsc
  ```

- **Rename a Chat Session:**
  ```bash
  ai --rename-chat "Old Title" "New Title"
  ```
  Shorthand:
  ```bash
  ai -rnc "Old Title" "New Title"
  ```

- **Delete a Chat Session:**
  ```bash
  ai --delete-chat "Chat Title"
  ```
  Shorthand:
  ```bash
  ai -delc "Chat Title"
  ```

- **Show Current Chat Title:**
  ```bash
  ai --current-chat-title
  ```
  Shorthand:
  ```bash
  ai -ct
  ```

### Messaging
- **Send a Message:**
  ```bash
  ai --send-message "Your message"
  ```
  Shorthand:  
  ```bash
  ai -m "Your message"
  ```
  Or simply:
  ```bash
  ai "Your message"
  ```

- **Start a Temporary Chat:**
  ```bash
  ai --temp-chat "Initial message"
  ```
  Shorthand:
  ```bash
  ai -tc "Initial message"
  ```

- **Edit Last Message:**
  ```bash
  ai --edit "Updated message"
  ```
  Shorthand:
  ```bash
  ai -e "Updated message"
  ```

- **Edit a Message at a Specific Index:**
  ```bash
  ai --edit 1 "Updated message"
  ```
  Shorthand:  
  ```bash
  ai -e 1 "Updated message"
  ```

- **List All Messages in Current Chat:**
  ```bash
  ai --list-messages
  ```
  Shorthand:
  ```bash
  ai -lsm
  ```

- **Clear All Temporary Chats:**
  ```bash
  ai --temp-flush
  ```

### System Prompt Management
- **Set Default System Prompt:**
  ```bash
  ai --default-system-prompt "Your default system prompt"
  ```

- **Update System Prompt for Active Chat:**
  ```bash
  ai --system-prompt "New system prompt for this chat"
  ```

### Shell Command Execution
- **Execute Shell Command (with context preservation):**
  ```bash
  ai --execute "your shell command"
  ```
  Shorthand:  
  ```bash
  ai -x "your shell command"
  ```

---

## Development & Contributing

Follow the same steps as described earlier.

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
