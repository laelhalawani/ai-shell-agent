# ai-shell-agent

A command-line AI chat application that allows you to interact with OpenAI's language models directly from your terminal. And help you running terminal commands and python code on it's own (it inputs it for you and you can still make your edits)

## Features

- Manage chat sessions
- Set and update system prompts
- Send and edit messages
- Temporary in-memory chat sessions
- API key management

## Installation

You can install the package from PyPI:

```bash
pip install ai-shell-agent
```

Alternatively, you can install it from the source:

```bash
git clone https://github.com/yourusername/ai-shell-agent.git
cd ai-shell-agent
pip install .
```

## Usage

### Setting the API Key

Before using the application, you need to set your OpenAI API key:

```bash
ai --set-api-key
```
If you don't set it up you will be prompted when trying to use other functionality.

### Quick conversation in session

To quickly initialize in-memory conversation, type

```bash
ai "your message here"
```

### Creating or Loading a Chat Session

To create or load a chat session with a specified title:

```bash
ai --chat "My Chat Session"
```

### Listing Available Chat Sessions

To list all available chat sessions:

```bash
ai --list-chats
```

### Renaming a Chat Session

To rename an existing chat session:

```bash
ai --rename-chat "Old Title" "New Title"
```

### Deleting a Chat Session

To delete a chat session:

```bash
ai --delete-chat "Chat Title"
```

### Setting the Default System Prompt

To set the default system prompt for new chats:

```bash
ai --default-system-prompt "Your default system prompt"
```

### Updating the System Prompt for the Active Chat Session

To update the system prompt for the active chat session:

```bash
ai --system-prompt "Your new system prompt"
```

### Sending a Message

To send a message to the active chat session:

```bash
ai --send-message "Your message"
```

### Starting a Temporary Chat Session

To start a temporary (in-memory) chat session with an initial message:

```bash
ai --temp-chat "Initial message"
```

### Editing a Previous Message

To edit a previous message at a given index:

```bash
ai --edit 1 "New message"
```

## Development

To contribute to the project, follow these steps:

1. Fork the repository.
2. Clone your forked repository:

    ```bash
    git clone https://github.com/yourusername/ai-shell-agent.git
    ```

3. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Run the tests:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI](https://openai.com) for providing the API.
- [Python](https://www.python.org) for being an awesome programming language.
