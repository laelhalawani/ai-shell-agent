# Usage

This guide covers how to use AI Shell Agent's various features.

## Setting the API Key

Before using the application, you need to set your OpenAI API key:

```bash
ai --set-api-key
```

If you don't set it up, you will be prompted when trying to use other functionality.

## Quick Conversation in Session

To quickly initialize an in-memory conversation, type:

```bash
ai "your message here"
```

## Creating or Loading a Chat Session

To create or load a chat session with a specified title:

```bash
ai --chat "My Chat Session"
```

## Listing Available Chat Sessions

To list all available chat sessions:

```bash
ai --list-chats
```

## Renaming a Chat Session

To rename an existing chat session:

```bash
ai --rename-chat "Old Title" "New Title"
```

## Deleting a Chat Session

To delete a chat session:

```bash
ai --delete-chat "Chat Title"
```

## Setting the Default System Prompt

To set the default system prompt for new chats:

```bash
ai --default-system-prompt "Your default system prompt"
```

## Updating the System Prompt for the Active Chat Session

To update the system prompt for the active chat session:

```bash
ai --system-prompt "Your new system prompt"
```

## Sending a Message

To send a message to the active chat session:

```bash
ai --send-message "Your message"
```

## Starting a Temporary Chat Session

To start a temporary (in-memory) chat session with an initial message:

```bash
ai --temp-chat "Initial message"
```

## Editing a Previous Message

To edit a previous message at a given index:

```bash
ai --edit 1 "New message"
```
