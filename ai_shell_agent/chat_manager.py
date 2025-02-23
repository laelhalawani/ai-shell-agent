import os
import json
import uuid
import logging
import subprocess

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    BaseMessage
)
from .tools import tools_functions, direct_windows_shell_tool, tools
from .prompts import default_system_prompt

CHAT_DIR = os.path.join("chats")
CHAT_MAP_FILE = os.path.join(CHAT_DIR, "chat_map.json")
SESSION_FILE = "session.json"
CONFIG_FILE = "config.json"
MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# Ensure the chats directory exists.
os.makedirs(CHAT_DIR, exist_ok=True)

def _read_json(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def _write_json(file_path: str, data: dict) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def _get_console_session_id() -> str:
    """Returns an identifier for a temporary console session."""
    return f"temp_{os.getpid()}"

def _read_messages(file_path: str) -> list[BaseMessage]:
    """Read and deserialize messages from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                messages_data = json.load(f)
                messages = []
                for msg in messages_data:
                    if msg["type"] == "system":
                        messages.append(SystemMessage(**msg))
                    elif msg["type"] == "human":
                        messages.append(HumanMessage(**msg))
                    elif msg["type"] == "ai":
                        messages.append(AIMessage(**msg))
                return messages
            except json.JSONDecodeError:
                return []
    return []

def _write_messages(file_path: str, messages: list[BaseMessage]) -> None:
    """Write messages to JSON file."""
    messages_data = [msg.model_dump() for msg in messages]
    with open(file_path, "w") as f:
        json.dump(messages_data, f, indent=4)

# ---------------------------
# Chat Session Management
# ---------------------------
def set_current_chat(chat_file: str) -> None:
    """
    Sets the current chat session.
    
    Parameters:
      chat_file (str): The filepath of the chat session to set as current.
    """
    _write_json(SESSION_FILE, {"current_chat": chat_file})

def get_current_chat() -> str:
    """
    Gets the current chat session.
    
    Returns:
      str: The filepath of the current chat session, or None if not set.
    """
    data = _read_json(SESSION_FILE)
    return data.get("current_chat", None)

def create_or_load_chat(title: str) -> str:
    """
    Creates or loads a chat session file based on the title.
    If a new chat is created, the default system prompt is added as the first message.
    
    Parameters:
      title (str): The chat session title.
      
    Returns:
      str: The filepath of the chat session JSON file.
    """
    chat_map = _read_json(CHAT_MAP_FILE)
    if (title in chat_map):
        chat_id = chat_map[title]
        logging.debug(f"Loading existing chat session: {title}")
    else:
        chat_id = str(uuid.uuid4())
        chat_map[title] = chat_id
        _write_json(CHAT_MAP_FILE, chat_map)
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if not os.path.exists(chat_file):
        logging.info(f"Creating new chat session: {title}")
        # New chat: add default system prompt
        config = _read_json(CONFIG_FILE)
        if "default_system_prompt" not in config:
            config["default_system_prompt"] = default_system_prompt
            _write_json(CONFIG_FILE, config)
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        initial_messages = [SystemMessage(content=default_prompt)]
        _write_messages(chat_file, initial_messages)
    set_current_chat(chat_file)
    return chat_file

def get_chat_titles_list() -> list:
    """Returns a list of all chat session titles."""
    chat_map = _read_json(CHAT_MAP_FILE)
    chats = list(chat_map.keys())
    chats_str = "\n - ".join(chats)
    logging.info(f"Chats: \n{chats_str}")
    return chats

def rename_chat(old_title: str, new_title: str) -> bool:
    """
    Renames an existing chat session.
    
    Parameters:
      old_title (str): The current chat title.
      new_title (str): The new chat title.
      
    Returns:
      bool: True if successful, False otherwise.
    """
    chat_map = _read_json(CHAT_MAP_FILE)
    if old_title in chat_map:
        chat_map[new_title] = chat_map.pop(old_title)
        _write_json(CHAT_MAP_FILE, chat_map)
        logging.info(f"Chat session renamed: {old_title} -> {new_title}")
        return True
    logging.error(f"Chat session not found: {old_title}")
    return False

def delete_chat(title: str) -> bool:
    """
    Deletes a chat session.
    
    Parameters:
      title (str): The title of the chat to delete.
      
    Returns:
      bool: True if successful, False otherwise.
    """
    chat_map = _read_json(CHAT_MAP_FILE)
    if title in chat_map:
        chat_id = chat_map.pop(title)
        _write_json(CHAT_MAP_FILE, chat_map)
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
            logging.info(f"Chat session deleted: {title}")
        return True
    logging.error(f"Chat session not found: {title}")
    return False

def save_session(chat_file: str) -> None:
    """
    Saves the active chat session to session.json.
    
    Parameters:
      chat_file (str): The filepath of the active chat session.
    """
    _write_json(SESSION_FILE, {"current_chat": chat_file})

def load_session() -> str:
    """
    Loads the active chat session from session.json.
    
    Returns:
      str: The filepath of the active chat session, or None if not set.
    """
    data = _read_json(SESSION_FILE)
    return data.get("current_chat", None)

# ---------------------------
# Messaging Functions
# ---------------------------
def _handle_tool_calls(messages: list[BaseMessage], ai_message: AIMessage) -> list[BaseMessage]:
    """Handle tool calls from AI response and append tool messages to conversation."""
    if not ai_message.tool_calls:
        return messages
    logging.debug(f"Tool function[0] type: {type(tools[0])}")
    logging.debug(f"Tool function[1] type: {type(tools[1])}")
    logging.debug(f"Tool function[0]: {tools[0]}")
    logging.debug(f"Tool function[1]: {tools[1]}")
    tools_dict = {
        "interactive_windows_shell_tool": tools[0],
        "run_python_code": tools[1]
    }
    logging.info(f"AI wants to run commands...")

    for tool_call in ai_message.tool_calls:
        tool_name = None
        try:
            tool_name = tool_call["name"]
            tool_call_id = tool_call["id"]
            if tool_name not in tools_dict:
                logging.error(f"Unknown tool: {tool_name}")
                continue
                
            tool = tools_dict[tool_name]
            logging.debug(f"Tool function: {tool}")
            logging.debug(f"Tool function type: {type(tool)}")

            
            tool_response:ToolMessage = tool.invoke(tool_call)
            tool_response.tool_call_id = tool_call_id
            logging.debug(f"Tool response added: {tool_response.content}")
            messages.append(tool_response)

        except Exception as e:
            logging.error(f"Error executing tool {tool_name}: {e}")
    return messages


def send_message(message: str) -> str:
    """
    Handles message sending in two scenarios:
    1. No current chat: Creates a new temp chat with system prompt
    2. Existing chat: Appends to existing conversation
    
    Parameters:
      message (str): The human message.
      
    Returns:
      str: The AI's response.
    """
    # Get or create chat session
    chat_file = get_current_chat()
    if not chat_file:
        console_session_id = _get_console_session_id()
        chat_file = create_or_load_chat(console_session_id)
    
    # Load existing messages and ensure they exist
    current_messages = _read_messages(chat_file) or []
    
    # Ensure system prompt exists at the start
    if len(current_messages) == 0 or not isinstance(current_messages[0], SystemMessage):
        config = _read_json(CONFIG_FILE)
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        current_messages.insert(0, SystemMessage(content=default_prompt))
    
    # Append new human message
    human_message = HumanMessage(content=message)
    current_messages.append(human_message)
    
    # Log human message with correct index
    human_count = sum(1 for msg in current_messages if isinstance(msg, HumanMessage))
    logging.info(f"User[{human_count}]: {message}")
    
    # Get AI response with complete history
    llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE).bind_tools(tools_functions)

    ai_response:AIMessage = None
    # Process response
    while True:
        ai_response:AIMessage = llm.invoke(current_messages)
        current_messages.append(ai_response)
        if len(ai_response.tool_calls or []) > 0:
            current_messages = _handle_tool_calls(current_messages, ai_response)
        else:
            logging.info(f"AI: {ai_response.content}")
            break
    
    # Save complete updated conversation
    _write_messages(chat_file, current_messages)
    return ai_response

def start_temp_chat(message: str) -> str:
    """
    Starts a temporary (in-memory) chat session with the default system prompt,
    appends the human message and the AI response (powered by ChatOpenAI with bound tools),
    and returns the AI's response.
    
    The human message is shown as 'User[N]: <message>' in the console
    but the content does not include 'User[N]'.
    
    Parameters:
      message (str): The initial message for the temporary chat.
      
    Returns:
      str: The AI's response.
    """
    console_session_id = _get_console_session_id()
    chat_file = create_or_load_chat(console_session_id)
    
    messages = _read_messages(chat_file)
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        config = _read_json(CONFIG_FILE)
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        messages.insert(0, SystemMessage(content=default_prompt))
    
    human_message_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
    human_index = human_message_count + 1
    
    messages.append(HumanMessage(content=message))
    logging.info(f"User[{human_index}]: {message}")
    
    llm = ChatOpenAI(model=MODEL, temperature=0.7).bind_tools(tools_functions)
    ai_response = llm.invoke(messages)
    
    messages.append(ai_response)
    
    # Handle tool calls if present, otherwise show AI response
    if len(ai_response.tool_calls or []) > 0:
        messages = _handle_tool_calls(messages, ai_response)
        response = ""  # Don't return AI text when tool calls were handled
    else:
        response = ai_response.content
        logging.info(f"AI: {response}")
    
    set_current_chat(chat_file)
    _write_messages(chat_file, messages)
    return response

def edit_message(index: int, new_message: str) -> bool:
    """
    Edits a previous message at the given index and truncates subsequent messages.
    
    Parameters:
      index (int): The index of the message to edit.
      new_message (str): The new content for the message.
      
    Returns:
      bool: True if successful, False otherwise.
    """
    chat_file = load_session()
    if not chat_file:
        return False
        
    messages = _read_messages(chat_file)
    if index < 0 or index >= len(messages):
        return False
        
    # Preserve message type while updating content
    message_type = type(messages[index])
    messages[index] = message_type(content=new_message)
    messages = messages[:index + 1]
    _write_messages(chat_file, messages)
    return True

def flush_temp_chats() -> None:
    """Removes all temporary chat sessions."""
    chat_map = _read_json(CHAT_MAP_FILE)
    # Identify titles beginning with "temp_"
    to_remove = [title for title in chat_map if title.startswith("temp_")]
    for title in to_remove:
        chat_id = chat_map.pop(title)
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
    _write_json(CHAT_MAP_FILE, chat_map)

# ---------------------------
# System Prompt Management
# ---------------------------
def set_default_system_prompt(prompt_text: str) -> None:
    """
    Sets the default system prompt in config.json.
    
    Parameters:
      prompt_text (str): The default system prompt.
    """
    config = _read_json(CONFIG_FILE)
    config["default_system_prompt"] = prompt_text
    _write_json(CONFIG_FILE, config)
    logging.info("Default system prompt saved to config.json")

def update_system_prompt(prompt_text: str) -> None:
    """
    Updates the system prompt for the active chat session.
    
    Parameters:
      prompt_text (str): The new system prompt.
    """
    chat_file = load_session()
    if not chat_file:
        logging.warning("No active chat session to update.")
        return
        
    messages = _read_messages(chat_file)
    messages.insert(0, SystemMessage(content=prompt_text))
    _write_messages(chat_file, messages)

def cmd(command: str) -> str:
    """
    Executes a shell command directly and adds both command and output 
    to chat history as a HumanMessage. Creates a temporary chat if no chat is active.
    
    Parameters:
      command (str): The shell command to execute.
      
    Returns:
      str: The command output.
    """
    # Get or create chat session
    chat_file = get_current_chat()
    if not chat_file:
        console_session_id = _get_console_session_id()
        chat_file = create_or_load_chat(console_session_id)
    
    # Load existing messages and ensure they exist
    current_messages = _read_messages(chat_file) or []
    
    # Ensure system prompt exists
    if not any(isinstance(msg, SystemMessage) for msg in current_messages):
        config = _read_json(CONFIG_FILE)
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        current_messages.insert(0, SystemMessage(content=default_prompt))
    
    # Execute command and get output
    output = direct_windows_shell_tool.invoke({"command": command})
    
    # Append new command message
    cmd_message = HumanMessage(content=f"CMD> {command}\n{output}")
    current_messages.append(cmd_message)
    
    # Save complete updated conversation
    _write_messages(chat_file, current_messages)
    return output

