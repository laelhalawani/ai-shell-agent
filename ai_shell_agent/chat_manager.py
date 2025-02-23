# File: ai_shell_agent/chat_manager.py
import os
import json
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    BaseMessage
)
from langchain_core.messages.utils import convert_to_openai_messages
from .tools import tools_functions, direct_windows_shell_tool, tools
from .prompts import default_system_prompt
from . import logger

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
                    elif msg["type"] == "tool":  # Added branch for tool messages
                        messages.append(ToolMessage(**msg))
                logger.debug(f"Read messages: {messages}")
                return messages
            except json.JSONDecodeError:
                return []
    return []

def _write_messages(file_path: str, messages: list[BaseMessage]) -> None:
    """Write messages to JSON file."""
    messages_data = [msg.model_dump() for msg in messages]
    logger.debug(f"Writing messages: {messages_data}")
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
    logger.debug(f"Setting current chat: {chat_file}")
    _write_json(SESSION_FILE, {"current_chat": chat_file})

def get_current_chat() -> str:
    """
    Gets the current chat session.
    
    Returns:
      str: The filepath of the current chat session, or None if not set.
    """
    data = _read_json(SESSION_FILE)
    logger.debug(f"Current chat data: {data}")
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
    logger.debug(f"Chat map: {chat_map}")
    if (title in chat_map):
        chat_id = chat_map[title]
        logger.debug(f"Loading existing chat session: {title}")
    else:
        chat_id = str(uuid.uuid4())
        chat_map[title] = chat_id
        _write_json(CHAT_MAP_FILE, chat_map)
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if not os.path.exists(chat_file):
        logger.info(f"Creating new chat session: {title}")
        # New chat: add default system prompt
        config = _read_json(CONFIG_FILE)
        logger.debug(f"Config: {config}")
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
    logger.info(f"Chats: \n{chats_str}")
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
    logger.debug(f"Chat map: {chat_map}")
    if old_title in chat_map:
        chat_map[new_title] = chat_map.pop(old_title)
        _write_json(CHAT_MAP_FILE, chat_map)
        logger.info(f"Chat session renamed: {old_title} -> {new_title}")
        return True
    logger.error(f"Chat session not found: {old_title}")
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
    logger.debug(f"Chat map: {chat_map}")
    if title in chat_map:
        chat_id = chat_map.pop(title)
        _write_json(CHAT_MAP_FILE, chat_map)
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
            logger.info(f"Chat session deleted: {title}")
        return True
    logger.error(f"Chat session not found: {title}")
    return False

def save_session(chat_file: str) -> None:
    """
    Saves the active chat session to session.json.
    
    Parameters:
      chat_file (str): The filepath of the active chat session.
    """
    logger.debug(f"Saving session: {chat_file}")
    _write_json(SESSION_FILE, {"current_chat": chat_file})

def load_session() -> str:
    """
    Loads the active chat session from session.json.
    
    Returns:
      str: The filepath of the active chat session, or None if not set.
    """
    data = _read_json(SESSION_FILE)
    logger.debug(f"Loaded session data: {data}")
    return data.get("current_chat", None)

# ---------------------------
# Messaging Functions
# ---------------------------
def _handle_tool_calls(ai_message: AIMessage) -> list[BaseMessage]:
    """Handle tool calls from AI response and append tool messages to conversation."""
    logger.debug(f"AI message tool calls: {ai_message.tool_calls}")
    if not ai_message.tool_calls:
        return []
    messages = []

    tools_dict = {
        "interactive_windows_shell_tool": tools[0],
        "run_python_code": tools[1]
    }
    logger.info(f"AI wants to run commands...")

    for tool_call in ai_message.tool_calls:
        tool_name = None
        try:
            tool_name = tool_call["name"]
            tool_call_id = tool_call["id"]
            if tool_name not in tools_dict:
                logger.error(f"Unknown tool: {tool_name}")
                continue
                
            tool = tools_dict[tool_name]
            tool_response: ToolMessage = tool.invoke(tool_call)
            tool_response.tool_call_id = tool_call_id
            messages.append(tool_response)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            # Append an error message to the conversation
            error_message = AIMessage(content=f"Error executing tool {tool_name}: {e}")
            messages.append(error_message)
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
    logger.debug(f"Chat file: {chat_file}")
    if not chat_file:
        console_session_id = _get_console_session_id()
        chat_file = create_or_load_chat(console_session_id)
    
    # Load existing messages and ensure they exist
    current_messages = _read_messages(chat_file) or []
    logger.debug(f"Current messages: {current_messages}")
    
    # Ensure system prompt exists at the start
    if len(current_messages) == 0 or not isinstance(current_messages[0], SystemMessage):
        config = _read_json(CONFIG_FILE)
        logger.debug(f"Config: {config}")
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        current_messages.insert(0, SystemMessage(content=default_prompt))
    
    # Append new human message
    human_message = HumanMessage(content=message)
    current_messages.append(human_message)
    logger.debug(f"Appended human message: {human_message}")
    # Log human message with correct index
    human_count = sum(1 for msg in current_messages if isinstance(msg, HumanMessage))
    logger.info(f"User[{human_count}]: {message}")
    
    # Get AI response with complete history
    llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE).bind_tools(tools_functions)

    ai_response: AIMessage = None
    # Process response
    while True:
        logger.debug(f"Current messages: {current_messages}")
        ai_response = llm.invoke(convert_to_openai_messages(current_messages))
        logger.debug(f"AI response: {ai_response}")
        current_messages.append(ai_response)
        
        if ai_response.tool_calls and len(ai_response.tool_calls) > 0:
            tool_messages = _handle_tool_calls(ai_response)
            current_messages.extend(tool_messages)
        else:
            logger.info(f"AI: {ai_response.content}")
            break
    _write_messages(chat_file, current_messages)
    return ai_response.content

def start_temp_chat(message: str) -> str:
    """
    Starts a temporary (in-memory) chat session with the default system prompt,
    appends the human message and the AI response (powered by ChatOpenAI with bound tools),
    and returns the final AI response.
    
    This function now uses a loop to process tool calls until no further tool calls
    are returned, ensuring that the final LLM response is obtained.
    
    Parameters:
      message (str): The initial message for the temporary chat.
      
    Returns:
      str: The final AI's response.
    """
    console_session_id = _get_console_session_id()
    logger.debug(f"Console session ID: {console_session_id}")
    chat_file = create_or_load_chat(console_session_id)
    logger.debug(f"Chat file: {chat_file}")
    
    current_messages = _read_messages(chat_file)
    logger.debug(f"Messages: {current_messages}")
    if not any(isinstance(msg, SystemMessage) for msg in current_messages):
        config = _read_json(CONFIG_FILE)
        logger.debug(f"Config: {config}")
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        current_messages.insert(0, SystemMessage(content=default_prompt))
    
    human_message_count = sum(1 for msg in current_messages if isinstance(msg, HumanMessage))
    human_index = human_message_count + 1
    
    current_messages.append(HumanMessage(content=message))
    logger.info(f"User[{human_index}]: {message}")
    
    llm = ChatOpenAI(model=MODEL, temperature=0.7).bind_tools(tools_functions)
    logger.debug(f"LLM: {llm}")
    
    # Loop to process tool calls until no tool calls are returned
    while True:
        ai_response: AIMessage = llm.invoke(convert_to_openai_messages(current_messages))
        logger.debug(f"AI response: {ai_response}")
        current_messages.append(ai_response)
        if ai_response.tool_calls and len(ai_response.tool_calls) > 0:
            tool_messages = _handle_tool_calls(ai_response)
            current_messages.extend(tool_messages)
        else:
            logger.info(f"AI: {ai_response.content}")
            break
    
    set_current_chat(chat_file)
    _write_messages(chat_file, current_messages)
    return ai_response.content

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
    logger.debug(f"Chat file: {chat_file}")
    if not chat_file:
        return False
        
    messages = _read_messages(chat_file)
    logger.debug(f"Messages: {messages}")
    if index < 0 or index >= len(messages):
        return False
        
    # Preserve message type while updating content
    message_type = type(messages[index])
    messages[index] = message_type(content=new_message)
    messages = messages[:index + 1]
    logger.debug(f"Edited messages: {messages}")
    _write_messages(chat_file, messages)
    return True

def flush_temp_chats() -> None:
    """Removes all temporary chat sessions."""
    chat_map = _read_json(CHAT_MAP_FILE)
    logger.debug(f"Chat map: {chat_map}")
    # Identify titles beginning with "temp_"
    to_remove = [title for title in chat_map if title.startswith("temp_")]
    for title in to_remove:
        chat_id = chat_map.pop(title)
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
    logger.debug(f"Updated chat map: {chat_map}")
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
    logger.debug(f"Config: {config}")
    config["default_system_prompt"] = prompt_text
    _write_json(CONFIG_FILE, config)
    logger.info("Default system prompt saved to config.json")

def update_system_prompt(prompt_text: str) -> None:
    """
    Updates the system prompt for the active chat session.
    
    Parameters:
      prompt_text (str): The new system prompt.
    """
    chat_file = load_session()
    logger.debug(f"Chat file: {chat_file}")
    if not chat_file:
        logger.warning("No active chat session to update.")
        return
        
    messages = _read_messages(chat_file)
    logger.debug(f"Messages: {messages}")
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
    logger.debug(f"Chat file: {chat_file}")
    if not chat_file:
        console_session_id = _get_console_session_id()
        chat_file = create_or_load_chat(console_session_id)
    
    # Load existing messages and ensure they exist
    current_messages = _read_messages(chat_file) or []
    logger.debug(f"Current messages: {current_messages}")
    
    # Ensure system prompt exists
    if not any(isinstance(msg, SystemMessage) for msg in current_messages):
        config = _read_json(CONFIG_FILE)
        logger.debug(f"Config: {config}")
        default_prompt = config.get("default_system_prompt", default_system_prompt)
        current_messages.insert(0, SystemMessage(content=default_prompt))
    
    # Execute command and get output
    output = direct_windows_shell_tool.invoke({"command": command})
    logger.debug(f"Command output: {output}")
    
    # Append new command message
    cmd_message = HumanMessage(content=f"CMD> {command}\n{output}")
    current_messages.append(cmd_message)
    logger.debug(f"Appended command message: {cmd_message}")
    
    # Save complete updated conversation
    _write_messages(chat_file, current_messages)
    return output
