"""
Chat session management module for AI Shell Agent.
Handles chat sessions, history, and the conversation flow with the LLM.
"""
# Standard imports
#import os
import json
#import uuid
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
#from pathlib import Path
import time

# External imports
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage
)
#from langchain_core.utils.function_calling import convert_to_openai_function

# Local imports
from . import logger
#from .config_manager import get_current_model, get_model_provider

# --- Import necessary components from the state manager ---
from .chat_state_manager import (
    get_current_chat,
    save_session,
    create_or_load_chat, # Keep for start_temp_chat internal call
    _read_chat_data,
    _write_chat_data,
    get_active_toolsets,
    # update_active_toolsets, # Not directly called here
    _get_console_session_id,
    rename_chat as rename_chat_state,
    delete_chat as delete_chat_state,
    flush_temp_chats as flush_temp_chats_state,
    get_current_chat_title,
    _update_message_in_chat, # Keep for edit_message internal call
    _get_chat_messages
)

# --- Import ONLY what's needed from aider_integration and the new registry ---
from .aider_integration_and_tools import get_active_coder_state, SIGNAL_PROMPT_NEEDED # Removed unused start_code_editor_tool import
from .tool_registry import get_all_tools_dict, get_all_openai_functions, get_all_tools

# --- Import the new LLM and Prompt builders ---
from .llm import get_llm

# --- Chat Session Management ---
def get_chat_titles_list():
    """Prints the list of available chat titles."""
    # Uses state_manager._read_json internally now via get_current_chat
    from .chat_state_manager import _read_json, CHAT_MAP_FILE, get_current_chat
    chat_map = _read_json(CHAT_MAP_FILE, {})
    current_chat_id = get_current_chat() # Get ID from state manager

    if not chat_map:
        print("No chats found.")
        return

    print("\nAvailable Chats:")
    # Sort by title (value) case-insensitively
    for chat_id, title in sorted(chat_map.items(), key=lambda item: item[1].lower()):
        current_marker = " <- Current" if chat_id == current_chat_id else ""
        print(f"- {title}{current_marker}")


def rename_chat(old_title: str, new_title: str) -> None:
    """Renames a chat session by calling the state manager."""
    success = rename_chat_state(old_title, new_title) # Delegate persistence
    if success:
        print(f"Renamed chat: {old_title} -> {new_title}") # Keep user feedback here
    # Error message handled within rename_chat_state now, avoid double printing
    # else:
    #     print(f"Failed to rename chat: {old_title} -> {new_title}")


def delete_chat(title: str) -> None:
    """Deletes a chat session by calling the state manager."""
    deleted = delete_chat_state(title) # Delegate persistence
    if deleted:
        print(f"Deleted chat: {title}")
    # Error message handled within delete_chat_state now
    # else:
    #     print(f"Chat not found: {title}") # Keep user feedback here


# ---------------------------
# Messaging Functions
# ---------------------------
def _handle_tool_calls(ai_message: AIMessage, chat_file: str) -> list[BaseMessage]:
    """
    Handle tool calls from AI response. Invokes tools and returns ToolMessages.
    Relies on tools themselves to update state if necessary.
    """
    logger.debug(f"Handling tool calls: {ai_message.tool_calls}")
    if not ai_message.tool_calls:
        return []

    messages: List[BaseMessage] = []
    tool_registry_dict = get_all_tools_dict() # Get tools from registry
    logger.info(f"AI wants to run {len(ai_message.tool_calls)} tool(s)...")

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_call_id = tool_call.get("id")
        tool_args = tool_call.get("args", {}) # Get args dict

        if not tool_name or not tool_call_id:
             logger.error(f"Invalid tool call structure received: {tool_call}")
             continue

        try:
            if tool_name not in tool_registry_dict:
                logger.error(f"Unknown tool requested: {tool_name}")
                error_content = f"Error: Tool '{tool_name}' not found or not available."
                messages.append(ToolMessage(content=error_content, tool_call_id=tool_call_id))
                continue

            tool = tool_registry_dict[tool_name]
            logger.debug(f"Invoking tool '{tool_name}' with args: {tool_args}")

            # Invoke the tool with its arguments dictionary
            tool_result_content = tool.invoke(tool_args)

            # Wrap the raw output in a ToolMessage
            tool_response = ToolMessage(content=str(tool_result_content), tool_call_id=tool_call_id)

            # Check for Aider input signal post-invocation
            if isinstance(tool_response.content, str) and tool_response.content.startswith(SIGNAL_PROMPT_NEEDED):
                prompt_details = tool_response.content[len(SIGNAL_PROMPT_NEEDED):].strip()
                logger.info(f"Tool '{tool_name}' indicated Aider needs input: {prompt_details}")

            messages.append(tool_response)

        except Exception as e:
            logger.error(f"Error running tool '{tool_name}' with args {tool_args}: {e}")
            logger.error(traceback.format_exc())
            error_content = f"Error executing tool '{tool_name}': {e}"
            # Use the original tool_call_id for the error message
            messages.append(ToolMessage(content=error_content, tool_call_id=tool_call_id))

    logger.debug(f"Returning {len(messages)} tool messages.")
    return messages

def _convert_message_dicts_to_langchain(messages: List[Dict]) -> List[BaseMessage]:
    """
    Converts a list of message dictionaries to LangChain messages.
    Handles both standard roles (system, human, ai) and tool messages.
    """
    lc_messages = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "human":
            lc_messages.append(HumanMessage(content=content))
        elif role == "ai":
            # Check for tool calls in the message
            tool_calls = msg.get("tool_calls", [])
            
            # Create AIMessage with tool_calls if present
            if tool_calls:
                lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
            else:
                lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            # Create a ToolMessage for each tool response
            tool_call_id = msg.get("tool_call_id", "unknown_tool_call")
            lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        else:
            logger.warning(f"Unknown message role: {role}")
            # Default to human message for unknown roles
            lc_messages.append(HumanMessage(content=content))
    
    return lc_messages

def _convert_langchain_to_message_dicts(messages: List[BaseMessage]) -> List[Dict]:
    """
    Converts LangChain messages back to dictionaries for storage.
    Handles both standard messages and tool messages.
    """
    msg_dicts = []
    
    for msg in messages:
        if isinstance(msg, SystemMessage):
            msg_dicts.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            msg_dicts.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            ai_dict = {"role": "ai", "content": msg.content}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                ai_dict["tool_calls"] = msg.tool_calls
            msg_dicts.append(ai_dict)
        elif isinstance(msg, ToolMessage):
            msg_dicts.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
        else:
            logger.warning(f"Unknown message type: {type(msg)}")
            # Default to storing as a human message
            msg_dicts.append({"role": "human", "content": str(msg.content)})
    
    return msg_dicts

def send_message(message: str) -> None:
    """
    Sends a message to the AI and displays the response.
    Handles message history, dynamic prompt/tool loading, and tool calls.
    Implements ReAct logic to automatically continue agent execution after tool calls.
    """
    chat_file = get_current_chat()
    if not chat_file:
        logger.warning("No active chat session. Starting temporary session.")
        start_temp_chat(message)
        return

    # Add human message using state manager helpers
    timestamp = datetime.now(timezone.utc).isoformat()
    human_msg_dict = {"role": "human", "content": message, "timestamp": timestamp}
    current_data = _read_chat_data(chat_file)
    current_data.setdefault("messages", []).append(human_msg_dict)
    _write_chat_data(chat_file, current_data)
    logger.info(f"Human: {message}")

    # ReAct loop - continue until we get an AI response without tool calls
    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"ReAct iteration {iteration}/{max_iterations}")
        
        # --- Prepare for LLM Call ---
        chat_history_dicts = _get_chat_messages(chat_file)
        
        # Validate messages before sending to the LLM to prevent OpenAI errors
        # Track which AI tool calls have matching tool responses
        validated_messages = []
        tool_call_ids_with_responses = set()
        
        # First pass: identify which tool call IDs have responses
        for msg in chat_history_dicts:
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                tool_call_ids_with_responses.add(msg.get("tool_call_id"))
        
        # Second pass: filter out AI messages with tool calls that don't have matching responses
        for msg in chat_history_dicts:
            if (msg.get("role") == "ai" and 
                "tool_calls" in msg and 
                msg["tool_calls"]):
                
                # Check if all tool calls in this AI message have responses
                all_tool_calls_have_responses = True
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("id") not in tool_call_ids_with_responses:
                        all_tool_calls_have_responses = False
                        logger.warning(f"Found AI message with tool call ID {tool_call.get('id')} without response. Filtering out tool calls.")
                        
                # If not all tool calls have responses, create a new message without tool_calls
                if not all_tool_calls_have_responses:
                    # Clone the message but without tool_calls
                    filtered_msg = {
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg.get("timestamp", "")
                    }
                    validated_messages.append(filtered_msg)
                else:
                    # Keep the original message with tool_calls
                    validated_messages.append(msg)
            else:
                # Keep all other messages unchanged
                validated_messages.append(msg)
        
        # Use the validated messages for the LLM call
        active_toolsets = get_active_toolsets(chat_file)
        lc_messages = _convert_message_dicts_to_langchain(validated_messages)
        
        if lc_messages and lc_messages[0].type == "system":
            logger.debug(f"System prompt (first message): {lc_messages[0].content[:200]}...") # Log beginning
        else:
            logger.warning("No system prompt found as the first message in history.")

        llm_instance = get_llm(active_toolsets=active_toolsets)

        try:
            # --- Invoke LLM ---
            ai_response = llm_instance.invoke(lc_messages)
            logger.info(f"AI Raw Response Content: {ai_response.content}")
            logger.debug(f"AI Full Response Object: {ai_response}")

            # --- Save AI response ---
            ai_msg_dict = {
                "role": "ai",
                "content": ai_response.content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
                # Ensure tool calls are serializable (should be list of dicts)
                try:
                    # Langchain tool_calls are typically already serializable list[dict]
                    ai_msg_dict["tool_calls"] = ai_response.tool_calls
                    logger.debug(f"AI response included tool calls: {ai_response.tool_calls}")
                except TypeError as json_err:
                    logger.error(f"Could not serialize tool_calls: {json_err}. Storing empty list.")
                    ai_msg_dict["tool_calls"] = []

            current_data = _read_chat_data(chat_file) # Re-read data before write
            current_data.setdefault("messages", []).append(ai_msg_dict)
            _write_chat_data(chat_file, current_data) # Save AI response

            # If this is the first response or the response doesn't have tool calls, print it to user
            if iteration == 1 or not (hasattr(ai_response, "tool_calls") and ai_response.tool_calls):
                print(ai_response.content) # Print message content to user

            # --- Handle Tool Calls ---
            has_tool_calls = hasattr(ai_response, "tool_calls") and ai_response.tool_calls
            if has_tool_calls:
                logger.info(f"AI made {len(ai_response.tool_calls)} tool call(s) - executing and continuing ReAct loop")
                tool_messages = _handle_tool_calls(ai_response, chat_file)

                # Save tool messages to chat history and print results
                if tool_messages: # Only proceed if tool handling produced messages
                    current_data = _read_chat_data(chat_file) # Re-read data again before write
                    tool_message_dicts = []
                    for tool_msg in tool_messages:
                        tool_msg_dict = {
                            "role": "tool",
                            "content": tool_msg.content,
                            "tool_call_id": tool_msg.tool_call_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        tool_message_dicts.append(tool_msg_dict)

                        # Print tool result/signal
                        if isinstance(tool_msg.content, str) and tool_msg.content.startswith(SIGNAL_PROMPT_NEEDED):
                            prompt_details = tool_msg.content[len(SIGNAL_PROMPT_NEEDED):].strip()
                            print(f"\n[Code Editor Input Required]: {prompt_details}")
                            # Exit ReAct loop on Aider input signal - needs human intervention
                            has_tool_calls = False  # Set to false to exit the loop
                        else:
                            tool_name_for_print = next((tc.get("name") for tc in ai_response.tool_calls if tc.get("id") == tool_msg.tool_call_id), "Unknown Tool")
                            print(f"\n[Tool Result - {tool_name_for_print} ({tool_msg.tool_call_id})]:\n{tool_msg.content}")

                    current_data.setdefault("messages", []).extend(tool_message_dicts)
                    _write_chat_data(chat_file, current_data) # Save data with tool results
                else:
                    # If tool execution failed completely, mark all tool calls as having no valid responses
                    logger.warning("Tool execution failed to produce any tool messages. Breaking ReAct loop to prevent API errors.")
                    has_tool_calls = False  # Force exit from the loop as there are no tool responses
                
                # Break the loop if there are no tool calls or if Aider needs input
                if not has_tool_calls:
                    logger.info("Breaking ReAct loop - no more tool calls or input needed")
                    break
                
                # Continue the loop to let the agent react to the tool output
                continue
            
            # No tool calls, break the ReAct loop
            logger.info("AI response has no tool calls, ReAct loop complete")
            break

        except Exception as e:
            logger.error(f"Error during LLM interaction or tool handling: {e}")
            logger.error(traceback.format_exc())
            print(f"An error occurred: {e}")
            break
    
    # Log if we hit the iteration limit
    if iteration >= max_iterations:
        logger.warning(f"ReAct loop hit maximum iterations ({max_iterations}), terminating")
        print(f"\n[Warning: Reached maximum number of tool calls ({max_iterations}). Process terminated.]")


def start_temp_chat(message: str) -> None:
    """
    Starts a temporary chat session with the given message.
    Uses state manager's create_or_load_chat which initializes prompt/toolsets.
    """
    console_id = _get_console_session_id() # For potential future use
    chat_title = f"Temp Chat {int(time.time())}"
    logger.info(f"Starting temporary chat: {chat_title}")
    # create_or_load_chat from state_manager handles initial prompt/toolsets
    chat_file = create_or_load_chat(chat_title)
    save_session(chat_file) # Set as current session
    send_message(message) # Send the initial message


def edit_message(index: Optional[int], new_message: str) -> None:
    """
    Edits a previous message in the chat using state manager helpers.
    If index is None, edits the last human message.
    Truncates history and resends the message.
    """
    chat_file = get_current_chat()
    if not chat_file:
        logger.error("No active chat session to edit.")
        print("No active chat session. Please create or load a chat first.")
        return

    # Use state manager helper to get messages
    chat_messages = _get_chat_messages(chat_file)

    target_index = -1
    if index is None:
        # Find the last human message index
        for i in range(len(chat_messages) - 1, -1, -1):
            if chat_messages[i].get("role") == "human":
                target_index = i
                break
        if target_index == -1:
            logger.error("No human messages found to edit.")
            print("No previous human message found to edit.")
            return
    else:
        # Validate provided index
        if not isinstance(index, int) or not (0 <= index < len(chat_messages)):
            logger.error(f"Message index {index} out of range (0-{len(chat_messages)-1}).")
            print(f"Error: Message index {index} is out of range.")
            return
        # Ensure it's a human message we're editing
        if chat_messages[index].get("role") != "human":
            logger.error(f"Cannot edit non-human message at index {index} (role: {chat_messages[index].get('role')}).")
            print(f"Error: Cannot edit message at index {index} - it's not a user message.")
            return
        target_index = index

    # Edit the message in the list (will be saved below)
    original_content = chat_messages[target_index].get("content")
    timestamp = datetime.now(timezone.utc).isoformat()
    edited_message_dict = {
        "role": "human",
        "content": new_message,
        "timestamp": timestamp,
        "edited": True,
        "original_content": original_content # Store original for reference
    }

    # Update chat data using state manager helpers
    chat_data = _read_chat_data(chat_file) # Read fresh data
    chat_data["messages"] = chat_data.get("messages", [])[:target_index] # Messages before edit
    chat_data["messages"].append(edited_message_dict) # Add edited message
    num_removed = len(chat_messages) - (target_index + 1)

    _write_chat_data(chat_file, chat_data) # Save truncated + edited history

    logger.info(f"Edited message at index {target_index}. Original: '{original_content}', New: '{new_message}'. Removed {num_removed} subsequent messages.")
    print(f"Message at index {target_index} edited. Removed {num_removed} subsequent messages.")

    # Resend the message to get a new response from the LLM
    send_message(new_message)


def flush_temp_chats() -> None:
    """Removes all temporary chat sessions."""
    removed_count = flush_temp_chats_state() # Delegates to state manager
    print(f"Removed {removed_count} temporary chats.")

# --- Additional Commands ---
def execute(command: str) -> str:
    """Executes a shell command directly using the direct terminal tool."""
    try:
        from .terminal_tools import direct_terminal_tool
        logger.info(f"Executing direct command: {command}")
        output = direct_terminal_tool.invoke({"command": command})
        print(output) # Print output to console
        return output
    except ImportError:
        logger.error("Could not import direct_terminal_tool.")
        return "Error: Direct terminal tool not available."
    except Exception as e:
        logger.error(f"Error executing direct command '{command}': {e}")
        error_msg = f"Error executing command: {e}"
        print(error_msg)
        return error_msg


def list_messages() -> None:
    """Lists all messages in the current chat."""
    chat_file = get_current_chat()
    if not chat_file:
        logger.error("No active chat session to list messages.")
        print("No active chat session. Please create or load a chat first.")
        return

    # Use state manager helper
    chat_messages = _get_chat_messages(chat_file)

    if not chat_messages:
        print("No messages in the current chat.")
        return

    print("\n--- Chat History ---")
    titles = {
        "system": "System",
        "human": "Human",
        "ai": "Assistant",
        "tool": "Tool",
    }

    for i, msg in enumerate(chat_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "") # Get timestamp if available

        print(f"\n[{i}] {titles.get(role, role.capitalize())} ({timestamp}):") # Display timestamp

        # Handle potential non-string content (e.g., structured tool output if not stringified)
        if not isinstance(content, str):
            try:
                content_str = json.dumps(content, indent=2)
            except TypeError:
                content_str = str(content)
        else:
            content_str = content

        # Truncate long messages for display only
        display_content = content_str
        if len(display_content) > 500:
            display_content = display_content[:500] + "..."

        print(display_content)

        # Show tool calls if present on AI message
        if role == "ai" and "tool_calls" in msg:
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                print("  [Tool Calls Initiated]:")
                for tc in tool_calls:
                    tool_name = tc.get('name', 'unknown_tool')
                    tool_args = tc.get('args', {})
                    tool_id = tc.get('id', 'no_id')
                    print(f"    - Tool: {tool_name}, Args: {tool_args}, ID: {tool_id}")
        # Show tool call ID if present on Tool message
        elif role == "tool" and "tool_call_id" in msg:
            tool_call_id = msg.get("tool_call_id")
            print(f"  [For Tool Call ID]: {tool_call_id}")

    print("\n--- End of History ---")


def current_chat_title() -> None:
    """Prints the title of the current chat."""
    # Delegates to state manager
    title = get_current_chat_title()
    if title:
        print(f"Current chat: {title}")
    else:
        print("No active chat session.")