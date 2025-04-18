"""
Chat session management module for AI Shell Agent.
Handles chat sessions, history, and the conversation flow with the LLM.
"""
import os
import json
import time
import argparse
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
import queue
from uuid import uuid4

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Rich imports for text processing
from rich.text import Text
from rich.panel import Panel

# Local imports
from . import logger
from .llm import get_llm
from .prompts.prompts import SYSTEM_PROMPT

# --- Console Manager ---
from .console_manager import get_console_manager

# --- Custom Errors ---
from .errors import PromptNeededError

# --- Import necessary components from the state manager ---
from .chat_state_manager import (
    get_current_chat,
    get_chat_messages,
    get_chat_map,
    create_or_load_chat,
    _write_chat_messages,       # Used directly for saving history
    # _get_chat_dir_path,       # Not needed directly, used by delete_chat_state
    # _write_chat_map,          # Not needed directly, used by delete_chat_state/rename_chat_state
    get_current_chat_title,
    get_enabled_toolsets,
    get_active_toolsets,
    rename_chat as rename_chat_state, # Import 'rename_chat' and alias it
    delete_chat as delete_chat_state  # Import 'delete_chat' and alias it
)

# --- Import tool integrations/registry ---
from .toolsets.toolsets import get_registered_toolsets, get_toolset_names
from .tool_registry import get_all_tools, get_all_tools_dict # Removed get_tool_by_name


# Get console manager instance
console = get_console_manager()

# --- Define signal constants --- 
SIGNAL_PROMPT_NEEDED = "[PROMPT_NEEDED]"

# --- Chat Session Management ---
def get_chat_titles_list():
    """Prints the list of available chat titles using console_manager."""
    chat_map = get_chat_map()
    current_chat_id = get_current_chat()
    
    if not chat_map:
        console.display_message("SYSTEM: ", "No chats found.", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        return
        
    console.display_message("SYSTEM: ", "Available Chats:", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    for chat_id, title in sorted(chat_map.items(), key=lambda item: item[1].lower()):
        marker = " <- Current" if chat_id == current_chat_id else ""
        console.console.print(f"- {title}{marker}")

def rename_chat(old_title: str, new_title: str) -> None:
    """Renames a chat session by calling the state manager."""
    if rename_chat_state(old_title, new_title):
        console.display_message("INFO: ", f"Renamed chat: {old_title} -> {new_title}", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
    else:
        console.display_message("ERROR: ", f"Failed to rename chat: {old_title}", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

def delete_chat(title: str) -> None:
    """Deletes a chat session by calling the state manager."""
    if delete_chat_state(title):
        console.display_message("INFO: ", f"Deleted chat: {title}", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
    else:
        console.display_message("ERROR: ", f"Failed to delete chat: {title}", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

# ---------------------------
# Tool Handling
# ---------------------------
# Define return types for _handle_tool_calls for clarity
ToolResult = str  # Simple string result for successful tool execution
ErrorResult = str  # Error message string when tool execution fails

def _handle_tool_calls(
    ai_message: AIMessage,
    chat_file: str,
    confirmed_inputs: Optional[Dict[str, str]] = None
) -> List[Union[ToolResult, ErrorResult, PromptNeededError]]:
    """
    Handle tool calls from AI response. Invokes tools.

    Args:
        ai_message: The AI message containing tool calls.
        chat_file: The current chat file ID.
        confirmed_inputs: A dictionary {tool_call_id: input_string} if this is a
                         re-invocation after user confirmation.

    Returns:
        A list containing results for each tool call:
        - Tool output string (ToolResult)
        - An ErrorResult string if an error occurred.
        - A PromptNeededError exception instance if HITL is required.
    """
    if not ai_message.tool_calls:
        return []
    
    logger.debug(f"Handling {len(ai_message.tool_calls)} tool calls. Confirmed input provided: {bool(confirmed_inputs)}")

    results = []
    tool_registry_dict = get_all_tools_dict()
    
    if not confirmed_inputs:
        confirmed_inputs = {}  # Ensure it's a dict

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_call_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})

        # Handle potential non-dict args
        if isinstance(tool_args, str):
            try: 
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                # Keep as string if not JSON
                pass

        # --- Basic Validation ---
        if not tool_name or not tool_call_id:
            logger.error(f"Invalid tool call structure: {tool_call}")
            results.append(f"Error: Invalid tool call structure (missing name or id): {tool_call}")
            continue

        logger.debug(f"Processing Tool Call: {tool_name}(args={tool_args}) ID: {tool_call_id}")

        # --- Check if confirmed input is provided for this specific tool call ---
        current_confirmed_input = confirmed_inputs.get(tool_call_id)

        # --- Get Tool Instance ---
        if tool_name not in tool_registry_dict:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            # Return error result directly
            results.append(f"Error: Tool '{tool_name}' not found.")
            continue

        tool_instance = tool_registry_dict[tool_name]

        try:
            # --- Extract primary argument value from tool_args ---
            primary_arg_value = None
            
            # If tool_args is a string, use it directly as the primary argument
            if isinstance(tool_args, str):
                primary_arg_value = tool_args
            # If tool_args is a dict with exactly one key, use its value
            elif isinstance(tool_args, dict) and len(tool_args) == 1:
                primary_arg_value = list(tool_args.values())[0]
            # If tool_args is a dict with more than one key, try to find a suitable primary argument
            elif isinstance(tool_args, dict) and len(tool_args) > 0:
                # Try to use a common primary argument name if present
                for common_arg_name in ['query', 'command', 'cmd', 'text', 'input', 'message', 'prompt']:
                    if common_arg_name in tool_args:
                        primary_arg_value = tool_args[common_arg_name]
                        break
                # If no common name found, just use the first value
                if primary_arg_value is None:
                    primary_arg_value = list(tool_args.values())[0]
                    
            # --- Tool Execution ---
            # Choose execution based on whether this is a HITL scenario
            if hasattr(tool_instance, 'requires_confirmation') and tool_instance.requires_confirmation:
                # This is a HITL tool - handle the two-phase execution
                if current_confirmed_input is None:
                    # Phase 1: First call, need user input - will raise PromptNeededError
                    if primary_arg_value is not None:
                        tool_result = tool_instance._run(primary_arg_value)
                    else:
                        tool_result = tool_instance._run()  # No args case
                    
                    # If we get here without a PromptNeededError, the tool didn't request input
                    results.append(tool_result)
                else:
                    # Phase 2: Re-execution with confirmed input
                    if primary_arg_value is not None:
                        tool_result = tool_instance._run(primary_arg_value, confirmed_input=current_confirmed_input)
                    else:
                        tool_result = tool_instance._run(confirmed_input=current_confirmed_input)
                    
                    results.append(tool_result)
            else:
                # Regular non-HITL tool - simply invoke it
                if primary_arg_value is not None:
                    tool_result = tool_instance._run(primary_arg_value)
                else:
                    # No args needed or couldn't determine primary arg
                    tool_result = tool_instance._run()
                
                # Ensure result is string
                if not isinstance(tool_result, str):
                    try:
                        tool_result = json.dumps(tool_result)
                    except TypeError:
                        tool_result = str(tool_result)
                
                results.append(tool_result)
            
            logger.debug(f"Tool '{tool_name}' returned result (success)")

        except PromptNeededError as pne:
            # HITL required. Pass the exception itself back to the main loop.
            logger.info(f"Tool '{tool_name}' raised PromptNeededError.")
            results.append(pne)

        except Exception as e:
            # Handle other execution errors
            logger.error(f"Error invoking tool '{tool_name}' with args {tool_args}: {e}", exc_info=True)
            results.append(f"Error executing tool '{tool_name}': {type(e).__name__}: {e}")

    return results

# --- Message Conversion Helpers ---
def _convert_message_dicts_to_langchain(message_dicts: List[Dict]) -> List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]:
    """Converts chat message dictionaries to LangChain message objects."""
    langchain_messages = []
    
    for msg in message_dicts:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "human":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "ai":
            tool_calls = msg.get("tool_calls", [])
            langchain_messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                langchain_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    
    return langchain_messages

def _convert_langchain_to_message_dicts(messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]) -> List[Dict]:
    """Converts LangChain message objects to chat message dictionaries."""
    message_dicts = []
    
    for msg in messages:
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if isinstance(msg, SystemMessage):
            message_dict = {"role": "system", "content": msg.content, "timestamp": timestamp}
        elif isinstance(msg, HumanMessage):
            message_dict = {"role": "human", "content": msg.content, "timestamp": timestamp}
        elif isinstance(msg, AIMessage):
            message_dict = {
                "role": "ai", 
                "content": msg.content, 
                "timestamp": timestamp
            }
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
        elif isinstance(msg, ToolMessage):
            message_dict = {
                "role": "tool", 
                "content": msg.content, 
                "timestamp": timestamp,
                "tool_call_id": msg.tool_call_id
            }
        else:
            logger.warning(f"Unknown message type: {type(msg)}. Skipping.")
            continue
        
        message_dicts.append(message_dict)
    
    return message_dicts

# --- send_message Refactor ---
def send_message(message: str) -> None:
    """Sends message, handles ReAct loop for tool calls using ConsoleManager."""
    chat_file = get_current_chat()
    if not chat_file:
        console.display_message("WARNING: ", "No active chat. Starting temp chat.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        start_temp_chat(message)
        return

    # 1. Add Human Message
    human_msg_dict = {"role": "human", "content": message, "timestamp": datetime.now(timezone.utc).isoformat()}
    current_messages = get_chat_messages(chat_file)
    current_messages.append(human_msg_dict)
    _write_chat_messages(chat_file, current_messages)
    logger.debug(f"Human message added to chat {chat_file}: {message[:100]}...")

    # 2. ReAct Loop Variables
    max_iterations = 100
    iteration = 0
    pending_prompt: Optional[Tuple[str, PromptNeededError]] = None  # Store (tool_call_id, error)

    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"ReAct iteration {iteration}/{max_iterations}")

        # --- Handle Pending User Input ---
        if pending_prompt:
            tool_call_id, prompt_error = pending_prompt
            pending_prompt = None  # Clear the pending prompt

            # Get user input via ConsoleManager
            confirmed_input_str = console.display_tool_prompt(prompt_error)

            # --- LINE CLEARING LOGIC ---
            if confirmed_input_str is not None:
                # If input was successful (not cancelled), clear the prompt line
                console.clear_current_line()
            # --- END LINE CLEARING LOGIC ---

            if confirmed_input_str is None:  # User cancelled
                logger.warning("User cancelled prompt. Stopping ReAct loop.")
                # Tool message indicating cancellation
                tool_response = ToolMessage(content="User cancelled input.", tool_call_id=tool_call_id)
                tool_message_dicts = _convert_langchain_to_message_dicts([tool_response])
                current_messages = get_chat_messages(chat_file)
                current_messages.extend(tool_message_dicts)
                _write_chat_messages(chat_file, current_messages)
                break  # Stop the loop

            # User provided input - display confirmation now (AFTER clearing the line)
            final_args = {prompt_error.edit_key: confirmed_input_str}
            console.display_tool_confirmation(prompt_error.tool_name, final_args)

            # --- Find the AI message associated with tool_call_id in history ---
            original_ai_message = None
            history_dicts = get_chat_messages(chat_file)
            for msg_dict in reversed(history_dicts):
                if msg_dict.get("role") == "ai" and "tool_calls" in msg_dict:
                    for tc in msg_dict["tool_calls"]:
                        if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                            # Convert just this dict to Langchain AIMessage
                            temp_lc_msgs = _convert_message_dicts_to_langchain([msg_dict])
                            if temp_lc_msgs and isinstance(temp_lc_msgs[0], AIMessage):
                                original_ai_message = temp_lc_msgs[0]
                                break
                    if original_ai_message:
                        break

            if not original_ai_message:
                logger.error(f"Could not find original AI message for tool_call_id {tool_call_id} to re-invoke tool.")
                console.display_message("ERROR: ", "Internal error: Failed to find tool context.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                break

            # Re-call _handle_tool_calls, passing the confirmed input for the specific tool call
            logger.debug(f"Re-invoking tool handling for {tool_call_id} with confirmed input.")
            confirmed_input_map = {tool_call_id: confirmed_input_str}
            tool_call_results = _handle_tool_calls(original_ai_message, chat_file, confirmed_inputs=confirmed_input_map)

            # Process results (expecting only one result now)
            if not tool_call_results:
                logger.error("Tool handling returned no results after confirmed input.")
                console.display_message("ERROR: ", "Tool execution failed after input.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                break

            result_item = tool_call_results[0]  # Process the single result
            # Get tool name for display
            tool_name = prompt_error.tool_name  # Use the tool name from the original prompt error

            # --- Determine final tool content ---
            tool_content = ""
            prompt_needed_again = False # Flag if prompt is needed again
            if isinstance(result_item, str): # Includes ToolResult and ErrorResult strings
                tool_content = result_item
                # --- Display condensed tool output passing the tool_name ---
                console.display_tool_output(tool_name, tool_content)
            elif isinstance(result_item, PromptNeededError):
                # Handle case where tool immediately asks for input again
                logger.warning("Tool requested input again immediately after receiving input.")
                tool_content = f"{SIGNAL_PROMPT_NEEDED} Tool '{result_item.tool_name}' requires further input unexpectedly."
                pending_prompt = (tool_call_id, result_item) # Set pending for next iteration
                prompt_needed_again = True
            else:
                logger.error(f"Unexpected result type after confirmed input: {type(result_item)}")
                tool_content = "Error: Unexpected tool result type."
                # Display error message directly
                console.display_message("ERROR: ", tool_content, console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

            # --- Find and Update Placeholder ToolMessage ---
            logger.debug(f"Attempting to find and update placeholder ToolMessage for ID: {tool_call_id}")
            current_messages = get_chat_messages(chat_file)
            message_updated = False
            placeholder_index = -1

            # Search backwards for efficiency
            for i in range(len(current_messages) - 1, -1, -1):
                msg = current_messages[i]
                if (msg.get("role") == "tool" and
                    msg.get("tool_call_id") == tool_call_id and
                    isinstance(msg.get("content"), str) and
                    msg.get("content", "").startswith(SIGNAL_PROMPT_NEEDED)):
                    placeholder_index = i
                    break

            if placeholder_index != -1:
                logger.debug(f"Found placeholder ToolMessage at index {placeholder_index}. Updating content.")
                current_messages[placeholder_index]["content"] = tool_content
                current_messages[placeholder_index]["timestamp"] = datetime.now(timezone.utc).isoformat()
                # Optionally add metadata about the update if needed
                # if "metadata" not in current_messages[placeholder_index]:
                #     current_messages[placeholder_index]["metadata"] = {}
                # current_messages[placeholder_index]["metadata"]["updated_after_hitl"] = True
                message_updated = True
            else:
                logger.error(f"Could not find placeholder ToolMessage for ID {tool_call_id} to update!")
                # Decide how to handle this - maybe append anyway as a fallback?
                # For now, we'll just log the error and proceed. The history might be inconsistent.
                # Consider adding a new message if the placeholder wasn't found, although it shouldn't happen.

            # Save the potentially modified history
            if message_updated:
                _write_chat_messages(chat_file, current_messages)
                logger.debug("Chat history updated with tool result after HITL.")
            else:
                logger.warning("Chat history NOT updated as placeholder message wasn't found.")
            # --- End Find and Update ---

            # --- Decide next step ---
            if prompt_needed_again:
                # If prompt needed again, loop will handle it based on pending_prompt being set
                continue
            elif "Error:" in tool_content: # Check if it was an error result string
                logger.warning("Tool execution resulted in an error after input. Stopping loop.")
                break # Stop loop if error occurred
            else:
                # Success after input, continue to next LLM call
                logger.debug("HITL step completed successfully. Proceeding to next LLM call.")
                continue # Explicitly continue loop

        # --- Normal LLM Invocation Flow ---
        chat_history_dicts = get_chat_messages(chat_file)
        # Basic validation
        if not chat_history_dicts:
            logger.error(f"No chat history found for {chat_file}")
            console.display_message("ERROR: ", "Chat history not found or empty.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            break
            
        lc_messages = _convert_message_dicts_to_langchain(chat_history_dicts)
        
        # --- LLM instantiation ---
        try:
            llm_instance = get_llm()
        except Exception as e:
            logger.error(f"LLM Init fail: {e}", exc_info=True)
            console.display_message("ERROR:", f"AI model initialization failed: {e}", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            break

        try:
            # --- START THINKING ---
            console.start_thinking()

            # --- Invoke LLM ---
            ai_response = llm_instance.invoke(lc_messages)
            logger.debug(f"AI Raw Response Content: {ai_response.content}")
            logger.debug(f"AI Raw Response Tool Calls: {getattr(ai_response, 'tool_calls', None)}")

            # --- Save AI response ---
            current_messages = get_chat_messages(chat_file)
            ai_msg_dict_list = _convert_langchain_to_message_dicts([ai_response])
            if not ai_msg_dict_list:
                logger.error("Failed to convert LLM response.")
                console.display_message("ERROR:", "AI response conversion failed.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                break
                
            ai_msg_dict = ai_msg_dict_list[0]
            current_messages.append(ai_msg_dict)
            _write_chat_messages(chat_file, current_messages)

            # --- Handle Tool Calls or AI Response ---
            has_tool_calls = bool(ai_msg_dict.get("tool_calls"))
            ai_content = ai_response.content

            if has_tool_calls:
                logger.info(f"AI requesting {len(ai_response.tool_calls)} tool call(s)...")
                # Process tools *without* confirmed input initially
                tool_call_results = _handle_tool_calls(ai_response, chat_file)

                tool_messages_to_save = []
                prompt_needed_for_next_iteration = None  # Track if any prompt is needed
                any_tool_ran_successfully = False # Track if we should show "Used tool"

                # Process results from potentially multiple tool calls
                for i, result_item in enumerate(tool_call_results):
                    # Get corresponding tool_call and id
                    tool_call = ai_response.tool_calls[i] # Get corresponding call
                    tool_call_id = tool_call.get("id", f"unknown_call_{i}")
                    tool_name = tool_call.get("name") # Get tool name
                    tool_args = tool_call.get("args", {}) # Get args
                    tool_content = ""

                    if isinstance(result_item, PromptNeededError):
                        logger.info(f"Tool {result_item.tool_name} needs input.")
                        tool_content = f"{SIGNAL_PROMPT_NEEDED} Tool '{result_item.tool_name}' requires input."
                        if not prompt_needed_for_next_iteration:  # Store the first prompt encountered
                            prompt_needed_for_next_iteration = (tool_call_id, result_item)

                    elif isinstance(result_item, str):  # ToolResult or ErrorResult
                        tool_content = result_item
                        if "Error:" not in tool_content:
                            any_tool_ran_successfully = True # Mark success
                            # Display "Used tool..." confirmation first
                            console.display_tool_confirmation(tool_name, tool_args)
                            # Display condensed tool output with tool name
                            console.display_tool_output(tool_name, tool_content)
                        else:
                            # Display error message directly
                            console.display_message("ERROR:", tool_content, console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
                    else:
                        logger.error(f"Unexpected result type from tool handling: {type(result_item)}")
                        tool_content = "Error: Unexpected tool result type."
                        console.display_message("ERROR:", tool_content, console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

                    # Create ToolMessage for history
                    tool_messages_to_save.append(ToolMessage(content=tool_content, tool_call_id=tool_call_id))

                # Save all tool messages
                if tool_messages_to_save:
                    tool_message_dicts = _convert_langchain_to_message_dicts(tool_messages_to_save)
                    current_messages = get_chat_messages(chat_file)
                    current_messages.extend(tool_message_dicts)
                    _write_chat_messages(chat_file, current_messages)

                # Decide next step based on whether a prompt is pending
                if prompt_needed_for_next_iteration:
                    logger.debug("Prompt needed, setting pending_prompt for next iteration.")
                    pending_prompt = prompt_needed_for_next_iteration
                    # Loop continues implicitly
                elif any_tool_ran_successfully or tool_messages_to_save: # If any tool ran or message saved
                    # Now start thinking for the next step ON A NEW LINE
                    console.start_thinking()
                else: # No tools ran, no prompts, maybe all errors?
                    logger.warning("Tool handling finished with no success or prompts.")
                    break # Stop the loop

            elif ai_content:
                # --- DISPLAY FINAL AI TEXT RESPONSE ---
                logger.info(f"AI: {ai_content[:100]}...")
                console.display_ai_response(ai_content)
                break  # End loop after final text response

            else:
                # --- Handle empty AI response ---
                logger.warning("AI response had no content and no tool calls.")
                console.display_message("WARNING:", "AI returned an empty response.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
                break  # End loop

        except Exception as e:
            logger.error(f"LLM/Tool Error in main loop: {e}", exc_info=True)
            console.display_message("ERROR:", f"AI interaction error: {e}", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            break  # End loop on error

    # --- Max iterations handling ---
    if iteration >= max_iterations:
        logger.warning("Hit maximum iterations of ReAct loop")
        console.display_message("WARNING:", f"Reached maximum of {max_iterations} AI interactions. Stopping.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)

# --- Temporary Chat Management ---
def start_temp_chat(message: str) -> None:
    """Starts a temporary chat session."""
    safe_ts = str(time.time()).replace('.', '_')
    chat_title = f"Temp Chat {safe_ts}"
    
    logger.info(f"Starting temporary chat: {chat_title}")
    chat_file = create_or_load_chat(chat_title)
    
    if chat_file:
        console.display_message("INFO: ", f"Started {chat_title}.", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        send_message(message)
    else:
        console.display_message("ERROR:", f"Could not start temp chat '{chat_title}'.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

def flush_temp_chats() -> None:
    """Removes all temporary chat sessions."""
    # Import state manager functions needed specifically here
    from .chat_state_manager import get_chat_map, get_current_chat, delete_chat_state

    chat_map = get_chat_map()
    current_chat_id = get_current_chat()

    temp_chats_to_remove = [] # Store titles to delete
    for chat_id, title in chat_map.items():
        if title.startswith("Temp Chat "):
            if chat_id == current_chat_id:
                # Don't delete current chat
                console.display_message("INFO: ", f"Skipping current temp chat: {title}", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
            else:
                temp_chats_to_remove.append(title) # Add title to list

    if not temp_chats_to_remove:
        console.display_message("INFO: ", "No temporary chats found to remove.", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return

    removed_count = 0
    for title in temp_chats_to_remove:
         # Use the public delete function which handles map update and directory removal
         if delete_chat_state(title): # delete_chat_state returns bool
             removed_count += 1
         else:
             # Error message printed within delete_chat_state/delete_chat
             logger.warning(f"Failed to delete temporary chat '{title}' during flush.")

    # delete_chat_state handles clearing the session if the current one is deleted,
    # but we explicitly skip deleting the current one above.

    # Map is updated internally by delete_chat_state calls
    console.display_message("INFO: ", f"Removed {removed_count} temporary chat(s).", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)

# --- Message Editing ---
def edit_message(idx: Optional[int], new_message: str) -> None:
    """
    Edits a message in the current chat and re-processes from that point.
    If idx is None, edits the last human message.
    """
    chat_file = get_current_chat()
    if not chat_file:
        console.display_message("ERROR: ", "No active chat to edit.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return
        
    if not new_message.strip():
        console.display_message("ERROR: ", "Cannot set empty message.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return
        
    current_messages = get_chat_messages(chat_file)
    if not current_messages:
        console.display_message("ERROR: ", "No messages to edit.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return
        
    # Find the message to edit
    if idx is None or idx == "last":
        # Find the last human message
        for i in range(len(current_messages) - 1, -1, -1):
            if current_messages[i].get("role") == "human":
                idx = i
                break
        if idx is None:
            console.display_message("ERROR: ", "No human messages found to edit.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
            return
    elif idx < 0 or idx >= len(current_messages):
        console.display_message("ERROR: ", f"Invalid index {idx}. Must be between 0 and {len(current_messages) - 1}.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return
        
    # Check if the message at idx is human
    if current_messages[idx].get("role") != "human":
        console.display_message("ERROR: ", f"Message at index {idx} is not a human message.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return
        
    # Store original timestamp
    original_timestamp = current_messages[idx].get("timestamp")
    
    # Update the message
    current_messages[idx]["content"] = new_message
    current_messages[idx]["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Add metadata about edit
    if "metadata" not in current_messages[idx]:
        current_messages[idx]["metadata"] = {}
    current_messages[idx]["metadata"]["edited"] = True
    current_messages[idx]["metadata"]["original_timestamp"] = original_timestamp
    
    # Truncate history after this message
    current_messages = current_messages[:idx + 1]
    
    # Save the updated message history
    _write_chat_messages(chat_file, current_messages)
    console.display_message("INFO: ", f"Edited message at position {idx}. Reprocessing...", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
    
    # Send the same message again to trigger reprocessing
    send_message(new_message)

# --- Direct Command Execution ---
def execute(command: str) -> str:
    """Executes a shell command directly using the internal tool."""
    try:
        from .toolsets.terminal.toolset import run_direct_terminal_command
        logger.info(f"Executing direct command: {command}")
        output = run_direct_terminal_command(command)
        console.display_message("INFO: ", f"Direct Execution Result:\n{output}", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return output
    except ImportError:
        err_msg = "Error: Direct terminal tool function not available."
        logger.error(err_msg)
        console.display_message("ERROR: ", err_msg, console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return err_msg
    except Exception as e:
        logger.error(f"Error executing direct command '{command}': {e}", exc_info=True)
        error_msg = f"Error executing command: {e}"
        console.display_message("ERROR: ", error_msg, console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return error_msg

# --- Message Listing ---
def list_messages() -> None:
    """Lists all messages in the current chat using Rich via ConsoleManager."""
    chat_file = get_current_chat()
    if not chat_file:
        console.display_message("WARNING:", "No active chat.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        return

    chat_messages = get_chat_messages(chat_file)
    if not chat_messages:
        console.display_message("INFO:", "No messages in chat.", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
        return

    chat_title = get_current_chat_title() or "Current Chat"
    console.display_message("SYSTEM:", f"\n--- Chat History for: {chat_title} ---", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    roles_map = {
        "system": console.STYLE_SYSTEM_CONTENT,
        "human": console.STYLE_USER_LABEL,
        "ai": console.STYLE_AI_CONTENT,
        "tool": console.STYLE_INFO_CONTENT,
    }
    titles = {"system": "System", "human": "Human", "ai": "AI", "tool": "Tool"}

    for i, msg in enumerate(chat_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts_str = msg.get("timestamp", "No timestamp")
        
        # Format timestamp
        try:
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            ts = datetime.fromisoformat(ts_str).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            ts = ts_str

        role_style = roles_map.get(role, console.STYLE_INFO_CONTENT)
        role_title = titles.get(role, role.capitalize())

        # Build panel content
        panel_content = Text()
        
        # Add tool call info or tool call reference where applicable
        if role == "ai" and msg.get("tool_calls"):
            panel_content.append("Tool Calls Initiated:\n", style=console.STYLE_SYSTEM_CONTENT)
            tool_calls_list = msg["tool_calls"] if isinstance(msg["tool_calls"], list) else []
            for tc in tool_calls_list:
                if isinstance(tc, dict):
                    tc_text = Text(f"- Tool: ", style=console.STYLE_SYSTEM_CONTENT)
                    tc_text.append(f"{tc.get('name', '?')}", style=console.STYLE_TOOL_NAME)
                    tc_text.append(f", Args: {json.dumps(tc.get('args', {}))}", style=console.STYLE_ARG_VALUE)
                    tc_text.append(f", ID: {tc.get('id', '?')}\n", style=console.STYLE_ARG_NAME)
                    panel_content.append(tc_text)
                else:
                    panel_content.append(f"- Invalid tool call entry: {tc}\n", style=console.STYLE_ERROR_CONTENT)
            panel_content.append("\n")
        elif role == "tool" and msg.get("tool_call_id"):
            panel_content.append(f"For Tool Call ID: {msg['tool_call_id']}\n\n", style=console.STYLE_ARG_NAME)
        
        if msg.get("metadata", {}).get("edited"):
            panel_content.append(f"[Edited - Original TS: {msg['metadata'].get('original_timestamp', 'N/A')}]\n\n", style=console.STYLE_SYSTEM_CONTENT)

        # Append main content (ensure it's a string)
        content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
        panel_content.append(content_str)

        # Output the panel
        from rich.panel import Panel
        console.console.print(
            Panel(
                panel_content,
                title=f"[{i}] {role_title}",
                subtitle=f"({ts})",
                border_style=role_style,
                title_align="left",
                subtitle_align="right"
            )
        )
    
    console.display_message("SYSTEM:", "--- End of History ---", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

# --- Toolset Listing ---
def list_toolsets() -> None:
    """Lists all available toolsets and their status."""
    chat_id = get_current_chat()
    
    if not chat_id:
        console.display_message("ERROR: ", "No active chat session.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        console.display_message("SYSTEM: ", "Please load or create a chat first (e.g., `ai -c <chat_title>`).", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
        return
    
    chat_title = get_current_chat_title()
    registered_toolsets = get_registered_toolsets()
    
    if not registered_toolsets:
        console.display_message("WARNING: ", "No toolsets found/registered.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        return
    
    active_toolsets = get_active_toolsets(chat_id)
    enabled_toolsets = get_enabled_toolsets(chat_id)
    
    console.display_message("SYSTEM: ", f"Toolsets for '{chat_title}':", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)
    
    # Combine all distinct toolset names
    all_toolset_names = sorted(set(list(registered_toolsets.values()) + active_toolsets + enabled_toolsets))
    
    # Headers for table columns
    from rich.table import Table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Toolset", style="cyan")
    table.add_column("Description", style="")
    table.add_column("Registered", style="")
    table.add_column("Enabled", style="")
    table.add_column("Active", style="")
    
    # Add rows for each toolset
    for ts_name in sorted(registered_toolsets.keys()):
        meta = registered_toolsets[ts_name]
        
        is_registered = "✓"
        is_enabled = "✓" if meta.name in enabled_toolsets else "-"
        is_active = "✓" if meta.name in active_toolsets else "-"
        
        description = meta.description or "No description"
        if len(description) > 60:
            description = description[:57] + "..."
            
        table.add_row(meta.name, description, is_registered, is_enabled, is_active)
    
    # Output the table
    console.console.print(table)
    
    # Output explanation
    explanation = """
• Registered: The toolset is installed and available for use
• Enabled: The toolset is enabled in the current chat and can be activated
• Active: The toolset is currently active and its tools are available to the AI

Note: Use --select-tools to change which toolsets are enabled.
      A toolset must be enabled before it can be activated/deactivated 
      by the AI during conversation.
"""
    console.display_message("SYSTEM: ", explanation, console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)