"""
Chat session management module for AI Shell Agent.
Handles chat sessions, history, and the conversation flow with the LLM.
"""
import json
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import time

# External imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage

# Local imports
from . import logger, console_io # console_io import remains important
from .llm import get_llm
# --- Import necessary components from the state manager ---
from .chat_state_manager import (
    get_current_chat, save_session, create_or_load_chat, 
    get_active_toolsets, get_enabled_toolsets,
    _get_console_session_id, rename_chat as rename_chat_state,
    delete_chat as delete_chat_state, flush_temp_chats as flush_temp_chats_state,
    get_current_chat_title, _update_message_in_chat, get_chat_messages, _write_chat_messages
)

# --- Import tool integrations/registry ---
try:
    from .toolsets.aider.integration.integration import SIGNAL_PROMPT_NEEDED
except ImportError:
    SIGNAL_PROMPT_NEEDED = "[SIGNAL_PLACEHOLDER_AIDER_UNAVAILABLE]" # Fallback if aider toolset fails

from .tool_registry import get_all_tools_dict # Needed for _handle_tool_calls

# --- Chat Session Management ---
def get_chat_titles_list():
    """Prints the list of available chat titles using console_io."""
    from .chat_state_manager import _read_json, CHAT_MAP_FILE
    chat_map = _read_json(CHAT_MAP_FILE, {})
    current_chat_id = get_current_chat()
    if not chat_map:
        console_io.print_system("No chats found.") # Corrected function name
        return
    console_io.print_system("\nAvailable Chats:") # Corrected function name
    for chat_id, title in sorted(chat_map.items(), key=lambda item: item[1].lower()):
        marker = " <- Current" if chat_id == current_chat_id else ""
        # Use console.print directly for simple list formatting
        # Assuming print_list_item is a valid method in your console_io or just use console.print
        try:
             console_io.print_list_item(f"{title}{marker}")
        except AttributeError: # Fallback if print_list_item doesn't exist
             console_io.console.print(f"- {title}{marker}")


def rename_chat(old_title: str, new_title: str) -> None:
    """Renames a chat session by calling the state manager."""
    # Use console_io for confirmation message
    if rename_chat_state(old_title, new_title):
        console_io.print_info(f"Renamed chat: {old_title} -> {new_title}")

def delete_chat(title: str) -> None:
    """Deletes a chat session by calling the state manager."""
    if delete_chat_state(title):
        console_io.print_info(f"Deleted chat: {title}")


# ---------------------------
# Messaging Functions
# ---------------------------
def _handle_tool_calls(ai_message: AIMessage, chat_file: str) -> list[BaseMessage]:
    """
    Handle tool calls from AI response. Invokes tools and returns ToolMessages.
    Tool execution handles its own console output via console_io.
    """
    if not ai_message.tool_calls: return []
    logger.debug(f"Handling {len(ai_message.tool_calls)} tool calls.")
    messages: List[BaseMessage] = []
    tool_registry_dict = get_all_tools_dict()

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_call_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})
        if isinstance(tool_args, str): # Handle non-dict args
             try: tool_args = json.loads(tool_args)
             except json.JSONDecodeError:
                  logger.warning(f"Tool '{tool_name}' received non-JSON string args: '{tool_args}'. Passing as is.")

        if not tool_name or not tool_call_id:
             logger.error(f"Invalid tool call structure: {tool_call}")
             messages.append(ToolMessage(content=f"Error: Invalid tool call structure (missing name or id): {tool_call}", tool_call_id=tool_call_id or f"invalid_call_{time.time()}"))
             continue

        logger.debug(f"Tool Call Requested: {tool_name}(args={tool_args}) ID: {tool_call_id}")

        if tool_name not in tool_registry_dict:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            # Print error via console_io as this happens *before* tool execution
            console_io.print_error(f"Tool '{tool_name}' not found in registry.")
            messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call_id))
            continue

        tool_instance = tool_registry_dict[tool_name]
        try:
            # *** Tool Execution Happens Here ***
            # The tool's _run method will now handle:
            # 1. Calling console_io.request_tool_edit (if HITL) -> stops spinner, prints prefix, prompts
            # 2. Calling console_io.print_tool_execution_info -> prints "Used tool..."
            # 3. Executing its core logic
            # 4. Calling console_io.print_tool_output -> prints "TOOL: ..."
            # 5. Returning the result string
            tool_result_content = tool_instance.invoke(input=tool_args)
            # *** End Tool Execution ***

            # Ensure result is string for ToolMessage
            if not isinstance(tool_result_content, str):
                try: tool_result_content = json.dumps(tool_result_content)
                except TypeError: tool_result_content = str(tool_result_content)

            # Create ToolMessage with the returned content
            tool_response = ToolMessage(content=tool_result_content, tool_call_id=tool_call_id)

            logger.debug(f"Tool '{tool_name}' returned. Result preview: {tool_result_content[:200]}...")

            # Keep Aider signal check logic here for logging, but primary check is in send_message
            if isinstance(tool_response.content, str) and tool_response.content.startswith(SIGNAL_PROMPT_NEEDED):
                logger.debug(f"Tool '{tool_name}' signaled Aider input needed.") # Log signal detection

            messages.append(tool_response)
        except Exception as e:
            # If the exception happens *during* invoke (e.g., subprocess error AFTER prompt)
            # The tool's _run should ideally catch it and return an error string.
            # If the exception is *before* or *outside* the tool's internal try/except, log it here.
            logger.error(f"Error invoking tool '{tool_name}' with args {tool_args}: {e}", exc_info=True)
            # Print error via console_io as this indicates a failure in the framework or tool setup
            console_io.print_error(f"Error running tool '{tool_name}': {e}")
            messages.append(ToolMessage(content=f"Error executing tool '{tool_name}': {type(e).__name__}: {e}", tool_call_id=tool_call_id))

    return messages

# --- Message Conversion Helpers ---
def _convert_message_dicts_to_langchain(messages: List[Dict]) -> List[BaseMessage]:
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system": lc_messages.append(SystemMessage(content=content))
        elif role == "human": lc_messages.append(HumanMessage(content=content))
        elif role == "ai":
            tool_calls = msg.get("tool_calls", [])
            invalid_tool_calls = msg.get("invalid_tool_calls", []) # Langchain might add this
            all_calls = tool_calls + invalid_tool_calls
            # Filter out calls missing 'id' which causes issues downstream
            valid_calls = [c for c in all_calls if isinstance(c, dict) and c.get("id")] # Ensure dict and id
            if valid_calls: lc_messages.append(AIMessage(content=content, tool_calls=valid_calls))
            else: lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown_tool_call")
            lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        else:
            logger.warning(f"Unknown message role '{role}'. Treating as Human.")
            lc_messages.append(HumanMessage(content=f"[{role.upper()}] {content}"))
    return lc_messages


def _convert_langchain_to_message_dicts(messages: List[BaseMessage]) -> List[Dict]:
    msg_dicts = []
    for msg in messages:
        timestamp = datetime.now(timezone.utc).isoformat()
        if isinstance(msg, SystemMessage):
            msg_dicts.append({"role": "system", "content": msg.content, "timestamp": timestamp})
        elif isinstance(msg, HumanMessage):
            msg_dicts.append({"role": "human", "content": msg.content, "timestamp": timestamp})
        elif isinstance(msg, AIMessage):
            ai_dict = {"role": "ai", "content": msg.content, "timestamp": timestamp}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                 # Ensure serializable list of dicts with IDs
                 valid_calls = [dict(c) for c in msg.tool_calls if isinstance(c, dict) and c.get("id")]
                 if valid_calls: ai_dict["tool_calls"] = valid_calls
            msg_dicts.append(ai_dict)
        elif isinstance(msg, ToolMessage):
            msg_dicts.append({
                "role": "tool", "content": msg.content,
                "tool_call_id": getattr(msg, 'tool_call_id', 'unknown_tool_call'),
                "timestamp": timestamp })
        else:
            logger.warning(f"Unknown message type {type(msg)}. Storing as Human.")
            msg_dicts.append({"role": "human", "content": f"[UNKNOWN TYPE: {type(msg).__name__}] {str(msg.content)}", "timestamp": timestamp})
    return msg_dicts


def send_message(message: str) -> None:
    """Sends message, handles ReAct loop for tool calls."""
    chat_file = get_current_chat()
    if not chat_file:
        console_io.print_warning("No active chat. Starting temp chat.")
        start_temp_chat(message)
        return

    # 1. Add Human Message
    human_msg_dict = {"role": "human", "content": message, "timestamp": datetime.now(timezone.utc).isoformat()}
    current_messages = get_chat_messages(chat_file); current_messages.append(human_msg_dict)
    _write_chat_messages(chat_file, current_messages)
    logger.debug(f"Human message added to chat {chat_file}: {message[:100]}...")

    # 2. ReAct Loop
    max_iterations = 100
    iteration = 0
    # Define ai_response outside the loop scope initially for the condition check removal
    ai_response = None

    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"ReAct iteration {iteration}/{max_iterations}")
        chat_history_dicts = get_chat_messages(chat_file)

        # --- Message Validation Logic - Direct implementation instead of calling undefined function ---
        validated_messages = []
        try:
            # Direct implementation of message validation
            # Simply using the chat history as is - this could be enhanced with actual validation if needed
            validated_messages = chat_history_dicts
            
            # If additional validation is needed, it can be implemented here
            # For example, checking message format, removing invalid messages, etc.
            logger.debug(f"Message validation completed: {len(validated_messages)} messages")
        except Exception as e:
            logger.error(f"Error during message validation: {e}", exc_info=True)
            console_io.print_error(f"Error validating messages: {e}")
            break # Exit on validation error
            
        if not validated_messages: # Check after validation
            logger.error("Validation resulted in empty history.")
            console_io.print_error("History processing failed.")
            console_io._stop_live()
            break

        lc_messages = _convert_message_dicts_to_langchain(validated_messages)
        # --- LLM instantiation (remains the same) ---
        try:
            llm_instance = get_llm()
        except Exception as e:
             logger.error(f"LLM Init fail: {e}", exc_info=True); console_io.print_error(f"AI model initialization failed: {e}")
             console_io._stop_live(); break

        try:
            # --- START THINKING (Unconditional before LLM invoke) ---
            console_io.start_ai_thinking() # ALWAYS show spinner before invoking LLM

            # --- Invoke LLM ---
            ai_response = llm_instance.invoke(lc_messages) # Assign to loop-scoped variable
            logger.debug(f"AI Raw Response Content: {ai_response.content}")
            logger.debug(f"AI Raw Response Tool Calls: {ai_response.tool_calls}")

            # --- Save AI response FIRST ---
            current_messages = get_chat_messages(chat_file) # Re-read before append
            ai_msg_dict_list = _convert_langchain_to_message_dicts([ai_response])
            if not ai_msg_dict_list:
                logger.error("Failed to convert LLM response to message format."); console_io.print_error("AI response conversion failed.")
                console_io._stop_live(); break
            ai_msg_dict = ai_msg_dict_list[0]
            current_messages.append(ai_msg_dict)
            _write_chat_messages(chat_file, current_messages)

            # --- Handle Tool Calls or AI Response ---
            has_tool_calls = bool(ai_msg_dict.get("tool_calls"))
            ai_content = ai_response.content

            if has_tool_calls:
                logger.info(f"AI requesting {len(ai_response.tool_calls)} tool call(s)...")
                # --- _handle_tool_calls invokes tools which handle their own I/O ---
                tool_messages = _handle_tool_calls(ai_response, chat_file)
                # --- Process Tool Results ---
                if tool_messages:
                    tool_message_dicts = _convert_langchain_to_message_dicts(tool_messages)
                    current_messages = get_chat_messages(chat_file); current_messages.extend(tool_message_dicts)
                    _write_chat_messages(chat_file, current_messages)
                    # --- Aider Signal Handling ---
                    from .toolsets.aider.integration.integration import is_file_editor_prompt_signal
                    for tm in tool_message_dicts:
                        # Log if we found a prompt signal from Aider
                        if tm.get("role") == "tool" and is_file_editor_prompt_signal(tm.get("content", "")):
                            logger.debug(f"Stopping ReAct loop due to File Editor prompt signal: {tm.get('content', '')[:50]}...")
                            console_io._stop_live() # Stop spinner but don't print anything
                            break # Break inner loop only
                    else: # No prompt signal found - continue ReAct loop
                        continue # Loop to let LLM process tool results
                else:
                    logger.warning("Tool execution failed or returned no messages. Breaking loop.")
                    console_io._stop_live()
                    break
            elif ai_content:
                # --- UPDATE DISPLAY WITH AI TEXT RESPONSE ---
                logger.info(f"AI: {ai_content[:100]}...")
                console_io.update_ai_response(ai_content) # Stops spinner, clears line, prints response
                break # End loop after text response
            else:
                # --- Handle empty AI response ---
                logger.warning("AI response had no content and no tool calls.")
                console_io.print_warning("AI returned an empty response.")
                console_io._stop_live()
                break

        except Exception as e:
            logger.error(f"LLM/Tool Error in main loop: {e}", exc_info=True)
            console_io.print_error(f"AI interaction error: {e}")
            console_io._stop_live()
            break

    # --- Max iterations handling ---
    if iteration >= max_iterations:
        logger.warning("Hit maximum iterations of ReAct loop")
        console_io.print_warning(f"Reached maximum of {max_iterations} AI interactions. Stopping.")
        console_io._stop_live()

# --- Other functions (start_temp_chat, edit_message, flush_temp_chats, execute, list_messages, list_toolsets) ---
# No changes required in these functions for the HITL flow.

def start_temp_chat(message: str) -> None:
    """Starts a temporary chat session."""
    safe_ts = str(time.time()).replace('.', '_')
    chat_title = f"Temp Chat {safe_ts}"
    logger.info(f"Starting temporary chat: {chat_title}")
    chat_file = create_or_load_chat(chat_title)  # create_or_load_chat handles session saving
    if chat_file:
        console_io.print_info(f"Started {chat_title}.")
        send_message(message)
    else:
        console_io.print_error(f"Could not start temp chat '{chat_title}'.")

def edit_message(index: Optional[int], new_message: str) -> None:
    """Edits a previous human message."""
    chat_file = get_current_chat()
    if not chat_file:
        console_io.print_error("No active chat.")
        return

    chat_messages = get_chat_messages(chat_file)
    if not chat_messages:
        console_io.print_error("No messages to edit.")
        return

    target_index = -1
    if index is None:  # Find last human message
        target_index = next((i for i in range(len(chat_messages) - 1, -1) if chat_messages[i].get("role") == "human"), -1)
        if target_index == -1:
            console_io.print_error("No previous user message found.")
            return
    else:  # Validate index
        if not isinstance(index, int) or not (0 <= index < len(chat_messages)):
            console_io.print_error(f"Index {index} out of range.")
            return
        if chat_messages[index].get("role") != "human":
            console_io.print_error(f"Cannot edit non-user message at index {index}.")
            return
        target_index = index

    original_content = chat_messages[target_index].get("content")
    edited_message_dict = {
        "role": "human", "content": new_message, "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {"edited": True, "original_content": original_content, "original_timestamp": chat_messages[target_index].get("timestamp")}
    }

    # Create a new messages list with the edited message and truncate subsequent messages
    new_messages = chat_messages[:target_index] + [edited_message_dict]
    original_len = len(chat_messages)
    num_removed = original_len - len(new_messages)

    # Save the updated messages
    _write_chat_messages(chat_file, new_messages)

    logger.info(f"Edited msg {target_index}. Removed {num_removed} subsequent.")
    console_io.print_info(f"Message {target_index} edited. Removed {num_removed} message(s). Resending...")
    # The original message is passed here, which might be slightly confusing if edited.
    # It should perhaps send the edited_message_dict["content"]? Yes.
    send_message(edited_message_dict["content"])


def flush_temp_chats() -> None:
    """Removes all temporary chat sessions."""
    removed_count = flush_temp_chats_state()
    console_io.print_info(f"Removed {removed_count} temporary chats.")

# --- Additional Commands ---
def execute(command: str) -> str:
    """Executes a shell command directly using the internal tool."""
    try:
        from .toolsets.terminal.toolset import run_direct_terminal_command
        logger.info(f"Executing direct command: {command}")
        output = run_direct_terminal_command(command)
        # Use console_io for printing the output - modified to use print_tool_output for consistency?
        # Let's use print_info for now as it's a direct user command, not an AI tool call result.
        console_io.print_info(f"Direct Execution Result:\n{output}")
        return output
    except ImportError:
        err_msg = "Error: Direct terminal tool function not available."
        logger.error(err_msg)
        console_io.print_error(err_msg)
        return err_msg
    except Exception as e:
        logger.error(f"Error executing direct command '{command}': {e}", exc_info=True)
        error_msg = f"Error executing command: {e}"
        console_io.print_error(error_msg)
        return error_msg

def list_messages() -> None:
    """Lists all messages in the current chat using Rich."""
    # ... (implementation remains the same) ...
    from rich.panel import Panel # Import Panel for better formatting
    from rich.text import Text # Import Text here

    chat_file = get_current_chat()
    if not chat_file:
        console_io.print_warning("No active chat.")
        return

    chat_messages = get_chat_messages(chat_file)
    if not chat_messages:
        console_io.print_info("No messages in chat.")
        return

    chat_title = get_current_chat_title() or "Current Chat"
    console_io.print_system(f"\n--- Chat History for: {chat_title} ---")

    roles_map = {
        "system": console_io.STYLE_SYSTEM_CONTENT, # Use content style for border
        "human": console_io.STYLE_USER,
        "ai": console_io.STYLE_AI_CONTENT, # Use content style for border
        "tool": console_io.STYLE_INFO_CONTENT, # Use content style for border
    }
    titles = {"system": "System", "human": "Human", "ai": "AI", "tool": "Tool"}

    for i, msg in enumerate(chat_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts_str = msg.get("timestamp", "No timestamp")
        try:
            # Try parsing with timezone, handle Z correctly
            if ts_str.endswith('Z'): ts_str = ts_str[:-1] + '+00:00'
            ts = datetime.fromisoformat(ts_str).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception: # Catch broad exceptions for parsing issues
            ts = ts_str # Fallback to original string

        role_style = roles_map.get(role, console_io.STYLE_INFO_CONTENT) # Default to info style
        role_title = titles.get(role, role.capitalize())

        # Format content
        content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
        display_content = content_str # Let Panel handle wrapping

        # Create the panel content with rich.text.Text
        panel_content = Text()

        # Add metadata like tool calls, edits inside the panel or as separate lines
        if role == "ai" and msg.get("tool_calls"):
             panel_content.append("Tool Calls Initiated:\n", style=console_io.STYLE_SYSTEM_CONTENT) # Use system content style
             # Ensure tool_calls is a list
             tool_calls_list = msg["tool_calls"] if isinstance(msg["tool_calls"], list) else []
             for tc in tool_calls_list:
                 # Ensure tc is a dict before accessing keys
                 if isinstance(tc, dict):
                     tc_text = Text(f"- Tool: ", style=console_io.STYLE_SYSTEM_CONTENT)
                     tc_text.append(f"{tc.get('name', '?')}", style=console_io.STYLE_TOOL_NAME)
                     tc_text.append(f", Args: {json.dumps(tc.get('args', {}))}", style=console_io.STYLE_ARG_VALUE)
                     tc_text.append(f", ID: {tc.get('id', '?')}\n", style=console_io.STYLE_ARG_NAME)
                     panel_content.append(tc_text)
                 else:
                      panel_content.append(f"- Invalid tool call entry: {tc}\n", style=console_io.STYLE_ERROR_CONTENT)
             panel_content.append("\n") # Add space before main content
        elif role == "tool" and msg.get("tool_call_id"):
             panel_content.append(f"For Tool Call ID: {msg['tool_call_id']}\n\n", style=console_io.STYLE_ARG_NAME)
        if msg.get("metadata", {}).get("edited"):
             panel_content.append(f"[Edited - Original TS: {msg['metadata'].get('original_timestamp', 'N/A')}]\n\n", style=console_io.STYLE_SYSTEM_CONTENT)

        # Append main content
        panel_content.append(display_content)

        # Create Panel
        console_io.console.print(
            Panel(
                panel_content,
                title=f"[{i}] {role_title}",
                subtitle=f"({ts})",
                border_style=role_style, # Use style for border
                title_align="left",
                subtitle_align="right"
            )
        )

    console_io.print_system("--- End of History ---")

# --- Toolset Management ---
def list_toolsets() -> None:
    """List all available toolsets and their status using console_io."""
    # ... (implementation remains the same) ...
    chat_file = get_current_chat()
    if not chat_file:
        console_io.print_error("No active chat session. Please create or load a chat first.")
        return

    enabled_toolsets = get_enabled_toolsets(chat_file)
    active_toolsets = get_active_toolsets(chat_file)

    console_io.print_info("--- Toolsets Status ---")

    from .toolsets.toolsets import get_registered_toolsets
    all_toolsets_meta = get_registered_toolsets()  # Dict[id, ToolsetMetadata]

    if not all_toolsets_meta:
        console_io.print_warning("No toolsets found/registered.")
        return

    for ts_id, metadata in sorted(all_toolsets_meta.items(), key=lambda item: item[1].name):
        toolset_name = metadata.name
        # Check against display names
        is_enabled = toolset_name in enabled_toolsets
        is_active = toolset_name in active_toolsets

        enabled_status = "[green]✓[/]" if is_enabled else "[dim]✗[/]"
        active_status = "[green]✓[/]" if is_active else "[dim]✗[/]"
        # Use console.print for rich formatting
        console_io.console.print(f"- {toolset_name} ([dim]ID: {ts_id}[/dim]): [Enabled: {enabled_status}] [Active: {active_status}]")

    console_io.console.print("\n[dim]Enabled: Toolset is available for use[/dim]")
    console_io.console.print("[dim]Active: Toolset's tools are currently offered to the LLM[/dim]")
    console_io.console.print("\nUse [bold]--select-tools[/bold] to manage enabled toolsets.")