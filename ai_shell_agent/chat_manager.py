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
from . import logger

# --- Import necessary components from the state manager ---
from .chat_state_manager import (
    get_current_chat, save_session, create_or_load_chat, 
    # _read_chat_data, - Remove old function
    # _write_chat_data, - Remove old function
    get_active_toolsets, get_enabled_toolsets,
    # update_active_toolsets, update_enabled_toolsets, # State updated via ai.py or tools
    # REMOVED: enable_toolset, disable_toolset, activate_toolset, deactivate_toolset, update_system_prompt_for_toolsets
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
    """Prints the list of available chat titles."""
    # Use internal helper from state manager - Correct
    from .chat_state_manager import _read_json, CHAT_MAP_FILE # Import locally if needed
    chat_map = _read_json(CHAT_MAP_FILE, {})
    current_chat_id = get_current_chat()
    if not chat_map: print("No chats found."); return
    print("\nAvailable Chats:")
    for chat_id, title in sorted(chat_map.items(), key=lambda item: item[1].lower()):
        marker = " <- Current" if chat_id == current_chat_id else ""
        print(f"- {title}{marker}")

def rename_chat(old_title: str, new_title: str) -> None:
    """Renames a chat session by calling the state manager."""
    if rename_chat_state(old_title, new_title): print(f"Renamed chat: {old_title} -> {new_title}")

def delete_chat(title: str) -> None:
    """Deletes a chat session by calling the state manager."""
    if delete_chat_state(title): print(f"Deleted chat: {title}")

# ---------------------------
# Messaging Functions
# ---------------------------
def _handle_tool_calls(ai_message: AIMessage, chat_file: str) -> list[BaseMessage]:
    """Handle tool calls from AI response. Invokes tools and returns ToolMessages."""
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

        logger.info(f"Tool Call Requested: {tool_name}(args={tool_args}) ID: {tool_call_id}")
        if tool_name not in tool_registry_dict:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call_id))
            continue

        tool_instance = tool_registry_dict[tool_name]
        try:
            tool_result_content = tool_instance.invoke(input=tool_args) # Invoke requires 'input' key
            if not isinstance(tool_result_content, str):
                try: tool_result_content = json.dumps(tool_result_content)
                except TypeError: tool_result_content = str(tool_result_content)
            tool_response = ToolMessage(content=tool_result_content, tool_call_id=tool_call_id)
            logger.info(f"Tool '{tool_name}' executed. Result preview: {tool_result_content[:200]}...")
            # Check for Aider signal
            if isinstance(tool_response.content, str) and tool_response.content.startswith(SIGNAL_PROMPT_NEEDED):
                logger.info(f"Tool '{tool_name}' signaled Aider input needed.")
            messages.append(tool_response)
        except Exception as e:
            logger.error(f"Error running tool '{tool_name}' with args {tool_args}: {e}", exc_info=True)
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
            valid_calls = [c for c in all_calls if c.get("id")]
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
    if not chat_file: logger.warning("No active chat. Starting temp."); start_temp_chat(message); return

    # 1. Add Human Message
    human_msg_dict = {"role": "human", "content": message, "timestamp": datetime.now(timezone.utc).isoformat()}
    current_messages = get_chat_messages(chat_file)
    current_messages.append(human_msg_dict)
    _write_chat_messages(chat_file, current_messages)
    logger.info(f"Human: {message}")

    # 2. ReAct Loop
    max_iterations = 10; iteration = 0; last_ai_content = None
    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"ReAct iteration {iteration}/{max_iterations}")
        chat_history_dicts = get_chat_messages(chat_file)

        # --- Message Validation Logic (Keep the existing robust logic) ---
        validated_messages = []
        pending_tool_calls = {}; responded_tool_call_ids = set(); skip_indices = set()
        for i, msg in enumerate(chat_history_dicts):
             if msg.get("role") == "ai" and msg.get("tool_calls"):
                  calls = {tc["id"]: tc for tc in msg["tool_calls"] if tc.get("id")};
                  if calls: pending_tool_calls[i] = calls
             elif msg.get("role") == "tool" and msg.get("tool_call_id"): responded_tool_call_ids.add(msg["tool_call_id"])
        for i, msg in enumerate(chat_history_dicts):
            if i in skip_indices: continue
            if i in pending_tool_calls:
                all_resp = all(tc_id in responded_tool_call_ids for tc_id in pending_tool_calls[i])
                if all_resp:
                    validated_messages.append(msg)
                    tool_resps = []
                    for j in range(i + 1, len(chat_history_dicts)):
                         next_msg = chat_history_dicts[j]
                         if next_msg.get("role") == "tool" and next_msg.get("tool_call_id") in pending_tool_calls[i]:
                              tool_resps.append(next_msg); skip_indices.add(j)
                         elif next_msg.get("role") == "ai": break
                    validated_messages.extend(tool_resps)
                else: logger.warning(f"Skipping AI msg {i}, pending tool calls."); # Skip AI msg and its tool responses
            else: validated_messages.append(msg)
        # --- End Validation ---

        if not validated_messages: logger.error("Validation empty history."); print("[Error]: History processing failed."); break
        lc_messages = _convert_message_dicts_to_langchain(validated_messages)
        if not lc_messages or lc_messages[0].type != "system": logger.warning("No system prompt found.")

        # --- Get LLM (No args needed now) ---
        try: llm_instance = get_llm()
        except Exception as e: logger.error(f"LLM Init fail: {e}", exc_info=True); print(f"\n[Error]: AI model init failed: {e}"); break

        try:
            # --- Invoke LLM ---
            ai_response: AIMessage = llm_instance.invoke(lc_messages)
            logger.debug(f"AI Raw Response Content: {ai_response.content}")

            # --- Save AI response ---
            ai_msg_dict_list = _convert_langchain_to_message_dicts([ai_response])
            if not ai_msg_dict_list: logger.error("AI resp convert fail."); print("\n[Error]: Internal error."); break
            ai_msg_dict = ai_msg_dict_list[0]
            current_messages = get_chat_messages(chat_file)
            current_messages.append(ai_msg_dict)
            _write_chat_messages(chat_file, current_messages)

            # --- Print & Handle Tool Calls ---
            has_tool_calls = bool(ai_msg_dict.get("tool_calls")) # Check saved dict
            if ai_response.content and (ai_response.content != last_ai_content or not has_tool_calls):
                 print(f"\n[AI]: {ai_response.content}"); last_ai_content = ai_response.content
            elif not ai_response.content and has_tool_calls: print(f"\n[AI]: Requesting tool execution...")
            logger.info(f"AI: {ai_response.content or '[No text content, using tools]'}")

            if has_tool_calls:
                logger.info(f"AI requesting {len(ai_response.tool_calls)} tool call(s)...")
                tool_messages = _handle_tool_calls(ai_response, chat_file) # Returns list[ToolMessage]
                if tool_messages:
                    tool_message_dicts = _convert_langchain_to_message_dicts(tool_messages)
                    current_messages = get_chat_messages(chat_file)
                    current_messages.extend(tool_message_dicts)
                    _write_chat_messages(chat_file, current_messages)
                    stop_loop = False
                    for tool_msg in tool_messages: # Process results/signals
                        tool_name = next((tc.get("name") for tc in ai_response.tool_calls if tc.get("id") == tool_msg.tool_call_id), "Unknown")
                        if isinstance(tool_msg.content, str) and tool_msg.content.startswith(SIGNAL_PROMPT_NEEDED):
                             print(f"\n[File Editor Input Required]: {tool_msg.content[len(SIGNAL_PROMPT_NEEDED):].strip()}"); stop_loop = True
                        else: print(f"\n[Tool Result - {tool_name}]:\n  " + str(tool_msg.content).replace('\n', '\n  '))
                    if stop_loop: logger.info("Aider input needed. Break loop."); break
                    continue # Loop back for LLM reaction
                else: logger.warning("Tool exec failed. Break loop."); break # Break if tools failed
            else: logger.debug("No tool calls. Loop complete."); break # Break if no tools called
        except Exception as e: logger.error(f"LLM/Tool Error: {e}", exc_info=True); print(f"\n[Error]: AI interaction error: {e}"); break
    if iteration >= max_iterations: logger.warning("Max iterations."); print(f"\n[Warning]: Max tool interactions ({max_iterations}) reached.")

def start_temp_chat(message: str) -> None:
    """Starts a temporary chat session."""
    safe_ts = str(time.time()).replace('.', '_'); chat_title = f"Temp Chat {safe_ts}"
    logger.info(f"Starting temporary chat: {chat_title}")
    chat_file = create_or_load_chat(chat_title) # create_or_load_chat handles session saving
    if chat_file: print(f"Started {chat_title}."); send_message(message)
    else: print(f"Error: Could not start temp chat '{chat_title}'.")

def edit_message(index: Optional[int], new_message: str) -> None:
    """Edits a previous human message."""
    chat_file = get_current_chat();
    if not chat_file: print("No active chat."); return
    chat_messages = get_chat_messages(chat_file)
    if not chat_messages: print("No messages to edit."); return

    target_index = -1
    if index is None: # Find last human message
        target_index = next((i for i in range(len(chat_messages) - 1, -1, -1) if chat_messages[i].get("role") == "human"), -1)
        if target_index == -1: print("Error: No previous user message found."); return
    else: # Validate index
        if not isinstance(index, int) or not (0 <= index < len(chat_messages)): print(f"Error: Index {index} out of range."); return
        if chat_messages[index].get("role") != "human": print(f"Error: Cannot edit non-user message at index {index}."); return
        target_index = index

    original_content = chat_messages[target_index].get("content")
    edited_message_dict = {
        "role": "human", "content": new_message, "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {"edited": True, "original_content": original_content, "original_timestamp": chat_messages[target_index].get("timestamp")} }

    # Create a new messages list with the edited message and truncate subsequent messages
    new_messages = chat_messages[:target_index] + [edited_message_dict]
    original_len = len(chat_messages)
    num_removed = original_len - len(new_messages)
    
    # Save the updated messages
    _write_chat_messages(chat_file, new_messages)

    logger.info(f"Edited msg {target_index}. Removed {num_removed} subsequent.")
    print(f"Message {target_index} edited. Removed {num_removed} message(s). Resending...")
    send_message(new_message)

def flush_temp_chats() -> None:
    """Removes all temporary chat sessions."""
    removed_count = flush_temp_chats_state()
    print(f"Removed {removed_count} temporary chats.")

# --- Additional Commands ---
def execute(command: str) -> str:
    """Executes a shell command directly using the internal tool."""
    try:
        # Import the runner function from the toolset module
        from .toolsets.terminal.toolset import run_direct_terminal_command
        logger.info(f"Executing direct command: {command}")
        output = run_direct_terminal_command(command)
        print(output)
        return output
    except ImportError:
        err_msg = "Error: Direct terminal tool function not available."
        logger.error(err_msg); print(err_msg); return err_msg
    except Exception as e:
        logger.error(f"Error executing direct command '{command}': {e}", exc_info=True)
        error_msg = f"Error executing command: {e}"; print(error_msg); return error_msg

def list_messages() -> None:
    """Lists all messages in the current chat."""
    chat_file = get_current_chat()
    if not chat_file: print("No active chat."); return
    chat_messages = get_chat_messages(chat_file)
    if not chat_messages: print("No messages in chat."); return

    print(f"\n--- Chat History for: {get_current_chat_title()} ---")
    titles = {"system": "System", "human": "Human", "ai": "AI", "tool": "Tool"}
    for i, msg in enumerate(chat_messages):
        role = msg.get("role", "unknown"); content = msg.get("content", "")
        ts_str = msg.get("timestamp", "No timestamp")
        try: ts = datetime.fromisoformat(ts_str).strftime('%Y-%m-%d %H:%M:%S %Z')
        except: ts = ts_str
        print(f"\n[{i}] {titles.get(role, role.capitalize())} ({ts}):")
        content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
        max_len = 1000
        display_content = content_str[:max_len] + f"... (truncated, {len(content_str)} chars)" if len(content_str) > max_len else content_str
        print("  " + display_content.replace('\n', '\n  '))
        if role == "ai" and msg.get("tool_calls"):
             print("    [Tool Calls Initiated]:")
             for tc in msg["tool_calls"]: print(f"      - Tool: {tc.get('name', '?')}, Args: {json.dumps(tc.get('args', {}))}, ID: {tc.get('id', '?')}")
        elif role == "tool" and msg.get("tool_call_id"): print(f"    [For Tool Call ID]: {msg['tool_call_id']}")
        if msg.get("metadata", {}).get("edited"): print(f"    [Edited - Original TS: {msg['metadata'].get('original_timestamp', 'N/A')}]")
    print("\n--- End of History ---")

# --- Toolset Management ---
def list_toolsets() -> None:
    """List all available toolsets and their status (enabled and active)."""
    chat_file = get_current_chat()
    if not chat_file:
        print("No active chat session. Please create or load a chat first.")
        return
    
    # Get current toolsets
    enabled_toolsets = get_enabled_toolsets(chat_file)
    active_toolsets = get_active_toolsets(chat_file)
    
    print("\n--- Toolsets Status ---")
    print("Available toolsets:")
    
    from .toolsets.toolsets import get_toolset_ids, get_registered_toolsets
    all_toolsets = get_toolset_ids()
    toolset_metadata = get_registered_toolsets()
    
    for toolset_id in sorted(all_toolsets):
        metadata = toolset_metadata.get(toolset_id)
        if not metadata:
            continue
            
        toolset_name = metadata.name
        enabled_status = "✓" if toolset_name in enabled_toolsets else "✗"
        active_status = "✓" if toolset_name in active_toolsets else "✗"
        print(f"- {toolset_name} (ID: {toolset_id}): [Enabled: {enabled_status}] [Active: {active_status}]")
    
    print("\nEnabled: Toolset is loaded and available for use")
    print("Active: Toolset is currently being used")
    print("\nUse --select-tools to manage enabled toolsets")