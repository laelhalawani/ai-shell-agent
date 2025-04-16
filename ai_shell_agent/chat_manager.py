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
from . import logger, console_io
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
        console_io.print_system_message("No chats found.", level="INFO")
        return
    console_io.print_system_message("\nAvailable Chats:", level="INFO")
    for chat_id, title in sorted(chat_map.items(), key=lambda item: item[1].lower()):
        marker = " <- Current" if chat_id == current_chat_id else ""
        # Use console.print directly for simple list formatting
        console_io.print_list_item(f"{title}{marker}")

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

        logger.debug(f"Tool Call Requested: {tool_name}(args={tool_args}) ID: {tool_call_id}") # Keep debug log

        if tool_name not in tool_registry_dict:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call_id))
            continue

        tool_instance = tool_registry_dict[tool_name]
        try:
            # *** Tool Execution Happens Here (potentially blocking for HITL) ***
            tool_result_content = tool_instance.invoke(input=tool_args)
            # *** End Tool Execution ***

            if not isinstance(tool_result_content, str):
                try: tool_result_content = json.dumps(tool_result_content)
                except TypeError: tool_result_content = str(tool_result_content)
            tool_response = ToolMessage(content=tool_result_content, tool_call_id=tool_call_id)

            logger.debug(f"Tool '{tool_name}' executed. Result preview: {tool_result_content[:200]}...") # Keep debug log

            # Keep Aider signal check logic here for logging, but primary check is in send_message
            if isinstance(tool_response.content, str) and tool_response.content.startswith(SIGNAL_PROMPT_NEEDED):
                logger.debug(f"Tool '{tool_name}' signaled Aider input needed.") # Log signal detection

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
    if not chat_file: 
        # Use console_io for warning
        console_io.print_warning("No active chat. Starting temp chat.")
        start_temp_chat(message)
        return

    # 1. Add Human Message (No print needed here, input is visible)
    human_msg_dict = {"role": "human", "content": message, "timestamp": datetime.now(timezone.utc).isoformat()}
    current_messages = get_chat_messages(chat_file)
    current_messages.append(human_msg_dict)
    _write_chat_messages(chat_file, current_messages)
    logger.debug(f"Human message added to chat {chat_file}: {message[:100]}...")

    # 2. ReAct Loop
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"ReAct iteration {iteration}/{max_iterations}")
        chat_history_dicts = get_chat_messages(chat_file)

        # --- Message Validation Logic ---
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
                else: logger.warning(f"Skipping AI msg {i}, pending tool calls.")
            else: validated_messages.append(msg)
        # --- End Validation ---

        if not validated_messages:
            logger.error("Validation resulted in empty history.")
            console_io.print_error("History processing failed.")
            break
            
        lc_messages = _convert_message_dicts_to_langchain(validated_messages)
        if not lc_messages or lc_messages[0].type != "system": 
            logger.warning("No system prompt found or first message is not system.")

        # --- Get LLM ---
        try: 
            llm_instance = get_llm()
        except Exception as e: 
            logger.error(f"LLM Init fail: {e}", exc_info=True)
            console_io.print_error(f"AI model initialization failed: {e}")
            break

        try:
            # --- START THINKING ---
            console_io.start_ai_thinking()
            
            # --- Invoke LLM ---
            ai_response: AIMessage = llm_instance.invoke(lc_messages)
            logger.debug(f"AI Raw Response Content: {ai_response.content}")
            logger.debug(f"AI Raw Response Tool Calls: {ai_response.tool_calls}")

            # --- Save AI response FIRST ---
            ai_msg_dict_list = _convert_langchain_to_message_dicts([ai_response])
            if not ai_msg_dict_list:
                logger.error("AI response conversion failed.")
                console_io.print_error("Internal error processing AI response.")
                # Need to stop thinking indicator here
                console_io._stop_live() # Use internal stop as we didn't update
                break
                
            ai_msg_dict = ai_msg_dict_list[0]
            current_messages = get_chat_messages(chat_file)
            current_messages.append(ai_msg_dict)
            _write_chat_messages(chat_file, current_messages)

            # --- Print & Handle Tool Calls ---
            has_tool_calls = bool(ai_msg_dict.get("tool_calls"))
            ai_content = ai_response.content

            if has_tool_calls:
                logger.info(f"AI requesting {len(ai_response.tool_calls)} tool call(s)...")

                # --- UPDATE LIVE DISPLAY TO 'USING TOOL' ---
                # For now, show the first tool call. Expand later if needed.
                first_tool_call = ai_response.tool_calls[0]
                tool_name = first_tool_call.get("name", "unknown_tool")
                tool_args = first_tool_call.get("args", {})
                console_io.update_ai_tool_call(tool_name, tool_args)

                # --- EXECUTE TOOL (This might block for HITL) ---
                # _handle_tool_calls itself should not print to console
                tool_messages = _handle_tool_calls(ai_response, chat_file)

                # --- Process Tool Results ---
                if tool_messages:
                    # Save tool results
                    tool_message_dicts = _convert_langchain_to_message_dicts(tool_messages)
                    current_messages = get_chat_messages(chat_file)  # Re-read after potential HITL delay
                    current_messages.extend(tool_message_dicts)
                    _write_chat_messages(chat_file, current_messages)

                    # Combine results for final display
                    # For now, just combine content strings simply. Improve later if needed.
                    combined_result_content = "\n".join(
                        [str(tm.content) for tm in tool_messages if hasattr(tm, 'content')]
                    )

                    # Check for Aider Signal *before* finalizing
                    aider_signal_output = None
                    if isinstance(combined_result_content, str) and combined_result_content.startswith(SIGNAL_PROMPT_NEEDED):
                        signal_info = combined_result_content[len(SIGNAL_PROMPT_NEEDED):].strip()
                        aider_signal_output = f"[File Editor Input Required]: {signal_info}"
                        # Remove signal from the result passed to finalize
                        combined_result_content = "" # Or some indicator? Let's use empty for now.
                        logger.info(f"Tool '{tool_name}' signaled Aider input needed.")

                    # --- UPDATE LIVE DISPLAY TO 'USED TOOL: RESULT' ---
                    console_io.finalize_ai_tool_result(combined_result_content)

                    # Print Aider signal *after* finalizing the tool result line
                    if aider_signal_output:
                         console_io.print_info(aider_signal_output)

                    # Continue loop for LLM reaction to tool results
                    continue
                else:
                    logger.warning("Tool execution failed or returned no messages. Breaking loop.")
                    console_io.print_error(f"Tool '{tool_name}' execution failed.")
                    # Stop thinking/tool indicator here as well
                    console_io._stop_live()
                    break  # Break if tool execution failed critically
            
            elif ai_content:
                 # --- UPDATE LIVE DISPLAY WITH AI TEXT RESPONSE ---
                 # This handles the case where the AI responds with text after a tool call,
                 # or responds with only text initially.
                 logger.info(f"AI: {ai_content[:100]}...")
                 console_io.update_ai_response(ai_content)
                 break # Break loop after AI text response
            else:
                 # AI responded with neither content nor tool calls (should be rare)
                 logger.warning("AI response had no content and no tool calls.")
                 console_io.print_warning("AI returned an empty response.")
                 console_io._stop_live() # Stop thinking indicator
                 break # Break loop

        except Exception as e:
            logger.error(f"LLM/Tool Error: {e}", exc_info=True)
            console_io.print_error(f"AI interaction error: {e}")
            console_io._stop_live() # Ensure thinking stops on error
            break  # Exit loop on error

    if iteration >= max_iterations:
        console_io._stop_live()  # Ensure thinking stops
        logger.warning("Max iterations reached.")
        console_io.print_warning(f"Max tool interactions ({max_iterations}) reached.")

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
    send_message(new_message)

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
        # Use console_io for printing the output
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
    from rich.panel import Panel # Import Panel for better formatting
    
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
        "system": console_io.STYLE_SYSTEM,
        "human": console_io.STYLE_USER, # Use User style
        "ai": console_io.STYLE_AI,
        "tool": console_io.STYLE_INFO, # Use Info style for Tool
    }
    titles = {"system": "System", "human": "Human", "ai": "AI", "tool": "Tool"}

    for i, msg in enumerate(chat_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts_str = msg.get("timestamp", "No timestamp")
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        except:
            ts = ts_str

        role_style = roles_map.get(role, console_io.STYLE_INFO) # Default to info style
        role_title = titles.get(role, role.capitalize())

        # Format content
        content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
        display_content = content_str # Let Panel handle wrapping

        # Create the panel content with rich.text.Text
        from rich.text import Text
        panel_content = Text()
        
        # Add metadata like tool calls, edits inside the panel or as separate lines
        if role == "ai" and msg.get("tool_calls"):
             panel_content.append("Tool Calls Initiated:\n", style=console_io.STYLE_SYSTEM)
             for tc in msg["tool_calls"]:
                  tc_text = Text(f"- Tool: ", style=console_io.STYLE_SYSTEM)
                  tc_text.append(f"{tc.get('name', '?')}", style=console_io.STYLE_TOOL_NAME)
                  tc_text.append(f", Args: {json.dumps(tc.get('args', {}))}", style=console_io.STYLE_ARG_VALUE)
                  tc_text.append(f", ID: {tc.get('id', '?')}\n", style=console_io.STYLE_ARG_NAME)
                  panel_content.append(tc_text)
             panel_content.append("\n") # Add space before main content
        elif role == "tool" and msg.get("tool_call_id"):
             panel_content.append(f"For Tool Call ID: {msg['tool_call_id']}\n\n", style=console_io.STYLE_ARG_NAME)
        if msg.get("metadata", {}).get("edited"):
             panel_content.append(f"[Edited - Original TS: {msg['metadata'].get('original_timestamp', 'N/A')}]\n\n", style=console_io.STYLE_SYSTEM)

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