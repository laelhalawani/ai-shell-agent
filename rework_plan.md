Okay, here is the finalized, detailed integration plan for incorporating Aider into AI Shell Agent using the threading/queue mechanism.

**I. Overall Goal & Approach**

*   **Goal:** Seamlessly integrate Aider's code editing capabilities into AI Shell Agent as a suite of Langchain Tools. The agent should manage the lifecycle of an Aider session (start, add/drop files, edit, undo, close) based on user instructions within a chat.
*   **Core Approach:** Use a separate thread for potentially blocking Aider operations (`coder.run`). Employ input/output queues (`queue.Queue`) for non-blocking communication between the Aider thread and the agent's tools. Maintain Aider's state through a combination of an in-memory `Coder` object for the active session and persistent storage within the chat's JSON file.

**II. High-Level Plan**

1.  **Implement `AiderIOStubWithQueues`:** A custom IO handler for Aider that intercepts blocking prompts and uses queues for communication.
2.  **Manage In-Memory State:** Create a mechanism (`active_coders` dictionary) to hold the live `Coder` instance, thread, queues, and IO stub for the active chat session.
3.  **Manage Persistent State:** Modify `chat_manager.py` to store and retrieve Aider's serializable state (file paths, commit hashes, Aider's internal chat history) within the chat JSON file's `aider_state` dictionary.
4.  **Implement Threading:** Run the core `coder.run()` method in a separate thread when the `RunCodeEditTool` is invoked.
5.  **Define/Refine Tools:**
    *   `StartCodeEditorTool`: Initializes both persistent and in-memory state.
    *   `RunCodeEditTool`: Starts the Aider thread, handles the *first* interaction (either a prompt request or the final result).
    *   `SubmitCodeEditorInputTool`: Sends the agent's input back to the waiting Aider thread and handles *subsequent* interactions.
    *   `AddFileTool`, `DropFileTool`, `ListFilesInEditorTool`, `ViewDiffTool`, `UndoLastEditTool`: Operate based on the current state (in-memory if active, persistent otherwise for `List`).
    *   `CloseCodeEditorTool`: Cleans up in-memory and persistent state.
6.  **Update Tool Availability:** The agent's core logic (outside these tool definitions) will need to determine which set of tools (basic or including Aider coding tools) are available based on whether `get_active_coder_state` returns a valid state for the current chat.

**III. Key Considerations**

*   **Concurrency:** Use `threading.Lock` to protect access to the shared `active_coders` dictionary.
*   **State Synchronization:** Ensure that relevant state changes made by the in-memory `Coder` instance (added/dropped files, commits, Aider's internal `done_messages`) are consistently saved back to the persistent `aider_state` in the chat JSON using `update_aider_state_from_coder`.
*   **Interactivity Bridge:** The `AiderIOStubWithQueues`, `RunCodeEditTool`, and `SubmitCodeEditorInputTool` form the crucial bridge. The message format on the `output_q` must clearly distinguish between needing input and completion/error. The formatting of the message *back to the agent* must be consistent.
*   **Error Handling:** Implement robust error handling for thread exceptions, tool execution errors (e.g., session not active), and potential queue issues.
*   **Aider as Library:** Strictly use Aider's Python classes and functions. No CLI calls.
*   **Aider's Internal History:** Saving `coder.done_messages` to the persistent `aider_state` is critical for Aider to maintain context correctly when resuming or running subsequent edits. This is separate from the AI Shell Agent's primary chat history.

**IV. Detailed Implementation Plan (File by File)**

1.  **`aider_integration.py` (Create/Heavily Modify)**

    *   **Imports:** `queue`, `threading`, `traceback`, `json`, `dataclasses`, relevant classes from `aider` (`Coder`, `InputOutput`, `GitRepo`, `Model`, `Commands`, `ANY_GIT_ERROR`), `logger`, etc.
    *   **`AiderIOStubWithQueues(InputOutput)` Class:**
        *   `__init__(self)`: Initialize `self.input_q = None`, `self.output_q = None`, `self.group_preferences = {}`, `self.never_prompts = set()`. Also initialize buffers for captured output/errors/warnings.
        *   `set_queues(self, input_q, output_q)`: Method to assign queues after instantiation.
        *   `confirm_ask(self, question, default="y", subject=None, explicit_yes_required=False, group=None, allow_never=False)`:
            *   *Purpose:* Intercept Aider's request for yes/no confirmation.
            *   Check `group_preferences` and `never_prompts` for early exit.
            *   Prepare `prompt_data = {'type': 'prompt', 'prompt_type': 'confirm', 'question': question, 'default': default, 'subject': subject, 'allow_never': allow_never, 'group_id': id(group) if group else None}`.
            *   Assert `self.output_q` is set. `self.output_q.put(prompt_data)`.
            *   Assert `self.input_q` is set. `raw_response = self.input_q.get()`.
            *   Process `raw_response` (lowercase, strip). Handle 'a', 's', 'd' for group/never logic, updating internal state (`group_preferences`, `never_prompts`).
            *   Determine boolean result based on response, default, and `explicit_yes_required`.
            *   Return the boolean result.
        *   `prompt_ask(self, question, default="", subject=None)`:
            *   *Purpose:* Intercept Aider's request for arbitrary text input.
            *   Prepare `prompt_data = {'type': 'prompt', 'prompt_type': 'input', 'question': question, 'default': default, 'subject': subject}`.
            *   Assert `self.output_q` is set. `self.output_q.put(prompt_data)`.
            *   Assert `self.input_q` is set. `raw_response = self.input_q.get()`.
            *   Return `raw_response`.
        *   `tool_output`, `tool_error`, `tool_warning`: Implement capture logic (append to internal lists).
        *   `get_captured_output`: Combine captured lists into a string and clear the lists.
    *   **`ActiveCoderState` Dataclass:**
        *   *Purpose:* Hold the live state for an active Aider session.
        *   Fields: `coder: Coder`, `thread: Optional[threading.Thread]`, `input_q: queue.Queue`, `output_q: queue.Queue`, `io_stub: AiderIOStubWithQueues`.
    *   **Global State:**
        *   `active_coders: Dict[str, ActiveCoderState] = {}`
        *   `active_coders_lock = threading.Lock()`
    *   **Helper Functions:**
        *   `get_active_coder_state(chat_file: str) -> Optional[ActiveCoderState]`: Get state safely with lock.
        *   `create_active_coder_state(chat_file: str, coder: Coder, io_stub: AiderIOStubWithQueues) -> ActiveCoderState`: Create state with lock.
        *   `remove_active_coder_state(chat_file: str)`: Remove state safely with lock (potentially join thread if needed).
    *   **`_run_aider_in_thread` Function:**
        *   *Purpose:* The target function for the Aider worker thread.
        *   Takes `coder`, `instruction`, `output_q` as args.
        *   Wraps `coder.run(with_message=instruction)` in a try/except block.
        *   On success: `output_q.put({'type': 'result', 'content': coder.io_stub.get_captured_output()})`.
        *   On exception: `output_q.put({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})`.
    *   **`update_aider_state_from_coder(chat_file: str, coder: Coder)` Function:**
        *   *Purpose:* Syncs the live coder state to the persistent `aider_state`.
        *   Gets current persistent `aider_state` using `chat_manager.get_aider_state()`.
        *   Updates the dictionary: `abs_fnames`, `abs_read_only_fnames`, `git_root` (if `coder.repo`), `aider_commit_hashes`, *`aider_done_messages = coder.done_messages`*.
        *   Saves the updated dictionary using `chat_manager.save_aider_state()`.
    *   **Tool Classes:**
        *   **`StartCodeEditorTool`:**
            *   `_run`: Gets `chat_file`. Calls `chat_manager.clear_aider_state`. Determines initial settings. Saves initial state via `chat_manager.save_aider_state`. Creates `io_stub = AiderIOStubWithQueues()`. Creates `coder = Coder.create(...)`. Creates queues. Stores in `active_coders` via `create_active_coder_state`. Calls `io_stub.set_queues(input_q, output_q)`. Returns success message.
        *   **`AddFileTool`, `DropFileTool`:**
            *   `_run(file_path: str)`: Gets `chat_file`. Locks. Gets `state`. Calls `state.coder.add_rel_fname(file_path)` or `state.coder.drop_rel_fname(...)`. Calls `update_aider_state_from_coder`. Unlocks. Returns captured output from `state.io_stub.get_captured_output()`.
        *   **`ListFilesInEditorTool`:**
            *   `_run`: Gets `chat_file`. Calls `chat_manager.get_aider_state`. Formats and returns list based on `aider_state['abs_fnames']`.
        *   **`RunCodeEditTool`:**
            *   `_run(instruction: str)`: Gets `chat_file`. Locks. Gets `state`. Checks `state.thread`. Clears `state.input_q`, `state.output_q`. Starts `_run_aider_in_thread` in `state.thread`. Unlocks. `message = state.output_q.get()`. Processes `message`:
                *   If `prompt`: Format `response_text` including subject, question, and guidance on special commands (Y/N/A/S/D if applicable based on `prompt_data`). Return `f"[CODE_EDITOR_INPUT_NEEDED] {response_text}"`.
                *   If `result`: Call `update_aider_state_from_coder`. Return `message['content']`. `state.thread = None`.
                *   If `error`: Return error. `state.thread = None`.
        *   **`SubmitCodeEditorInputTool`:**
            *   `_run(user_response: str)`: Gets `chat_file`. Locks. Gets `state`. Checks `state.thread`. Puts `user_response` (raw string) onto `state.input_q`. Unlocks. `message = state.output_q.get()`. Processes `message` identically to `RunCodeEditTool`.
        *   **`ViewDiffTool`, `UndoLastEditTool`:**
            *   `_run`: Gets `chat_file`. Locks. Gets `state`. Ensures `state.coder.commands` exists. Calls `state.coder.commands.raw_cmd_diff("")` or `state.coder.commands.raw_cmd_undo(None)`. Calls `update_aider_state_from_coder` (especially important for Undo). Unlocks. Returns captured output from `state.io_stub.get_captured_output()`.
        *   **`CloseCodeEditorTool`:**
            *   `_run`: Gets `chat_file`. Locks. Calls `remove_active_coder_state`. Unlocks. Calls `chat_manager.clear_aider_state`. Returns success message.

2.  **`chat_manager.py` (Modifications)**

    *   **`_read_chat_data`:**
        *   *Purpose:* Read both agent messages and persistent aider state.
        *   Loads JSON. Deserializes agent `messages`. Loads `aider_state` dictionary, ensuring `aider_done_messages` exists as a list (default to `[]` if missing).
    *   **`_write_chat_data`:**
        *   *Purpose:* Write both agent messages and persistent aider state.
        *   Takes `messages` and `aider_state`. Serializes `messages`. Filters `None` from `aider_state` (except `aider_done_messages`). Dumps combined structure to JSON.
    *   **`get_aider_state(chat_file: str)`:**
        *   *Purpose:* Provide easy access to just the persistent aider state.
        *   Calls `_read_chat_data`, returns only the `aider_state` dictionary.
    *   **`save_aider_state(chat_file: str, state_dict: Dict[str, Any])`:**
        *   *Purpose:* Update only the persistent aider state, leaving agent messages intact.
        *   Calls `messages, _ = _read_chat_data(chat_file)`. Calls `_write_chat_data(chat_file, messages, state_dict)`.
    *   **`clear_aider_state(chat_file: str)`:**
        *   *Purpose:* Remove the persistent aider state.
        *   Calls `messages, _ = _read_chat_data(chat_file)`. Calls `_write_chat_data(chat_file, messages, {})`.

3.  **`tools.py` (Modifications)**

    *   Import tool instances (`start_code_editor_tool`, `add_code_file_tool`, ..., `submit_code_editor_input_tool`, `close_code_editor_tool`) from `aider_integration.py`.
    *   Add these instances to the main `tools` list (or manage their availability elsewhere).
    *   Update `tools_functions` list comprehension.

4.  **`ai_shell_agent/ai.py` (Implied Changes)**

    *   *Purpose:* Orchestrate tool availability and handle the multi-step edit process.
    *   Modify the agent execution loop (or wherever tools are selected/called).
    *   Before providing tools to the LLM for a turn, check if an Aider session is active for the current `chat_file` using `aider_integration.get_active_coder_state(chat_file)`.
    *   If active, provide the full set of Aider coding tools. If not, provide only `StartCodeEditorTool`.
    *   After receiving an LLM response requesting a tool call:
        *   If the response from a tool (specifically `RunCodeEditTool` or `SubmitCodeEditorInputTool`) starts with `[CODE_EDITOR_INPUT_NEEDED]`, parse the rest of the message. Present this request to the LLM in the next turn, expecting it to use `SubmitCodeEditorInputTool`.
        *   Otherwise, process the tool response normally.

**V. Flow Comparison**

| Action             | Aider Original CLI Flow                                     | Integrated Flow                                                                                                                                                                                                                         |
| :----------------- | :---------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Start**          | `aider <files>` (args parsed, Coder created)                | Agent calls `StartCodeEditorTool` -> Creates persistent & in-memory state, Coder instance, queues.                                                                                                                                     |
| **Add File**       | User types `/add <file>` -> `commands.run()` -> `coder.add` | Agent calls `AddFileTool` -> Retrieves active coder -> `coder.add` -> Saves persistent state.                                                                                                                                                |
| **Request Edit**   | User types `prompt` -> `coder.run()` blocks                 | Agent calls `RunCodeEditTool` -> Starts thread (`_run_aider_in_thread`) -> `coder.run()` inside thread.                                                                                                                                     |
| **Confirmation**   | `io.confirm_ask` blocks CLI, waits for user `y/n` input   | `AiderIOStubWithQueues.confirm_ask` puts prompt on `output_q` -> `RunCodeEditTool` receives, returns `[CODE_EDITOR_INPUT_NEEDED]` -> Agent calls `SubmitCodeEditorInputTool` -> Puts response on `input_q` -> IO stub unblocks, returns bool. |
| **Text Input**     | `io.prompt_ask` blocks CLI, waits for user text             | `AiderIOStubWithQueues.prompt_ask` puts prompt on `output_q` -> `RunCodeEditTool` receives, returns `[CODE_EDITOR_INPUT_NEEDED]` -> Agent calls `SubmitCodeEditorInputTool` -> Puts response on `input_q` -> IO stub unblocks, returns string.   |
| **Edit Completion**| `coder.run()` returns, prints output, waits for next prompt | `_run_aider_in_thread` puts `{'type': 'result', ...}` on `output_q` -> `RunCodeEditTool`/`SubmitCodeEditorInputTool` receives -> Updates persistent state -> Returns final content to agent.                                             |
| **Undo**           | User types `/undo` -> `commands.run()` -> `coder.undo`      | Agent calls `UndoLastEditTool` -> Retrieves active coder -> `coder.commands.raw_cmd_undo` -> Saves persistent state.                                                                                                                         |
| **Close**          | User types `/exit` or Ctrl-D                                | Agent calls `CloseCodeEditorTool` -> Removes in-memory state -> Clears persistent state.                                                                                                                                                 |
| **State**          | Mostly in-memory `Coder` attributes, history files on disk  | In-memory `ActiveCoderState` (live Coder, thread, queues) + Persistent `aider_state` in chat JSON (files, commits, Aider's message history).                                                                                                |

This finalized plan provides a detailed roadmap for the integration, addressing the core challenges and incorporating the refinements discussed.


# DEVELOPMENT PHASES

## Phase 1: Modify `ai_shell_agent\chat_manager.py` for Persistent Aider State

*Goal: Update the chat manager to read, write, and clear Aider-specific state within the chat JSON files.*

1.  **Backup:** Create a backup of your current `ai_shell_agent\chat_manager.py`.
2.  **Replace `_read_chat_data`:** Update this function to load an `aider_state` dictionary alongside the `messages`. Ensure `aider_done_messages` is handled correctly.

    ```python
    # Replace the existing _read_chat_data function with this:
    def _read_chat_data(file_path: str) -> Tuple[List[BaseMessage], Dict[str, Any]]:
        """Read and deserialize agent messages and the full aider_state from JSON file."""
        agent_messages = []
        aider_state = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding='utf-8') as f:  # Specify encoding
                    data = json.load(f)

                # Load agent's messages (unchanged logic)
                messages_data = data.get("messages", [])
                for msg_data in messages_data:
                    msg_type = msg_data.get("type")
                    # Ensure 'type' exists before accessing 'model_dump' or specific classes
                    if msg_type:
                        try:
                            # Dynamically find the message class based on 'type'
                            # Assumes message types match class names like HumanMessage, AIMessage, etc.
                            # Ensure necessary classes are imported at the top of the file
                            message_class_name = msg_type.capitalize() + "Message"
                            # Handle potential variations like 'ai' -> 'AIMessage'
                            if msg_type == "ai": message_class_name = "AIMessage"
                            if msg_type == "human": message_class_name = "HumanMessage"
                            if msg_type == "system": message_class_name = "SystemMessage"
                            if msg_type == "tool": message_class_name = "ToolMessage"

                            # Get the class object (assuming it's imported)
                            message_class = globals().get(message_class_name)

                            if message_class and issubclass(message_class, BaseMessage):
                                # Recreate the message object
                                # We need to handle 'additional_kwargs' which might not be a direct parameter
                                # Extract standard parameters first
                                standard_params = {k: v for k, v in msg_data.items() if k in message_class.__fields__}
                                # Put remaining items into additional_kwargs if the class supports it
                                if 'additional_kwargs' in message_class.__fields__:
                                     standard_params['additional_kwargs'] = {k: v for k, v in msg_data.items() if k not in message_class.__fields__}
                                     # Remove the explicit 'type' if it's not part of the model fields but was in the dump
                                     if 'type' in standard_params['additional_kwargs'] and 'type' not in message_class.__fields__:
                                         del standard_params['additional_kwargs']['type']


                                agent_messages.append(message_class(**standard_params))
                            else:
                                logger.warning(f"Could not find or validate message class for type: {msg_type}")
                        except Exception as e:
                             logger.error(f"Error deserializing message: {msg_data}. Error: {e}")
                    else:
                        logger.warning(f"Message data missing 'type' field: {msg_data}")


                # Load the entire aider_state dictionary
                aider_state = data.get("aider_state", {})
                # Ensure aider_done_messages is present and a list
                if "aider_done_messages" not in aider_state:
                    aider_state["aider_done_messages"] = []
                elif not isinstance(aider_state.get("aider_done_messages"), list):
                    logger.warning(f"aider_done_messages in state (file: {file_path}) is not a list, resetting.")
                    aider_state["aider_done_messages"] = []

                #logger.debug(f"Read {len(agent_messages)} agent messages from {file_path}")
                #logger.debug(f"Read aider_state keys: {list(aider_state.keys())} from {file_path}")
                #logger.debug(f"Read {len(aider_state.get('aider_done_messages', []))} aider messages from state in {file_path}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {file_path}")
                return [], {} # Return empty on decode error
            except Exception as e:
                logger.error(f"Error reading chat data from {file_path}: {e}\n{traceback.format_exc()}")
                return [], {} # Return empty on other errors

        # Ensure aider_state always contains aider_done_messages list, even if file didn't exist or was empty/corrupt
        if "aider_done_messages" not in aider_state:
            aider_state["aider_done_messages"] = []

        return agent_messages, aider_state
    ```

3.  **Replace `_write_chat_data`:** Update this function to save the `aider_state` dictionary alongside the `messages`.

    ```python
    # Replace the existing _write_chat_data function with this:
    import traceback # Add at the top if not already present

    def _write_chat_data(file_path: str, messages: List[BaseMessage], aider_state: Dict[str, Any]) -> None:
        """Write agent messages and the full aider_state to JSON file."""
        try:
            # Serialize agent messages
            messages_data = []
            for msg in messages:
                try:
                    # Use model_dump if available (Pydantic v2), otherwise dict()
                    if hasattr(msg, 'model_dump'):
                        messages_data.append(msg.model_dump())
                    else:
                        messages_data.append(msg.dict())
                except Exception as e:
                    logger.error(f"Error serializing message: {msg}. Error: {e}")
                    # Optionally skip the message or add placeholder

            # Ensure aider_done_messages exists in the state being written
            if "aider_done_messages" not in aider_state:
                aider_state["aider_done_messages"] = []
            elif not isinstance(aider_state.get("aider_done_messages"), list):
                logger.warning(f"aider_done_messages in state (file: {file_path}) is not a list before writing, resetting.")
                aider_state["aider_done_messages"] = []

            # Filter out None values from the rest of aider_state *before* saving
            # We explicitly keep aider_done_messages even if empty
            filtered_aider_state = {
                k: v for k, v in aider_state.items()
                if v is not None or k == "aider_done_messages"  # Keep aider_done_messages
            }

            data_to_write = {
                "messages": messages_data,
                "aider_state": filtered_aider_state
            }
            #logger.debug(f"Writing {len(messages_data)} agent messages to {file_path}")
            #logger.debug(f"Writing aider_state keys: {list(filtered_aider_state.keys())} to {file_path}")

            # Ensure we log how many Aider messages are being saved
            aider_messages_count = len(filtered_aider_state.get("aider_done_messages", []))
            #logger.debug(f"Writing {aider_messages_count} aider_done_messages to {file_path}")

            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding='utf-8') as f:  # Specify encoding
                json.dump(data_to_write, f, indent=4, default=str) # Add default=str for non-serializable types

        except TypeError as e:
             logger.error(f"Serialization error writing to {file_path}: {e}")
             logger.error(f"Data that failed: {data_to_write}") # Log the problematic data
        except Exception as e:
            logger.error(f"Error writing chat data to {file_path}: {e}\n{traceback.format_exc()}")
    ```

4.  **Implement `get_aider_state`:** Add this new function.

    ```python
    # Add this new function to chat_manager.py
    def get_aider_state(chat_file: str) -> Dict[str, Any]:
        """Loads the aider_state dictionary from the chat file."""
        if not chat_file or not os.path.exists(chat_file):
            logger.debug(f"Chat file {chat_file} not found or not specified for get_aider_state.")
            return {"aider_done_messages": []} # Return default empty state
        try:
            _, aider_state = _read_chat_data(chat_file)
            # Ensure the essential key exists even if read failed partially
            if "aider_done_messages" not in aider_state:
                 aider_state["aider_done_messages"] = []
            return aider_state
        except Exception as e:
            logger.error(f"Failed to get aider state from {chat_file}: {e}")
            return {"aider_done_messages": []} # Return default on error
    ```

5.  **Implement `save_aider_state`:** Add this new function.

    ```python
    # Add this new function to chat_manager.py
    def save_aider_state(chat_file: str, state_dict: Dict[str, Any]):
        """Saves the aider_state dictionary to the chat file, preserving messages."""
        if not chat_file:
             logger.error("Cannot save aider state: chat_file path is empty.")
             return
        try:
            messages, _ = _read_chat_data(chat_file)
            # Ensure messages is a list even if read failed
            if not isinstance(messages, list):
                logger.warning(f"Read invalid messages from {chat_file}, using empty list.")
                messages = []
            _write_chat_data(chat_file, messages, state_dict)
            #logger.debug(f"Aider state saved to {chat_file}")
        except Exception as e:
            logger.error(f"Failed to save aider state to {chat_file}: {e}\n{traceback.format_exc()}")

    ```

6.  **Implement `clear_aider_state`:** Add this new function.

    ```python
    # Add this new function to chat_manager.py
    def clear_aider_state(chat_file: str):
        """Clears the aider_state dictionary in the chat file, preserving messages."""
        if not chat_file:
             logger.error("Cannot clear aider state: chat_file path is empty.")
             return
        try:
            messages, _ = _read_chat_data(chat_file)
             # Ensure messages is a list even if read failed
            if not isinstance(messages, list):
                logger.warning(f"Read invalid messages from {chat_file} before clearing aider state, using empty list.")
                messages = []
            _write_chat_data(chat_file, messages, {}) # Write with an empty aider_state dict
            logger.info(f"Aider state cleared for chat file: {chat_file}")
        except Exception as e:
             logger.error(f"Failed to clear aider state for {chat_file}: {e}\n{traceback.format_exc()}")

    ```

7.  **Imports:** Ensure all necessary imports are present at the top of `chat_manager.py`:
    ```python
    import os
    import json
    import uuid
    import traceback # Add this
    from typing import Optional, Dict, Any, Tuple, List
    from langchain_core.messages import ( # Make sure these specific classes are imported
        HumanMessage,
        AIMessage,
        SystemMessage,
        ToolMessage,
        BaseMessage
    )
    # ... keep other existing imports ...
    from . import logger # Ensure logger is imported
    ```

## Phase 2: Implement `ai_shell_agent\aider_integration.py` 

*Goal: Create the core integration logic, including the IO Stub, state management, threading, and Tool classes.*

1.  **Create/Replace `ai_shell_agent\aider_integration.py`:** Create this new file or replace its existing content with the following structure. *Note: This is a large file, implement it section by section.*

    ```python
    # File: ai_shell_agent/aider_integration.py
    """
    Aider integration for AI Shell Agent.

    Handles communication with the Aider library, state management,
    and provides Langchain Tools for agent interaction.
    """

    import os
    import sys
    import queue
    import threading
    import traceback
    import json
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import Dict, Any, Optional, List, Set, Tuple, Union

    # Aider imports (ensure aider-chat is installed)
    try:
        from aider.coders import Coder
        from aider.models import Model, models # Import the 'models' module itself for sanity_check_models
        from aider.repo import GitRepo, ANY_GIT_ERROR
        from aider.io import InputOutput
        from aider.utils import format_content # If needed for formatting output, otherwise remove
        from aider.commands import Commands
    except ImportError as e:
        raise ImportError(
            "Aider package not found or incomplete. Please install it:"
            " pip install aider-chat"
        ) from e

    # Langchain import for BaseTool
    try:
        from langchain.tools import BaseTool
    except ImportError:
         # Fallback for older langchain/langchain-core versions if needed
         from langchain_core.tools import BaseTool


    # Local imports (relative)
    from . import logger
    # Import necessary functions from chat_manager and config_manager
    # These will be used by the tools and helper functions
    from .config_manager import get_current_model, get_api_key_for_model, normalize_model_name
    from .chat_manager import (
        get_current_chat,
        get_aider_state,
        save_aider_state,
        clear_aider_state
    )

    # --- Constants ---
    SIGNAL_PROMPT_NEEDED = "[CODE_EDITOR_INPUT_NEEDED]"

    # --- Custom IO Stub ---
    class AiderIOStubWithQueues(InputOutput):
        """
        An InputOutput stub for Aider that uses queues for interaction
        and captures output.
        """
        def __init__(self, *args, **kwargs):
            # Initialize with minimal necessary defaults for non-interactive use
            # Ensure 'yes' is True for automatic confirmations where possible internally
            # But external confirmations will be routed via queues.
            super().__init__(pretty=False, yes=True, fancy_input=False)
            self.input_q: Optional[queue.Queue] = None
            self.output_q: Optional[queue.Queue] = None
            self.group_preferences: Dict[int, str] = {} # group_id -> preference ('yes'/'no'/'all'/'skip')
            self.never_prompts: Set[Tuple[str, Optional[str]]] = set() # (question, subject)

            # Buffers for capturing output
            self.captured_output: List[str] = []
            self.captured_errors: List[str] = []
            self.captured_warnings: List[str] = []
            self.tool_output_lock = threading.Lock() # Protect buffer access

        def set_queues(self, input_q: queue.Queue, output_q: queue.Queue):
            """Assign the input and output queues."""
            self.input_q = input_q
            self.output_q = output_q

        def tool_output(self, *messages, log_only=False, bold=False):
            """Capture tool output."""
            msg = " ".join(map(str, messages))
            with self.tool_output_lock:
                 self.captured_output.append(msg)
            # logger.debug(f"AiderIOStub Output: {msg}") # Optional: log if needed
            # Also call parent for potential logging if needed by aider internals
            super().tool_output(*messages, log_only=True) # Ensure log_only=True for parent

        def tool_error(self, message="", strip=True):
            """Capture error messages."""
            msg = str(message).strip() if strip else str(message)
            with self.tool_output_lock:
                self.captured_errors.append(msg)
            # logger.error(f"AiderIOStub Error: {msg}") # Optional: log if needed
            super().tool_error(message, strip=strip) # Call parent for potential logging

        def tool_warning(self, message="", strip=True):
            """Capture warning messages."""
            msg = str(message).strip() if strip else str(message)
            with self.tool_output_lock:
                self.captured_warnings.append(msg)
            # logger.warning(f"AiderIOStub Warning: {msg}") # Optional: log if needed
            super().tool_warning(message, strip=strip) # Call parent for potential logging

        def get_captured_output(self, include_warnings=True, include_errors=True) -> str:
            """Returns all captured output, warnings, and errors as a single string and clears buffers."""
            with self.tool_output_lock:
                output = "\n".join(self.captured_output)
                warnings = "\n".join([f"WARNING: {w}" for w in self.captured_warnings]) if include_warnings and self.captured_warnings else ""
                errors = "\n".join([f"ERROR: {e}" for e in self.captured_errors]) if include_errors and self.captured_errors else ""

                full_output = "\n".join(filter(None, [output, warnings, errors]))

                # Clear buffers after getting output
                self.captured_output = []
                self.captured_errors = []
                self.captured_warnings = []
                return full_output.strip()

        # --- Intercept Blocking Methods ---

        def confirm_ask(self, question, default="y", subject=None, explicit_yes_required=False, group=None, allow_never=False):
            """Intercepts confirm_ask, sends prompt data via output_q, waits for input_q."""
            logger.debug(f"Intercepted confirm_ask: {question} (Subject: {subject})")
            if not self.input_q or not self.output_q:
                logger.error("Queues not set for AiderIOStubWithQueues confirm_ask.")
                raise RuntimeError("Aider IO Queues not initialized.")

            question_id = (question, subject)
            group_id = id(group) if group else None

            # 1. Check internal state for early exit (never/all/skip)
            if question_id in self.never_prompts:
                logger.debug(f"confirm_ask: Answering 'no' due to 'never_prompts' for {question_id}")
                return False
            if group_id and group_id in self.group_preferences:
                preference = self.group_preferences[group_id]
                logger.debug(f"confirm_ask: Using group preference '{preference}' for group {group_id}")
                if preference == 'skip':
                    return False
                if preference == 'all' and not explicit_yes_required:
                    return True
                # If preference is 'yes' or 'no', we still need to ask the LLM/user
                # via the queue, but maybe we can pre-fill the answer?
                # For now, let's just proceed to asking via queue.

            # 2. Send prompt details to the main thread via output_q
            prompt_data = {
                'type': 'prompt',
                'prompt_type': 'confirm',
                'question': question,
                'default': default,
                'subject': subject,
                'explicit_yes_required': explicit_yes_required,
                'allow_never': allow_never,
                'group_id': group_id # Include group_id for context
            }
            logger.debug(f"Putting prompt data on output_q: {prompt_data}")
            self.output_q.put(prompt_data)

            # 3. Block and wait for the response from the main thread via input_q
            logger.debug("Waiting for response on input_q...")
            raw_response = self.input_q.get()
            logger.debug(f"Received raw response from input_q: '{raw_response}'")

            # 4. Process the response
            response = str(raw_response).lower().strip()
            result = False # Default to no

            # Handle 'never'/'don't ask'
            if allow_never and response.startswith('d'):
                self.never_prompts.add(question_id)
                logger.debug(f"Added {question_id} to never_prompts.")
                result = False # 'Don't ask' implies 'no' for this instance
            # Handle group preferences 'all'/'skip'
            elif group_id:
                 if response.startswith('a') and not explicit_yes_required:
                     self.group_preferences[group_id] = 'all'
                     logger.debug(f"Set group {group_id} preference to 'all'.")
                     result = True
                 elif response.startswith('s'):
                     self.group_preferences[group_id] = 'skip'
                     logger.debug(f"Set group {group_id} preference to 'skip'.")
                     result = False
                 elif response.startswith('y'):
                     result = True
                 elif response.startswith('n'):
                     result = False
                 else: # Fallback to default if response is unclear for group
                     result = default.lower().startswith('y')
            # Handle regular yes/no
            elif response.startswith('y'):
                result = True
            elif response.startswith('n'):
                result = False
            else: # Use default if response is empty or unclear
                result = default.lower().startswith('y')

            # Override based on explicit_yes_required
            if explicit_yes_required and not response.startswith('y'):
                result = False

            logger.debug(f"confirm_ask returning: {result}")
            return result

        def prompt_ask(self, question, default="", subject=None):
            """Intercepts prompt_ask, sends prompt data via output_q, waits for input_q."""
            logger.debug(f"Intercepted prompt_ask: {question} (Subject: {subject})")
            if not self.input_q or not self.output_q:
                logger.error("Queues not set for AiderIOStubWithQueues prompt_ask.")
                raise RuntimeError("Aider IO Queues not initialized.")

            prompt_data = {
                'type': 'prompt',
                'prompt_type': 'input',
                'question': question,
                'default': default,
                'subject': subject
            }
            logger.debug(f"Putting prompt data on output_q: {prompt_data}")
            self.output_q.put(prompt_data)

            logger.debug("Waiting for response on input_q...")
            raw_response = self.input_q.get()
            logger.debug(f"Received raw response from input_q: '{raw_response}'")

            # Return the raw response, or default if empty
            return raw_response if raw_response else default

        # --- Override other methods to be non-interactive or no-op ---
        def get_input(self, *args, **kwargs):
            err = "AiderIOStubWithQueues.get_input() called unexpectedly in tool mode."
            logger.error(err)
            # In tool mode, input comes via RunCodeEditTool or SubmitCodeEditorInputTool
            raise NotImplementedError(err)

        def user_input(self, *args, **kwargs):
            """No-op as AI Shell Agent manages user input."""
            pass

        def ai_output(self, *args, **kwargs):
            """No-op as AI Shell Agent handles AI output."""
            pass

        def append_chat_history(self, *args, **kwargs):
            """No-op as AI Shell Agent manages chat history externally."""
            pass

    # --- Active Coder State Management ---
    @dataclass
    class ActiveCoderState:
        """Holds the live state for an active Aider session."""
        coder: Coder
        thread: Optional[threading.Thread] = None
        input_q: queue.Queue = field(default_factory=queue.Queue)
        output_q: queue.Queue = field(default_factory=queue.Queue)
        io_stub: AiderIOStubWithQueues = field(default_factory=AiderIOStubWithQueues)
        lock: threading.Lock = field(default_factory=threading.Lock) # Lock for this specific session

        def __post_init__(self):
             # Ensure the io_stub has the queues associated with this state
             self.io_stub.set_queues(self.input_q, self.output_q)

    # Global dictionary to store active coder states, keyed by chat_file path
    active_coders: Dict[str, ActiveCoderState] = {}
    # Lock to protect access to the active_coders dictionary itself
    _active_coders_dict_lock = threading.Lock()

    def get_active_coder_state(chat_file: str) -> Optional[ActiveCoderState]:
        """Safely retrieves the active coder state for a given chat file."""
        with _active_coders_dict_lock:
            return active_coders.get(chat_file)

    def create_active_coder_state(chat_file: str, coder: Coder) -> ActiveCoderState:
        """Creates and stores a new active coder state."""
        with _active_coders_dict_lock:
            if chat_file in active_coders:
                 logger.warning(f"Overwriting existing active coder state for {chat_file}")
                 # Potentially add cleanup logic here if needed before overwriting
            # Create the state, which includes creating the IO stub and queues
            state = ActiveCoderState(coder=coder)
            # Associate the coder with the io_stub explicitly if needed by Coder.create/internals
            coder.io = state.io_stub
             # Initialize Commands if not already done by Coder.create
            if not hasattr(coder, 'commands') or coder.commands is None:
                try:
                    coder.commands = Commands(io=state.io_stub, coder=coder)
                    logger.debug(f"Initialized Commands for coder {chat_file}")
                except Exception as e:
                    logger.warning(f"Could not initialize Commands for coder {chat_file}: {e}")

            active_coders[chat_file] = state
            logger.info(f"Created active Aider session for: {chat_file}")
            return state

    def remove_active_coder_state(chat_file: str):
        """Removes the active coder state and potentially cleans up resources."""
        with _active_coders_dict_lock:
            state = active_coders.pop(chat_file, None)
            if state:
                logger.info(f"Removed active Aider session for: {chat_file}")
                # Optional: Add logic to signal the thread to stop if it's running,
                # or wait for it if necessary, though letting it finish naturally might be okay.
                # if state.thread and state.thread.is_alive():
                #     logger.debug(f"Waiting for Aider thread to finish for {chat_file}...")
                #     state.thread.join(timeout=5.0) # Add a timeout
                #     if state.thread.is_alive():
                #          logger.warning(f"Aider thread for {chat_file} did not terminate cleanly.")
            else:
                 logger.debug(f"No active Aider session found to remove for: {chat_file}")

    # --- Aider Coder Recreation/Update ---
    def recreate_coder(chat_file: str, io_stub: AiderIOStubWithQueues) -> Optional[Coder]:
        """
        Recreates the Aider Coder instance from saved persistent state.
        This version is passive and primarily for use within StartCodeEditorTool.
        """
        try:
            aider_state = get_aider_state(chat_file) # Use chat_manager function
            if not aider_state or not aider_state.get("enabled", False):
                logger.debug(f"Aider state not found, empty, or not enabled for {chat_file}")
                return None

            logger.debug(f"Recreating Coder for {chat_file} with state keys: {list(aider_state.keys())}")

            # --- Model and Config Setup ---
            # Prioritize state, then agent's current, then default
            main_model_name = aider_state.get('main_model_name')
            if not main_model_name:
                main_model_name = get_current_model() # Agent's current model
                logger.warning(f"Aider main_model_name not in state, using agent's current: {main_model_name}")
            if not main_model_name: # Should not happen if agent setup is correct
                 logger.error("Cannot determine main model name for Aider Coder recreation.")
                 return None

            weak_model_name = aider_state.get('weak_model_name')
            editor_model_name = aider_state.get('editor_model_name')
            # Ensure edit format comes from state or model default
            edit_format = aider_state.get('edit_format')
            editor_edit_format = aider_state.get('editor_edit_format')

            # Ensure API keys are set in the environment (Coder relies on this)
            api_key, env_var = get_api_key_for_model(main_model_name)
            if not api_key:
                logger.error(f"API Key ({env_var}) not found for model {main_model_name}. Cannot recreate Coder.")
                return None # Cannot proceed without API key

            # Instantiate the main model object
            # This might fail if the model name is invalid or requires specific setup not handled here
            try:
                 main_model = Model(
                     main_model_name,
                     weak_model=weak_model_name,
                     editor_model=editor_model_name,
                     editor_edit_format=editor_edit_format
                 )
                 # If edit_format wasn't in state, get it from the instantiated model
                 if edit_format is None:
                      edit_format = main_model.edit_format
                      logger.debug(f"Edit format not in state, using model default: {edit_format}")

            except Exception as e:
                 logger.error(f"Failed to instantiate main_model '{main_model_name}': {e}")
                 return None

            # --- Load Aider History ---
            aider_done_messages = aider_state.get("aider_done_messages", [])
            if not isinstance(aider_done_messages, list):
                 logger.warning("aider_done_messages in state is not a list, using empty list.")
                 aider_done_messages = []
            logger.debug(f"Loading {len(aider_done_messages)} messages from aider_state for Coder history.")

            # --- Git Repo Setup ---
            repo = None
            git_root = aider_state.get("git_root")
            abs_fnames = aider_state.get("abs_fnames", [])
            abs_read_only_fnames = aider_state.get("abs_read_only_fnames", [])
            # Use all known files for repo context if available
            fnames_for_repo = abs_fnames + abs_read_only_fnames
            if git_root:
                try:
                    # Make sure GitRepo is available
                    if 'GitRepo' not in globals():
                        raise ImportError("GitRepo not imported successfully.")

                    repo_root_path = Path(git_root)
                    if not repo_root_path.is_dir():
                        logger.warning(f"Git root directory specified in state does not exist or is not a directory: {git_root}. Attempting without git.")
                        git_root = None # Invalidate git_root if path is gone/invalid
                    else:
                        # Ensure fnames_for_repo contains absolute paths relative to the *correct* root
                        # It assumes saved paths were absolute or relative to the original root.
                        # Best practice is to save absolute paths.
                        abs_fnames_for_repo = [str(Path(p).resolve()) for p in fnames_for_repo]

                        repo = GitRepo(io=io_stub, fnames=abs_fnames_for_repo, git_dname=str(repo_root_path))
                        # Verify the repo root matches the state to catch potential mismatches
                        if str(Path(repo.root).resolve()) != str(repo_root_path.resolve()):
                             logger.warning(f"Detected Git root '{repo.root}' differs from state '{git_root}'. Using detected root.")
                             # Update git_root for Coder instantiation if needed, or decide policy
                             # For now, we'll trust the detected repo object's root
                             git_root = repo.root # Use the root from the created repo object

                except ImportError:
                    logger.warning("GitPython not installed or GitRepo not available, git features disabled.")
                    git_root = None # Ensure git_root is None if git can't be used
                    repo = None
                except ANY_GIT_ERROR as e:
                    logger.error(f"Error initializing GitRepo at {git_root}: {e}")
                    git_root = None # Invalidate on error
                    repo = None
                except Exception as e: # Catch other potential errors
                    logger.error(f"Unexpected error initializing GitRepo at {git_root}: {e}")
                    git_root = None
                    repo = None
            else:
                logger.debug("No git_root found in state, proceeding without GitRepo.")


            # --- Prepare Explicit Config for Coder.create ---
            coder_kwargs = dict(
                main_model=main_model,
                edit_format=edit_format,
                io=io_stub, # Use the provided stub
                repo=repo,
                fnames=abs_fnames, # Use absolute paths from state
                read_only_fnames=abs_read_only_fnames, # Use absolute paths from state
                done_messages=aider_done_messages, # Use history from state
                cur_messages=[], # Always start cur_messages fresh for a tool run

                # Pass other relevant settings from state or defaults
                auto_commits=aider_state.get("auto_commits", True),
                dirty_commits=aider_state.get("dirty_commits", True),
                use_git=bool(repo),
                # Set other flags based on state or agent defaults as needed
                # Example: map_tokens, verbose, stream etc. might come from aider_state
                # For now, using sensible defaults for tool usage:
                map_tokens=aider_state.get("map_tokens", 0), # Default to 0 if not in state
                verbose=False, # Keep tool runs non-verbose unless specified
                stream=False, # Tool runs are typically non-streaming
                suggest_shell_commands=False, # Safer default for tool usage
            )

            # --- Instantiate Coder ---
            coder = Coder.create(**coder_kwargs)
            coder.root = git_root or os.getcwd() # Set root explicitly

             # Initialize Commands instance for tools that might need it (like /diff, /undo)
            if not hasattr(coder, 'commands') or coder.commands is None:
                try:
                    # Ensure Commands class is available
                    if 'Commands' not in globals():
                         raise ImportError("Commands class not imported successfully.")
                    coder.commands = Commands(io=io_stub, coder=coder)
                    logger.debug(f"Initialized Commands for coder {chat_file} during recreation")
                except ImportError:
                    logger.warning("Could not import Commands module from Aider.")
                    # Proceed without commands if not critical for all tools
                except Exception as e:
                     logger.warning(f"Could not initialize Commands for coder {chat_file}: {e}")


            logger.info(f"Coder successfully recreated for {chat_file}")
            return coder

        except Exception as e:
            logger.error(f"Failed to recreate Coder for {chat_file}: {e}")
            logger.error(traceback.format_exc())
            return None

    def update_aider_state_from_coder(chat_file: str, coder: Coder) -> None:
        """Update the persistent Aider state from a Coder instance."""
        try:
            # Get the current state to update it, preserving unrelated fields
            aider_state = get_aider_state(chat_file)
            if not aider_state:
                aider_state = {} # Initialize if it doesn't exist

            # Ensure 'enabled' flag is set (it should be if we have a coder)
            aider_state["enabled"] = True

            # --- Update fields based on the coder ---
            aider_state["main_model_name"] = coder.main_model.name
            aider_state["edit_format"] = coder.edit_format
            aider_state["weak_model_name"] = getattr(coder.main_model.weak_model, 'name', None)
            aider_state["editor_model_name"] = getattr(coder.main_model.editor_model, 'name', None)
            aider_state["editor_edit_format"] = getattr(coder.main_model, 'editor_edit_format', None)

            aider_state["abs_fnames"] = sorted(list(coder.abs_fnames))
            aider_state["abs_read_only_fnames"] = sorted(list(getattr(coder, "abs_read_only_fnames", [])))

            # Update git info if available
            if hasattr(coder, "repo") and coder.repo:
                aider_state["git_root"] = coder.repo.root
                # Ensure commit hashes are strings and sorted for consistency
                aider_state["aider_commit_hashes"] = sorted(list(map(str, coder.aider_commit_hashes)))
            else:
                # Clear git info if repo is no longer present
                aider_state.pop("git_root", None)
                aider_state.pop("aider_commit_hashes", None)

            # *** Crucially, save Aider's internal conversation history ***
            # Filter out any non-serializable content if necessary, though usually dicts are fine
            try:
                 aider_state["aider_done_messages"] = coder.done_messages
                 logger.debug(f"Saving {len(coder.done_messages)} messages to aider_done_messages state for {chat_file}.")
            except Exception as e:
                 logger.error(f"Failed to serialize aider_done_messages for {chat_file}: {e}")
                 # Decide on fallback: save empty list or raise error? Saving empty might be safer.
                 aider_state["aider_done_messages"] = []


            # Update other relevant config flags if they were changed
            aider_state["auto_commits"] = getattr(coder, "auto_commits", True)
            aider_state["dirty_commits"] = getattr(coder, "dirty_commits", True)
            # Add other flags as needed, e.g., map_tokens

            # --- Save the updated state ---
            save_aider_state(chat_file, aider_state)
            logger.debug(f"Aider state updated from coder for {chat_file}")

        except Exception as e:
            logger.error(f"Failed to update persistent Aider state for {chat_file}: {e}")
            logger.error(traceback.format_exc())

    # --- Threading Logic ---
    def _run_aider_in_thread(coder: Coder, instruction: str, output_q: queue.Queue):
        """Target function for the Aider worker thread."""
        thread_name = threading.current_thread().name
        logger.info(f"Aider worker thread '{thread_name}' started for instruction: {instruction[:50]}...")
        try:
            # Ensure the coder uses the stub with the correct queues assigned
            if not isinstance(coder.io, AiderIOStubWithQueues) or coder.io.output_q != output_q:
                 logger.error(f"Thread {thread_name}: Coder IO setup is incorrect!")
                 raise RuntimeError("Coder IO setup incorrect in thread.")

            # Clear any previous output in the stub before running
            coder.io.get_captured_output()

            # The main blocking call
            coder.run(with_message=instruction)

            # Get accumulated output from the run
            final_output = coder.io.get_captured_output()
            logger.info(f"Thread {thread_name}: coder.run completed successfully.")
            output_q.put({'type': 'result', 'content': final_output})

        except Exception as e:
            logger.error(f"Thread {thread_name}: Exception during coder.run: {e}")
            logger.error(traceback.format_exc())
            # Capture any output accumulated before the error
            error_output = coder.io.get_captured_output()
            error_message = f"Error during code edit: {e}\nOutput before error:\n{error_output}"
            output_q.put({'type': 'error', 'message': error_message, 'traceback': traceback.format_exc()})
        finally:
             logger.info(f"Aider worker thread '{thread_name}' finished.")


    # --- Tool Definitions ---

    # Tool: Start/Resume Session
    class StartCodeEditorTool(BaseTool):
        name = "start_code_editor"
        description = (
            "Initializes or resumes the code editing session for the current chat."
            " Must be called before adding files or requesting edits in a new agent session,"
            " or to start a new coding task."
        )

        def _run(self, args: str = "") -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found. Please start or load a chat first."

            with _active_coders_dict_lock: # Lock the dictionary access
                if chat_file in active_coders:
                    logger.info(f"Aider session already active for {chat_file}.")
                    # Optionally, reset parts of the existing state if needed, or just confirm.
                    # For now, just confirm it's active.
                    state = active_coders[chat_file]
                    # Ensure the io_stub's buffers are cleared if resuming an existing active session
                    state.io_stub.get_captured_output()
                    return f"Code editor session is already active (Model: {state.coder.main_model.name}). Ready for files and edits."

                # --- Session Not Active in Memory: Try to Resume from Persistent State ---
                logger.debug(f"No active session for {chat_file}, attempting to recreate from state...")
                # Create a temporary IO stub just for recreation attempt
                temp_io_stub = AiderIOStubWithQueues()
                coder = recreate_coder(chat_file, temp_io_stub)

                if coder:
                    # --- Resumed Successfully ---
                    logger.info(f"Resuming Aider session for {chat_file} from persistent state.")
                    # Create the *actual* active state using the recreated coder
                    state = create_active_coder_state(chat_file, coder)
                    # No need to call update_aider_state_from_coder here, state is already loaded
                    # Capture any output from recreation (e.g., warnings)
                    recreation_output = temp_io_stub.get_captured_output()
                    if recreation_output:
                         logger.warning(f"Output during coder recreation for {chat_file}: {recreation_output}")
                    return f"Code editor session resumed (Model: {coder.main_model.name}). Ready for files and edits."
                else:
                    # --- Failed to Resume: Start Fresh ---
                    logger.info(f"No valid persistent state found for {chat_file}, starting fresh.")
                    clear_aider_state(chat_file) # Clear any potentially corrupt state

                    # Determine initial model based on agent's current config
                    current_main_model_name = get_current_model()
                    if not current_main_model_name:
                        return "Error: Could not determine the current model for the agent."

                    try:
                        # Instantiate a temporary model to get defaults
                        temp_model = Model(current_main_model_name)
                        default_edit_format = temp_model.edit_format
                        default_weak_model_name = getattr(temp_model.weak_model, 'name', None)
                        default_editor_model_name = getattr(temp_model.editor_model, 'name', None)
                        default_editor_edit_format = getattr(temp_model, 'editor_edit_format', None)
                    except Exception as e:
                        logger.warning(f"Could not get defaults for {current_main_model_name}: {e}. Using basic defaults.")
                        default_edit_format = 'whole' # A safe fallback
                        default_weak_model_name = None
                        default_editor_model_name = None
                        default_editor_edit_format = None

                    # Define initial persistent state
                    initial_state = {
                        "enabled": True,
                        "main_model_name": current_main_model_name,
                        "edit_format": default_edit_format,
                        "weak_model_name": default_weak_model_name,
                        "editor_model_name": default_editor_model_name,
                        "editor_edit_format": default_editor_edit_format,
                        "abs_fnames": [],
                        "abs_read_only_fnames": [],
                        "aider_done_messages": [],
                        "aider_commit_hashes": [],
                        "git_root": None, # Will be determined later if git is used
                        "auto_commits": True, # Default setting
                        "dirty_commits": True, # Explicitly match Coder default
                         # Add other necessary default fields if coder expects them
                    }
                    save_aider_state(chat_file, initial_state)

                    # Now create the Coder instance for the active session
                    try:
                         # We need an IO stub first for Coder.create
                         fresh_io_stub = AiderIOStubWithQueues()
                         fresh_coder = Coder.create(
                              main_model=Model(current_main_model_name), # Re-instantiate model
                              edit_format=default_edit_format,
                              io=fresh_io_stub,
                              # Pass other necessary defaults
                              fnames=[],
                              read_only_fnames=[],
                              done_messages=[],
                              cur_messages=[],
                              auto_commits=True,
                              dirty_commits=True,
                              # Attempt git repo detection for the new session
                              use_git=True # Assume try git by default
                         )
                         # If repo was found, update persistent state
                         if fresh_coder.repo:
                              initial_state["git_root"] = fresh_coder.repo.root
                              save_aider_state(chat_file, initial_state) # Save updated git_root

                         # Create the active state
                         create_active_coder_state(chat_file, fresh_coder)
                         return f"New code editor session started (Model: {current_main_model_name}, Format: {default_edit_format}). Ready for files and edits."
                    except Exception as e:
                         logger.error(f"Failed to create new Coder instance for {chat_file}: {e}")
                         logger.error(traceback.format_exc())
                         # Clean up potentially inconsistent state
                         clear_aider_state(chat_file)
                         remove_active_coder_state(chat_file)
                         return f"Error: Failed to initialize code editor. {e}"

        async def _arun(self, args: str = "") -> str:
            # Consider running sync in threadpool if needed, but might be okay sync
            return self._run(args)

    # Tool: Add File
    class AddFileTool(BaseTool):
        name = "add_code_file"
        description = (
             "Adds a file to the code editing session context. Argument must be the relative or"
              " absolute file path. Use list_code_files to see current files."
              " Use start_code_editor first if the session is not active."
        )

        def _run(self, file_path: str) -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found."

            state = get_active_coder_state(chat_file)
            if not state:
                return "Error: Code editor session not active. Use start_code_editor first."

            # Acquire lock for the specific session to modify coder state
            with state.lock:
                coder = state.coder
                io_stub = state.io_stub
                original_fnames = set(coder.abs_fnames)

                try:
                    # Resolve the path - Coder methods often expect relative paths
                    # but let's resolve first for checks, then get relative.
                    abs_path_to_add = Path(file_path).resolve()

                    # Basic existence check
                    if not abs_path_to_add.exists():
                         return f"Error: File '{file_path}' (resolved to '{abs_path_to_add}') does not exist."
                    if not abs_path_to_add.is_file():
                         return f"Error: Path '{file_path}' (resolved to '{abs_path_to_add}') is not a file."


                    # Use coder's method to get relative path based on its root
                    try:
                        rel_path_to_add = coder.get_rel_fname(str(abs_path_to_add))
                    except ValueError as e:
                         # This can happen if the file is outside the coder's root (e.g., different drive)
                         return f"Error: Cannot add file '{file_path}'. It seems to be outside the project root defined by the coder ({coder.root}): {e}"


                    # Add the file using the coder's method
                    coder.add_rel_fname(rel_path_to_add)

                    # Check if the file was actually added (might have been there already)
                    if abs_path_to_add in original_fnames:
                         msg = f"File {rel_path_to_add} was already in the context."
                    else:
                         msg = f"Successfully added {rel_path_to_add} to the context."
                         # Update persistent state only if a file was actually added
                         update_aider_state_from_coder(chat_file, coder)


                    # Return captured output plus success/already-added message
                    captured = io_stub.get_captured_output()
                    return f"{msg}\n{captured}".strip()

                except Exception as e:
                    logger.error(f"Error in AddFileTool for {file_path}: {e}")
                    logger.error(traceback.format_exc())
                    # Return captured output plus error message
                    captured = io_stub.get_captured_output()
                    return f"Error adding file {file_path}: {e}\n{captured}".strip()

        async def _arun(self, file_path: str) -> str:
            return self._run(file_path)

    # Tool: Drop File
    class DropFileTool(BaseTool):
        name = "drop_code_file"
        description = (
            "Removes a file from the code editing session context. Argument must be the"
            " relative or absolute file path that was previously added."
            " Use list_code_files to see current files."

        )

        def _run(self, file_path: str) -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found."

            state = get_active_coder_state(chat_file)
            if not state:
                return "Error: Code editor session not active." # No need for start_code_editor hint here

            # Acquire lock for the specific session
            with state.lock:
                 coder = state.coder
                 io_stub = state.io_stub

                 try:
                    # Resolve path and get relative path using coder's root
                    abs_path_to_drop = Path(file_path).resolve()
                    try:
                         rel_path_to_drop = coder.get_rel_fname(str(abs_path_to_drop))
                    except ValueError:
                        # If it can't be made relative, it's likely not in the context anyway
                         return f"File '{file_path}' could not be found in the context (possibly outside project root)."

                    # Check if it's actually in the context before trying to drop
                    if str(abs_path_to_drop) not in coder.abs_fnames:
                         # Check read-only files as well
                         if str(abs_path_to_drop) in getattr(coder, 'abs_read_only_fnames', set()):
                              # If it's read-only, drop it from there
                              if hasattr(coder, 'abs_read_only_fnames'):
                                   coder.abs_read_only_fnames.discard(str(abs_path_to_drop))
                                   update_aider_state_from_coder(chat_file, coder)
                                   msg = f"Successfully dropped read-only file {rel_path_to_drop}."
                                   captured = io_stub.get_captured_output()
                                   return f"{msg}\n{captured}".strip()
                              else:
                                   return f"File {rel_path_to_drop} is read-only and could not be dropped (internal state issue)."
                         else:
                              return f"File {rel_path_to_drop} not found in the editable context."

                    # Attempt to drop using Coder's method
                    success = coder.drop_rel_fname(rel_path_to_drop)

                    if success:
                        # Update persistent state
                        update_aider_state_from_coder(chat_file, coder)
                        msg = f"Successfully dropped {rel_path_to_drop}."
                    else:
                        # This case might be redundant due to the check above, but keep for safety
                        msg = f"File {rel_path_to_drop} not found in context."

                    # Return captured output plus status message
                    captured = io_stub.get_captured_output()
                    return f"{msg}\n{captured}".strip()

                 except Exception as e:
                     logger.error(f"Error in DropFileTool for {file_path}: {e}")
                     logger.error(traceback.format_exc())
                     captured = io_stub.get_captured_output()
                     return f"Error dropping file {file_path}: {e}\n{captured}".strip()

        async def _arun(self, file_path: str) -> str:
            return self._run(file_path)

    # Tool: List Files
    class ListFilesInEditorTool(BaseTool):
        name = "list_code_files"
        description = "Lists all files currently in the code editing session context (both editable and read-only)."

        def _run(self, args: str = "") -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found."

            # Get state from persistent storage, as this doesn't require a live coder
            try:
                aider_state = get_aider_state(chat_file)
                if not aider_state or not aider_state.get("enabled", False):
                     return "Code editor session not active or not initialized."

                editable_files = aider_state.get("abs_fnames", [])
                read_only_files = aider_state.get("abs_read_only_fnames", [])

                if not editable_files and not read_only_files:
                    return "No files are currently added to the code editing session."

                # Get the common root to show relative paths if possible
                # Prefer git_root if available, otherwise try to determine from files
                root = aider_state.get("git_root")
                if not root:
                     all_files = editable_files + read_only_files
                     if all_files:
                          # Find common directory - this might be slow for many files
                          try:
                               root = os.path.commonpath(all_files)
                               # Ensure it's a directory
                               if not os.path.isdir(root):
                                    root = os.path.dirname(root)
                          except ValueError: # Happens if files are on different drives
                               root = None
                     if not root: root = os.getcwd() # Fallback

                output = ""
                if editable_files:
                    output += "Editable files in context:\n"
                    files_list = []
                    for f in sorted(editable_files):
                        try:
                             rel_path = os.path.relpath(f, root)
                             files_list.append(f"- {rel_path}")
                        except ValueError:
                             files_list.append(f"- {f} (abs path)") # Show absolute if relpath fails
                    output += "\n".join(files_list) + "\n"

                if read_only_files:
                    output += "\nRead-only files in context:\n"
                    files_list = []
                    for f in sorted(read_only_files):
                        try:
                             rel_path = os.path.relpath(f, root)
                             files_list.append(f"- {rel_path}")
                        except ValueError:
                             files_list.append(f"- {f} (abs path)")
                    output += "\n".join(files_list) + "\n"

                return output.strip()

            except Exception as e:
                logger.error(f"Error in ListFilesInEditorTool: {e}")
                logger.error(traceback.format_exc())
                return f"Error listing files: {e}"

        async def _arun(self, args: str = "") -> str:
            return self._run(args)

    # Tool: Run Edit
    class RunCodeEditTool(BaseTool):
        name = "edit_code"
        description = (
             "Requests code changes based on the provided natural language instruction."
             " Edits files currently in the session context."
             " Requires an active code editor session."
             " May return '[CODE_EDITOR_INPUT_NEEDED]' if Aider requires confirmation (e.g., to create a file or overwrite dirty changes)."
        )
        # Potentially add args_schema here if needed

        def _run(self, instruction: str) -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found."

            state = get_active_coder_state(chat_file)
            if not state:
                return "Error: Code editor session not active. Use start_code_editor first."

            # Acquire lock for the specific session
            with state.lock:
                 # Check if another edit is already in progress
                 if state.thread and state.thread.is_alive():
                      # This shouldn't happen if agent logic is correct, but good safeguard
                      return "Error: Another code edit is already in progress for this session."

                 # Clear queues before starting
                 while not state.input_q.empty(): state.input_q.get()
                 while not state.output_q.empty(): state.output_q.get()

                 # Start the Aider process in a new thread
                 state.thread = threading.Thread(
                      target=_run_aider_in_thread,
                      args=(state.coder, instruction, state.output_q),
                      name=f"AiderWorker-{os.path.basename(chat_file)}" # Add chat file basename to thread name
                 )
                 state.thread.daemon = True # Allow main program to exit even if thread is stuck
                 state.thread.start()
                 logger.info(f"Started Aider worker thread for {chat_file}")

            # Release lock before waiting on queue
            logger.debug(f"Main thread waiting for first message from output_q for {chat_file}...")
            try:
                 # Wait for the first message from the worker thread (prompt needed or result/error)
                 # Add a timeout? Long edits might take time, but blocking indefinitely is risky.
                 # Let's start without a timeout and see.
                 message = state.output_q.get() # Blocks here
                 logger.debug(f"Main thread received message from output_q: {message.get('type')}")
            except queue.Empty:
                 # This should ideally not happen with get() unless a timeout is added
                 logger.error(f"Timeout or queue error waiting for Aider response ({chat_file}).")
                 # Attempt to cleanup state?
                 remove_active_coder_state(chat_file) # Assume the session is borked
                 return "Error: Timed out waiting for Aider response."
            except Exception as e:
                 logger.error(f"Exception waiting on output_q for {chat_file}: {e}")
                 remove_active_coder_state(chat_file)
                 return f"Error: Exception while waiting for Aider: {e}"

            # Process the received message
            message_type = message.get('type')

            if message_type == 'prompt':
                prompt_data = message # It's the whole dict
                # Format the prompt information for the agent
                prompt_type = prompt_data.get('prompt_type', 'unknown')
                question = prompt_data.get('question', 'Input needed')
                subject = prompt_data.get('subject')
                default = prompt_data.get('default')
                allow_never = prompt_data.get('allow_never')

                response_guidance = f"Please respond to the following prompt using the 'submit_code_editor_input' tool. Prompt: '{question}'"
                if subject: response_guidance += f" (Regarding: {subject[:100]}{'...' if len(subject)>100 else ''})"
                if default: response_guidance += f" [Default: {default}]"

                if prompt_type == 'confirm':
                     options = "(yes/no"
                     if prompt_data.get('group_id'): options += "/all/skip"
                     if allow_never: options += "/don't ask"
                     options += ")"
                     response_guidance += f" Options: {options}"
                # Add more guidance if needed for 'input' type

                # Return the signal and the formatted guidance
                return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"

            elif message_type == 'result':
                 # Edit completed successfully in the thread
                 logger.info(f"Aider edit completed successfully for {chat_file}.")
                 with state.lock: # Re-acquire lock to update state
                      # Update persistent state from the now-finished coder
                      update_aider_state_from_coder(chat_file, state.coder)
                      state.thread = None # Mark thread as finished
                 return f"Edit completed. Output:\n{message.get('content', 'No output captured.')}"

            elif message_type == 'error':
                 # Error occurred in the thread
                 logger.error(f"Aider edit failed for {chat_file}.")
                 # Error message already contains captured output + exception
                 error_content = message.get('message', 'Unknown error')
                 # Optionally log full traceback here if needed
                 # logger.error(f"Full traceback:\n{message.get('traceback', 'No traceback available.')}")
                 # Session might be corrupted, clean up active state
                 remove_active_coder_state(chat_file)
                 return f"Error during edit:\n{error_content}"
            else:
                 # Should not happen
                 logger.error(f"Received unknown message type from Aider thread: {message_type}")
                 remove_active_coder_state(chat_file)
                 return f"Error: Unknown response type '{message_type}' from Aider process."

        async def _arun(self, instruction: str) -> str:
             # Consider running sync in threadpool if needed
            return self._run(instruction)

    # Tool: Submit Input
    class SubmitCodeEditorInputTool(BaseTool):
        name = "submit_code_editor_input"
        description = (
             "Provides the required input (e.g., 'yes', 'no', 'all', 'skip', 'don't ask', or text) "
             "when the code editor signals '[CODE_EDITOR_INPUT_NEEDED]'."
             " Sends the input back to the ongoing Aider process."
             " Will return the next prompt, the final result, or an error."
        )
        args_schema: Optional[Type[BaseModel]] = None # Define if using Pydantic

        # Simplified: takes a single string argument
        def _run(self, user_response: str) -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found."

            state = get_active_coder_state(chat_file)
            if not state:
                # This indicates an agent logic error - trying to submit input when no session active
                return "Error: No active code editor session to submit input to."

            # Acquire lock for the specific session
            with state.lock:
                 # Check if the thread is actually running and waiting
                 if not state.thread or not state.thread.is_alive():
                      # Could happen if thread finished unexpectedly or was closed
                      # Clean up just in case
                      remove_active_coder_state(chat_file)
                      return "Error: The code editing process is not waiting for input."

                 # Send the raw user response to the waiting Aider thread
                 logger.debug(f"Putting user response on input_q: '{user_response}' for {chat_file}")
                 state.input_q.put(user_response)

            # Release lock before waiting on queue
            logger.debug(f"Main thread waiting for *next* message from output_q for {chat_file}...")
            try:
                # Wait for the Aider thread's *next* action (could be another prompt, result, or error)
                 message = state.output_q.get() # Blocks here
                 logger.debug(f"Main thread received message from output_q: {message.get('type')}")
            except queue.Empty:
                 logger.error(f"Timeout or queue error waiting for Aider response ({chat_file}).")
                 remove_active_coder_state(chat_file)
                 return "Error: Timed out waiting for Aider response after submitting input."
            except Exception as e:
                 logger.error(f"Exception waiting on output_q for {chat_file}: {e}")
                 remove_active_coder_state(chat_file)
                 return f"Error: Exception while waiting for Aider after submitting input: {e}"

            # Process the message (identical logic to RunCodeEditTool's processing part)
            message_type = message.get('type')

            if message_type == 'prompt':
                prompt_data = message
                prompt_type = prompt_data.get('prompt_type', 'unknown')
                question = prompt_data.get('question', 'Input needed')
                subject = prompt_data.get('subject')
                default = prompt_data.get('default')
                allow_never = prompt_data.get('allow_never')

                response_guidance = f"Aider requires further input. Please respond using 'submit_code_editor_input'. Prompt: '{question}'"
                if subject: response_guidance += f" (Regarding: {subject[:100]}{'...' if len(subject)>100 else ''})"
                if default: response_guidance += f" [Default: {default}]"
                if prompt_type == 'confirm':
                     options = "(yes/no"
                     if prompt_data.get('group_id'): options += "/all/skip"
                     if allow_never: options += "/don't ask"
                     options += ")"
                     response_guidance += f" Options: {options}"

                return f"{SIGNAL_PROMPT_NEEDED} {response_guidance}"

            elif message_type == 'result':
                 logger.info(f"Aider edit completed successfully for {chat_file} after input.")
                 with state.lock: # Re-acquire lock
                      update_aider_state_from_coder(chat_file, state.coder)
                      state.thread = None
                 return f"Edit completed. Output:\n{message.get('content', 'No output captured.')}"

            elif message_type == 'error':
                 logger.error(f"Aider edit failed for {chat_file} after input.")
                 error_content = message.get('message', 'Unknown error')
                 remove_active_coder_state(chat_file)
                 return f"Error during edit:\n{error_content}"
            else:
                 logger.error(f"Received unknown message type from Aider thread: {message_type}")
                 remove_active_coder_state(chat_file)
                 return f"Error: Unknown response type '{message_type}' from Aider process."

        async def _arun(self, user_response: str) -> str:
            # Consider running sync in threadpool if needed
            return self._run(user_response)

    # Tool: View Diff
    class ViewDiffTool(BaseTool):
        name = "view_code_diff"
        description = "Shows the git diff of changes made since the start of the last edit request or /commit."

        def _run(self, args: str = "") -> str:
            chat_file = get_current_chat()
            if not chat_file: return "Error: No active chat session found."

            state = get_active_coder_state(chat_file)
            if not state: return "Error: Code editor session not active."

            with state.lock: # Lock access to the coder/commands object
                 coder = state.coder
                 io_stub = state.io_stub
                 if not coder.repo:
                     return "Error: Git repository not available. Cannot show diff."

                 # Ensure commands object is available
                 if not hasattr(coder, 'commands') or coder.commands is None:
                      return "Error: Aider Commands object not initialized, cannot run diff."

                 try:
                      # Use the raw command to bypass interactive prompts and capture output via stub
                      # Ensure output buffers are clear before calling
                      io_stub.get_captured_output()
                      coder.commands.raw_cmd_diff("") # Pass empty args for default diff behavior
                      captured = io_stub.get_captured_output()
                      return f"Diff output:\n{captured}" if captured else "No changes detected since last commit/message."
                 except ANY_GIT_ERROR as e:
                      logger.error(f"Git error during diff: {e}")
                      logger.error(traceback.format_exc())
                      captured = io_stub.get_captured_output() # Get any partial error output
                      return f"Error viewing diff: {e}\n{captured}".strip()
                 except Exception as e:
                      logger.error(f"Error in ViewDiffTool: {e}")
                      logger.error(traceback.format_exc())
                      captured = io_stub.get_captured_output()
                      return f"Error viewing diff: {e}\n{captured}".strip()

        async def _arun(self, args: str = "") -> str:
            return self._run(args)

    # Tool: Undo Last Edit
    class UndoLastEditTool(BaseTool):
        name = "undo_last_code_edit"
        description = "Undoes the last code edit commit made by Aider in the current session, if possible."

        def _run(self, args: str = "") -> str:
            chat_file = get_current_chat()
            if not chat_file: return "Error: No active chat session found."

            state = get_active_coder_state(chat_file)
            if not state: return "Error: Code editor session not active."

            with state.lock: # Lock access
                 coder = state.coder
                 io_stub = state.io_stub
                 if not coder.repo:
                     return "Error: Cannot undo. Git repository not found or not configured."

                 if not hasattr(coder, 'commands') or coder.commands is None:
                      return "Error: Aider Commands object not initialized, cannot run undo."

                 try:
                      # Ensure output buffers are clear before calling
                      io_stub.get_captured_output()
                      # Use the raw command to bypass potential interactive prompts
                      coder.commands.raw_cmd_undo(None) # Pass None for default undo behavior

                      # IMPORTANT: Update persistent state after undo changes commit hashes
                      update_aider_state_from_coder(chat_file, coder)

                      captured = io_stub.get_captured_output()
                      return f"Undo attempt finished. Output:\n{captured}".strip()

                 except ANY_GIT_ERROR as e:
                      logger.error(f"Git error during undo: {e}")
                      logger.error(traceback.format_exc())
                      # State might be inconsistent, but try to capture output
                      captured = io_stub.get_captured_output()
                      return f"Error during undo: {e}\n{captured}".strip()
                 except Exception as e:
                      logger.error(f"Unexpected error during undo: {e}")
                      logger.error(traceback.format_exc())
                      captured = io_stub.get_captured_output()
                      return f"Error during undo: {e}\n{captured}".strip()

        async def _arun(self, args: str = "") -> str:
            return self._run(args)

    # Tool: Close Session
    class CloseCodeEditorTool(BaseTool):
        name = "close_code_editor"
        description = (
            "Closes the code editor session for the current chat, clearing its specific context"
            " like added files and edit history. Call this when code editing tasks are complete."
        )

        def _run(self, args: str = "") -> str:
            chat_file = get_current_chat()
            if not chat_file:
                return "Error: No active chat session found to close the editor for."

            try:
                 # Remove the in-memory state first
                 remove_active_coder_state(chat_file)
                 # Then clear the persistent state
                 clear_aider_state(chat_file)
                 return "Code editor session closed and context cleared successfully."
            except Exception as e:
                 logger.error(f"Error closing code editor for {chat_file}: {e}")
                 logger.error(traceback.format_exc())
                 return f"Error closing code editor: {e}"

        async def _arun(self, args: str = "") -> str:
            return self._run(args)


    # --- Tool Instances ---
    start_code_editor_tool = StartCodeEditorTool()
    add_code_file_tool = AddFileTool()
    drop_code_file_tool = DropFileTool()
    list_code_files_tool = ListFilesInEditorTool()
    edit_code_tool = RunCodeEditTool()
    submit_code_editor_input_tool = SubmitCodeEditorInputTool()
    view_diff_tool = ViewDiffTool()
    undo_last_edit_tool = UndoLastEditTool()
    close_code_editor_tool = CloseCodeEditorTool()

    # List of tools provided by this module
    aider_tools = [
        start_code_editor_tool,
        add_code_file_tool,
        drop_code_file_tool,
        list_code_files_tool,
        edit_code_tool,
        submit_code_editor_input_tool,
        view_diff_tool,
        undo_last_edit_tool,
        close_code_editor_tool,
    ]
    ```

## Phase 3: Modify `ai_shell_agent\tools.py` 

*Goal: Integrate the Aider tools into the main tool list.*

1.  **Backup:** Create a backup of your current `ai_shell_agent\tools.py`.
2.  **Modify Imports:** Import the `aider_tools` list from the new integration module.
3.  **Update `tools` List:** Replace the individual Aider tool imports with the imported list.

    ```python
    # File: ai_shell_agent/tools.py
    import subprocess
    from langchain.tools import BaseTool, tool
    from langchain_core.utils.function_calling import convert_to_openai_function
    from langchain_experimental.tools.python.tool import PythonREPLTool
    from prompt_toolkit import prompt
    from . import logger

    # --- REMOVE individual aider tool imports ---
    # from .aider_integration import (
    #     start_code_editor_tool,
    #     add_code_file_tool,
    #     ... etc ...
    # )

    # --- ADD import for the list of aider tools ---
    from .aider_integration import aider_tools

    # Keep ConsoleTool_HITL, ConsoleTool_Direct, run_python_code, python_repl_tool etc.
    # ... (existing tool definitions like ConsoleTool_Direct remain) ...
    class ConsoleTool_HITL(BaseTool):
        # ... existing implementation ...
        pass

    class ConsoleTool_Direct(BaseTool):
        # ... existing implementation ...
        pass

    python_repl_tool = PythonREPLTool()
    interactive_windows_shell_tool = ConsoleTool_HITL()
    direct_windows_shell_tool = ConsoleTool_Direct()

    def run_python_code(code: str) -> str:
         """
         Wrapper around PythonREPLTool for convenience.

         Args:
             code (str): The Python code to execute.

         Returns:
             str: The output from executing the Python code.
         """
         return python_repl_tool.invoke({"command": code})


    # List of all available tools
    # Combine existing tools with the new aider_tools list
    tools = [
        interactive_windows_shell_tool, # Example existing tool
        python_repl_tool,           # Example existing tool
        # Add other non-Aider tools here if you have them
    ] + aider_tools # Add all the Aider tools

    # Update the tools_functions list (no change in logic needed here)
    tools_functions = [convert_to_openai_function(t) for t in tools]

    ```

## Phase 4: Modify `ai_shell_agent/ai.py` (Guidance) 

*Goal: Adapt the main agent logic to manage tool availability and handle the interactive Aider flow.*

*   **Location:** Find the part of your `main()` function (or wherever the agent's main execution loop resides) where the list of available tools is determined and passed to the LLM.
*   **Conditional Tool Availability:**
    *   Before invoking the LLM for a turn, get the current chat file: `chat_file = get_current_chat()`
    *   Check if an Aider session is active:
        ```python
        from .aider_integration import get_active_coder_state, aider_tools, start_code_editor_tool # Import necessary items
        from .tools import tools as all_base_tools # Rename original list or filter aider tools out

        active_state = get_active_coder_state(chat_file)

        if active_state:
            # Session is active, provide all Aider tools + base tools
            available_tools = all_base_tools # Assumes all_base_tools already includes aider_tools
            available_tool_functions = [convert_to_openai_function(t) for t in available_tools]
            # Or construct dynamically:
            # base_tools_only = [t for t in all_base_tools if t not in aider_tools]
            # available_tools = base_tools_only + aider_tools
            # available_tool_functions = [convert_to_openai_function(t) for t in available_tools]

        else:
            # Session not active, provide only StartCodeEditorTool + base tools
            # Filter out all aider tools except the start tool
            base_tools_only = [t for t in all_base_tools if t not in aider_tools]
            available_tools = base_tools_only + [start_code_editor_tool]
            available_tool_functions = [convert_to_openai_function(t) for t in available_tools]

        # Pass available_tool_functions to your LLM invocation
        # llm.bind_tools(available_tool_functions) # Or however your agent passes tools
        ```
*   **Handling `[CODE_EDITOR_INPUT_NEEDED]`:**
    *   After invoking a tool (specifically `edit_code` or `submit_code_editor_input`), check the result string.
    *   Find the part of your code that processes the `ToolMessage` content returned after a tool call.
        ```python
        from .aider_integration import SIGNAL_PROMPT_NEEDED # Import the signal constant

        # Inside your loop where you handle tool_messages:
        # for tool_msg in tool_messages:
        #    tool_output = tool_msg.content
        #    ...

        if isinstance(tool_output, str) and tool_output.startswith(SIGNAL_PROMPT_NEEDED):
            # Aider needs input!
            prompt_details = tool_output[len(SIGNAL_PROMPT_NEEDED):].strip()
            logger.info(f"Aider needs input: {prompt_details}")

            # Construct a message to send back to the LLM, asking it to provide the input
            # This message becomes the next 'user' or 'system' message in the history
            # before calling the LLM again.
            input_request_message = HumanMessage(
                 content=f"Aider requires input to proceed. {prompt_details}"
                 # You might want to make this a SystemMessage or format it differently
                 # depending on how you want the LLM to react.
            )
            current_messages.append(input_request_message) # Add to history

            # Continue the loop to call the LLM again with the request for input
            # The LLM should ideally respond by calling 'submit_code_editor_input'
            continue # Or break and re-invoke LLM depending on loop structure

        else:
            # Process normal tool output
            logger.info(f"Tool output: {tool_output}")
            # Add tool_output to history as usual
            # current_messages.append(tool_msg) # Append the original ToolMessage
        ```

## Phase 5: Dependencies 

*   **Ensure `aider-chat` is installed:** Add `aider-chat` to your project's dependencies (e.g., `pyproject.toml`, `requirements.txt`). Make sure you install a version compatible with the code structure used here (recent versions should be fine).

    ```toml
    # Example for pyproject.toml
    [tool.poetry.dependencies]
    # ... other dependencies
    aider-chat = "^0.48.0" # Use an appropriate version specifier
    ```

    Or in `requirements.txt`:

    ```
    # requirements.txt
    # ... other requirements
    aider-chat>=0.48.0
    ```
*   **Run `pip install -e .`** (or your project's install command) after updating dependencies.

## LEAVE FOR QA: 
1.  **Testing:** Thoroughly test the entire workflow:
    *   Start session (`/start_code_editor`).
    *   Add files (`/add_code_file`).
    *   List files (`/list_code_files`).
    *   Request edits (`/edit_code`).
    *   Handle confirmations (`/submit_code_editor_input` when prompted). Test 'yes', 'no', 'all', 'skip', 'don't ask'.
    *   View diffs (`/view_code_diff`).
    *   Undo edits (`/undo_last_code_edit`).
    *   Drop files (`/drop_code_file`).
    *   Close session (`/close_code_editor`).
    *   Test restarting the agent and resuming an Aider session (`/start_code_editor` again).
    *   Test error conditions (e.g., file not found, git errors).
2.  **Refinement:** Adjust logging levels (`logger.debug`, `logger.info`, etc.) as needed for clarity during operation and debugging. Refine the prompt formatting in `[CODE_EDITOR_INPUT_NEEDED]` if the LLM struggles to understand the request.


# Improvements 

Okay, this is a good refinement. We want `ai-shell-agent` to have its own persistent configuration for the Aider settings (edit format, models) that overrides whatever might be saved in Aider's own internal state or defaults.

Here's the detailed plan to implement the `--select-edit-mode` and `--select-coder-models` options:

## Phase 1: Update `ai_shell_agent/config_manager.py`

*Goal: Add functions to manage persistent configuration for Aider's edit format and specific models used within the coder context.*

1.  **Define Valid Edit Formats:** Add a list of valid Aider edit formats with descriptions.

    ```python
    # Add near the top of config_manager.py
    AIDER_EDIT_FORMATS = {
        "whole": "LLM sends back the entire file content.",
        "diff": "LLM sends back search/replace blocks.",
        "diff-fenced": "LLM sends back search/replace blocks within a fenced code block (good for Gemini).",
        "udiff": "LLM sends back simplified unified diff format (good for GPT-4 Turbo).",
        "architect": "High-level planning by main model, detailed edits by editor model.",
        # editor-* formats are typically set automatically with architect mode or --editor-model
        # but allow explicit selection if needed.
        "editor-diff": "Streamlined diff format for editor models.",
        "editor-whole": "Streamlined whole format for editor models.",
    }
    DEFAULT_AIDER_EDIT_FORMAT = None # Let Aider/Model decide by default
    ```

2.  **Implement Edit Format Get/Set/Prompt:**

    ```python
    # Add these functions to config_manager.py

    def get_aider_edit_format() -> Optional[str]:
        """Gets the configured Aider edit format."""
        config = _read_config()
        # Return None if not set, allowing Aider's defaults to take precedence
        return config.get("aider_edit_format", DEFAULT_AIDER_EDIT_FORMAT)

    def set_aider_edit_format(edit_format: Optional[str]) -> None:
        """Sets the Aider edit format in the config."""
        config = _read_config()
        if edit_format is None:
            config.pop("aider_edit_format", None) # Remove key if set to None (use default)
            logger.info("Aider edit format reset to default (model-specific).")
        elif edit_format in AIDER_EDIT_FORMATS:
            config["aider_edit_format"] = edit_format
            logger.info(f"Aider edit format set to: {edit_format}")
        else:
            logger.error(f"Invalid edit format '{edit_format}'. Not setting.")
            return
        _write_config(config)

    def prompt_for_edit_mode_selection() -> Optional[str]:
        """Prompts the user to select an Aider edit format."""
        current_format = get_aider_edit_format()
        print("\nSelect Aider Edit Format:")
        print("-------------------------")
        i = 0
        valid_choices = {}
        # Option 0: Use Default
        print(f"  0: Default (Let Aider/Model choose based on the main model) {'<- Current Setting' if current_format is None else ''}")
        valid_choices['0'] = None

        # List other formats
        format_list = sorted(AIDER_EDIT_FORMATS.keys())
        for idx, fmt in enumerate(format_list, 1):
            description = AIDER_EDIT_FORMATS[fmt]
            marker = " <- Current Setting" if fmt == current_format else ""
            print(f"  {idx}: {fmt}{marker} - {description}")
            valid_choices[str(idx)] = fmt

        while True:
            try:
                choice = input(f"Enter choice (0-{len(format_list)}): ").strip()
                if choice in valid_choices:
                    selected_format = valid_choices[choice]
                    set_aider_edit_format(selected_format)
                    return selected_format
                elif not choice: # Allow empty input to keep current setting
                     print(f"Keeping current setting: {current_format or 'Default'}")
                     return current_format
                else:
                    print("Invalid choice. Please try again.")
            except EOFError:
                print("\nSelection cancelled.")
                return current_format # Keep current on Ctrl+D
            except KeyboardInterrupt:
                 print("\nSelection cancelled.")
                 return current_format # Keep current on Ctrl+C
    ```

3.  **Implement Coder Model Get/Set/Prompt:**

    ```python
    # Add these functions to config_manager.py

    def _get_aider_model(config_key: str) -> Optional[str]:
        """Helper to get a specific aider model config."""
        config = _read_config()
        return config.get(config_key) # Returns None if not set

    def _set_aider_model(config_key: str, model_name: Optional[str]) -> None:
        """Helper to set a specific aider model config."""
        config = _read_config()
        normalized_name = normalize_model_name(model_name) if model_name else None
        if normalized_name:
             config[config_key] = normalized_name
             logger.info(f"Set {config_key} to: {normalized_name}")
        elif config_key in config:
             del config[config_key] # Remove if set to None
             logger.info(f"Reset {config_key} to default.")
        _write_config(config)

    def get_aider_main_model() -> Optional[str]:
        """Gets the configured main/architect model for Aider."""
        return _get_aider_model("aider_main_model")

    def set_aider_main_model(model_name: Optional[str]) -> None:
        """Sets the main/architect model for Aider."""
        _set_aider_model("aider_main_model", model_name)

    def get_aider_editor_model() -> Optional[str]:
        """Gets the configured editor model for Aider."""
        return _get_aider_model("aider_editor_model")

    def set_aider_editor_model(model_name: Optional[str]) -> None:
        """Sets the editor model for Aider."""
        _set_aider_model("aider_editor_model", model_name)

    def get_aider_weak_model() -> Optional[str]:
        """Gets the configured weak model for Aider."""
        return _get_aider_model("aider_weak_model")

    def set_aider_weak_model(model_name: Optional[str]) -> None:
        """Sets the weak model for Aider."""
        _set_aider_model("aider_weak_model", model_name)

    def _prompt_for_single_coder_model(role_name: str, current_value: Optional[str]) -> Optional[str]:
        """Helper to prompt for one of the coder models."""
        print(f"\n--- Select Aider {role_name} Model ---")
        # Use the existing model selection prompt but adapt the message
        # Pass the current value so it can be displayed
        # Need to slightly modify prompt_for_model_selection or create a variant
        # For now, let's assume prompt_for_model_selection can take a current_value hint

        # Reusing prompt_for_model_selection; it needs modification to accept a current value hint
        # and maybe a different prompt message. Let's define a new function for clarity.

        print("Available models (same as agent models):")
        all_model_names = sorted(list(set(ALL_MODELS.values()))) # Use ALL_MODELS from config_manager
        for model in all_model_names:
             marker = " <- Current Value" if model == current_value else ""
             print(f"- {model}{marker}")
             # Add aliases maybe?
             aliases = [alias for alias, full_name in ALL_MODELS.items() if full_name == model and alias != model]
             if aliases: print(f"  (aliases: {', '.join(aliases)})")

        prompt_msg = (f"Enter model name for {role_name} role"
                      f" (leave empty to keep '{current_value or 'Default'}',"
                      f" enter 'none' to use default): ")

        while True:
            selected = input(prompt_msg).strip()
            if not selected:
                print(f"Keeping current setting: {current_value or 'Default'}")
                return current_value # Keep current
            elif selected.lower() == 'none':
                 print(f"Resetting {role_name} model to default.")
                 return None # Use None to signify default/reset
            else:
                 normalized_model = normalize_model_name(selected)
                 if normalized_model in all_model_names:
                      return normalized_model # Return the selected normalized name
                 else:
                      print(f"Error: Unknown model '{selected}'. Please choose from the list or enter 'none'.")


    def prompt_for_coder_models_selection() -> None:
        """Runs the multi-step wizard to select Aider coder models."""
        print("\n--- Configure Aider Coder Models ---")
        print("Select the models Aider should use for different coding tasks.")
        print("Leave input empty at any step to keep the current setting.")
        print("Enter 'none' to reset a model to its default behavior.")

        current_main = get_aider_main_model()
        selected_main = _prompt_for_single_coder_model("Main/Architect", current_main)
        if selected_main != current_main: # Only set if changed
            set_aider_main_model(selected_main)

        current_editor = get_aider_editor_model()
        selected_editor = _prompt_for_single_coder_model("Editor", current_editor)
        if selected_editor != current_editor:
            set_aider_editor_model(selected_editor)

        current_weak = get_aider_weak_model()
        selected_weak = _prompt_for_single_coder_model("Weak (Commits/Summaries)", current_weak)
        if selected_weak != current_weak:
            set_aider_weak_model(selected_weak)

        print("\nAider coder models configuration updated.")

    ```

## Phase 2: Update `ai_shell_agent/ai.py`

*Goal: Add the new command-line arguments and the logic to call the selection wizards.*

1.  **Add Arguments:** Add `--select-edit-mode` and `--select-coder-models` to the `ArgumentParser` in the `main` function.

    ```python
    # In main() function, within parser definition:
    # ... other arguments ...

    # Aider Configuration Wizards
    parser.add_argument("--select-edit-mode", action="store_true", help="Interactively select the edit format Aider should use.")
    parser.add_argument("--select-coder-models", action="store_true", help="Interactively select the models Aider should use for coding tasks.")

    # ... rest of the arguments ...
    ```

2.  **Call Wizards:** Add logic in `main()` to call the new prompt functions. Place this *after* `ensure_api_key()` and *before* any commands that might initiate an Aider session (like `-m`, `-tc`, `message`).

    ```python
    # In main() function, after ensure_api_key() and before command handling:

    # Handle Aider configuration wizards if arguments are present
    if args.select_edit_mode:
        prompt_for_edit_mode_selection()
        # Optionally return here if this is the only action desired
        # return # Uncomment if you want the script to exit after selection

    if args.select_coder_models:
        # We need API keys potentially checked for the models we might select,
        # so ensure_api_key() should ideally run before this or be handled within.
        # Let's assume ensure_api_key() for the *main agent* model is sufficient for now.
        prompt_for_coder_models_selection()
        # Optionally return here
        # return # Uncomment if desired

    # ... rest of the command handling (if args.execute:, if args.chat:, etc.) ...
    ```
3.  **Import new functions:** Add the necessary imports at the top of `ai.py`.
    ```python
    from .config_manager import (
        # ... existing imports ...
        prompt_for_edit_mode_selection, # New import
        prompt_for_coder_models_selection, # New import
    )
    ```


## Phase 3: Update `ai_shell_agent/aider_integration.py`

*Goal: Make `StartCodeEditorTool` and `recreate_coder` respect the new configuration settings.*

1.  **Import Getters:** Import the new getter functions from `config_manager`.

    ```python
    # Add to imports at the top of aider_integration.py
    from .config_manager import (
        # ... existing imports ...
        get_aider_edit_format,
        get_aider_main_model,
        get_aider_editor_model,
        get_aider_weak_model
    )
    ```

2.  **Modify `recreate_coder`:** Update the logic for determining models and edit format.

    ```python
    # Inside recreate_coder function:
    # ... after reading aider_state ...

    # --- Model and Config Setup ---
    # Priority: Agent Config -> Persistent State -> Agent Current Model
    main_model_name_cfg = get_aider_main_model()
    main_model_name_state = aider_state.get('main_model_name')
    main_model_name_agent = get_current_model() # Agent's primary model

    main_model_name = main_model_name_cfg or main_model_name_state or main_model_name_agent
    if not main_model_name:
         logger.error("Cannot determine main model name for Aider Coder recreation.")
         return None
    logger.debug(f"Using main model: {main_model_name} (Source: {'Config' if main_model_name_cfg else 'State' if main_model_name_state else 'Agent'})")

    # Other models (Priority: Agent Config -> Persistent State -> None/Default)
    editor_model_name = get_aider_editor_model() or aider_state.get('editor_model_name')
    weak_model_name = get_aider_weak_model() or aider_state.get('weak_model_name')
    logger.debug(f"Using editor model: {editor_model_name or 'Default'}")
    logger.debug(f"Using weak model: {weak_model_name or 'Default'}")


    # Edit format (Priority: Agent Config -> Persistent State -> Model Default)
    edit_format_cfg = get_aider_edit_format()
    edit_format_state = aider_state.get('edit_format')
    # We'll determine the final edit_format after instantiating the model if needed

    editor_edit_format = aider_state.get('editor_edit_format') # Editor format primarily from state for now

    # Ensure API key for the *determined* main_model_name
    api_key, env_var = get_api_key_for_model(main_model_name)
    if not api_key:
        logger.error(f"API Key ({env_var}) not found for model {main_model_name}. Cannot recreate Coder.")
        return None

    try:
        main_model_instance = Model(
             main_model_name,
             weak_model=weak_model_name, # Pass potentially overridden models
             editor_model=editor_model_name,
             editor_edit_format=editor_edit_format # Pass format from state
        )
        # Determine final edit format
        # If config is set, use it. If not, use state. If neither, use model default.
        edit_format = edit_format_cfg or edit_format_state or main_model_instance.edit_format
        logger.debug(f"Using edit format: {edit_format} (Source: {'Config' if edit_format_cfg else 'State' if edit_format_state else 'Model Default'})")

    except Exception as e:
         logger.error(f"Failed to instantiate main_model '{main_model_name}': {e}")
         return None

    # --- Load Aider History --- (Keep existing logic)
    aider_done_messages = aider_state.get("aider_done_messages", [])
    # ... rest of history loading ...

    # --- Git Repo Setup --- (Keep existing logic)
    # ... git repo setup ...

    # --- Prepare Explicit Config for Coder.create ---
    coder_kwargs = dict(
        main_model=main_model_instance, # Use the instance created above
        edit_format=edit_format, # Use the final determined edit format
        io=io_stub,
        repo=repo,
        fnames=abs_fnames,
        read_only_fnames=abs_read_only_fnames,
        done_messages=aider_done_messages,
        cur_messages=[],
        auto_commits=aider_state.get("auto_commits", True),
        dirty_commits=aider_state.get("dirty_commits", True),
        use_git=bool(repo),
        map_tokens=aider_state.get("map_tokens", 0),
        verbose=False,
        stream=False,
        suggest_shell_commands=False,
    )

    # --- Instantiate Coder --- (Keep existing logic)
    coder = Coder.create(**coder_kwargs)
    # ... rest of coder setup (root, commands) ...

    # ---> IMPORTANT: Immediately update persistent state with potentially resolved values <---
    # This ensures the state reflects overrides even if the session was just recreated
    # This replaces the update call that was previously suggested to be only in StartCodeEditorTool
    # Because recreate_coder is now passive, the caller (StartCodeEditorTool) needs to handle
    # saving the state *after* associating it with the active session. Let's remove this update here.
    # update_aider_state_from_coder(chat_file, coder) # REMOVED FROM HERE

    logger.info(f"Coder successfully recreated for {chat_file}")
    return coder

    # ... rest of recreate_coder ...
    ```

3.  **Modify `StartCodeEditorTool._run`:** Update the "Start Fresh" and "Resumed Successfully" sections to use the new config getters and to update the persistent state correctly *after* creating the active state.

    ```python
    # Inside StartCodeEditorTool._run method:

    # ---> In the "Resumed Successfully" block (after `coder = recreate_coder(...)`) <---
    if coder:
        logger.info(f"Resuming Aider session for {chat_file} from persistent state.")
        # Create the active state using the recreated coder
        state = create_active_coder_state(chat_file, coder) # This now associates the ACTUAL io_stub

        # ---> Update persistent state AFTER creating active state <---
        # This ensures the saved state reflects any overrides applied during recreation
        update_aider_state_from_coder(chat_file, coder)

        recreation_output = temp_io_stub.get_captured_output()
        if recreation_output:
             logger.warning(f"Output during coder recreation for {chat_file}: {recreation_output}")
        # Update return message to show effective settings
        return (f"Code editor session resumed (Main: {coder.main_model.name},"
                 f" Editor: {getattr(coder.main_model.editor_model, 'name', 'Default')},"
                 f" Weak: {getattr(coder.main_model.weak_model, 'name', 'Default')},"
                 f" Format: {coder.edit_format}). Ready for files and edits.")

    # ---> In the "Failed to Resume: Start Fresh" block <---
    else:
        logger.info(f"No valid persistent state found for {chat_file}, starting fresh.")
        clear_aider_state(chat_file)

        # --- Determine initial settings using config overrides ---
        main_model_name = get_aider_main_model() or get_current_model()
        editor_model_name = get_aider_editor_model() # Defaults to None if not set
        weak_model_name = get_aider_weak_model()     # Defaults to None if not set
        edit_format = get_aider_edit_format()       # Defaults to None if not set
        # Note: editor_edit_format is not directly configured via wizard yet

        if not main_model_name:
            return "Error: Could not determine the main model for the agent or Aider config."

        # Ensure API key for the main model
        api_key, env_var = get_api_key_for_model(main_model_name)
        if not api_key:
             # Handle missing API key - maybe call ensure_api_key_for_model?
             # For now, return error, assuming ensure_api_key ran earlier in ai.py
             return f"Error: API Key ({env_var}) not found for model {main_model_name}."

        # Instantiate the main model to get defaults if edit_format is still None
        final_edit_format = edit_format
        final_editor_edit_format = None # Let Model/Coder handle this initially
        try:
            temp_model = Model(
                main_model_name,
                weak_model=weak_model_name,
                editor_model=editor_model_name
                # No editor_edit_format here yet
            )
            if final_edit_format is None: # If no config override and no state value (which is none here)
                final_edit_format = temp_model.edit_format
            # Get the default editor format if an editor model exists
            final_editor_edit_format = getattr(temp_model, 'editor_edit_format', None)

        except Exception as e:
             logger.warning(f"Could not get model defaults for {main_model_name}: {e}. Using basic defaults.")
             if final_edit_format is None: final_edit_format = 'whole' # Safe fallback

        # --- Define initial persistent state (using determined settings) ---
        initial_state = {
            "enabled": True,
            "main_model_name": main_model_name,
            "edit_format": final_edit_format,
            "weak_model_name": weak_model_name,
            "editor_model_name": editor_model_name,
            "editor_edit_format": final_editor_edit_format,
            "abs_fnames": [],
            "abs_read_only_fnames": [],
            "aider_done_messages": [],
            "aider_commit_hashes": [],
            "git_root": None,
            "auto_commits": True,
            "dirty_commits": True,
        }
        # Save initial state (without git_root initially)
        save_aider_state(chat_file, initial_state)

        # --- Now create the Coder instance for the active session ---
        try:
             fresh_io_stub = AiderIOStubWithQueues()
             # Re-instantiate model with final determined names
             fresh_main_model = Model(
                  main_model_name,
                  weak_model=weak_model_name,
                  editor_model=editor_model_name,
                  editor_edit_format=final_editor_edit_format # Pass determined format
             )
             fresh_coder = Coder.create(
                  main_model=fresh_main_model,
                  edit_format=final_edit_format, # Use determined format
                  io=fresh_io_stub,
                  fnames=[], read_only_fnames=[], done_messages=[], cur_messages=[],
                  auto_commits=True, dirty_commits=True, use_git=True
             )

             # If repo was found, update persistent state *again* with git_root
             if fresh_coder.repo:
                  initial_state["git_root"] = fresh_coder.repo.root
                  save_aider_state(chat_file, initial_state)

             # Create the active state
             create_active_coder_state(chat_file, fresh_coder)

             # Update persistent state *after* creating active state
             # This ensures the saved state reflects the actual coder created
             update_aider_state_from_coder(chat_file, fresh_coder)

             # Update return message
             return (f"New code editor session started (Main: {fresh_coder.main_model.name},"
                      f" Editor: {getattr(fresh_coder.main_model.editor_model, 'name', 'Default')},"
                      f" Weak: {getattr(fresh_coder.main_model.weak_model, 'name', 'Default')},"
                      f" Format: {fresh_coder.edit_format}). Ready for files and edits.")
        except Exception as e:
             logger.error(f"Failed to create new Coder instance for {chat_file}: {e}")
             logger.error(traceback.format_exc())
             clear_aider_state(chat_file)
             remove_active_coder_state(chat_file)
             return f"Error: Failed to initialize code editor. {e}"

    ```

4.  **Modify `update_aider_state_from_coder`:** Ensure it saves the *effective* settings currently used by the `coder` instance. *Self-correction: The existing `update_aider_state_from_coder` already reads from the `coder` instance, so it should correctly save the effective settings after overrides have been applied during creation/recreation. No change needed here.*

## Phase 4: Testing

1.  Run `aider --select-edit-mode` and choose an option (e.g., `udiff`). Verify `config.json` is updated.
2.  Run `aider --select-coder-models`. Select different models for main, editor, and weak roles. Verify `config.json`.
3.  Start aider with a specific chat (`aider -c my-coding-chat`).
4.  Start the code editor (`/start_code_editor`). Verify the output message reflects the configured models/format from steps 1 & 2, not necessarily the agent's default model or Aider's defaults.
5.  Add a file (`/add_code_file test.py`).
6.  Request an edit (`/edit_code "Refactor this"`).
7.  Close the agent (Ctrl+C).
8.  Restart the agent with the same chat (`aider -c my-coding-chat`).
9.  Start the code editor again (`/start_code_editor`). Verify it *resumes* and the output message *still* reflects the models/format configured in steps 1 & 2.
10. Reset a setting (e.g., run `aider --select-edit-mode` and choose `0` for Default). Verify `config.json` removes the key. Start aider again and verify the output message reflects the default behavior.

This completes the implementation steps for adding the configuration overrides.