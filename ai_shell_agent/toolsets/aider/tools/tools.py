# =========================================================================
# File: ai_shell_agent/toolsets/aider/tools/tools.py
# =========================================================================
"""
Tool definitions for the AI Code Copilot (Aider) toolset.
"""
import os
import threading
import traceback
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Type

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, Field

# Local Imports from parent toolset and core app
from .... import logger
from ....errors import PromptNeededError
from ....chat_state_manager import get_current_chat, get_toolset_data_path
from ....utils.file_io import read_json # Need this for ListFilesInContext

# Import toolset-specific components
from ..integration.integration import (
    ensure_active_coder_state, update_aider_state_from_coder,
    remove_active_coder_state, SIGNAL_PROMPT_NEEDED, TIMEOUT,
    _run_aider_in_thread, ANY_GIT_ERROR # Import ANY_GIT_ERROR here
)
from ..prompts import AIDER_TOOLSET_PROMPT
from ..texts import get_text # Import toolset-specific texts
from ..settings import ( # Keep settings import if needed (not directly used here currently)
    AIDER_DEFAULT_MAIN_MODEL, AIDER_DEFAULT_EDITOR_MODEL, AIDER_DEFAULT_WEAK_MODEL,
    AIDER_DEFAULT_EDIT_FORMAT, AIDER_DEFAULT_AUTO_COMMITS, AIDER_DEFAULT_DIRTY_COMMITS
)
from .schemas import ( # Import schemas from the sibling file
    NoArgsSchema, FilePathSchema, InstructionSchema, UserResponseSchema
)


# --- Tool Classes ---

class AiderUsageGuideTool(BaseTool):
    name: str = get_text("tools.usage_guide.name")
    description: str = get_text("tools.usage_guide.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Returns the usage instructions for the Aider toolset."""
        logger.debug(f"AiderUsageGuideTool invoked.")
        return AIDER_TOOLSET_PROMPT

    async def _arun(self) -> str:
        return await run_in_executor(None, self._run)


class AddFileToConext(BaseTool):
    name: str = get_text("tools.add_file.name")
    description: str = get_text("tools.add_file.description")
    args_schema: Type[BaseModel] = FilePathSchema

    def _run(self, file_path: str) -> str:
        """Adds a file to the Aider context, recreating state if needed."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.add_file.error.no_chat")

        # Use toolset_id defined in parent or pass it explicitly
        toolset_id = "aider" # Hardcode or get from parent if needed
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.add_file.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub

        try:
            abs_path_to_add = str(Path(file_path).resolve())

            if not os.path.exists(abs_path_to_add):
                return get_text("tools.add_file.error.not_exists", path=file_path, abs_path=abs_path_to_add)
            if coder.repo and coder.root != os.getcwd():
                try:
                    if hasattr(Path, 'is_relative_to'): # Use modern Path method if available
                        if not Path(abs_path_to_add).is_relative_to(Path(coder.root)):
                             return get_text("tools.add_file.error.outside_root", path=file_path, root=coder.root)
                    else: # Fallback for older Python versions
                         rel_path_check = os.path.relpath(abs_path_to_add, coder.root)
                         if rel_path_check.startswith('..'):
                              return get_text("tools.add_file.error.outside_root", path=file_path, root=coder.root)
                except ValueError: # Handles cross-drive paths on Windows
                     return get_text("tools.add_file.error.different_drive", path=file_path, root=coder.root)

            rel_path = coder.get_rel_fname(abs_path_to_add)
            coder.add_rel_fname(rel_path)

            update_aider_state_from_coder(aider_json_path, coder)
            logger.info(f"Added file {rel_path} and updated persistent state for {chat_id}")

            if abs_path_to_add in coder.abs_fnames:
                return get_text("tools.add_file.success", path=rel_path, output=io_stub.get_captured_output())
            else:
                logger.error(f"File {abs_path_to_add} not found in coder.abs_fnames after adding.")
                return get_text("tools.add_file.warn_confirm_failed", path=rel_path)

        except Exception as e:
            logger.error(f"Error in AddFileTool: {e}", exc_info=True)
            return get_text("tools.add_file.error.generic", path=file_path, error=e, output=io_stub.get_captured_output())

    async def _arun(self, file_path: str) -> str:
        return await run_in_executor(None, self._run, file_path)


class RemoveFileFromContext(BaseTool):
    name: str = get_text("tools.remove_file.name")
    description: str = get_text("tools.remove_file.description")
    args_schema: Type[BaseModel] = FilePathSchema

    def _run(self, file_path: str) -> str:
        """Removes a file from the Aider context."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.remove_file.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.remove_file.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub

        try:
            abs_path_to_drop = str(Path(file_path).resolve())
            rel_path_to_drop = coder.get_rel_fname(abs_path_to_drop)

            success = coder.drop_rel_fname(rel_path_to_drop)

            if success:
                update_aider_state_from_coder(aider_json_path, coder)
                return get_text("tools.remove_file.success", path=file_path, output=io_stub.get_captured_output())
            else:
                return get_text("tools.remove_file.info_not_found", path=file_path, output=io_stub.get_captured_output())

        except Exception as e:
            logger.error(f"Error in DropFileTool: {e}", exc_info=True)
            return get_text("tools.remove_file.error.generic", path=file_path, error=e, output=io_stub.get_captured_output())

    async def _arun(self, file_path: str) -> str:
        return await run_in_executor(None, self._run, file_path)


class ListFilesInContext(BaseTool):
    name: str = get_text("tools.list_files.name")
    description: str = get_text("tools.list_files.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Lists all files in the Aider context."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.list_files.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        aider_state = read_json(aider_json_path, default_value=None) # Use imported read_json
        if not aider_state or not aider_state.get("enabled", False):
            return get_text("tools.list_files.error.init_failed")

        try:
            files = aider_state.get("abs_fnames", [])
            if not files:
                return get_text("tools.list_files.info_empty")

            root = aider_state.get("git_root", os.getcwd())
            files_list = []
            for f in files:
                try:
                    rel_path = os.path.relpath(f, root)
                    files_list.append(rel_path)
                except ValueError:
                    files_list.append(f)

            return get_text("tools.list_files.success", file_list="\n".join(f"- {f}" for f in sorted(files_list)))

        except Exception as e:
            logger.error(f"Error in ListFilesInEditorTool: {e}", exc_info=True)
            return get_text("tools.list_files.error.generic", error=e)

    async def _arun(self) -> str:
        return await run_in_executor(None, self._run)


class RequestEdits(BaseTool):
    name: str = get_text("tools.request_edit.name")
    description: str = get_text("tools.request_edit.description")
    args_schema: Type[BaseModel] = InstructionSchema

    def _run(self, instruction: str) -> str:
        """Runs Aider's main edit loop in a background thread."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.request_edit.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.request_edit.error.init_failed")

        if not state.coder.abs_fnames:
             return get_text("tools.request_edit.error.no_files")

        with state.lock:
            if state.thread and state.thread.is_alive():
                logger.warning(f"An edit is already in progress for {chat_id}. Please wait or submit input if needed.")
                return get_text("tools.request_edit.error.in_progress")

            if state.coder.io is not state.io_stub:
                 logger.warning("Correcting coder IO instance mismatch.")
                 state.coder.io = state.io_stub

            logger.info(f"Starting Aider worker thread for: {chat_id}")
            state.thread = threading.Thread(
                target=_run_aider_in_thread,
                args=(state.coder, instruction, state.output_q),
                daemon=True,
                name=f"AiderWorker-{chat_id[:8]}"
            )
            state.thread.start()

            update_aider_state_from_coder(aider_json_path, state.coder)

        logger.debug(f"Main thread waiting for initial message from output_q for {chat_id}...")
        try:
             message = state.output_q.get(timeout=TIMEOUT)
             logger.debug(f"Main thread received initial message: {message.get('type')}")
        except queue.Empty:
             logger.error(f"Timeout waiting for initial Aider response ({chat_id}).")
             remove_active_coder_state(aider_json_path)
             return get_text("tools.request_edit.error.timeout")
        except Exception as e:
              logger.error(f"Exception waiting on output_q for {chat_id}: {e}")
              remove_active_coder_state(aider_json_path)
              return get_text("tools.request_edit.error.queue_exception", error=e)


        message_type = message.get('type')

        if message_type == 'prompt':
            prompt_data = message
            question = prompt_data.get('question', 'Input needed')
            subject = prompt_data.get('subject')
            default = prompt_data.get('default')
            allow_never = prompt_data.get('allow_never')

            response_guidance = get_text("tools.request_edit.prompt_guidance", prompt=question)
            if subject: response_guidance += get_text("tools.request_edit.prompt_subject", subject=subject[:100]+'...' if len(subject)>100 else subject)
            if default: response_guidance += get_text("tools.request_edit.prompt_default", default=default)
            if prompt_data.get('prompt_type', 'unknown') == 'confirm':
                 options = "(yes/no"
                 if prompt_data.get('group_id'): options += "/all/skip"
                 if allow_never: options += "/don't ask"
                 options += ")"
                 response_guidance += get_text("tools.request_edit.prompt_confirm_options", options=options)

            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)

            return get_text("tools.request_edit.prompt_needed", signal=SIGNAL_PROMPT_NEEDED, guidance=response_guidance)

        elif message_type == 'result':
            logger.info(f"Aider edit completed successfully for {chat_id}.")
            with state.lock:
                update_aider_state_from_coder(aider_json_path, state.coder)
                state.thread = None
            return get_text("tools.request_edit.success", output=message.get('content', 'No output captured.'))

        elif message_type == 'error':
            logger.error(f"Aider edit failed for {chat_id}.")
            error_content = message.get('message', 'Unknown error')

            try:
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
            except Exception:
                pass

            remove_active_coder_state(aider_json_path)
            return get_text("tools.request_edit.error.edit_failed", content=error_content)
        else:
             logger.error(f"Received unknown message type from Aider thread: {message_type}")
             remove_active_coder_state(aider_json_path)
             return get_text("tools.request_edit.error.unknown_response", type=message_type)

    async def _arun(self, instruction: str) -> str:
        return await run_in_executor(None, self._run, instruction)


class ViewDiffs(BaseTool):
    name: str = get_text("tools.view_diffs.name")
    description: str = get_text("tools.view_diffs.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Shows the diff of changes made by Aider."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.view_diffs.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.view_diffs.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub

        try:
            if not coder.repo:
                return get_text("tools.view_diffs.error.no_repo")

            if hasattr(coder, 'commands') and coder.commands:
                coder.commands.raw_cmd_diff("")
                captured = io_stub.get_captured_output()
                return get_text("tools.view_diffs.success", diff=captured) if captured else get_text("tools.view_diffs.info_no_changes")
            else:
                diff = coder.repo.get_unstaged_changes()

                if not diff:
                    return get_text("tools.view_diffs.info_no_changes")

                return get_text("tools.view_diffs.success", diff=diff)

        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during diff: {e}")
            return get_text("tools.view_diffs.error.git_error", error=e, output=io_stub.get_captured_output()).strip()

        except Exception as e:
            logger.error(f"Error in ViewDiffTool: {e}")
            return get_text("tools.view_diffs.error.generic", error=e, output=io_stub.get_captured_output())

    async def _arun(self) -> str:
        return await run_in_executor(None, self._run())

class UndoLastEditTool(BaseTool):
    name: str = get_text("tools.undo_edit.name")
    description: str = get_text("tools.undo_edit.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Undoes the last edit commit made by Aider."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.undo_edit.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.undo_edit.error.init_failed")

        coder = state.coder
        io_stub = state.io_stub

        try:
            if not coder.repo:
                return get_text("tools.undo_edit.error.no_repo")

            try:
                from aider.commands import Commands
                if not hasattr(coder, "commands"):
                    coder.commands = Commands(io=io_stub, coder=coder)
            except ImportError:
                return get_text("tools.undo_edit.error.cmd_module_missing")

            coder.commands.raw_cmd_undo(None)
            update_aider_state_from_coder(aider_json_path, coder)

            return get_text("tools.undo_edit.success", output=io_stub.get_captured_output()).strip()

        except ANY_GIT_ERROR as e:
            logger.error(f"Git error during undo: {e}")
            return get_text("tools.undo_edit.error.git_error", error=e, output=io_stub.get_captured_output()).strip()

        except Exception as e:
            logger.error(f"Unexpected error during undo: {e}")
            return get_text("tools.undo_edit.error.generic", error=e, output=io_stub.get_captured_output()).strip()

    async def _arun(self) -> str:
        return await run_in_executor(None, self._run())

class CloseCodeCopilot(BaseTool):
    name: str = get_text("tools.close_editor.name")
    description: str = get_text("tools.close_editor.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Clears the Aider state and cleans up runtime state."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.close_editor.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)

        state_cleared_msg = ""
        try:
            remove_active_coder_state(aider_json_path)
            logger.info(f"Aider state cleared for {chat_id}")
            state_cleared_msg = get_text("tools.close_editor.info_success")
        except Exception as e:
            logger.error(f"Error clearing Aider state: {e}", exc_info=True)
            state_cleared_msg = get_text("tools.close_editor.warn_error")

        return state_cleared_msg.strip()

    async def _arun(self) -> str:
        return await run_in_executor(None, self._run())


class SubmitInput_HITL(BaseTool):
    name: str = get_text("tools.submit_input.name")
    description: str = get_text("tools.submit_input.description")
    args_schema: Type[BaseModel] = UserResponseSchema
    requires_confirmation: bool = True

    def _run(self, user_response: str, confirmed_input: Optional[str] = None) -> str:
        """Handles submitting input to Aider using the PromptNeededError approach."""
        chat_id = get_current_chat()
        if not chat_id:
            return get_text("tools.submit_input.error.no_chat")

        toolset_id = "aider"
        aider_json_path = get_toolset_data_path(chat_id, toolset_id)
        state = ensure_active_coder_state(aider_json_path)
        if not state:
            return get_text("tools.submit_input.error.init_failed")

        if confirmed_input is None:
            logger.debug(f"SubmitInputTool: Raising PromptNeededError")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"user_response": user_response},
                edit_key="user_response",
                prompt_suffix=get_text("tools.submit_input.prompt_suffix")
            )

        edited_response = confirmed_input

        if not edited_response.strip():
             return get_text("tools.submit_input.error.empty_confirmed")

        processed_result = ""

        with state.lock:
            if not state.thread or not state.thread.is_alive():
                remove_active_coder_state(aider_json_path)
                return get_text("tools.submit_input.error.not_waiting")

            logger.debug(f"Putting confirmed response on input_q: '{edited_response[:50]}...' for {chat_id}")
            state.input_q.put(edited_response)

        logger.debug(f"Main thread waiting for response from output_q for {chat_id}...")
        try:
            message = state.output_q.get(timeout=TIMEOUT)
            logger.debug(f"Main thread received message from output_q: {message.get('type')}")

            message_type = message.get('type')
            if message_type == 'prompt':
                prompt_data = message
                question = prompt_data.get('question', 'Input needed')
                subject = prompt_data.get('subject')
                default = prompt_data.get('default')

                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)

                prompt_msg = question
                if subject:
                    prompt_msg += get_text("tools.submit_input.prompt_subject", subject=subject[:50]+'...' if len(subject) > 50 else subject)

                new_args = {"user_response": default or ""}

                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args=new_args,
                    edit_key="user_response",
                    prompt_suffix=get_text("tools.submit_input.prompt_hitl_suffix", prompt=prompt_msg)
                )

            elif message_type == 'result':
                logger.info(f"Aider edit completed successfully for {chat_id} after input.")
                with state.lock:
                    update_aider_state_from_coder(aider_json_path, state.coder)
                    state.thread = None
                processed_result = get_text("tools.submit_input.success", output=message.get('content', 'No output captured.'))

            elif message_type == 'error':
                logger.error(f"Aider edit failed for {chat_id} after input.")
                error_content = message.get('message', 'Unknown error')
                try:
                    with state.lock:
                        update_aider_state_from_coder(aider_json_path, state.coder)
                except Exception:
                    pass
                remove_active_coder_state(aider_json_path)
                processed_result = get_text("tools.submit_input.error.edit_failed", content=error_content)

            else:
                logger.error(f"Unknown message type from Aider thread: {message_type}")
                try:
                    with state.lock:
                        update_aider_state_from_coder(aider_json_path, state.coder)
                except Exception:
                    pass
                remove_active_coder_state(aider_json_path)
                processed_result = get_text("tools.submit_input.error.unknown_response", type=message_type)

        except queue.Empty:
            logger.error(f"Timeout waiting for Aider response ({chat_id}).")
            remove_active_coder_state(aider_json_path)
            processed_result = get_text("tools.submit_input.error.timeout")
        except PromptNeededError:
            raise
        except Exception as e:
            logger.error(f"Exception waiting on output_q for {chat_id}: {e}", exc_info=True)
            remove_active_coder_state(aider_json_path)
            processed_result = get_text("tools.submit_input.error.queue_exception", error=e)

        return processed_result

    async def _arun(self, user_response: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, user_response, confirmed_input)


# --- Instantiate Tools ---
aider_usage_guide_tool = AiderUsageGuideTool()
add_code_file_tool = AddFileToConext()
drop_code_file_tool = RemoveFileFromContext()
list_code_files_tool = ListFilesInContext()
edit_code_tool = RequestEdits()
submit_code_editor_input_tool = SubmitInput_HITL()
view_diff_tool = ViewDiffs()
undo_last_edit_tool = UndoLastEditTool()
close_code_editor_tool = CloseCodeCopilot()