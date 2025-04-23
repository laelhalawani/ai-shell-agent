# ai_shell_agent/toolsets/files/toolset.py
"""
File Manager toolset: Provides tools for direct file and directory manipulation.
Includes tools for create, read, delete, copy, move, rename, find, edit, history,
and backup management. Handles user confirmation for destructive/modifying actions.
"""
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Union

# Pydantic and Langchain
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor

# Local Imports
from ... import logger, ROOT_DIR # Import ROOT_DIR for config fallback path
from ...tool_registry import register_tools
from ...utils.file_io import read_json, write_json
from ...utils.env import ensure_dotenv_key
from ...errors import PromptNeededError
from ...console_manager import get_console_manager
from ...chat_state_manager import (
    get_current_chat,
    get_toolset_data_path,
)
from ...texts import get_text as get_main_text
from .settings import (
    FILES_HISTORY_LIMIT, FIND_FUZZY_DEFAULT, FIND_THRESHOLD_DEFAULT,
    FIND_LIMIT_DEFAULT, FIND_WORKERS_DEFAULT
)
from .prompts import FILES_TOOLSET_PROMPT # Import the updated prompt text
from .texts import get_text # Toolset-specific texts
from .integration.find_logic import find_files_with_logic

console = get_console_manager()

# --- Toolset Metadata ---
toolset_id = "files"
toolset_name = get_text("toolset.name")
toolset_description = get_text("toolset.description")
toolset_required_secrets: Dict[str, str] = {}

# --- State Management Helpers ---
def _get_history_path(chat_id: str) -> Path:
    return get_toolset_data_path(chat_id, toolset_id)

def _read_toolset_state(chat_id: str) -> Dict:
    state_path = _get_history_path(chat_id)
    # No longer need pending_edit/pending_delete in default state
    default = {"history": []}
    state = read_json(state_path, default_value=default)
    if "history" not in state: state["history"] = []
    return state

def _write_toolset_state(chat_id: str, state_data: Dict) -> bool:
    state_path = _get_history_path(chat_id)
    state_data.setdefault("history", [])
    return write_json(state_path, state_data)

def _log_history_event(chat_id: str, event_data: Dict) -> None:
    if not chat_id: return
    try:
        state = _read_toolset_state(chat_id)
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        state["history"].append(event_data)
        limit = FILES_HISTORY_LIMIT # Use loaded setting
        if limit > 0 and len(state["history"]) > limit:
             state["history"] = state["history"][-limit:]
        if not _write_toolset_state(chat_id, state):
            logger.error(f"Failed to write state after logging history event for chat {chat_id}")
        else:
            logger.debug(f"Logged file history event for chat {chat_id}: {event_data.get('operation')}")
    except Exception as e:
        logger.error(f"Failed to log file history event for chat {chat_id}: {e}", exc_info=True)

# --- Configuration Function ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path],
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
) -> Dict:
    """
    Configuration function for the File Manager toolset.
    Prompts user for history retrieval limit, using defaults from settings.
    """
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat"
    console.display_message(
        get_main_text("common.labels.system"),
        get_text("config.header"),
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )
    config_to_prompt = current_chat_config if current_chat_config is not None else {}
    final_config = {}
    try:
        default_limit = config_to_prompt.get("history_retrieval_limit", FILES_HISTORY_LIMIT)
        limit_str = console.prompt_for_input(
            get_text("config.prompt_limit"),
            default=str(default_limit)
        ).strip()
        try:
            limit = int(limit_str) if limit_str else default_limit
            if limit < 0: limit = 0
            final_config["history_retrieval_limit"] = limit
        except ValueError:
            console.display_message(
                get_main_text("common.labels.warning"),
                get_text("config.warn_invalid"),
                console.STYLE_WARNING_LABEL,
                console.STYLE_WARNING_CONTENT
            )
            final_config["history_retrieval_limit"] = default_limit
    except (KeyboardInterrupt, EOFError):
        console.display_message(
            get_main_text("common.labels.warning"),
            get_text("config.warn_cancel"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}
    except Exception as e:
        logger.error(f"Error during File Manager configuration: {e}", exc_info=True)
        console.display_message(
            get_main_text("common.labels.error"),
            get_text("config.error_generic", error=e),
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}
    save_success_global = True
    save_success_local = True
    try:
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(global_config_path, final_config)
        logger.info(f"File Manager configuration saved to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to save File Manager config to global path {global_config_path}: {e}")
    if local_config_path:
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.info(f"File Manager configuration saved to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to save File Manager config to local path {local_config_path}: {e}")
    if save_success_global and save_success_local:
        console.display_message(
            get_main_text("common.labels.info"),
            get_text("config.info_saved"),
            console.STYLE_INFO_LABEL,
            console.STYLE_INFO_CONTENT
        )
    else:
        console.display_message(
            get_main_text("common.labels.error"),
            get_text("config.error_save_failed"),
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )
    return final_config

# --- Tool Schemas ---
class NoArgsSchema(BaseModel): pass
class PathSchema(BaseModel):
    path: str = Field(description=get_text("schemas.path.path_desc"))
class RestorePathSchema(BaseModel): # Uses renamed schema key from texts.py
    path_or_paths: Union[str, List[str]] = Field(description=get_text("schemas.restore_file.paths_desc"))
    backup_id: Optional[str] = Field(None, description=get_text("schemas.restore_file.backup_id_desc"))
class CreateSchema(BaseModel):
    path: str = Field(description=get_text("schemas.create.path_desc"))
    content: Optional[str] = Field(None, description=get_text("schemas.create.content_desc"))
    is_directory: bool = Field(False, description=get_text("schemas.create.is_directory_desc"))
class OverwriteSchema(BaseModel): # Uses renamed schema key from texts.py
    path: str = Field(description=get_text("schemas.overwrite_file.path_desc"))
    new_content: str = Field(description=get_text("schemas.overwrite_file.content_desc"))
class FindReplaceSchema(BaseModel): # Uses renamed schema key from texts.py
    path: str = Field(description=get_text("schemas.find_replace.path_desc"))
    find_text: str = Field(description=get_text("schemas.find_replace.find_text_desc"))
    replace_text: str = Field(description=get_text("schemas.find_replace.replace_text_desc"))
    summary: str = Field(description=get_text("schemas.find_replace.summary_desc"))
class FromToSchema(BaseModel):
    from_path: str = Field(description=get_text("schemas.from_to.from_path_desc"))
    to_path: str = Field(description=get_text("schemas.from_to.to_path_desc"))
class RenameSchema(BaseModel):
    path: str = Field(description=get_text("schemas.rename.path_desc"))
    new_name: str = Field(description=get_text("schemas.rename.new_name_desc"))
class FindSchema(BaseModel):
    query: str = Field(description=get_text("schemas.find.query_desc"))
    directory: Optional[str] = Field(None, description=get_text("schemas.find.directory_desc"))

# --- Tool Classes ---

class FileManagerUsageGuideTool(BaseTool):
    name: str = get_text("tools.usage_guide.name")
    description: str = get_text("tools.usage_guide.description")
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        logger.debug(f"FileManagerUsageGuideTool invoked.")
        return FILES_TOOLSET_PROMPT

    async def _arun(self) -> str: return await run_in_executor(None, self._run)

class Create(BaseTool):
    name: str = get_text("tools.create.name")
    description: str = get_text("tools.create.description")
    args_schema: Type[BaseModel] = CreateSchema
    requires_confirmation: bool = False # Create is generally safe

    def _run(self, path: str, content: Optional[str] = None, is_directory: bool = False) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.create.error.no_chat")
        target_path = Path(path).resolve()
        try:
            if target_path.exists():
                return get_text("tools.create.error.exists", path=str(target_path))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if is_directory:
                if content:
                    return get_text("tools.create.error.content_for_dir", path=str(target_path))
                target_path.mkdir()
                op_type = "directory"
                log_data = {"operation": "create_dir", "path": str(target_path)}
            else:
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content or "")
                op_type = "file"
                log_data = {"operation": "create_file", "path": str(target_path)}
            _log_history_event(chat_id, log_data)
            return get_text("tools.create.success", type=op_type, path=str(target_path))
        except Exception as e:
            logger.error(f"Error creating path {target_path}: {e}", exc_info=True)
            return get_text("tools.create.error.generic", path=str(target_path), error=e)

    async def _arun(self, path: str, content: Optional[str] = None, is_directory: bool = False) -> str:
        return await run_in_executor(None, self._run, path, content, is_directory)

class Read(BaseTool):
    name: str = get_text("tools.read.name")
    description: str = get_text("tools.read.description")
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = False # Read is safe

    def _run(self, path: str) -> str:
        target_path = Path(path).resolve()
        max_len = 4000 # Max length to display in output
        try:
            if not target_path.is_file():
                return get_text("tools.read.error.not_file", path=str(target_path))
            content = target_path.read_text(encoding='utf-8', errors='replace')
            truncated_suffix = get_text("tools.read.truncated_suffix")
            display_content = (content[:max_len] + truncated_suffix) if len(content) > max_len else content
            return get_text("tools.read.success", path=str(target_path), content=display_content)
        except FileNotFoundError:
            return get_text("tools.read.error.not_found", path=str(target_path))
        except Exception as e:
            logger.error(f"Error reading file {target_path}: {e}", exc_info=True)
            return get_text("tools.read.error.generic", path=str(target_path), error=e)

    async def _arun(self, path: str) -> str: return await run_in_executor(None, self._run, path)

# --- NEW MERGED DELETE TOOL ---
class DeleteFileOrDir(BaseTool):
    name: str = get_text("tools.delete_file_or_dir.name")
    description: str = get_text("tools.delete_file_or_dir.description")
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = True # Delete is destructive

    def _run(self, path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.delete_file_or_dir.error.no_chat")
        target_path = Path(path).resolve()

        if confirmed_input is None:
            # Proposal phase: Check existence and raise PromptNeededError
            if not target_path.exists():
                return get_text("tools.delete_file_or_dir.error.not_exists", path=str(target_path))
            op_type = "directory" if target_path.is_dir() else "file"
            prompt_message = get_text("tools.delete_file_or_dir.confirm_prompt", type=op_type, path=str(target_path))
            logger.debug(f"Requesting confirmation to delete {op_type}: '{target_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"path": path}, # Pass original path arg
                edit_key="path", # User confirms the path they intend to delete
                prompt_suffix=get_text("tools.delete_file_or_dir.confirm_suffix")
            )
        else:
            # Execution phase: User confirmed the path
            final_path_str = confirmed_input.strip()
            if final_path_str != path:
                 logger.warning(f"User confirmed path '{final_path_str}' does not match proposed path '{path}'. Aborting delete.")
                 return get_text("tools.delete_file_or_dir.error.path_mismatch")

            final_target_path = Path(final_path_str).resolve()
            logger.info(f"Executing confirmed delete for: '{final_target_path}'")

            # Re-check existence *after* confirmation
            if not final_target_path.exists():
                logger.error(f"Target path {final_target_path} not found during delete confirmation.")
                return get_text("tools.delete_file_or_dir.error.target_missing", path=str(final_target_path))

            # Perform deletion
            log_data = {"path": str(final_target_path)}
            op_type = ""
            try:
                if final_target_path.is_dir():
                    shutil.rmtree(final_target_path)
                    op_type = "directory"
                    log_data["operation"] = "delete_dir_confirmed"
                elif final_target_path.is_file():
                    os.remove(final_target_path)
                    op_type = "file"
                    log_data["operation"] = "delete_file_confirmed"
                else:
                    raise OSError(f"Path exists but is neither file nor directory: {final_target_path}")

                _log_history_event(chat_id, log_data)
                return get_text("tools.delete_file_or_dir.success_confirm", type=op_type, path=str(final_target_path))

            except Exception as delete_e:
                 logger.error(f"Failed to delete {final_target_path}: {delete_e}", exc_info=True)
                 return get_text("tools.delete_file_or_dir.error.delete_failed", path=str(final_target_path), error=delete_e)

    async def _arun(self, path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, confirmed_input)


# --- NEW MERGED OVERWRITE TOOL ---
class OverwriteFile(BaseTool):
    name: str = get_text("tools.overwrite_file.name")
    description: str = get_text("tools.overwrite_file.description")
    args_schema: Type[BaseModel] = OverwriteSchema
    requires_confirmation: bool = True # Overwriting is destructive

    def _run(self, path: str, new_content: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.overwrite_file.error.no_chat")
        target_path = Path(path).resolve()

        if confirmed_input is None:
            # Proposal phase: Check file exists, show diff/content, raise PromptNeededError
            if not target_path.is_file():
                return get_text("tools.overwrite_file.error.not_file", path=str(target_path))
            try:
                original_content = target_path.read_text(encoding='utf-8', errors='replace')
                max_display_len = 500 # Limit displayed content length in prompt
                display_original = (original_content[:max_display_len] + "\n...") if len(original_content) > max_display_len else original_content
                display_new = (new_content[:max_display_len] + "\n...") if len(new_content) > max_display_len else new_content
                prompt_message = get_text("tools.overwrite_file.confirm_prompt",
                                          original_file_path=str(target_path),
                                          original_file_content=display_original,
                                          new_content=display_new)
                logger.debug(f"Requesting confirmation to overwrite file '{target_path}'")
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"path": path, "new_content": new_content},
                    edit_key="new_content", # User confirms the *new content*
                    prompt_suffix=get_text("tools.overwrite_file.confirm_suffix")
                )
            except Exception as e:
                logger.error(f"Error reading original file {target_path} for overwrite proposal: {e}", exc_info=True)
                return get_text("tools.overwrite_file.error.read_failed", path=str(target_path), error=e)
        else:
            # Execution phase: User confirmed the new_content
            final_new_content = confirmed_input

            # Re-check file existence
            if not target_path.is_file():
                logger.error(f"Target file {target_path} not found or not a file during overwrite confirmation.")
                return get_text("tools.overwrite_file.error.target_missing", path=str(target_path))

            # Create backup
            backup_path = None
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                backup_path = target_path.with_suffix(f"{target_path.suffix}.bak.{timestamp}")
                shutil.copy2(target_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            except Exception as backup_e:
                 logger.error(f"Failed to create backup for {target_path}: {backup_e}", exc_info=True)
                 return get_text("tools.overwrite_file.error.backup_failed", path=str(target_path), error=backup_e)

            # Write new content
            try:
                target_path.write_text(final_new_content, encoding='utf-8')
                logger.info(f"Successfully overwrote file '{target_path}'")
            except Exception as write_e:
                logger.error(f"Failed to write new content to {target_path}: {write_e}", exc_info=True)
                return get_text("tools.overwrite_file.error.write_failed", path=str(target_path), error=write_e)

            # Log success
            log_data = {
                "operation": "overwrite_confirmed",
                "path": str(target_path),
                "backup_path": str(backup_path) if backup_path else None,
            }
            _log_history_event(chat_id, log_data)
            return get_text("tools.overwrite_file.success_confirm", path=str(target_path), backup_path=str(backup_path))

    async def _arun(self, path: str, new_content: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, new_content, confirmed_input)

# --- NEW FIND AND REPLACE TOOL ---
class FindAndReplaceInFile(BaseTool):
    name: str = get_text("tools.find_and_replace_in_file.name")
    description: str = get_text("tools.find_and_replace_in_file.description")
    args_schema: Type[BaseModel] = FindReplaceSchema
    requires_confirmation: bool = True # Modifies file content

    def _run(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.find_and_replace_in_file.error.no_chat")
        target_path = Path(path).resolve()

        if confirmed_input is None:
            # Proposal phase: Check file, raise PromptNeededError confirming the summary
            if not target_path.is_file():
                return get_text("tools.find_and_replace_in_file.error.not_file", path=str(target_path))

            logger.debug(f"Requesting confirmation for find/replace in '{target_path}' with summary: {summary}")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"path": path, "find_text": find_text, "replace_text": replace_text, "summary": summary},
                edit_key="summary", # User confirms the summary
                prompt_suffix=get_text("tools.find_and_replace_in_file.confirm_suffix")
            )
        else:
            # Execution phase: User confirmed the summary
            final_summary = confirmed_input.strip()
            if not final_summary:
                 return get_text("tools.find_and_replace_in_file.error.empty_summary")

            logger.info(f"Executing confirmed find/replace on '{target_path}' with summary: {final_summary}")

            # Re-check file existence
            if not target_path.is_file():
                logger.error(f"Target file {target_path} not found or not a file during find/replace confirmation.")
                return get_text("tools.find_and_replace_in_file.error.target_missing", path=str(target_path))

            # Create backup
            backup_path = None
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                backup_path = target_path.with_suffix(f"{target_path.suffix}.bak.{timestamp}")
                shutil.copy2(target_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            except Exception as backup_e:
                 logger.error(f"Failed to create backup for {target_path}: {backup_e}", exc_info=True)
                 return get_text("tools.find_and_replace_in_file.error.backup_failed", path=str(target_path), error=backup_e)

            # Perform find/replace
            try:
                original_content = target_path.read_text(encoding='utf-8', errors='replace')
                new_content = original_content.replace(find_text, replace_text)

                if new_content == original_content:
                     try:
                         if backup_path: backup_path.unlink()
                     except OSError: pass
                     logger.info(f"Find/replace resulted in no changes for file '{target_path}'")
                     return get_text("tools.find_and_replace_in_file.info_no_change", path=str(target_path))

                target_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Successfully performed find/replace in file '{target_path}'")

            except Exception as write_e:
                logger.error(f"Failed to read or write file {target_path} during find/replace: {write_e}", exc_info=True)
                return get_text("tools.find_and_replace_in_file.error.write_failed", path=str(target_path), error=write_e)

            # Log success
            log_data = {
                "operation": "find_replace_confirmed",
                "path": str(target_path),
                "backup_path": str(backup_path) if backup_path else None,
                "summary": final_summary,
                "find": find_text,
                "replace": replace_text
            }
            _log_history_event(chat_id, log_data)
            return get_text("tools.find_and_replace_in_file.success_confirm", path=str(target_path), backup_path=str(backup_path))

    async def _arun(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, find_text, replace_text, summary, confirmed_input)

class Copy(BaseTool):
    name: str = get_text("tools.copy.name")
    description: str = get_text("tools.copy.description")
    args_schema: Type[BaseModel] = FromToSchema
    requires_confirmation: bool = True

    def _run(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.copy.error.no_chat")
        source_path_str = from_path

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to copy from '{from_path}' to '{to_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"from_path": from_path, "to_path": to_path},
                edit_key="from_path", # Confirm source
                prompt_suffix=get_text("tools.copy.confirm_suffix")
            )
        else:
            final_from_str = confirmed_input.strip()
            final_to_str = to_path
            source_path = Path(final_from_str).resolve()
            dest_path = Path(final_to_str).resolve()

            if str(source_path) != str(Path(from_path).resolve()):
                 logger.warning(f"Confirmed source path '{source_path}' differs from proposed '{from_path}'. Aborting copy.")
                 return get_text("tools.copy.error.path_mismatch")

            logger.info(f"Executing confirmed copy from '{source_path}' to '{dest_path}'")
            try:
                if not source_path.exists():
                    return get_text("tools.copy.error.source_not_exists", path=str(source_path))
                if dest_path.exists():
                    return get_text("tools.copy.error.dest_exists", path=str(dest_path))
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                log_data = {"from_path": str(source_path), "to_path": str(dest_path)}
                if source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=False)
                    op_type = "directory"
                    log_data["operation"] = "copy_dir_confirmed"
                else:
                    shutil.copy2(source_path, dest_path)
                    op_type = "file"
                    log_data["operation"] = "copy_file_confirmed"
                _log_history_event(chat_id, log_data)
                return get_text("tools.copy.success", type=op_type, from_path=str(source_path), to_path=str(dest_path))
            except Exception as e:
                logger.error(f"Error copying {source_path} to {dest_path}: {e}", exc_info=True)
                return get_text("tools.copy.error.generic", from_path=str(source_path), to_path=str(dest_path), error=e)

    async def _arun(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, from_path, to_path, confirmed_input)

class Move(BaseTool):
    name: str = get_text("tools.move.name")
    description: str = get_text("tools.move.description")
    args_schema: Type[BaseModel] = FromToSchema
    requires_confirmation: bool = True

    def _run(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.move.error.no_chat")
        source_path_str = from_path

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to move from '{from_path}' to '{to_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"from_path": from_path, "to_path": to_path},
                edit_key="from_path", # Confirm source
                prompt_suffix=get_text("tools.move.confirm_suffix")
            )
        else:
            final_from_str = confirmed_input.strip()
            final_to_str = to_path
            source_path = Path(final_from_str).resolve()
            dest_path = Path(final_to_str).resolve()

            if str(source_path) != str(Path(from_path).resolve()):
                 logger.warning(f"Confirmed source path '{source_path}' differs from proposed '{from_path}'. Aborting move.")
                 return get_text("tools.move.error.path_mismatch")

            logger.info(f"Executing confirmed move from '{source_path}' to '{dest_path}'")
            try:
                if not source_path.exists():
                    return get_text("tools.move.error.source_not_exists", path=str(source_path))
                if dest_path.exists():
                    return get_text("tools.move.error.dest_exists", path=str(dest_path))
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))
                log_data = { "operation": "move_confirmed", "from_path": str(source_path), "to_path": str(dest_path) }
                _log_history_event(chat_id, log_data)
                return get_text("tools.move.success", from_path=str(source_path), to_path=str(dest_path))
            except Exception as e:
                logger.error(f"Error moving {source_path} to {dest_path}: {e}", exc_info=True)
                return get_text("tools.move.error.generic", from_path=str(source_path), to_path=str(dest_path), error=e)

    async def _arun(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, from_path, to_path, confirmed_input)

class Rename(BaseTool):
    name: str = get_text("tools.rename.name")
    description: str = get_text("tools.rename.description")
    args_schema: Type[BaseModel] = RenameSchema
    requires_confirmation: bool = True

    def _run(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.rename.error.no_chat")
        original_path = Path(path).resolve()
        original_new_name = new_name.strip()

        if os.path.sep in original_new_name or (os.altsep and os.altsep in original_new_name):
            return get_text("tools.rename.error.invalid_new_name", new_name=original_new_name)
        if not original_new_name:
             return get_text("tools.rename.error.empty_new_name")

        if confirmed_input is None:
             logger.debug(f"Requesting confirmation to rename '{original_path}' to '{original_new_name}'")
             raise PromptNeededError(
                 tool_name=self.name,
                 proposed_args={"path": path, "new_name": original_new_name},
                 edit_key="new_name", # Confirm new name
                 prompt_suffix=get_text("tools.rename.confirm_suffix")
             )
        else:
            final_new_name = confirmed_input.strip()
            if not final_new_name: return get_text("tools.rename.error.empty_new_name")
            if os.path.sep in final_new_name or (os.altsep and os.altsep in final_new_name):
                 return get_text("tools.rename.error.invalid_new_name", new_name=final_new_name)

            logger.info(f"Executing confirmed rename of '{original_path}' to '{final_new_name}'")
            new_path = original_path.with_name(final_new_name)

            try:
                if not original_path.exists():
                    return get_text("tools.rename.error.path_not_exists", path=str(original_path))
                if new_path.exists():
                    return get_text("tools.rename.error.target_exists", path=str(new_path))
                original_path.rename(new_path)
                log_data = { "operation": "rename_confirmed", "from_path": str(original_path), "to_path": str(new_path) }
                _log_history_event(chat_id, log_data)
                return get_text("tools.rename.success", path=str(original_path), new_path=str(new_path))
            except Exception as e:
                logger.error(f"Error renaming {original_path} to {new_path}: {e}", exc_info=True)
                return get_text("tools.rename.error.generic", path=str(original_path), new_path=str(new_path), error=e)

    async def _arun(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, new_name, confirmed_input)

class Find(BaseTool):
    name: str = get_text("tools.find.name")
    description: str = get_text("tools.find.description")
    args_schema: Type[BaseModel] = FindSchema
    requires_confirmation: bool = False # Find is safe

    def _run(self, query: str, directory: Optional[str] = None) -> str:
        start_dir_path = Path(directory).resolve() if directory else Path.cwd()
        start_dir_str = str(start_dir_path)
        fuzzy_enabled = FIND_FUZZY_DEFAULT
        fuzzy_threshold = FIND_THRESHOLD_DEFAULT
        result_limit = FIND_LIMIT_DEFAULT
        try:
            matches, permission_warning = find_files_with_logic(
                pattern=query, directory=start_dir_path, glob_pattern="**/*",
                fuzzy=fuzzy_enabled, threshold=fuzzy_threshold, limit=result_limit
            )
            if matches is None:
                return permission_warning or get_text("tools.find.error.generic", error="Unknown error during search")
            if not matches:
                no_match_msg = get_text("tools.find.info_no_matches", query=query, directory=start_dir_str)
                return f"{no_match_msg}{f' ({permission_warning})' if permission_warning else ''}"
            else:
                relative_matches = []
                for p in matches:
                     try: relative_matches.append(str(p.relative_to(start_dir_path)))
                     except ValueError: relative_matches.append(str(p))
                matches_str = "\n".join(f"- {m}" for m in relative_matches)
                result_str = get_text("tools.find.success", count=len(matches), query=query, directory=start_dir_str, matches=matches_str)
                if len(matches) >= result_limit:
                    result_str += get_text("tools.find.info_limit_reached")
                if permission_warning:
                     result_str += f"\n\nWARNING: {permission_warning}"
                return result_str
        except Exception as e:
            logger.error(f"Unexpected error in Find tool execution for query '{query}' in '{start_dir_str}': {e}", exc_info=True)
            return get_text("tools.find.error.generic", error=e)

    async def _arun(self, query: str, directory: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, query, directory)

class CheckExist(BaseTool):
    name: str = get_text("tools.check_exist.name")
    description: str = get_text("tools.check_exist.description")
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = False # Check is safe

    def _run(self, path: str) -> str:
        target_path = Path(path) # Don't resolve here
        try:
            if target_path.exists():
                type_str = "directory" if target_path.is_dir() else "file" if target_path.is_file() else "special file"
                return get_text("tools.check_exist.success_exists", path=path, type=type_str)
            else:
                return get_text("tools.check_exist.success_not_exists", path=path)
        except Exception as e:
            logger.error(f"Error checking existence of path '{path}': {e}", exc_info=True)
            return get_text("tools.check_exist.error.generic", path=path, error=e)

    async def _arun(self, path: str) -> str: return await run_in_executor(None, self._run, path)

class ShowHistory(BaseTool):
    name: str = get_text("tools.history.name")
    description: str = get_text("tools.history.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    requires_confirmation: bool = False # Show history is safe

    def _run(self) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.history.error.no_chat")
        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            if not history: return get_text("tools.history.info_empty")
            limit = FILES_HISTORY_LIMIT
            try: # Attempt to read limit from config
                config_path = get_toolset_data_path(chat_id, toolset_id)
                tool_config = read_json(config_path, default_value=None)
                if tool_config is not None and "history_retrieval_limit" in tool_config:
                     limit = int(tool_config["history_retrieval_limit"])
                else: # Fallback to global config
                    global_config_path = Path(ROOT_DIR / "data" / "toolsets" / f"{toolset_id}.json")
                    global_config = read_json(global_config_path, default_value=None)
                    if global_config is not None and "history_retrieval_limit" in global_config:
                         limit = int(global_config["history_retrieval_limit"])
                if limit < 0: limit = 0
            except (ValueError, TypeError, FileNotFoundError) as e:
                logger.warning(f"Could not read history limit config for chat {chat_id}. Using default {FILES_HISTORY_LIMIT}. Error: {e}")
            recent_history = history[-limit:]
            total_history = len(history)
            actual_shown = len(recent_history)
            output = get_text("tools.history.header", count=actual_shown, total=total_history)
            for event in reversed(recent_history):
                ts = event.get('timestamp', 'Timestamp missing')
                op = event.get('operation', 'Unknown').replace("_confirmed", "") # Clean up op name
                path = event.get('path')
                from_p = event.get('from_path')
                to_p = event.get('to_path')
                backup = event.get('backup_path')
                summary = event.get('summary')
                find_q = event.get('find')
                replace_q = event.get('replace')
                ts_formatted = ""
                try: ts_formatted = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime('%Y-%m-%d %H:%M:%S')
                except: pass
                line = get_text("tools.history.line_format", ts=ts_formatted, op=op.upper())
                if path: line += f": {path}"
                if from_p: line += f": From {from_p}"
                if to_p: line += f" To {to_p}"
                if summary: line += f" (Summary: {summary})"
                if find_q: line += f" [Find: '{find_q[:20]}...']"
                if replace_q: line += f" [Replace: '{replace_q[:20]}...']"
                if backup: line += f" [Backup: {Path(backup).name}]"
                output += line + "\n"
            return output.strip()
        except Exception as e:
            logger.error(f"Error retrieving file history for chat {chat_id}: {e}", exc_info=True)
            return get_text("tools.history.error.generic", error=e)

    async def _arun(self) -> str: return await run_in_executor(None, self._run)

# --- RENAMED RESTORE TOOL ---
class RestoreFile(BaseTool):
    name: str = get_text("tools.restore_file.name")
    description: str = get_text("tools.restore_file.description")
    args_schema: Type[BaseModel] = RestorePathSchema
    requires_confirmation: bool = True # Restoring overwrites current file

    def _run(self, path_or_paths: Union[str, List[str]], backup_id: Optional[str] = None, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.restore_file.error.no_chat")
        target_paths_str = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
        if not target_paths_str: return get_text("tools.restore_file.error.no_paths")
        resolved_target_paths = [str(Path(p).resolve()) for p in target_paths_str]
        backup_filter = backup_id.strip() if backup_id else None
        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            backups_to_restore = {}
            errors = []
            found_backups_summary = []

            # Find backups
            for target_path_str in resolved_target_paths:
                latest_match = None; latest_match_ts = datetime.min.replace(tzinfo=timezone.utc)
                for event in reversed(history):
                    event_ts_str = event.get("timestamp")
                    event_path = event.get("path")
                    event_backup = event.get("backup_path")
                    event_op = event.get("operation")
                    is_edit_op = event_op in ["overwrite_confirmed", "find_replace_confirmed"]

                    if is_edit_op and event_path == target_path_str and event_backup:
                         if backup_filter and backup_filter not in event_backup: continue
                         try:
                             event_ts = datetime.fromisoformat(event_ts_str.replace("Z", "+00:00"))
                             if event_ts > latest_match_ts:
                                  if Path(event_backup).is_file():
                                       latest_match_ts = event_ts
                                       latest_match = event_backup
                                  else: logger.warning(f"Backup {event_backup} in history but not found. Skipping.")
                         except Exception as e: logger.debug(f"Timestamp parse error: {e}"); continue
                if latest_match:
                    backups_to_restore[target_path_str] = latest_match
                    found_backups_summary.append(f"'{target_path_str}' from backup '{Path(latest_match).name}'")
                else:
                    filter_msg = f" matching '{backup_filter}'" if backup_filter else ""
                    errors.append(get_text("tools.restore_file.error.no_backup_found", path=target_path_str, filter=filter_msg))

            if not backups_to_restore:
                return "\n".join(errors) if errors else get_text("tools.restore_file.error.no_valid_backups")

            proposal_summary = "Propose restoring:\n" + "\n".join(f"- {s}" for s in found_backups_summary)
            if errors: proposal_summary += "\n\nErrors finding backups:\n" + "\n".join(f"- {e}" for e in errors)

            if confirmed_input is None:
                logger.debug(f"Requesting confirmation for restore:\n{proposal_summary}")
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={ "path_or_paths": path_or_paths, "backup_id": backup_id, "summary": proposal_summary },
                    edit_key="summary", # Confirm summary
                    prompt_suffix=get_text("tools.restore_file.confirm_suffix")
                )
            else:
                if not confirmed_input or confirmed_input.lower().strip() in ["no", "cancel", "revert"]:
                    logger.info("Restore confirmation denied by user.")
                    return get_text("tools.restore_file.info_cancel")

                logger.info(f"Executing confirmed restore for {len(backups_to_restore)} files.")
                results = []
                success_count = 0
                for target_path_str, backup_path_str in backups_to_restore.items():
                    target_path = Path(target_path_str); backup_path = Path(backup_path_str)
                    try:
                        if not backup_path.is_file(): raise FileNotFoundError(f"Backup disappeared: {backup_path}")
                        if target_path.exists() and not target_path.is_file(): raise IsADirectoryError(f"Target not file: {target_path}")
                        shutil.copy2(backup_path, target_path)
                        success_count += 1
                        results.append(get_text("tools.restore_file.success_single", path=str(target_path), backup_name=backup_path.name))
                        _log_history_event(chat_id, { "operation": "restore_confirmed", "path": str(target_path), "backup_path": str(backup_path) })
                    except Exception as e:
                        logger.error(f"Failed to restore {target_path} from {backup_path}: {e}", exc_info=True)
                        results.append(get_text("tools.restore_file.error.single_failed", path=str(target_path), error=e))

                final_message = f"Restore attempt finished for {len(backups_to_restore)} file(s). Success: {success_count}.\n" + "\n".join(results)
                if errors: final_message += "\n\nErrors finding backups:\n" + "\n".join(f"- {e}" for e in errors)
                return final_message.strip()
        except Exception as e:
            logger.error(f"Unexpected error during restore process: {e}", exc_info=True)
            return get_text("tools.restore_file.error.generic", error=e)

    async def _arun(self, path_or_paths: Union[str, List[str]], backup_id: Optional[str] = None, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path_or_paths, backup_id, confirmed_input)

# --- RENAMED CLEANUP TOOL ---
class CleanupFileBackups(BaseTool):
    name: str = get_text("tools.cleanup_file_backups.name")
    description: str = get_text("tools.cleanup_file_backups.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    requires_confirmation: bool = True

    def _run(self, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.cleanup_file_backups.error.no_chat")
        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            backup_paths_to_delete = set()
            for event in history:
                is_edit_op = event.get("operation") in ["overwrite_confirmed", "find_replace_confirmed"]
                if is_edit_op and event.get("backup_path"):
                    backup_paths_to_delete.add(event["backup_path"])

            num_backups = len(backup_paths_to_delete)
            confirm_prompt = get_text("tools.cleanup_file_backups.prompt_confirm", count=num_backups) if num_backups > 0 else get_text("tools.cleanup_file_backups.prompt_confirm_no_backups")

            if confirmed_input is None:
                if num_backups == 0 and not history:
                     return get_text("tools.cleanup_file_backups.info_no_work")
                logger.debug(f"Requesting confirmation: '{confirm_prompt}' for chat {chat_id}.")
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"confirm": confirm_prompt},
                    edit_key="confirm", # Confirm the prompt message
                    prompt_suffix=get_text("tools.cleanup_file_backups.confirm_suffix")
                )
            else:
                if confirmed_input.lower().strip() != 'yes' and confirmed_input.strip() != confirm_prompt:
                    logger.info("Cleanup confirmation denied.")
                    return get_text("tools.cleanup_file_backups.info_cancel")

                logger.info(f"Executing confirmed cleanup for chat {chat_id}.")
                deleted_count = 0; errors = []; delete_success = True
                if num_backups > 0:
                    logger.info(f"Deleting {num_backups} backup files...")
                    for backup_path_str in backup_paths_to_delete:
                        try:
                            backup_path = Path(backup_path_str)
                            if backup_path.is_file(): backup_path.unlink()
                            else: logger.warning(f"Backup not found/not file: {backup_path}")
                            deleted_count += 1 # Increment even if not found, as it's cleared from history
                        except Exception as e:
                            logger.error(f"Error deleting backup {backup_path_str}: {e}")
                            errors.append(backup_path_str); delete_success = False

                result_msg = ""
                if delete_success:
                    try:
                        current_state = _read_toolset_state(chat_id)
                        current_state["history"] = [] # Clear history
                        if _write_toolset_state(chat_id, current_state):
                            logger.info(f"Cleared file history for chat {chat_id}.")
                            result_msg = get_text("tools.cleanup_file_backups.success", deleted_count=deleted_count, total_count=num_backups)
                        else:
                            logger.error(f"Failed to write state after clearing history for chat {chat_id}.")
                            result_msg = get_text("tools.cleanup_file_backups.warn_delete_success_history_fail", deleted_count=deleted_count, total_count=num_backups)
                            if errors: result_msg += get_text("tools.cleanup_file_backups.warn_errors_suffix", errors=', '.join(errors))
                    except Exception as e:
                        logger.error(f"Error clearing history/writing state for chat {chat_id}: {e}", exc_info=True)
                        result_msg = get_text("tools.cleanup_file_backups.error_clear_history_failed", deleted_count=deleted_count, total_count=num_backups, error=e)
                        if errors: result_msg += get_text("tools.cleanup_file_backups.warn_errors_suffix", errors=', '.join(errors))
                else:
                    logger.warning(f"Backup deletion failed. History NOT cleared for chat {chat_id}.")
                    result_msg = get_text("tools.cleanup_file_backups.warn_delete_failed_history_kept", deleted_count=deleted_count, total_count=num_backups)
                    result_msg += get_text("tools.cleanup_file_backups.warn_errors_suffix", errors=', '.join(errors))

                return result_msg
        except Exception as e:
            logger.error(f"Error finalizing file changes for chat {chat_id}: {e}", exc_info=True)
            return get_text("tools.cleanup_file_backups.error.generic", error=e)

    async def _arun(self, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, confirmed_input)


# --- Instantiate Tools (Updated List) ---
file_manager_usage_guide_tool = FileManagerUsageGuideTool()
create_tool = Create()
read_tool = Read()
delete_tool = DeleteFileOrDir() # New merged tool
overwrite_file_tool = OverwriteFile() # New merged tool
find_replace_tool = FindAndReplaceInFile() # New merged tool
copy_tool = Copy()
move_tool = Move()
rename_tool = Rename()
find_tool = Find()
exists_tool = CheckExist()
history_tool = ShowHistory()
restore_tool = RestoreFile() # Renamed tool
cleanup_tool = CleanupFileBackups() # Renamed tool

# --- Define Toolset Structure (Updated List) ---
toolset_tools: List[BaseTool] = [
    file_manager_usage_guide_tool,
    create_tool,
    read_tool,
    delete_tool, # Merged
    overwrite_file_tool, # Merged/Renamed
    find_replace_tool, # Merged/Renamed
    copy_tool,
    move_tool,
    rename_tool,
    find_tool,
    exists_tool,
    history_tool,
    restore_tool, # Renamed
    cleanup_tool, # Renamed
]

# --- Register Tools ---
# Ensure tools are registered using the updated instances
register_tools(toolset_tools)
logger.debug(f"Registered File Manager toolset ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")