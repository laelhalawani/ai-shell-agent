"""
File Manager toolset: Provides tools for direct file and directory manipulation.
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
from ... import logger
from ...tool_registry import register_tools
from ...utils.file_io import read_json, write_json
from ...utils.env import ensure_dotenv_key
from ...errors import PromptNeededError
from ...console_manager import get_console_manager
from ...chat_state_manager import (
    get_current_chat,
    get_toolset_data_path,
    check_and_configure_toolset # Keep import if needed elsewhere
)
from ...texts import get_text as get_main_text
from .settings import (
    FILES_HISTORY_LIMIT, FIND_FUZZY_DEFAULT, FIND_THRESHOLD_DEFAULT,
    FIND_LIMIT_DEFAULT, FIND_WORKERS_DEFAULT
) # Import all settings variables
from .prompts import FILES_TOOLSET_PROMPT
from .texts import get_text # Keep toolset-specific get_text
from .integration.find_logic import find_files_with_logic # Import the new find_files_with_logic function

console = get_console_manager()

# --- Toolset Metadata ---
toolset_id = "files"
toolset_name = get_text("toolset.name")
toolset_description = get_text("toolset.description")
toolset_required_secrets: Dict[str, str] = {}

# --- State Management Helpers ---
def _get_history_path(chat_id: str) -> Path:
    # Renaming slightly for clarity as it now holds more than just history
    return get_toolset_data_path(chat_id, toolset_id)

def _read_toolset_state(chat_id: str) -> Dict:
    state_path = _get_history_path(chat_id)
    # Ensure default includes 'history', 'pending_edit' and 'pending_delete' keys
    default = {"history": [], "pending_edit": None, "pending_delete": None}
    state = read_json(state_path, default_value=default)
    # Ensure keys exist even if file existed but was missing them
    if "history" not in state: state["history"] = []
    if "pending_edit" not in state: state["pending_edit"] = None
    if "pending_delete" not in state: state["pending_delete"] = None
    return state

def _write_toolset_state(chat_id: str, state_data: Dict) -> bool:
    state_path = _get_history_path(chat_id)
    # Ensure all potential keys are present before writing if modifying partially
    state_data.setdefault("history", [])
    state_data.setdefault("pending_edit", None)
    state_data.setdefault("pending_delete", None)
    return write_json(state_path, state_data) # write_json returns bool

def _log_history_event(chat_id: str, event_data: Dict) -> None:
    if not chat_id: return
    try:
        state = _read_toolset_state(chat_id) # Use the new reader
        # History processing remains the same
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        state["history"].append(event_data)
        limit = FILES_HISTORY_LIMIT # Use loaded setting
        if limit > 0 and len(state["history"]) > limit:
             state["history"] = state["history"][-limit:]
        if not _write_toolset_state(chat_id, state): # Use the new writer
            logger.error(f"Failed to write state after logging history event for chat {chat_id}")
        else:
            logger.debug(f"Logged file history event for chat {chat_id}: {event_data.get('operation')}")
    except Exception as e:
        logger.error(f"Failed to log file history event for chat {chat_id}: {e}", exc_info=True)

# --- Configuration Function ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path], # Keep Optional for standard signature
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
) -> Dict:
    """
    Configuration function for the File Manager toolset.
    Prompts user for history retrieval limit, using defaults from settings.
    """
    # --- Determine context and print header ---
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat" # Or fetch actual chat title if needed

    # Print the main configuration header using console manager
    console.display_message(
        get_main_text("common.labels.system"),
        get_text("config.header"), 
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )

    # Use the config provided by configure_toolset_cli which could be global or local
    config_to_prompt = current_chat_config if current_chat_config is not None else {}
    final_config = {}

    try:
        # Determine the default value based on the current config or the setting default
        default_limit = config_to_prompt.get("history_retrieval_limit", FILES_HISTORY_LIMIT)

        # Prompt using console manager - text key already updated to remove colon
        limit_str = console.prompt_for_input(
            get_text("config.prompt_limit"),
            default=str(default_limit) # Pass default to prompt_for_input
        ).strip()

        try:
            limit = int(limit_str) if limit_str else default_limit
            if limit < 0: limit = 0
            final_config["history_retrieval_limit"] = limit
        except ValueError:
            # Use console manager for warning
            console.display_message(
                get_main_text("common.labels.warning"),
                get_text("config.warn_invalid"),
                console.STYLE_WARNING_LABEL,
                console.STYLE_WARNING_CONTENT
            )
            final_config["history_retrieval_limit"] = default_limit

    except (KeyboardInterrupt, EOFError):
        # Message handled by prompt_for_input
        # Just return the existing config without changes
        console.display_message(
            get_main_text("common.labels.warning"),
            get_text("config.warn_cancel"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}
    except Exception as e:
        logger.error(f"Error during File Manager configuration: {e}", exc_info=True)
        # Use console manager for error
        console.display_message(
            get_main_text("common.labels.error"),
            get_text("config.error_generic", error=e),
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )
        # Return original config on error
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}

    save_success_global = True
    save_success_local = True

    # Save to global path
    try:
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(global_config_path, final_config)
        logger.info(f"File Manager configuration saved to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to save File Manager config to global path {global_config_path}: {e}")

    # Save to local path if it exists (i.e., not global-only context)
    if local_config_path:
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.info(f"File Manager configuration saved to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to save File Manager config to local path {local_config_path}: {e}")

    # Display final status message using console manager
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
class RestorePathSchema(BaseModel):
    path_or_paths: Union[str, List[str]] = Field(description=get_text("schemas.restore.paths_desc"))
    backup_id: Optional[str] = Field(None, description=get_text("schemas.restore.backup_id_desc"))
class CreateSchema(BaseModel):
    path: str = Field(description=get_text("schemas.create.path_desc"))
    content: Optional[str] = Field(None, description=get_text("schemas.create.content_desc"))
    is_directory: bool = Field(False, description=get_text("schemas.create.is_directory_desc"))
class EditSchemaNew(BaseModel): # This schema is used by the remaining Edit tool
    path: str = Field(description=get_text("schemas.edit_new.path_desc"))
    new_content: str = Field(description=get_text("schemas.edit_new.content_desc"))
class ConfirmationSchema(BaseModel): # Used by ConfirmEdit
    confirmation: str = Field(description=get_text("schemas.confirm_edit.confirmation_desc"))
class ConfirmDeleteSchema(BaseModel): # Used by ConfirmDelete
    confirmation: str = Field(description=get_text("schemas.confirm_delete.confirmation_desc"))
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

    def _run(self, path: str) -> str:
        target_path = Path(path).resolve()
        max_len = 4000

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

class ProposeDelete(BaseTool): 
    name: str = get_text("tools.delete.name")
    description: str = get_text("tools.delete.description")
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = False # Proposes only

    def _run(self, path: str) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.delete.error.no_chat")
        target_path = Path(path).resolve()

        try:
            # 1. Check existence
            if not target_path.exists():
                return get_text("tools.delete.error.not_exists", path=str(target_path))

            # 2. Check for other pending actions
            state = _read_toolset_state(chat_id)
            if state.get("pending_edit") or state.get("pending_delete"):
                logger.warning(f"Attempted to propose delete for {target_path} while another action is pending.")
                return get_text("tools.delete.error.pending_action")

            # 3. Store pending delete
            pending_data = {"path": str(target_path)}
            state["pending_delete"] = pending_data

            # 4. Write updated state
            if not _write_toolset_state(chat_id, state):
                logger.error(f"Failed to write pending delete state for chat {chat_id}, path {target_path}")
                state["pending_delete"] = None # Clean up memory state
                return get_text("tools.delete.error.state_write_failed", path=str(target_path))

            # 5. Return confirmation prompt
            op_type = "directory" if target_path.is_dir() else "file"
            logger.info(f"Proposed deletion for {op_type} {target_path}, pending confirmation.")
            return get_text(
                "tools.delete.msg_confirm_prompt",
                type=op_type,
                path=str(target_path)
            )

        except Exception as e:
            logger.error(f"Error proposing deletion for path {target_path}: {e}", exc_info=True)
            return get_text("tools.delete.error.generic", path=str(target_path), error=e)

    async def _arun(self, path: str) -> str:
        return await run_in_executor(None, self._run, path)

class Edit(BaseTool): # Renamed from EditNew
    name: str = get_text("tools.edit.name")
    description: str = get_text("tools.edit.description")
    args_schema: Type[BaseModel] = EditSchemaNew # Use the correct schema
    requires_confirmation: bool = False # This tool only proposes

    def _run(self, path: str, new_content: str) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.edit.error.no_chat")
        target_path = Path(path).resolve()
        try:
            if not target_path.is_file():
                return get_text("tools.edit.error.not_file", path=str(target_path))
            try:
                original_content = target_path.read_text(encoding='utf-8', errors='replace')
                max_display_len = 500
                display_original = (original_content[:max_display_len] + "\n...") if len(original_content) > max_display_len else original_content
                display_new = (new_content[:max_display_len] + "\n...") if len(new_content) > max_display_len else new_content
            except Exception as e:
                logger.error(f"Error reading original file {target_path} for edit proposal: {e}", exc_info=True)
                return get_text("tools.edit.error.read_failed", path=str(target_path), error=e)
            state = _read_toolset_state(chat_id)
            if state.get("pending_edit"):
                logger.warning(f"Attempted to propose edit for {target_path} while another edit is pending.")
                return get_text("tools.edit.error.pending_edit_exists")
            pending_data = {"path": str(target_path), "new_content": new_content}
            state["pending_edit"] = pending_data
            if not _write_toolset_state(chat_id, state):
                logger.error(f"Failed to write pending edit state for chat {chat_id}, path {target_path}")
                state["pending_edit"] = None
                return get_text("tools.edit.error.state_write_failed", path=str(target_path))
            logger.info(f"Proposed edit for {target_path}, pending confirmation.")
            return get_text(
                "tools.edit.msg_confirm_prompt",
                original_file_path=str(target_path),
                original_file_content=display_original,
                new_content=display_new
            )
        except Exception as e:
            logger.error(f"Unexpected error proposing edit for file {target_path}: {e}", exc_info=True)
            return get_text("tools.edit.error.generic", path=str(target_path), error=e)

    async def _arun(self, path: str, new_content: str) -> str:
        return await run_in_executor(None, self._run, path, new_content)

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
                edit_key="from_path"
            )
        else:
            final_from_str = confirmed_input
            final_to_str = to_path
            source_path = Path(final_from_str).resolve()
            dest_path = Path(final_to_str).resolve()
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
                    log_data["operation"] = "copy_dir"
                else:
                    shutil.copy2(source_path, dest_path)
                    op_type = "file"
                    log_data["operation"] = "copy_file"

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
                edit_key="from_path"
            )
        else:
            final_from_str = confirmed_input
            final_to_str = to_path
            source_path = Path(final_from_str).resolve()
            dest_path = Path(final_to_str).resolve()
            logger.info(f"Executing confirmed move from '{source_path}' to '{dest_path}'")

            try:
                if not source_path.exists():
                    return get_text("tools.move.error.source_not_exists", path=str(source_path))
                if dest_path.exists():
                    return get_text("tools.move.error.dest_exists", path=str(dest_path))

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))

                log_data = {
                    "operation": "move",
                    "from_path": str(source_path),
                    "to_path": str(dest_path)
                 }
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
        original_new_name = new_name

        if confirmed_input is None:
             if os.path.sep in new_name or (os.altsep and os.altsep in new_name):
                  return get_text("tools.rename.error.invalid_new_name", new_name=new_name)
             logger.debug(f"Requesting confirmation to rename '{original_path}' to '{new_name}'")
             raise PromptNeededError(
                 tool_name=self.name,
                 proposed_args={"path": path, "new_name": new_name},
                 edit_key="new_name"
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

                log_data = {
                    "operation": "rename",
                    "from_path": str(original_path),
                    "to_path": str(new_path)
                }
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

    def _run(self, query: str, directory: Optional[str] = None) -> str:
        start_dir_path = Path(directory).resolve() if directory else Path.cwd()
        start_dir_str = str(start_dir_path) # For use in get_text and logging

        # Load settings for the find operation
        fuzzy_enabled = FIND_FUZZY_DEFAULT
        fuzzy_threshold = FIND_THRESHOLD_DEFAULT
        result_limit = FIND_LIMIT_DEFAULT

        try:
            # Call the new find logic from the integration module
            matches, permission_warning = find_files_with_logic(
                pattern=query,
                directory=start_dir_path,
                glob_pattern="**/*", # Hardcoded glob pattern for now
                fuzzy=fuzzy_enabled,
                threshold=fuzzy_threshold,
                limit=result_limit
            )

            # Handle potential errors from find_files_with_logic
            if matches is None:
                # Error occurred, permission_warning contains the error message
                return permission_warning or get_text("tools.find.error.generic", error="Unknown error during search")

            # Process results
            if not matches:
                no_match_msg = get_text("tools.find.info_no_matches", query=query, directory=start_dir_str)
                return f"{no_match_msg}{f' ({permission_warning})' if permission_warning else ''}"
            else:
                # Convert Path objects to relative strings for display
                relative_matches = []
                for p in matches:
                     try:
                          relative_matches.append(str(p.relative_to(start_dir_path)))
                     except ValueError: # Handle cases like different drives on Windows
                          relative_matches.append(str(p))

                matches_str = "\n".join(f"- {m}" for m in relative_matches)
                result_str = get_text("tools.find.success", count=len(matches), query=query, directory=start_dir_str, matches=matches_str)

                # Check if the result was limited
                if len(matches) >= result_limit:
                    result_str += get_text("tools.find.info_limit_reached")

                # Append permission warning if it occurred
                if permission_warning:
                     result_str += f"\n\nWARNING: {permission_warning}"

                return result_str

        except Exception as e:
            # Catch any unexpected errors here
            logger.error(f"Unexpected error in Find tool execution for query '{query}' in '{start_dir_str}': {e}", exc_info=True)
            return get_text("tools.find.error.generic", error=e)

    async def _arun(self, query: str, directory: Optional[str] = None) -> str:
        # Run the synchronous _run method in an executor
        return await run_in_executor(None, self._run, query, directory)

class CheckExist(BaseTool):
    name: str = get_text("tools.check_exist.name")
    description: str = get_text("tools.check_exist.description")
    args_schema: Type[BaseModel] = PathSchema

    def _run(self, path: str) -> str:
        target_path = Path(path)
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

    def _run(self) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.history.error.no_chat")

        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])

            if not history:
                return get_text("tools.history.info_empty")

            limit = FILES_HISTORY_LIMIT
            try:
                config_path = get_toolset_data_path(chat_id, toolset_id)
                tool_config = read_json(config_path, default_value=None)
                if tool_config is not None and "history_retrieval_limit" in tool_config:
                     limit = int(tool_config["history_retrieval_limit"])
                else:
                    global_config_path = Path(f"data/toolsets/{toolset_id}.json")
                    global_config = read_json(global_config_path, default_value=None)
                    if global_config is not None and "history_retrieval_limit" in global_config:
                         limit = int(global_config["history_retrieval_limit"])

                if limit < 0: limit = 0

            except (ValueError, TypeError, FileNotFoundError) as e:
                logger.warning(f"Could not read/parse history limit from config for chat {chat_id}. Using default from settings ({FILES_HISTORY_LIMIT}). Error: {e}")

            recent_history = history[-limit:]
            total_history = len(history)
            actual_shown = len(recent_history)

            output = get_text("tools.history.header", count=actual_shown, total=total_history)
            for event in reversed(recent_history):
                ts = event.get('timestamp', 'Timestamp missing')
                op = event.get('operation', 'Unknown operation')
                path = event.get('path')
                from_p = event.get('from_path')
                to_p = event.get('to_path')
                backup = event.get('backup_path')
                summary = event.get('summary')

                ts_formatted = ""
                try:
                    ts_formatted = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime('%Y-%m-%d %H:%M:%S')
                except: pass

                line = get_text("tools.history.line_format", ts=ts_formatted, op=op.upper())
                if path: line += f": {path}" # Keep details appended for now
                if from_p: line += f": From {from_p}"
                if to_p: line += f" To {to_p}"
                if summary: line += f" (Summary: {summary})"
                if backup: line += f" [Backup: {backup}]"
                output += line + "\n"

            return output.strip()

        except Exception as e:
            logger.error(f"Error retrieving file history for chat {chat_id}: {e}", exc_info=True)
            return get_text("tools.history.error.generic", error=e)

    async def _arun(self) -> str: return await run_in_executor(None, self._run)

class RestoreFromBackup(BaseTool):
    name: str = get_text("tools.restore.name")
    description: str = get_text("tools.restore.description")
    args_schema: Type[BaseModel] = RestorePathSchema
    requires_confirmation: bool = True

    def _run(self, path_or_paths: Union[str, List[str]], backup_id: Optional[str] = None, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.restore.error.no_chat")

        target_paths_str = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
        if not target_paths_str: return "Error: No paths provided for restoration." # Basic validation

        resolved_target_paths = [str(Path(p).resolve()) for p in target_paths_str]
        backup_filter = backup_id.strip() if backup_id else None

        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            backups_to_restore = {} # Dict[target_path_str, backup_path_str]
            errors = []
            found_backups_summary = [] # List of strings describing found backups

            # Find the latest matching backup for each requested path
            for target_path_str in resolved_target_paths:
                latest_match = None
                latest_match_ts = datetime.min.replace(tzinfo=timezone.utc)

                for event in reversed(history):
                    event_ts_str = event.get("timestamp")
                    event_path = event.get("path")
                    event_backup = event.get("backup_path")
                    event_op = event.get("operation")

                    # Check if it's a confirmed edit for the correct path with a backup recorded
                    if (event_op == "edit_confirmed" or event_op == "edit") and event_path == target_path_str and event_backup:
                         # Check backup_id filter if provided
                         if backup_filter and backup_filter not in event_backup:
                             continue # Filter mismatch

                         # Check timestamp
                         try:
                             event_ts = datetime.fromisoformat(event_ts_str.replace("Z", "+00:00"))
                             if event_ts > latest_match_ts:
                                  # Verify backup file exists *now*
                                  if Path(event_backup).is_file():
                                       latest_match_ts = event_ts
                                       latest_match = event_backup
                                  else:
                                       logger.warning(f"Backup file {event_backup} found in history for {target_path_str} but does not exist on disk. Skipping.")
                         except Exception as e: 
                             logger.debug(f"Error parsing timestamp for backup event: {e}")
                             continue # Ignore events with bad timestamps

                if latest_match:
                    backups_to_restore[target_path_str] = latest_match
                    found_backups_summary.append(f"'{target_path_str}' from backup '{Path(latest_match).name}'")
                else:
                    filter_msg = f" matching '{backup_filter}'" if backup_filter else ""
                    errors.append(get_text("tools.restore.error.no_backup_found", path=target_path_str, filter=filter_msg))

            if not backups_to_restore:
                return "\n".join(errors) if errors else get_text("tools.restore.error.no_valid_backups")

            # --- Confirmation Step ---
            proposal_summary = "Propose restoring:\n" + "\n".join(f"- {s}" for s in found_backups_summary)
            if errors: proposal_summary += "\n\nErrors finding backups:\n" + "\n".join(f"- {e}" for e in errors)

            if confirmed_input is None:
                logger.debug(f"Requesting confirmation for restore actions:\n{proposal_summary}")
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={
                        "path_or_paths": path_or_paths, # Pass original args
                        "backup_id": backup_id,
                        "summary": proposal_summary # Include summary for confirmation prompt context
                    },
                    edit_key="summary", # User confirms the summary of actions
                    prompt_suffix=get_text("tools.restore.prompt_suffix")
                )
            else:
                # Check if user confirmed the proposal
                if not confirmed_input or confirmed_input.lower().strip() in ["no", "cancel", "revert"]:
                    logger.info("Restore confirmation denied by user.")
                    return "Restore operation cancelled."

                logger.info(f"Executing confirmed restore for {len(backups_to_restore)} files.")
                results = []
                success_count = 0

                for target_path_str, backup_path_str in backups_to_restore.items():
                    target_path = Path(target_path_str)
                    backup_path = Path(backup_path_str)
                    try:
                        # Re-check target and backup just before restore
                        if not backup_path.is_file():
                            raise FileNotFoundError(f"Backup file disappeared: {backup_path}")
                        if target_path.exists() and not target_path.is_file():
                            raise IsADirectoryError(f"Target path exists but is not a file: {target_path}")

                        # Perform restore
                        shutil.copy2(backup_path, target_path)
                        success_count += 1
                        results.append(get_text("tools.restore.success_single", path=str(target_path), backup_name=backup_path.name))

                        # Log individual restore event
                        _log_history_event(chat_id, {
                            "operation": "restore_confirmed",
                            "path": str(target_path),
                            "backup_path": str(backup_path)
                        })
                    except Exception as e:
                        logger.error(f"Failed to restore {target_path} from {backup_path}: {e}", exc_info=True)
                        results.append(get_text("tools.restore.error.single_failed", path=str(target_path), error=e))

                final_message = f"Restore attempt finished for {len(backups_to_restore)} file(s). Success: {success_count}.\n" + "\n".join(results)
                # Append original errors if any backups weren't found initially
                if errors: final_message += "\n\nErrors finding backups:\n" + "\n".join(f"- {e}" for e in errors)
                return final_message.strip()

        except Exception as e:
            logger.error(f"Unexpected error during restore process: {e}", exc_info=True)
            return get_text("tools.restore.error.generic", error=e)

    async def _arun(self, path_or_paths: Union[str, List[str]], backup_id: Optional[str] = None, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path_or_paths, backup_id, confirmed_input)

class FinalizeChanges(BaseTool):
    name: str = get_text("tools.finalize.name")
    description: str = get_text("tools.finalize.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    requires_confirmation: bool = True

    def _run(self, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.finalize.error.no_chat")

        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            backup_paths_to_delete = set()

            # Find all unique backup paths from history
            for event in history:
                if event.get("backup_path") and \
                   (event.get("operation") == "edit" or event.get("operation") == "edit_confirmed"):
                    backup_paths_to_delete.add(event["backup_path"])

            num_backups = len(backup_paths_to_delete)

            if num_backups == 0:
                # Also check if history is empty - if both are empty, nothing to do
                if not history:
                    return "No backup files or history entries found to finalize."
                else:
                    # If history exists but no backups, ask only about clearing history
                    confirm_prompt = "No backup files found. Clear file history for this chat?"
            else:
                # If backups exist, use the text from JSON
                confirm_prompt = get_text("tools.finalize.prompt_confirm", count=num_backups)

            if confirmed_input is None:
                # Confirmation prompt text now depends on whether backups were found
                logger.debug(f"Requesting confirmation: '{confirm_prompt}' for chat {chat_id}.")
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"confirm": confirm_prompt}, # Show appropriate prompt
                    edit_key="confirm", # Have user confirm the prompt message
                    prompt_suffix=get_text("tools.finalize.prompt_suffix")
                )
            else:
                # Check confirmation against the expected prompt OR 'yes'
                if confirmed_input.lower().strip() != 'yes' and confirmed_input.strip() != confirm_prompt:
                    logger.info("Finalize confirmation denied.")
                    return get_text("tools.finalize.info_cancel")

                logger.info(f"Executing confirmed finalization for chat {chat_id}.")
                deleted_count = 0
                errors = []
                delete_success = True # Track if deletion had errors

                # Delete backups first
                if num_backups > 0:
                    logger.info(f"Deleting {num_backups} backup files...")
                    for backup_path_str in backup_paths_to_delete:
                        try:
                            backup_path = Path(backup_path_str)
                            if backup_path.exists() and backup_path.is_file():
                                os.remove(backup_path)
                                deleted_count += 1
                            else:
                                logger.warning(f"Backup file not found or not a file during finalize: {backup_path}")
                        except Exception as e:
                            logger.error(f"Error deleting backup {backup_path_str}: {e}")
                            errors.append(backup_path_str)
                            delete_success = False # Mark failure

                result_msg = ""
                # Clear history ONLY if deletion was successful
                if delete_success:
                    try:
                        # Read state again before modifying history
                        current_state = _read_toolset_state(chat_id)
                        current_state["history"] = [] # Clear the history list
                        # Keep pending_edit/pending_delete as they might exist if finalize was called mid-action
                        
                        if _write_toolset_state(chat_id, current_state):
                            logger.info(f"Successfully cleared file history for chat {chat_id}.")
                            result_msg = get_text("tools.finalize.success", deleted_count=deleted_count, total_count=num_backups)
                        else:
                            logger.error(f"Failed to write state after clearing history for chat {chat_id}.")
                            # Report deletion success but state failure
                            result_msg = f"Deleted {deleted_count} of {num_backups} backup files, but failed to clear history state. Please check logs."
                            # Append errors if they occurred during delete attempt before state save failed
                            if errors:
                                result_msg += get_text("tools.finalize.warn_errors", errors=', '.join(errors))
                    except Exception as e:
                        logger.error(f"Error clearing history or writing state for chat {chat_id}: {e}", exc_info=True)
                        # Report deletion success but history clear failure
                        result_msg = f"Deleted {deleted_count} of {num_backups} backup files, but failed to clear history: {e}"
                        if errors:
                            result_msg += get_text("tools.finalize.warn_errors", errors=', '.join(errors))
                else:
                    # Deletion failed, do not clear history
                    logger.warning(f"Backup deletion encountered errors. History for chat {chat_id} will NOT be cleared.")
                    result_msg = f"Finalized file changes attempt finished. Deleted {deleted_count} of {num_backups} backup files."
                    result_msg += get_text("tools.finalize.warn_errors", errors=', '.join(errors)) # Report errors

                return result_msg

        except Exception as e:
            logger.error(f"Error finalizing file changes for chat {chat_id}: {e}", exc_info=True)
            return get_text("tools.finalize.error.generic", error=e)

    async def _arun(self, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, confirmed_input)

class ConfirmEdit(BaseTool):
    name: str = get_text("tools.confirm_edit.name")
    description: str = get_text("tools.confirm_edit.description")
    args_schema: Type[BaseModel] = ConfirmationSchema
    # This tool acts on the confirmation, no further HITL needed within it
    requires_confirmation: bool = False

    def _run(self, confirmation: str) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.confirm_edit.error.no_chat")

        state_path = _get_history_path(chat_id)
        pending_path_str = None # Keep track for error messages

        try:
            # 1. Read current state
            state = _read_toolset_state(chat_id)
            pending_edit_data = state.get("pending_edit")

            # 2. Check if an edit is actually pending
            if not pending_edit_data or not isinstance(pending_edit_data, dict):
                logger.warning(f"ConfirmEdit called for chat {chat_id}, but no pending edit found in state.")
                return get_text("tools.confirm_edit.error.no_pending")

            # 3. Extract pending data
            pending_path_str = pending_edit_data.get("path")
            pending_new_content = pending_edit_data.get("new_content")
            if not pending_path_str or pending_new_content is None: # new_content can be empty string
                 logger.error(f"Invalid pending edit data found for chat {chat_id}: {pending_edit_data}")
                 # Clear invalid state and report error
                 state["pending_edit"] = None
                 _write_toolset_state(chat_id, state) # Attempt to clear bad state
                 return get_text("tools.confirm_edit.error.state_read_failed") # Generic error

            # 4. IMPORTANT: Clear pending edit from state *before* acting
            state["pending_edit"] = None
            if not _write_toolset_state(chat_id, state):
                # If state cannot be cleared, it's risky to proceed.
                logger.error(f"Failed to clear pending edit state for chat {chat_id} *before* acting.")
                return get_text("tools.confirm_edit.error.state_write_failed")

            # 5. Process confirmation
            action = confirmation.lower().strip()
            target_path = Path(pending_path_str) # Already resolved in EditNew tool

            if action == 'yes':
                logger.info(f"Confirmed 'yes' for pending edit on {target_path}")
                # 5a. Verify target file still exists and is a file
                if not target_path.is_file():
                    logger.error(f"Target file {target_path} not found or not a file during edit confirmation.")
                    return get_text("tools.confirm_edit.error.target_missing", path=str(target_path))

                # 5b. Create backup
                backup_path = None
                try:
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    backup_path = target_path.with_suffix(f"{target_path.suffix}.bak.{timestamp}")
                    shutil.copy2(target_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                except Exception as backup_e:
                     logger.error(f"Failed to create backup for {target_path}: {backup_e}", exc_info=True)
                     return get_text("tools.confirm_edit.error.backup_failed", path=str(target_path), error=backup_e)

                # 5c. Write new content
                try:
                    target_path.write_text(pending_new_content, encoding='utf-8')
                except Exception as write_e:
                    logger.error(f"Failed to write new content to {target_path}: {write_e}", exc_info=True)
                    return get_text("tools.confirm_edit.error.write_failed", path=str(target_path), error=write_e)

                # 5d. Log the successful confirmed edit
                log_data = {
                    "operation": "edit_confirmed", # Distinguish from proposal
                    "path": str(target_path),
                    "backup_path": str(backup_path) if backup_path else None,
                }
                _log_history_event(chat_id, log_data)

                return get_text("tools.confirm_edit.success_confirm", path=str(target_path), backup_path=str(backup_path))

            elif action == 'no' or action == 'revert':
                logger.info(f"Confirmed '{action}' for pending edit on {target_path}. Edit cancelled.")
                return get_text("tools.confirm_edit.success_revert", path=str(target_path))
            else:
                logger.warning(f"Invalid confirmation '{confirmation}' received for pending edit on {target_path}.")
                return get_text("tools.confirm_edit.error.invalid_confirmation", confirmation=confirmation)

        except Exception as e:
            logger.error(f"Unexpected error during edit confirmation for {pending_path_str or 'unknown path'}: {e}", exc_info=True)
            try:
                state = _read_toolset_state(chat_id)
                if state.get("pending_edit"):
                     state["pending_edit"] = None
                     _write_toolset_state(chat_id, state)
            except Exception: pass
            return get_text("tools.confirm_edit.error.generic", path=(pending_path_str or "unknown"), error=e)

    async def _arun(self, confirmation: str) -> str:
        return await run_in_executor(None, self._run, confirmation)

class ConfirmDelete(BaseTool):
    name: str = get_text("tools.confirm_delete.name")
    description: str = get_text("tools.confirm_delete.description")
    args_schema: Type[BaseModel] = ConfirmDeleteSchema
    requires_confirmation: bool = False # Acts on confirmation

    def _run(self, confirmation: str) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.confirm_delete.error.no_chat")

        pending_path_str = None # For error messages

        try:
            # 1. Read state and check for pending delete
            state = _read_toolset_state(chat_id)
            pending_delete_data = state.get("pending_delete")

            if not pending_delete_data or not isinstance(pending_delete_data, dict):
                logger.warning(f"ConfirmDelete called for chat {chat_id}, but no pending delete found.")
                return get_text("tools.confirm_delete.error.no_pending")

            # 2. Extract path and clear pending state *immediately*
            pending_path_str = pending_delete_data.get("path")
            if not pending_path_str:
                 logger.error(f"Invalid pending delete data for chat {chat_id}: {pending_delete_data}")
                 state["pending_delete"] = None
                 _write_toolset_state(chat_id, state) # Attempt cleanup
                 return get_text("tools.confirm_delete.error.state_read_failed")

            state["pending_delete"] = None
            if not _write_toolset_state(chat_id, state):
                logger.error(f"Failed to clear pending delete state for {chat_id} before acting.")
                return get_text("tools.confirm_delete.error.state_write_failed")

            # 3. Process confirmation
            action = confirmation.lower().strip()
            target_path = Path(pending_path_str) # Already resolved in ProposeDelete

            if action == 'yes':
                logger.info(f"Confirmed 'yes' for pending delete of {target_path}")

                # 3a. Re-check existence
                if not target_path.exists():
                    logger.error(f"Target path {target_path} not found during delete confirmation.")
                    return get_text("tools.confirm_delete.error.target_missing", path=str(target_path))

                # 3b. Perform deletion
                log_data = {"path": str(target_path)}
                op_type = ""
                try:
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                        op_type = "directory"
                        log_data["operation"] = "delete_dir_confirmed"
                    elif target_path.is_file():
                        os.remove(target_path)
                        op_type = "file"
                        log_data["operation"] = "delete_file_confirmed"
                    else:
                        raise OSError(f"Path exists but is neither file nor directory: {target_path}")

                except Exception as delete_e:
                     logger.error(f"Failed to delete {target_path}: {delete_e}", exc_info=True)
                     return get_text("tools.confirm_delete.error.delete_failed", path=str(target_path), error=delete_e)

                # 3c. Log success
                _log_history_event(chat_id, log_data)
                return get_text("tools.confirm_delete.success_confirm", type=op_type, path=str(target_path))

            elif action == 'no' or action == 'cancel':
                logger.info(f"Confirmed '{action}' for pending delete of {target_path}. Delete cancelled.")
                return get_text("tools.confirm_delete.success_cancel", path=str(target_path))
            else:
                logger.warning(f"Invalid confirmation '{confirmation}' received for pending delete of {target_path}.")
                return get_text("tools.confirm_delete.error.invalid_confirmation", confirmation=confirmation)

        except Exception as e:
            logger.error(f"Unexpected error during delete confirmation for {pending_path_str or 'unknown path'}: {e}", exc_info=True)
            try:
                state = _read_toolset_state(chat_id)
                if state.get("pending_delete"):
                     state["pending_delete"] = None
                     _write_toolset_state(chat_id, state)
            except Exception: pass
            return get_text("tools.confirm_delete.error.generic", path=(pending_path_str or "unknown"), error=e)

    async def _arun(self, confirmation: str) -> str:
        return await run_in_executor(None, self._run, confirmation)

# --- Instantiate Tools ---
file_manager_usage_guide_tool = FileManagerUsageGuideTool()
create_tool = Create()
read_tool = Read()
delete_tool = ProposeDelete()
edit_tool = Edit()
copy_tool = Copy()
move_tool = Move()
rename_tool = Rename()
find_tool = Find()
exists_tool = CheckExist()
history_tool = ShowHistory()
restore_tool = RestoreFromBackup()
finalize_tool = FinalizeChanges()
confirm_edit_tool = ConfirmEdit()
confirm_delete_tool = ConfirmDelete()

# --- Define Toolset Structure ---
toolset_tools: List[BaseTool] = [
    file_manager_usage_guide_tool,
    create_tool,
    read_tool,
    delete_tool,
    edit_tool,
    copy_tool,
    move_tool,
    rename_tool,
    find_tool,
    exists_tool,
    history_tool,
    restore_tool,
    finalize_tool,
    confirm_edit_tool,
    confirm_delete_tool,
]

# --- Register Tools ---
register_tools(toolset_tools)

logger.debug(f"Registered File Manager toolset ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")