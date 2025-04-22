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
from .settings import FILES_HISTORY_LIMIT # Import toolset specific settings
from .prompts import FILES_TOOLSET_PROMPT
from .texts import get_text # <--- ADDED IMPORT

console = get_console_manager()

# --- Toolset Metadata ---
toolset_id = "files"
toolset_name = get_text("toolset.name") # MODIFIED
toolset_description = get_text("toolset.description") # MODIFIED
toolset_required_secrets: Dict[str, str] = {}

# --- State Management Helpers ---
def _get_history_path(chat_id: str) -> Path:
    return get_toolset_data_path(chat_id, toolset_id)

def _read_history_state(chat_id: str) -> Dict:
    history_path = _get_history_path(chat_id)
    return read_json(history_path, default_value={"history": []})

def _write_history_state(chat_id: str, state_data: Dict) -> None:
    history_path = _get_history_path(chat_id)
    write_json(history_path, state_data)

def _log_history_event(chat_id: str, event_data: Dict) -> None:
    if not chat_id: return
    try:
        state = _read_history_state(chat_id)
        if "history" not in state or not isinstance(state["history"], list):
            state["history"] = []

        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        state["history"].append(event_data)
        # Apply history limit BEFORE writing
        limit = FILES_HISTORY_LIMIT # Default
        try: # Try to read chat-specific limit
            config_path = get_toolset_data_path(chat_id, toolset_id)
            tool_config = read_json(config_path, default_value=None)
            if tool_config is not None and "history_retrieval_limit" in tool_config:
                 limit = int(tool_config["history_retrieval_limit"])
                 if limit < 0: limit = 0
        except Exception: pass # Ignore errors reading limit, use default

        if limit > 0 and len(state["history"]) > limit:
             state["history"] = state["history"][-limit:] # Keep only the last 'limit' items

        _write_history_state(chat_id, state)
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
        get_text("common.labels.system"),
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
                get_text("common.labels.warning"),
                get_text("config.warn_invalid"),
                console.STYLE_WARNING_LABEL,
                console.STYLE_WARNING_CONTENT
            )
            final_config["history_retrieval_limit"] = default_limit

    except (KeyboardInterrupt, EOFError):
        # Message handled by prompt_for_input
        # Just return the existing config without changes
        console.display_message(
            get_text("common.labels.warning"),
            get_text("config.warn_cancel"),
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )
        return current_chat_config if current_chat_config is not None else {"history_retrieval_limit": FILES_HISTORY_LIMIT}
    except Exception as e:
        logger.error(f"Error during File Manager configuration: {e}", exc_info=True)
        # Use console manager for error
        console.display_message(
            get_text("common.labels.error"),
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
            get_text("common.labels.info"),
            get_text("config.info_saved"),
            console.STYLE_INFO_LABEL,
            console.STYLE_INFO_CONTENT
        )
    else:
        console.display_message(
            get_text("common.labels.error"),
            get_text("config.error_save_failed"),
            console.STYLE_ERROR_LABEL,
            console.STYLE_ERROR_CONTENT
        )

    return final_config

# --- Tool Schemas ---
class NoArgsSchema(BaseModel): pass
class PathSchema(BaseModel):
    path: str = Field(description=get_text("schemas.path.path_desc")) # MODIFIED
class CreateSchema(BaseModel):
    path: str = Field(description=get_text("schemas.create.path_desc")) # MODIFIED
    content: Optional[str] = Field(None, description=get_text("schemas.create.content_desc")) # MODIFIED
    is_directory: bool = Field(False, description=get_text("schemas.create.is_directory_desc")) # MODIFIED
class EditSchemaSimple(BaseModel):
    path: str = Field(description=get_text("schemas.edit_simple.path_desc")) # MODIFIED
    find_text: str = Field(description=get_text("schemas.edit_simple.find_text_desc")) # MODIFIED
    replace_text: str = Field(description=get_text("schemas.edit_simple.replace_text_desc")) # MODIFIED
    summary: str = Field(description=get_text("schemas.edit_simple.summary_desc")) # MODIFIED
class FromToSchema(BaseModel):
    from_path: str = Field(description=get_text("schemas.from_to.from_path_desc")) # MODIFIED
    to_path: str = Field(description=get_text("schemas.from_to.to_path_desc")) # MODIFIED
class RenameSchema(BaseModel):
    path: str = Field(description=get_text("schemas.rename.path_desc")) # MODIFIED
    new_name: str = Field(description=get_text("schemas.rename.new_name_desc")) # MODIFIED
class FindSchema(BaseModel):
    query: str = Field(description=get_text("schemas.find.query_desc")) # MODIFIED
    directory: Optional[str] = Field(None, description=get_text("schemas.find.directory_desc")) # MODIFIED

# --- Tool Classes ---

class FileManagerUsageGuideTool(BaseTool):
    name: str = get_text("tools.usage_guide.name") # MODIFIED
    description: str = get_text("tools.usage_guide.description") # MODIFIED
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        logger.debug(f"FileManagerUsageGuideTool invoked.")
        return FILES_TOOLSET_PROMPT

    async def _arun(self) -> str: return await run_in_executor(None, self._run)

class Create(BaseTool):
    name: str = get_text("tools.create.name") # MODIFIED
    description: str = get_text("tools.create.description") # MODIFIED
    args_schema: Type[BaseModel] = CreateSchema

    def _run(self, path: str, content: Optional[str] = None, is_directory: bool = False) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.create.error.no_chat") # MODIFIED
        target_path = Path(path).resolve()

        try:
            if target_path.exists():
                return get_text("tools.create.error.exists", path=str(target_path)) # MODIFIED

            target_path.parent.mkdir(parents=True, exist_ok=True)

            if is_directory:
                if content:
                    return get_text("tools.create.error.content_for_dir", path=str(target_path)) # MODIFIED
                target_path.mkdir()
                op_type = "directory"
                log_data = {"operation": "create_dir", "path": str(target_path)}
            else:
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content or "")
                op_type = "file"
                log_data = {"operation": "create_file", "path": str(target_path)}

            _log_history_event(chat_id, log_data)
            return get_text("tools.create.success", type=op_type, path=str(target_path)) # MODIFIED

        except Exception as e:
            logger.error(f"Error creating path {target_path}: {e}", exc_info=True)
            return get_text("tools.create.error.generic", path=str(target_path), error=e) # MODIFIED

    async def _arun(self, path: str, content: Optional[str] = None, is_directory: bool = False) -> str:
        return await run_in_executor(None, self._run, path, content, is_directory)

class Read(BaseTool):
    name: str = get_text("tools.read.name") # MODIFIED
    description: str = get_text("tools.read.description") # MODIFIED
    args_schema: Type[BaseModel] = PathSchema

    def _run(self, path: str) -> str:
        target_path = Path(path).resolve()
        max_len = 4000

        try:
            if not target_path.is_file():
                return get_text("tools.read.error.not_file", path=str(target_path)) # MODIFIED

            content = target_path.read_text(encoding='utf-8', errors='replace')
            truncated_suffix = get_text("tools.read.truncated_suffix") # MODIFIED
            display_content = (content[:max_len] + truncated_suffix) if len(content) > max_len else content
            return get_text("tools.read.success", path=str(target_path), content=display_content) # MODIFIED

        except FileNotFoundError:
            return get_text("tools.read.error.not_found", path=str(target_path)) # MODIFIED
        except Exception as e:
            logger.error(f"Error reading file {target_path}: {e}", exc_info=True)
            return get_text("tools.read.error.generic", path=str(target_path), error=e) # MODIFIED

    async def _arun(self, path: str) -> str: return await run_in_executor(None, self._run, path)

class Delete(BaseTool):
    name: str = get_text("tools.delete.name") # MODIFIED
    description: str = get_text("tools.delete.description") # MODIFIED
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = True

    def _run(self, path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.delete.error.no_chat") # MODIFIED
        target_path_str = path

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to delete: {target_path_str}")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"path": target_path_str},
                edit_key="path"
            )
        else:
            final_path_str = confirmed_input
            target_path = Path(final_path_str).resolve()
            logger.info(f"Executing confirmed delete: {target_path}")

            try:
                if not target_path.exists():
                    return get_text("tools.delete.error.not_exists", path=str(target_path)) # MODIFIED

                log_data = {"path": str(target_path)}
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                    op_type = "directory"
                    log_data["operation"] = "delete_dir"
                else:
                    os.remove(target_path)
                    op_type = "file"
                    log_data["operation"] = "delete_file"

                _log_history_event(chat_id, log_data)
                return get_text("tools.delete.success", type=op_type, path=str(target_path)) # MODIFIED

            except Exception as e:
                logger.error(f"Error deleting path {target_path}: {e}", exc_info=True)
                return get_text("tools.delete.error.generic", path=str(target_path), error=e) # MODIFIED

    async def _arun(self, path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, confirmed_input)

class Edit(BaseTool):
    name: str = get_text("tools.edit.name") # MODIFIED
    description: str = get_text("tools.edit.description") # MODIFIED
    args_schema: Type[BaseModel] = EditSchemaSimple
    requires_confirmation: bool = True

    def _run(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.edit.error.no_chat") # MODIFIED
        target_path = Path(path).resolve()

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to edit '{target_path}' with summary: {summary}")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"path": path, "find_text": find_text, "replace_text": replace_text, "summary": summary},
                edit_key="summary"
            )
        else:
            final_summary = confirmed_input
            logger.info(f"Executing confirmed edit on '{target_path}' with summary: {final_summary}")

            try:
                if not target_path.is_file():
                    return get_text("tools.edit.error.not_file", path=str(target_path)) # MODIFIED

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                backup_path = target_path.with_suffix(f"{target_path.suffix}.bak.{timestamp}")
                shutil.copy2(target_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

                content = target_path.read_text(encoding='utf-8', errors='replace')
                new_content = content.replace(find_text, replace_text)

                if new_content == content:
                     try: backup_path.unlink()
                     except OSError: pass
                     return get_text("tools.edit.info_no_change", path=str(target_path)) # MODIFIED

                target_path.write_text(new_content, encoding='utf-8')

                log_data = {
                    "operation": "edit",
                    "path": str(target_path),
                    "backup_path": str(backup_path),
                    "summary": final_summary,
                    "find": find_text,
                    "replace": replace_text
                }
                _log_history_event(chat_id, log_data)

                return get_text("tools.edit.success", path=str(target_path), summary=final_summary, backup_path=str(backup_path)) # MODIFIED

            except Exception as e:
                logger.error(f"Error editing file {target_path}: {e}", exc_info=True)
                return get_text("tools.edit.error.generic", path=str(target_path), error=e) # MODIFIED

    async def _arun(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, find_text, replace_text, summary, confirmed_input)

class Copy(BaseTool):
    name: str = get_text("tools.copy.name") # MODIFIED
    description: str = get_text("tools.copy.description") # MODIFIED
    args_schema: Type[BaseModel] = FromToSchema
    requires_confirmation: bool = True

    def _run(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.copy.error.no_chat") # MODIFIED
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
                    return get_text("tools.copy.error.source_not_exists", path=str(source_path)) # MODIFIED
                if dest_path.exists():
                    return get_text("tools.copy.error.dest_exists", path=str(dest_path)) # MODIFIED

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
                return get_text("tools.copy.success", type=op_type, from_path=str(source_path), to_path=str(dest_path)) # MODIFIED

            except Exception as e:
                logger.error(f"Error copying {source_path} to {dest_path}: {e}", exc_info=True)
                return get_text("tools.copy.error.generic", from_path=str(source_path), to_path=str(dest_path), error=e) # MODIFIED

    async def _arun(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, from_path, to_path, confirmed_input)

class Move(BaseTool):
    name: str = get_text("tools.move.name") # MODIFIED
    description: str = get_text("tools.move.description") # MODIFIED
    args_schema: Type[BaseModel] = FromToSchema
    requires_confirmation: bool = True

    def _run(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.move.error.no_chat") # MODIFIED
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
                    return get_text("tools.move.error.source_not_exists", path=str(source_path)) # MODIFIED
                if dest_path.exists():
                    return get_text("tools.move.error.dest_exists", path=str(dest_path)) # MODIFIED

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))

                log_data = {
                    "operation": "move",
                    "from_path": str(source_path),
                    "to_path": str(dest_path)
                 }
                _log_history_event(chat_id, log_data)
                return get_text("tools.move.success", from_path=str(source_path), to_path=str(dest_path)) # MODIFIED

            except Exception as e:
                logger.error(f"Error moving {source_path} to {dest_path}: {e}", exc_info=True)
                return get_text("tools.move.error.generic", from_path=str(source_path), to_path=str(dest_path), error=e) # MODIFIED

    async def _arun(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, from_path, to_path, confirmed_input)

class Rename(BaseTool):
    name: str = get_text("tools.rename.name") # MODIFIED
    description: str = get_text("tools.rename.description") # MODIFIED
    args_schema: Type[BaseModel] = RenameSchema
    requires_confirmation: bool = True

    def _run(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.rename.error.no_chat") # MODIFIED
        original_path = Path(path).resolve()
        original_new_name = new_name

        if confirmed_input is None:
             if os.path.sep in new_name or (os.altsep and os.altsep in new_name):
                  return get_text("tools.rename.error.invalid_new_name", new_name=new_name) # MODIFIED
             logger.debug(f"Requesting confirmation to rename '{original_path}' to '{new_name}'")
             raise PromptNeededError(
                 tool_name=self.name,
                 proposed_args={"path": path, "new_name": new_name},
                 edit_key="new_name"
             )
        else:
            final_new_name = confirmed_input.strip()
            if not final_new_name: return get_text("tools.rename.error.empty_new_name") # MODIFIED
            if os.path.sep in final_new_name or (os.altsep and os.altsep in final_new_name):
                 return get_text("tools.rename.error.invalid_new_name", new_name=final_new_name) # MODIFIED

            logger.info(f"Executing confirmed rename of '{original_path}' to '{final_new_name}'")
            new_path = original_path.with_name(final_new_name)

            try:
                if not original_path.exists():
                    return get_text("tools.rename.error.path_not_exists", path=str(original_path)) # MODIFIED
                if new_path.exists():
                    return get_text("tools.rename.error.target_exists", path=str(new_path)) # MODIFIED

                original_path.rename(new_path)

                log_data = {
                    "operation": "rename",
                    "from_path": str(original_path),
                    "to_path": str(new_path)
                }
                _log_history_event(chat_id, log_data)
                return get_text("tools.rename.success", path=str(original_path), new_path=str(new_path)) # MODIFIED

            except Exception as e:
                logger.error(f"Error renaming {original_path} to {new_path}: {e}", exc_info=True)
                return get_text("tools.rename.error.generic", path=str(original_path), new_path=str(new_path), error=e) # MODIFIED

    async def _arun(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, new_name, confirmed_input)

class Find(BaseTool):
    name: str = get_text("tools.find.name") # MODIFIED
    description: str = get_text("tools.find.description") # MODIFIED
    args_schema: Type[BaseModel] = FindSchema

    def _run(self, query: str, directory: Optional[str] = None) -> str:
        start_dir = Path(directory).resolve() if directory else Path.cwd()
        query_lower = query.lower()
        matches = []
        limit = 50
        start_dir_str = str(start_dir) # For use in get_text

        try:
            if not start_dir.is_dir():
                return get_text("tools.find.error.dir_not_found", directory=start_dir_str) # MODIFIED

            logger.info(f"Searching for '{query}' in '{start_dir_str}'...")
            for item in start_dir.rglob('*'):
                if query_lower in item.name.lower():
                    matches.append(str(item.relative_to(start_dir)))
                    if len(matches) >= limit:
                        break

            if not matches:
                return get_text("tools.find.info_no_matches", query=query, directory=start_dir_str) # MODIFIED
            else:
                matches_str = "\n".join(f"- {m}" for m in matches)
                result_str = get_text("tools.find.success", count=len(matches), query=query, directory=start_dir_str, matches=matches_str) # MODIFIED
                if len(matches) >= limit:
                    result_str += get_text("tools.find.info_limit_reached") # MODIFIED
                return result_str

        except PermissionError:
             logger.warning(f"Permission denied during search in {start_dir_str}.")
             if matches:
                  matches_str = "\n".join(f"- {m}" for m in matches)
                  return get_text("tools.find.warn_permission", count=len(matches), directory=start_dir_str, matches=matches_str) # MODIFIED
             else:
                  return get_text("tools.find.error_permission", directory=start_dir_str) # MODIFIED
        except Exception as e:
            logger.error(f"Error finding files matching '{query}' in {start_dir_str}: {e}", exc_info=True)
            return get_text("tools.find.error.generic", error=e) # MODIFIED

    async def _arun(self, query: str, directory: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, query, directory)

class CheckExist(BaseTool):
    name: str = get_text("tools.check_exist.name") # MODIFIED
    description: str = get_text("tools.check_exist.description") # MODIFIED
    args_schema: Type[BaseModel] = PathSchema

    def _run(self, path: str) -> str:
        target_path = Path(path)
        try:
            if target_path.exists():
                type_str = "directory" if target_path.is_dir() else "file" if target_path.is_file() else "special file"
                return get_text("tools.check_exist.success_exists", path=path, type=type_str) # MODIFIED
            else:
                return get_text("tools.check_exist.success_not_exists", path=path) # MODIFIED
        except Exception as e:
            logger.error(f"Error checking existence of path '{path}': {e}", exc_info=True)
            return get_text("tools.check_exist.error.generic", path=path, error=e) # MODIFIED

    async def _arun(self, path: str) -> str: return await run_in_executor(None, self._run, path)

class ShowHistory(BaseTool):
    name: str = get_text("tools.history.name") # MODIFIED
    description: str = get_text("tools.history.description") # MODIFIED
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.history.error.no_chat") # MODIFIED

        try:
            state = _read_history_state(chat_id)
            history = state.get("history", [])

            if not history:
                return get_text("tools.history.info_empty") # MODIFIED

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

            output = get_text("tools.history.header", count=actual_shown, total=total_history) # MODIFIED
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

                line = get_text("tools.history.line_format", ts=ts_formatted, op=op.upper()) # MODIFIED
                if path: line += f": {path}" # Keep details appended for now
                if from_p: line += f": From {from_p}"
                if to_p: line += f" To {to_p}"
                if summary: line += f" (Summary: {summary})"
                if backup: line += f" [Backup: {backup}]"
                output += line + "\n"

            return output.strip()

        except Exception as e:
            logger.error(f"Error retrieving file history for chat {chat_id}: {e}", exc_info=True)
            return get_text("tools.history.error.generic", error=e) # MODIFIED

    async def _arun(self) -> str: return await run_in_executor(None, self._run)

class Restore(BaseTool):
    name: str = get_text("tools.restore.name") # MODIFIED
    description: str = get_text("tools.restore.description") # MODIFIED
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = True

    def _run(self, path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.restore.error.no_chat") # MODIFIED
        target_path_str = path

        try:
            state = _read_history_state(chat_id)
            history = state.get("history", [])
            latest_backup_path_str = None
            latest_backup_ts = None

            for event in reversed(history):
                if event.get("operation") == "edit" and event.get("path") == target_path_str:
                    backup_path = event.get("backup_path")
                    event_ts = event.get("timestamp")
                    if backup_path:
                        latest_backup_path_str = backup_path
                        latest_backup_ts = event_ts
                        break

            if not latest_backup_path_str:
                return get_text("tools.restore.error.no_backup_history", path=target_path_str) # MODIFIED

            backup_path = Path(latest_backup_path_str)
            if not backup_path.exists():
                return get_text("tools.restore.error.backup_missing", backup_path=latest_backup_path_str) # MODIFIED

            if confirmed_input is None:
                logger.debug(f"Requesting confirmation to restore '{target_path_str}' from backup '{backup_path}' (created around {latest_backup_ts})")
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"path": target_path_str},
                    edit_key="path",
                    prompt_suffix=get_text("tools.restore.prompt_suffix", backup_name=backup_path.name) # MODIFIED
                )
            else:
                final_path_str = confirmed_input
                target_path = Path(final_path_str).resolve()
                logger.info(f"Executing confirmed restore of '{target_path}' from '{backup_path}'")

                if target_path.exists() and not target_path.is_file():
                     return get_text("tools.restore.error.target_not_file", path=str(target_path)) # MODIFIED

                shutil.copy2(backup_path, target_path)

                log_data = {
                    "operation": "restore",
                    "path": str(target_path),
                    "backup_path": str(backup_path)
                }
                _log_history_event(chat_id, log_data)
                return get_text("tools.restore.success", path=str(target_path), backup_name=backup_path.name) # MODIFIED

        except Exception as e:
            logger.error(f"Error restoring file {target_path_str}: {e}", exc_info=True)
            return get_text("tools.restore.error.generic", path=target_path_str, error=e) # MODIFIED

    async def _arun(self, path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, confirmed_input)

class FinalizeChanges(BaseTool):
    name: str = get_text("tools.finalize.name") # MODIFIED
    description: str = get_text("tools.finalize.description") # MODIFIED
    args_schema: Type[BaseModel] = NoArgsSchema
    requires_confirmation: bool = True

    def _run(self, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.finalize.error.no_chat") # MODIFIED

        try:
            state = _read_history_state(chat_id)
            history = state.get("history", [])
            backup_paths_to_delete = set()
            edit_count = 0

            for event in history:
                if event.get("operation") == "edit":
                    edit_count += 1
                    backup_path = event.get("backup_path")
                    if backup_path:
                        backup_paths_to_delete.add(backup_path)

            num_backups = len(backup_paths_to_delete)

            if num_backups == 0:
                return get_text("tools.finalize.info_no_backups") # MODIFIED

            if confirmed_input is None:
                logger.debug(f"Requesting confirmation to delete {num_backups} backup files for chat {chat_id}.")
                confirm_prompt = get_text("tools.finalize.prompt_confirm", count=num_backups) # MODIFIED
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"confirm": confirm_prompt}, # Show info in prompt
                    edit_key="confirm",
                    prompt_suffix=get_text("tools.finalize.prompt_suffix") # MODIFIED
                )
            else:
                if confirmed_input.lower().strip() != 'yes':
                    return get_text("tools.finalize.info_cancel") # MODIFIED

                logger.info(f"Executing confirmed finalization: Deleting {num_backups} backups for chat {chat_id}.")
                deleted_count = 0
                errors = []
                for backup_path_str in backup_paths_to_delete:
                    try:
                        backup_path = Path(backup_path_str)
                        if backup_path.exists():
                            os.remove(backup_path)
                            deleted_count += 1
                        else:
                             logger.warning(f"Backup file not found during finalize: {backup_path}")
                    except Exception as e:
                        logger.error(f"Error deleting backup {backup_path_str}: {e}")
                        errors.append(backup_path_str)

                log_data = {
                    "operation": "finalize",
                    "deleted_backups": deleted_count,
                    "failed_deletions": errors
                }
                _log_history_event(chat_id, log_data)

                result_msg = get_text("tools.finalize.success", deleted_count=deleted_count, total_count=num_backups) # MODIFIED
                if errors:
                    result_msg += get_text("tools.finalize.warn_errors", errors=', '.join(errors)) # MODIFIED
                return result_msg

        except Exception as e:
            logger.error(f"Error finalizing file changes for chat {chat_id}: {e}", exc_info=True)
            return get_text("tools.finalize.error.generic", error=e) # MODIFIED

    async def _arun(self, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, confirmed_input)

# --- Instantiate Tools ---
file_manager_usage_guide_tool = FileManagerUsageGuideTool()
create_tool = Create()
read_tool = Read()
delete_tool = Delete()
edit_tool = Edit()
copy_tool = Copy()
move_tool = Move()
rename_tool = Rename()
find_tool = Find()
exists_tool = CheckExist()
history_tool = ShowHistory()
restore_tool = Restore()
finalize_tool = FinalizeChanges()

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
]

# --- Register Tools ---
register_tools(toolset_tools)

logger.debug(f"Registered File Manager toolset ({toolset_id}) with tools: {[t.name for t in toolset_tools]}")