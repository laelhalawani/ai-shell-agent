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
from ...utils import read_json, write_json, ensure_dotenv_key
from ...errors import PromptNeededError
from ...console_manager import get_console_manager
from ...chat_state_manager import (
    get_current_chat,
    get_toolset_data_path,
    check_and_configure_toolset # Keep import if needed elsewhere
)
from .prompts import FILES_TOOLSET_PROMPT

console = get_console_manager()

# --- Toolset Metadata ---
toolset_id = "files"
toolset_name = "File Manager"
toolset_description = "Provides tools for direct file and directory manipulation (create, read, edit, delete, copy, move, find, history)."

# --- Toolset Configuration ---
toolset_config_defaults: Dict[str, Any] = {
    "history_retrieval_limit": 20 # Default number of history items to show
}
toolset_required_secrets: Dict[str, str] = {} # No secrets currently needed

# --- State Management Helpers ---
def _get_history_path(chat_id: str) -> Path:
    """Gets the path to the history state file for this toolset in the chat."""
    return get_toolset_data_path(chat_id, toolset_id) # Uses state manager helper

def _read_history_state(chat_id: str) -> Dict:
    """Reads the history state file, returning defaults if not found."""
    history_path = _get_history_path(chat_id)
    # Default structure includes an empty history list
    return read_json(history_path, default_value={"history": []})

def _write_history_state(chat_id: str, state_data: Dict) -> None:
    """Writes the history state file."""
    history_path = _get_history_path(chat_id)
    write_json(history_path, state_data)

def _log_history_event(chat_id: str, event_data: Dict) -> None:
    """Adds an event to the chat's history state for this toolset."""
    if not chat_id: return # Should not happen
    try:
        state = _read_history_state(chat_id)
        if "history" not in state or not isinstance(state["history"], list):
            state["history"] = [] # Ensure history list exists

        # Add timestamp if not present
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        state["history"].append(event_data)
        _write_history_state(chat_id, state)
        logger.debug(f"Logged file history event for chat {chat_id}: {event_data.get('operation')}")
    except Exception as e:
        logger.error(f"Failed to log file history event for chat {chat_id}: {e}", exc_info=True)

# --- Configuration Function ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Path,
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
) -> Dict:
    """
    Configuration function for the File Manager toolset.
    Prompts user for history retrieval limit.
    """
    logger.info(f"Configuring File Manager toolset. Global: {global_config_path}, Local: {local_config_path}")
    config_to_prompt = current_chat_config or toolset_config_defaults
    final_config = {}

    console.display_message("SYSTEM:", "\n--- Configure File Manager Manager ---", console.STYLE_SYSTEM_LABEL, console.STYLE_SYSTEM_CONTENT)

    try:
        # Prompt for history limit
        limit_str = console.prompt_for_input(
            "Enter max number of history items to show",
            default=str(config_to_prompt.get("history_retrieval_limit", 20))
        ).strip()

        # Validate and set limit
        try:
            limit = int(limit_str) if limit_str else config_to_prompt.get("history_retrieval_limit", 20)
            if limit < 0: limit = 0 # Non-negative
            final_config["history_retrieval_limit"] = limit
        except ValueError:
            console.display_message("WARNING:", "Invalid number entered. Using previous/default limit.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
            final_config["history_retrieval_limit"] = config_to_prompt.get("history_retrieval_limit", 20)

    except (KeyboardInterrupt, EOFError):
        console.display_message("WARNING:", "\nConfiguration cancelled. Using previous/default values.", console.STYLE_WARNING_LABEL, console.STYLE_WARNING_CONTENT)
        return current_chat_config or toolset_config_defaults
    except Exception as e:
        logger.error(f"Error during File Manager configuration: {e}", exc_info=True)
        console.display_message("ERROR:", f"\nConfiguration error: {e}", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)
        return current_chat_config or toolset_config_defaults

    # Save the final configuration to BOTH local and global paths
    save_success = True
    try: write_json(local_config_path, final_config)
    except Exception as e: save_success = False; logger.error(f"Failed to save config to {local_config_path}: {e}")
    try: write_json(global_config_path, final_config)
    except Exception as e: save_success = False; logger.error(f"Failed to save config to {global_config_path}: {e}")

    if save_success: console.display_message("INFO:", "\nFile Manager configuration saved.", console.STYLE_INFO_LABEL, console.STYLE_INFO_CONTENT)
    else: console.display_message("ERROR:", "\nFailed to save File Manager configuration.", console.STYLE_ERROR_LABEL, console.STYLE_ERROR_CONTENT)

    return final_config

# --- Tool Schemas ---
class NoArgsSchema(BaseModel): pass
class PathSchema(BaseModel):
    path: str = Field(description="The absolute or relative path to the file or directory.")
class CreateSchema(BaseModel):
    path: str = Field(description="The absolute or relative path for the new file or directory.")
    content: Optional[str] = Field(None, description="Optional text content to write if creating a file.")
    is_directory: bool = Field(False, description="Set to true if creating a directory, false (default) for a file.")
class EditSchemaSimple(BaseModel):
    path: str = Field(description="The path to the file to edit.")
    find_text: str = Field(description="The exact text content to find within the file.")
    replace_text: str = Field(description="The text content to replace the 'find_text' with.")
    summary: str = Field(description="A brief summary of the edit being performed.")
class FromToSchema(BaseModel):
    from_path: str = Field(description="The source path of the file or directory.")
    to_path: str = Field(description="The destination path.")
class RenameSchema(BaseModel):
    path: str = Field(description="The current path of the file or directory to rename.")
    new_name: str = Field(description="The desired new name for the file or directory (just the final component, not the full path).")
class FindSchema(BaseModel):
    query: str = Field(description="The filename or directory name pattern to search for.")
    directory: Optional[str] = Field(None, description="Optional directory to start the search from (defaults to current working directory). Search includes subdirectories.")

# --- Tool Classes ---

class FileManagerUsageGuideTool(BaseTool):
    name: str = "file_manager_usage_guide"
    description: str = "Displays usage instructions and context for the File Manager toolset."
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        """Returns the usage instructions for the File Manager toolset."""
        logger.debug(f"FileManagerUsageGuideTool invoked.")
        # Simply return the static prompt content
        return FILES_TOOLSET_PROMPT

    async def _arun(self) -> str: return await run_in_executor(None, self._run)

class Create(BaseTool):
    name: str = "create_file_or_dir"
    description: str = "Creates a new file or directory at the specified path. Optionally writes content if creating a file."
    args_schema: Type[BaseModel] = CreateSchema

    def _run(self, path: str, content: Optional[str] = None, is_directory: bool = False) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        target_path = Path(path).resolve()

        try:
            if target_path.exists():
                return f"Error: Path already exists: {target_path}"

            target_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent exists

            if is_directory:
                if content:
                    return f"Error: Cannot provide content when creating a directory: {target_path}"
                target_path.mkdir()
                op_type = "directory"
                log_data = {"operation": "create_dir", "path": str(target_path)}
            else:
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content or "")
                op_type = "file"
                log_data = {"operation": "create_file", "path": str(target_path)}

            _log_history_event(chat_id, log_data)
            return f"Successfully created {op_type}: {target_path}"

        except Exception as e:
            logger.error(f"Error creating path {target_path}: {e}", exc_info=True)
            return f"Error creating path {target_path}: {e}"

    async def _arun(self, path: str, content: Optional[str] = None, is_directory: bool = False) -> str:
        return await run_in_executor(None, self._run, path, content, is_directory)

class Read(BaseTool):
    name: str = "read_file_content"
    description: str = "Reads the content of a specified file as text."
    args_schema: Type[BaseModel] = PathSchema

    def _run(self, path: str) -> str:
        target_path = Path(path).resolve()
        max_len = 4000 # Limit output size

        try:
            if not target_path.is_file():
                return f"Error: Path is not a file or does not exist: {target_path}"

            content = target_path.read_text(encoding='utf-8', errors='replace')
            display_content = (content[:max_len] + "\n... (truncated)") if len(content) > max_len else content
            return f"Content of {target_path}:\n---\n{display_content}\n---"

        except FileNotFoundError:
            return f"Error: File not found: {target_path}"
        except Exception as e:
            logger.error(f"Error reading file {target_path}: {e}", exc_info=True)
            return f"Error reading file {target_path}: {e}"

    async def _arun(self, path: str) -> str: return await run_in_executor(None, self._run, path)

class Delete(BaseTool):
    name: str = "delete_file_or_dir"
    description: str = "Deletes a specified file or directory. Requires user confirmation."
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = True

    def _run(self, path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        target_path_str = path # Keep original path string for prompt

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to delete: {target_path_str}")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"path": target_path_str},
                edit_key="path" # User confirms/edits the path
            )
        else:
            # User confirmed, use the potentially edited path
            final_path_str = confirmed_input
            target_path = Path(final_path_str).resolve()
            logger.info(f"Executing confirmed delete: {target_path}")

            try:
                if not target_path.exists():
                    return f"Error: Path does not exist: {target_path}"

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
                return f"Successfully deleted {op_type}: {target_path}"

            except Exception as e:
                logger.error(f"Error deleting path {target_path}: {e}", exc_info=True)
                return f"Error deleting path {target_path}: {e}"

    async def _arun(self, path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, confirmed_input)

class Edit(BaseTool):
    name: str = "edit_file"
    description: str = "Edits a file by replacing occurrences of 'find_text' with 'replace_text'. Creates a backup first. Requires user confirmation of the edit summary and target file."
    args_schema: Type[BaseModel] = EditSchemaSimple
    requires_confirmation: bool = True

    def _run(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        target_path = Path(path).resolve()

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to edit '{target_path}' with summary: {summary}")
            # We ask the user to confirm the summary (intent) for the given path.
            raise PromptNeededError(
                tool_name=self.name,
                # Provide all original args for context, even though only summary is the edit_key
                proposed_args={"path": path, "find_text": find_text, "replace_text": replace_text, "summary": summary},
                edit_key="summary" # User confirms/edits the summary
            )
        else:
            # User confirmed, use the potentially edited summary and original path/texts
            final_summary = confirmed_input
            logger.info(f"Executing confirmed edit on '{target_path}' with summary: {final_summary}")

            try:
                if not target_path.is_file():
                    return f"Error: Path is not a file or does not exist: {target_path}"

                # Create backup
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                backup_path = target_path.with_suffix(f"{target_path.suffix}.bak.{timestamp}")
                shutil.copy2(target_path, backup_path) # copy2 preserves metadata
                logger.info(f"Created backup: {backup_path}")

                # Read, replace, write
                content = target_path.read_text(encoding='utf-8', errors='replace')
                new_content = content.replace(find_text, replace_text)

                if new_content == content:
                     # Clean up backup if no changes were made
                     try: backup_path.unlink()
                     except OSError: pass # Ignore if backup deletion fails
                     return f"Edit completed on {target_path}. No changes made (find_text not found?). Original file untouched. No backup created."

                target_path.write_text(new_content, encoding='utf-8')

                # Log success
                log_data = {
                    "operation": "edit",
                    "path": str(target_path),
                    "backup_path": str(backup_path),
                    "summary": final_summary,
                    "find": find_text, # Log find/replace for audit? Maybe too verbose? Optional.
                    "replace": replace_text
                }
                _log_history_event(chat_id, log_data)

                return f"Successfully edited {target_path}.\nSummary: {final_summary}\nBackup created at: {backup_path}"

            except Exception as e:
                logger.error(f"Error editing file {target_path}: {e}", exc_info=True)
                # Attempt to clean up backup on error? Maybe not, leave it for inspection.
                return f"Error editing file {target_path}: {e}"

    async def _arun(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, find_text, replace_text, summary, confirmed_input)

class Copy(BaseTool):
    name: str = "copy_file_or_dir"
    description: str = "Copies a file or directory from 'from_path' to 'to_path'. Requires user confirmation of the source path."
    args_schema: Type[BaseModel] = FromToSchema
    requires_confirmation: bool = True

    def _run(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        source_path_str = from_path # Keep original for prompt

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to copy from '{from_path}' to '{to_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"from_path": from_path, "to_path": to_path},
                edit_key="from_path" # User confirms the source
            )
        else:
            # User confirmed source, use potentially edited source and original destination
            final_from_str = confirmed_input
            final_to_str = to_path
            source_path = Path(final_from_str).resolve()
            dest_path = Path(final_to_str).resolve()
            logger.info(f"Executing confirmed copy from '{source_path}' to '{dest_path}'")

            try:
                if not source_path.exists():
                    return f"Error: Source path does not exist: {source_path}"
                if dest_path.exists():
                    return f"Error: Destination path already exists: {dest_path}. Overwriting not supported."

                dest_path.parent.mkdir(parents=True, exist_ok=True)

                log_data = {"from_path": str(source_path), "to_path": str(dest_path)}
                if source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=False) # No overwrite
                    op_type = "directory"
                    log_data["operation"] = "copy_dir"
                else:
                    shutil.copy2(source_path, dest_path) # copy2 preserves metadata
                    op_type = "file"
                    log_data["operation"] = "copy_file"

                _log_history_event(chat_id, log_data)
                return f"Successfully copied {op_type} from {source_path} to {dest_path}"

            except Exception as e:
                logger.error(f"Error copying {source_path} to {dest_path}: {e}", exc_info=True)
                return f"Error copying {source_path} to {dest_path}: {e}"

    async def _arun(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, from_path, to_path, confirmed_input)

class Move(BaseTool):
    name: str = "move_file_or_dir"
    description: str = "Moves (renames) a file or directory from 'from_path' to 'to_path'. Requires user confirmation of the source path."
    args_schema: Type[BaseModel] = FromToSchema
    requires_confirmation: bool = True

    def _run(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        source_path_str = from_path

        if confirmed_input is None:
            logger.debug(f"Requesting confirmation to move from '{from_path}' to '{to_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"from_path": from_path, "to_path": to_path},
                edit_key="from_path" # User confirms the source
            )
        else:
            # User confirmed source, use potentially edited source and original destination
            final_from_str = confirmed_input
            final_to_str = to_path
            source_path = Path(final_from_str).resolve()
            dest_path = Path(final_to_str).resolve()
            logger.info(f"Executing confirmed move from '{source_path}' to '{dest_path}'")

            try:
                if not source_path.exists():
                    return f"Error: Source path does not exist: {source_path}"
                if dest_path.exists():
                    return f"Error: Destination path already exists: {dest_path}. Overwriting not supported."

                dest_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.move(str(source_path), str(dest_path)) # shutil.move handles files/dirs

                log_data = {
                    "operation": "move", # General 'move' covers rename across/within dirs
                    "from_path": str(source_path),
                    "to_path": str(dest_path)
                 }
                _log_history_event(chat_id, log_data)
                return f"Successfully moved {source_path} to {dest_path}"

            except Exception as e:
                logger.error(f"Error moving {source_path} to {dest_path}: {e}", exc_info=True)
                return f"Error moving {source_path} to {dest_path}: {e}"

    async def _arun(self, from_path: str, to_path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, from_path, to_path, confirmed_input)


class Rename(BaseTool):
    name: str = "rename_file_or_dir"
    description: str = "Renames a file or directory within the same parent directory. Requires user confirmation of the new name."
    args_schema: Type[BaseModel] = RenameSchema
    requires_confirmation: bool = True

    def _run(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        original_path = Path(path).resolve()
        original_new_name = new_name # Keep original for prompt

        if confirmed_input is None:
             # Validate proposed new_name early
             if os.path.sep in new_name or (os.altsep and os.altsep in new_name):
                  return f"Error: 'new_name' ({new_name}) cannot contain path separators. Use 'move_file_or_dir' to move between directories."
             logger.debug(f"Requesting confirmation to rename '{original_path}' to '{new_name}'")
             raise PromptNeededError(
                 tool_name=self.name,
                 proposed_args={"path": path, "new_name": new_name},
                 edit_key="new_name" # User confirms the new name part
             )
        else:
            # User confirmed, use original path and potentially edited new_name
            final_new_name = confirmed_input.strip()
            if not final_new_name: return "Error: Confirmed new name cannot be empty."
            if os.path.sep in final_new_name or (os.altsep and os.altsep in final_new_name):
                 return f"Error: Confirmed 'new_name' ({final_new_name}) cannot contain path separators."

            logger.info(f"Executing confirmed rename of '{original_path}' to '{final_new_name}'")
            new_path = original_path.with_name(final_new_name)

            try:
                if not original_path.exists():
                    return f"Error: Original path does not exist: {original_path}"
                if new_path.exists():
                    return f"Error: Target path already exists: {new_path}"

                original_path.rename(new_path)

                log_data = {
                    "operation": "rename",
                    "from_path": str(original_path),
                    "to_path": str(new_path)
                }
                _log_history_event(chat_id, log_data)
                return f"Successfully renamed {original_path} to {new_path}"

            except Exception as e:
                logger.error(f"Error renaming {original_path} to {new_path}: {e}", exc_info=True)
                return f"Error renaming {original_path} to {new_path}: {e}"

    async def _arun(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, new_name, confirmed_input)


class Find(BaseTool):
    name: str = "find_files"
    description: str = "Searches for files or directories matching the query within a specified directory (or current directory) and its subdirectories."
    args_schema: Type[BaseModel] = FindSchema

    def _run(self, query: str, directory: Optional[str] = None) -> str:
        start_dir = Path(directory).resolve() if directory else Path.cwd()
        query_lower = query.lower()
        matches = []
        limit = 50 # Limit number of results

        try:
            if not start_dir.is_dir():
                return f"Error: Search directory does not exist or is not a directory: {start_dir}"

            logger.info(f"Searching for '{query}' in '{start_dir}'...")
            for item in start_dir.rglob('*'): # Recursive glob
                if query_lower in item.name.lower():
                    matches.append(str(item.relative_to(start_dir))) # Show relative paths
                    if len(matches) >= limit:
                        break

            if not matches:
                return f"No files or directories found matching '{query}' in {start_dir}."
            else:
                result_str = f"Found {len(matches)} match(es) for '{query}' in {start_dir}:\n"
                result_str += "\n".join(f"- {m}" for m in matches)
                if len(matches) >= limit:
                    result_str += "\n... (result limit reached)"
                return result_str

        except PermissionError:
             logger.warning(f"Permission denied during search in {start_dir}.")
             # Return partial results if any found before error
             if matches:
                  return f"Found {len(matches)} match(es) before encountering permission error in {start_dir}:\n" + "\n".join(f"- {m}" for m in matches) + "\n(Search incomplete due to permissions)"
             else:
                  return f"Error: Permission denied while searching in {start_dir}."
        except Exception as e:
            logger.error(f"Error finding files matching '{query}' in {start_dir}: {e}", exc_info=True)
            return f"Error finding files: {e}"

    async def _arun(self, query: str, directory: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, query, directory)


class CheckExist(BaseTool):
    name: str = "check_path_exists"
    description: str = "Checks if a given file or directory path exists in the file system."
    args_schema: Type[BaseModel] = PathSchema

    def _run(self, path: str) -> str:
        target_path = Path(path) # Don't resolve immediately, check exactly what user provided
        try:
            if target_path.exists():
                type_str = "directory" if target_path.is_dir() else "file" if target_path.is_file() else "special file"
                return f"Path exists: '{path}' (Type: {type_str})"
            else:
                return f"Path does not exist: '{path}'"
        except Exception as e: # Catch potential OS errors on checking invalid paths
            logger.error(f"Error checking existence of path '{path}': {e}", exc_info=True)
            return f"Error checking existence of path '{path}': {e}"

    async def _arun(self, path: str) -> str: return await run_in_executor(None, self._run, path)


class ShowHistory(BaseTool):
    name: str = "show_file_change_history"
    description: str = "Displays the recent history of file operations performed by this toolset in the current chat."
    args_schema: Type[BaseModel] = NoArgsSchema

    def _run(self) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."

        try:
            # Read history state for this chat
            state = _read_history_state(chat_id)
            history = state.get("history", [])

            if not history:
                return "No file operations recorded in history for this chat."

            # Read global config to get retrieval limit
            # Need to read the toolset's *own* config, not the main agent config
            config_path = get_toolset_data_path(chat_id, toolset_id) # Check local first
            tool_config = read_json(config_path, default_value=None)
            if tool_config is None: # Fallback to global if local not found
                 global_config_path = Path(f"data/toolsets/{toolset_id}.json") # Construct global path
                 tool_config = read_json(global_config_path, default_value=toolset_config_defaults)

            limit = tool_config.get("history_retrieval_limit", toolset_config_defaults["history_retrieval_limit"])

            # Get the last 'limit' items
            recent_history = history[-limit:]

            output = f"Recent File Operations (Last {len(recent_history)} of {len(history)} total):\n"
            for event in reversed(recent_history): # Show newest first
                ts = event.get('timestamp', 'Timestamp missing')
                op = event.get('operation', 'Unknown operation')
                path = event.get('path')
                from_p = event.get('from_path')
                to_p = event.get('to_path')
                backup = event.get('backup_path')
                summary = event.get('summary')

                ts_formatted = ""
                try: # Format timestamp nicely
                    ts_formatted = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime('%Y-%m-%d %H:%M:%S')
                except: pass

                line = f"- [{ts_formatted}] {op.upper()}"
                if path: line += f": {path}"
                if from_p: line += f": From {from_p}"
                if to_p: line += f" To {to_p}"
                if summary: line += f" (Summary: {summary})"
                if backup: line += f" [Backup: {backup}]"
                output += line + "\n"

            return output.strip()

        except Exception as e:
            logger.error(f"Error retrieving file history for chat {chat_id}: {e}", exc_info=True)
            return f"Error retrieving file history: {e}"

    async def _arun(self) -> str: return await run_in_executor(None, self._run)


class Restore(BaseTool):
    name: str = "restore_file_from_backup"
    description: str = "Restores a file to the state saved in its most recent backup created by the 'edit_file' tool. Requires user confirmation."
    args_schema: Type[BaseModel] = PathSchema
    requires_confirmation: bool = True

    def _run(self, path: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."
        target_path_str = path # Keep original for prompt/logging

        try:
            # Find the latest backup for this path
            state = _read_history_state(chat_id)
            history = state.get("history", [])
            latest_backup_path_str = None
            latest_backup_ts = None

            # Iterate backwards to find the most recent edit event for this path
            for event in reversed(history):
                if event.get("operation") == "edit" and event.get("path") == target_path_str:
                    backup_path = event.get("backup_path")
                    event_ts = event.get("timestamp")
                    if backup_path: # Found an edit event with a backup path
                        latest_backup_path_str = backup_path
                        latest_backup_ts = event_ts
                        break # Stop searching

            if not latest_backup_path_str:
                return f"Error: No backup found in history for file: {target_path_str}"

            backup_path = Path(latest_backup_path_str)
            if not backup_path.exists():
                return f"Error: Backup file recorded in history does not exist: {backup_path}"

            # --- Confirmation ---
            if confirmed_input is None:
                logger.debug(f"Requesting confirmation to restore '{target_path_str}' from backup '{backup_path}' (created around {latest_backup_ts})")
                # Use the path as the edit key, but rely on description/prompt context
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"path": target_path_str},
                    edit_key="path",
                    prompt_suffix=f"(Confirm restore from backup: {backup_path.name}) > "
                )
            else:
                # User confirmed, use the potentially edited path
                final_path_str = confirmed_input
                target_path = Path(final_path_str).resolve()
                logger.info(f"Executing confirmed restore of '{target_path}' from '{backup_path}'")

                if not target_path.is_file():
                     # Check if target exists *after* confirmation, maybe user deleted it?
                     if target_path.exists():
                          return f"Error: Target path exists but is not a file: {target_path}"
                     # If target doesn't exist, that's fine, restore will create it

                shutil.copy2(backup_path, target_path) # Restore by copying backup over

                log_data = {
                    "operation": "restore",
                    "path": str(target_path),
                    "backup_path": str(backup_path)
                }
                _log_history_event(chat_id, log_data)
                return f"Successfully restored {target_path} from backup {backup_path.name}"

        except Exception as e:
            logger.error(f"Error restoring file {target_path_str}: {e}", exc_info=True)
            return f"Error restoring file {target_path_str}: {e}"

    async def _arun(self, path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, confirmed_input)


class FinalizeChanges(BaseTool):
    name: str = "finalize_file_changes"
    description: str = "Deletes all backup files created by this toolset in the current chat session."
    args_schema: Type[BaseModel] = NoArgsSchema
    requires_confirmation: bool = True

    def _run(self, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return "Error: No active chat session."

        try:
            # Find all unique backup paths
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
                return "No backup files found to delete."

            # --- Confirmation ---
            if confirmed_input is None:
                logger.debug(f"Requesting confirmation to delete {num_backups} backup files for chat {chat_id}.")
                # Use a dummy edit key since we just need a yes/no
                raise PromptNeededError(
                    tool_name=self.name,
                    proposed_args={"confirm": f"Delete {num_backups} backup files?"}, # Show info in prompt
                    edit_key="confirm",
                    prompt_suffix="(Type 'yes' to confirm) > "
                )
            else:
                # User confirmed (we expect 'yes' or similar based on prompt)
                if confirmed_input.lower().strip() != 'yes':
                    return "Finalization cancelled by user."

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

                # Log finalization
                log_data = {
                    "operation": "finalize",
                    "deleted_backups": deleted_count,
                    "failed_deletions": errors
                }
                _log_history_event(chat_id, log_data)

                # Prepare result message
                result_msg = f"Finalized file changes. Deleted {deleted_count} of {num_backups} backup files."
                if errors:
                    result_msg += f"\nErrors occurred deleting: {', '.join(errors)}"
                return result_msg

        except Exception as e:
            logger.error(f"Error finalizing file changes for chat {chat_id}: {e}", exc_info=True)
            return f"Error finalizing file changes: {e}"

    async def _arun(self, confirmed_input: Optional[str] = None) -> str:
        # Need to pass the confirmed input correctly
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
    file_manager_usage_guide_tool, # Added
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