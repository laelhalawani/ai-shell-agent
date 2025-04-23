# =========================================================================
# File: ai_shell_agent/toolsets/files/tools/tools.py
# =========================================================================
"""
Tool definitions for the File Manager toolset.
"""
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Union

# Pydantic and Langchain
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor

# Local Imports from parent toolset and core app
from .... import logger
from ....errors import PromptNeededError
from ....chat_state_manager import get_current_chat
# Import state helper from the new state module
from ..state import _log_history_event, _read_toolset_state, _write_toolset_state # Import from new state module
from ..integration.find_logic import find_files_with_logic
from ..texts import get_text # Import toolset-specific texts
from ..settings import (
    FILES_HISTORY_LIMIT, FIND_FUZZY_DEFAULT, FIND_THRESHOLD_DEFAULT,
    FIND_LIMIT_DEFAULT, FIND_WORKERS_DEFAULT
)
from ..prompts import FILES_TOOLSET_PROMPT # Import prompt for usage guide
from .schemas import ( # Import schemas from the sibling file
    NoArgsSchema, PathSchema, RestorePathSchema, CreateSchema, OverwriteSchema,
    FindReplaceSchema, FromToSchema, RenameSchema, FindSchema
)
from ....utils.file_io import read_json # Need this for history tool config read

# --- Constants ---
AFFIRMATIVE_CONFIRMATIONS = ['yes', 'y', 'confirm']

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
            # --- Proposal Phase ---
            if not target_path.exists():
                return get_text("tools.delete_file_or_dir.error.not_exists", path=str(target_path))
            op_type = "directory" if target_path.is_dir() else "file"
            # --- MODIFICATION: Format the prompt string ---
            prompt_message = get_text(
                "tools.delete_file_or_dir.confirm_prompt",
                tool_name=self.name,
                type=op_type,
                path=str(target_path)
            )
            logger.debug(f"Requesting confirmation to delete {op_type}: '{target_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                # Pass the formatted prompt itself
                proposed_args={"confirmation_prompt": prompt_message, "_path": path},
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            # Check if user confirmed affirmatively
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                 logger.info(f"User rejected delete action for '{path}'")
                 return get_text("tools.common.info_rejected")

            # User confirmed, proceed using the original 'path' argument
            final_target_path = Path(path).resolve() # Re-resolve original path for safety
            logger.info(f"Executing confirmed delete for: '{final_target_path}'")

            if not final_target_path.exists():
                # Check again before deletion in case state changed
                logger.error(f"Target path {final_target_path} not found during delete confirmation.")
                return get_text("tools.delete_file_or_dir.error.target_missing", path=str(final_target_path))

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
                    # Should not happen if exists() check passed, but handle defensively
                    raise OSError(f"Path exists but is neither file nor directory: {final_target_path}")
            except (OSError, shutil.Error, PermissionError) as delete_e:
                 logger.error(f"Failed to delete {final_target_path}: {delete_e}", exc_info=True)
                 return get_text("tools.delete_file_or_dir.error.delete_failed", path=str(final_target_path), error=delete_e)
            except Exception as e:
                 logger.error(f"Unexpected error during delete operation for {final_target_path}: {e}", exc_info=True)
                 return get_text("tools.delete_file_or_dir.error.generic", path=str(final_target_path), error=e)

            _log_history_event(chat_id, log_data)
            return get_text("tools.delete_file_or_dir.success_confirm", type=op_type, path=str(final_target_path))

    async def _arun(self, path: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, confirmed_input)

class OverwriteFile(BaseTool):
    name: str = get_text("tools.overwrite_file.name")
    description: str = get_text("tools.overwrite_file.description")
    args_schema: Type[BaseModel] = OverwriteSchema
    requires_confirmation: bool = True

    def _run(self, path: str, new_content: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.overwrite_file.error.no_chat")
        target_path = Path(path).resolve()

        if confirmed_input is None:
            # --- Proposal Phase ---
            if not target_path.is_file():
                return get_text("tools.overwrite_file.error.not_file", path=str(target_path))

            # --- MODIFICATION: Format the prompt string ---
            max_snippet_len = 200 # Limit snippet length in prompt
            new_content_snippet = (new_content[:max_snippet_len] + "...") if len(new_content) > max_snippet_len else new_content
            prompt_message = get_text(
                "tools.overwrite_file.confirm_prompt",
                tool_name=self.name,
                file_path=str(target_path),
                new_content_snippet=new_content_snippet
            )

            logger.debug(f"Requesting confirmation to overwrite file '{target_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                # Pass formatted prompt and original args
                proposed_args={"confirmation_prompt": prompt_message, "_path": path, "_new_content": new_content},
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                 logger.info(f"User rejected overwrite action for '{path}'")
                 return get_text("tools.common.info_rejected")

            # User confirmed, proceed using the original 'path' and 'new_content' arguments
            final_target_path = Path(path).resolve() # Re-resolve original path
            final_new_content = new_content # Use original new_content

            logger.info(f"Executing confirmed overwrite for: '{final_target_path}'")

            if not final_target_path.is_file():
                logger.error(f"Target file {final_target_path} not found or not a file during overwrite confirmation.")
                return get_text("tools.overwrite_file.error.target_missing", path=str(final_target_path))

            backup_path = None
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                backup_path = final_target_path.with_suffix(f"{final_target_path.suffix}.bak.{timestamp}")
                shutil.copy2(final_target_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
                final_target_path.write_text(final_new_content, encoding='utf-8')
                logger.info(f"Successfully overwrote file '{final_target_path}'")
            except (FileNotFoundError, PermissionError, OSError, shutil.Error) as op_e:
                 if backup_path is None or not backup_path.exists():
                     logger.error(f"Failed to create backup for {final_target_path}: {op_e}", exc_info=True)
                     return get_text("tools.overwrite_file.error.backup_failed", path=str(final_target_path), error=op_e)
                 else:
                     logger.error(f"Failed to write new content to {final_target_path} after backup: {op_e}", exc_info=True)
                     return get_text("tools.overwrite_file.error.write_failed", path=str(final_target_path), error=op_e)
            except Exception as e:
                 logger.error(f"Unexpected error during overwrite operation for {final_target_path}: {e}", exc_info=True)
                 return get_text("tools.overwrite_file.error.generic", path=str(final_target_path), error=e)

            log_data = {
                "operation": "overwrite_confirmed", "path": str(final_target_path),
                "backup_path": str(backup_path) if backup_path else None,
            }
            _log_history_event(chat_id, log_data)
            return get_text("tools.overwrite_file.success_confirm", path=str(final_target_path), backup_path=str(backup_path))

    async def _arun(self, path: str, new_content: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, new_content, confirmed_input)

class FindAndReplaceInFile(BaseTool):
    name: str = get_text("tools.find_and_replace_in_file.name")
    description: str = get_text("tools.find_and_replace_in_file.description")
    args_schema: Type[BaseModel] = FindReplaceSchema
    requires_confirmation: bool = True

    def _run(self, path: str, find_text: str, replace_text: str, summary: str, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.find_and_replace_in_file.error.no_chat")
        target_path = Path(path).resolve()

        if confirmed_input is None:
            # --- Proposal Phase ---
            if not target_path.is_file():
                return get_text("tools.find_and_replace_in_file.error.not_file", path=str(target_path))

            # --- MODIFICATION: Format the prompt string ---
            prompt_message = get_text(
                "tools.find_and_replace_in_file.confirm_prompt",
                tool_name=self.name,
                file_path=str(target_path),
                summary=summary
            )
            logger.debug(f"Requesting confirmation for find/replace in '{target_path}' with summary: {summary}")
            raise PromptNeededError(
                tool_name=self.name,
                # Pass formatted prompt and original args
                proposed_args={
                    "confirmation_prompt": prompt_message,
                    "_path": path, "_find_text": find_text, "_replace_text": replace_text, "_summary": summary
                },
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                 logger.info(f"User rejected find/replace action for '{path}'")
                 return get_text("tools.common.info_rejected")

            # User confirmed, proceed using original arguments
            final_target_path = Path(path).resolve() # Re-resolve original path
            final_find_text = find_text
            final_replace_text = replace_text
            final_summary = summary # For logging

            logger.info(f"Executing confirmed find/replace on '{final_target_path}' with summary: {final_summary}")

            if not final_target_path.is_file():
                logger.error(f"Target file {final_target_path} not found or not a file during find/replace confirmation.")
                return get_text("tools.find_and_replace_in_file.error.target_missing", path=str(final_target_path))

            backup_path = None
            original_content = None
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                backup_path = final_target_path.with_suffix(f"{final_target_path.suffix}.bak.{timestamp}")
                shutil.copy2(final_target_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
                original_content = final_target_path.read_text(encoding='utf-8', errors='replace')
                new_content = original_content.replace(final_find_text, final_replace_text)

                if new_content == original_content:
                     # No change, delete backup and inform user
                     try:
                         if backup_path: backup_path.unlink(missing_ok=True)
                     except OSError: pass # Ignore error if backup deletion fails
                     logger.info(f"Find/replace resulted in no changes for file '{final_target_path}'")
                     return get_text("tools.find_and_replace_in_file.info_no_change", path=str(final_target_path))

                # Write the changed content
                final_target_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Successfully performed find/replace in file '{final_target_path}'")
            except (FileNotFoundError, PermissionError, OSError, shutil.Error) as op_e:
                 if backup_path is None or not backup_path.exists():
                      logger.error(f"Failed to create backup for {final_target_path}: {op_e}", exc_info=True)
                      return get_text("tools.find_and_replace_in_file.error.backup_failed", path=str(final_target_path), error=op_e)
                 elif original_content is None:
                      logger.error(f"Failed to read file {final_target_path} during find/replace: {op_e}", exc_info=True)
                      return get_text("tools.find_and_replace_in_file.error.write_failed", path=str(final_target_path), error=op_e)
                 else:
                      logger.error(f"Failed to write file {final_target_path} during find/replace: {op_e}", exc_info=True)
                      return get_text("tools.find_and_replace_in_file.error.write_failed", path=str(final_target_path), error=op_e)
            except Exception as e:
                 logger.error(f"Unexpected error during find/replace operation for {final_target_path}: {e}", exc_info=True)
                 return get_text("tools.find_and_replace_in_file.error.generic", path=str(final_target_path), error=e)

            log_data = {
                "operation": "find_replace_confirmed", "path": str(final_target_path),
                "backup_path": str(backup_path) if backup_path else None,
                "summary": final_summary, "find": final_find_text, "replace": final_replace_text
            }
            _log_history_event(chat_id, log_data)
            return get_text("tools.find_and_replace_in_file.success_confirm", path=str(final_target_path), backup_path=str(backup_path))

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
        dest_path_str = to_path

        if confirmed_input is None:
            # --- Proposal Phase ---
            # --- MODIFICATION: Format the prompt string ---
            prompt_message = get_text(
                "tools.copy.confirm_prompt",
                tool_name=self.name,
                from_path=from_path,
                to_path=to_path
            )
            logger.debug(f"Requesting confirmation to copy from '{from_path}' to '{to_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"confirmation_prompt": prompt_message, "_from_path": from_path, "_to_path": to_path},
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                 logger.info(f"User rejected copy action from '{from_path}' to '{to_path}'")
                 return get_text("tools.common.info_rejected")

            # User confirmed, proceed using the original 'from_path' and 'to_path' arguments
            source_path = Path(from_path).resolve()
            dest_path = Path(to_path).resolve()

            logger.info(f"Executing confirmed copy from '{source_path}' to '{dest_path}'")

            if not source_path.exists():
                return get_text("tools.copy.error.source_not_exists", path=str(source_path))
            if dest_path.exists():
                return get_text("tools.copy.error.dest_exists", path=str(dest_path))

            try:
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
            except (FileNotFoundError, PermissionError, OSError, shutil.Error) as copy_e:
                logger.error(f"Error copying {source_path} to {dest_path}: {copy_e}", exc_info=True)
                return get_text("tools.copy.error.generic", from_path=str(source_path), to_path=str(dest_path), error=copy_e)
            except Exception as e:
                logger.error(f"Unexpected error during copy operation for {source_path}: {e}", exc_info=True)
                return get_text("tools.copy.error.generic", from_path=str(source_path), to_path=str(dest_path), error=e)

            _log_history_event(chat_id, log_data)
            return get_text("tools.copy.success", type=op_type, from_path=str(source_path), to_path=str(dest_path))

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
        dest_path_str = to_path

        if confirmed_input is None:
            # --- Proposal Phase ---
            # --- MODIFICATION: Format the prompt string ---
            prompt_message = get_text(
                "tools.move.confirm_prompt",
                tool_name=self.name,
                from_path=from_path,
                to_path=to_path
            )
            logger.debug(f"Requesting confirmation to move from '{from_path}' to '{to_path}'")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"confirmation_prompt": prompt_message, "_from_path": from_path, "_to_path": to_path},
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                 logger.info(f"User rejected move action from '{from_path}' to '{to_path}'")
                 return get_text("tools.common.info_rejected")

            # User confirmed, proceed using the original 'from_path' and 'to_path' arguments
            source_path = Path(from_path).resolve()
            dest_path = Path(to_path).resolve()

            logger.info(f"Executing confirmed move from '{source_path}' to '{dest_path}'")

            if not source_path.exists():
                return get_text("tools.move.error.source_not_exists", path=str(source_path))
            if dest_path.exists():
                return get_text("tools.move.error.dest_exists", path=str(dest_path))

            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))
            except (FileNotFoundError, PermissionError, OSError, shutil.Error) as move_e:
                logger.error(f"Error moving {source_path} to {dest_path}: {move_e}", exc_info=True)
                return get_text("tools.move.error.generic", from_path=str(source_path), to_path=str(dest_path), error=move_e)
            except Exception as e:
                logger.error(f"Unexpected error during move operation for {source_path}: {e}", exc_info=True)
                return get_text("tools.move.error.generic", from_path=str(source_path), to_path=str(dest_path), error=e)

            log_data = { "operation": "move_confirmed", "from_path": str(source_path), "to_path": str(dest_path) }
            _log_history_event(chat_id, log_data)
            return get_text("tools.move.success", from_path=str(source_path), to_path=str(dest_path))

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
        original_path_str = path
        original_new_name = new_name.strip()

        if os.path.sep in original_new_name or (os.altsep and os.altsep in original_new_name):
            return get_text("tools.rename.error.invalid_new_name", new_name=original_new_name)
        if not original_new_name:
             return get_text("tools.rename.error.empty_new_name")

        if confirmed_input is None:
             # --- Proposal Phase ---
             # --- MODIFICATION: Format the prompt string ---
             prompt_message = get_text(
                 "tools.rename.confirm_prompt",
                 tool_name=self.name,
                 path=path,
                 new_name=original_new_name
             )
             logger.debug(f"Requesting confirmation to rename '{path}' to '{original_new_name}'")
             raise PromptNeededError(
                 tool_name=self.name,
                 proposed_args={"confirmation_prompt": prompt_message, "_path": path, "_new_name": original_new_name},
                 edit_key="confirmation_prompt",
                 prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
             )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                logger.info(f"User rejected rename action for '{path}'")
                return get_text("tools.common.info_rejected")

            # User confirmed, proceed using original arguments
            original_path = Path(path).resolve()
            final_new_name = new_name.strip() # Use original new_name

            logger.info(f"Executing confirmed rename of '{original_path}' to '{final_new_name}'")

            # Re-validate final_new_name just in case (although not editable in this flow)
            if not final_new_name: return get_text("tools.rename.error.empty_new_name")
            if os.path.sep in final_new_name or (os.altsep and os.altsep in final_new_name):
                 return get_text("tools.rename.error.invalid_new_name", new_name=final_new_name)

            new_path = original_path.with_name(final_new_name)

            if not original_path.exists():
                return get_text("tools.rename.error.path_not_exists", path=str(original_path))
            if new_path.exists():
                return get_text("tools.rename.error.target_exists", path=str(new_path))

            try:
                original_path.rename(new_path)
            except (FileNotFoundError, PermissionError, OSError) as rename_e:
                logger.error(f"Error renaming {original_path} to {new_path}: {rename_e}", exc_info=True)
                return get_text("tools.rename.error.generic", path=str(original_path), new_path=str(new_path), error=rename_e)
            except Exception as e:
                logger.error(f"Unexpected error during rename operation for {original_path}: {e}", exc_info=True)
                return get_text("tools.rename.error.generic", path=str(original_path), new_path=str(new_path), error=e)

            log_data = { "operation": "rename_confirmed", "from_path": str(original_path), "to_path": str(new_path) }
            _log_history_event(chat_id, log_data)
            return get_text("tools.rename.success", path=str(original_path), new_path=str(new_path))

    async def _arun(self, path: str, new_name: str, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path, new_name, confirmed_input)

class Find(BaseTool):
    name: str = get_text("tools.find.name")
    description: str = get_text("tools.find.description")
    args_schema: Type[BaseModel] = FindSchema
    requires_confirmation: bool = False

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
    requires_confirmation: bool = False

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
    requires_confirmation: bool = False

    def _run(self) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.history.error.no_chat")
        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            if not history: return get_text("tools.history.info_empty")
            limit = FILES_HISTORY_LIMIT # Default limit from settings

            # Attempt to read configured limit
            try:
                # Assume 'files' toolset uses its own ID for data path
                from ....chat_state_manager import get_toolset_data_path, get_toolset_global_config_path # Need global too
                from .... import ROOT_DIR # Import root for global fallback path construction

                # Try chat-specific config first
                config_path = get_toolset_data_path(chat_id, "files") # Use toolset_id
                tool_config = read_json(config_path, default_value=None)

                if tool_config is not None and "history_retrieval_limit" in tool_config:
                     limit = int(tool_config["history_retrieval_limit"])
                else: # Fallback to global toolset config
                    # Need to construct global path correctly
                    global_config_path = get_toolset_global_config_path("files") # Use toolset_id
                    global_config = read_json(global_config_path, default_value=None)
                    if global_config is not None and "history_retrieval_limit" in global_config:
                         limit = int(global_config["history_retrieval_limit"])
                if limit < 0: limit = 0
            except (ValueError, TypeError, FileNotFoundError) as e:
                logger.warning(f"Could not read history limit config for chat {chat_id}. Using default {FILES_HISTORY_LIMIT}. Error: {e}")
            # End reading configured limit

            recent_history = history[-limit:]
            total_history = len(history)
            actual_shown = len(recent_history)
            output = get_text("tools.history.header", count=actual_shown, total=total_history)
            for event in reversed(recent_history):
                ts = event.get('timestamp', 'Timestamp missing')
                op = event.get('operation', 'Unknown').replace("_confirmed", "")
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

class RestoreFile(BaseTool):
    name: str = get_text("tools.restore_file.name")
    description: str = get_text("tools.restore_file.description")
    args_schema: Type[BaseModel] = RestorePathSchema
    requires_confirmation: bool = True

    def _run(self, path_or_paths: Union[str, List[str]], backup_id: Optional[str] = None, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.restore_file.error.no_chat")
        target_paths_str = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
        if not target_paths_str: return get_text("tools.restore_file.error.no_paths")
        resolved_target_paths = [str(Path(p).resolve()) for p in target_paths_str]
        backup_filter = backup_id.strip() if backup_id else None

        # --- Logic to find backups (moved outside proposal/execution) ---
        backups_to_restore = {}
        errors = []
        found_backups_summary = []
        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            for target_path_str in resolved_target_paths:
                latest_match = None
                latest_match_ts = datetime.min.replace(tzinfo=timezone.utc)
                for event in reversed(history):
                    event_ts_str = event.get("timestamp")
                    event_path = event.get("path")
                    event_backup = event.get("backup_path")
                    event_op = event.get("operation")
                    # Check if it's an operation that creates a backup
                    is_edit_op = event_op in ["overwrite_confirmed", "find_replace_confirmed"]
                    if is_edit_op and event_path == target_path_str and event_backup:
                         if backup_filter and backup_filter not in event_backup: continue
                         try:
                             event_ts = datetime.fromisoformat(event_ts_str.replace("Z", "+00:00"))
                             if event_ts > latest_match_ts:
                                  if Path(event_backup).is_file():
                                       latest_match_ts = event_ts
                                       latest_match = event_backup
                                  else:
                                       logger.warning(f"Backup '{event_backup}' for '{target_path_str}' listed in history but not found on disk.")
                         except Exception as e:
                              logger.debug(f"Timestamp parse error or file check error: {e}")
                              continue
                if latest_match:
                    backups_to_restore[target_path_str] = latest_match
                    found_backups_summary.append(f"'{target_path_str}' from backup '{Path(latest_match).name}'")
                else:
                    filter_msg = f" matching '{backup_filter}'" if backup_filter else ""
                    errors.append(get_text("tools.restore_file.error.no_backup_found", path=target_path_str, filter=filter_msg))
        except Exception as find_e:
            logger.error(f"Error occurred while finding backups for restore: {find_e}", exc_info=True)
            return get_text("tools.restore_file.error.generic", error=f"finding backups: {find_e}")
        # --- End logic to find backups ---

        if confirmed_input is None:
            # --- Proposal Phase ---
            if not backups_to_restore:
                return "\n".join(errors) if errors else get_text("tools.restore_file.error.no_valid_backups")

            # --- MODIFICATION: Format the prompt string ---
            details_string = "\n".join(f"- {s}" for s in found_backups_summary)
            prompt_message = get_text(
                "tools.restore_file.confirm_prompt",
                tool_name=self.name,
                count=len(found_backups_summary),
                details=details_string
            )
            if errors:
                 prompt_message += "\n\nErrors/Warnings finding backups:\n" + "\n".join(f"- {e}" for e in errors)

            logger.debug(f"Requesting confirmation for restore:\n{prompt_message}")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={
                    "confirmation_prompt": prompt_message,
                    "_path_or_paths": path_or_paths, "_backup_id": backup_id
                 },
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                logger.info("Restore confirmation denied by user.")
                return get_text("tools.common.info_rejected")

            # User confirmed, proceed using the found backups_to_restore map
            logger.info(f"Executing confirmed restore for {len(backups_to_restore)} files.")
            results = []
            success_count = 0
            for target_path_str, backup_path_str in backups_to_restore.items():
                target_path = Path(target_path_str)
                backup_path = Path(backup_path_str)
                try:
                    if not backup_path.is_file():
                        raise FileNotFoundError(f"Backup disappeared before restore: {backup_path}")
                    if target_path.exists() and not target_path.is_file():
                        raise IsADirectoryError(f"Cannot restore: Target path exists and is not a file: {target_path}")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, target_path)
                    success_count += 1
                    results.append(get_text("tools.restore_file.success_single", path=str(target_path), backup_name=backup_path.name))
                    _log_history_event(chat_id, { "operation": "restore_confirmed", "path": str(target_path), "backup_path": str(backup_path) })
                except (FileNotFoundError, IsADirectoryError, PermissionError, OSError, shutil.Error) as restore_e:
                    logger.error(f"Failed to restore {target_path} from {backup_path}: {restore_e}", exc_info=True)
                    results.append(get_text("tools.restore_file.error.single_failed", path=str(target_path), error=restore_e))
                except Exception as e:
                     logger.error(f"Unexpected error restoring {target_path}: {e}", exc_info=True)
                     results.append(get_text("tools.restore_file.error.generic", error=f"restoring {target_path}: {e}"))

            # Use success count and original error list for final message
            final_message = get_text("tools.restore_file.success", count=len(backups_to_restore), success_count=success_count)
            if results: final_message += "\n" + "\n".join(results)
            if errors: final_message += "\n\nErrors/Warnings finding backups:\n" + "\n".join(f"- {e}" for e in errors)
            return final_message.strip()

    async def _arun(self, path_or_paths: Union[str, List[str]], backup_id: Optional[str] = None, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, path_or_paths, backup_id, confirmed_input)

class CleanupFileBackups(BaseTool):
    name: str = get_text("tools.cleanup_file_backups.name")
    description: str = get_text("tools.cleanup_file_backups.description")
    args_schema: Type[BaseModel] = NoArgsSchema
    requires_confirmation: bool = True

    def _run(self, confirmed_input: Optional[str] = None) -> str:
        chat_id = get_current_chat()
        if not chat_id: return get_text("tools.cleanup_file_backups.error.no_chat")

        # --- Logic to find backups (moved outside proposal/execution) ---
        backup_paths_to_delete = set()
        has_history = False
        try:
            state = _read_toolset_state(chat_id)
            history = state.get("history", [])
            has_history = bool(history)
            for event in history:
                is_backup_op = event.get("operation") in ["overwrite_confirmed", "find_replace_confirmed", "restore_confirmed"]
                if is_backup_op and event.get("backup_path"):
                    backup_paths_to_delete.add(event["backup_path"])
        except Exception as find_e:
             logger.error(f"Error finding backups/reading history for cleanup: {find_e}", exc_info=True)
             return get_text("tools.cleanup_file_backups.error.generic", error=f"finding backups: {find_e}")
        # --- End logic to find backups ---

        num_backups = len(backup_paths_to_delete)
        if num_backups == 0 and not has_history:
            return get_text("tools.cleanup_file_backups.info_no_work")

        if confirmed_input is None:
            # --- Proposal Phase ---
            # --- MODIFICATION: Format the prompt string ---
            if num_backups > 0:
                prompt_template = get_text("tools.cleanup_file_backups.prompt_confirm")
                prompt_message = prompt_template.format(tool_name=self.name, count=num_backups)
            else:
                prompt_template = get_text("tools.cleanup_file_backups.prompt_confirm_no_backups")
                prompt_message = prompt_template.format(tool_name=self.name)

            logger.debug(f"Requesting confirmation: '{prompt_message}' for chat {chat_id}.")
            raise PromptNeededError(
                tool_name=self.name,
                proposed_args={"confirmation_prompt": prompt_message},
                edit_key="confirmation_prompt",
                prompt_suffix=get_text("tools.common.confirm_suffix_yes_no")
            )
        else:
            # --- Execution Phase ---
            if confirmed_input.lower().strip() not in AFFIRMATIVE_CONFIRMATIONS:
                logger.info("Cleanup confirmation denied.")
                return get_text("tools.common.info_rejected")

            logger.info(f"Executing confirmed cleanup for chat {chat_id}.")
            deleted_count = 0
            errors = []
            delete_success = True

            if num_backups > 0:
                logger.info(f"Deleting {num_backups} backup files...")
                for backup_path_str in backup_paths_to_delete:
                    try:
                        backup_path = Path(backup_path_str)
                        if backup_path.is_file():
                            backup_path.unlink()
                            logger.debug(f"Deleted backup: {backup_path}")
                        else:
                            logger.warning(f"Backup path not found or not a file during cleanup: {backup_path}")
                        # Increment count even if file was already missing, as it's effectively "cleaned" from our perspective
                        deleted_count += 1
                    except (FileNotFoundError, PermissionError, OSError) as del_e:
                        # Log error but don't increment deleted_count if actual delete failed
                        logger.error(f"Error deleting backup {backup_path_str}: {del_e}")
                        errors.append(backup_path_str)
                        delete_success = False
                    except Exception as e:
                        logger.error(f"Unexpected error deleting backup {backup_path_str}: {e}", exc_info=True)
                        errors.append(backup_path_str)
                        delete_success = False

            history_cleared_successfully = False
            # Clear history ONLY if backup deletion was successful (or no backups to delete)
            if delete_success:
                try:
                    current_state = _read_toolset_state(chat_id)
                    current_state["history"] = []
                    if _write_toolset_state(chat_id, current_state):
                        history_cleared_successfully = True
                        logger.info(f"Cleared file history for chat {chat_id}.")
                    else:
                        logger.error(f"Failed to write state after clearing history for chat {chat_id}.")
                except Exception as clear_e:
                    logger.error(f"Error clearing history/writing state for chat {chat_id}: {clear_e}", exc_info=True)
            else:
                 logger.warning(f"Backup deletion failed. History NOT cleared for chat {chat_id}.")

            # Construct final message based on outcomes
            if delete_success and history_cleared_successfully:
                result_msg = get_text("tools.cleanup_file_backups.success", deleted_count=deleted_count, total_count=num_backups)
            elif delete_success and not history_cleared_successfully:
                result_msg = get_text("tools.cleanup_file_backups.warn_delete_success_history_fail", deleted_count=deleted_count, total_count=num_backups)
                if errors: result_msg += " " + get_text("tools.cleanup_file_backups.warn_errors_suffix", errors=', '.join(errors))
            else: # delete_success is False
                result_msg = get_text("tools.cleanup_file_backups.warn_delete_failed_history_kept", deleted_count=deleted_count, total_count=num_backups)
                result_msg += " " + get_text("tools.cleanup_file_backups.warn_errors_suffix", errors=', '.join(errors))

            return result_msg.strip()

    async def _arun(self, confirmed_input: Optional[str] = None) -> str:
        return await run_in_executor(None, self._run, confirmed_input)


# --- Instantiate Tools ---
file_manager_usage_guide_tool = FileManagerUsageGuideTool()
create_tool = Create()
read_tool = Read()
delete_tool = DeleteFileOrDir()
overwrite_file_tool = OverwriteFile()
find_replace_tool = FindAndReplaceInFile()
copy_tool = Copy()
move_tool = Move()
rename_tool = Rename()
find_tool = Find()
exists_tool = CheckExist()
history_tool = ShowHistory()
restore_tool = RestoreFile()
cleanup_tool = CleanupFileBackups()
