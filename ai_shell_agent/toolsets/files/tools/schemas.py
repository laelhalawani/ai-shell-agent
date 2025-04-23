# =========================================================================
# File: ai_shell_agent/toolsets/files/tools/schemas.py
# =========================================================================
"""
Pydantic schemas for the File Manager toolset tools.
"""
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Type

# Import toolset-specific text getter
from ..texts import get_text

# --- Tool Schemas ---
class NoArgsSchema(BaseModel):
    """Input schema for tools that require no arguments."""
    pass

class PathSchema(BaseModel):
    path: str = Field(..., description=get_text("schemas.path.path_desc"))

class RestorePathSchema(BaseModel):
    path_or_paths: Union[str, List[str]] = Field(..., description=get_text("schemas.restore_file.paths_desc"))
    backup_id: Optional[str] = Field(None, description=get_text("schemas.restore_file.backup_id_desc"))

class CreateSchema(BaseModel):
    path: str = Field(..., description=get_text("schemas.create.path_desc"))
    content: Optional[str] = Field(None, description=get_text("schemas.create.content_desc"))
    is_directory: bool = Field(False, description=get_text("schemas.create.is_directory_desc"))

class OverwriteSchema(BaseModel):
    path: str = Field(..., description=get_text("schemas.overwrite_file.path_desc"))
    new_content: str = Field(..., description=get_text("schemas.overwrite_file.content_desc"))

class FindReplaceSchema(BaseModel):
    path: str = Field(..., description=get_text("schemas.find_replace.path_desc"))
    find_text: str = Field(..., description=get_text("schemas.find_replace.find_text_desc"))
    replace_text: str = Field(..., description=get_text("schemas.find_replace.replace_text_desc"))
    summary: str = Field(..., description=get_text("schemas.find_replace.summary_desc"))

class FromToSchema(BaseModel):
    from_path: str = Field(..., description=get_text("schemas.from_to.from_path_desc"))
    to_path: str = Field(..., description=get_text("schemas.from_to.to_path_desc"))

class RenameSchema(BaseModel):
    path: str = Field(..., description=get_text("schemas.rename.path_desc"))
    new_name: str = Field(..., description=get_text("schemas.rename.new_name_desc"))

class FindSchema(BaseModel):
    query: str = Field(..., description=get_text("schemas.find.query_desc"))
    directory: Optional[str] = Field(None, description=get_text("schemas.find.directory_desc"))