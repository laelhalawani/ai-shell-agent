# =========================================================================
# File: ai_shell_agent/toolsets/aider/tools/schemas.py
# =========================================================================
"""
Pydantic schemas for the AI Code Copilot (Aider) toolset tools.
"""
from pydantic import BaseModel, Field
from typing import Optional, Type

# Import toolset-specific text getter
from ..texts import get_text

# --- Tool Input Schemas ---
class NoArgsSchema(BaseModel):
    """Input schema for tools that require no arguments."""
    pass

class FilePathSchema(BaseModel):
    """Input schema for tools accepting a file path."""
    file_path: str = Field(..., description=get_text("schemas.file_path.path_desc"))

class InstructionSchema(BaseModel):
    """Input schema for tools accepting a natural language instruction."""
    instruction: str = Field(..., description=get_text("schemas.instruction.instruction_desc"))

class UserResponseSchema(BaseModel):
    """Input schema for tools accepting a user response to a prompt."""
    user_response: str = Field(..., description=get_text("schemas.user_response.response_desc"))