# ai_shell_agent/paths.py
"""
Defines core filesystem paths for the application.
This module should have minimal dependencies to avoid circular imports.
"""
from pathlib import Path

# Define ROOT_DIR based on this file's location
# Assumes paths.py is in ai_shell_agent/
ROOT_DIR = Path(__file__).parent.parent.resolve()