# ai_shell_agent/paths.py
"""
Defines core filesystem paths for the application.
This module should have minimal dependencies to avoid circular imports.
"""
from pathlib import Path

# Define ROOT_DIR based on this file's location
# Assumes paths.py is in ai_shell_agent/
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Directory where the package (ai_shell_agent) is installed
# Path(__file__).resolve().parent will point to .../site-packages/ai_shell_agent/
PACKAGE_INSTALL_DIR = Path(__file__).resolve().parent
DEFAULT_DOTENV_PATH = PACKAGE_INSTALL_DIR / ".env"