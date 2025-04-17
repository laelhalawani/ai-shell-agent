"""
AI Shell Agent package.

Provides AI-powered command line tools integration.
"""
import os
import logging
import sys
from pathlib import Path
# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Default to INFO level
    format='%(levelname)s [%(filename)s:%(funcName)s:%(lineno)d]: %(message)s',  # Added line number
)
logger = logging.getLogger('ai_shell_agent')

# Disable all logging messages
# logging.disable(logging.CRITICAL)

# Get the root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
CHATS_DIR = DATA_DIR / 'chats'
# Define global toolset config directory
TOOLSETS_GLOBAL_CONFIG_DIR = DATA_DIR / 'toolsets'

# Create necessary directories if not exist
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(TOOLSETS_GLOBAL_CONFIG_DIR, exist_ok=True)

# Import errors module for custom exceptions
from . import errors

# Import key modules to ensure they are initialized early
from . import tool_registry
# Import the console_manager (replaces console_io)
from . import console_manager
# Import the central toolset registry which triggers discovery
from . import toolsets

logger.debug("AI Shell Agent package initialized.")
