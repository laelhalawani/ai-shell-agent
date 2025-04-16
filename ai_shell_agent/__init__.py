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
#logging.disable(logging.CRITICAL)

# Get the root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
CHATS_DIR = DATA_DIR / 'chats'
# --- NEW: Define global toolset config directory ---
TOOLSETS_GLOBAL_CONFIG_DIR = DATA_DIR / 'toolsets'

# Create necessary directories if not exist
os.makedirs(CHATS_DIR, exist_ok=True)
# --- NEW: Ensure global toolset config directory exists ---
os.makedirs(TOOLSETS_GLOBAL_CONFIG_DIR, exist_ok=True)


# Import key modules to ensure they are initialized early
# This ensures tool registrations happen in the correct order
from . import tool_registry
# --- Import the central toolset registry which triggers discovery ---
from . import toolsets # ADDED IMPORT - Triggers discovery in toolsets/toolsets.py

logger.debug("AI Shell Agent package initialized.")
