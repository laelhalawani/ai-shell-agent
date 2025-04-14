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
    level=logging.INFO,
    format='%(levelname)s [%(filename)s:%(funcName)s]: %(message)s',
)
logger = logging.getLogger('ai_shell_agent')

# Import key modules to ensure they are initialized early
# This ensures tool registrations happen in the correct order
from . import tool_registry
from . import terminal_tools  # Import terminal_tools early to register its tools
from . import aider_integration_and_tools  # Also import aider tools early

# Get the root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
CHATS_DIR = DATA_DIR / 'chats'

# Create necessary directories if not exist
os.makedirs(CHATS_DIR, exist_ok=True)
