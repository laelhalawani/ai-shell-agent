"""
AI Shell Agent package.

Provides AI-powered command line tools integration.
"""
import os
import logging # Keep logging import
import sys
from pathlib import Path

# --- Define Logger Object FIRST ---
# Get the specific logger for this application
logger = logging.getLogger('ai_shell_agent')
# Set the logging level EARLY - handler level will respect this later
logger.setLevel(logging.DEBUG) # Change to logging.INFO for less verbose output
# Prevent propagation EARLY
logger.propagate = False
# --- END Define Logger Object FIRST ---

# --- Import Rich Handler and Console Manager AFTER logger object exists ---
from rich.logging import RichHandler
# Now it's safe to import console_manager as it can import the 'logger' object above
from .console_manager import get_console_manager
# --- END Import Rich Handler and Console Manager ---


# --- Import Paths AFTER logger exists but BEFORE modules needing paths ---
from .paths import ROOT_DIR # <--- IMPORT ROOT_DIR

# --- Setup Paths and Directories (using imported ROOT_DIR) ---
DATA_DIR = ROOT_DIR / 'data'
CHATS_DIR = DATA_DIR / 'chats'
TOOLSETS_GLOBAL_CONFIG_DIR = DATA_DIR / 'toolsets'

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(TOOLSETS_GLOBAL_CONFIG_DIR, exist_ok=True)
# --- END Setup Paths and Directories ---


# --- Logging Configuration with Rich - Handler Setup ---
# 1. Get the console instance from the manager
# Ensure the console manager is initialized
console_manager_instance = get_console_manager()
rich_console = console_manager_instance.console # Get the rich.Console instance

# 2. Configure the RichHandler
# Use console's stderr setting, enable markup and tracebacks
# show_path=False makes logs a bit cleaner for this app, keep True if you prefer file:line
rich_handler = RichHandler(
    console=rich_console,
    level=logging.DEBUG, # Set handler level explicitly too (or omit to inherit logger level)
    show_time=False, # Time is less critical for CLI interaction logs
    show_level=True,
    show_path=True, # Set back to True for debugging line numbers
    markup=True, # Allow rich markup in log messages
    rich_tracebacks=True, # Use rich for formatting tracebacks
    tracebacks_show_locals=False # Set True to see local variables in tracebacks (verbose)
)

# 3. Add the RichHandler to the logger object created earlier
logger.addHandler(rich_handler)

# --- End Logging Configuration ---

# Import errors module for custom exceptions
from . import errors

# Import key modules to ensure they are initialized early
from . import tool_registry
# Import the central toolset registry which triggers discovery
from . import toolsets

logger.debug("AI Shell Agent package initialized with Rich logging.")

# --- Optional: Disable logging ---
# Uncomment the line below to disable all logging messages below CRITICAL
# logging.disable(logging.CRITICAL)
# --- End Optional: Disable logging ---
