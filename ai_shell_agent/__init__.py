import os
import logging

# Set up logging
logger = logging.getLogger("ai_shell_agent")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create data directory in the installation directory
def ensure_data_directories():
    """Ensure that the necessary data directories exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    chat_dir = os.path.join(data_dir, "chats")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(chat_dir, exist_ok=True)
    
    return data_dir, chat_dir

# Initialize directories
ensure_data_directories()
