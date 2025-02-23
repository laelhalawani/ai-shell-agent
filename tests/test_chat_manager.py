import os
import json
import tempfile
import shutil
import pytest
from pathlib import Path

# Import functions from chat_manager
import shellai.chat_manager as chat_manager

@pytest.fixture(scope="function")
def temp_chat_env(tmp_path):
    """
    Setup a temporary environment for chat files:
      - Overrides CHAT_DIR, CHAT_MAP_FILE, SESSION_FILE, and CONFIG_FILE in chat_manager.
    Cleans up after tests.
    """
    # Create a temporary directory
    temp_dir = tmp_path / "chat_env"
    temp_dir.mkdir()
    
    # Override file paths in chat_manager
    chat_manager.CHAT_DIR = str(temp_dir / "chats")
    os.makedirs(chat_manager.CHAT_DIR, exist_ok=True)
    chat_manager.CHAT_MAP_FILE = str(temp_dir / "chat_map.json")
    chat_manager.SESSION_FILE = str(temp_dir / "session.json")
    chat_manager.CONFIG_FILE = str(temp_dir / "config.json")
    
    # Ensure fresh start
    for f in [chat_manager.CHAT_MAP_FILE, chat_manager.SESSION_FILE, chat_manager.CONFIG_FILE]:
        with open(f, "w") as fp:
            json.dump({}, fp)
    
    yield temp_dir
    # Cleanup is handled by pytest's tmp_path

def test_create_and_list_chat(temp_chat_env):
    title = "UnitTest Chat"
    chat_file = chat_manager.create_or_load_chat(title)
    assert os.path.exists(chat_file)
    
    # Listing chats should include our new chat
    chats = chat_manager.list_chats()
    assert title in chats

def test_rename_chat(temp_chat_env):
    old_title = "Old Chat"
    new_title = "Renamed Chat"
    # Create a chat first
    _ = chat_manager.create_or_load_chat(old_title)
    renamed = chat_manager.rename_chat(old_title, new_title)
    assert renamed is True
    chats = chat_manager.list_chats()
    assert new_title in chats
    assert old_title not in chats

def test_delete_chat(temp_chat_env):
    title = "Chat To Delete"
    chat_file = chat_manager.create_or_load_chat(title)
    # Ensure file exists
    assert os.path.exists(chat_file)
    deleted = chat_manager.delete_chat(title)
    assert deleted is True
    chats = chat_manager.list_chats()
    assert title not in chats
    # File should no longer exist
    assert not os.path.exists(chat_file)

def test_send_message_and_conversation_flow(temp_chat_env):
    title = "Conversation Test"
    chat_file = chat_manager.create_or_load_chat(title)
    chat_manager.save_session(chat_file)
    
    response = chat_manager.send_message("Hello, how are you?")
    assert response.startswith("AI:")
    
    # Read chat history from file and check that both user and AI messages are present.
    history = json.load(open(chat_file, "r"))
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello, how are you?"
    assert history[1]["role"] == "ai"
    assert "You said" in history[1]["content"]

def test_edit_message(temp_chat_env):
    title = "Edit Message Test"
    chat_file = chat_manager.create_or_load_chat(title)
    chat_manager.save_session(chat_file)
    # Send a message and then edit it
    _ = chat_manager.send_message("Original message")
    # Edit the AI response at index 1
    success = chat_manager.edit_message(1, "Edited AI response")
    assert success is True
    history = json.load(open(chat_file, "r"))
    assert history[1]["content"] == "Edited AI response"
    # Ensure only messages up to index 1 remain.
    assert len(history) == 2

def test_start_temp_chat():
    response = chat_manager.start_temp_chat("Temp message")
    assert "AI (temp):" in response

def test_set_and_update_system_prompt(temp_chat_env):
    # Test default system prompt update
    default_prompt = "Default system prompt for new chats."
    chat_manager.set_default_system_prompt(default_prompt)
    config = json.load(open(chat_manager.CONFIG_FILE, "r"))
    assert config.get("default_system_prompt") == default_prompt

    # Test updating system prompt in an active chat
    title = "System Prompt Test"
    chat_file = chat_manager.create_or_load_chat(title)
    chat_manager.save_session(chat_file)
    chat_manager.send_message("User message")
    new_system_prompt = "Updated system prompt."
    chat_manager.update_system_prompt(new_system_prompt)
    history = json.load(open(chat_file, "r"))
    assert history[0]["role"] == "system"
    assert history[0]["content"] == new_system_prompt
