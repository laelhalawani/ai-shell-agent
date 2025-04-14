import os
import pytest
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from ai_shell_agent.aider_integration_and_tools import (
    AiderIOStub,
    recreate_coder,
    update_aider_from_coder,
    start_code_editor_tool,
    add_code_file_tool,
    drop_code_file_tool,
    list_code_files_tool,
    edit_code_tool,
    view_diff_tool
)

# Setup fixtures

@pytest.fixture(scope="function")
def temp_chat_env(tmp_path):
    """Setup temporary environment for chat files and test files."""
    temp_dir = tmp_path / "test_aider_env"
    temp_dir.mkdir()
    
    # Create temp test files
    test_files_dir = temp_dir / "test_files"
    test_files_dir.mkdir()
    
    # Create a sample file to use with the editor
    sample_file = test_files_dir / "sample.py"
    with open(sample_file, "w") as f:
        f.write("def hello():\n    return 'Hello, world!'\n")
    
    # Mock chat file
    chat_file = temp_dir / "test_chat.json"
    
    # Create a minimal initial chat state with aider_state
    initial_data = {
        "messages": [
            {"type": "system", "content": "Test system prompt"}
        ],
        "aider_state": {
            "enabled": True,
            "main_model_name": "gpt-4o-mini",
            "edit_format": "diff",
            "abs_fnames": [],
            "abs_read_only_fnames": [],
            "aider_commit_hashes": [],
            "git_root": None,
            "auto_commits": True
        }
    }
    
    with open(chat_file, "w") as f:
        json.dump(initial_data, f)
    
    # Yield the directory and chat file path
    yield {"temp_dir": temp_dir, "chat_file": str(chat_file), "test_files_dir": test_files_dir}
    
    # No cleanup as pytest handles tmp_path

@pytest.fixture
def mock_get_current_chat():
    """Mock for get_current_chat function."""
    with patch("ai_shell_agent.chat_manager.get_current_chat") as mock:
        yield mock

# Tests for AiderIOStub

def test_aider_io_stub_output():
    """Test that AiderIOStub captures output correctly."""
    io = AiderIOStub()
    io.tool_output("Test message")
    io.tool_warning("Warning message")
    io.tool_error("Error message")
    
    # Get output
    output = io.get_captured_output()
    
    assert "Test message" in output
    assert "WARNING: Warning message" in output
    assert "ERROR: Error message" in output

def test_aider_io_stub_confirm():
    """Test that AiderIOStub auto-confirms."""
    io = AiderIOStub()
    result = io.confirm_ask("Confirm?")
    assert result is True

# Tests for editor tools

@patch("ai_shell_agent.aider_integration.recreate_coder")
def test_start_code_editor_tool(mock_recreate, mock_get_current_chat, temp_chat_env):
    """Test the start_code_editor_tool."""
    # Setup
    chat_file = temp_chat_env["chat_file"]
    mock_get_current_chat.return_value = chat_file
    
    # Execute
    result = start_code_editor_tool._run()
    
    # Verify
    assert "Code editor initialized" in result
    
    # Read back the chat file to check that state was updated
    with open(chat_file, "r") as f:
        data = json.load(f)
        assert data["aider_state"]["enabled"] is True

@patch("ai_shell_agent.aider_integration.recreate_coder")
def test_add_code_file_tool(mock_recreate, mock_get_current_chat, temp_chat_env):
    """Test adding a file to the code editor."""
    # Setup
    chat_file = temp_chat_env["chat_file"]
    mock_get_current_chat.return_value = chat_file
    
    # Create a mock coder
    mock_coder = MagicMock()
    mock_coder.abs_fnames = set()
    mock_coder.get_rel_fname = lambda x: os.path.basename(x)
    
    def mock_add_rel_fname(fname):
        # Simulate adding a file to the coder
        mock_coder.abs_fnames.add(str(Path(temp_chat_env["test_files_dir"]) / fname))
        return True
    
    mock_coder.add_rel_fname = mock_add_rel_fname
    mock_recreate.return_value = mock_coder
    
    # Execute - add the sample file
    sample_file_path = str(Path(temp_chat_env["test_files_dir"]) / "sample.py")
    result = add_code_file_tool._run(sample_file_path)
    
    # Verify
    assert "Successfully added" in result
    
    # Check that the file was added to state
    with open(chat_file, "r") as f:
        data = json.load(f)
        assert sample_file_path in data["aider_state"]["abs_fnames"]

@patch("ai_shell_agent.aider_integration.recreate_coder")
def test_list_code_files_tool(mock_recreate, mock_get_current_chat, temp_chat_env):
    """Test listing files in the code editor."""
    # Setup
    chat_file = temp_chat_env["chat_file"]
    mock_get_current_chat.return_value = chat_file
    
    # Read the chat data and add a file to the state
    with open(chat_file, "r") as f:
        data = json.load(f)
    
    sample_file_path = str(Path(temp_chat_env["test_files_dir"]) / "sample.py")
    data["aider_state"]["abs_fnames"] = [sample_file_path]
    
    with open(chat_file, "w") as f:
        json.dump(data, f)
    
    # Execute
    result = list_code_files_tool._run()
    
    # Verify
    assert "Files in code editor" in result
    assert "sample.py" in result

@patch("ai_shell_agent.aider_integration.recreate_coder")
def test_drop_code_file_tool(mock_recreate, mock_get_current_chat, temp_chat_env):
    """Test removing a file from the code editor."""
    # Setup
    chat_file = temp_chat_env["chat_file"]
    mock_get_current_chat.return_value = chat_file
    
    # Add a file to the state
    with open(chat_file, "r") as f:
        data = json.load(f)
    
    sample_file_path = str(Path(temp_chat_env["test_files_dir"]) / "sample.py")
    data["aider_state"]["abs_fnames"] = [sample_file_path]
    
    with open(chat_file, "w") as f:
        json.dump(data, f)
    
    # Create a mock coder
    mock_coder = MagicMock()
    mock_coder.abs_fnames = {sample_file_path}
    mock_coder.get_rel_fname = lambda x: os.path.basename(x)
    
    def mock_drop_rel_fname(fname):
        # Simulate dropping a file from the coder
        if sample_file_path in mock_coder.abs_fnames:
            mock_coder.abs_fnames.remove(sample_file_path)
            return True
        return False
    
    mock_coder.drop_rel_fname = mock_drop_rel_fname
    mock_recreate.return_value = mock_coder
    
    # Execute
    result = drop_code_file_tool._run(sample_file_path)
    
    # Verify
    assert "Successfully dropped" in result
    
    # Check that the file was removed from state
    with open(chat_file, "r") as f:
        data = json.load(f)
        assert sample_file_path not in data["aider_state"]["abs_fnames"]

# Test for edit_code_tool would be more complex and require more mocking
# of the Aider coder, which might be beyond the scope of unit testing.
# Consider integration tests for this functionality.