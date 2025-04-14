# ai_shell_agent/llm.py
"""
Handles LLM instantiation, configuration, and dynamic tool binding
based on active toolsets.
"""
from typing import List, Optional, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage # For type hinting if needed later
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import BaseTool

from . import logger
from .config_manager import get_current_model, get_model_provider
from .tool_registry import get_all_tools_dict, get_tool, get_all_tools # Import registry functions

# --- Toolset Definitions ---
# Tool names MUST match the 'name' attribute of the tool classes
TOOLSET_MEMBERS: Dict[str, List[str]] = {
    "Terminal": [
        "terminal",             # TerminalTool_HITL
        "python_repl",          # PythonREPLTool
        # "shell_windows_direct" # Keep internal? Or add if LLM needs direct exec? For now, keep out.
    ],
    "File Editor": [
        "include_file",         # AddFileTool
        "exclude_file",         # DropFileTool
        "list_files",           # ListFilesInEditorTool
        "request_edit",         # RunCodeEditTool
        "submit_editor_input",  # SubmitCodeEditorInputTool
        "view_diff",            # ViewDiffTool
        "undo_last_edit",       # UndoLastEditTool
        "close_file_editor",    # CloseCodeEditorTool
    ]
    # Add more toolsets here (e.g., "FileSystem": ["list_dir", "read_file", ...])
}

# Maps toolset name to the *name* of its starter tool
TOOLSET_STARTERS: Dict[str, str] = {
    "Terminal": "start_terminal",    # To be created
    "File Editor": "start_file_editor", # Exists in aider_integration.py
}

# --- LLM Instantiation and Binding ---

def get_llm(active_toolsets: Optional[List[str]] = None) -> BaseChatModel:
    """
    Get the LLM instance based on the current model configuration,
    binding tools dynamically based on the active toolsets.

    Args:
        active_toolsets: A list of names of the currently active toolsets.

    Returns:
        A configured LangChain BaseChatModel instance with tools bound.
    """
    if active_toolsets is None:
        active_toolsets = []
        logger.debug("No active toolsets provided, defaulting to empty list")
    else:
        logger.info(f"Preparing LLM with active toolsets: {active_toolsets}")

    # Use set for faster lookups
    active_toolsets = set(active_toolsets)

    model_name = get_current_model()
    provider = get_model_provider(model_name)
    all_tools_dict = get_all_tools_dict() # Get all registered tools

    bound_tools: List[BaseTool] = []
    bound_tool_names: set[str] = set() # Keep track of added tools

    logger.debug(f"Available registered tools: {list(all_tools_dict.keys())}")

    # 1. Add tools from ACTIVE toolsets
    for toolset_name in active_toolsets:
        if toolset_name in TOOLSET_MEMBERS:
            logger.debug(f"Processing active toolset: '{toolset_name}'")
            for tool_name in TOOLSET_MEMBERS[toolset_name]:
                tool = all_tools_dict.get(tool_name)
                if tool and tool_name not in bound_tool_names:
                    logger.debug(f"Adding active tool: '{tool_name}' from toolset '{toolset_name}'")
                    bound_tools.append(tool)
                    bound_tool_names.add(tool_name)
                elif not tool:
                    logger.warning(f"Tool '{tool_name}' in active toolset '{toolset_name}' not found in registry.")
                else:
                    logger.debug(f"Tool '{tool_name}' already added, skipping duplicate")
        else:
            logger.warning(f"Unknown toolset name: '{toolset_name}' (not defined in TOOLSET_MEMBERS)")

    # 2. Add STARTER tools for INACTIVE toolsets
    for toolset_name, starter_tool_name in TOOLSET_STARTERS.items():
         if toolset_name not in active_toolsets:
             logger.debug(f"Processing inactive toolset: '{toolset_name}' with starter tool: '{starter_tool_name}'")
             tool = all_tools_dict.get(starter_tool_name)
             if tool and starter_tool_name not in bound_tool_names:
                 logger.debug(f"Adding starter tool: '{starter_tool_name}' for inactive toolset '{toolset_name}'")
                 bound_tools.append(tool)
                 bound_tool_names.add(starter_tool_name)
             elif not tool:
                 logger.warning(f"Starter tool '{starter_tool_name}' for toolset '{toolset_name}' not found in registry.")
             else:
                 logger.debug(f"Starter tool '{starter_tool_name}' already added, skipping duplicate")

    # Convert selected tools to the format expected by the LLM provider
    if not bound_tools:
         logger.warning("No tools could be bound to the LLM based on active/inactive toolsets.")
         # Decide behavior: return LLM without tools or raise error? Let's return without tools.

    tool_functions = [convert_to_openai_function(t) for t in bound_tools]
    
    logger.info(f"Final tools bound to LLM ({len(bound_tools)}): {[t.name for t in bound_tools]}")

    # Instantiate the LLM
    llm: BaseChatModel
    if provider == "openai":
        logger.debug(f"Using OpenAI provider with model: {model_name}")
        llm = ChatOpenAI(model=model_name)
    elif provider == "google":
        logger.debug(f"Using Google provider with model: {model_name}")
        llm = ChatGoogleGenerativeAI(model=model_name)
    else:
        logger.warning(f"Unsupported provider '{provider}' for model '{model_name}'. Defaulting to OpenAI.")
        llm = ChatOpenAI(model=model_name) # Default fallback

    # Bind the tools
    if tool_functions:
        logger.debug(f"Binding {len(tool_functions)} tools to LLM")
        return llm.bind_tools(tool_functions)
    else:
        logger.warning("No tools were bound to the LLM - returning raw LLM instance")
        return llm # Return LLM without tools if none were selected/found