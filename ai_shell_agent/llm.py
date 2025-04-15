# ai_shell_agent/llm.py
"""
Handles LLM instantiation, configuration, and dynamic tool binding
based on enabled/active toolsets for the current chat.
"""
from typing import List, Optional, Dict, Set

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage # For type hinting if needed later
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import BaseTool

from . import logger
from .config_manager import get_current_model, get_model_provider
# Import toolset registry and state manager functions
from .toolsets.toolsets import get_registered_toolsets, ToolsetMetadata # Correct import
from .chat_state_manager import get_current_chat, get_enabled_toolsets, get_active_toolsets # Correct import

# --- LLM Instantiation and Binding ---

def get_llm() -> BaseChatModel:
    """
    Get the LLM instance based on the current model configuration,
    binding tools dynamically based on the enabled/active toolsets
    for the *current chat session*.

    Returns:
        A configured LangChain BaseChatModel instance with appropriate tools bound.
        Returns LLM without tools if no chat session or no tools applicable.
    """
    chat_file = get_current_chat()
    if not chat_file:
        logger.warning("get_llm called without an active chat session. Returning LLM without tools.")
        enabled_toolsets_names = []
        active_toolsets_names = []
    else:
        # Fetch current state for this specific LLM invocation
        # Use display names from state manager
        enabled_toolsets_names = get_enabled_toolsets(chat_file)
        active_toolsets_names = get_active_toolsets(chat_file)

    logger.info(f"Preparing LLM for chat '{chat_file or 'None'}'. Enabled: {enabled_toolsets_names}, Active: {active_toolsets_names}")

    model_name = get_current_model()
    provider = get_model_provider(model_name)
    all_registered_toolsets: Dict[str, ToolsetMetadata] = get_registered_toolsets() # Dict[id, ToolsetMetadata]

    bound_tools: List[BaseTool] = []
    bound_tool_names: Set[str] = set() # Track names to avoid duplicates

    enabled_set = set(enabled_toolsets_names) # Set of display names
    active_set = set(active_toolsets_names) # Set of display names

    # Iterate through registered toolsets to decide which tools to bind
    for toolset_id, metadata in all_registered_toolsets.items():
        toolset_name = metadata.name # Use display name for checking against state
        if toolset_name in enabled_set:
            tools_to_bind_for_this_set = []
            if toolset_name in active_set:
                # Toolset is Enabled and Active: Bind its main tools AND the closer tool
                logger.debug(f"Adding tools for active toolset: '{toolset_name}'")
                tools_to_bind_for_this_set.extend(metadata.tools) # Add all main tools

                # Add closer tool (convention: close_<toolset_id>) if exists
                closer_tool_name = f"close_{toolset_id}"
                # Check if closer tool is in the list of tools for this toolset
                closer_tool_instance = next((t for t in metadata.tools if t.name == closer_tool_name), None)
                 # Special case for File Editor closer
                if metadata.id == "aider": closer_tool_instance = next((t for t in metadata.tools if t.name == "close_file_editor"), None)

                if closer_tool_instance:
                     # We expect the closer to be *part* of metadata.tools, so it should already be included.
                     # Log if found, but no need to add again unless it wasn't in the list somehow.
                     logger.debug(f"  - Closer tool '{closer_tool_instance.name}' identified.")
                     if closer_tool_instance not in tools_to_bind_for_this_set:
                          logger.warning(f"  - Adding closer tool {closer_tool_instance.name} explicitly (was missing from tool list?)")
                          tools_to_bind_for_this_set.append(closer_tool_instance)
                else:
                     logger.debug(f"  - No conventional closer tool found for '{toolset_name}'.")

            else:
                # Toolset is Enabled but Inactive: Bind only its starter tool
                if metadata.start_tool:
                    logger.debug(f"Adding starter tool for inactive enabled toolset: '{toolset_name}'")
                    tools_to_bind_for_this_set.append(metadata.start_tool)
                else:
                     logger.debug(f"Toolset '{toolset_name}' is enabled but has no starter tool defined.")

            # Add the selected tools for this set to the main list, checking duplicates
            for tool in tools_to_bind_for_this_set:
                if tool.name not in bound_tool_names:
                    logger.debug(f"  - Binding tool: {tool.name}")
                    bound_tools.append(tool)
                    bound_tool_names.add(tool.name)
                # else: Tool already bound from another set (less likely with this structure)

    # Instantiate the LLM
    llm: BaseChatModel
    if provider == "openai":
        logger.debug(f"Using OpenAI provider with model: {model_name}")
        llm = ChatOpenAI(model=model_name) # <-- temperature removed
    elif provider == "google":
        logger.debug(f"Using Google provider with model: {model_name}")
        # Keep convert_system_message_to_human=True if needed for Google models
        llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True) # <-- temperature removed
    else:
        logger.warning(f"Unsupported provider '{provider}'. Defaulting to OpenAI.")
        # Also remove temperature from the fallback
        llm = ChatOpenAI(model=model_name) # <-- temperature removed

    # Bind the selected tools
    if bound_tools:
        logger.info(f"Final tools bound to LLM ({len(bound_tools)}): {sorted(list(bound_tool_names))}")
        try:
             # Recommended method for models supporting native tool calling
             return llm.bind_tools(tools=bound_tools)
        except TypeError as e:
             logger.error(f"Failed bind_tools (model: {model_name}): {e}. Trying fallback.", exc_info=True)
             try: # Fallback using older method with OpenAI format tools
                 from langchain_core.utils.function_calling import convert_to_openai_tool
                 openai_tools = [convert_to_openai_tool(t) for t in bound_tools]
                 return llm.bind(tools=openai_tools) # Might work for some models/versions
             except Exception as bind_e:
                 logger.error(f"Fallback tool binding failed: {bind_e}. Returning LLM without tools.", exc_info=True)
                 return llm # Return raw LLM if all binding fails
    else:
        logger.warning("No tools were bound to the LLM for this chat/state.")
        return llm # Return LLM without tools