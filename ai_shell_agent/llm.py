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
from .config_manager import ConfigManager, get_api_key, get_current_model, get_model_provider, get_translation_model 
# Import toolset registry and state manager functions
from .toolsets.toolsets import get_registered_toolsets, ToolsetMetadata # Correct import
from .chat_state_manager import get_current_chat, get_enabled_toolsets # get_active_toolsets should NOT be here

# --- LLM Instantiation and Binding ---

def get_llm() -> BaseChatModel:
    """
    Get the LLM instance based on the current model configuration,
    binding tools dynamically based on the *enabled* toolsets
    for the current chat session.

    Returns:
        A configured LangChain BaseChatModel instance with appropriate tools bound.
        Returns LLM without tools if no chat session or no tools applicable.
    """
    chat_file = get_current_chat()
    if not chat_file:
        logger.warning("get_llm called without an active chat session. Returning LLM without tools.")
        enabled_toolsets_names = []
    else:
        # Fetch current state for this specific LLM invocation
        # Use display names from state manager
        enabled_toolsets_names = get_enabled_toolsets(chat_file)
        # Ensure active_toolsets_names is NOT fetched here

    logger.info(f"Preparing LLM for chat '{chat_file or 'None'}'. Enabled: {enabled_toolsets_names}")

    model_name = get_current_model()
    provider = get_model_provider(model_name)
    all_registered_toolsets: Dict[str, ToolsetMetadata] = get_registered_toolsets() # Dict[id, ToolsetMetadata]

    bound_tools: List[BaseTool] = []
    bound_tool_names: Set[str] = set() # Track names to avoid duplicates

    enabled_set = set(enabled_toolsets_names) # Set of display names
    # Ensure active_set is NOT defined or used here

    # Iterate through registered toolsets to decide which tools to bind
    for toolset_id, metadata in all_registered_toolsets.items():
        toolset_name = metadata.name # Use display name for checking against state
        if toolset_name in enabled_set:
            # If toolset is enabled, bind ALL its tools
            logger.debug(f"Adding tools for enabled toolset: '{toolset_name}'")

            # Iterate through all tools defined in the toolset's metadata
            for tool in metadata.tools: # metadata.tools now includes the UsageGuideTool
                if tool.name not in bound_tool_names:
                    logger.debug(f"  - Binding tool: {tool.name}")
                    bound_tools.append(tool)
                    bound_tool_names.add(tool.name)
                # else: Tool already bound
        # Ensure there is NO 'else:' block here related to inactive toolsets

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

def get_llm_plain() -> Optional[BaseChatModel]:
    """
    Get a basic LLM instance based on the current model configuration,
    WITHOUT binding any tools. Used for general tasks.

    Returns:
        A LangChain BaseChatModel instance or None if instantiation fails.
    """
    model_name = get_current_model()
    provider = get_model_provider(model_name)
    logger.info(f"Preparing plain LLM instance (Provider: {provider}, Model: {model_name})")

    llm: Optional[BaseChatModel] = None
    try:
        if provider == "openai":
            llm = ChatOpenAI(model=model_name)
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)
        else:
            logger.warning(f"Unsupported provider '{provider}' for plain LLM. Defaulting to OpenAI.")
            llm = ChatOpenAI(model=model_name)
        return llm
    except Exception as e:
        logger.error(f"Failed to instantiate plain LLM (Provider: {provider}, Model: {model_name}): {e}", exc_info=True)
        return None

def get_translation_llm() -> Optional[BaseChatModel]:
    """
    Get a basic LLM instance specifically for translation tasks,
    based on the configured translation model, WITHOUT binding any tools.

    Returns:
        A LangChain BaseChatModel instance or None if instantiation fails.
    """
    # Get the specifically configured translation model
    model_name = get_translation_model()
    provider = get_model_provider(model_name)
    logger.info(f"Preparing translation LLM instance (Provider: {provider}, Model: {model_name})")

    llm: Optional[BaseChatModel] = None
    try:
        # Instantiate based on provider for the translation model
        if provider == "openai":
            llm = ChatOpenAI(model=model_name)
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)
        else:
            logger.warning(f"Unsupported provider '{provider}' for translation LLM. Defaulting to OpenAI.")
            llm = ChatOpenAI(model=model_name)
        return llm
    except Exception as e:
        logger.error(f"Failed to instantiate translation LLM (Provider: {provider}, Model: {model_name}): {e}", exc_info=True)
        return None