# ai_shell_agent/prompts/prompts.py
"""
Builds the final system prompt based on active toolsets.
"""
from typing import List, Optional

from .. import logger
from .default_prompt import BASE_SYSTEM_PROMPT
from .terminal_prompt import TERMINAL_PROMPT_SNIPPET
from .file_editor_prompt import FILE_EDITOR_TOOLSET_INTRO
from .additional_prompt import FINAL_WORDS

# Define mapping from toolset names to their prompt snippets
TOOLSET_PROMPTS = {
    "Terminal": TERMINAL_PROMPT_SNIPPET,
    "File Editor": FILE_EDITOR_TOOLSET_INTRO,
    # Add other toolsets here if needed
}

# Define names of starter tools for messaging
# This duplicates info from llm.py, maybe centralize later?
TOOLSET_STARTERS_NAMES = {
     "Terminal": "start_terminal",
     "File Editor": "start_file_editor",
}

def build_prompt(active_toolsets: Optional[List[str]] = None) -> str:
    """
    Dynamically builds the system prompt based on active toolsets.
    
    Args:
        active_toolsets: List of active toolset names, e.g., ["Terminal", "File Editor"]
        
    Returns:
        Complete system prompt with all appropriate sections.
    """
    if active_toolsets is None:
        active_toolsets = []
        logger.debug("No active toolsets provided to build_prompt, defaulting to empty list")
    
    logger.info(f"Building system prompt with active toolsets: {active_toolsets}")
    
    # Get the base prompt that's always included
    prompt_parts = [BASE_SYSTEM_PROMPT]
    
    # Get all available inactive toolsets for information section
    all_toolset_names = set(TOOLSET_PROMPTS.keys())
    inactive_sets = all_toolset_names - set(active_toolsets)
    
    # Add prompts for active toolsets
    for toolset_name in active_toolsets:
        if toolset_name in TOOLSET_PROMPTS:
            logger.debug(f"Adding prompt snippet for active toolset: {toolset_name}")
            prompt_parts.append(TOOLSET_PROMPTS[toolset_name])
        else:
            logger.warning(f"Unknown toolset requested: '{toolset_name}'. Skipping.")
    
    # Add information about inactive toolsets that could be activated
    inactive_toolsets_info = []
    logger.debug(f"Adding information about {len(inactive_sets)} inactive toolsets")
    
    for toolset_name in sorted(inactive_sets):
        if toolset_name in TOOLSET_STARTERS_NAMES:
            starter_tool = TOOLSET_STARTERS_NAMES[toolset_name]
            inactive_toolsets_info.append(
                f"- {toolset_name}: can be started using `{starter_tool}`"
            )
    
    if inactive_toolsets_info:
        inactive_section = "**Additional Available Tools:**\n" + "\n".join(inactive_toolsets_info)
        prompt_parts.append(inactive_section)
    
    # Combine all parts with spacing
    final_prompt = "\n\n".join(prompt_parts)
    logger.debug(f"Built system prompt ({len(final_prompt)} chars) with {len(active_toolsets)} active toolsets")
    
    return final_prompt