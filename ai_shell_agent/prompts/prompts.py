# ai_shell_agent/prompts/prompts.py
"""
Builds the final system prompt based on active toolsets.
"""
from typing import List, Optional

from .. import logger
from .default_prompt import BASE_SYSTEM_PROMPT
from .terminal_prompt import TERMINAL_PROMPT_SNIPPET
from .ai_editor_prompt import AI_EDITOR_TOOLSET_INTRO

# Define mapping from toolset names to their prompt snippets
TOOLSET_PROMPTS = {
    "Terminal": TERMINAL_PROMPT_SNIPPET,
    "AI Editor": AI_EDITOR_TOOLSET_INTRO,
    # Add other toolsets here if needed
}

# Define names of starter tools for messaging
# This duplicates info from llm.py, maybe centralize later?
TOOLSET_STARTERS_NAMES = {
     "Terminal": "start_terminal",
     "AI Editor": "start_ai_editor",
}

def build_prompt(active_toolsets: Optional[List[str]] = None) -> str:
    """
    Constructs the system prompt by combining the base prompt
    with snippets for the currently active toolsets.

    Args:
        active_toolsets: A list of names of the currently active toolsets.
                         Defaults to an empty list if None.

    Returns:
        The combined system prompt string.
    """
    if active_toolsets is None:
        active_toolsets = []
        logger.debug("No active toolsets provided to build_prompt, defaulting to empty list")
    else:
        logger.info(f"Building system prompt with active toolsets: {active_toolsets}")

    # Ensure active_toolsets contains unique elements (case-sensitive)
    active_toolsets = sorted(list(set(active_toolsets)))
    if len(active_toolsets) != len(set(active_toolsets)):
        logger.debug("Removed duplicate toolsets from active_toolsets")

    prompt_parts = [BASE_SYSTEM_PROMPT]
    prompt_parts.append("\n--- ACTIVE TOOLSETS ---")

    # Add snippets for active toolsets
    if active_toolsets:
        for toolset_name in active_toolsets:
            if toolset_name in TOOLSET_PROMPTS:
                logger.debug(f"Adding prompt snippet for active toolset: {toolset_name}")
                prompt_parts.append(f"\n# TOOLSET: {toolset_name}\n{TOOLSET_PROMPTS[toolset_name]}")
            else:
                logger.warning(f"No prompt snippet found for toolset: {toolset_name}")
                prompt_parts.append(f"\n# TOOLSET: {toolset_name} (No specific instructions available)")
    else:
        logger.debug("No active toolsets, adding placeholder message")
        prompt_parts.append("\nNo specific toolsets are currently active.")

    # Mention available starter tools for inactive sets
    inactive_sets = [name for name in TOOLSET_PROMPTS if name not in active_toolsets]
    if inactive_sets:
        logger.debug(f"Adding information about {len(inactive_sets)} inactive toolsets")
        prompt_parts.append("\n--- AVAILABLE TOOLSETS (Activate to Use) ---")
        for name in inactive_sets:
            starter = TOOLSET_STARTERS_NAMES.get(name)
            if starter:
                prompt_parts.append(f"- {name}: Activate using the `{starter}` tool.")
            else:
                logger.warning(f"No starter tool defined for toolset: {name}")
                prompt_parts.append(f"- {name}: (No specific starter tool defined)")

    prompt_parts.append("\n-------------------------\n") # Footer separator
    prompt_parts.append("Remember to use the available tools corresponding to the ACTIVE toolsets.")

    final_prompt = "\n".join(prompt_parts).strip()
    logger.debug(f"Built system prompt ({len(final_prompt)} chars) with {len(active_toolsets)} active toolsets")
    return final_prompt