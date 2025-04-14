# ai_shell_agent/prompts/prompts.py
"""
Builds the final system prompt based on enabled and active toolsets.
"""

from .default_prompt import BASE_SYSTEM_PROMPT
from .additional_prompt import FINAL_WORDS

SYSTEM_PROMPT = f"""\
{BASE_SYSTEM_PROMPT}

{FINAL_WORDS}
"""