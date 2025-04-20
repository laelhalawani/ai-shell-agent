# ai_shell_agent/prompts/prompts.py
"""
Builds the final system prompt based on enabled and active toolsets.
"""

from .default_prompt import BASE_SYSTEM_PROMPT
from .additional_prompt import FINAL_WORDS

# --- Translation Prompt Template ---
TRANSLATION_PROMPT_TEMPLATE = """\
You are a senior {target_language} translator localizing software application labels and UI elements.
You are currently translating the value for the key '{translation_key}', which is part of the '{top_level_key}' section.
For context, here is the full '{top_level_key}' section:
{top_level_keys_value_dict_str}

Please translate the following text string:
'{object_text_path_value}'

Translate ONLY the text string itself. Preserve any original variable placeholders like `{{variable_name}}` or `%s` or `%d` exactly as they appear in the original string. Maintain the original meaning and tone appropriate for a CLI application UI.

Output ONLY the translated text string in {target_language}:"""

SYSTEM_PROMPT = f"""\
{BASE_SYSTEM_PROMPT}

{FINAL_WORDS}
"""