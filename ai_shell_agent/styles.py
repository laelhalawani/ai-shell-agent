# ai_shell_agent/styles.py
"""
Loads console styling (colors, Rich styles, PTK styles) from default_style.json
and provides them as constants and objects. Assumes the JSON file exists and is valid.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from rich.style import Style as RichStyle
from prompt_toolkit.styles import Style as PTKStyle, merge_styles, StyleTransformation
from prompt_toolkit.filters import has_focus, is_done
from .utils.file_io import read_json
from . import logger

_styles_file = Path(__file__).parent / 'styles' / 'default_style.json'


_styles_data = read_json(_styles_file) # Load the JSON file

# Initialize containers
_colors: Dict[str, str] = {}
_rich_styles_objects: Dict[str, RichStyle] = {}
_ptk_styles_dict: Dict[str, str] = {}
_DEFAULT_RICH_STYLE = RichStyle() # Keep a basic default for error cases during processing

def _process_styles():
    """Processes the loaded style data."""
    global _colors, _rich_styles_objects, _ptk_styles_dict

    if not _styles_data: # Check if loading failed
        logger.error("Style data is empty, cannot process styles.")
        return

    try:
        _colors = _styles_data.get('colors', {})
        if not _colors: logger.warning(f"No 'colors' section found in {_styles_file}. Default colors will be used.")

        # --- Process Rich Styles ---
        rich_styles_config = _styles_data.get('rich_styles', {})
        for name, config in rich_styles_config.items():
            if not isinstance(config, dict):
                logger.warning(f"Invalid Rich style config for '{name}' (expected dict, got {type(config)}). Skipping.")
                continue
            try:
                resolved_config = config.copy()
                if 'color' in resolved_config:
                    # Directly access color, assuming key exists after _colors loaded
                    color_ref = resolved_config['color']
                    if isinstance(color_ref, str) and color_ref.startswith("colors."):
                         color_name = color_ref.split('.', 1)[1]
                         resolved_config['color'] = _colors.get(color_name, color_ref) # Use original if not found
                    # else: use value directly if not a ref
                style_args = {k: resolved_config.get(k) for k in ['color', 'bgcolor', 'bold', 'italic', 'underline', 'strike', 'dim'] if resolved_config.get(k) is not None}
                _rich_styles_objects[name.upper()] = RichStyle(**style_args)
            except Exception as e:
                logger.error(f"Error processing Rich style '{name}': {e}")
                _rich_styles_objects[name.upper()] = _DEFAULT_RICH_STYLE

        # --- Process PTK Styles ---
        ptk_styles_config = _styles_data.get('ptk_styles', {})
        for name, definition in ptk_styles_config.items():
            if not isinstance(definition, str):
                logger.warning(f"Invalid PTK style definition for '{name}' (expected str, got {type(definition)}). Skipping.")
                continue
            try:
                # Resolve colors within the definition string
                parts = definition.split()
                resolved_parts = []
                for part in parts:
                    if ':' in part:
                        prefix, value = part.split(':', 1)
                        resolved_color = value
                        if value.startswith("colors."):
                            color_name = value.split('.', 1)[1]
                            resolved_color = _colors.get(color_name, value)
                        resolved_parts.append(f"{prefix}:{resolved_color}")
                    else:
                        resolved_parts.append(part)
                _ptk_styles_dict[name] = " ".join(resolved_parts)
            except Exception as e:
                logger.error(f"Error processing PTK style '{name}': {e}")
                _ptk_styles_dict[name] = definition # Use original on error

    except Exception as e:
        logger.critical(f"Error processing loaded styles data: {e}. Styling will be incomplete.", exc_info=True)


# --- Process loaded styles ---
_process_styles()


# --- Define Constants directly from processed data ---
# These will raise KeyError if processing failed or keys are missing in JSON

try:
    # Export Colors
    COLOR_AI = _colors['ai']
    COLOR_USER = _colors['user']
    COLOR_INFO = _colors['info']
    COLOR_WARNING = _colors['warning']
    COLOR_ERROR = _colors['error']
    COLOR_SYSTEM = _colors['system']
    COLOR_TOOL = _colors['tool']
    COLOR_COMMAND = _colors['command']
    COLOR_DIM_TEXT = _colors['dim_text']
    COLOR_NEUTRAL = _colors['neutral']
    COLOR_CODE = _colors['code'] # Assuming 'code' is defined in colors

    # Export Rich Style Objects (use .get with default for safety after processing)
    STYLE_AI_LABEL = _rich_styles_objects.get('AI_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_AI_CONTENT = _rich_styles_objects.get('AI_CONTENT', _DEFAULT_RICH_STYLE)
    STYLE_USER_LABEL = _rich_styles_objects.get('USER_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_INFO_LABEL = _rich_styles_objects.get('INFO_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_INFO_CONTENT = _rich_styles_objects.get('INFO_CONTENT', _DEFAULT_RICH_STYLE)
    STYLE_WARNING_LABEL = _rich_styles_objects.get('WARNING_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_WARNING_CONTENT = _rich_styles_objects.get('WARNING_CONTENT', _DEFAULT_RICH_STYLE)
    STYLE_ERROR_LABEL = _rich_styles_objects.get('ERROR_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_ERROR_CONTENT = _rich_styles_objects.get('ERROR_CONTENT', _DEFAULT_RICH_STYLE)
    STYLE_SYSTEM_LABEL = _rich_styles_objects.get('SYSTEM_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_SYSTEM_CONTENT = _rich_styles_objects.get('SYSTEM_CONTENT', _DEFAULT_RICH_STYLE)
    STYLE_TOOL_NAME = _rich_styles_objects.get('TOOL_NAME', _DEFAULT_RICH_STYLE)
    STYLE_ARG_NAME = _rich_styles_objects.get('ARG_NAME', _DEFAULT_RICH_STYLE)
    STYLE_ARG_VALUE = _rich_styles_objects.get('ARG_VALUE', _DEFAULT_RICH_STYLE)
    STYLE_THINKING = _rich_styles_objects.get('THINKING', _DEFAULT_RICH_STYLE)
    STYLE_INPUT_OPTION = _rich_styles_objects.get('INPUT_OPTION', _DEFAULT_RICH_STYLE)
    STYLE_COMMAND_LABEL = _rich_styles_objects.get('COMMAND_LABEL', _DEFAULT_RICH_STYLE)
    STYLE_COMMAND_CONTENT = _rich_styles_objects.get('COMMAND_CONTENT', _DEFAULT_RICH_STYLE)
    STYLE_TOOL_OUTPUT_DIM = _rich_styles_objects.get('TOOL_OUTPUT_DIM', _DEFAULT_RICH_STYLE)
    STYLE_CODE = _rich_styles_objects.get('CODE', _DEFAULT_RICH_STYLE)

    # Export PTK Style Dictionary Object
    PTK_STYLE = PTKStyle.from_dict(_ptk_styles_dict)

except KeyError as e:
    logger.critical(f"Missing expected key when defining style constants: {e}. Styling is broken.")
    # Set defaults to prevent import errors, but log the critical failure
    COLOR_AI = COLOR_USER = COLOR_INFO = COLOR_WARNING = COLOR_ERROR = COLOR_SYSTEM = COLOR_TOOL = COLOR_COMMAND = COLOR_DIM_TEXT = COLOR_NEUTRAL = COLOR_CODE = "#FFFFFF"
    STYLE_AI_LABEL = STYLE_AI_CONTENT = STYLE_USER_LABEL = STYLE_INFO_LABEL = STYLE_INFO_CONTENT = STYLE_WARNING_LABEL = STYLE_WARNING_CONTENT = STYLE_ERROR_LABEL = STYLE_ERROR_CONTENT = STYLE_SYSTEM_LABEL = STYLE_SYSTEM_CONTENT = STYLE_TOOL_NAME = STYLE_ARG_NAME = STYLE_ARG_VALUE = STYLE_THINKING = STYLE_INPUT_OPTION = STYLE_COMMAND_LABEL = STYLE_COMMAND_CONTENT = STYLE_TOOL_OUTPUT_DIM = STYLE_CODE = _DEFAULT_RICH_STYLE
    PTK_STYLE = PTKStyle.from_dict({})
    raise # Re-raise to halt execution


if __name__ == '__main__':
    # Example usage for testing the loader
    print(f"--- Style Loader Test (Simplified) ---")
    print(f"Loaded Styles Data Keys: {_styles_data.keys() if _styles_data else 'None'}")
    print(f"Colors Processed: {len(_colors)}")
    print(f"Rich Styles Processed: {len(_rich_styles_objects)}")
    print(f"PTK Styles Processed: {len(_ptk_styles_dict)}")
    print(f"\nExample COLOR_AI: {COLOR_AI}")
    print(f"Example STYLE_AI_LABEL: {STYLE_AI_LABEL}")
    print(f"Example PTK_STYLE type: {type(PTK_STYLE)}")