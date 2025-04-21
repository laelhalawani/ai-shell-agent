Okay, let's rewrite the `updating_styles.md` guide to reflect the new JSON-based styling system.

```markdown
# Customizing Styles in AI Shell Agent

This guide explains how to update colors and styles for the AI Shell Agent interface. The application uses a centralized JSON file to define colors and map them to different UI elements for both standard console output (`rich`) and interactive prompts (`prompt_toolkit`).

## Style System Overview

Styles are defined in a single JSON file:

*   **File:** `ai_shell_agent/styles/default_style.json`

This file contains three main sections:

1.  **`colors`**: Defines named color variables (using hex codes). This is the recommended place to change base colors.
2.  **`rich_styles`**: Defines styles used by the `rich` library for displaying messages (AI responses, system info, errors, etc.). These styles often *refer* to the colors defined in the `colors` section.
3.  **`ptk_styles`**: Defines styles used by the `prompt_toolkit` library for interactive input prompts (user input line, HITL prompts, selection prompts). These styles also often *refer* to the colors defined in the `colors` section.

The application automatically processes this JSON file via `ai_shell_agent/styles.py` to create the necessary style objects for `rich` and `prompt_toolkit`.

## Modifying Styles: The Recommended Way (Changing Colors)

The easiest and most consistent way to change the appearance is to modify the hex color codes in the `colors` section of `default_style.json`.

**Example: Change AI message color from purple to green**

1.  **Open:** `ai_shell_agent/styles/default_style.json`
2.  **Locate:** The `colors` section.
3.  **Find:** The `"ai"` key. Its default value might be `"#B19CD9"` (a light purple).
4.  **Change:** The hex value to your desired green, for example `"#90EE90"` (light green).

    ```json
    {
      "colors": {
        "ai": "#90EE90", // Changed from #B19CD9
        "user": "#FFAA99",
        "info": "#8FD9A8",
        // ... other colors ...
      },
      // ... rest of the file ...
    }
    ```
5.  **Save** the file.
6.  **Restart** the `ai` agent.

Because the `rich_styles` (like `ai_label`, `ai_content`) and `ptk_styles` (like `style_ai_label`, `style_ai_content`) refer to `"colors.ai"`, changing this single color value will automatically update all relevant UI elements that use the "ai" color.

## Advanced Modifications (Changing Style Attributes)

You can also modify specific style attributes (like bold, underline, background color) or assign different colors directly within the `rich_styles` and `ptk_styles` sections.

### Modifying Rich Styles (`rich_styles`)

*   **Structure:** A dictionary where keys are style identifiers (e.g., `"error_label"`) and values are configuration objects.
*   **Configuration:**
    *   `"color"`: Can be a direct hex string (`"#FF0000"`) or a reference (`"colors.error"`).
    *   `"bgcolor"`: Background color (hex or reference).
    *   `"bold"`, `"italic"`, `"underline"`, `"strike"`, `"dim"`: Boolean values (`true` or `false`).

*   **Example: Make error labels bold and underlined, keeping the referenced error color:**

    ```json
    {
      // ... colors ...
      "rich_styles": {
        // ... other styles ...
        "error_label": { "color": "colors.error", "bold": true, "underline": true }, // Added underline
        "error_content": { "color": "colors.error" },
        // ... other styles ...
      },
      // ... ptk_styles ...
    }
    ```

### Modifying Prompt Toolkit Styles (`ptk_styles`)

*   **Structure:** A dictionary where keys are style class names (e.g., `"style_error_label"`, `"prompt.prefix"`) and values are style strings.
*   **Style String Format:** Uses `prompt_toolkit`'s format (e.g., `bold`, `italic`, `underline`, `fg:#RRGGBB`, `bg:colorname`, `fg:colors.error`). See `prompt_toolkit` documentation for details. Color references like `"colors.some_color"` are resolved automatically.

*   **Example: Add an underline to the error label in prompts:**

    ```json
    {
      // ... colors ...
      // ... rich_styles ...
      "ptk_styles": {
        // ... other styles ...
        "style_error_label": "bold underline fg:colors.error", // Added underline
        "style_error_content": "fg:colors.error",
        // ... other styles ...
      }
    }
    ```

**Consistency Note:** If you change attributes directly in `rich_styles` or `ptk_styles` without using color references, you might need to manually update both sections to keep the appearance perfectly consistent between standard output and interactive prompts. Using the `colors` section is generally preferred for color changes.

## Adding New Styles

If extending the application requires new styled elements:

1.  **(Optional) Define a new color** in the `"colors"` section if needed (e.g., `"special_alert": "#FF8C00"`).
2.  **Add a new entry** to `"rich_styles"` (e.g., `"special_alert_label": { "color": "colors.special_alert", "italic": true }`). The key (`"special_alert_label"`) will be used to generate the Python constant `STYLE_SPECIAL_ALERT_LABEL`.
3.  **Add a corresponding entry** to `"ptk_styles"` (e.g., `"style_special_alert_label": "italic fg:colors.special_alert"`). The key should generally follow the `style_` prefix convention for consistency, but can be anything `prompt_toolkit` accepts.
4.  After restarting the agent, you can import and use the new Rich style constant (e.g., `from .styles import STYLE_SPECIAL_ALERT_LABEL`) in your Python code. The PTK style will be automatically included in the main `PTK_STYLE` object used by `prompt_toolkit_prompt`.

## Style Reference

This table maps the style keys in `default_style.json` to their purpose in the UI:

| `colors` Key         | `rich_styles` Key(s)           | `ptk_styles` Key(s)                       | Purpose                                     |
| :------------------- | :----------------------------- | :---------------------------------------- | :------------------------------------------ |
| `ai`                 | `ai_label`, `ai_content`, `thinking` | `style_ai_label`, `style_ai_content`, `style_thinking` | AI messages, thinking indicator           |
| `user`               | `user_label`                   | `style_user_label`                        | User input labels (in history listing)      |
| `info`               | `info_label`, `info_content`   | `style_info_label`, `style_info_content`  | Informational messages                    |
| `warning`            | `warning_label`, `warning_content` | `style_warning_label`, `style_warning_content` | Warning messages                        |
| `error`              | `error_label`, `error_content` | `style_error_label`, `style_error_content` | Error messages                          |
| `system`             | `system_label`, `system_content` | `style_system_label`, `style_system_content` | System messages, UI instructions        |
| `tool`               | `tool_name`                    | `style_tool_name`, `prompt.toolname`      | Tool names in output/prompts            |
| `command`            | `command_label`, `command_content`, `code` | `style_command_label`, `style_command_content` | Direct command execution, Code snippets |
| `dim_text`           | `arg_name`, `arg_value`, `tool_output_dim` | `style_arg_name`, `style_arg_value`, `style_tool_output_dim`, `prompt.argname`, `prompt.argvalue` | Dimmed text (args, condensed output)    |
| `neutral`            | *(Used directly if needed)*    | *(Used directly if needed)*               | General neutral color (e.g., white)       |
| `prompt_prefix`      | *(N/A)*                        | `prompt.prefix`                           | Prefix text for user input prompts      |
| `prompt_default_hint`| *(N/A)*                        | `default`                                 | Default value hint in prompts           |
| `input_text`         | *(N/A)*                        | `""` (Empty key)                          | Default text color for user input       |
| `option_underline`   | `input_option`                 | `style_input_option`                      | Underlined options in selection lists   |
| `code`               | `code`                         | *(Covered by command styles)*             | Standalone code block style             |

*Note: Some `ptk_styles` keys like `prompt.prefix` or `""` are specific class names used by `prompt_toolkit` itself.*

## Testing Changes

After modifying `default_style.json`, **you must restart the `ai` agent** for the changes to take effect. The JSON file is read only once during startup.

Test various commands (`ai "hi"`, `ai --list-chats`, `ai --select-model`, triggering HITL prompts, etc.) to ensure your style changes appear correctly in all relevant parts of the UI.

## Best Practices

*   **Use the `colors` Section:** Prefer changing colors in the `colors` section for consistency across Rich and Prompt Toolkit elements.
*   **Maintain Contrast:** Ensure text remains readable against your terminal's background color.
*   **Keep Styles Consistent:** Use similar colors/styles for related UI elements (e.g., all error messages).
*   **Consider Accessibility:** Avoid color combinations that might be difficult to read for users with color vision deficiencies.
*   **Test in Different Terminals:** Styles can sometimes render slightly differently across various terminal applications (Windows Terminal, WSL, macOS Terminal, GNOME Terminal, etc.).

By editing the `default_style.json` file, you can easily customize the look and feel of the AI Shell Agent.
```