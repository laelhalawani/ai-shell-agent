# Customizing Styles in AI Shell Agent

This guide explains how to update colors and styles for different types of messages in the AI Shell Agent. Whether you want to change the AI responses to be green instead of blue, or modify any other visual aspects of the application, this document will walk you through the process.

## Style System Overview

AI Shell Agent uses two style systems that need to be kept in sync:

1. **Rich Styles** - Used for console output via the `rich` library
2. **Prompt Toolkit Styles** - Used for interactive prompts via the `prompt_toolkit` library

Both style systems are defined in `ai_shell_agent/console_manager.py`. When changing styles, you need to update both to ensure consistent appearance.

## Step 1: Locate the Style Definitions

Open `ai_shell_agent/console_manager.py` and find the style definitions near the top of the file. You'll see two main sections:

```python
# Rich styles
STYLE_AI_LABEL = RichStyle(color="blue", bold=True)
STYLE_AI_CONTENT = RichStyle(color="blue")
STYLE_USER_LABEL = RichStyle(color="purple", bold=True)
# ... more Rich styles ...

# Prompt Toolkit styles
PTK_STYLE = PTKStyle.from_dict({
    'style_ai_label':          'bold fg:ansiblue',
    'style_user_label':        'bold fg:ansimagenta',
    # ... more PT styles ...
})
```

## Step 2: Understand Style Formats

### Rich Styles

Rich styles use descriptive color names or RGB hex codes:

```python
RichStyle(color="green", bold=True)  # Named color
RichStyle(color="#00FF00", bold=True)  # Hex color
```

Available attributes:
- `color` - Text color (named color or hex)
- `bgcolor` - Background color
- `bold` - Bold text (True/False)
- `italic` - Italic text (True/False)
- `underline` - Underlined text (True/False)
- `dim` - Dimmed text (True/False)

### Prompt Toolkit Styles

Prompt Toolkit uses a different format:

```python
'bold fg:ansigreen'  # Bold green text
'bg:ansiblue'        # Blue background
```

Available formats:
- `fg:color` - Foreground color
- `bg:color` - Background color
- `bold` - Bold text
- `italic` - Italic text
- `underline` - Underlined text

For colors, use either:
- ANSI colors: `ansiblack`, `ansired`, `ansigreen`, `ansiyellow`, `ansiblue`, `ansimagenta`, `ansicyan`, `ansigray`
- Hex colors: `#RRGGBB`

## Step 3: Modify Styles

To change a style, update both the Rich and Prompt Toolkit definitions. For example, to change AI messages from blue to green:

```python
# Change Rich style from blue to green
STYLE_AI_LABEL = RichStyle(color="green", bold=True)
STYLE_AI_CONTENT = RichStyle(color="green")

# Change corresponding Prompt Toolkit style
PTK_STYLE = PTKStyle.from_dict({
    # ... other styles ...
    'style_ai_label':          'bold fg:ansigreen',
    'style_ai_content':        'fg:ansigreen',
    # ... other styles ...
})
```

## Common Customization Examples

### Example 1: Swap AI and INFO Colors

To make AI messages green and INFO messages blue:

```python
# Rich styles
STYLE_AI_LABEL = RichStyle(color="green", bold=True)
STYLE_AI_CONTENT = RichStyle(color="green")
STYLE_INFO_LABEL = RichStyle(color="blue", bold=True)
STYLE_INFO_CONTENT = RichStyle(color="blue")

# PT styles
PTK_STYLE = PTKStyle.from_dict({
    # ... other styles ...
    'style_ai_label':          'bold fg:ansigreen',
    'style_ai_content':        'fg:ansigreen',
    'style_info_label':        'bold fg:ansiblue',
    'style_info_content':      'fg:ansiblue',
    # ... other styles ...
})
```

### Example 2: Use Custom Hex Colors

To use specific hex colors:

```python
# Rich styles
STYLE_WARNING_LABEL = RichStyle(color="#FFA500", bold=True)  # Orange
STYLE_WARNING_CONTENT = RichStyle(color="#FFA500")

# PT styles
PTK_STYLE = PTKStyle.from_dict({
    # ... other styles ...
    'style_warning_label':     'bold fg:#FFA500',
    'style_warning_content':   'fg:#FFA500',
    # ... other styles ...
})
```

### Example 3: Add Background Colors

To add background colors to labels:

```python
# Rich styles
STYLE_ERROR_LABEL = RichStyle(color="white", bgcolor="red", bold=True)

# PT styles
PTK_STYLE = PTKStyle.from_dict({
    # ... other styles ...
    'style_error_label':       'bold fg:ansiwhite bg:ansired',
    # ... other styles ...
})
```

## Using a Single Color Definition Source

While it's not possible to completely unify the two styling systems due to their different formats, you could create a single source of truth for color values by using hex colors exclusively and defining them as constants with semantic names:

```python
# Define colors as semantic constants
AI_LABEL_COLOR = "#0000FF"       # Blue for AI labels
AI_CONTENT_COLOR = "#0000FF"     # Blue for AI content
INFO_LABEL_COLOR = "#00FF00"     # Green for info labels
INFO_CONTENT_COLOR = "#00FF00"   # Green for info content
WARNING_COLOR = "#FFFF00"        # Yellow for warnings
ERROR_COLOR = "#FF0000"          # Red for errors
SYSTEM_COLOR = "#00FFFF"         # Cyan for system messages
USER_LABEL_COLOR = "#FF00FF"     # Magenta for user labels
COMMAND_COLOR = "#FF00FF"        # Magenta for command content
DIM_TEXT_COLOR = "#888888"       # Dim gray for less important text

# Rich styles
STYLE_AI_LABEL = RichStyle(color=AI_LABEL_COLOR, bold=True)
STYLE_AI_CONTENT = RichStyle(color=AI_CONTENT_COLOR)
STYLE_ERROR_LABEL = RichStyle(color=ERROR_COLOR, bold=True)
STYLE_ERROR_CONTENT = RichStyle(color=ERROR_COLOR)
# ...etc

# Prompt Toolkit styles
PTK_STYLE = PTKStyle.from_dict({
    'style_ai_label':          f'bold fg:{AI_LABEL_COLOR}',
    'style_ai_content':        f'fg:{AI_CONTENT_COLOR}',
    'style_error_label':       f'bold fg:{ERROR_COLOR}',
    'style_error_content':     f'fg:{ERROR_COLOR}',
    # ...etc
})
```

This approach provides several advantages:

1. **Single Source of Truth**: Color values are defined once and used consistently
2. **Semantic Naming**: Color constants are named by their purpose, not their value
3. **Easy Updates**: Changing a color requires modifying only one line
4. **Maintainability**: New developers can understand the purpose of each color at a glance

The trade-off is that you lose access to the built-in named color systems of both libraries, but the improved maintainability is often worth it, especially if you're using custom brand colors.

## Style Categories

Here are all the style categories you can modify:

| Category | Purpose |
|----------|---------|
| `AI_LABEL` / `AI_CONTENT` | AI messages and responses |
| `USER_LABEL` | User input labels |
| `INFO_LABEL` / `INFO_CONTENT` | Informational messages |
| `WARNING_LABEL` / `WARNING_CONTENT` | Warning messages |
| `ERROR_LABEL` / `ERROR_CONTENT` | Error messages |
| `SYSTEM_LABEL` / `SYSTEM_CONTENT` | System messages |
| `TOOL_NAME` | Tool names in output |
| `ARG_NAME` / `ARG_VALUE` | Tool argument names and values |
| `THINKING` | "AI: Thinking..." indicator |
| `INPUT_OPTION` | Input options |
| `COMMAND_LABEL` / `COMMAND_CONTENT` | Command input/output |
| `TOOL_OUTPUT_DIM` | Dimmed tool output |

## Testing Changes

After making style changes, run the AI Shell Agent to verify that your changes look as expected across all types of interactions. Pay attention to both standard output and interactive prompts, as they use different styling systems.

## Best Practices

1. **Maintain Contrast**: Ensure text remains readable on your terminal background
2. **Keep Styles Consistent**: Use similar colors for related functionality
3. **Consider Accessibility**: Some color combinations may be difficult to read for users with color vision deficiencies
4. **Keep Both Style Systems in Sync**: Always update both Rich and Prompt Toolkit styles
5. **Test in Different Terminals**: Styles may render differently across terminal applications

By following this guide, you can customize the visual appearance of AI Shell Agent to match your preferences or branding requirements.