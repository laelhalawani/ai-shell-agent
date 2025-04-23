"""
Contains prompt fragments related to using the Cognitive tools.
This content is returned as a ToolMessage when the toolset is activated.
"""

COGNITIVE_TOOLSET_PROMPT = """\
You have activated the 'Cognitive' tools.
Use these tools for internal reasoning and planning before executing complex tasks.
- `analyse`: Use this tool to break down the user's request, identify information needs, consider potential challenges, and document your reasoning. This helps structure your thought process. The output is saved internally and not shown directly to the user.
- `plan`: Use this tool to outline the sequence of steps or commands you intend to execute, especially for multi-step tasks. This allows you to verify the plan before starting execution. The output is saved internally and not shown directly to the user.

Use these tools to improve the quality and reliability of your responses and actions.
"""