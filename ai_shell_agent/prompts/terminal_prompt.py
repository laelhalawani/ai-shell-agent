# ai_shell_agent/prompts/terminal_prompt.py
"""
Contains prompt fragments related to using the Terminal toolset.
Includes OS-specific instructions.
"""
import platform

OS_SYSTEM = platform.system() # Use platform.system() for broader compatibility

TERMINAL_TOOLSET_INTRO = """\

You have access to a 'Terminal' toolset which allows you to execute commands directly in the user's system shell.
Use the `terminal` tool to run commands.
Use this tool for operations like file management, system information retrieval, software installation, and network diagnostics, among others.
The terminal supports various commands as listed below.
"""

WINDOWS_CMD_GUIDANCE = """\
You are interacting with a Windows system using CMD.
Key Commands:
- List directory: `dir`
- Change directory: `cd <directory>`
- Show current directory: `cd`
- System info: `systeminfo`
- Running processes: `tasklist`
- Network info: `ipconfig /all`
- Environment variables: `set` or `echo %VARNAME%`
- Set environment variable: `set VAR=value`
- Run multiple commands: `command1 && command2`

Always use CMD syntax. Be mindful of path separators (`\\`).
When asked for technical support or to perform system tasks, run the necessary commands directly.
Only provide explanations after executing the commands, using their output.
Do not ask for confirmation before running commands unless the user explicitly requests it or the action is potentially destructive.
"""

LINUX_BASH_GUIDANCE = """\
You are interacting with a Linux system using a Bash-like shell.
Key Commands:
- List directory: `ls -la`
- Change directory: `cd /path/to/directory`
- Show current directory: `pwd`
- System info: `uname -a` or `cat /etc/os-release`
- Running processes: `ps aux` or `top -bn1`
- Network info: `ip a` or `ifconfig`
- Environment variables: `env` or `echo $VARNAME`
- Set environment variable: `export VAR=value` (for current session)
- Run multiple commands: `command1 && command2`

Always use Bash syntax. Be mindful of path separators (`/`).
When asked for technical support or to perform system tasks, run the necessary commands directly.
Only provide explanations after executing the commands, using their output.
Do not ask for confirmation before running commands unless the user explicitly requests it or the action is potentially destructive.
"""

MACOS_ZSH_GUIDANCE = """\
You are interacting with a macOS system using a Zsh/Bash-like shell.
Key Commands:
- List directory: `ls -la`
- Change directory: `cd /path/to/directory`
- Show current directory: `pwd`
- System info: `uname -a` or `sw_vers`
- Running processes: `ps aux` or `top -l 1`
- Network info: `ifconfig`
- Environment variables: `env` or `echo $VARNAME`
- Set environment variable: `export VAR=value` (for current session)
- Run multiple commands: `command1 && command2`

Always use Zsh/Bash syntax. Be mindful of path separators (`/`).
When asked for technical support or to perform system tasks, run the necessary commands directly.
Only provide explanations after executing the commands, using their output.
Do not ask for confirmation before running commands unless the user explicitly requests it or the action is potentially destructive.
"""

UNKNOWN_SYSTEM_GUIDANCE = """\
The operating system could not be automatically determined.
You can try running commands like `uname -a`, `ver`, `sw_vers`, or `cat /etc/os-release` to identify the system (Windows, Linux, macOS).
Once identified, use the appropriate command syntax for that system.
Remember to use the `terminal` tool for execution.
"""

def get_terminal_guidance() -> str:
    """Returns OS-specific terminal guidance."""
    if OS_SYSTEM == "Windows":
        return WINDOWS_CMD_GUIDANCE
    elif OS_SYSTEM == "Linux":
        return LINUX_BASH_GUIDANCE
    elif OS_SYSTEM == "Darwin": # Darwin is the system name for macOS
        return MACOS_ZSH_GUIDANCE
    else:
        return UNKNOWN_SYSTEM_GUIDANCE

TERMINAL_PROMPT_SNIPPET = TERMINAL_TOOLSET_INTRO + get_terminal_guidance()