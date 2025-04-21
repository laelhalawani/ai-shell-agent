"""
Prompt content for the File System toolset.
"""

FILES_TOOLSET_PROMPT = """\
You have activated the 'File System' manager.
It allows you to directly interact with the user's file system, to perform operations such as creating, deleting, moving, copying, and editing files and directories.
Use these tools carefully, especially delete, edit, copy, move, rename, restore, and finalize which require user confirmation.
Always check path existence before attempting operations like read, delete, edit, copy, move, rename.
When editing, backups are automatically created with a `.bak.<timestamp>` suffix.
"""