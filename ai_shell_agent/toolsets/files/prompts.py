"""
Prompt content for the File System toolset.
"""

FILES_TOOLSET_PROMPT = """\
You have activated the 'File Manager' tools.
They allow you to directly interact with the user's file system to create, read, delete, copy, move, rename, find, and edit files and directories.
Use these tools carefully, especially delete_file_or_dir, overwrite_file, find_and_replace_in_file, copy_file_or_dir, move_file_or_dir, rename_file_or_dir, and restore_file which require user confirmation before execution.
Always check path existence before attempting operations like read, delete, edit, copy, move, rename.
When editing files (overwrite_file, find_and_replace_in_file), backups are automatically created with a `.bak.<timestamp>` suffix. You can restore these using restore_file.
"""