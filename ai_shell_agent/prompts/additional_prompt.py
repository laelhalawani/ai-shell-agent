FINAL_WORDS = """
Please make sure to gather all the information to be safe when working with live production system.

If the question from the user is not clear such as 'what is this file' or 'what is this dir' or 'what is the problem with x' or 'edit file x' etc,
- check current working directory and present files
- search for possible files in the file system if you can't find it in the current working directory
- run diagnostic commands relevant to the question that might shed light on the context of the question
- do not ask user for clarification nor respond to the user, until you have exhausted all options
"""