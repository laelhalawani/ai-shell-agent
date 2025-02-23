import logging

# Create a logger for the ai_shell_agent package
logger = logging.getLogger("ai_shell_agent")
logger.setLevel(logging.INFO)

# Create a console handler and set the level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)
