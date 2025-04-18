# How to Add Additional Tools

Our architecture now emphasizes a fully modular approach. Tool integration is localized entirely within each toolset's own module. Follow these revised steps:

## 1. Create a New Tool Directory

- Navigate to the `ai_shell_agent/toolsets` directory.
- Create a new subdirectory for your tool. For example, for a tool named "mytool", create a folder: `ai_shell_agent/toolsets/mytool`.

## 2. Add Required Files

Within your new tool directory, include the following files:

- **__init__.py**: Can be empty or include initialization code.
- **toolset.py**: The main file for your tool.
- **Optional: prompts.py**: If your tool requires custom prompts, add this file modeled after existing implementations.

## 3. Register and Configure Your Tool (Auto-Discovery and Local Registration)

Tool registration and configuration now happens locally within each toolset's own `toolset.py`. To integrate your tool:

- Within `toolset.py`, create instances of your tool classes (inheriting from BaseTool or the appropriate abstract class).
- Organize these instances into lists:
  
  ```python
  # Example tool instances
  tool_instance = MyTool()
  toolset_tools = [tool_instance]
  toolset_start_tool = []  # if you have a preferred start tool; leave empty otherwise
  ```

- At the bottom of the file, import `register_tools` from `ai_shell_agent.tool_registry` and register your tool instances by calling:

  ```python
  from ai_shell_agent.tool_registry import register_tools
  register_tools(toolset_start_tool + toolset_tools)
  ```

- **Note:** Also ensure your toolset supplies fallback default configuration values and necessary secrets.

This approach makes your tool automatically discovered without needing manual edits in central files.

## 4. Implement Toolset-Specific Configuration

Tool configuration is now handled within the toolset itself, eliminating the need to modify a central configuration file. In your `toolset.py`:

- **Define Default Configurations:** Set up a dictionary named `toolset_config_defaults` for default values.

  ```python
  toolset_config_defaults: Dict = {}
  ```

- **Specify Required Secrets:** Declare any required secrets in a dictionary called `toolset_required_secrets`, mapping environment variable names to their descriptions.

  ```python
  toolset_required_secrets: Dict[str, str] = {}
  ```

- **Implement Configuration Logic and Validation:** Create a function called `configure_toolset` with the signature:

  ```python
  def configure_toolset(global_config_path: Path, local_config_path: Path, dotenv_path: Path, current_chat_config: Optional[Dict]) -> Dict:
      # Prompt user for configuration values, utilize ensure_dotenv_key from ai_shell_agent.utils for secrets,
      # and save the final configuration to both local_config_path and global_config_path.
      # Validate the configuration parameters to ensure no required value is missing.
      return final_config
  ```

This function should be fully self-contained and manage any configuration for your tool.

## 5. Test Your Integration

After setting up your tool:

- Run the agent and observe the logging output to verify that your new tool is discovered and configured correctly.
- Troubleshoot any issues with tool discovery or configuration within your `toolset.py`.

## Additional Notes

- **Auto-Discovery:** The system automatically scans the `ai_shell_agent/toolsets` directory for subdirectories containing a `toolset.py` file and loads the tool instances defined therein.
- **Local Registration & Configuration:** By handling both registration and configuration within each toolset's own module, there is no need to modify central files like `ai_shell_agent/tool_registry.py` or `ai_shell_agent/config_manager.py` for tool-specific logic.
- **Reference Implementations:** For guidance, review the implementations in `ai_shell_agent/toolsets/aider/` and `ai_shell_agent/toolsets/terminal/`.
- **Future Enhancements:** Keep an eye on updates to helper functions in modules like `ai_shell_agent/llm.py` that could further simplify tool integrations.

By following these revised steps, you'll ensure that your tool integrates seamlessly into our updated modular framework.
