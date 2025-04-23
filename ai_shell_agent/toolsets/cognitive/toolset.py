# ai_shell_agent/toolsets/cognitive/toolset.py
"""
Defines the tools and metadata for the Cognitive toolset.
Allows the agent to perform internal analysis and planning steps.
"""
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

# Langchain imports
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import run_in_executor
from pydantic import Field, BaseModel

# Local imports
from ... import logger
from ...tool_registry import register_tools
from ...console_manager import get_console_manager # For configure_toolset info message
from ...utils.file_io import write_json, read_json # For configure_toolset
from ...texts import get_text as get_main_text # For configure_toolset info message
from .texts import get_text # Toolset specific texts

# --- Toolset Metadata ---
toolset_id = "cognitive"
toolset_name = get_text("toolset.name")
toolset_description = get_text("toolset.description")
toolset_required_secrets: Dict[str, str] = {}

# Get console manager instance
console = get_console_manager()

# --- Configuration Function (No actual settings) ---
def configure_toolset(
    global_config_path: Path,
    local_config_path: Optional[Path],
    dotenv_path: Path,
    current_chat_config: Optional[Dict]
) -> Dict:
    """
    Configuration function for the Cognitive toolset.
    Currently, this toolset has no configurable options.
    """
    final_config = {} # No settings to configure
    is_global_only = local_config_path is None
    context_name = "Global Defaults" if is_global_only else "Current Chat"

    # Display context header
    console.display_message(
        get_main_text("common.labels.system"),
        get_text("config.header").format(context_name=context_name),
        console.STYLE_SYSTEM_LABEL,
        console.STYLE_SYSTEM_CONTENT
    )

    logger.info(f"Configuring Cognitive toolset ({context_name}). No user-configurable settings currently.")
    # Display info message using toolset's text
    console.display_message(
        get_main_text("common.labels.info"),
        get_text("config.info_no_settings"),
        console.STYLE_INFO_LABEL,
        console.STYLE_INFO_CONTENT
    )

    save_success_global = True
    save_success_local = True

    # Attempt to write empty config files to mark as configured
    try:
        global_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(global_config_path, final_config)
        logger.debug(f"Wrote empty config for Cognitive toolset to global path: {global_config_path}")
    except Exception as e:
         save_success_global = False
         logger.error(f"Failed to write empty config for Cognitive toolset to global path {global_config_path}: {e}")

    if not is_global_only and local_config_path:
        try:
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(local_config_path, final_config)
            logger.debug(f"Wrote empty config for Cognitive toolset to local path: {local_config_path}")
        except Exception as e:
             save_success_local = False
             logger.error(f"Failed to write empty config for Cognitive toolset to local path {local_config_path}: {e}")

    if not save_success_global or not save_success_local:
        console.display_message(
            get_main_text("common.labels.warning"),
            get_text("config.warn_write_failed"), # Use toolset text key
            console.STYLE_WARNING_LABEL,
            console.STYLE_WARNING_CONTENT
        )

    return final_config

# --- Tool Input Schemas ---
class AnalyseArgsSchema(BaseModel):
    analysis: str = Field(..., description=get_text("schemas.analyse.analysis_desc"))

class PlanArgsSchema(BaseModel):
    plan: str = Field(..., description=get_text("schemas.plan.plan_desc"))

# --- Tool Classes ---
class AnalyseTool(BaseTool):
    """Tool for the agent to internally analyse the task at hand."""
    name: str = get_text("tools.analyse.name")
    description: str = get_text("tools.analyse.description")
    args_schema: Type[BaseModel] = AnalyseArgsSchema
    requires_confirmation: bool = False # Internal tool, no confirmation needed

    def _run(self, analysis: str) -> str:
        """Saves the analysis as an internal message."""
        logger.debug(f"AnalyseTool executed. Analysis length: {len(analysis)}")
        # The return value will be automatically converted to a ToolMessage
        # by the chat_manager because it's not a PromptNeededError.
        return get_text("tools.analyse.success", analysis=analysis)

    async def _arun(self, analysis: str) -> str:
        return await run_in_executor(None, self._run, analysis)

class PlanTool(BaseTool):
    """Tool for the agent to internally lay out a plan for complex tasks."""
    name: str = get_text("tools.plan.name")
    description: str = get_text("tools.plan.description")
    args_schema: Type[BaseModel] = PlanArgsSchema
    requires_confirmation: bool = False # Internal tool, no confirmation needed

    def _run(self, plan: str) -> str:
        """Saves the plan as an internal message."""
        logger.debug(f"PlanTool executed. Plan length: {len(plan)}")
        # The return value will be automatically converted to a ToolMessage
        return get_text("tools.plan.success", plan=plan)

    async def _arun(self, plan: str) -> str:
        return await run_in_executor(None, self._run, plan)

# --- Tool Instances ---
analyse_tool = AnalyseTool()
plan_tool = PlanTool()

# --- Toolset Definition ---
toolset_tools: List[BaseTool] = [
    analyse_tool,
    plan_tool,
]

# --- Register Tools ---
register_tools(toolset_tools)
logger.debug(f"Cognitive toolset tools registered: {[t.name for t in toolset_tools]}")