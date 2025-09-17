"""
DurgasAI Pages Package
Contains individual page modules for the Streamlit application
"""

from .chat_page import ChatPage
from .tools_page import ToolsPage
from .templates_page import TemplatesPage
from .monitor_page import MonitorPage
from .settings_page import SettingsPage
from .google_agent_page import GoogleAgentPage

__all__ = [
    "ChatPage",
    "ToolsPage", 
    "TemplatesPage",
    "MonitorPage",
    "SettingsPage",
    "GoogleAgentPage"
]
