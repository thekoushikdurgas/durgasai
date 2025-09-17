"""
DurgasAI Main UI Application
Integrates LangGraph chatbot with existing DurgasAI tools and templates
"""

import streamlit as st
import json
import os
from pathlib import Path

# Import page modules
from ui.pages import ChatPage, ToolsPage, TemplatesPage, MonitorPage, SettingsPage, GoogleAgentPage
from ui.pages.workflows_page import WorkflowsPage
# from ui.components.sidebar import Sidebar
from ui.components.template_manager import TemplateManager
from utils.config_manager import ConfigManager
from utils.tool_manager import DurgasAIToolManager
from utils.workflow_service import WorkflowService
from utils.logging_utils import setup_logging_from_config, log_startup, log_shutdown, log_info, log_error
from utils.model_service import get_model_service, get_chat_service, get_logger_service
from models import ChatSession, ChatHistory, LogEntry, LogLevel

class DurgasAIApp:
    """Main DurgasAI application class"""
    
    def __init__(self):
        # Initialize configuration and logging first
        self.config_manager = ConfigManager()
        
        # Setup logging from configuration
        setup_logging_from_config(self.config_manager)
        log_startup("DurgasAI Application")
        
        try:
            self.tool_manager = DurgasAIToolManager()
            self.workflow_service = WorkflowService(tool_manager=self.tool_manager)
            self.initialize_session_state()
            self.template_manager = TemplateManager()
            
            # Initialize page instances
            self.chat_page = ChatPage(self.config_manager, self.tool_manager, self.template_manager)
            self.tools_page = ToolsPage(self.tool_manager)
            self.workflows_page = WorkflowsPage(self.workflow_service)
            self.templates_page = TemplatesPage(self.template_manager)
            self.monitor_page = MonitorPage()
            self.settings_page = SettingsPage(self.config_manager)
            self.google_agent_page = GoogleAgentPage(self.config_manager)
            
            log_info("DurgasAI application initialized successfully", "main")
            
        except Exception as e:
            log_error("Failed to initialize DurgasAI application", "main", e)
            raise
    
    def initialize_session_state(self):
        """Initialize Streamlit session state with enhanced model support"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "chat_session" not in st.session_state:
            # Create a new chat session using the enhanced model
            chat_service = get_chat_service()
            st.session_state.chat_session = chat_service.create_session(
                model="gpt-4o",
                user_id="default_user"
            )
        
        if "selected_template" not in st.session_state:
            st.session_state.selected_template = "default_assistant"
        
        if "selected_tools" not in st.session_state:
            st.session_state.selected_tools = []
        
        if "current_page" not in st.session_state:
            config = self.config_manager.get_config()
            st.session_state.current_page = config.get("ui", {}).get("default_page", "Chat")
        
        if "api_keys" not in st.session_state:
            self.sync_api_keys_from_config()
        
        # Initialize model services
        if "model_service" not in st.session_state:
            st.session_state.model_service = get_model_service()
        
        if "logger_service" not in st.session_state:
            st.session_state.logger_service = get_logger_service()
    
    def sync_api_keys_from_config(self):
        """Sync API keys from config to session state"""
        # Get decrypted API keys from config
        api_keys_config = self.config_manager.get_all_api_keys()
        st.session_state.api_keys = {
            "openai": api_keys_config.get("openai", ""),
            "anthropic": api_keys_config.get("anthropic", ""),
            "google": api_keys_config.get("google", ""),
            "deepseek": api_keys_config.get("deepseek", ""),
            "groq": api_keys_config.get("groq", ""),
            "mistral": api_keys_config.get("mistral", ""),
            "perplexity": api_keys_config.get("perplexity", ""),
            "huggingface": api_keys_config.get("huggingface", ""),
            "tavily": api_keys_config.get("tavily", ""),
            "web_search": api_keys_config.get("web_search", ""),
            "exa": api_keys_config.get("exa", ""),
            "serpapi": api_keys_config.get("serpapi", ""),
            "github": api_keys_config.get("github", ""),
            "openrouter": api_keys_config.get("openrouter", ""),
            "cohere": api_keys_config.get("cohere", ""),
            "nvidia": api_keys_config.get("nvidia", ""),
        }
        # log_info(f' "openai": {api_keys_config.get("openai", "")}, "anthropic": {api_keys_config.get("anthropic", "")}, "google": {api_keys_config.get("google", "")}, "deepseek": {api_keys_config.get("deepseek", "")}, "groq": {api_keys_config.get("groq", "")}, "mistral": {api_keys_config.get("mistral", "")}, "perplexity": {api_keys_config.get("perplexity", "")}, "huggingface": {api_keys_config.get("huggingface", "")}, "tavily": {api_keys_config.get("tavily", "")}, "web_search": {api_keys_config.get("web_search", "")}, "exa": {api_keys_config.get("exa", "")}, "serpapi": {api_keys_config.get("serpapi", "")}')
    def sync_api_keys_to_config(self):
        """Sync API keys from session state to config"""
        if "api_keys" in st.session_state:
            api_keys_dict = {}
            for provider, key in st.session_state.api_keys.items():
                if key and key.strip():  # Only include non-empty keys
                    api_keys_dict[provider] = key.strip()
            
            self.config_manager.update_api_keys(api_keys_dict)
            log_info("API keys synced from session state to config", "main")
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        config = self.config_manager.get_config()
        st.set_page_config(
            page_title=config.get("ui", {}).get("page_title", "DurgasAI"),
            page_icon=config.get("ui", {}).get("page_icon", "🤖"),
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
                    
    def render_main_content(self):
        with st.sidebar:
            # Page navigation
            pages = ["Chat", "Tools", "Workflows", "Templates", "Google Agent", "System Monitor", "Settings"]
            selected_page = st.selectbox(
                "Select Page",
                pages,
                index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
            )
            st.session_state.current_page = selected_page
            
        """Render the main content based on selected page"""
        if st.session_state.current_page == "Chat":
            self.chat_page.render()
        elif st.session_state.current_page == "Tools":
            self.tools_page.render()
        elif st.session_state.current_page == "Workflows":
            self.workflows_page.render()
        elif st.session_state.current_page == "Templates":
            self.templates_page.render()
        elif st.session_state.current_page == "Google Agent":
            self.google_agent_page.render()
        elif st.session_state.current_page == "System Monitor":
            self.monitor_page.render()
        elif st.session_state.current_page == "Settings":
            self.settings_page.render()
    
    
    def run(self):
        """Run the main application"""
        self.setup_page_config()
        self.render_main_content()

def load_durgasai_app():
    """Load and run the DurgasAI application"""
    try:
        app = DurgasAIApp()
        log_info("Starting DurgasAI application", "main")
        app.run()
    except Exception as e:
        log_error("Error loading DurgasAI application", "main", e)
        st.error(f"Error loading DurgasAI application: {str(e)}")
        st.exception(e)
    finally:
        log_shutdown("DurgasAI Application")
