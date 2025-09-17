"""
Chat Page Module
Handles the main chat interface functionality
"""

import streamlit as st
from ui.components.sidebar import Sidebar
from ..components.chatbot_interface import ChatbotInterface


class ChatPage:
    """Chat page implementation"""
    
    def __init__(self, config_manager, tool_manager, template_manager):
        self.config_manager = config_manager
        self.tool_manager = tool_manager
        self.template_manager = template_manager
        self.sidebar = Sidebar(self.template_manager, self.config_manager)
    
    def render(self):
        """Render the main chat interface"""
        # Initialize chatbot interface
        self.render_sidebar()
            
        chatbot = ChatbotInterface(
            config_manager=self.config_manager,
            tool_manager=self.tool_manager,
            template_manager=self.template_manager
        )
        
        # Render chat interface
        chatbot.render()
    
    def render_sidebar(self):
        """Render the sidebar with navigation and configuration"""
        self.sidebar.render()
