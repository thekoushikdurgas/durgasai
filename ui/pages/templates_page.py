"""
Templates Page Module
Handles the templates management functionality
"""

import streamlit as st
from ..components.template_manager import TemplateManagerUI


class TemplatesPage:
    """Templates page implementation"""
    
    def __init__(self, template_manager):
        self.template_manager = template_manager
    
    def render(self):
        """Render the templates management page"""
        st.title("📋 Templates Management")
        st.markdown("Manage AI assistant templates and personalities")

        template_manager_ui = TemplateManagerUI(self.template_manager)
        template_manager_ui.render()
