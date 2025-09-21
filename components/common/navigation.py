"""
Navigation Component.

Handles application navigation and common UI elements.
"""

import streamlit as st
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import debug


class Navigation:
    """Component for application navigation."""
    
    def __init__(self):
        """Initialize navigation component."""
        debug("Navigation initialized", "navigation")
    
    def render_sidebar_navigation(self, navigation_items: dict) -> str:
        """
        Render sidebar navigation using the new expandable design.
        
        Args:
            navigation_items: Dict of display names to page IDs
            
        Returns:
            Selected page ID
        """
        with st.sidebar:
            # Main title
            st.title("🤖 DurgasAI")
            st.markdown("---")
            
            # Initialize session state for navigation if not exists
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = 'ai_agent'  # Default page
            
            # Define navigation sections
            navigation_sections = self._organize_navigation_sections(navigation_items)
            
            # Render each section
            for section_title, section_items in navigation_sections.items():
                with st.expander(section_title, expanded=True):
                    for item_name, page_id in section_items:
                        # Create a button for each navigation item
                        if st.button(item_name, key=f"nav_{page_id}", use_container_width=True):
                            st.session_state.selected_page = page_id
                            st.rerun()
                        
                        # Highlight the current page
                        if st.session_state.selected_page == page_id:
                            st.markdown("**← Current Page**")
            
            return st.session_state.selected_page
    
    def _organize_navigation_sections(self, navigation_items: dict) -> Dict[str, List[tuple]]:
        """
        Organize navigation items into logical sections.
        
        Args:
            navigation_items: Dict of display names to page IDs
            
        Returns:
            Dict of section titles to lists of (item_name, page_id) tuples
        """
        sections = {
            "✨ CORE FEATURES": [],
            "🧩 AI MODELS": [],
            "🔧 TOOLS & UTILITIES": []
        }
        
        # Categorize navigation items
        for item_name, page_id in navigation_items.items():
            if any(keyword in item_name.lower() for keyword in ['home', 'ai agent', 'huggingface']):
                sections["✨ CORE FEATURES"].append((item_name, page_id))
            elif any(keyword in item_name.lower() for keyword in ['model', 'catalog', 'facial']):
                sections["🧩 AI MODELS"].append((item_name, page_id))
            else:
                sections["🔧 TOOLS & UTILITIES"].append((item_name, page_id))
        
        # Remove empty sections
        return {k: v for k, v in sections.items() if v}


def display_usage_tips() -> None:
    """Display usage tips and help information."""
    with st.expander("💡 Usage Tips & Help", expanded=False):
        st.markdown("""
        ### Getting Started
        1. **API Token**: Get your free HuggingFace API token from [here](https://huggingface.co/settings/tokens)
        2. **Model Selection**: Choose from various AI models based on your needs
        3. **System Prompt**: Customize the AI's behavior and personality
        4. **Start Chatting**: Type your message and press Enter!
        
        ### Model Recommendations
        - **🎯 Mistral 7B**: Best for instruction-following and reasoning
        - **💬 Zephyr 7B**: Excellent for conversations and general queries  
        - **📚 Flan T5**: Great for educational and factual questions
        - **🗣️ DialoGPT**: Optimized for dialogue and chat interactions
        - **🌟 BlenderBot**: Good for open-domain conversations
        
        ### Tips for Better Results
        - Be specific and clear in your questions
        - Use system prompts to guide the AI's responses
        - Adjust temperature for creativity vs consistency
        - Try different models for different types of tasks
        
        ### Troubleshooting
        - **API Errors**: Check your API token and internet connection
        - **Slow Responses**: Try a smaller model or reduce max tokens
        - **Memory Issues**: Clear chat history or restart the app
        - **Model Loading**: Wait for the model to fully initialize
        """)
