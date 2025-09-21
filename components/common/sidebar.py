"""
Enhanced Sidebar Component.

Implements the expandable sidebar design pattern with organized navigation sections.
"""

import streamlit as st
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any, Callable

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation


class EnhancedSidebar:
    """Enhanced sidebar component with expandable sections and organized navigation."""
    
    def __init__(self, title: str = "🤖 DurgasAI"):
        """
        Initialize the enhanced sidebar component.
        
        Args:
            title: Title to display at the top of the sidebar
        """
        self.title = title
        debug("EnhancedSidebar initialized", "sidebar")
    
    def render(self, navigation_config: Dict[str, Any]) -> str:
        """
        Render the enhanced sidebar with expandable sections.
        
        This method creates the complete sidebar interface including:
        - Navigation sections with expandable groups
        - Individual navigation items with icons
        - Usage tips and help information
        - Session state management for selected page
        
        Args:
            navigation_config: Configuration dictionary containing:
                - sections: List of navigation sections with items
                - default_page: Default page to select if none chosen
                
        Returns:
            str: The ID of the currently selected page
            
        Note:
            The sidebar uses Streamlit's session state to persist the
            selected page across app reruns.
        """
        debug("Starting enhanced sidebar rendering", "sidebar")
        
        with st.sidebar:
            debug("Sidebar context established", "sidebar")
            
            # Initialize session state for navigation if not exists
            if 'selected_page' not in st.session_state:
                default_page = navigation_config.get('default_page', 'ai_agent')
                st.session_state.selected_page = default_page
                debug(f"Initialized selected_page to default: {default_page}", "sidebar")
            else:
                debug(f"Using existing selected_page: {st.session_state.selected_page}", "sidebar")
            
            # Render navigation sections
            sections = navigation_config.get('sections', [])
            debug(f"Rendering {len(sections)} navigation sections", "sidebar")
            
            for i, section in enumerate(sections):
                debug(f"Rendering section {i+1}/{len(sections)}: {section.get('title', 'Unknown')}", "sidebar")
                self._render_section(section)
            
            debug("All navigation sections rendered", "sidebar")
            
            # Add usage tips at the bottom
            debug("Rendering usage tips section", "sidebar")
            self._render_usage_tips()
            debug("Usage tips section rendered", "sidebar")
            
            selected_page = st.session_state.selected_page
            debug(f"Sidebar rendering complete, selected page: {selected_page}", "sidebar")
            
            return selected_page
    
    def _render_section(self, section: Dict[str, Any]) -> None:
        """
        Render a single navigation section.
        
        Args:
            section: Section configuration with title, items, and options
        """
        section_title = section.get('title', 'Section')
        items = section.get('items', [])
        expanded = section.get('expanded', True)
        
        with st.expander(section_title, expanded=expanded):
            for item in items:
                self._render_navigation_item(item)
    
    def _render_navigation_item(self, item: Dict[str, Any]) -> None:
        """
        Render a single navigation item with comprehensive logging and error handling.
        
        This method creates individual navigation buttons with proper styling,
        state management, and user interaction tracking.
        
        Args:
            item: Item configuration with name, page_id, and options
            
        Features:
        - Navigation button creation with proper styling
        - Current page highlighting
        - User interaction logging
        - Error handling for malformed items
        - State management and page switching
        
        Logging:
        - Navigation item rendering details
        - Button click events
        - Page switching operations
        - Error handling for invalid items
        """
        debug("Starting navigation item rendering", "sidebar")
        
        try:
            # Extract item configuration with validation
            item_name = item.get('name', 'Unnamed')
            page_id = item.get('page_id', 'unknown')
            is_default = item.get('default', False)
            icon = item.get('icon', '')
            
            debug("Navigation item configuration extracted", "sidebar",
                  item_name=item_name,
                  page_id=page_id,
                  is_default=is_default,
                  icon=icon)
            
            # Validate item configuration
            if not item_name or not page_id:
                warning("Invalid navigation item configuration", "sidebar",
                       item_name=item_name,
                       page_id=page_id)
                return
            
            # Format display name with icon
            display_name = f"{icon} {item_name}" if icon else item_name
            debug("Display name formatted", "sidebar",
                  display_name=display_name,
                  has_icon=bool(icon))
            
            # Determine button styling based on current page
            current_page = st.session_state.selected_page
            is_current_page = current_page == page_id
            is_default_page = is_default and current_page == 'ai_agent'
            is_active = is_current_page or is_default_page
            
            color = "primary" if is_active else "secondary"
            
            debug("Button styling determined", "sidebar",
                  current_page=current_page,
                  is_current_page=is_current_page,
                  is_default_page=is_default_page,
                  is_active=is_active,
                  button_color=color)
            
            # Create navigation button with error handling
            button_key = f"nav_{page_id}"
            try:
                if st.button(display_name, key=button_key, use_container_width=True, type=color):
                    debug(f"Navigation button clicked: {item_name}", "sidebar",
                          page_id=page_id,
                          button_key=button_key,
                          from_page=current_page)
                    
                    # Update selected page
                    previous_page = current_page
                    st.session_state.selected_page = page_id
                    
                    debug("Page selection updated", "sidebar",
                          previous_page=previous_page,
                          new_page=page_id)
                    
                    # Log navigation action for analytics
                    log_user_action("page_navigation_clicked",
                                  page_id=page_id,
                                  page_name=item_name,
                                  previous_page=previous_page,
                                  is_default=is_default,
                                  icon=icon)
                    
                    # Trigger page rerun
                    st.rerun()
                    debug("Page rerun triggered", "sidebar")
                
                debug("Navigation button rendered successfully", "sidebar",
                      page_id=page_id,
                      button_key=button_key)
                
            except Exception as e:
                error("Error rendering navigation button", "sidebar", e,
                      page_id=page_id,
                      item_name=item_name,
                      button_key=button_key)
                raise
            
        except Exception as e:
            error("Critical error in navigation item rendering", "sidebar", e,
                  item_data=item)
            # Continue with other items even if one fails
            return
        
        debug("Navigation item rendering completed", "sidebar",
              page_id=page_id,
              item_name=item_name)
        
    
    def _render_usage_tips(self) -> None:
        """Render usage tips section at the bottom of the sidebar."""
        with st.expander("💡 Quick Tips", expanded=False):
            st.markdown("""
            **Getting Started:**
            - Set up your HuggingFace API token in Settings
            - Start with AI Agent or HuggingFace Chat
            - Explore specialized processors and tools
            
            **Key Features:**
            - 🤖 **AI Agent**: Pre-configured chat models
            - 🤗 **HuggingFace**: Direct API access to 500k+ models
            - 🔤 **Text Processing**: Tokenizers, padding, truncation
            - 🖼️ **Vision**: Image/video processors, backbones
            - 🎵 **Audio**: Feature extractors for speech/audio
            - 🚀 **Pipelines**: ML workflows and app deployment
            - 🎛️ **Dashboard**: Unified monitoring and management
            
            **Advanced Tools:**
            - Custom model development and sharing
            - Component customization with LoRA
            - Web server inference deployment
            - Comprehensive analytics and debugging
            """)
    
    @staticmethod
    def create_navigation_config() -> Dict[str, Any]:
        """
        Create the default navigation configuration for DurgasAI.
        
        Returns:
            Navigation configuration dictionary
        """
        return {
            'default_page': 'ai_agent',
            'sections': [
                {
                    'title': '🏠 MAIN',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Home',
                            'page_id': 'home',
                            'icon': '🏠'
                        },
                        {
                            'name': 'AI Agent',
                            'page_id': 'ai_agent',
                            'icon': '🤖',
                            'default': True
                        },
                        {
                            'name': 'Chat',
                            'page_id': 'chat',
                            'icon': '💬'
                        },
                        {
                            'name': 'HuggingFace Chat',
                            'page_id': 'huggingface',
                            'icon': '🤗'
                        }
                    ]
                },
                {
                    'title': '🔤 TEXT PROCESSING',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Tokenizers',
                            'page_id': 'tokenizers',
                            'icon': '🔤'
                        },
                        {
                            'name': 'Tokenizer Summary',
                            'page_id': 'tokenizer_summary',
                            'icon': '📋'
                        },
                        {
                            'name': 'Padding & Truncation',
                            'page_id': 'padding_truncation',
                            'icon': '📏'
                        }
                    ]
                },
                {
                    'title': '🖼️ VISION & MEDIA',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Image Processors',
                            'page_id': 'image_processors',
                            'icon': '🖼️'
                        },
                        {
                            'name': 'Video Processors',
                            'page_id': 'video_processors',
                            'icon': '🎥'
                        },
                        {
                            'name': 'Backbones',
                            'page_id': 'backbones',
                            'icon': '🏗️'
                        },
                        {
                            'name': 'Processors (Multimodal)',
                            'page_id': 'processors',
                            'icon': '🔄'
                        }
                    ]
                },
                {
                    'title': '🎵 AUDIO & FEATURES',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Feature Extractors',
                            'page_id': 'feature_extractors',
                            'icon': '🎵'
                        }
                    ]
                },
                {
                    'title': '🚀 PIPELINES & APPS',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Pipelines',
                            'page_id': 'pipeline',
                            'icon': '🔄'
                        },
                        {
                            'name': 'ML Apps',
                            'page_id': 'ml_apps',
                            'icon': '🚀'
                        },
                        {
                            'name': 'Auto Classes',
                            'page_id': 'auto_classes',
                            'icon': '⚡'
                        }
                    ]
                },
                {
                    'title': '🧩 AI MODELS',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Model Catalog',
                            'page_id': 'model_catalog',
                            'icon': '📚'
                        },
                        {
                            'name': 'Facial AI Studio',
                            'page_id': 'facial_ai',
                            'icon': '🎭'
                        },
                        {
                            'name': 'Custom Models',
                            'page_id': 'custom_models',
                            'icon': '🔧'
                        },
                        {
                            'name': 'Component Customization',
                            'page_id': 'component_customization',
                            'icon': '🎯'
                        },
                        {
                            'name': 'Web Server Inference',
                            'page_id': 'web_server_inference',
                            'icon': '🌐'
                        }
                    ]
                },
                {
                    'title': '🎛️ MANAGEMENT & MONITORING',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Preprocessor Dashboard',
                            'page_id': 'preprocessor_dashboard',
                            'icon': '🎛️'
                        },
                        {
                            'name': 'Analytics',
                            'page_id': 'analytics',
                            'icon': '📊'
                        },
                        {
                            'name': 'Debug Dashboard',
                            'page_id': 'debug_dashboard',
                            'icon': '🔧'
                        },
                        {
                            'name': 'Debug',
                            'page_id': 'debug',
                            'icon': '🐛'
                        }
                    ]
                },
                {
                    'title': '⚙️ CONFIGURATION',
                    'expanded': True,
                    'items': [
                        {
                            'name': 'Settings',
                            'page_id': 'settings',
                            'icon': '⚙️'
                        },
                        {
                            'name': 'Help',
                            'page_id': 'help',
                            'icon': '❓'
                        }
                    ]
                }
            ]
        }


def create_durgasai_sidebar() -> EnhancedSidebar:
    """
    Create a configured DurgasAI sidebar instance.
    
    Returns:
        Configured EnhancedSidebar instance
    """
    return EnhancedSidebar(title="🤖 DurgasAI")