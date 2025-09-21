"""
Page modules for DurgasAI application.

This package contains individual page modules, each focused on a specific
functionality and kept under 200 lines for maintainability.
"""

# Import all page render functions for easy access
from .home_page import render_home_page
from .chat_page import render_chat_page
from .settings_page import render_settings_page
from .analytics_page import render_analytics_page
from .help_page import render_help_page
from .debug_page import render_debug_page

__all__ = [
    'render_home_page',
    'render_chat_page', 
    'render_settings_page',
    'render_analytics_page',
    'render_help_page',
    'render_debug_page'
]
