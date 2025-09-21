"""
Debug Page for DurgasAI Application.

Provides debugging information, system monitoring, and troubleshooting tools.

This module serves as the main debug interface for DurgasAI, providing:
- System information and resource monitoring
- Session state inspection and management
- Log file analysis and viewing
- Model status and configuration debugging
- Performance metrics and optimization insights
- Troubleshooting tools and utilities

The debug page is essential for developers and advanced users to
diagnose issues and optimize application performance.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

# Import logging utilities with fallback
try:
    from utils.logger import debug, info, warning, error, log_user_action
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    # Fallback logging functions
    def debug(msg, component="debug_page", **kwargs): pass
    def info(msg, component="debug_page", **kwargs): pass
    def warning(msg, component="debug_page", **kwargs): pass
    def error(msg, component="debug_page", **kwargs): pass
    def log_user_action(action, **kwargs): pass

# Import the existing debug dashboard
try:
    from page.debug_dashboard import render_debug_dashboard
    DEBUG_DASHBOARD_AVAILABLE = True
except ImportError:
    DEBUG_DASHBOARD_AVAILABLE = False
    def render_debug_dashboard():
        st.error("Debug dashboard not available - missing dependencies")


def render_debug_page() -> None:
    """
    Render the debug page using the existing debug dashboard with enhanced logging.
    
    This function provides access to comprehensive debugging tools including:
    - System information and monitoring
    - Session state inspection
    - Log analysis and viewing
    - Performance metrics
    - Troubleshooting utilities
    
    The debug page is crucial for maintaining and optimizing DurgasAI.
    """
    debug("Starting debug page render", "debug_page")
    
    try:
        # Log debug page access (important for security monitoring)
        log_user_action("debug_page_accessed", 
                       timestamp=datetime.now().isoformat(),
                       logging_available=LOGGING_AVAILABLE,
                       dashboard_available=DEBUG_DASHBOARD_AVAILABLE)
        
        # Render debug dashboard with error handling
        debug("Rendering debug dashboard component", "debug_page")
        try:
            render_debug_dashboard()
            debug("Debug dashboard rendered successfully", "debug_page")
            
            # Log successful debug page render
            info("Debug page rendered successfully", "debug_page",
                 render_timestamp=datetime.now().isoformat(),
                 dashboard_available=DEBUG_DASHBOARD_AVAILABLE)
                 
        except Exception as e:
            error("Failed to render debug dashboard", "debug_page", e)
            st.error("❌ Failed to load debug dashboard")
            
            # Show basic fallback debug info
            st.markdown("## 🔧 Debug Information")
            st.markdown("Debug dashboard temporarily unavailable.")
            
            # Show basic system info as fallback
            st.markdown("### Basic System Information")
            st.write(f"**Python Version:** {sys.version}")
            st.write(f"**Logging Available:** {LOGGING_AVAILABLE}")
            st.write(f"**Dashboard Available:** {DEBUG_DASHBOARD_AVAILABLE}")
            
    except Exception as e:
        # Handle any unexpected errors during debug page render
        error("Critical error rendering debug page", "debug_page", e)
        st.error("🚨 A critical error occurred in the debug page. This is concerning.")
        
        # Show minimal fallback content for debugging the debug page
        st.markdown("## 🔧 Debug Page Error")
        st.markdown(f"Error: {str(e)}")
        st.markdown("Please check the logs for more information.")
