"""
Home Page for DurgasAI Application.

Displays welcome information, feature overview, and getting started guide.

This module serves as the main landing page for DurgasAI, providing:
- Welcome message and application overview
- Feature highlights and capabilities
- Available model showcase
- Getting started guide for new users
- Navigation to other application features

The page is designed to be informative and welcoming while guiding
users toward the main AI Agent functionality.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from components.common.metrics_display import display_model_overview
from components.common.navigation import display_usage_tips

# Import logging utilities with fallback
try:
    from utils.logger import debug, info, warning, error, log_user_action
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    # Fallback logging functions
    def debug(msg, component="home_page", **kwargs): pass
    def info(msg, component="home_page", **kwargs): pass
    def warning(msg, component="home_page", **kwargs): pass
    def error(msg, component="home_page", **kwargs): pass
    def log_user_action(action, **kwargs): pass


def render_home_page() -> None:
    """
    Render the home page with welcome information and feature overview.
    
    This function creates the main landing page for DurgasAI, including:
    - Welcome header with branding
    - Feature overview and capabilities
    - Available models showcase
    - Getting started guide
    - Navigation hints
    
    The page serves as the entry point for new users and provides
    quick access to key application features.
    """
    debug("Starting home page render", "home_page")
    
    try:
        # Log page access for analytics
        log_user_action("home_page_visited", 
                       timestamp=datetime.now().isoformat(),
                       logging_available=LOGGING_AVAILABLE)
        
        # Render main header with enhanced styling
        debug("Rendering main header", "home_page")
        st.markdown('<h1 class="main-header">🤖 Welcome to DurgasAI</h1>', unsafe_allow_html=True)
    
        # Render main content sections with logging
        debug("Rendering introduction and features section", "home_page")
        st.markdown("""
        ## Advanced AI Agent Platform
        
        Welcome to **DurgasAI**, your comprehensive AI assistant powered by cutting-edge language models 
        from HuggingFace and enhanced with LangChain for superior conversational capabilities.
        
        ### 🌟 Key Features
        
        - **Multiple AI Models**: Choose from various state-of-the-art models
        - **LangChain Integration**: Advanced conversation memory and context management
        - **Customizable Prompts**: Tailor the AI's behavior to your needs
        - **Real-time Chat**: Smooth, responsive conversational interface
        - **Export Conversations**: Save and share your chat sessions
        - **Model Analytics**: Track usage and performance metrics
        
        ### 🚀 Available Models
        """)
        
        # Display available models using component with error handling
        debug("Rendering model overview component", "home_page")
        try:
            display_model_overview()
            debug("Model overview component rendered successfully", "home_page")
        except Exception as e:
            error("Failed to render model overview component", "home_page", e)
            st.error("❌ Failed to load model overview. Please check the configuration.")
    
        # Render use cases section
        debug("Rendering use cases section", "home_page")
        st.markdown("""
        ### 🎯 Use Cases
        
        - **💼 Business**: Customer support, content creation, data analysis
        - **🎓 Education**: Tutoring, research assistance, learning support  
        - **💻 Development**: Code review, documentation, debugging help
        - **🎨 Creative**: Writing, brainstorming, creative projects
        - **🔬 Research**: Information gathering, summarization, analysis
        
        ### 🚀 Getting Started
        
        1. Navigate to the **AI Agent** page
        2. Enter your HuggingFace API token (get one [here](https://huggingface.co/settings/tokens))
        3. Select your preferred AI model
        4. Customize the system prompt (optional)
        5. Start chatting!
        
        ---
        
        Ready to begin? Head over to the **🤖 AI Agent** page to start your conversation!
        """)
        
        # Display usage tips with error handling
        debug("Rendering usage tips component", "home_page")
        try:
            display_usage_tips()
            debug("Usage tips component rendered successfully", "home_page")
        except Exception as e:
            error("Failed to render usage tips component", "home_page", e)
            st.warning("⚠️ Usage tips temporarily unavailable")
        
        # Log successful page render
        info("Home page rendered successfully", "home_page",
             logging_available=LOGGING_AVAILABLE,
             render_timestamp=datetime.now().isoformat())
        
    except Exception as e:
        # Handle any unexpected errors during page render
        error("Critical error rendering home page", "home_page", e)
        st.error("🚨 An error occurred while loading the home page. Please refresh and try again.")
        
        # Show basic fallback content
        st.markdown("## 🤖 DurgasAI")
        st.markdown("Please refresh the page to access the full application.")
