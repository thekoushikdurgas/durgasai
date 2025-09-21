"""
Help Page for DurgasAI Application.

Provides user documentation, troubleshooting guides, and support information.

This module serves as the comprehensive help center for DurgasAI, offering:
- User guide and getting started instructions
- Troubleshooting guides for common issues
- Model selection guidance and recommendations
- Best practices for optimal usage
- Support information and resources

The page is designed to be self-service, helping users resolve
issues independently and learn advanced features.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from components.common.navigation import display_usage_tips

# Import logging utilities with fallback
try:
    from utils.logger import debug, info, warning, error, log_user_action
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    # Fallback logging functions
    def debug(msg, component="help_page", **kwargs): pass
    def info(msg, component="help_page", **kwargs): pass
    def warning(msg, component="help_page", **kwargs): pass
    def error(msg, component="help_page", **kwargs): pass
    def log_user_action(action, **kwargs): pass


def render_help_page() -> None:
    """
    Render the help page with documentation and support information.
    
    This function creates a comprehensive help center including:
    - User guide and getting started instructions
    - Troubleshooting guides for common issues
    - Model selection guidance
    - Best practices for optimal usage
    - Support information and resources
    
    All sections include detailed logging for user behavior analytics.
    """
    debug("Starting help page render", "help_page")
    
    try:
        # Log page access for analytics
        log_user_action("help_page_visited", 
                       timestamp=datetime.now().isoformat(),
                       logging_available=LOGGING_AVAILABLE)
        
        # Render main header
        debug("Rendering help page header", "help_page")
        st.markdown('<h1 class="main-header">❓ Help & Documentation</h1>', unsafe_allow_html=True)
        
        # User Guide Section
        debug("Rendering user guide section", "help_page")
        _render_user_guide()
        
        # Troubleshooting Section
        debug("Rendering troubleshooting section", "help_page")
        _render_troubleshooting()
        
        # Model Guide Section
        debug("Rendering model guide section", "help_page")
        _render_model_guide()
        
        # Best Practices Section
        debug("Rendering best practices section", "help_page")
        _render_best_practices()
        
        # Support Section
        debug("Rendering support information section", "help_page")
        _render_support_info()
        
        # Display usage tips with error handling
        debug("Rendering usage tips component", "help_page")
        try:
            display_usage_tips()
            debug("Usage tips component rendered successfully", "help_page")
        except Exception as e:
            error("Failed to render usage tips component", "help_page", e)
            st.warning("⚠️ Usage tips temporarily unavailable")
        
        # Log successful page render
        info("Help page rendered successfully", "help_page",
             sections_rendered=["user_guide", "troubleshooting", "model_guide", "best_practices", "support"],
             render_timestamp=datetime.now().isoformat())
        
    except Exception as e:
        # Handle any unexpected errors during page render
        error("Critical error rendering help page", "help_page", e)
        st.error("🚨 An error occurred while loading the help page. Please refresh and try again.")
        
        # Show basic fallback content
        st.markdown("## ❓ Help & Documentation")
        st.markdown("Please refresh the page to access the full help documentation.")


def _render_user_guide() -> None:
    """Render user guide section."""
    st.markdown("""
    ## 📚 User Guide
    
    ### Getting Started
    
    1. **Setup API Token**
       - Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
       - Create a new token (read access is sufficient)
       - Enter the token in the AI Agent page sidebar
    
    2. **Choose a Model**
       - Select from available models based on your needs
       - API models require internet connection
       - Local models run on your device (requires more resources)
    
    3. **Customize System Prompt**
       - Use predefined prompts or create your own
       - System prompts guide the AI's behavior and responses
       - Try different prompts for different use cases
    """)


def _render_troubleshooting() -> None:
    """Render troubleshooting section."""
    st.markdown("""
    ### 🔧 Troubleshooting
    
    **Common Issues:**
    
    - **"No model loaded"**: Make sure to initialize a model first
    - **API errors**: Check your internet connection and API token
    - **Slow responses**: Try a smaller model or reduce max tokens
    - **Memory issues**: Clear chat history or restart the application
    
    **Performance Tips:**
    
    - Use API models for faster responses
    - Adjust temperature for desired creativity level
    - Keep conversations focused for better context
    - Clear chat history periodically
    """)


def _render_model_guide() -> None:
    """Render model selection guide."""
    st.markdown("""
    ### 🤖 Model Guide
    
    **Mistral 7B Instruct**
    - Best for: Instructions, reasoning, coding
    - Strengths: Following complex instructions, logical reasoning
    - Use when: You need precise, instruction-following responses
    
    **Zephyr 7B Beta**
    - Best for: General conversation, Q&A
    - Strengths: Natural dialogue, helpfulness
    - Use when: You want natural, conversational interactions
    
    **Flan T5 Large**
    - Best for: Educational content, factual questions
    - Strengths: Factual accuracy, educational responses
    - Use when: You need reliable, educational information
    
    **DialoGPT Medium**
    - Best for: Casual chat, dialogue
    - Strengths: Conversational flow, personality
    - Use when: You want engaging, casual conversations
    
    **BlenderBot 400M**
    - Best for: Open-domain chat, creative conversations
    - Strengths: Creativity, diverse topics
    - Use when: You want creative, diverse conversations
    """)


def _render_best_practices() -> None:
    """Render best practices section."""
    st.markdown("""
    ### 📋 Best Practices
    
    1. **Be Specific**: Clear, specific questions get better answers
    2. **Use Context**: Reference previous messages for continuity
    3. **Experiment**: Try different models and settings
    4. **Save Important Chats**: Export conversations you want to keep
    5. **Provide Feedback**: Note what works well for future reference
    """)


def _render_support_info() -> None:
    """Render support information."""
    st.markdown("""
    ### 🆘 Support
    
    If you encounter issues:
    1. Check the troubleshooting section above
    2. Try clearing your browser cache
    3. Restart the application
    4. Check your internet connection
    5. Verify your API token is valid
    
    ### 📖 Additional Resources
    
    - [HuggingFace Documentation](https://huggingface.co/docs)
    - [LangChain Documentation](https://python.langchain.com/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """)
