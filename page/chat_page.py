"""
Chat Page for DurgasAI Application.

Main chat interface with AI models, separated from complex UI logic.

This module provides the main chat interface for DurgasAI, featuring:
- Clean separation between UI logic and chat functionality
- Modular component architecture for maintainability
- Comprehensive error handling and logging
- Model configuration and parameter controls
- Real-time chat interface with AI models

The page serves as a simplified alternative to the full AI Agent page,
focusing on core chat functionality with streamlined UI.
"""

import streamlit as st
from typing import Dict, Any
from pathlib import Path
import sys
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_manager import ModelManager

# Import logging utilities with enhanced error handling
try:
    from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    # Fallback logging functions and classes
    def debug(msg, component="chat_page", **kwargs): pass
    def info(msg, component="chat_page", **kwargs): pass
    def warning(msg, component="chat_page", **kwargs): pass
    def error(msg, component="chat_page", **kwargs): pass
    def log_user_action(action, **kwargs): pass
    
    class LoggedOperation:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

from components.chat.chat_interface import ChatInterface
from components.model.model_selector import ModelSelector
from components.model.parameter_controls import ParameterControls
from components.common.navigation import display_usage_tips


def render_chat_page(model_manager: ModelManager) -> None:
    """
    Render the main chat page with comprehensive logging and error handling.
    
    This function orchestrates the complete chat page rendering including:
    - Model manager validation
    - Chat interface initialization
    - Sidebar configuration setup
    - Model status display
    - Error handling and recovery
    
    Args:
        model_manager (ModelManager): The model manager instance for AI operations
    
    Raises:
        Exception: Re-raises any critical errors after logging
    """
    debug("Starting chat page render", "chat_page", 
          model_manager_available=bool(model_manager),
          logging_available=LOGGING_AVAILABLE)
    
    try:
        # Log page access for analytics
        log_user_action("chat_page_visited", 
                       timestamp=datetime.now().isoformat(),
                       model_manager_type=type(model_manager).__name__)
        
        with LoggedOperation("render_chat_page", "pages"):
            # Validate model manager
            if not model_manager:
                error("Model manager not provided to chat page", "chat_page")
                st.error("🚨 Model manager not available. Please restart the application.")
                return
            
            # Initialize chat interface with error handling
            debug("Initializing chat interface component", "chat_page")
            try:
                chat_interface = ChatInterface(model_manager)
                debug("Chat interface initialized successfully", "chat_page")
            except Exception as e:
                error("Failed to initialize chat interface", "chat_page", e)
                st.error("❌ Failed to initialize chat interface. Please refresh the page.")
                return
            
            # Render sidebar configuration
            debug("Rendering sidebar configuration", "chat_page")
            try:
                config = _render_sidebar_config(model_manager)
                debug("Sidebar configuration rendered", "chat_page", 
                      config_keys=list(config.keys()) if config else [])
            except Exception as e:
                error("Failed to render sidebar configuration", "chat_page", e)
                st.error("❌ Failed to load configuration panel")
                config = {}
            
            # Render main chat interface header
            debug("Rendering main chat interface header", "chat_page")
            st.markdown('<h1 class="main-header">🤖 AI Agent Chat</h1>', unsafe_allow_html=True)
            
            # Display model status with error handling
            debug("Displaying model status", "chat_page")
            try:
                _display_model_status()
                debug("Model status displayed successfully", "chat_page")
            except Exception as e:
                error("Failed to display model status", "chat_page", e)
                st.warning("⚠️ Model status temporarily unavailable")
            
            # Render chat interface with error handling
            debug("Rendering chat interface component", "chat_page")
            try:
                chat_interface.render(config)
                debug("Chat interface rendered successfully", "chat_page")
                
                # Log successful page render
                info("Chat page rendered successfully", "chat_page",
                     render_timestamp=datetime.now().isoformat(),
                     config_provided=bool(config))
                     
            except Exception as e:
                error("Failed to render chat interface", "chat_page", e)
                st.error("❌ Failed to load chat interface. Please refresh the page.")
                
    except Exception as e:
        # Handle any unexpected errors during page render
        error("Critical error rendering chat page", "chat_page", e)
        st.error("🚨 A critical error occurred while loading the chat page. Please refresh and try again.")
        
        # Show basic fallback content
        st.markdown("## 🤖 AI Agent Chat")
        st.markdown("Please refresh the page to access the chat interface.")


def _render_sidebar_config(model_manager: ModelManager) -> Dict[str, Any]:
    """
    Render sidebar configuration and return config dict with comprehensive logging.
    
    This function handles the complete sidebar configuration including:
    - Model selection component rendering
    - Parameter controls setup
    - Model control buttons
    - Usage tips display
    - Configuration merging and validation
    
    Args:
        model_manager (ModelManager): The model manager instance
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary
        
    Raises:
        Exception: Re-raises any critical errors after logging
    """
    debug("Starting sidebar configuration render", "chat_page")
    
    try:
        with st.sidebar:
            # Render sidebar header
            debug("Rendering sidebar header", "chat_page")
            st.markdown("### 🤖 AI Model Configuration")
            
            # Model selection component with error handling
            debug("Initializing model selector component", "chat_page")
            try:
                model_selector = ModelSelector()
                model_config = model_selector.render()
                debug("Model selector rendered successfully", "chat_page", 
                      config_type=type(model_config).__name__)
            except Exception as e:
                error("Failed to render model selector", "chat_page", e)
                st.error("❌ Failed to load model selection")
                model_config = {}
            
            # Parameter controls component with error handling
            debug("Initializing parameter controls component", "chat_page")
            try:
                param_controls = ParameterControls()
                # Extract the actual ModelConfig object from the model_config dict
                actual_model_config = model_config.get('model_config') if isinstance(model_config, dict) else model_config
                parameters = param_controls.render(actual_model_config)
                debug("Parameter controls rendered successfully", "chat_page",
                      parameters_count=len(parameters) if parameters else 0)
            except Exception as e:
                error("Failed to render parameter controls", "chat_page", e)
                st.error("❌ Failed to load parameter controls")
                parameters = {}
            
            # Model control buttons with error handling
            debug("Rendering model control buttons", "chat_page")
            try:
                _render_model_controls(model_manager, model_config, parameters)
                debug("Model control buttons rendered successfully", "chat_page")
            except Exception as e:
                error("Failed to render model controls", "chat_page", e)
                st.error("❌ Failed to load model controls")
            
            # Usage tips with error handling
            debug("Rendering usage tips in sidebar", "chat_page")
            try:
                display_usage_tips()
                debug("Usage tips rendered successfully in sidebar", "chat_page")
            except Exception as e:
                error("Failed to render usage tips in sidebar", "chat_page", e)
                st.warning("⚠️ Usage tips temporarily unavailable")
            
            # Merge configuration with validation
            debug("Merging configuration dictionaries", "chat_page")
            try:
                merged_config = {
                    **model_config,
                    **parameters
                }
                debug("Configuration merged successfully", "chat_page",
                      total_config_keys=len(merged_config))
                return merged_config
            except Exception as e:
                error("Failed to merge configuration", "chat_page", e)
                return {}
                
    except Exception as e:
        error("Critical error in sidebar configuration", "chat_page", e)
        st.error("🚨 Failed to load sidebar configuration")
        return {}


def _render_model_controls(model_manager: ModelManager, model_config: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """Render model control buttons."""
    st.markdown("### 🚀 Model Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Model", type="primary", use_container_width=True):
            _handle_model_load(model_manager, model_config, parameters)
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            _handle_chat_clear(model_manager)


def _handle_model_load(model_manager: ModelManager, model_config: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """Handle model loading."""
    api_token = model_config.get('api_token', '')
    selected_model = model_config.get('selected_model', '')
    system_prompt = model_config.get('system_prompt', '')
    
    if not api_token and model_config.get('requires_api_token', True):
        st.error("❌ API token required for this model")
        return
    
    with st.spinner("Loading model..."):
        success = model_manager.initialize_model(selected_model, api_token, system_prompt)
        
        if success:
            st.session_state.model_loaded = True
            st.session_state.current_model = model_config.get('model_name', selected_model)


def _handle_chat_clear(model_manager: ModelManager) -> None:
    """Handle chat clearing."""
    from core.application_state import ApplicationState
    
    # Clear messages
    ApplicationState.set('messages', [])
    ApplicationState.set('total_messages', 0)
    
    # Clear model chat history
    model_manager.clear_chat_history()
    
    st.success("Chat cleared!")


def _display_model_status() -> None:
    """Display current model status."""
    if st.session_state.get("model_loaded", False):
        current_model = st.session_state.get("current_model", "Unknown")
        st.success(f"✅ **{current_model}** is ready for conversation!")
    else:
        st.warning("⚠️ Please load a model from the sidebar to start chatting.")
