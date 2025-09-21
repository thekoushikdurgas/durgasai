"""
Model Status Component.

Displays current model status, performance metrics, and health information.
"""

import streamlit as st
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.application_state import ApplicationState
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation


class ModelStatus:
    """Component for displaying model status and metrics."""
    
    def __init__(self):
        """Initialize model status component."""
        debug("ModelStatus initialized", "model_status")
    
    def render(self, detailed: bool = False) -> None:
        """
        Render model status information with comprehensive logging and error handling.
        
        This method displays the current model status, including loading state,
        performance metrics, and detailed information when requested.
        
        Args:
            detailed: Whether to show detailed status information and metrics
            
        Features:
        - Model loading status display
        - Performance metrics when available
        - Quick start guide for new users
        - Error handling and user feedback
        - User interaction tracking
        
        Logging:
        - Model status retrieval and validation
        - Status rendering performance
        - User interaction with status display
        - Error handling for status retrieval
        - Analytics for status display usage
        """
        debug("Starting model status rendering", "model_status",
              detailed_mode=detailed)
        
        try:
            with LoggedOperation("model_status_rendering", "model_status"):
                # Retrieve model status with error handling
                try:
                    model_loaded = ApplicationState.is_model_loaded()
                    current_model = ApplicationState.get_current_model()
                    
                    debug("Model status retrieved", "model_status",
                          model_loaded=model_loaded,
                          current_model=current_model,
                          detailed_mode=detailed)
                    
                except Exception as e:
                    error("Error retrieving model status", "model_status", e)
                    st.error("🚨 Error retrieving model status. Please try again.")
                    log_user_action("model_status_retrieval_failed", error=str(e))
                    return
                
                # Log status display action
                log_user_action("model_status_displayed",
                              model_loaded=model_loaded,
                              current_model=current_model,
                              detailed_mode=detailed)
                
                # Render appropriate status based on model state
                if model_loaded and current_model:
                    debug("Rendering loaded model status", "model_status",
                          model_name=current_model,
                          detailed=detailed)
                    
                    try:
                        self._render_loaded_status(current_model, detailed)
                        debug("Loaded model status rendered successfully", "model_status")
                        
                    except Exception as e:
                        error("Error rendering loaded model status", "model_status", e,
                              model_name=current_model,
                              detailed=detailed)
                        st.error("🚨 Error displaying model status. Please try again.")
                        raise
                        
                else:
                    debug("Rendering not loaded model status", "model_status",
                          model_loaded=model_loaded,
                          current_model=current_model)
                    
                    try:
                        self._render_not_loaded_status()
                        debug("Not loaded model status rendered successfully", "model_status")
                        
                    except Exception as e:
                        error("Error rendering not loaded model status", "model_status", e)
                        st.error("🚨 Error displaying model status. Please try again.")
                        raise
                
                debug("Model status rendering completed successfully", "model_status",
                      model_loaded=model_loaded,
                      detailed_mode=detailed)
                
        except Exception as e:
            error("Critical error in model status rendering", "model_status", e,
                  detailed_mode=detailed)
            st.error("🚨 An error occurred while displaying model status.")
            log_user_action("model_status_display_failed", error=str(e))
            raise
        
        debug("Model status rendering process completed", "model_status")
    
    def _render_loaded_status(self, model_name: str, detailed: bool) -> None:
        """Render status for loaded model."""
        st.success(f"✅ **{model_name}** is ready for conversation!")
        
        if detailed:
            # Show detailed metrics
            last_response_time = ApplicationState.get("last_model_response_time")
            total_messages = ApplicationState.get("total_messages", 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if last_response_time:
                    st.metric("Last Response Time", f"{last_response_time:.2f}s")
                else:
                    st.metric("Last Response Time", "N/A")
            
            with col2:
                st.metric("Total Messages", total_messages)
    
    def _render_not_loaded_status(self) -> None:
        """Render status when no model is loaded."""
        st.warning("⚠️ Please load a model from the sidebar to start chatting.")
        
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            1. **Enter API Token**: Add your HuggingFace API token in the sidebar
            2. **Select Model**: Choose from available AI models
            3. **Configure Parameters**: Adjust temperature, max tokens, etc.
            4. **Load Model**: Click the "Load Model" button
            5. **Start Chatting**: Type your message in the chat input
            """)
